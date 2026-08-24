"""
vector_queries.py
==================
pgvector-backed query & upsert functions, following the same
conventions as db/queries.py (async, asyncpg, `async with get_db()`,
plain-dict return rows) so server/api routes can import from here the
same way they import from db/queries.py.

Every similarity search orders by the pgvector `<=>` cosine-distance
operator and reports `1 - distance` as `similarity` (1.0 = identical,
0.0 = orthogonal, negative = opposed) so callers/UI don't have to
mentally invert a distance metric.
"""

from datetime import date
from typing import Optional

from db.connection import get_db
from db.embeddings import (
    embed_text,
    embed_species,
    embed_edna_sequence,
    build_grid_profile_text,
    vector_literal,
)


# ---------------------------------------------------------------------
# Upserts — call these after any insert/update to species, edna_samples,
# or research_documents so their embeddings stay in sync. See
# backfill_embeddings.py for a one-shot batch version over existing rows.
# ---------------------------------------------------------------------

async def upsert_species_embedding(species_id: str, vec: Optional[list] = None, row: dict = None):
    """Pass either a precomputed `vec`, or a `row` dict (as returned by
    get_all_species()) to compute it here."""
    if vec is None:
        if row is None:
            raise ValueError("upsert_species_embedding needs either vec= or row=")
        vec = embed_species(
            row.get("common_name"), row.get("scientific_name"), row.get("family"),
            row.get("habitat_type"), row.get("conservation_status"),
            row.get("commercial_value"), row.get("min_sst"), row.get("max_sst"),
        )
    async with get_db() as db:
        await db.execute(
            """
            UPDATE species
            SET description_embedding = $2::vector, embedding_updated_at = now()
            WHERE id = $1
            """,
            species_id, vector_literal(vec),
        )
    return vec


async def upsert_edna_embedding(edna_id: str, sequence_fragment: str):
    vec = embed_edna_sequence(sequence_fragment)
    async with get_db() as db:
        await db.execute(
            "UPDATE edna_samples SET sequence_embedding = $2::vector WHERE id = $1",
            edna_id, vector_literal(vec),
        )
    return vec


async def insert_research_document(title: str, raw_text: str, source: str = None,
                                     url: str = None, domain: str = None,
                                     published_date=None) -> str:
    """Inserts the parent document row only. Use ingest_research.py's
    `ingest_document()` for the full chunk + embed + insert pipeline —
    this is exposed separately for callers that want to manage
    chunking themselves."""
    async with get_db() as db:
        row = await db.fetchrow(
            """
            INSERT INTO research_documents (title, source, url, domain, published_date, raw_text)
            VALUES ($1, $2, $3, $4, $5, $6)
            RETURNING id
            """,
            title, source, url, domain, published_date, raw_text,
        )
        return str(row["id"])


async def insert_research_chunk(document_id: str, chunk_index: int, content: str,
                                  token_count: int = None):
    vec = embed_text(content)
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO research_chunks (document_id, chunk_index, content, token_count, embedding)
            VALUES ($1, $2, $3, $4, $5::vector)
            ON CONFLICT (document_id, chunk_index) DO UPDATE
                SET content = EXCLUDED.content, embedding = EXCLUDED.embedding
            """,
            document_id, chunk_index, content, token_count, vector_literal(vec),
        )
    return vec


# ---------------------------------------------------------------------
# Semantic search
# ---------------------------------------------------------------------

async def semantic_search_species(query_text: str, top_k: int = 5):
    """Natural-language species search, e.g. 'small schooling fish that
    tolerates warm shallow water' — matches on the embedded description,
    not just exact field values."""
    qvec = vector_literal(embed_text(query_text))
    async with get_db() as db:
        rows = await db.fetch(
            """
            SELECT id, common_name, scientific_name, family, habitat_type,
                   conservation_status, commercial_value, min_sst, max_sst,
                   1 - (description_embedding <=> $1::vector) AS similarity
            FROM species
            WHERE description_embedding IS NOT NULL
            ORDER BY description_embedding <=> $1::vector
            LIMIT $2
            """,
            qvec, top_k,
        )
        return [dict(r) for r in rows]


async def find_similar_species(species_id: str, top_k: int = 5):
    """'More like this' for a given species — e.g. surfacing ecological
    analogues when planning bycatch-aware advisories."""
    async with get_db() as db:
        rows = await db.fetch(
            """
            SELECT b.id, b.common_name, b.scientific_name, b.family, b.habitat_type,
                   1 - (b.description_embedding <=> a.description_embedding) AS similarity
            FROM species a
            JOIN species b ON b.id != a.id
            WHERE a.id = $1
              AND a.description_embedding IS NOT NULL
              AND b.description_embedding IS NOT NULL
            ORDER BY b.description_embedding <=> a.description_embedding
            LIMIT $2
            """,
            species_id, top_k,
        )
        return [dict(r) for r in rows]


async def hybrid_species_search(query_text: str, sst: float = None,
                                  salinity: float = None, top_k: int = 10):
    """Combines semantic similarity with hard physical-condition
    filters in ONE SQL round trip — the 'unified platform' claim made
    literal at the query layer: instead of a keyword/vector search
    system bolted onto a separate structured database, one query can
    say 'find species like a mackerel, that also tolerate 29C water'."""
    qvec = vector_literal(embed_text(query_text))
    conditions = ["description_embedding IS NOT NULL"]
    params: list = [qvec]
    if sst is not None:
        params.append(sst)
        conditions.append(f"(min_sst IS NULL OR max_sst IS NULL OR ${len(params)} BETWEEN min_sst AND max_sst)")
    if salinity is not None:
        params.append(salinity)
        conditions.append(f"(min_salinity IS NULL OR max_salinity IS NULL OR ${len(params)} BETWEEN min_salinity AND max_salinity)")
    params.append(top_k)
    sql = f"""
        SELECT id, common_name, scientific_name, family, habitat_type, commercial_value,
               min_sst, max_sst, min_salinity, max_salinity,
               1 - (description_embedding <=> $1::vector) AS similarity
        FROM species
        WHERE {' AND '.join(conditions)}
        ORDER BY description_embedding <=> $1::vector
        LIMIT ${len(params)}
    """
    async with get_db() as db:
        rows = await db.fetch(sql, *params)
        return [dict(r) for r in rows]


async def find_similar_edna_sequences(sequence_fragment: str, top_k: int = 10,
                                        min_similarity: float = 0.0):
    """Alignment-free nearest-neighbor search over eDNA sequence
    fragments via k-mer composition. Good for a coarse first pass on a
    fresh/degraded field sample before running a full alignment against
    a reference database."""
    qvec = vector_literal(embed_edna_sequence(sequence_fragment))
    async with get_db() as db:
        rows = await db.fetch(
            """
            SELECT e.id, e.sample_id, e.marker_gene, e.detection_confidence,
                   e.collection_date, s.common_name, s.scientific_name,
                   1 - (e.sequence_embedding <=> $1::vector) AS similarity
            FROM edna_samples e
            JOIN species s ON e.species_id = s.id
            WHERE e.sequence_embedding IS NOT NULL
            ORDER BY e.sequence_embedding <=> $1::vector
            LIMIT $2
            """,
            qvec, top_k,
        )
        return [dict(r) for r in rows if r["similarity"] >= min_similarity]


async def semantic_search_research(query_text: str, top_k: int = 5, domain: str = None):
    """RAG-style retrieval over ingested research paper chunks — the
    'NLP for Research Ingestion' innovation from README.md, now backed
    by an actual retrievable store instead of just a stated goal."""
    qvec = vector_literal(embed_text(query_text))
    params: list = [qvec]
    domain_clause = ""
    if domain:
        params.append(domain)
        domain_clause = f"AND d.domain = ${len(params)}"
    params.append(top_k)
    sql = f"""
        SELECT c.id AS chunk_id, c.content, c.chunk_index,
               d.id AS document_id, d.title, d.source, d.url, d.domain, d.published_date,
               1 - (c.embedding <=> $1::vector) AS similarity
        FROM research_chunks c
        JOIN research_documents d ON c.document_id = d.id
        WHERE c.embedding IS NOT NULL {domain_clause}
        ORDER BY c.embedding <=> $1::vector
        LIMIT ${len(params)}
    """
    async with get_db() as db:
        rows = await db.fetch(sql, *params)
        return [dict(r) for r in rows]


async def semantic_search_grids(query_text: str, top_k: int = 5, as_of: date = None):
    """e.g. 'warm hypoxic water with tuna eDNA present' -> matching grid
    cells, most recent profile per cell on or before `as_of` (defaults
    to latest available)."""
    qvec = vector_literal(embed_text(query_text))
    async with get_db() as db:
        rows = await db.fetch(
            """
            SELECT DISTINCT ON (h3_index) h3_index, profile_date, avg_sst, avg_salinity,
                   avg_chlorophyll, avg_dissolved_oxygen, species_richness,
                   dominant_species, profile_text,
                   1 - (profile_embedding <=> $1::vector) AS similarity
            FROM grid_ecological_profiles
            WHERE profile_embedding IS NOT NULL
              AND ($3::date IS NULL OR profile_date <= $3)
            ORDER BY h3_index, profile_date DESC, profile_embedding <=> $1::vector
            LIMIT $2
            """,
            qvec, top_k, as_of,
        )
        # DISTINCT ON above guarantees one row per cell (latest date);
        # re-sort that result set by similarity for the final ranking.
        return sorted([dict(r) for r in rows], key=lambda r: r["similarity"], reverse=True)[:top_k]


async def find_similar_grids(h3_index: int, profile_date_: date = None, top_k: int = 5):
    async with get_db() as db:
        anchor = await db.fetchrow(
            """
            SELECT profile_embedding FROM grid_ecological_profiles
            WHERE h3_index = $1 AND ($2::date IS NULL OR profile_date = $2)
              AND profile_embedding IS NOT NULL
            ORDER BY profile_date DESC LIMIT 1
            """,
            h3_index, profile_date_,
        )
        if anchor is None or anchor["profile_embedding"] is None:
            return []
        rows = await db.fetch(
            """
            SELECT DISTINCT ON (h3_index) h3_index, profile_date, avg_sst, avg_salinity,
                   species_richness, dominant_species, profile_text,
                   1 - (profile_embedding <=> $1::vector) AS similarity
            FROM grid_ecological_profiles
            WHERE h3_index != $2 AND profile_embedding IS NOT NULL
            ORDER BY h3_index, profile_date DESC
            """,
            anchor["profile_embedding"], h3_index,
        )
        return sorted([dict(r) for r in rows], key=lambda r: r["similarity"], reverse=True)[:top_k]
