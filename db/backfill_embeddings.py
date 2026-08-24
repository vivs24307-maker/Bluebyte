"""
backfill_embeddings.py
=======================
One-shot batch job: computes and stores embeddings for every
species / edna_samples row that doesn't have one yet (description_embedding
/ sequence_embedding IS NULL). Run this once after applying
schema_pgvector.sql on a database that already has data (e.g. right
after schema_postgis.sql + db/load_synthetic_dataset.py, before pgvector existed), or
any time bulk inserts happened outside the vector_queries upsert helpers.

Usage:
    python -m db.backfill_embeddings
"""

import asyncio

from db.connection import db_manager
from db.embeddings import embed_species, embed_edna_sequence, vector_literal, using_real_model


async def backfill_species(conn) -> int:
    rows = await conn.fetch(
        """
        SELECT id, common_name, scientific_name, family, habitat_type,
               conservation_status, commercial_value, min_sst, max_sst
        FROM species WHERE description_embedding IS NULL
        """
    )
    for r in rows:
        vec = embed_species(
            r["common_name"], r["scientific_name"], r["family"], r["habitat_type"],
            r["conservation_status"], r["commercial_value"], r["min_sst"], r["max_sst"],
        )
        await conn.execute(
            "UPDATE species SET description_embedding = $2::vector, embedding_updated_at = now() WHERE id = $1",
            r["id"], vector_literal(vec),
        )
    return len(rows)


async def backfill_edna(conn) -> int:
    rows = await conn.fetch(
        "SELECT id, sequence_fragment FROM edna_samples WHERE sequence_embedding IS NULL"
    )
    for r in rows:
        vec = embed_edna_sequence(r["sequence_fragment"] or "")
        await conn.execute(
            "UPDATE edna_samples SET sequence_embedding = $2::vector WHERE id = $1",
            r["id"], vector_literal(vec),
        )
    return len(rows)


async def backfill_all():
    print(f"Text embedding backend: {'sentence-transformers (all-MiniLM-L6-v2)' if using_real_model() else 'offline hashing fallback'}")
    pool = await db_manager.connect()
    async with pool.acquire() as conn:
        n_species = await backfill_species(conn)
        n_edna = await backfill_edna(conn)
    print(f"Backfilled {n_species} species embeddings, {n_edna} eDNA sequence embeddings")


if __name__ == "__main__":
    asyncio.run(backfill_all())