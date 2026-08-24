-- =====================================================================
-- schema_pgvector.sql
-- Vector-embedding layer for BlueByte AI, added on top of
-- schema_postgis.sql + schema_postgis_addendum.sql.
--
-- Fills the "Vector DB (Pinecone)" box from the architecture diagram
-- in README.md using pgvector INSIDE the same Postgres instance we
-- already run (timescale/timescaledb-ha:pg16), instead of standing up
-- a second managed service. One database, one connection pool, one
-- transaction boundary for structured + vector queries — see
-- vector_queries.hybrid_species_search() for why that matters.
--
-- Run this AFTER schema_postgis.sql and schema_postgis_addendum.sql:
--   psql "$BLUEBYTE_DATABASE_URL" -f schema_postgis.sql
--   psql "$BLUEBYTE_DATABASE_URL" -f schema_postgis_addendum.sql
--   psql "$BLUEBYTE_DATABASE_URL" -f schema_pgvector.sql
--
-- (docker-compose.db.yml already mounts all three into
--  docker-entrypoint-initdb.d/ in this order, so a fresh
--  `docker compose up` applies all of them automatically.)
-- =====================================================================

CREATE EXTENSION IF NOT EXISTS vector;
-- If your Postgres image doesn't bundle pgvector, either switch the
-- image in docker-compose.db.yml to one that does (e.g.
-- pgvector/pgvector:pg16, or install the `postgresql-16-pgvector`
-- package on the host) — timescale/timescaledb-ha:pg16 ships it as of
-- recent releases, but pin/verify the tag if this CREATE EXTENSION fails.

-- ---------------------------------------------------------------------
-- 1. TEXT/SEMANTIC EMBEDDINGS on existing tables
--    Dimension 384 == sentence-transformers/all-MiniLM-L6-v2 output
--    (see db/embeddings.py — swap models there, not here, if you need
--    a different dimension; update this column's vector(N) to match).
-- ---------------------------------------------------------------------
ALTER TABLE species
    ADD COLUMN IF NOT EXISTS description_embedding vector(384),
    ADD COLUMN IF NOT EXISTS embedding_updated_at TIMESTAMPTZ;

-- HNSW: query-time accuracy/speed without needing to pick & retrain an
-- IVFFlat list count as data grows — good default for a dataset this
-- size (dozens to low-thousands of species/chunks, not billions).
CREATE INDEX IF NOT EXISTS idx_species_embedding
    ON species USING hnsw (description_embedding vector_cosine_ops);

-- ---------------------------------------------------------------------
-- 2. eDNA SEQUENCE EMBEDDINGS — alignment-free k-mer composition
--    vectors (dimension 256 == 4^4 possible 4-mers over {A,C,G,T}).
--    See db/embeddings.py:kmer_frequency_vector(). This is a coarse
--    pre-filter for "which reference sequences look similar to this
--    fragment" — NOT a replacement for BLAST/alignment confirmation,
--    but it works on partial/degraded fragments where alignment tools
--    struggle, and it's a single indexed SQL query instead of an
--    external BLAST call.
-- ---------------------------------------------------------------------
ALTER TABLE edna_samples
    ADD COLUMN IF NOT EXISTS sequence_embedding vector(256);

CREATE INDEX IF NOT EXISTS idx_edna_sequence_embedding
    ON edna_samples USING hnsw (sequence_embedding vector_cosine_ops);

-- ---------------------------------------------------------------------
-- 3. RESEARCH DOCUMENT INGESTION — the "NLP for Research Ingestion"
--    innovation from README.md / INNOVATION.md, previously undocumented
--    in the schema. Chunked + embedded so a natural-language question
--    ("what does research say about hypoxia in the Bay of Bengal?")
--    retrieves the most relevant passages (RAG-style) instead of only
--    full-document keyword search.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS research_documents (
    id             UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    title          TEXT NOT NULL,
    source         TEXT,               -- e.g. 'CMFRI', 'INCOIS', 'NCBI', 'internal-report'
    url            TEXT,
    domain         TEXT,               -- 'oceanography' | 'fisheries' | 'biodiversity' | 'cross-domain'
    published_date DATE,
    raw_text       TEXT NOT NULL,
    ingested_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS research_chunks (
    id            UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    document_id   UUID NOT NULL REFERENCES research_documents(id) ON DELETE CASCADE,
    chunk_index   INT NOT NULL,
    content       TEXT NOT NULL,
    token_count   INT,
    embedding     vector(384),
    created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (document_id, chunk_index)
);

CREATE INDEX IF NOT EXISTS idx_research_chunks_embedding
    ON research_chunks USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_research_chunks_document
    ON research_chunks (document_id);

-- ---------------------------------------------------------------------
-- 4. GRID ECOLOGICAL PROFILES — fused cross-domain embedding per
--    H3 cell, the concrete implementation of INNOVATION.md's
--    "Tri-Domain Cross-Attention Knowledge Network" idea at the
--    storage layer: one vector per grid summarizing its physical
--    conditions (SST/salinity/chlorophyll/DO) AND its observed
--    biology (species richness, eDNA-detected taxa) as a single
--    natural-language profile, embedded — so "find grids ecologically
--    similar to grid X" or "find grids like: warm, hypoxic, tuna
--    eDNA present" is one ORDER BY ... <=> ... query instead of a
--    hand-built multi-factor scoring function.
-- ---------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS grid_ecological_profiles (
    id                    UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    h3_index              BIGINT NOT NULL,
    profile_date          DATE NOT NULL,
    avg_sst               DOUBLE PRECISION,
    avg_salinity          DOUBLE PRECISION,
    avg_chlorophyll       DOUBLE PRECISION,
    avg_dissolved_oxygen  DOUBLE PRECISION,
    reading_count         INT,
    species_richness      INT,          -- distinct species detected via eDNA in this cell
    dominant_species      TEXT[],
    profile_text          TEXT,         -- the natural-language description that was embedded
                                         -- (kept for debugging/explainability of the embedding)
    profile_embedding     vector(384),
    computed_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
    UNIQUE (h3_index, profile_date)
);

CREATE INDEX IF NOT EXISTS idx_grid_profile_embedding
    ON grid_ecological_profiles USING hnsw (profile_embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS idx_grid_profile_h3 ON grid_ecological_profiles (h3_index, profile_date DESC);
