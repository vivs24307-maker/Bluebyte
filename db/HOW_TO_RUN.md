# BlueByte AI — DB Layer: How to Run

Covers the existing PostGIS/TimescaleDB layer plus what's new: **vector
embeddings (pgvector)** and a **synthetic test dataset** covering every
parameter in the schema.

## 0. What's new here

| File | Purpose |
|---|---|
| `schema_pgvector.sql` | Enables pgvector, adds embedding columns to `species`/`edna_samples`, adds `research_documents`/`research_chunks` (NLP research ingestion) and `grid_ecological_profiles` (fused cross-domain grid embeddings) |
| `embeddings.py` | Text embeddings (sentence-transformers, offline-fallback) + DNA k-mer embeddings + chunking |
| `vector_queries.py` | All semantic/similarity search + upsert functions |
| `ingest_research.py` | Chunk + embed + store research papers/reports |
| `grid_profiles.py` | Computes fused physical+biological embedding per H3 cell |
| `backfill_embeddings.py` | One-shot: embed any existing species/eDNA rows that don't have an embedding yet |
| `synthetic_data/generate_synthetic_dataset.py` | Generates a full test dataset (CSV/JSON) for every table |
| `load_synthetic_dataset.py` | Loads that dataset into Postgres end-to-end (ETL, embeddings, grid profiles) |
| `demo_vector_search.py` | Runs every vector-search capability and prints results |
| `stream_loader.py` | Fixed one import bug (`from etl_pipeline` → `from db.etl_pipeline`) so it actually runs as part of the `db` package |

## 1. Prerequisites

```bash
docker --version        # Docker + Docker Compose
python3 --version       # 3.11+
```

## 2. Start the database

From the repo root:

```bash
cd db
docker compose -f docker-compose.db.yml up -d
```

This starts Postgres 16 with PostGIS + TimescaleDB + pgvector, and
auto-applies, **in order**: `schema_postgis.sql` →
`schema_postgis_addendum.sql` → `schema_pgvector.sql` (all three are
mounted into `docker-entrypoint-initdb.d/`).

Check it came up healthy:

```bash
docker compose -f docker-compose.db.yml ps
docker exec -it bluebyte_postgres psql -U bluebyte -d bluebyte -c "\dx"
# should list postgis, timescaledb, uuid-ossp, vector
```

> If `CREATE EXTENSION vector` failed silently, your image tag doesn't
> bundle pgvector — check `docker logs bluebyte_postgres` for the
> `03_schema_pgvector.sql` output, and see the comment at the top of
> `docker-compose.db.yml` for an alternate image.

Optional: pgAdmin is also started at `http://localhost:5050`
(`dev@bluebyte.local` / `bluebyte_dev`).

## 3. Install Python dependencies

From the repo root:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r db/requirements.txt
```

`sentence-transformers` is in there for higher-quality text embeddings.
It's optional — if it or its model download isn't available,
`embeddings.py` automatically falls back to a deterministic offline
hashing embedding of the same dimension, so **everything below still
works with zero internet access**, just with less semantically-rich
text search (DNA k-mer search is unaffected either way — it never
depended on a downloaded model).

Set the DB connection string if you're not using the default:

```bash
export BLUEBYTE_DATABASE_URL="postgresql://bluebyte:bluebyte_dev@localhost:5432/bluebyte"
```

## 4. Generate the synthetic test dataset

```bash
python3 -m db.synthetic_data.generate_synthetic_dataset
```

Writes CSV/JSON files to `db/synthetic_data/data/`:

- `ocean_grids.csv` — 20 grid polygons across the Indian EEZ
- `species.csv` — 13 species (mackerel, sardine, hilsa, tuna, shrimp,
  shark, turtle, ...) with rich text fields for semantic search to
  actually differentiate on
- `buoy_readings.csv` — 6 sensors, 128 readings total (120 normal + 8
  deliberately injected outliers — heatwave SST spikes, hypoxia DO
  crashes) so `etl_pipeline.py`'s outlier flagging has real anomalies
  to catch
- `river_discharge.csv` — 4 gauge stations × 48 hourly readings each
  (192 rows total)
- `edna_samples.csv` — 93 samples; every species gets a deterministic
  reference DNA sequence (`_reference_sequence()`), and each sample is
  that reference with 1–3 point mutations — so nearest-neighbor search
  on the embeddings has a genuine "same species clusters together"
  signal to find
- `fishing_zones.csv` — 8 zones, `alerts.csv` — 6 alerts, matching the
  existing schema
- `research_documents.json` — 5 short synthetic abstracts referencing
  real species from the registry, for testing the research-paper RAG
  search

Generation is fully reproducible (hardcoded `random.Random(42)`) but
does NOT currently take CLI flags — `--seed`/`--out-dir` don't exist.
To change the seed or output location, edit the constants at the top
of `generate_synthetic_dataset.py` directly.

## 5. Load it into Postgres

```bash
python3 -m db.load_synthetic_dataset
```

This, in order:
1. Inserts ocean grids, species, fishing zones, alerts directly
2. Streams buoy readings + river discharge through
   `stream_loader.py` → `etl_pipeline.OutlierPreservingETL` (the real
   outlier-flagging ingestion path — not a shortcut insert)
3. Inserts eDNA samples, resolving each to its `species_id`
4. Ingests the research documents (chunk + embed + store)
5. Backfills embeddings for every species / eDNA row
6. Computes fused ecological-profile embeddings per H3 grid cell

Flags: `--skip-embeddings`, `--skip-grid-profiles`, `--data-dir <path>`.

(If you'd rather use a smaller, hand-written seed set instead of the
full synthetic dataset, that path — `db/seed_data.py` — was removed
when the synthetic_data/ pipeline replaced it. Use `python3 -m
db.synthetic_data.generate_synthetic_dataset` followed by `python3 -m
db.load_synthetic_dataset` instead; run `python3 -m
db.backfill_embeddings` afterward to add embeddings to it.)

## 6. Try the vector search

```bash
python3 -m db.demo_vector_search
```

Runs, against whatever's loaded:
1. Semantic species search ("small oily schooling fish, high commercial value")
2. "More like this" for a given species
3. Hybrid search: semantic query + hard SST/salinity filters in one query
4. eDNA nearest-neighbor search on a simulated field fragment
5. RAG-style research search ("why is dissolved oxygen dropping near Kerala?")
6. Grid ecological similarity search ("warm hypoxic water")

Or call the functions directly:

```python
from db import (
    semantic_search_species, find_similar_species, hybrid_species_search,
    find_similar_edna_sequences, semantic_search_research, semantic_search_grids,
)

await semantic_search_species("reef fish tolerant of warm shallow water", top_k=5)
await find_similar_edna_sequences(raw_sequence_from_lab, top_k=10)
await semantic_search_research("marine heatwave impact on chlorophyll", top_k=3)
```

These are exported from `db/__init__.py` exactly like the existing
`get_buoy_readings`, `get_active_alerts`, etc. — same import style
(`from db import queries` / `from db import ...`) the API routes
already use, no changes needed there to start calling them.

## 7. Running the API on top of this (unchanged)

```bash
pip install -r requirements.txt   # repo root — FastAPI, etc.
python3 start.py
```

`server/api/routes/ocean.py` and friends already fall back to sample
data if the DB import fails, so the API runs fine even before you've
done any of the above — steps 1-6 are what makes the *real* data path
(and the new vector search) light up instead of the fallback.

## 8. Re-running / refreshing

- New buoy/eDNA data arrives → run `backfill_embeddings.py` to embed
  anything new.
- Want fresher grid profiles → `python3 -m db.grid_profiles --days 7`
  (or however wide a window makes sense for your data density).
- Want a bigger/different synthetic dataset → edit the `random.Random(42)`
  seed or the count parameters (`num_normal`, `num_outliers`,
  `num_stations`, etc.) directly in
  `generate_synthetic_dataset.py` (sensor count, days, mutation rate, etc.)
  and re-run steps 4-5. `ON CONFLICT DO NOTHING`/`DO UPDATE` clauses
  throughout mean re-running the loader is safe.