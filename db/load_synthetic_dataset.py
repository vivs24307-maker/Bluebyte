"""
load_synthetic_dataset.py
==========================
Loads the CSV/JSON files produced by
db/synthetic_data/generate_synthetic_dataset.py into Postgres, end to
end:

  1. ocean_grids, species, fishing_zones, alerts  -> direct inserts
  2. buoy_readings, river_discharge               -> streamed through
     stream_loader.iter_csv_chunks() into
     etl_pipeline.OutlierPreservingETL (exercises the real outlier-
     flagging ingestion path, not a shortcut insert)
  3. edna_samples                                 -> inserted, resolving
     each row's scientific_name to species.id
  4. research_documents.json                      -> ingest_research.py
  5. embeddings for species / edna_samples         -> backfill_embeddings.py
  6. grid_ecological_profiles                       -> grid_profiles.py

Run schema_postgis.sql + schema_postgis_addendum.sql + schema_pgvector.sql
first (docker-compose.db.yml applies all three automatically on a
fresh container). Then:

    python -m db.synthetic_data.generate_synthetic_dataset
    python -m db.load_synthetic_dataset
"""

import argparse
import asyncio
import csv
import json
import os
from datetime import datetime, timezone

from db.connection import db_manager
from db.etl_pipeline import OutlierPreservingETL, compute_h3_index
from db.stream_loader import iter_csv_chunks
from db.ingest_research import ingest_batch_from_file
from db.backfill_embeddings import backfill_all
from db.grid_profiles import refresh_all_grid_profiles

DEFAULT_DATA_DIR = os.path.join(os.path.dirname(__file__), "synthetic_data", "data")


def _parse_bool(v):
    return str(v).strip().lower() in ("1", "true", "t", "yes")


async def load_ocean_grids(conn, path: str) -> int:
    n = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            await conn.execute(
                """
                INSERT INTO ocean_grids (grid_code, area_name, geom)
                VALUES ($1, $2, ST_SetSRID(ST_GeomFromText($3), 4326)::geography)
                ON CONFLICT (grid_code) DO NOTHING
                """,
                row["grid_code"], row["area_name"], row["polygon_wkt"],
            )
            n += 1
    return n


async def load_species(conn, path: str) -> dict:
    """Returns {scientific_name: species_id} for use resolving eDNA rows."""
    sci_to_id = {}
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            existing = await conn.fetchrow("SELECT id FROM species WHERE scientific_name = $1", row["scientific_name"])
            if existing:
                sci_to_id[row["scientific_name"]] = existing["id"]
                continue
            rec = await conn.fetchrow(
                """
                INSERT INTO species (common_name, scientific_name, family, habitat_type,
                                      conservation_status, min_sst, max_sst, min_salinity,
                                      max_salinity, min_depth, max_depth, commercial_value)
                VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12)
                RETURNING id
                """,
                row["common_name"], row["scientific_name"], row["family"], row["habitat_type"],
                row["conservation_status"], float(row["min_sst"]), float(row["max_sst"]),
                float(row["min_salinity"]), float(row["max_salinity"]),
                float(row["min_depth"]), float(row["max_depth"]), row["commercial_value"],
            )
            sci_to_id[row["scientific_name"]] = rec["id"]
    return sci_to_id


async def load_edna_samples(conn, path: str, sci_to_id: dict) -> int:
    n = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            species_id = sci_to_id.get(row["scientific_name"])
            if species_id is None:
                continue
            lat, lon = float(row["lat"]), float(row["lon"])
            await conn.execute(
                """
                INSERT INTO edna_samples (sample_id, species_id, geom, h3_index, marker_gene,
                                           sequence_fragment, detection_confidence, collection_date)
                VALUES ($1, $2, ST_SetSRID(ST_MakePoint($4, $3), 4326)::geography, $5, $6, $7, $8, $9)
                ON CONFLICT (sample_id) DO NOTHING
                """,
                row["sample_id"], species_id, lat, lon, compute_h3_index(lat, lon),
                row["marker_gene"], row["sequence_fragment"], float(row["detection_confidence"]),
                datetime.fromisoformat(row["collection_date"]),
            )
            n += 1
    return n


async def load_fishing_zones(conn, path: str) -> int:
    n = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            await conn.execute(
                """
                INSERT INTO fishing_zones (zone_name, geom, radius_km, pfz_score,
                                            dominant_species, valid_from, valid_until)
                VALUES ($1, ST_SetSRID(ST_MakePoint($3, $2), 4326)::geography, $4, $5, $6, $7, $8)
                """,
                row["zone_name"], float(row["lat"]), float(row["lon"]), float(row["radius_km"]),
                float(row["pfz_score"]), row["dominant_species"],
                datetime.fromisoformat(row["valid_from"]), datetime.fromisoformat(row["valid_until"]),
            )
            n += 1
    return n


async def load_alerts(conn, path: str) -> int:
    n = 0
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            await conn.execute(
                """
                INSERT INTO alerts (alert_type, severity, sensor_id, geom, message, created_at, acknowledged)
                VALUES ($1, $2, $3, ST_SetSRID(ST_MakePoint($5, $4), 4326)::geography, $6, $7, $8)
                """,
                row["alert_type"], row["severity"], row["sensor_id"],
                float(row["lat"]), float(row["lon"]), row["message"],
                datetime.fromisoformat(row["created_at"]), _parse_bool(row["acknowledged"]),
            )
            n += 1
    return n


async def load_buoy_and_river(etl: OutlierPreservingETL, data_dir: str) -> dict:
    buoy_totals = {"accepted": 0, "rejected_structural": 0, "flagged_outliers": 0}
    for chunk in iter_csv_chunks(os.path.join(data_dir, "buoy_readings.csv"), chunk_size=2000):
        result = await etl.ingest_buoy_batch(chunk)
        buoy_totals["accepted"] += result.accepted
        buoy_totals["rejected_structural"] += result.rejected_structural
        buoy_totals["flagged_outliers"] += result.flagged_outliers

    river_totals = {"accepted": 0, "rejected_structural": 0, "flagged_outliers": 0}
    river_column_map = {
        "station_name": "station_id",      # generator emits a human-readable
                                              # name; river_discharge.station_id
                                              # just needs a stable identifier,
                                              # a name string works fine as one
        "discharge_m3_s": "discharge_cumecs",  # same physical unit (cumecs IS
                                                  # m3/s), just a naming mismatch
        # turbidity_ntu has no corresponding column in river_discharge and is
        # intentionally dropped here rather than forced into water_level_m,
        # which is a different physical quantity — silently mismapping units
        # would be worse than just not loading it. Add a turbidity_ntu column
        # to the schema first if you want this value stored.
    }
    for chunk in iter_csv_chunks(os.path.join(data_dir, "river_discharge.csv"),
                                   chunk_size=2000, column_map=river_column_map):
        result = await etl.ingest_river_discharge_batch(chunk)
        river_totals["accepted"] += result.accepted
        river_totals["rejected_structural"] += result.rejected_structural
        river_totals["flagged_outliers"] += result.flagged_outliers

    return {"buoy": buoy_totals, "river": river_totals}


async def load_all(data_dir: str = DEFAULT_DATA_DIR, skip_embeddings: bool = False,
                     skip_grid_profiles: bool = False):
    pool = await db_manager.connect()
    etl = OutlierPreservingETL(pool)

    async with pool.acquire() as conn:
        n_grids = await load_ocean_grids(conn, os.path.join(data_dir, "ocean_grids.csv"))
        print(f"Loaded {n_grids} ocean grids")

        sci_to_id = await load_species(conn, os.path.join(data_dir, "species.csv"))
        print(f"Loaded {len(sci_to_id)} species")

        n_edna = await load_edna_samples(conn, os.path.join(data_dir, "edna_samples.csv"), sci_to_id)
        print(f"Loaded {n_edna} eDNA samples")

        n_zones = await load_fishing_zones(conn, os.path.join(data_dir, "fishing_zones.csv"))
        print(f"Loaded {n_zones} fishing zones")

        alerts_path = os.path.join(data_dir, "alerts.csv")
        if os.path.exists(alerts_path):
            n_alerts = await load_alerts(conn, alerts_path)
            print(f"Loaded {n_alerts} alerts")
        else:
            print(f"[WARN] {alerts_path} not found \u2014 skipping alerts "
                  f"(run generate_synthetic_dataset.py first)")

    totals = await load_buoy_and_river(etl, data_dir)
    print(f"Loaded buoy readings: {totals['buoy']}")
    print(f"Loaded river discharge: {totals['river']}")

    research_path = os.path.join(data_dir, "research_documents.json")
    if os.path.exists(research_path):
        results = await ingest_batch_from_file(research_path)
        print(f"Ingested {len(results)} research documents "
              f"({sum(r['chunk_count'] for r in results)} chunks)")

    if not skip_embeddings:
        await backfill_all()

    if not skip_grid_profiles:
        n_cells = await refresh_all_grid_profiles(window_days=90)
        print(f"Computed ecological profiles for {n_cells} grid cells")

    print("Synthetic dataset load complete.")


async def _main():
    parser = argparse.ArgumentParser(description="Load the synthetic dataset into Postgres")
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)
    parser.add_argument("--skip-embeddings", action="store_true")
    parser.add_argument("--skip-grid-profiles", action="store_true")
    args = parser.parse_args()
    await load_all(args.data_dir, args.skip_embeddings, args.skip_grid_profiles)


if __name__ == "__main__":
    asyncio.run(_main())