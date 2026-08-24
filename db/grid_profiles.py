"""
grid_profiles.py
=================
Computes the fused physical + biological "ecological profile" per H3
cell (grid_ecological_profiles, see schema_pgvector.sql section 4) and
its embedding, so semantic_search_grids()/find_similar_grids() in
vector_queries.py have something to search over.

Run as a batch job (cron / Airflow / manual) on a schedule — this is
deliberately NOT computed inline during buoy ingestion, for the same
reason graph_bridge.py's edge computation isn't dual-written at ingest
time: sensor ingestion needs to stay fast and simple, and "recompute
the fused ecological summary for a cell" is an analytical job that can
run independently on its own cadence (e.g. hourly/daily).

Usage:
    python -m db.grid_profiles              # refresh all cells with data today
    python -m db.grid_profiles --days 7      # use a 7-day averaging window
"""

import argparse
import asyncio
from datetime import date, datetime, timedelta, timezone

from db.connection import db_manager
from db.embeddings import build_grid_profile_text, embed_text, vector_literal

DEFAULT_WINDOW_DAYS = 1


async def compute_and_upsert_profile(conn, h3_index: int, window_days: int = DEFAULT_WINDOW_DAYS,
                                       profile_date: date = None):
    profile_date = profile_date or datetime.now(timezone.utc).date()

    env = await conn.fetchrow(
        """
        SELECT avg(sst) AS avg_sst, avg(salinity) AS avg_salinity,
               avg(chlorophyll_a) AS avg_chlorophyll, avg(dissolved_oxygen) AS avg_do,
               count(*) AS reading_count
        FROM buoy_readings
        WHERE h3_index = $1 AND ts >= now() - make_interval(days => $2)
        """,
        h3_index, window_days,
    )

    species_rows = await conn.fetch(
        """
        SELECT DISTINCT s.common_name
        FROM edna_samples e
        JOIN species s ON e.species_id = s.id
        WHERE e.h3_index = $1
        """,
        h3_index,
    )
    species_names = [r["common_name"] for r in species_rows]

    profile_text = build_grid_profile_text(
        avg_sst=env["avg_sst"], avg_salinity=env["avg_salinity"],
        avg_chlorophyll=env["avg_chlorophyll"], avg_do=env["avg_do"],
        species_richness=len(species_names),
        dominant_species=species_names[:5] if species_names else None,
    )
    vec = embed_text(profile_text)

    await conn.execute(
        """
        INSERT INTO grid_ecological_profiles (
            h3_index, profile_date, avg_sst, avg_salinity, avg_chlorophyll,
            avg_dissolved_oxygen, reading_count, species_richness,
            dominant_species, profile_text, profile_embedding
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11::vector)
        ON CONFLICT (h3_index, profile_date) DO UPDATE SET
            avg_sst = EXCLUDED.avg_sst, avg_salinity = EXCLUDED.avg_salinity,
            avg_chlorophyll = EXCLUDED.avg_chlorophyll,
            avg_dissolved_oxygen = EXCLUDED.avg_dissolved_oxygen,
            reading_count = EXCLUDED.reading_count,
            species_richness = EXCLUDED.species_richness,
            dominant_species = EXCLUDED.dominant_species,
            profile_text = EXCLUDED.profile_text,
            profile_embedding = EXCLUDED.profile_embedding,
            computed_at = now()
        """,
        h3_index, profile_date, env["avg_sst"], env["avg_salinity"], env["avg_chlorophyll"],
        env["avg_do"], env["reading_count"], len(species_names),
        species_names or None, profile_text, vector_literal(vec),
    )
    return profile_text


async def refresh_all_grid_profiles(window_days: int = DEFAULT_WINDOW_DAYS) -> int:
    pool = await db_manager.connect()
    async with pool.acquire() as conn:
        cells = await conn.fetch(
            """
            SELECT DISTINCT h3_index FROM buoy_readings
            WHERE ts >= now() - make_interval(days => $1)
            UNION
            SELECT DISTINCT h3_index FROM edna_samples WHERE h3_index IS NOT NULL
            """,
            window_days,
        )
        for row in cells:
            await compute_and_upsert_profile(conn, row["h3_index"], window_days=window_days)
        return len(cells)


async def _main():
    parser = argparse.ArgumentParser(description="Refresh grid_ecological_profiles + embeddings")
    parser.add_argument("--days", type=int, default=DEFAULT_WINDOW_DAYS,
                         help="Averaging window in days for buoy readings (default: %(default)s)")
    args = parser.parse_args()
    n = await refresh_all_grid_profiles(window_days=args.days)
    print(f"Refreshed ecological profiles for {n} H3 cells (window: {args.days}d)")


if __name__ == "__main__":
    asyncio.run(_main())
