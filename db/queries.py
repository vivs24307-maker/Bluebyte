"""
queries.py — Postgres/PostGIS/H3 replacement for the SQLite queries.

Every function keeps the SAME NAME AND SIGNATURE as the original
db/queries.py, so server/api/routes/*.py and websocket_manager.py can
import from db/__init__.py unchanged. What changed under the hood:

  * lat/lon BETWEEN scans      -> ST_DWithin on a geography column
    (fixes the "flat 111km/degree applied to longitude too" bug from
    the original review — geography math handles the cos(latitude)
    contraction correctly)
  * species filtered by SST only -> now filters BOTH sst and salinity
    (fixes the "salinity parameter silently ignored" bug)
  * grid_id FK lookup for eDNA  -> spatial ST_Within join against
    ocean_grids.geom (edna_samples no longer carries a grid_id FK in
    the new schema; grid_id here refers to ocean_grids.id, resolved
    spatially instead of via stored foreign key)
  * bbox grid lookup            -> ST_Contains on the grid polygon
  * new inserts compute h3_index at write time via h3-py

Rows are returned as plain dicts (asyncpg.Record supports dict()),
same shape callers already expect.
"""

from datetime import datetime, timezone
from typing import Optional
from db.connection import get_db
from db.etl_pipeline import compute_h3_index as _h3_index  # single source of
# truth for lat/lon -> H3 int conversion, so this can't drift out of sync
# with etl_pipeline.py's version again


async def get_buoy_readings(sensor_id: str, limit: int = 100):
    async with get_db() as db:
        rows = await db.fetch(
            """
            SELECT id, sensor_id, ts, ST_Y(geom::geometry) AS lat, ST_X(geom::geometry) AS lon,
                   h3_index, sst, salinity, chlorophyll_a, dissolved_oxygen, wave_height,
                   current_velocity, current_direction, is_outlier, outlier_method,
                   outlier_fields, z_score_sst, z_score_salinity, z_score_do
            FROM buoy_readings
            WHERE sensor_id = $1
            ORDER BY ts DESC
            LIMIT $2
            """,
            sensor_id, limit,
        )
        return [dict(row) for row in rows]


async def get_readings_in_area(lat: float, lon: float, radius_km: float, hours: int = 24):
    async with get_db() as db:
        rows = await db.fetch(
            """
            SELECT id, sensor_id, ts, ST_Y(geom::geometry) AS lat, ST_X(geom::geometry) AS lon,
                   h3_index, sst, salinity, chlorophyll_a, dissolved_oxygen, wave_height,
                   current_velocity, current_direction, is_outlier, outlier_method,
                   outlier_fields, z_score_sst, z_score_salinity, z_score_do
            FROM buoy_readings
            WHERE ST_DWithin(geom, ST_SetSRID(ST_MakePoint($2, $1), 4326)::geography, $3)
              AND ts >= now() - make_interval(hours => $4)
            ORDER BY ts DESC
            """,
            lat, lon, radius_km * 1000.0, hours,  # ST_DWithin on geography takes meters
        )
        return [dict(row) for row in rows]


async def get_species_for_conditions(sst: float, salinity: float = None):
    async with get_db() as db:
        rows = await db.fetch(
            """
            SELECT * FROM species
            WHERE $1 BETWEEN min_sst AND max_sst
              AND ($2::double precision IS NULL OR min_salinity IS NULL
                   OR $2 BETWEEN min_salinity AND max_salinity)
            """,
            sst, salinity,
        )
        return [dict(row) for row in rows]


async def get_edna_detections(grid_id: str):
    """grid_id is ocean_grids.id (UUID). Resolved spatially via
    ST_Within against the grid polygon, since edna_samples stores a
    geography point + h3_index rather than a stored grid_id FK."""
    async with get_db() as db:
        rows = await db.fetch(
            """
            SELECT e.id, e.sample_id, e.species_id, e.marker_gene, e.sequence_fragment,
                   e.detection_confidence, e.collection_date,
                   s.common_name, s.scientific_name
            FROM edna_samples e
            JOIN species s ON e.species_id = s.id
            JOIN ocean_grids g ON ST_Within(e.geom::geometry, g.geom::geometry)
            WHERE g.id = $1
            ORDER BY e.collection_date DESC
            """,
            grid_id,
        )
        return [dict(row) for row in rows]


async def get_active_alerts():
    async with get_db() as db:
        rows = await db.fetch(
            """
            SELECT id, alert_type, severity, sensor_id,
                   ST_Y(geom::geometry) AS lat, ST_X(geom::geometry) AS lon,
                   message, created_at, acknowledged
            FROM alerts
            WHERE acknowledged = FALSE
            ORDER BY created_at DESC
            """
        )
        return [dict(row) for row in rows]


async def get_recent_alerts(limit: int = 20, severity: Optional[str] = None):
    """Unlike get_active_alerts (unacknowledged only), this returns ALL
    alerts regardless of acknowledgement status — what /alerts/recent
    actually needs. Added because the route previously had nowhere in
    db/queries.py to get this from and fell back to hardcoded samples."""
    async with get_db() as db:
        if severity:
            rows = await db.fetch(
                """
                SELECT id, alert_type, severity, sensor_id,
                       ST_Y(geom::geometry) AS lat, ST_X(geom::geometry) AS lon,
                       message, created_at, acknowledged
                FROM alerts
                WHERE severity = $1
                ORDER BY created_at DESC
                LIMIT $2
                """,
                severity, limit,
            )
        else:
            rows = await db.fetch(
                """
                SELECT id, alert_type, severity, sensor_id,
                       ST_Y(geom::geometry) AS lat, ST_X(geom::geometry) AS lon,
                       message, created_at, acknowledged
                FROM alerts
                ORDER BY created_at DESC
                LIMIT $1
                """,
                limit,
            )
        return [dict(row) for row in rows]


async def get_alert_stats():
    """Summary counts for /alerts/stats, computed in SQL rather than
    pulling every row into Python."""
    async with get_db() as db:
        total_row = await db.fetchrow(
            "SELECT count(*) AS total, count(*) FILTER (WHERE NOT acknowledged) AS active FROM alerts"
        )
        by_severity = await db.fetch(
            "SELECT severity, count(*) AS n FROM alerts GROUP BY severity"
        )
        by_type = await db.fetch(
            "SELECT alert_type, count(*) AS n FROM alerts GROUP BY alert_type"
        )
        return {
            "total": total_row["total"],
            "active": total_row["active"],
            "by_severity": {r["severity"]: r["n"] for r in by_severity},
            "by_type": {r["alert_type"]: r["n"] for r in by_type},
        }


async def get_fishing_zones():
    async with get_db() as db:
        rows = await db.fetch(
            """
            SELECT id, zone_name, ST_Y(geom::geometry) AS center_lat,
                   ST_X(geom::geometry) AS center_lon, radius_km, pfz_score,
                   dominant_species, valid_from, valid_until
            FROM fishing_zones
            WHERE valid_from <= now() AND valid_until >= now()
            ORDER BY pfz_score DESC
            """
        )
        return [dict(row) for row in rows]


async def insert_buoy_reading(data: dict):
    """NOTE: for real ingestion, prefer etl_pipeline.OutlierPreservingETL
    (adds MAD-based outlier scoring). This function is kept for direct/
    manual inserts and API-driven single-row writes where outlier
    scoring against a rolling baseline isn't the point (e.g. a
    dashboard test insert)."""
    lat, lon = data["lat"], data["lon"]
    ts = data.get("timestamp") or data.get("ts") or datetime.now(timezone.utc)
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)

    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO buoy_readings (
                sensor_id, ts, geom, h3_index, sst, salinity, chlorophyll_a,
                dissolved_oxygen, wave_height, current_velocity, current_direction,
                is_outlier, outlier_method, z_score_sst, z_score_salinity, z_score_do
            ) VALUES (
                $1, $2, ST_SetSRID(ST_MakePoint($4, $3), 4326)::geography, $5,
                $6, $7, $8, $9, $10, $11, $12, $13, $14, $15, $16, $17
            )
            ON CONFLICT (sensor_id, ts) DO NOTHING
            """,
            data.get("sensor_id"), ts, lat, lon, _h3_index(lat, lon),
            data.get("sst"), data.get("salinity"), data.get("chlorophyll_a"),
            data.get("dissolved_oxygen"), data.get("wave_height"),
            data.get("current_velocity"), data.get("current_direction"),
            data.get("anomaly_flag", False), data.get("anomaly_reason"),
            data.get("z_score_sst", 0.0), data.get("z_score_do", 0.0),
        )


async def insert_alert(data: dict):
    async with get_db() as db:
        await db.execute(
            """
            INSERT INTO alerts (alert_type, severity, sensor_id, geom, message, created_at, acknowledged)
            VALUES ($1, $2, $3, ST_SetSRID(ST_MakePoint($5, $4), 4326)::geography, $6, $7, $8)
            """,
            data.get("alert_type"), data.get("severity"), data.get("sensor_id"),
            data.get("lat"), data.get("lon"), data.get("message"),
            data.get("created_at", datetime.now(timezone.utc)), data.get("acknowledged", False),
        )


async def get_all_species():
    async with get_db() as db:
        rows = await db.fetch("SELECT * FROM species")
        return [dict(row) for row in rows]


async def get_all_grids():
    """Backs server/api/routes/ocean.py's GET /ocean-data/grids — was
    calling a function that didn't exist in this module (always fell
    through to the hardcoded sample data). Returns grid centroid, since
    that's what the route's SAMPLE_GRIDS fallback shape expects
    (lat_center/lon_center), alongside the full polygon as GeoJSON for
    callers that want it."""
    async with get_db() as db:
        rows = await db.fetch(
            """
            SELECT id, grid_code, area_name,
                   ST_Y(ST_Centroid(geom::geometry)) AS lat_center,
                   ST_X(ST_Centroid(geom::geometry)) AS lon_center,
                   ST_AsGeoJSON(geom::geometry) AS geojson
            FROM ocean_grids
            ORDER BY grid_code
            """
        )
        return [dict(row) for row in rows]


async def get_grid_by_coordinates(lat: float, lon: float):
    async with get_db() as db:
        row = await db.fetchrow(
            """
            SELECT id, grid_code, area_name
            FROM ocean_grids
            WHERE ST_Contains(geom::geometry, ST_SetSRID(ST_MakePoint($2, $1), 4326))
            LIMIT 1
            """,
            lat, lon,
        )
        return dict(row) if row else None