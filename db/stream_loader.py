"""
stream_loader.py
=================
Bounded-memory streaming loaders for real marine datasets (NetCDF grids
from sources like INCOIS/ARGO/Copernicus, and large CSV exports from
buoy networks or CWC river-gauge stations), feeding into
etl_pipeline.OutlierPreservingETL in fixed-size chunks.

Nothing here loads a full file into memory: NetCDF is read via xarray
with chunked/lazy access (dask-backed), and CSV is streamed row-by-row
with csv.DictReader — so file size is decoupled from peak memory use.

Requires: pip install xarray netCDF4 dask asyncpg
"""

import asyncio
import csv
from datetime import datetime, timezone
from typing import Iterator

import numpy as np
import xarray as xr

from db.etl_pipeline import OutlierPreservingETL

DEFAULT_CHUNK_SIZE = 5000  # rows per batch handed to the ETL — tune against
                            # available memory and DB round-trip cost


# ---------------------------------------------------------------------
# CSV streaming
# ---------------------------------------------------------------------
def iter_csv_chunks(path: str, chunk_size: int = DEFAULT_CHUNK_SIZE,
                     column_map: dict = None) -> Iterator[list[dict]]:
    """
    Yields lists of row-dicts of at most `chunk_size` rows, reading the
    file incrementally — never holds more than one chunk in memory.

    column_map lets you rename source CSV headers to the ETL's expected
    keys, e.g. {"buoy_id": "sensor_id", "latitude": "lat", "longitude": "lon"}.
    """
    column_map = column_map or {}
    batch = []
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for raw_row in reader:
            row = {column_map.get(k, k): v for k, v in raw_row.items()}
            # Type coercion — CSV gives you strings for everything
            for numeric_field in ("lat", "lon", "sst", "salinity", "chlorophyll_a",
                                    "dissolved_oxygen", "wave_height",
                                    "current_velocity", "current_direction",
                                    "discharge_cumecs", "water_level_m"):
                if numeric_field in row and row[numeric_field] not in (None, ""):
                    try:
                        row[numeric_field] = float(row[numeric_field])
                    except ValueError:
                        row[numeric_field] = None  # let structural validation
                                                     # in the ETL catch/reject it,
                                                     # rather than silently
                                                     # coercing a bad value to 0
            batch.append(row)
            if len(batch) >= chunk_size:
                yield batch
                batch = []
    if batch:
        yield batch


# ---------------------------------------------------------------------
# NetCDF streaming (e.g. gridded SST/salinity products)
# ---------------------------------------------------------------------
def iter_netcdf_chunks(path: str, sensor_id_prefix: str = "GRID",
                        chunk_size: int = DEFAULT_CHUNK_SIZE,
                        time_dim: str = "time", lat_dim: str = "lat",
                        lon_dim: str = "lon",
                        var_map: dict = None) -> Iterator[list[dict]]:
    """
    Streams a gridded NetCDF file (time x lat x lon, e.g. SST/salinity
    reanalysis products) into flat row-dicts, one time-slice at a time,
    using xarray's lazy/dask-backed loading so the whole grid is never
    materialized in memory at once.

    var_map maps NetCDF variable names to the ETL's expected fields,
    e.g. {"analysed_sst": "sst", "so": "salinity"}.
    """
    var_map = var_map or {"sst": "sst", "salinity": "salinity"}

    # chunks={} enables dask lazy loading — data is only pulled from disk
    # as each .values access below actually touches it, one time-step
    # at a time, not the full 3D array up front.
    ds = xr.open_dataset(path, chunks={})

    try:
        times = ds[time_dim].values
        lats = ds[lat_dim].values
        lons = ds[lon_dim].values

        batch = []
        for t_idx, t_val in enumerate(times):
            slice_ds = ds.isel({time_dim: t_idx})  # lazy — pulls only this slice

            ts = _to_utc_datetime(t_val)

            # Vectorized extraction of this one time-slice, then flatten
            # to per-cell rows. Still bounded: one lat x lon slice at a
            # time, not the full time series.
            field_arrays = {}
            for nc_var, out_field in var_map.items():
                if nc_var in slice_ds:
                    field_arrays[out_field] = np.asarray(slice_ds[nc_var].values)

            if not field_arrays:
                continue

            for i, lat in enumerate(lats):
                for j, lon in enumerate(lons):
                    row = {
                        "sensor_id": f"{sensor_id_prefix}-{lat:.2f}-{lon:.2f}",
                        "ts": ts,
                        "lat": float(lat),
                        "lon": float(lon),
                    }
                    has_value = False
                    for out_field, arr in field_arrays.items():
                        val = arr[i, j]
                        if not np.isnan(val):
                            row[out_field] = float(val)
                            has_value = True
                    if not has_value:
                        continue  # skip pure-fill/land-mask cells; this is a
                                   # missing-data skip, not an outlier rejection —
                                   # no statistical judgment involved

                    batch.append(row)
                    if len(batch) >= chunk_size:
                        yield batch
                        batch = []

        if batch:
            yield batch
    finally:
        ds.close()


def _to_utc_datetime(t_val) -> datetime:
    ts = np.datetime64(t_val, "s").astype("datetime64[s]").astype(datetime)
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return ts


# ---------------------------------------------------------------------
# Orchestration: stream a file straight into the outlier-preserving ETL
# ---------------------------------------------------------------------
async def stream_csv_to_db(etl: OutlierPreservingETL, path: str,
                            column_map: dict = None,
                            chunk_size: int = DEFAULT_CHUNK_SIZE,
                            target: str = "buoy") -> dict:
    """target: 'buoy' or 'river' — routes to the matching ETL method."""
    totals = {"accepted": 0, "rejected_structural": 0, "flagged_outliers": 0}
    for chunk in iter_csv_chunks(path, chunk_size=chunk_size, column_map=column_map):
        if target == "river":
            result = await etl.ingest_river_discharge_batch(chunk)
        else:
            result = await etl.ingest_buoy_batch(chunk)
        totals["accepted"] += result.accepted
        totals["rejected_structural"] += result.rejected_structural
        totals["flagged_outliers"] += result.flagged_outliers
    return totals


async def stream_netcdf_to_db(etl: OutlierPreservingETL, path: str,
                               var_map: dict = None,
                               chunk_size: int = DEFAULT_CHUNK_SIZE) -> dict:
    totals = {"accepted": 0, "rejected_structural": 0, "flagged_outliers": 0}
    for chunk in iter_netcdf_chunks(path, var_map=var_map, chunk_size=chunk_size):
        result = await etl.ingest_buoy_batch(chunk)
        totals["accepted"] += result.accepted
        totals["rejected_structural"] += result.rejected_structural
        totals["flagged_outliers"] += result.flagged_outliers
    return totals


async def _demo():
    import asyncpg
    pool = await asyncpg.create_pool(dsn="postgresql://bluebyte:bluebyte_dev@localhost:5432/bluebyte")
    etl = OutlierPreservingETL(pool)

    # Example: a large CWC river-gauge CSV export, streamed in 5k-row chunks
    totals = await stream_csv_to_db(
        etl, "river_discharge_export.csv",
        column_map={"station": "station_id", "latitude": "lat", "longitude": "lon"},
        target="river",
    )
    print("River CSV ingest totals:", totals)

    # Example: a gridded SST NetCDF product
    totals = await stream_netcdf_to_db(
        etl, "sst_grid.nc", var_map={"analysed_sst": "sst"},
    )
    print("NetCDF SST ingest totals:", totals)

    await pool.close()


if __name__ == "__main__":
    asyncio.run(_demo())