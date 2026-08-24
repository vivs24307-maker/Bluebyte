"""
graph_bridge.py
Hexagonal spatial gridding utility using direct H3 v4 API integration.

FIXED: every function here previously required string H3 addresses
(H3 v4's native representation), but etl_pipeline.compute_h3_index()
stores H3 cells as BIGINT integers (via h3.str_to_int(...)) to match
the buoy_readings/river_discharge/edna_samples.h3_index columns. Any
caller passing a DB row's h3_index straight into these functions would
fail or behave wrong. Now every function accepts either representation
(int or str) and normalizes internally, so it works correctly whether
called with a raw DB row's h3_index or a fresh lat/lon computation.
"""

import h3
from typing import List, Optional, Union

H3Index = Union[int, str]  # a cell can arrive as either representation


def _to_h3_str(cell: H3Index) -> str:
    """Normalizes either representation to H3's native string form,
    which is what the h3-py v4 API actually expects."""
    return h3.int_to_str(cell) if isinstance(cell, int) else cell


def get_h3_neighbors(h3_index: H3Index, k: int = 1, as_int: bool = True) -> List[H3Index]:
    """
    Returns neighboring H3 cells within ring distance k.

    as_int controls the return type: True (default) returns BIGINT-
    compatible ints, matching the DB's h3_index columns — set False if
    you specifically need H3's native string form for something else.
    """
    try:
        neighbors = h3.grid_disk(_to_h3_str(h3_index), k)
        if as_int:
            return [h3.str_to_int(n) for n in neighbors]
        return list(neighbors)
    except Exception as e:
        print(f"[H3 Bridge] Error computing disk for index {h3_index}: {e}")
        return []


def assign_lat_lon_to_h3(lat: float, lon: float, resolution: int = 7, as_int: bool = True) -> Optional[H3Index]:
    """
    Converts latitude and longitude coordinates into an H3 cell.

    as_int=True (default) matches etl_pipeline.compute_h3_index() and
    the DB's BIGINT h3_index columns — use the same resolution (6) as
    etl_pipeline.H3_RESOLUTION if you want cells to actually match rows
    already in the database; this function's own default (7) is finer-
    grained and will NOT line up with existing DB rows unless you pass
    resolution=6 explicitly.
    """
    try:
        cell_str = h3.latlng_to_cell(lat, lon, resolution)
        return h3.str_to_int(cell_str) if as_int else cell_str
    except Exception as e:
        print(f"[H3 Bridge] Indexing error at lat={lat}, lon={lon}: {e}")
        return None


def compute_h3_distance(origin_h3: H3Index, destination_h3: H3Index) -> Optional[int]:
    """
    Calculates grid distance (in cell steps) between two H3 cells.
    Accepts either int or str for each argument independently.
    """
    try:
        return h3.grid_distance(_to_h3_str(origin_h3), _to_h3_str(destination_h3))
    except Exception as e:
        print(f"[H3 Bridge] Distance computation failed: {e}")
        return None