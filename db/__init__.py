from .connection import get_db, db_manager
from .queries import (
    get_buoy_readings,
    get_readings_in_area,
    get_species_for_conditions,
    get_edna_detections,
    get_active_alerts,
    get_fishing_zones,
    insert_buoy_reading,
    insert_alert,
    get_all_species,
    get_grid_by_coordinates,
    get_all_grids
)

# Vector-search layer (pgvector) — see schema_pgvector.sql / embeddings.py.
# Imported lazily-safe: these only touch the DB when actually called, so
# importing db/ doesn't require pgvector to be installed/enabled unless
# one of these is used.
from .vector_queries import (
    semantic_search_species,
    find_similar_species,
    hybrid_species_search,
    find_similar_edna_sequences,
    semantic_search_research,
    semantic_search_grids,
    find_similar_grids,
    upsert_species_embedding,
    upsert_edna_embedding,
    insert_research_document,
    insert_research_chunk,
)

__all__ = [
    'get_db',
    'db_manager',
    'get_buoy_readings',
    'get_readings_in_area',
    'get_species_for_conditions',
    'get_edna_detections',
    'get_active_alerts',
    'get_fishing_zones',
    'insert_buoy_reading',
    'insert_alert',
    'get_all_species',
    'get_grid_by_coordinates',
    'get_all_grids',
    'semantic_search_species',
    'find_similar_species',
    'hybrid_species_search',
    'find_similar_edna_sequences',
    'semantic_search_research',
    'semantic_search_grids',
    'find_similar_grids',
    'upsert_species_embedding',
    'upsert_edna_embedding',
    'insert_research_document',
    'insert_research_chunk',
]
