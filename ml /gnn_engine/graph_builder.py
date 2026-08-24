"""
graph_builder.py
Bridges PostGIS database data into a PyTorch Geometric HeteroData graph structure.
Enforces deterministic spatial adjacency and consistent node IDs.
"""

import os
import sys
import asyncio
import numpy as np
import torch
from torch_geometric.data import HeteroData
from sklearn.neighbors import BallTree
import nest_asyncio

# Enable nested event loops for seamless integration in sync/async environments
nest_asyncio.apply()

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from db.connection import db_manager

async def fetch_graph_data_from_db():
    """
    Pulls real species catalog, spatial grids, and eDNA occurrences from PostGIS.
    """
    pool = await db_manager.connect()
    async with pool.acquire() as conn:
        species_records = await conn.fetch("""
            SELECT id, common_name, min_sst, max_sst, min_salinity, max_salinity, min_depth, max_depth 
            FROM species 
            ORDER BY id ASC
        """)
        
        grid_records = await conn.fetch("""
            SELECT id, grid_code, ST_Y(ST_Centroid(geom::geometry)) as lat, ST_X(ST_Centroid(geom::geometry)) as lon 
            FROM ocean_grids 
            ORDER BY id ASC
        """)
        
        edna_records = await conn.fetch("""
            SELECT id, species_id, detection_confidence, ST_Y(geom::geometry) as lat, ST_X(geom::geometry) as lon 
            FROM edna_samples 
            ORDER BY id ASC
        """)
        
        buoy_averages = await conn.fetch("""
            SELECT 
                COALESCE(AVG(sst), 28.5) as mean_sst, 
                COALESCE(AVG(salinity), 34.5) as mean_sal, 
                COALESCE(AVG(dissolved_oxygen), 6.2) as mean_do,
                COALESCE(AVG(chlorophyll_a), 1.5) as mean_chl
            FROM buoy_readings
        """)
        
    return species_records, grid_records, edna_records, buoy_averages[0]

def build_hetero_graph(seed: int = 42) -> HeteroData:
    """
    Constructs the multi-modal HeteroData object from database records.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    loop = asyncio.get_event_loop()
    species_records, grid_records, edna_records, buoy_stats = loop.run_until_complete(fetch_graph_data_from_db())

    data = HeteroData()

    # ---------------------------------------------------------
    # 1. Species Nodes (Thermal, Saline, and Depth Tolerances)
    # ---------------------------------------------------------
    data['species'].num_nodes = len(species_records)
    species_features = []
    species_id_map = {}

    for idx, row in enumerate(species_records):
        species_id_map[row['id']] = idx
        species_features.append([
            float(row['min_sst'] or 24.0),
            float(row['max_sst'] or 32.0),
            float(row['min_salinity'] or 30.0),
            float(row['max_salinity'] or 36.0),
            float(row['min_depth'] or 0.0),
            float(row['max_depth'] or 100.0)
        ])
        
    data['species'].x = torch.tensor(species_features, dtype=torch.float)

    # ---------------------------------------------------------
    # 2. Grid Nodes (Ocean Physics & Spatial Coordinates)
    # ---------------------------------------------------------
    data['grid'].num_nodes = len(grid_records)
    if len(grid_records) > 0:
        grid_coords = np.array([[float(r['lat']), float(r['lon'])] for r in grid_records])
        
        # Node features: [Lat, Lon, Baseline SST, Baseline Salinity, Baseline DO]
        base_sst = float(buoy_stats['mean_sst'])
        base_sal = float(buoy_stats['mean_sal'])
        base_do = float(buoy_stats['mean_do'])
        
        grid_feats = []
        for lat, lon in grid_coords:
            grid_feats.append([lat, lon, base_sst, base_sal, base_do])
            
        data['grid'].x = torch.tensor(grid_feats, dtype=torch.float)

        # Build Grid-to-Grid Spatial Adjacency (BallTree Haversine k-NN)
        tree = BallTree(np.radians(grid_coords), metric='haversine')
        k = min(5, len(grid_coords))
        _, indices = tree.query(np.radians(grid_coords), k=k)

        src, dst = [], []
        for i in range(len(indices)):
            for j in indices[i]:
                if i != j:
                    src.append(i)
                    dst.append(j)
        data['grid', 'adjacent_to', 'grid'].edge_index = torch.tensor([src, dst], dtype=torch.long)
    else:
        data['grid'].x = torch.empty((0, 5), dtype=torch.float)
        data['grid', 'adjacent_to', 'grid'].edge_index = torch.empty((2, 0), dtype=torch.long)

    # ---------------------------------------------------------
    # 3. eDNA Nodes & Cross-Domain Biological Edges
    # ---------------------------------------------------------
    data['edna'].num_nodes = len(edna_records)
    edna_features = []
    edna_src, edna_dst = [], []

    for i, r in enumerate(edna_records):
        conf = float(r['detection_confidence'] if r['detection_confidence'] is not None else 0.85)
        edna_features.append([conf, float(r['lat']), float(r['lon'])])
        
        # Link eDNA detection to registered species
        target_species_id = r['species_id']
        if target_species_id in species_id_map:
            edna_src.append(i)
            edna_dst.append(species_id_map[target_species_id])

    data['edna'].x = torch.tensor(edna_features, dtype=torch.float) if edna_features else torch.empty((0, 3), dtype=torch.float)
    data['edna', 'detects', 'species'].edge_index = (
        torch.tensor([edna_src, edna_dst], dtype=torch.long) if edna_src else torch.empty((2, 0), dtype=torch.long)
    )

    return data