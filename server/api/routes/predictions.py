"""
BlueByte AI — Predictions REST API Routes
Provides endpoints for Potential Fishing Zone (PFZ) predictions,
species biodiversity inference (GNN), and vessel route optimization.
"""
import sys
import os
import logging
import math
from typing import Optional
from fastapi import APIRouter, Query, HTTPException

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", ".."))

logger = logging.getLogger("API-Predictions")
router = APIRouter()


# ── Pre-computed sample PFZ zones for instant demo ───────────────────────────

SAMPLE_PFZ_ZONES = [
    {
        "id": "PFZ-001",
        "zone_name": "Goa Offshore Upwelling Zone",
        "center_lat": 15.1,
        "center_lon": 73.2,
        "radius_km": 45,
        "pfz_score": 0.91,
        "avg_sst": 28.2,
        "avg_chlorophyll": 2.8,
        "dominant_species": ["Indian Mackerel", "Oil Sardine"],
        "advisory": "High probability fishing zone — optimal SST and chlorophyll convergence detected.",
    },
    {
        "id": "PFZ-002",
        "zone_name": "Kerala Upwelling Belt",
        "center_lat": 9.8,
        "center_lon": 75.5,
        "radius_km": 55,
        "pfz_score": 0.86,
        "avg_sst": 27.8,
        "avg_chlorophyll": 3.1,
        "dominant_species": ["Oil Sardine", "Penaeid Shrimp"],
        "advisory": "Strong chlorophyll bloom — sardine aggregation likely.",
    },
    {
        "id": "PFZ-003",
        "zone_name": "Visakhapatnam Continental Shelf",
        "center_lat": 17.2,
        "center_lon": 83.8,
        "radius_km": 40,
        "pfz_score": 0.78,
        "avg_sst": 29.1,
        "avg_chlorophyll": 1.9,
        "dominant_species": ["Hilsa", "Bombay Duck"],
        "advisory": "Moderate fishing zone — suitable for hilsa during monsoon season.",
    },
    {
        "id": "PFZ-004",
        "zone_name": "Lakshadweep Deep Channel",
        "center_lat": 10.8,
        "center_lon": 72.1,
        "radius_km": 35,
        "pfz_score": 0.82,
        "avg_sst": 28.6,
        "avg_chlorophyll": 2.2,
        "dominant_species": ["Yellowfin Tuna", "Indian Mackerel"],
        "advisory": "Pelagic species corridor — recommended for pole-and-line tuna fishing.",
    },
    {
        "id": "PFZ-005",
        "zone_name": "Andaman East Basin",
        "center_lat": 12.0,
        "center_lon": 93.5,
        "radius_km": 50,
        "pfz_score": 0.74,
        "avg_sst": 29.4,
        "avg_chlorophyll": 1.7,
        "dominant_species": ["Yellowfin Tuna"],
        "advisory": "Deep-sea tuna grounds — long-line fishing recommended.",
    },
]


SAMPLE_SPECIES_PREDICTIONS = [
    {"species": "Indian Mackerel", "scientific_name": "Rastrelliger kanagurta", "confidence": 0.92, "habitat_match": "Excellent", "edna_confirmed": True},
    {"species": "Oil Sardine", "scientific_name": "Sardinella longiceps", "confidence": 0.87, "habitat_match": "Good", "edna_confirmed": True},
    {"species": "Yellowfin Tuna", "scientific_name": "Thunnus albacares", "confidence": 0.71, "habitat_match": "Moderate", "edna_confirmed": False},
    {"species": "Penaeid Shrimp", "scientific_name": "Penaeus indicus", "confidence": 0.65, "habitat_match": "Moderate", "edna_confirmed": True},
    {"species": "Bombay Duck", "scientific_name": "Harpadon nehereus", "confidence": 0.48, "habitat_match": "Low", "edna_confirmed": False},
    {"species": "Hilsa", "scientific_name": "Tenualosa ilisha", "confidence": 0.33, "habitat_match": "Low", "edna_confirmed": False},
]


@router.get("/predictions/pfz")
async def get_pfz_predictions():
    """
    Get Potential Fishing Zone predictions.
    Combines SST + Chlorophyll + Current data to identify optimal fishing areas.
    """
    # Try using the clustering algorithm
    try:
        from server.algorithms.clustering import identify_pfz_zones
        # Generate sample observations for clustering
        import random
        random.seed(42)
        observations = []
        # Cluster around known fishing zones
        for zone in SAMPLE_PFZ_ZONES:
            for _ in range(8):
                observations.append({
                    "lat": zone["center_lat"] + random.uniform(-0.3, 0.3),
                    "lon": zone["center_lon"] + random.uniform(-0.3, 0.3),
                    "sst": zone["avg_sst"] + random.uniform(-0.5, 0.5),
                    "chlorophyll_a": zone["avg_chlorophyll"] + random.uniform(-0.3, 0.3),
                })
        computed_zones = identify_pfz_zones(observations)
        if computed_zones:
            return {
                "status": "ok",
                "source": "dbscan_clustering",
                "count": len(computed_zones),
                "zones": [_normalize_pfz_zone(z, from_clustering=True) for z in computed_zones],
            }
    except Exception as e:
        logger.debug(f"Clustering not available, using sample PFZ data: {e}")

    return {
        "status": "ok",
        "source": "precomputed",
        "count": len(SAMPLE_PFZ_ZONES),
        "zones": [_normalize_pfz_zone(z, from_clustering=False) for z in SAMPLE_PFZ_ZONES],
    }


def _normalize_pfz_zone(zone: dict, from_clustering: bool) -> dict:
    """Both PFZ response paths (DBSCAN clustering and the hardcoded
    sample fallback) previously used different key names — neither
    matched what frontend/react_app/src/hooks/useApi.ts actually reads
    (z.lat, z.lon, z.species, z.confidence), so the map overlay was
    silently broken (NaN coordinates, species always 'Mixed')
    regardless of which backend path served the response.

    This normalizes both shapes into the one the frontend expects,
    while keeping the original fields alongside for anything else that
    might want them (e.g. bounding_box for the clustering path)."""
    if from_clustering:
        lat, lon = zone.get("centroid", (None, None))
        return {
            **zone,
            "lat": lat,
            "lon": lon,
            "species": (zone.get("dominant_species") or [None])[0]
                if isinstance(zone.get("dominant_species"), list) else zone.get("dominant_species"),
            "confidence": zone.get("pfz_score"),
        }
    else:
        dominant = zone.get("dominant_species")
        return {
            **zone,
            "lat": zone.get("center_lat"),
            "lon": zone.get("center_lon"),
            "species": dominant[0] if isinstance(dominant, list) and dominant else dominant,
            "confidence": zone.get("pfz_score"),
        }


@router.get("/predictions/species/{grid_id}")
async def predict_species_in_grid(
    grid_id: str,
    sst: float = Query(28.5, description="Current SST at grid location"),
    salinity: float = Query(35.0, description="Current salinity at grid location"),
    chlorophyll: float = Query(2.0, description="Current chlorophyll-a concentration"),
    dissolved_oxygen: float = Query(5.5, description="Current dissolved oxygen level"),
):
    """
    Predict which marine species are likely present in a given ocean grid cell
    using the GNN Knowledge Graph or fallback rule-based model.
    """
    # Try using the GNN predictor
    try:
        from ml.gnn_engine import MarineBiodiversityPredictor
        predictor = MarineBiodiversityPredictor()
        predictions = predictor.predict_species_in_grid(grid_id, sst, salinity, chlorophyll, dissolved_oxygen)
        biodiversity_score = predictor.get_biodiversity_score(grid_id)
        edna_refs = predictor.get_edna_cross_references(grid_id)
        return {
            "status": "ok",
            "source": "gnn_model",
            "grid_id": grid_id,
            "environmental_conditions": {
                "sst": sst, "salinity": salinity,
                "chlorophyll_a": chlorophyll, "dissolved_oxygen": dissolved_oxygen,
            },
            "biodiversity_score": biodiversity_score,
            "species_predictions": predictions,
            "edna_cross_references": edna_refs,
        }
    except Exception as e:
        logger.debug(f"GNN predictor not available, using sample data: {e}")

    # Fallback: return sample predictions with SST-adjusted confidence
    adjusted_predictions = []
    for sp in SAMPLE_SPECIES_PREDICTIONS:
        confidence = sp["confidence"]
        # Slightly adjust confidence based on SST
        if 27 <= sst <= 30:
            confidence = min(1.0, confidence * 1.05)
        elif sst < 25 or sst > 31:
            confidence = confidence * 0.7
        adjusted_predictions.append({**sp, "confidence": round(confidence, 3)})

    adjusted_predictions.sort(key=lambda x: x["confidence"], reverse=True)
    return {
        "status": "ok",
        "source": "rule_based_fallback",
        "grid_id": grid_id,
        "environmental_conditions": {
            "sst": sst, "salinity": salinity,
            "chlorophyll_a": chlorophyll, "dissolved_oxygen": dissolved_oxygen,
        },
        "biodiversity_score": round(sum(p["confidence"] for p in adjusted_predictions) / len(adjusted_predictions), 3),
        "species_predictions": adjusted_predictions,
    }


@router.get("/predictions/route")
async def get_optimal_route(
    start_lat: float = Query(..., description="Starting latitude"),
    start_lon: float = Query(..., description="Starting longitude"),
    end_lat: float = Query(..., description="Destination latitude"),
    end_lon: float = Query(..., description="Destination longitude"),
):
    """
    Calculate optimal vessel navigation route considering ocean currents.
    Uses A* pathfinding with flow-vector cost weighting.
    """
    try:
        from server.algorithms.pathfinding import OceanGrid
        grid = OceanGrid()
        grid.generate_sample_currents()
        route_result = grid.find_route(start_lat, start_lon, end_lat, end_lon)

        if route_result is None:
            raise HTTPException(status_code=404, detail="No viable route found between the given coordinates.")

        waypoints, total_cost = route_result

        # Calculate straight-line distance for fuel savings comparison
        def haversine(lat1, lon1, lat2, lon2):
            R = 6371.0
            dlat, dlon = math.radians(lat2 - lat1), math.radians(lon2 - lon1)
            a = math.sin(dlat / 2) ** 2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
            return R * 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        straight_distance = haversine(start_lat, start_lon, end_lat, end_lon)

        return {
            "status": "ok",
            "source": "astar_current_aware",
            "start": {"lat": start_lat, "lon": start_lon},
            "end": {"lat": end_lat, "lon": end_lon},
            "waypoints": [{"lat": w[0], "lon": w[1]} for w in waypoints],
            "total_cost": round(total_cost, 2),
            "straight_line_distance_km": round(straight_distance, 2),
            "num_waypoints": len(waypoints),
            "estimated_fuel_saving_percent": round(max(0, min(25, (1 - total_cost / (straight_distance * 1.3)) * 100)), 1),
        }
    except ImportError:
        raise HTTPException(status_code=503, detail="Pathfinding module not available.")
    except Exception as e:
        logger.error(f"Route calculation error: {e}")
        # Return a simple straight-line fallback
        return {
            "status": "ok",
            "source": "straight_line_fallback",
            "start": {"lat": start_lat, "lon": start_lon},
            "end": {"lat": end_lat, "lon": end_lon},
            "waypoints": [
                {"lat": start_lat, "lon": start_lon},
                {"lat": (start_lat + end_lat) / 2, "lon": (start_lon + end_lon) / 2},
                {"lat": end_lat, "lon": end_lon},
            ],
            "total_cost": 0,
            "num_waypoints": 3,
            "note": "Fallback straight-line route (current-aware routing unavailable)",
        }


@router.get("/predictions/biodiversity-map")
async def get_biodiversity_map():
    """
    Get biodiversity predictions for all ocean grids as a GeoJSON FeatureCollection.
    Used by the frontend to render the biodiversity heatmap overlay.
    """
    try:
        from ml.gnn_engine import MarineBiodiversityPredictor
        predictor = MarineBiodiversityPredictor()
        geojson = predictor.predict_all_grids()
        return {"status": "ok", "source": "gnn_model", "geojson": geojson}
    except Exception as e:
        logger.debug(f"GNN batch prediction not available: {e}")

    # Fallback GeoJSON
    features = []
    sample_grids = [
        ("Goa Shelf", 15.4, 73.8, 0.82), ("Mumbai Offshore", 18.9, 72.8, 0.68),
        ("Kochi Basin", 9.9, 76.3, 0.88), ("Visakhapatnam Shelf", 17.7, 83.5, 0.74),
        ("Lakshadweep Atoll", 10.6, 72.6, 0.91), ("Andaman Trench", 11.6, 92.7, 0.79),
        ("Gulf of Mannar", 9.0, 79.0, 0.85), ("Sundarbans Delta", 21.9, 88.9, 0.72),
    ]
    for name, lat, lon, score in sample_grids:
        features.append({
            "type": "Feature",
            "geometry": {"type": "Point", "coordinates": [lon, lat]},
            "properties": {"name": name, "biodiversity_score": score},
        })
    return {
        "status": "ok",
        "source": "sample",
        "geojson": {"type": "FeatureCollection", "features": features},
    }