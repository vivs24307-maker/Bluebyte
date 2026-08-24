"""
generate_synthetic_dataset.py
Generates reproducible synthetic marine datasets within realistic oceanic bounds
(Arabian Sea, Bay of Bengal, Laccadive Sea) and exports CSV/JSON files.
"""

import os
import csv
import json
import random
import math
from datetime import datetime, timedelta, timezone

# Distinct marine bounding boxes strictly off the Indian subcontinent
MARINE_ZONES = [
    {"name": "Arabian Sea (West Coast)", "lat": (10.0, 20.0), "lon": (66.0, 72.0)},
    {"name": "Bay of Bengal (East Coast)", "lat": (11.0, 20.0), "lon": (82.0, 89.0)},
    {"name": "Laccadive Sea (South)", "lat": (6.0, 9.5), "lon": (74.5, 78.5)}
]

# Canonical 13-species registry with thermal/salinity envelopes
SPECIES_REGISTRY = [
    {"id": 1, "common_name": "Indian Mackerel", "scientific_name": "Rastrelliger kanagurta", "family": "Scombridae", "habitat_type": "Pelagic", "conservation_status": "Least Concern", "min_sst": 26.0, "max_sst": 31.0, "min_salinity": 33.0, "max_salinity": 36.0, "min_depth": 0, "max_depth": 100, "commercial_value": "High", "rarity_factor": 1.0},
    {"id": 2, "common_name": "Oil Sardine", "scientific_name": "Sardinella longiceps", "family": "Clupeidae", "habitat_type": "Pelagic", "conservation_status": "Least Concern", "min_sst": 27.0, "max_sst": 30.5, "min_salinity": 33.5, "max_salinity": 35.5, "min_depth": 0, "max_depth": 50, "commercial_value": "High", "rarity_factor": 1.0},
    {"id": 3, "common_name": "Hilsa Shad", "scientific_name": "Tenualosa ilisha", "family": "Clupeidae", "habitat_type": "Anadromous", "conservation_status": "Least Concern", "min_sst": 25.0, "max_sst": 30.0, "min_salinity": 10.0, "max_salinity": 34.0, "min_depth": 0, "max_depth": 50, "commercial_value": "Very High", "rarity_factor": 0.8},
    {"id": 4, "common_name": "Bombay Duck", "scientific_name": "Harpadon nehereus", "family": "Synodontidae", "habitat_type": "Demersal", "conservation_status": "Least Concern", "min_sst": 24.0, "max_sst": 29.0, "min_salinity": 32.0, "max_salinity": 35.0, "min_depth": 10, "max_depth": 200, "commercial_value": "Medium", "rarity_factor": 0.9},
    {"id": 5, "common_name": "Yellowfin Tuna", "scientific_name": "Thunnus albacares", "family": "Scombridae", "habitat_type": "Pelagic", "conservation_status": "Near Threatened", "min_sst": 20.0, "max_sst": 30.0, "min_salinity": 34.0, "max_salinity": 36.5, "min_depth": 0, "max_depth": 250, "commercial_value": "Very High", "rarity_factor": 0.5},
    {"id": 6, "common_name": "Penaeid Shrimp", "scientific_name": "Penaeus indicus", "family": "Penaeidae", "habitat_type": "Benthic", "conservation_status": "Least Concern", "min_sst": 26.0, "max_sst": 32.0, "min_salinity": 30.0, "max_salinity": 35.0, "min_depth": 5, "max_depth": 80, "commercial_value": "High", "rarity_factor": 1.0},
    {"id": 7, "common_name": "Skipjack Tuna", "scientific_name": "Katsuwonus pelamis", "family": "Scombridae", "habitat_type": "Pelagic", "conservation_status": "Least Concern", "min_sst": 22.0, "max_sst": 30.0, "min_salinity": 34.0, "max_salinity": 37.0, "min_depth": 0, "max_depth": 260, "commercial_value": "High", "rarity_factor": 0.7},
    {"id": 8, "common_name": "Silver Pomfret", "scientific_name": "Pampus argenteus", "family": "Stromateidae", "habitat_type": "Benthopelagic", "conservation_status": "Least Concern", "min_sst": 25.0, "max_sst": 30.0, "min_salinity": 32.0, "max_salinity": 35.5, "min_depth": 5, "max_depth": 110, "commercial_value": "Very High", "rarity_factor": 0.6},
    {"id": 9, "common_name": "Cobia", "scientific_name": "Rachycentron canadum", "family": "Rachycentridae", "habitat_type": "Pelagic", "conservation_status": "Least Concern", "min_sst": 24.0, "max_sst": 32.0, "min_salinity": 31.0, "max_salinity": 36.0, "min_depth": 0, "max_depth": 150, "commercial_value": "Medium", "rarity_factor": 0.6},
    {"id": 10, "common_name": "Green Sea Turtle", "scientific_name": "Chelonia mydas", "family": "Cheloniidae", "habitat_type": "Reef/Pelagic", "conservation_status": "Endangered", "min_sst": 22.0, "max_sst": 30.0, "min_salinity": 32.0, "max_salinity": 36.0, "min_depth": 0, "max_depth": 80, "commercial_value": "Protected", "rarity_factor": 0.2},
    {"id": 11, "common_name": "Whale Shark", "scientific_name": "Rhincodon typus", "family": "Rhincodontidae", "habitat_type": "Pelagic", "conservation_status": "Endangered", "min_sst": 21.0, "max_sst": 30.0, "min_salinity": 33.0, "max_salinity": 36.0, "min_depth": 0, "max_depth": 500, "commercial_value": "Protected", "rarity_factor": 0.15},
    {"id": 12, "common_name": "Blacktip Reef Shark", "scientific_name": "Carcharhinus melanopterus", "family": "Carcharhinidae", "habitat_type": "Reef", "conservation_status": "Vulnerable", "min_sst": 24.0, "max_sst": 31.0, "min_salinity": 33.0, "max_salinity": 36.0, "min_depth": 0, "max_depth": 75, "commercial_value": "Protected", "rarity_factor": 0.25},
    {"id": 13, "common_name": "Seahorse", "scientific_name": "Hippocampus kuda", "family": "Syngnathidae", "habitat_type": "Benthic", "conservation_status": "Vulnerable", "min_sst": 25.0, "max_sst": 31.0, "min_salinity": 30.0, "max_salinity": 35.0, "min_depth": 1, "max_depth": 50, "commercial_value": "Protected", "rarity_factor": 0.3}
]

def get_random_marine_coordinate(rng):
    zone = rng.choice(MARINE_ZONES)
    lat = round(rng.uniform(*zone["lat"]), 4)
    lon = round(rng.uniform(*zone["lon"]), 4)
    return lat, lon

def generate_ocean_grids(rng, num_grids=20):
    grids = []
    for i in range(num_grids):
        lat_base, lon_base = get_random_marine_coordinate(rng)
        polygon_wkt = (
            f"POLYGON(({lon_base} {lat_base}, {lon_base+0.5:.4f} {lat_base}, "
            f"{lon_base+0.5:.4f} {lat_base+0.5:.4f}, {lon_base} {lat_base+0.5:.4f}, {lon_base} {lat_base}))"
        )
        grids.append({
            "grid_code": f"GRID-{i:03d}",
            "area_name": f"Marine Sector {i:02d}",
            "polygon_wkt": polygon_wkt
        })
    return grids

def _reference_sequence(scientific_name: str, length: int = 60) -> str:
    """Deterministic per-species reference sequence, seeded off the
    species ID rather than the shared generator rng — so this is
    independently callable (e.g. from demo_vector_search.py) without
    needing generate_edna_samples() to have run first, and always
    returns the same sequence for a given species regardless of call
    order elsewhere."""
    for sp in SPECIES_REGISTRY:
        if sp["scientific_name"] == scientific_name:
            local_rng = random.Random(1000 + sp["id"])
            return "".join(local_rng.choices(["A", "C", "T", "G"], k=length))
    raise ValueError(f"Unknown species: {scientific_name}")


def generate_edna_samples(rng, species_list):
    samples = []
    sample_counter = 0
    for sp in species_list:
        count = max(2, int(12 * sp["rarity_factor"]))
        base_seq = _reference_sequence(sp["scientific_name"])  # same source
        # of truth demo_vector_search.py's probe sequence is built from,
        # so a probe genuinely resembles this species' seeded samples

        for _ in range(count):
            lat, lon = get_random_marine_coordinate(rng)
            seq_chars = list(base_seq)
            for _ in range(rng.randint(1, 3)):
                mut_idx = rng.randint(0, len(seq_chars) - 1)
                seq_chars[mut_idx] = rng.choice(["A", "C", "T", "G"])
            
            samples.append({
                "sample_id": f"EDNA-{sample_counter:04d}",
                "species_id": sp["id"],
                "scientific_name": sp["scientific_name"],
                "lat": lat,
                "lon": lon,
                "marker_gene": rng.choice(["COI", "12S", "16S"]),
                "sequence_fragment": "".join(seq_chars),
                "detection_confidence": round(rng.uniform(0.72, 0.99), 4),
                "collection_date": (datetime.now(timezone.utc) - timedelta(days=rng.randint(1, 45))).isoformat()
            })
            sample_counter += 1
    return samples

def generate_buoy_readings(rng, num_normal=120, num_outliers=8):
    sensor_ids = [f"BUOY-{i:02d}" for i in range(1, 7)]
    now = datetime.now(timezone.utc)
    readings = []
    
    # 1. Normal readings (all use "ts")
    for _ in range(num_normal):
        lat, lon = get_random_marine_coordinate(rng)
        ts = now - timedelta(hours=rng.randint(1, 120))
        readings.append({
            "sensor_id": rng.choice(sensor_ids),
            "ts": ts.isoformat(),
            "lat": lat,
            "lon": lon,
            "sst": round(rng.uniform(27.0, 30.5), 2),
            "salinity": round(rng.uniform(33.5, 35.5), 2),
            "chlorophyll_a": round(rng.uniform(0.2, 4.5), 2),
            "dissolved_oxygen": round(rng.uniform(5.5, 7.8), 2),
            "wave_height": round(rng.uniform(0.5, 2.5), 2),
            "current_velocity": round(rng.uniform(0.2, 1.8), 2),
            "current_direction": round(rng.uniform(0, 360), 1),
            "is_injected_outlier": False
        })
        
    # 2. Extreme shocks / Outliers (all use "ts")
    for i in range(num_outliers):
        lat, lon = get_random_marine_coordinate(rng)
        ts = now - timedelta(hours=rng.randint(1, 120))
        is_heatwave = (i % 2 == 0)
        
        readings.append({
            "sensor_id": rng.choice(sensor_ids),
            "ts": ts.isoformat(),
            "lat": lat,
            "lon": lon,
            "sst": round(rng.uniform(34.5, 37.8), 2) if is_heatwave else round(rng.uniform(27.0, 30.5), 2),
            "salinity": round(rng.uniform(33.5, 35.5), 2),
            "chlorophyll_a": round(rng.uniform(0.2, 4.5), 2),
            "dissolved_oxygen": round(rng.uniform(1.0, 2.2), 2) if not is_heatwave else round(rng.uniform(5.5, 7.8), 2),
            "wave_height": round(rng.uniform(0.5, 2.5), 2),
            "current_velocity": round(rng.uniform(0.2, 1.8), 2),
            "current_direction": round(rng.uniform(0, 360), 1),
            "is_injected_outlier": True
        })
        
    return readings

def generate_fishing_zones(rng, num_zones=8):
    zones = []
    now = datetime.now(timezone.utc)
    for i in range(num_zones):
        lat, lon = get_random_marine_coordinate(rng)
        zones.append({
            "zone_name": f"PFZ-{i+1:02d}",
            "lat": lat,
            "lon": lon,
            "radius_km": round(rng.uniform(15.0, 45.0), 1),
            "pfz_score": round(rng.uniform(0.75, 0.98), 3),
            "dominant_species": rng.choice(["Indian Mackerel", "Yellowfin Tuna", "Oil Sardine", "Silver Pomfret"]),
            "valid_from": now.isoformat(),
            "valid_until": (now + timedelta(days=3)).isoformat()
        })
    return zones

def generate_river_discharge(rng, num_stations=4):
    stations = [
        {"name": "Ganges Estuary", "lat": 21.8, "lon": 88.1},
        {"name": "Brahmaputra Outflow", "lat": 22.1, "lon": 90.5},
        {"name": "Godavari Mouth", "lat": 16.7, "lon": 82.3},
        {"name": "Narmada Gulf", "lat": 21.6, "lon": 72.6}
    ]
    readings = []
    now = datetime.now(timezone.utc)
    for st in stations[:num_stations]:
        for h in range(48):
            ts = now - timedelta(hours=h)
            base_flow = 12000.0 if "Ganges" in st["name"] else 4500.0
            readings.append({
                "station_name": st["name"],
                "lat": st["lat"],
                "lon": st["lon"],
                "ts": ts.isoformat(),
                "discharge_m3_s": round(base_flow + rng.uniform(-500, 1500), 2),
                "turbidity_ntu": round(rng.uniform(15.0, 85.0), 1)
            })
    return readings

def generate_alerts(rng, num_alerts=6):
    """Realistic mixed alert set — anomaly, vessel, and biodiversity
    types, matching what server/api/routes/alerts.py and the frontend
    expect (alert_type, severity, sensor_id, lat, lon, message,
    created_at, acknowledged)."""
    templates = [
        ("MARINE_HEATWAVE", "critical", "Rapid SST spike (+{delta}\u00b0C) detected \u2014 potential coral bleaching risk"),
        ("HYPOXIA_DEAD_ZONE", "high", "Severe hypoxia (DO={do} mg/L) \u2014 fish kill / biomass mortality risk"),
        ("ALGAL_BLOOM_BURST", "high", "Abnormal chlorophyll surge (Chl-a={chl} mg/m\u00b3) \u2014 possible harmful algal bloom"),
        ("EQUIPMENT_FAILURE", "medium", "Sensor connection lost \u2014 no readings for over 2 hours"),
        ("RIVER_DISCHARGE_SPIKE", "high", "Discharge {discharge} m\u00b3/s exceeds seasonal baseline \u2014 possible flood overflow"),
        ("LOW_SALINITY_EVENT", "medium", "Salinity dropped below {sal} PSU \u2014 possible freshwater intrusion"),
    ]
    now = datetime.now(timezone.utc)
    alerts = []
    for i in range(num_alerts):
        alert_type, severity, template = templates[i % len(templates)]
        lat, lon = get_random_marine_coordinate(rng)
        message = template.format(
            delta=round(rng.uniform(2.5, 5.0), 1), do=round(rng.uniform(1.0, 2.2), 1),
            chl=round(rng.uniform(6.0, 12.0), 1), discharge=round(rng.uniform(13000, 18000)),
            sal=round(rng.uniform(28.0, 31.0), 1),
        )
        alerts.append({
            "alert_type": alert_type,
            "severity": severity,
            "sensor_id": rng.choice([f"BUOY-{n:02d}" for n in range(1, 7)]),
            "lat": lat,
            "lon": lon,
            "message": message,
            "created_at": (now - timedelta(hours=rng.randint(1, 48))).isoformat(),
            "acknowledged": rng.random() < 0.3,
        })
    return alerts


def generate_research_documents(rng, species_list, num_docs=5):
    """Short synthetic research abstracts referencing real species from
    the registry, for db/ingest_research.py's vector-search demo."""
    docs = []
    for i in range(num_docs):
        sp = rng.choice(species_list)
        docs.append({
            "doc_id": f"RESEARCH-{i:03d}",
            "title": f"Habitat range and thermal tolerance of {sp['common_name']} in Indian coastal waters",
            "text": (
                f"{sp['scientific_name']} ({sp['common_name']}) is commonly observed in "
                f"waters between {sp['min_sst']}\u2013{sp['max_sst']}\u00b0C SST and "
                f"{sp['min_salinity']}\u2013{sp['max_salinity']} PSU salinity. Conservation status: "
                f"{sp['conservation_status']}. This synthetic abstract exists to exercise the "
                f"research-document ingestion and vector search pipeline."
            ),
            "source": "synthetic_dataset",
            "published_date": (datetime.now(timezone.utc) - timedelta(days=rng.randint(30, 800))).isoformat(),
        })
    return docs


def main():
    rng = random.Random(42)
    output_dir = os.path.join(os.path.dirname(__file__), "data")
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Species CSV
    species_path = os.path.join(output_dir, "species.csv")
    with open(species_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SPECIES_REGISTRY[0].keys())
        writer.writeheader()
        writer.writerows(SPECIES_REGISTRY)
    print(f"Generated {len(SPECIES_REGISTRY)} species -> {species_path}")

    # 2. Ocean Grids CSV
    grids = generate_ocean_grids(rng, 20)
    grids_path = os.path.join(output_dir, "ocean_grids.csv")
    with open(grids_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=grids[0].keys())
        writer.writeheader()
        writer.writerows(grids)
    print(f"Generated {len(grids)} ocean grids -> {grids_path}")

    # 3. eDNA Samples CSV
    edna = generate_edna_samples(rng, SPECIES_REGISTRY)
    edna_path = os.path.join(output_dir, "edna_samples.csv")
    with open(edna_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=edna[0].keys())
        writer.writeheader()
        writer.writerows(edna)
    print(f"Generated {len(edna)} eDNA samples -> {edna_path}")

    # 4. Buoy Readings CSV
    buoys = generate_buoy_readings(rng, 120, 8)
    buoys_path = os.path.join(output_dir, "buoy_readings.csv")
    with open(buoys_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=buoys[0].keys())
        writer.writeheader()
        writer.writerows(buoys)
    print(f"Generated {len(buoys)} buoy telemetry points -> {buoys_path}")

    # 5. Fishing Zones CSV
    zones = generate_fishing_zones(rng, 8)
    zones_path = os.path.join(output_dir, "fishing_zones.csv")
    with open(zones_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=zones[0].keys())
        writer.writeheader()
        writer.writerows(zones)
    print(f"Generated {len(zones)} fishing zones -> {zones_path}")

    # 6. River Discharge CSV
    rivers = generate_river_discharge(rng, 4)
    rivers_path = os.path.join(output_dir, "river_discharge.csv")
    with open(rivers_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rivers[0].keys())
        writer.writeheader()
        writer.writerows(rivers)
    print(f"Generated {len(rivers)} river telemetry points -> {rivers_path}")

    # 7. Alerts CSV — previously never generated, causing
    # load_synthetic_dataset.py to crash with FileNotFoundError
    alerts = generate_alerts(rng, 6)
    alerts_path = os.path.join(output_dir, "alerts.csv")
    with open(alerts_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=alerts[0].keys())
        writer.writeheader()
        writer.writerows(alerts)
    print(f"Generated {len(alerts)} alerts -> {alerts_path}")

    # 8. Research documents JSON — previously never generated
    research_docs = generate_research_documents(rng, SPECIES_REGISTRY, 5)
    research_path = os.path.join(output_dir, "research_documents.json")
    with open(research_path, "w") as f:
        json.dump(research_docs, f, indent=2)
    print(f"Generated {len(research_docs)} research documents -> {research_path}")

if __name__ == "__main__":
    main()