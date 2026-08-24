"""
predict.py
Complete inference engine providing single-cell occupancy prediction,
GeoJSON aggregate biodiversity maps, eDNA cross-referencing, and seasonal shifts.
"""

import os
import math
import numpy as np
import torch
from typing import Dict, List, Any, Optional

from .model import MarineGNN
from .graph_builder import build_hetero_graph

# ---------------------------------------------------------------------------
# Fallback Heuristic Model
# ---------------------------------------------------------------------------

class FallbackHabitatModel:
    @staticmethod
    def score_habitat(sst: float, salinity: float, depth: float, species_meta: Dict[str, Any]) -> float:
        min_sst = species_meta.get("min_sst", 24.0)
        max_sst = species_meta.get("max_sst", 32.0)
        opt_sst = (min_sst + max_sst) / 2.0
        sst_tol = max((max_sst - min_sst) / 2.0, 1.0)
        sst_score = math.exp(-0.5 * ((sst - opt_sst) / sst_tol) ** 2)

        min_sal = species_meta.get("min_salinity", 30.0)
        max_sal = species_meta.get("max_salinity", 36.0)
        opt_sal = (min_sal + max_sal) / 2.0
        sal_tol = max((max_sal - min_sal) / 2.0, 1.0)
        sal_score = math.exp(-0.5 * ((salinity - opt_sal) / sal_tol) ** 2)

        min_depth = species_meta.get("min_depth", 0.0)
        max_depth = species_meta.get("max_depth", 200.0)
        if min_depth <= depth <= max_depth:
            depth_score = 1.0
        else:
            dist = min(abs(depth - min_depth), abs(depth - max_depth))
            depth_score = math.exp(-0.5 * (dist / 50.0) ** 2)

        return float(np.clip(0.5 * sst_score + 0.3 * sal_score + 0.2 * depth_score, 0.01, 0.99))


# ---------------------------------------------------------------------------
# Marine Biodiversity Predictor
# ---------------------------------------------------------------------------

class MarineBiodiversityPredictor:
    def __init__(
        self,
        model_path: Optional[str] = "checkpoints/gnn_model.pt",
        seed: int = 42,
        device: Optional[str] = None
    ):
        self.seed = seed
        self.device = torch.device(device if device else ("cuda" if torch.cuda.is_available() else "cpu"))
        
        # Deterministic graph building
        self.graph = build_hetero_graph(seed=self.seed)
        self.graph = self.graph.to(self.device)
        
        self.model = MarineGNN(hidden_channels=64, out_channels=1).to(self.device)
        self.is_trained = False
        
        if model_path and os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path, map_location=self.device))
                self.model.eval()
                self.is_trained = True
            except Exception as e:
                print(f"[Predictor] Checkpoint load failed: {e}. Fallback active.")
        else:
            print("[Predictor] Operating in rule-based fallback mode.")

        self.fallback = FallbackHabitatModel()

    def predict_species_in_existing_grid(self, grid_idx: int, species_idx: int) -> Dict[str, Any]:
        if self.is_trained:
            with torch.no_grad():
                out = self.model(self.graph.x_dict, self.graph.edge_index_dict)
                prob = torch.sigmoid(out['grid'][grid_idx]).item()
                return {
                    "grid_index": grid_idx,
                    "species_index": species_idx,
                    "probability": round(prob, 4),
                    "prediction_source": "trained_gnn"
                }
        else:
            sst_feat = float(self.graph['grid'].x[grid_idx][2].item()) if self.graph['grid'].x.shape[1] > 2 else 28.5
            sal_feat = float(self.graph['grid'].x[grid_idx][3].item()) if self.graph['grid'].x.shape[1] > 3 else 34.5
            depth_feat = 50.0

            spec_meta = {
                "min_sst": float(self.graph['species'].x[species_idx][0].item()),
                "max_sst": float(self.graph['species'].x[species_idx][1].item()),
                "min_salinity": float(self.graph['species'].x[species_idx][2].item()),
                "max_salinity": float(self.graph['species'].x[species_idx][3].item()),
            }
            prob = self.fallback.score_habitat(sst_feat, sal_feat, depth_feat, spec_meta)
            return {
                "grid_index": grid_idx,
                "species_index": species_idx,
                "probability": round(prob, 4),
                "prediction_source": "ecological_fallback"
            }

    def predict_all_grids(self) -> Dict[str, Any]:
        num_grids = self.graph['grid'].num_nodes
        num_species = self.graph['species'].num_nodes
        features = []

        for g_idx in range(num_grids):
            species_probs = []
            for s_idx in range(num_species):
                pred = self.predict_species_in_existing_grid(g_idx, s_idx)
                species_probs.append(pred["probability"])

            probs_arr = np.array(species_probs)
            norm_probs = probs_arr / (np.sum(probs_arr) + 1e-9)
            shannon_index = -float(np.sum([p * np.log(p + 1e-9) for p in norm_probs if p > 0]))
            biodiversity_score = float(np.clip(shannon_index / np.log(max(num_species, 2)), 0.05, 0.99))

            feature = {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        float(self.graph['grid'].x[g_idx][1].item()), # Lon
                        float(self.graph['grid'].x[g_idx][0].item())  # Lat
                    ]
                },
                "properties": {
                    "grid_index": g_idx,
                    "biodiversity_score": round(biodiversity_score, 4),
                    "mean_species_probability": round(float(np.mean(probs_arr)), 4),
                    "dominant_species_index": int(np.argmax(probs_arr)),
                    "prediction_source": "trained_gnn" if self.is_trained else "ecological_fallback"
                }
            }
            features.append(feature)

        return {"type": "FeatureCollection", "features": features}

    def get_edna_cross_references(self, species_idx: int) -> List[Dict[str, Any]]:
        results = []
        if ('edna', 'detects', 'species') not in self.graph.edge_index_dict:
            return results

        edge_index = self.graph['edna', 'detects', 'species'].edge_index
        matched_edna_nodes = edge_index[0][edge_index[1] == species_idx]

        for edna_idx in matched_edna_nodes.tolist():
            conf = float(self.graph['edna'].x[edna_idx][0].item())
            results.append({
                "edna_node_index": edna_idx,
                "target_species_index": species_idx,
                "detection_confidence": round(conf, 4),
                "is_validated": conf >= 0.85
            })
        return results

    def seasonal_species_shift(self, species_idx: int) -> Dict[str, Any]:
        seasons = {
            "pre_monsoon": {"sst_delta": 1.5, "sal_delta": 0.5},
            "monsoon": {"sst_delta": -2.0, "sal_delta": -3.0},
            "post_monsoon": {"sst_delta": 0.0, "sal_delta": 1.0}
        }
        
        seasonal_results = {}
        original_features = self.graph['grid'].x.clone()

        for season_name, deltas in seasons.items():
            perturbed_features = original_features.clone()
            perturbed_features[:, 2] += deltas["sst_delta"] # Perturb SST feature
            perturbed_features[:, 3] += deltas["sal_delta"] # Perturb Salinity feature

            self.graph['grid'].x = perturbed_features
            
            probs = [self.predict_species_in_existing_grid(g, species_idx)["probability"] for g in range(self.graph['grid'].num_nodes)]

            seasonal_results[season_name] = {
                "sst_offset_celsius": deltas["sst_delta"],
                "salinity_offset_psu": deltas["sal_delta"],
                "average_habitat_suitability": round(float(np.mean(probs)), 4)
            }

        self.graph['grid'].x = original_features
        return seasonal_results