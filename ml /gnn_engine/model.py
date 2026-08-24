import math

try:
    import torch
    import torch.nn.functional as F
    from torch_geometric.nn import HeteroConv, GATConv
    HAS_PYG = True
except ImportError:
    HAS_PYG = False

try:
    from .graph_builder import SPECIES_METADATA
except ImportError:
    from graph_builder import SPECIES_METADATA

if HAS_PYG:
    class HeteroGNN(torch.nn.Module):
        REL_SPECS = [
            ('Species', 'species_occurs_in_grid', 'OceanGrid'),
            ('eDNAMarker', 'edna_detected_in_grid', 'OceanGrid'),
            ('eDNAMarker', 'edna_identifies_species', 'Species'),
            ('OceanGrid', 'grid_correlates_with_grid', 'OceanGrid'),
        ]

        def _build_conv_layer(self, hidden_channels):
            convs = {}
            for rel in self.REL_SPECS:
                src_type, _, dst_type = rel
                if src_type == dst_type:
                    convs[rel] = GATConv(-1, hidden_channels)
                else:
                    convs[rel] = GATConv((-1, -1), hidden_channels, add_self_loops=False)
            return HeteroConv(convs, aggr='sum')

        def __init__(self, hidden_channels, out_channels, dropout=0.3):
            super().__init__()
            self.dropout = dropout
            # A simple 2-layer HeteroGAT
            self.conv1 = self._build_conv_layer(hidden_channels)
            self.conv2 = self._build_conv_layer(hidden_channels)

            # Link prediction head for species -> grid
            self.lin = torch.nn.Linear(hidden_channels * 2, 1)

        def forward(self, x_dict, edge_index_dict, edge_label_index):
            x_dict_out = self.conv1(x_dict, edge_index_dict)
            x_dict_out = {key: F.dropout(F.relu(x), p=self.dropout, training=self.training) for key, x in x_dict_out.items()}
            # HeteroConv only returns node types that appear as a DESTINATION
            # in at least one edge type. eDNAMarker is source-only (it only
            # points at OceanGrid/Species, nothing points at it), so it gets
            # silently dropped here. Carry its previous-layer features forward
            # so conv2 still has a valid source tensor for it.
            for key, val in x_dict.items():
                if key not in x_dict_out:
                    x_dict_out[key] = val
            x_dict = x_dict_out

            x_dict_out2 = self.conv2(x_dict, edge_index_dict)
            for key, val in x_dict.items():
                if key not in x_dict_out2:
                    x_dict_out2[key] = val
            x_dict = x_dict_out2

            # Predict edges for ('Species', 'species_occurs_in_grid', 'OceanGrid')
            z_src = x_dict['Species'][edge_label_index[0]]
            z_dst = x_dict['OceanGrid'][edge_label_index[1]]
            z = torch.cat([z_src, z_dst], dim=-1)
            return self.lin(z).squeeze(-1)

        @torch.no_grad()
        def explain_by_ablation(self, x_dict, edge_index_dict, edge_label_index):
            """Explainability via ablation: runs the REAL forward() once as a
            baseline, then once more per relation type with that relation's
            edges removed, and reports how much the predicted confidence
            shifts. A relation whose removal barely changes the score wasn't
            doing much work for this prediction; one whose removal causes a
            big drop was load-bearing.

            This replaces an earlier attempt that summed GATConv's internal
            (post-softmax) attention weights per relation -- that number is
            mathematically guaranteed to be ~1.0 for any relation with at
            least one incoming edge (softmax over incoming edges always sums
            to 1), so it never actually measured relative importance. Ablation
            measures a real, observable effect instead, and reuses forward()
            directly instead of re-implementing the conv stack by hand.
            """
            self.eval()
            baseline = torch.sigmoid(self(x_dict, edge_index_dict, edge_label_index))

            impact_by_relation = {}
            for rel in edge_index_dict:
                ablated_edges = dict(edge_index_dict)
                empty = torch.zeros((2, 0), dtype=torch.long)
                ablated_edges[rel] = empty
                ablated_score = torch.sigmoid(self(x_dict, ablated_edges, edge_label_index))
                impact_by_relation[rel[1]] = float((baseline - ablated_score).abs().item())

            return float(baseline.item()), impact_by_relation

    def load_pretrained(path="gnn_link_predictor.pt", hidden_channels=32):
        """Loads weights saved by train.py. Returns None if the checkpoint
        doesn't exist yet (e.g. training hasn't been run in this environment)."""
        import os
        model = HeteroGNN(hidden_channels=hidden_channels, out_channels=hidden_channels)
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location="cpu"))
            model.eval()
            return model
        return None


class FallbackModel:
    def __init__(self):
        self.species_meta = SPECIES_METADATA

    def score_habitat(self, grid_features, sp):
        # grid_features = [sst, salinity, chlorophyll, do, depth, lat, lon]
        sst, salinity, chlorophyll, do, depth, lat, lon = grid_features

        opt_sst = (sp["min_sst"] + sp["max_sst"]) / 2
        sst_sigma = (sp["max_sst"] - sp["min_sst"]) / 2 if sp["max_sst"] != sp["min_sst"] else 1.0

        # Gaussian similarity for SST
        sst_score = math.exp(-0.5 * ((sst - opt_sst) / sst_sigma)**2)

        opt_depth = (sp["min_depth"] + sp["max_depth"]) / 2
        depth_sigma = (sp["max_depth"] - sp["min_depth"]) / 2 if sp["max_depth"] != sp["min_depth"] else 1.0
        depth_score = math.exp(-0.5 * ((depth - opt_depth) / depth_sigma)**2)

        # Salinity optimal is roughly 34 PSU for marine fish
        salinity_score = math.exp(-0.5 * ((salinity - 34.0) / 2.0)**2)

        # Combining
        score = (0.5 * sst_score) + (0.3 * depth_score) + (0.2 * salinity_score)

        # DO influence - below 3.0 is hypoxic
        if do < 3.0:
            score *= 0.2

        return max(0.0, min(1.0, score))

    def predict(self, grid_features):
        results = []
        for sp in self.species_meta:
            score = self.score_habitat(grid_features, sp)
            results.append({
                "species_name": sp["name"],
                "scientific_name": sp["scientific"],
                "confidence": score,
                "habitat_match": score > 0.6
            })
        results.sort(key=lambda x: x["confidence"], reverse=True)
        return results