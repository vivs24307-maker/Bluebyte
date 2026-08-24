"""
train.py
Model training pipeline with deterministic seed assignment and checkpoint saving.
"""

import os
import torch
import torch.nn.functional as F
import torch.optim as optim

from .model import MarineGNN
from .graph_builder import build_hetero_graph

def train_model(epochs: int = 100, seed: int = 42, checkpoint_dir: str = "checkpoints/"):
    print(f"[Training] Building graph with deterministic seed={seed}...")
    graph = build_hetero_graph(seed=seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    graph = graph.to(device)
    
    model = MarineGNN(hidden_channels=64, out_channels=1).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    print(f"[Training] Starting training across {epochs} epochs on {device}...")
    model.train()
    
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        out = model(graph.x_dict, graph.edge_index_dict)
        
        # Synthetic habitat occurrence target tensor
        dummy_targets = torch.randint(0, 2, (graph['grid'].num_nodes, 1), dtype=torch.float, device=device)
        
        loss = F.binary_cross_entropy_with_logits(out['grid'], dummy_targets)
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 1:
            print(f"Epoch {epoch:03d}/{epochs:03d} - Loss: {loss.item():.4f}")

    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, "gnn_model.pt")
    torch.save(model.state_dict(), checkpoint_path)
    print(f"[Training] Training complete. Saved checkpoint to {checkpoint_path}")

if __name__ == "__main__":
    train_model()