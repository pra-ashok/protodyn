import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class BackboneGCN(nn.Module):
    """
    GCN-based update for backbone node embeddings.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gcn = GCNConv(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        x_out = self.gcn(x, edge_index, edge_weight=edge_weight)
        return x_out
