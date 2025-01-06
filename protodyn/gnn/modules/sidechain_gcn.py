import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv

class SideChainGCN(nn.Module):
    """
    Simple GCN-based update for side-chain node embeddings.
    Wraps a GCNConv. We can stack more if multiple layers are needed.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.gcn = GCNConv(in_dim, out_dim)

    def forward(self, x, edge_index, edge_weight=None):
        """
        Args:
            x          : (N, in_dim) node features
            edge_index : (2, E)
            edge_weight: (E,) or None (scalar weights)
        Returns:
            updated_x  : (N, out_dim)
        """
        x_out = self.gcn(x, edge_index, edge_weight=edge_weight)
        return x_out
