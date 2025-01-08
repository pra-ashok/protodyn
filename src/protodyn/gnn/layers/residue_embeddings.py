import torch
import torch.nn as nn

class ResidueEmbedding(nn.Module):
    """
    E(R_i, R_j) residue-type embeddings.
    """
    def __init__(self, num_res_types, embed_dim):
        super().__init__()
        self.embed = nn.Embedding(num_res_types, embed_dim)

    def forward(self, residue_idx_i, residue_idx_j):
        """
        Args:
            residue_idx_i, residue_idx_j: (E,) integer residue type indices
        Returns:
            (E, 2 * embed_dim) embedding for each edge
        """
        emb_i = self.embed(residue_idx_i)
        emb_j = self.embed(residue_idx_j)
        return torch.cat([emb_i, emb_j], dim=-1)
