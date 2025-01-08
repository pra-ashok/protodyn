import torch
import torch.nn as nn

class LayerNormLinear(nn.Module):
    """
    A small module for: Linear(LayerNorm(x)).
    Used in node initialization steps.
    """
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.norm = nn.LayerNorm(input_dim)
        self.lin = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        # Normalize along the last dimension, then linear
        return self.lin(self.norm(x))
