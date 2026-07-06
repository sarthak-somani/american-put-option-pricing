"""MLP architecture for the neural American put pricer."""

import torch
from torch import nn


class PutPricerNet(nn.Module):
    """Maps [S0, K, T, r, sigma] -> predicted American put price."""

    def __init__(self, input_dim: int = 5, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)


def load_artifact(path: str, map_location: str = "cpu") -> dict:
    """Load a saved pipeline artifact (model state + scaling stats + metadata).

    weights_only=False is required: the artifact bundles x_mean/x_std as numpy
    arrays alongside the model state dict, which torch's default weights_only=True
    unpickler (torch>=2.6) rejects. Only load artifacts you saved yourself.
    """
    return torch.load(path, map_location=map_location, weights_only=False)
