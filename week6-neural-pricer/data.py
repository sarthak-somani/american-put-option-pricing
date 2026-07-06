"""Synthetic option contract sampling and binomial label generation for the Week 6 neural pricer."""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "binomial-tree"))
from american_put import crr_put_price  # noqa: E402

FEATURE_ORDER = ["S0", "K", "T", "r", "sigma"]

RANGES = {
    "S0": (60.0, 140.0),
    "K": (80.0, 120.0),
    "T": (0.05, 2.0),
    "r": (0.00, 0.10),
    "sigma": (0.10, 0.50),
}


def sample_contracts(n: int, seed: int = 42) -> np.ndarray:
    """Sample n synthetic option contracts as rows [S0, K, T, r, sigma]."""
    rng = np.random.default_rng(seed)
    S0 = rng.uniform(*RANGES["S0"], size=n)
    K = rng.uniform(*RANGES["K"], size=n)
    T = rng.uniform(*RANGES["T"], size=n)
    r = rng.uniform(*RANGES["r"], size=n)
    sigma = rng.uniform(*RANGES["sigma"], size=n)
    return np.column_stack([S0, K, T, r, sigma])


def generate_labels(X: np.ndarray, steps: int = 500) -> np.ndarray:
    """Label each contract with the Week 4 CRR American put price."""
    y = np.empty(len(X), dtype=np.float64)
    for i, (S0, K, T, r, sigma) in enumerate(X):
        y[i] = crr_put_price(
            S0=float(S0),
            K=float(K),
            T=float(T),
            r=float(r),
            sigma=float(sigma),
            steps=steps,
            american=True,
        )
    return y


def check_labels(X: np.ndarray, y: np.ndarray) -> None:
    """Verify labels are finite, non-negative, and at least intrinsic value."""
    intrinsic = np.maximum(X[:, 1] - X[:, 0], 0.0)

    assert np.isfinite(y).all(), "non-finite labels found"
    assert (y >= -1e-10).all(), "negative labels found"

    violations = np.sum(y + 1e-8 < intrinsic)
    assert violations == 0, f"{violations} labels fall below intrinsic value"

    print(f"label range: [{y.min():.6f}, {y.max():.6f}], mean={y.mean():.6f}")
    print("intrinsic violations: 0")


def save_dataset(path: str, X: np.ndarray, y: np.ndarray, steps: int, seed: int) -> None:
    np.savez_compressed(
        path,
        X=X,
        y=y,
        steps=steps,
        seed=seed,
        feature_order=np.array(FEATURE_ORDER),
        ranges=np.array([RANGES[f] for f in FEATURE_ORDER]),
    )


def load_dataset(path: str):
    data = np.load(path)
    return data["X"], data["y"], int(data["steps"]), int(data["seed"])
