"""Train/val/test split and train-only feature standardization."""

import numpy as np


def train_val_test_split(X: np.ndarray, y: np.ndarray, seed: int = 42, ratios=(0.8, 0.1, 0.1)):
    assert abs(sum(ratios) - 1.0) < 1e-9, "ratios must sum to 1"

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(X))
    n_train = int(ratios[0] * len(X))
    n_val = int(ratios[1] * len(X))

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    return (
        X[train_idx], y[train_idx],
        X[val_idx], y[val_idx],
        X[test_idx], y[test_idx],
    )


def standardize(X_train: np.ndarray, X_val: np.ndarray, X_test: np.ndarray):
    """Standardize using training-set statistics only."""
    x_mean = X_train.mean(axis=0)
    x_std = X_train.std(axis=0)
    x_std = np.where(x_std == 0, 1.0, x_std)

    X_train_s = (X_train - x_mean) / x_std
    X_val_s = (X_val - x_mean) / x_std
    X_test_s = (X_test - x_mean) / x_std

    return X_train_s, X_val_s, X_test_s, x_mean, x_std
