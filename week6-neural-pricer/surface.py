"""Surface comparison (NN vs binomial) and finance sanity checks."""

import sys

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import numpy as np
import torch

from data import crr_put_price

S_GRID = np.linspace(60, 140, 41)
T_GRID = np.linspace(0.05, 2.0, 40)
K_FIXED = 100.0
R_FIXED = 0.05
SIGMA_FIXED = 0.25


def nn_predict_price(model, X_raw: np.ndarray, x_mean: np.ndarray, x_std: np.ndarray) -> np.ndarray:
    X_scaled = (X_raw - x_mean) / x_std
    X_t = torch.tensor(X_scaled, dtype=torch.float32)
    model.eval()
    with torch.no_grad():
        return model(X_t).numpy().reshape(-1)


def build_surface(model, x_mean, x_std, K=K_FIXED, r=R_FIXED, sigma=SIGMA_FIXED, steps=500):
    rows = [[S0, K, T, r, sigma] for T in T_GRID for S0 in S_GRID]
    X_surface = np.array(rows)

    nn_surface = nn_predict_price(model, X_surface, x_mean, x_std).reshape(len(T_GRID), len(S_GRID))
    binomial_surface = np.array([
        crr_put_price(S0, K, T, r, sigma, steps, american=True)
        for T in T_GRID
        for S0 in S_GRID
    ]).reshape(len(T_GRID), len(S_GRID))

    abs_error_surface = np.abs(nn_surface - binomial_surface)
    return nn_surface, binomial_surface, abs_error_surface


def run_sanity_checks(model, x_mean, x_std, K=K_FIXED, T=1.0, r=R_FIXED, sigma=SIGMA_FIXED) -> dict:
    slice_rows = np.array([[S0, K, T, r, sigma] for S0 in np.linspace(60, 140, 81)])
    slice_pred = nn_predict_price(model, slice_rows, x_mean, x_std)

    monotonic_violations = int(np.sum(np.diff(slice_pred) > 1e-4))
    negative_predictions = int(np.sum(slice_pred < -1e-6))
    intrinsic = np.maximum(slice_rows[:, 1] - slice_rows[:, 0], 0.0)
    intrinsic_violations = int(np.sum(slice_pred + 1e-4 < intrinsic))

    results = {
        "monotonic_violations": monotonic_violations,
        "negative_predictions": negative_predictions,
        "intrinsic_violations": intrinsic_violations,
    }
    print("monotonic violations:", monotonic_violations)
    print("negative predictions:", negative_predictions)
    print("below intrinsic:", intrinsic_violations)
    return results


def plot_surfaces(nn_surface, binomial_surface, abs_error_surface, path):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
    titles = ["NN surface", "Binomial surface", "Abs error"]
    surfaces = [nn_surface, binomial_surface, abs_error_surface]
    cmaps = ["viridis", "viridis", "magma"]

    for ax, surf, title, cmap in zip(axes, surfaces, titles, cmaps):
        im = ax.pcolormesh(S_GRID, T_GRID, surf, shading="auto", cmap=cmap)
        ax.set_xlabel("S0")
        ax.set_ylabel("T (years)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax)

    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.show()
    plt.close()
