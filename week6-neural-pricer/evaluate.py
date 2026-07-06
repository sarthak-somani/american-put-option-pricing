"""Quant-style error evaluation: aggregate metrics, moneyness buckets, scatter plot."""

import sys

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import numpy as np


def compute_metrics(pred: np.ndarray, true: np.ndarray) -> dict:
    err = pred - true
    return {
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err ** 2))),
        "max_abs": float(np.max(np.abs(err))),
    }


def moneyness_bucket_mae(X: np.ndarray, err: np.ndarray) -> dict:
    moneyness = X[:, 0] / X[:, 1]
    buckets = {
        "deep ITM put": moneyness < 0.85,
        "near ATM": (moneyness >= 0.85) & (moneyness <= 1.15),
        "deep OTM put": moneyness > 1.15,
    }

    results = {}
    for name, mask in buckets.items():
        n = int(mask.sum())
        mae = float(np.mean(np.abs(err[mask]))) if n > 0 else float("nan")
        results[name] = {"n": n, "mae": mae}
        print(f"{name:12s}: n={n:4d}, MAE={mae:.4f}")
    return results


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, path: str):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, s=8, alpha=0.35)
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.plot([lo, hi], [lo, hi], color="black", linewidth=1)
    plt.xlabel("Binomial price")
    plt.ylabel("NN predicted price")
    plt.title("Predicted vs Binomial American Put Prices")
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.show()
    plt.close()
