"""End-to-end Week 6 pipeline: sample -> label -> split/scale -> train -> evaluate -> surface -> save."""

import os

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from data import FEATURE_ORDER, RANGES, check_labels, generate_labels, sample_contracts, save_dataset
from evaluate import compute_metrics, moneyness_bucket_mae, plot_pred_vs_true
from model import PutPricerNet
from split import standardize, train_val_test_split
from surface import build_surface, plot_surfaces, run_sanity_checks
from train import plot_learning_curve, train_model

HERE = os.path.dirname(__file__)
FIGURES_DIR = os.path.join(HERE, "figures")
ARTIFACTS_DIR = os.path.join(HERE, "artifacts")

N_CONTRACTS = 12_000
LABEL_STEPS = 500
SEED = 42


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    print("=== Part A: dataset generation ===")
    X = sample_contracts(N_CONTRACTS, seed=SEED)
    y = generate_labels(X, steps=LABEL_STEPS)
    check_labels(X, y)
    save_dataset(os.path.join(ARTIFACTS_DIR, "week6_option_data.npz"), X, y, LABEL_STEPS, SEED)

    print("\n=== Part B: split, scale, train ===")
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, seed=SEED)
    X_train_s, X_val_s, X_test_s, x_mean, x_std = standardize(X_train, X_val, X_test)

    torch.manual_seed(SEED)
    X_train_t = torch.tensor(X_train_s, dtype=torch.float32)
    y_train_t = torch.tensor(y_train.reshape(-1, 1), dtype=torch.float32)
    X_val_t = torch.tensor(X_val_s, dtype=torch.float32)
    y_val_t = torch.tensor(y_val.reshape(-1, 1), dtype=torch.float32)
    X_test_t = torch.tensor(X_test_s, dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, y_train_t), batch_size=256, shuffle=True)

    model = PutPricerNet(input_dim=5, hidden=128)
    history, best_state = train_model(model, train_loader, X_val_t, y_val_t, epochs=300, lr=1e-3)
    plot_learning_curve(history, os.path.join(FIGURES_DIR, "learning_curve.png"))

    print("\n=== Part C: evaluation and finance checks ===")
    model.eval()
    with torch.no_grad():
        pred_test = model(X_test_t).numpy().reshape(-1)

    err = pred_test - y_test
    metrics = compute_metrics(pred_test, y_test)
    print(f"MAE:     {metrics['mae']:.4f}")
    print(f"RMSE:    {metrics['rmse']:.4f}")
    print(f"Max abs: {metrics['max_abs']:.4f}")

    moneyness_bucket_mae(X_test, err)
    plot_pred_vs_true(y_test, pred_test, os.path.join(FIGURES_DIR, "pred_vs_binomial.png"))

    nn_surface, binomial_surface, abs_error_surface = build_surface(
        model, x_mean, x_std, steps=LABEL_STEPS
    )
    plot_surfaces(
        nn_surface, binomial_surface, abs_error_surface,
        os.path.join(FIGURES_DIR, "surface_comparison.png"),
    )
    run_sanity_checks(model, x_mean, x_std)

    artifact = {
        "model_state": best_state,
        "x_mean": x_mean,
        "x_std": x_std,
        "feature_order": FEATURE_ORDER,
        "label_steps": LABEL_STEPS,
        "ranges": RANGES,
    }
    torch.save(artifact, os.path.join(ARTIFACTS_DIR, "week6_neural_pricer.pt"))
    print("\nSaved artifact to artifacts/week6_neural_pricer.pt")


if __name__ == "__main__":
    main()
