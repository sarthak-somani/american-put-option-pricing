"""Sanity tests for the Week 6 neural pricer pipeline components."""

import numpy as np
import torch
from torch import nn

from data import RANGES, check_labels, generate_labels, sample_contracts
from model import PutPricerNet, load_artifact
from split import standardize, train_val_test_split
from surface import run_sanity_checks


def test_sample_contracts_shape_and_ranges():
    X = sample_contracts(500, seed=1)
    assert X.shape == (500, 5)

    for i, feature in enumerate(["S0", "K", "T", "r", "sigma"]):
        lo, hi = RANGES[feature]
        assert X[:, i].min() >= lo
        assert X[:, i].max() <= hi


def test_sample_contracts_reproducible_with_seed():
    X1 = sample_contracts(100, seed=7)
    X2 = sample_contracts(100, seed=7)
    assert np.array_equal(X1, X2)


def test_generate_labels_passes_sanity_checks():
    X = sample_contracts(50, seed=2)
    y = generate_labels(X, steps=25)
    check_labels(X, y)  # raises AssertionError on failure


def test_standardize_uses_train_stats_only():
    X = sample_contracts(300, seed=3)
    y = generate_labels(X, steps=10)
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y, seed=42)

    X_train_s, X_val_s, X_test_s, x_mean, x_std = standardize(X_train, X_val, X_test)

    assert np.allclose(X_train_s.mean(axis=0), 0.0, atol=1e-8)
    assert np.allclose(X_train_s.std(axis=0), 1.0, atol=1e-8)

    # val/test must be scaled with train stats, not their own
    expected_val_s = (X_val - x_mean) / x_std
    assert np.allclose(X_val_s, expected_val_s)


def test_put_pricer_net_output_shape():
    model = PutPricerNet(input_dim=5, hidden=16)
    x = torch.randn(10, 5)
    out = model(x)
    assert out.shape == (10, 1)


def test_artifact_round_trips_with_numpy_scaling_stats(tmp_path):
    model = PutPricerNet(input_dim=5, hidden=8)
    x_mean = np.array([100.0, 100.0, 1.0, 0.05, 0.25])
    x_std = np.array([20.0, 10.0, 0.5, 0.03, 0.1])

    artifact = {
        "model_state": model.state_dict(),
        "x_mean": x_mean,
        "x_std": x_std,
        "feature_order": ["S0", "K", "T", "r", "sigma"],
        "label_steps": 500,
        "ranges": RANGES,
    }
    path = tmp_path / "artifact.pt"
    torch.save(artifact, path)

    loaded = load_artifact(str(path))  # must not raise on torch's default weights_only=True
    assert np.array_equal(loaded["x_mean"], x_mean)
    assert np.array_equal(loaded["x_std"], x_std)
    assert loaded["label_steps"] == 500

    reloaded_model = PutPricerNet(input_dim=5, hidden=8)
    reloaded_model.load_state_dict(loaded["model_state"])


def test_sanity_checks_report_zero_violations_for_intrinsic_stub():
    class IntrinsicStub(nn.Module):
        """A model that returns exact intrinsic value: monotonic, non-negative, at the bound."""

        def __init__(self, x_mean, x_std):
            super().__init__()
            self.x_mean = torch.tensor(x_mean, dtype=torch.float32)
            self.x_std = torch.tensor(x_std, dtype=torch.float32)

        def forward(self, x):
            raw = x * self.x_std + self.x_mean
            S0 = raw[:, 0]
            K = raw[:, 1]
            price = torch.clamp(K - S0, min=0.0)
            return price.reshape(-1, 1)

    x_mean = np.array([100.0, 100.0, 1.0, 0.05, 0.25])
    x_std = np.array([20.0, 10.0, 0.5, 0.03, 0.1])
    stub = IntrinsicStub(x_mean, x_std)

    results = run_sanity_checks(stub, x_mean, x_std)
    assert results["monotonic_violations"] == 0
    assert results["negative_predictions"] == 0
    assert results["intrinsic_violations"] == 0
