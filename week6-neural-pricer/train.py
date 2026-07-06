"""Training loop with best-validation checkpointing and learning-curve plotting."""

import copy
import sys

import matplotlib
if "ipykernel" not in sys.modules:
    matplotlib.use("Agg")

import numpy as np
import torch
from torch import nn


def train_model(model, train_loader, X_val_t, y_val_t, epochs: int = 300, lr: float = 1e-3):
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train": [], "val": []}
    best_val = float("inf")
    best_state = None

    for epoch in range(epochs):
        model.train()
        batch_losses = []

        for xb, yb in train_loader:
            pred = model(xb)
            loss = loss_fn(pred, yb)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_val_t)
            val_loss = loss_fn(val_pred, y_val_t).item()

        train_loss = float(np.mean(batch_losses))
        history["train"].append(train_loss)
        history["val"].append(val_loss)

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

        if epoch % 25 == 0:
            print(f"{epoch:03d} train={train_loss:.6f} val={val_loss:.6f}")

    model.load_state_dict(best_state)
    return history, best_state


def plot_learning_curve(history: dict, path: str):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7, 4))
    plt.plot(history["train"], label="train MSE")
    plt.plot(history["val"], label="validation MSE")
    plt.yscale("log")
    plt.xlabel("Epoch")
    plt.ylabel("MSE")
    plt.title("Week 6 Neural Pricer Learning Curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.show()
    plt.close()
