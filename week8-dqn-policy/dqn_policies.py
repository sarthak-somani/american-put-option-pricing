"""Policy and diagnostic-grid helpers for a trained QNetwork.

`network_exercise_region` / `network_q_margin` return the same
(time_fractions, moneyness_grid, grid) shape that
../week7-rl-formulation/evaluate.py's `policy_exercise_region` /
`policy_q_margin` return, so Week 7's `plot_exercise_region` / `plot_q_margin`
can be reused unchanged for the DQN's plots -- only the grid-computation
needs a network-specific version (vectorized as one batched forward pass,
since a network -- unlike a table -- is cheap to query densely).
"""

import numpy as np
import torch

from dqn import expand_state


def greedy_action(model, state):
    """argmax_a Q(state, a) for a single raw [time_fraction, moneyness] state."""
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = expand_state(state)
        x_t = torch.tensor(x, dtype=torch.float32, device=device).unsqueeze(0)
        q = model(x_t)
    return int(torch.argmax(q, dim=1).item())


def make_dqn_policy(model):
    """Wrap a trained QNetwork as a policy_fn(state) -> action, for run_policy etc."""

    def policy(state):
        return greedy_action(model, state)

    return policy


def _grid_states(steps, n_money, money_min, money_max):
    time_fractions = np.arange(steps + 1) / steps
    moneyness_grid = np.linspace(money_min, money_max, n_money)
    tf_mesh, m_mesh = np.meshgrid(time_fractions, moneyness_grid, indexing="ij")
    states = np.stack(
        [tf_mesh.ravel(), 1.0 - tf_mesh.ravel(), m_mesh.ravel()], axis=1
    ).astype(np.float32)
    return time_fractions, moneyness_grid, states


def network_q_values(model, steps, n_money=121, money_min=0.5, money_max=1.5):
    """Batched Q(hold), Q(exercise) over a dense (time, moneyness) grid."""
    time_fractions, moneyness_grid, states = _grid_states(steps, n_money, money_min, money_max)
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        q = model(torch.tensor(states, dtype=torch.float32, device=device)).cpu().numpy()
    shape = (len(time_fractions), n_money)
    q_hold = q[:, 0].reshape(shape)
    q_exercise = q[:, 1].reshape(shape)
    return time_fractions, moneyness_grid, q_hold, q_exercise


def network_exercise_region(model, steps, n_money=121, money_min=0.5, money_max=1.5):
    """0/1 greedy exercise decision over a dense grid (same shape as Week 7's
    policy_exercise_region output, so it plugs into plot_exercise_region)."""
    tf, mg, q_hold, q_exercise = network_q_values(model, steps, n_money, money_min, money_max)
    exercise_grid = (q_exercise > q_hold).astype(int)
    return tf, mg, exercise_grid


def network_q_margin(model, steps, n_money=121, money_min=0.5, money_max=1.5):
    """Q(exercise) - Q(hold) over a dense grid (same shape as Week 7's
    policy_q_margin output, so it plugs into plot_q_margin)."""
    tf, mg, q_hold, q_exercise = network_q_values(model, steps, n_money, money_min, money_max)
    return tf, mg, q_exercise - q_hold
