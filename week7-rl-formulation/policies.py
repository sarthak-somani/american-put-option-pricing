"""Baseline and tabular-Q exercise policies over the [time_fraction, moneyness] state."""

import numpy as np

# Moneyness bins are non-uniform: coarse away from the money, fine across [0.7, 1.0]
# where the American put exercise boundary actually lives (see the Week 4 boundary,
# which stays within roughly this band for T in [0, 1] at these baseline parameters).
MONEY_MIN = 0.5
MONEY_MAX = 1.5
DEFAULT_MONEY_EDGES = np.concatenate([
    np.linspace(MONEY_MIN, 0.7, 7)[:-1],  # 6 coarse bins below the boundary region
    np.linspace(0.7, 1.0, 19),            # 18 fine bins across the boundary region
    np.linspace(1.0, MONEY_MAX, 7)[1:],   # 6 coarse bins above
])


def always_hold_policy(state):
    """Never exercise early; equivalent to European-style exercise at expiry."""
    return 0


def immediate_exercise_policy(state):
    """Exercise on the very first decision. Good only when already deep ITM."""
    return 1


def make_random_policy(seed=0):
    """Uniform random hold/exercise -- a weak, noisy debugging baseline."""
    rng = np.random.default_rng(seed)

    def policy(state):
        return int(rng.integers(0, 2))

    return policy


def discretize_state(state, n_time=20, money_edges=DEFAULT_MONEY_EDGES):
    """Bin [time_fraction, moneyness] into a (t_bin, m_bin) Q-table index.

    Moneyness bins are non-uniform (see DEFAULT_MONEY_EDGES): finer resolution
    across [0.7, 1.0], where the exercise boundary actually lives, coarser outside.
    Out-of-range moneyness clips to the nearest edge bin, same as before.
    """
    time_fraction, moneyness = state
    n_money = len(money_edges) - 1
    t_bin = int(np.clip(time_fraction * n_time, 0, n_time - 1))
    m_bin = int(np.clip(np.searchsorted(money_edges, moneyness, side="right") - 1, 0, n_money - 1))
    return t_bin, m_bin


def make_q_policy(Q, n_time=20, money_edges=DEFAULT_MONEY_EDGES):
    """Greedy policy from a trained tabular Q-function."""

    def policy(state):
        s_idx = discretize_state(state, n_time, money_edges)
        return int(np.argmax(Q[s_idx]))

    return policy
