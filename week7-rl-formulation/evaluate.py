"""Policy evaluation and exercise-region diagnostics for the American put env.

Three distinct kinds of evaluation are provided, and they answer different
questions:

- `run_policy` Monte Carlo simulates episodes through the stochastic
  environment and reports average payoff / exercise rate / exercise timing.
  It answers "how does this policy behave and pay off in practice."
- `policy_exercise_region` queries a policy directly on a dense, deterministic
  (time, moneyness) grid. It answers "what is this policy's exercise region,"
  which is the right way to compare a policy's *shape* against the Week 4
  binomial exercise boundary -- Monte Carlo samples are noisy and sparse near
  any particular (time, moneyness) point, a grid query is not.
- `policy_q_margin` (tabular-Q policies only) queries Q(exercise) - Q(hold) on
  the same kind of dense grid. It answers "how confidently" the policy prefers
  exercise, which a 0/1 region plot collapses away -- near-zero margins flag
  under-trained cells, not settled decisions.
"""

import numpy as np


def run_policy(env, policy_fn, episodes=1000, seed=None):
    """Monte Carlo evaluation of a policy over many simulated episodes.

    Reports both the raw (undiscounted) mean payoff and the properly
    discounted mean payoff (reward * env.discount ** elapsed_steps). The
    discounted figure is the one comparable to a Black-Scholes/binomial
    price; the raw figure is a diagnostic only -- summing undiscounted
    payoff overstates value because it ignores present-value time cost.
    """
    if seed is not None:
        env.reset(seed=seed)

    raw_payoffs = np.zeros(episodes)
    discounted_payoffs = np.zeros(episodes)
    exercised = np.zeros(episodes, dtype=bool)
    exercise_steps = []
    exercise_moneyness = []

    for i in range(episodes):
        state = env.reset()
        done = False
        reward = 0.0
        info = {"reason": "expiry", "step": 0}

        while not done:
            action = policy_fn(state)
            state, reward, done, info = env.step(action)

        raw_payoffs[i] = reward
        discounted_payoffs[i] = reward * (env.discount ** info["step"])
        if info["reason"] == "exercise":
            exercised[i] = True
            exercise_steps.append(info["step"])
            exercise_moneyness.append(float(state[1]))

    return {
        "raw_payoff_mean": float(raw_payoffs.mean()),
        "discounted_payoff_mean": float(discounted_payoffs.mean()),
        "discounted_payoff_std": float(discounted_payoffs.std()),
        "exercise_rate": float(exercised.mean()),
        "exercise_steps": np.array(exercise_steps, dtype=int),
        "exercise_moneyness": np.array(exercise_moneyness, dtype=float),
    }


def policy_exercise_region(policy_fn, steps, n_money=121, money_min=0.5, money_max=1.5):
    """Query a policy's exercise decision over a dense (time, moneyness) grid.

    Returns (time_fractions, moneyness_grid, exercise_grid) where
    exercise_grid[i, j] is 1 if the policy exercises at
    (time_fractions[i], moneyness_grid[j]), else 0.
    """
    time_fractions = np.arange(steps + 1) / steps
    moneyness_grid = np.linspace(money_min, money_max, n_money)
    exercise_grid = np.zeros((len(time_fractions), n_money), dtype=int)

    for i, tf in enumerate(time_fractions):
        for j, m in enumerate(moneyness_grid):
            state = np.array([tf, m], dtype=np.float32)
            exercise_grid[i, j] = policy_fn(state)

    return time_fractions, moneyness_grid, exercise_grid


def policy_q_margin(Q, steps, n_time=20, money_edges=None, n_points=121, money_min=None, money_max=None):
    """Evaluate Q(exercise) - Q(hold) over a dense (time, moneyness) grid.

    A 0/1 exercise-region plot only shows *which side* of the decision boundary
    a state falls on. The margin shows *how confidently*: a near-zero margin
    marks a genuine close call (or, just as often for a coarse table, a cell
    that was barely visited during training and never moved far from its zero
    initialization) rather than a settled "hold." Since Q is itself a lookup
    table, the margin is piecewise constant within each (t_bin, m_bin) cell --
    querying it on a dense grid just reveals the shape of those cells, which
    are finer near the boundary region by construction (see
    policies.DEFAULT_MONEY_EDGES).
    """
    from policies import DEFAULT_MONEY_EDGES, MONEY_MAX, MONEY_MIN, discretize_state

    if money_edges is None:
        money_edges = DEFAULT_MONEY_EDGES
    if money_min is None:
        money_min = MONEY_MIN
    if money_max is None:
        money_max = MONEY_MAX

    time_fractions = np.arange(steps + 1) / steps
    moneyness_grid = np.linspace(money_min, money_max, n_points)
    margin = np.zeros((len(time_fractions), n_points))

    for i, tf in enumerate(time_fractions):
        for j, m in enumerate(moneyness_grid):
            t_bin, m_bin = discretize_state((tf, m), n_time=n_time, money_edges=money_edges)
            margin[i, j] = Q[t_bin, m_bin, 1] - Q[t_bin, m_bin, 0]

    return time_fractions, moneyness_grid, margin


def boundary_to_moneyness(boundary, T, K):
    """Convert a Week 4 (time, stock_price) boundary into (time_fraction, moneyness)."""
    if not boundary:
        return np.array([]), np.array([])
    times, stocks = zip(*boundary)
    time_fractions = np.array(times) / T
    moneyness = np.array(stocks) / K
    return time_fractions, moneyness


def plot_exercise_region(time_fractions, moneyness_grid, exercise_grid, title, ref_boundary=None, save_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.pcolormesh(
        time_fractions, moneyness_grid, exercise_grid.T,
        cmap="RdBu_r", shading="auto", vmin=0, vmax=1,
    )
    ax.set_xlabel("time fraction (t / T)")
    ax.set_ylabel("moneyness (S / K)")
    ax.set_title(title)

    if ref_boundary is not None:
        ref_t, ref_m = ref_boundary
        if len(ref_t) > 0:
            ax.plot(ref_t, ref_m, color="black", linewidth=2, label="Week 4 binomial boundary")
            ax.legend(loc="upper right")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_q_margin(time_fractions, moneyness_grid, margin, title, ref_boundary=None, save_path=None):
    import matplotlib.pyplot as plt

    vmax = float(np.max(np.abs(margin))) or 1.0
    fig, ax = plt.subplots(figsize=(7, 5))
    mesh = ax.pcolormesh(
        time_fractions, moneyness_grid, margin.T,
        cmap="RdBu_r", shading="auto", vmin=-vmax, vmax=vmax,
    )
    fig.colorbar(mesh, ax=ax, label="Q(exercise) - Q(hold)")
    ax.set_xlabel("time fraction (t / T)")
    ax.set_ylabel("moneyness (S / K)")
    ax.set_title(title)

    if ref_boundary is not None:
        ref_t, ref_m = ref_boundary
        if len(ref_t) > 0:
            ax.plot(ref_t, ref_m, color="black", linewidth=2, label="Week 4 binomial boundary")
            ax.legend(loc="upper right")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_exercise_step_histogram(results_by_policy, steps, save_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for name, result in results_by_policy.items():
        if len(result["exercise_steps"]) == 0:
            continue
        ax.hist(
            result["exercise_steps"], bins=np.arange(0, steps + 2) - 0.5,
            alpha=0.5, label=name,
        )
    ax.set_xlabel("exercise step")
    ax.set_ylabel("episode count")
    ax.set_title("When each policy chooses to exercise")
    ax.legend()

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig
