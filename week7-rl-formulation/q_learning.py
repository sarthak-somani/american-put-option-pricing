"""Tabular Q-learning prototype for the American put exercise decision.

This is intentionally a toy: state space is discretized into a small
(time x moneyness) grid so the learning target stays concrete and debuggable.
It is a prototype for Week 8, not a production pricer.
"""

import numpy as np

from policies import DEFAULT_MONEY_EDGES, MONEY_MAX, MONEY_MIN, discretize_state

N_ACTIONS = 2


def train_q_learning(
    env,
    episodes=5000,
    n_time=20,
    money_edges=DEFAULT_MONEY_EDGES,
    alpha=0.05,
    epsilon=0.15,
    seed=123,
    exploring_starts=True,
    exploring_start_fraction=0.5,
):
    """Epsilon-greedy tabular Q-learning against `env`, with a warmup-then-refine
    exploring-starts schedule.

    With `exploring_starts=True` (default), the *first* `exploring_start_fraction`
    of episodes begin from a state drawn uniformly over (time, moneyness) rather
    than always from (t=0, S=S0). Under a t=0-only start, spot only reaches
    extreme moneyness values by randomly walking there over many steps, so those
    (state, action) cells get visited too rarely to learn anything -- they stay
    at their zero initialization and the greedy policy silently defaults to
    "hold" there. This warmup phase directly injects experience across the whole
    table instead of relying on the price process to wander everywhere on its
    own.

    The *remaining* episodes reset normally, at contract inception. This refine
    phase matters because pure exploring starts (fraction=1.0) spend a fixed
    episode budget spread across the entire state space, which undertrains the
    specific near-the-money, early-time region that a real t=0 rollout -- and
    any evaluation of this contract -- actually lives in. Splitting the budget
    gets both: full-table coverage from the warmup, then targeted refinement of
    the region that determines the contract's actual price.

    Evaluation (evaluate.run_policy) never uses exploring starts: real episodes
    always begin at contract inception, so evaluation should simulate that, not
    the training distribution.

    Exploration (epsilon-greedy) uses its own RNG (separate from env.rng, which
    drives the simulated price path) so exploration noise and price-path noise
    don't get entangled when reasoning about reproducibility.
    """
    rng = np.random.default_rng(seed)
    n_money = len(money_edges) - 1
    Q = np.zeros((n_time, n_money, N_ACTIONS), dtype=np.float64)
    warmup_episodes = int(episodes * exploring_start_fraction) if exploring_starts else 0

    for ep in range(episodes):
        if ep < warmup_episodes:
            start_step = int(rng.integers(0, env.steps))
            start_moneyness = rng.uniform(MONEY_MIN, MONEY_MAX)
            state = env.reset(step_count=start_step, spot=start_moneyness * env.K)
        else:
            state = env.reset()
        done = False

        while not done:
            s_idx = discretize_state(state, n_time, money_edges)

            if rng.random() < epsilon:
                action = int(rng.integers(0, N_ACTIONS))
            else:
                action = int(np.argmax(Q[s_idx]))

            next_state, reward, done, _ = env.step(action)
            ns_idx = discretize_state(next_state, n_time, money_edges)

            target = reward if done else reward + env.discount * np.max(Q[ns_idx])
            Q[s_idx + (action,)] += alpha * (target - Q[s_idx + (action,)])

            state = next_state

    return Q
