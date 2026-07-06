"""End-to-end Week 7 driver: environment sanity, policy comparison, tabular Q-learning,
and exercise-region plots against the Week 4 binomial boundary.

Run with: python pipeline.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")  # headless script: only ever saves figures, never shows them

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "binomial-tree"))
from american_put import crr_put_price, crr_put_with_boundary  # noqa: E402

from environment import AmericanPutEnv
from policies import (
    always_hold_policy,
    immediate_exercise_policy,
    make_q_policy,
    make_random_policy,
)
from evaluate import (
    boundary_to_moneyness,
    plot_exercise_region,
    plot_exercise_step_histogram,
    plot_q_margin,
    policy_exercise_region,
    policy_q_margin,
    run_policy,
)
from q_learning import train_q_learning

ENV_KWARGS = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, steps=50)
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")


def main():
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # --- Reference: Week 4 binomial price + exercise boundary on the same grid ---
    ref_price, ref_boundary = crr_put_with_boundary(**ENV_KWARGS)
    euro_price = crr_put_price(**ENV_KWARGS, american=False)
    ref_t, ref_m = boundary_to_moneyness(ref_boundary, ENV_KWARGS["T"], ENV_KWARGS["K"])
    print(f"Week 4 binomial American put price ({ENV_KWARGS['steps']} steps): {ref_price:.4f}")
    print(f"Week 4 binomial European put price ({ENV_KWARGS['steps']} steps): {euro_price:.4f}")

    # --- Part B: sample episodes under a random policy, show exercise/expiry reasons ---
    env = AmericanPutEnv(**ENV_KWARGS, seed=11)
    random_policy = make_random_policy(seed=11)
    print("\nFive sample episodes (random policy):")
    for i in range(5):
        state = env.reset()
        done = False
        while not done:
            action = random_policy(state)
            state, reward, done, info = env.step(action)
        print(f"  episode {i}: reason={info['reason']}, step={info['step']}, "
              f"reward={reward:.4f}, final moneyness={state[1]:.4f}")

    # Random policy exercises almost immediately in expectation (mean ~2 steps for a
    # fair coin), so it rarely survives to expiry -- show that expiry termination
    # also works correctly via always-hold.
    state = env.reset()
    done = False
    while not done:
        state, reward, done, info = env.step(env.HOLD)
    print(f"  (always-hold, for contrast): reason={info['reason']}, step={info['step']}, "
          f"reward={reward:.4f}, final moneyness={state[1]:.4f}")

    # --- Part C: train tabular Q-learning (warmup-then-refine: see q_learning.py) ---
    print("\nTraining tabular Q-learning (20000 episodes, 50% exploring-start warmup)...")
    train_env = AmericanPutEnv(**ENV_KWARGS, seed=42)
    Q = train_q_learning(train_env, episodes=20000, seed=123)
    q_policy = make_q_policy(Q)

    # --- Part C: policy comparison over >= 1000 episodes ---
    eval_episodes = 2000
    policies = {
        "always_hold": always_hold_policy,
        "immediate_exercise": immediate_exercise_policy,
        "random": make_random_policy(seed=99),
        "q_learned": q_policy,
    }

    print(f"\nPolicy comparison over {eval_episodes} episodes:")
    print(f"{'policy':20s} {'raw payoff':>12s} {'disc. payoff':>13s} {'exercise rate':>14s}")
    results = {}
    for name, policy_fn in policies.items():
        eval_env = AmericanPutEnv(**ENV_KWARGS, seed=2024)
        result = run_policy(eval_env, policy_fn, episodes=eval_episodes, seed=2024)
        results[name] = result
        print(f"{name:20s} {result['raw_payoff_mean']:12.4f} "
              f"{result['discounted_payoff_mean']:13.4f} {result['exercise_rate']:14.2%}")

    print(f"\nReference (Week 4 binomial, {ENV_KWARGS['steps']} steps): "
          f"American={ref_price:.4f}, European={euro_price:.4f}")

    # --- Figures ---
    plot_exercise_step_histogram(
        {k: v for k, v in results.items() if k != "always_hold"},
        steps=ENV_KWARGS["steps"],
        save_path=os.path.join(FIGURES_DIR, "exercise_step_histogram.png"),
    )

    tf, mg, grid = policy_exercise_region(q_policy, ENV_KWARGS["steps"])
    plot_exercise_region(
        tf, mg, grid, "Learned Q-policy exercise region vs. Week 4 boundary",
        ref_boundary=(ref_t, ref_m),
        save_path=os.path.join(FIGURES_DIR, "q_policy_exercise_region.png"),
    )

    tf, mg, grid = policy_exercise_region(random_policy, ENV_KWARGS["steps"])
    plot_exercise_region(
        tf, mg, grid, "Random policy exercise region (debugging baseline)",
        ref_boundary=(ref_t, ref_m),
        save_path=os.path.join(FIGURES_DIR, "random_policy_exercise_region.png"),
    )

    tf, mg, margin = policy_q_margin(Q, ENV_KWARGS["steps"])
    plot_q_margin(
        tf, mg, margin, "Learned Q-margin [Q(exercise) - Q(hold)] vs. Week 4 boundary",
        ref_boundary=(ref_t, ref_m),
        save_path=os.path.join(FIGURES_DIR, "q_policy_margin.png"),
    )

    print(f"\nFigures written to {FIGURES_DIR}")


if __name__ == "__main__":
    main()
