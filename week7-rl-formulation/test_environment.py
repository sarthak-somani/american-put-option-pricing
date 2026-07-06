"""Invariant tests for the American put exercise environment, policies, and Q-learning.

These check the environment cannot be "gamed" by a broken formulation: no
leakage, no double payoff, no stepping past done, and correct terminal
handling -- the pitfalls this week's instructional material calls out
explicitly.
"""

import numpy as np
import pytest

from environment import AmericanPutEnv
from policies import (
    always_hold_policy,
    discretize_state,
    immediate_exercise_policy,
    make_q_policy,
    make_random_policy,
)
from evaluate import run_policy
from q_learning import train_q_learning

ENV_KWARGS = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, steps=50)


def test_state_has_no_leakage():
    """State must be exactly [time_fraction, moneyness] -- no future price info."""
    env = AmericanPutEnv(**ENV_KWARGS, seed=1)
    state = env.reset()
    assert state.shape == (2,)
    assert state[0] == 0.0
    assert state[1] == pytest.approx(env.S0 / env.K)


def test_exercise_payoff_nonnegative_and_exact():
    env = AmericanPutEnv(**ENV_KWARGS, seed=1)
    env.reset()
    _, reward, done, info = env.step(env.EXERCISE)
    assert reward >= 0.0
    assert reward == pytest.approx(max(env.K - env.S0, 0.0))
    assert done
    assert info["reason"] == "exercise"


def test_cannot_step_after_done():
    env = AmericanPutEnv(**ENV_KWARGS, seed=1)
    env.reset()
    env.step(env.EXERCISE)
    with pytest.raises(RuntimeError):
        env.step(env.HOLD)


def test_invalid_action_rejected():
    env = AmericanPutEnv(**ENV_KWARGS, seed=1)
    env.reset()
    with pytest.raises(ValueError):
        env.step(2)


def test_hold_never_pays_reward_until_terminal():
    """Reward must be 0.0 on every hold step before expiry (no double-counting)."""
    env = AmericanPutEnv(**ENV_KWARGS, seed=2)
    env.reset()
    for _ in range(env.steps - 1):
        _, reward, done, info = env.step(env.HOLD)
        assert reward == 0.0
        assert not done
        assert info["reason"] == "hold"


def test_episode_terminates_at_expiry_with_correct_payoff():
    env = AmericanPutEnv(**ENV_KWARGS, seed=3)
    env.reset()
    reward, done, info = 0.0, False, {}
    for _ in range(env.steps):
        _, reward, done, info = env.step(env.HOLD)
    assert done
    assert info["reason"] == "expiry"
    assert reward == pytest.approx(max(env.K - env.spot, 0.0))
    with pytest.raises(RuntimeError):
        env.step(env.HOLD)


def test_random_policy_always_terminates():
    """A random hold/exercise policy must always reach exercise or expiry, never loop."""
    env = AmericanPutEnv(**ENV_KWARGS, seed=7)
    policy = make_random_policy(seed=7)
    for _ in range(20):
        state = env.reset()
        done = False
        n_steps = 0
        while not done:
            action = policy(state)
            state, reward, done, info = env.step(action)
            n_steps += 1
            assert n_steps <= env.steps + 1
        assert reward >= 0.0
        assert info["reason"] in ("exercise", "expiry")


def test_immediate_exercise_policy_always_exercises_at_step_zero():
    env = AmericanPutEnv(**ENV_KWARGS, seed=5)
    result = run_policy(env, immediate_exercise_policy, episodes=200)
    assert result["exercise_rate"] == 1.0
    assert np.all(result["exercise_steps"] == 0)
    # discount ** 0 == 1, so raw and discounted payoff must match exactly
    assert result["raw_payoff_mean"] == pytest.approx(result["discounted_payoff_mean"])


def test_always_hold_policy_never_exercises_early():
    env = AmericanPutEnv(**ENV_KWARGS, seed=5)
    result = run_policy(env, always_hold_policy, episodes=200)
    assert result["exercise_rate"] == 0.0
    assert len(result["exercise_steps"]) == 0


def test_discounting_reduces_value_relative_to_raw_payoff():
    """Whenever expiry payoff is positive, discounted value must be <= raw payoff."""
    env = AmericanPutEnv(**ENV_KWARGS, seed=11)
    result = run_policy(env, always_hold_policy, episodes=500)
    assert result["discounted_payoff_mean"] <= result["raw_payoff_mean"] + 1e-9


def test_discretize_state_clips_to_grid_bounds():
    assert discretize_state((-0.5, 0.0), n_time=20) == (0, 0)
    assert discretize_state((2.0, 5.0), n_time=20) == (19, 29)


def test_discretize_state_is_finer_near_the_boundary_region():
    """Moneyness bins are non-uniform: fine across [0.7, 1.0], coarse outside it.

    Two points 0.03 apart inside the fine region must land in different bins;
    the same 0.03 gap far outside it (near 0.55, deep in the coarse region)
    should not, since that region uses much wider bins.
    """
    _, m_bin_a = discretize_state((0.5, 0.80))
    _, m_bin_b = discretize_state((0.5, 0.83))
    assert m_bin_a != m_bin_b

    _, m_bin_c = discretize_state((0.5, 0.51))
    _, m_bin_d = discretize_state((0.5, 0.52))
    assert m_bin_c == m_bin_d


def test_reset_defaults_to_contract_inception():
    env = AmericanPutEnv(**ENV_KWARGS, seed=9)
    state = env.reset()
    assert state[0] == 0.0
    assert env.spot == env.S0


def test_reset_accepts_explicit_exploring_start():
    env = AmericanPutEnv(**ENV_KWARGS, seed=9)
    state = env.reset(step_count=20, spot=90.0)
    assert state[0] == pytest.approx(20 / env.steps)
    assert env.spot == 90.0
    assert not env.done

    # Episode still behaves normally (terminates, pays correctly) from this start.
    _, reward, done, info = env.step(env.EXERCISE)
    assert done
    assert reward == pytest.approx(max(env.K - 90.0, 0.0))
    assert info["reason"] == "exercise"


def test_reset_rejects_invalid_exploring_start():
    env = AmericanPutEnv(**ENV_KWARGS, seed=9)
    with pytest.raises(ValueError):
        env.reset(step_count=env.steps, spot=100.0)  # step_count must be < steps
    with pytest.raises(ValueError):
        env.reset(step_count=0, spot=-10.0)  # spot must be positive


def test_q_learning_exploring_starts_visits_far_from_the_money_states():
    """With exploring starts, training should reach moneyness bins a t=0 rollout
    over 50 steps would rarely touch -- the whole point of improvement #1."""
    env = AmericanPutEnv(**ENV_KWARGS, seed=42)
    Q = train_q_learning(env, episodes=3000, seed=1, exploring_starts=True)

    deep_itm_state = (0.5, 0.55)   # far below the money
    deep_otm_state = (0.5, 1.45)   # far above the money
    assert np.any(Q[discretize_state(deep_itm_state)] != 0.0)
    assert np.any(Q[discretize_state(deep_otm_state)] != 0.0)


def test_exploring_start_fraction_zero_matches_exploring_starts_disabled():
    """fraction=0.0 means no warmup episodes at all -- must be bit-identical to
    exploring_starts=False given the same seed, since both take the same
    (no-exploring-start) branch on every episode."""
    env_a = AmericanPutEnv(**ENV_KWARGS, seed=42)
    Q_no_warmup = train_q_learning(env_a, episodes=500, seed=1, exploring_start_fraction=0.0)

    env_b = AmericanPutEnv(**ENV_KWARGS, seed=42)
    Q_disabled = train_q_learning(env_b, episodes=500, seed=1, exploring_starts=False)

    assert np.array_equal(Q_no_warmup, Q_disabled)


def test_warmup_then_refine_achieves_full_table_coverage():
    """The warmup phase alone must be enough to visit every (state, action)
    cell given a reasonable episode budget -- the whole point of splitting the
    schedule instead of using exploring starts for 100% of episodes."""
    env = AmericanPutEnv(**ENV_KWARGS, seed=42)
    Q = train_q_learning(env, episodes=10000, seed=123, exploring_start_fraction=0.5)
    unvisited = np.sum((Q[:, :, 0] == 0.0) & (Q[:, :, 1] == 0.0))
    assert unvisited == 0


def test_q_learning_produces_finite_table_and_valid_greedy_policy():
    env = AmericanPutEnv(**ENV_KWARGS, seed=42)
    Q = train_q_learning(env, episodes=200, seed=1)
    assert np.all(np.isfinite(Q))

    policy = make_q_policy(Q)
    state = env.reset()
    action = policy(state)
    assert action in (env.HOLD, env.EXERCISE)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
