"""Fast unit/smoke tests for the Week 8 DQN pipeline.

Deliberately does not run a full 20,000-episode training loop -- that
belongs in pipeline.py. Tests here use tiny episode counts so the suite
stays fast, matching ../week6-neural-pricer/test_neural_pricer.py's and
../week7-rl-formulation/test_environment.py's convention of testing
components, not full pipelines.
"""

import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

from dqn import (
    QNetwork,
    ReplayBuffer,
    compute_dqn_loss,
    expand_state,
    load_artifact,
    save_artifact,
)
from dqn_policies import greedy_action, make_dqn_policy, network_exercise_region, network_q_margin
from diagnostics import boundary_monotonicity_report, theoretical_bounds_check
from train import epsilon_schedule, exploring_start_schedule, lr_schedule, set_seeds, train_dqn

# ../week7-rl-formulation has its own same-named pipeline.py; a bare `import pipeline`
# after that directory is on sys.path (below) resolves ambiguously depending on path
# order. Load *this* directory's pipeline.py by explicit file path to sidestep it.
_pipeline_path = os.path.join(os.path.dirname(__file__), "pipeline.py")
_spec = importlib.util.spec_from_file_location("week8_pipeline", _pipeline_path)
_week8_pipeline = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_week8_pipeline)
select_best_checkpoint = _week8_pipeline.select_best_checkpoint

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week7-rl-formulation"))
from environment import AmericanPutEnv  # noqa: E402


ENV_KWARGS = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, steps=50)


def make_env(seed=1):
    return AmericanPutEnv(**ENV_KWARGS, seed=seed)


# --- state expansion -----------------------------------------------------

def test_expand_state_appends_time_to_expiry():
    state = np.array([0.3, 1.2], dtype=np.float32)
    expanded = expand_state(state)
    assert expanded.shape == (3,)
    assert expanded[0] == pytest.approx(0.3)
    assert expanded[1] == pytest.approx(0.7)
    assert expanded[2] == pytest.approx(1.2)


def test_expand_state_no_future_leakage():
    """expand_state is a pure function of the current state only -- it must
    not require or accept anything about future steps."""
    import inspect
    params = inspect.signature(expand_state).parameters
    assert list(params) == ["state"]


# --- network ---------------------------------------------------------------

def test_qnetwork_output_shape():
    net = QNetwork(state_dim=3, hidden_dim=16, action_dim=2)
    x = torch.zeros((5, 3), dtype=torch.float32)
    out = net(x)
    assert out.shape == (5, 2)


# --- replay buffer -----------------------------------------------------------

def test_replay_buffer_push_and_len():
    buf = ReplayBuffer(capacity=10)
    for i in range(5):
        buf.push(np.array([0.0, 1.0]), 0, 0.0, np.array([0.1, 1.0]), False)
    assert len(buf) == 5


def test_replay_buffer_evicts_at_capacity():
    buf = ReplayBuffer(capacity=10)
    for i in range(15):
        buf.push(np.array([float(i), 1.0]), 0, 0.0, np.array([float(i + 1), 1.0]), False)
    assert len(buf) == 10
    # oldest entries (state time_fraction 0..4) should have been evicted
    remaining_first_coords = {t[0][0] for t in buf.buffer}
    assert remaining_first_coords == set(float(i) for i in range(5, 15))


def test_replay_buffer_sample_uses_given_rng():
    buf = ReplayBuffer(capacity=100)
    for i in range(100):
        buf.push(np.array([float(i), 1.0]), i % 2, float(i), np.array([float(i), 1.0]), False)
    rng_a = np.random.default_rng(0)
    rng_b = np.random.default_rng(0)
    sample_a = buf.sample(8, rng_a)
    sample_b = buf.sample(8, rng_b)
    assert sample_a[2] == sample_b[2]  # same rewards drawn given the same seeded rng


# --- Double DQN loss ---------------------------------------------------------

def test_compute_dqn_loss_finite_and_backprop():
    online = QNetwork(state_dim=3, hidden_dim=8)
    target = QNetwork(state_dim=3, hidden_dim=8)
    target.load_state_dict(online.state_dict())

    batch = (
        [np.array([0.1, 0.9, 1.0], dtype=np.float32)] * 4,
        [0, 1, 0, 1],
        [0.0, 5.0, 0.0, 0.0],
        [np.array([0.12, 0.88, 0.98], dtype=np.float32)] * 4,
        [False, True, False, False],
    )
    loss, q_selected = compute_dqn_loss(online, target, batch, discount=0.999)
    assert torch.isfinite(loss)
    assert loss.item() >= 0.0

    online.zero_grad()
    loss.backward()
    grads = [p.grad for p in online.parameters()]
    assert all(g is not None for g in grads)
    assert any(torch.any(g != 0) for g in grads)
    # target network must not receive gradients
    assert all(p.grad is None for p in target.parameters())


def test_compute_dqn_loss_masks_terminal_bootstrap():
    """A done=True transition's target must equal reward alone (no next-state
    bootstrap), regardless of what the networks predict for next_state."""
    online = QNetwork(state_dim=3, hidden_dim=8)
    target = QNetwork(state_dim=3, hidden_dim=8)

    state = np.array([0.5, 0.5, 0.9], dtype=np.float32)
    next_state = np.array([0.5, 0.5, 0.9], dtype=np.float32)
    batch = ([state], [1], [7.5], [next_state], [True])

    with torch.no_grad():
        q_selected_pred = online(torch.tensor(state, dtype=torch.float32).unsqueeze(0))[0, 1].item()

    loss, _ = compute_dqn_loss(online, target, batch, discount=0.99)
    expected_loss = torch.nn.functional.smooth_l1_loss(
        torch.tensor([q_selected_pred]), torch.tensor([7.5])
    )
    assert loss.item() == pytest.approx(expected_loss.item(), abs=1e-5)


# --- epsilon schedule --------------------------------------------------------

def test_epsilon_schedule_monotonic_and_bounded():
    episodes = 1000
    values = [epsilon_schedule(e, episodes) for e in range(episodes)]
    assert all(v <= 1.0 + 1e-9 for v in values)
    assert all(v >= 0.05 - 1e-9 for v in values)
    assert all(values[i] >= values[i + 1] - 1e-9 for i in range(len(values) - 1))


def test_epsilon_schedule_hits_floor_near_target_fraction():
    episodes = 2000
    target_fraction = 0.8
    eps_at_target = epsilon_schedule(int(episodes * target_fraction), episodes, decay_target_fraction=target_fraction)
    assert eps_at_target == pytest.approx(0.05, abs=1e-3)
    eps_well_before = epsilon_schedule(int(episodes * 0.3), episodes, decay_target_fraction=target_fraction)
    assert eps_well_before > 0.2  # still exploring well before the target fraction


# --- diagnostics --------------------------------------------------------------

def test_boundary_monotonicity_clean_grid():
    # Each row: exercise (1) at low moneyness, hold (0) at high moneyness -> 1 flip.
    grid = np.array([[1, 1, 1, 0, 0, 0] for _ in range(10)])
    report = boundary_monotonicity_report(grid)
    assert report["violation_rows"] == 0
    assert report["monotonicity_score"] == pytest.approx(1.0)


def test_boundary_monotonicity_speckled_grid():
    grid = np.array([[1, 0, 1, 0, 1, 0] for _ in range(10)])  # 5 flips per row
    report = boundary_monotonicity_report(grid)
    assert report["violation_rows"] == 10
    assert report["monotonicity_score"] == pytest.approx(0.0)
    assert report["max_flips_in_a_row"] == 5


def test_theoretical_bounds_check_flags_value_above_binomial():
    result = theoretical_bounds_check(rl_value=20.0, rl_se=0.01, european_price=7.4, american_price=7.95)
    assert result["exceeds_binomial"] is True


def test_theoretical_bounds_check_accepts_value_between_bounds():
    result = theoretical_bounds_check(rl_value=7.6, rl_se=0.05, european_price=7.4, american_price=7.95)
    assert result["exceeds_binomial"] is False
    assert result["below_always_hold"] is False


def test_theoretical_bounds_check_below_always_hold_is_not_a_bug_flag():
    # A legitimate but bad policy (e.g. immediate exercise at S0=K) can score
    # far below the always-hold price without any bug -- this must be
    # reported separately from exceeds_binomial, not folded into one verdict.
    result = theoretical_bounds_check(rl_value=0.0, rl_se=0.0, european_price=7.4, american_price=7.95)
    assert result["below_always_hold"] is True
    assert result["exceeds_binomial"] is False


# --- policy wrapper / grids ---------------------------------------------------

def test_greedy_action_is_valid_and_deterministic():
    net = QNetwork(state_dim=3, hidden_dim=8)
    state = np.array([0.4, 1.05], dtype=np.float32)
    a1 = greedy_action(net, state)
    a2 = greedy_action(net, state)
    assert a1 in (0, 1)
    assert a1 == a2


def test_network_exercise_region_and_margin_shapes():
    net = QNetwork(state_dim=3, hidden_dim=8)
    tf, mg, grid = network_exercise_region(net, steps=10, n_money=15)
    assert grid.shape == (11, 15)
    assert set(np.unique(grid)).issubset({0, 1})

    tf2, mg2, margin = network_q_margin(net, steps=10, n_money=15)
    assert margin.shape == (11, 15)
    assert np.array_equal(tf, tf2)


# --- artifact save/load -------------------------------------------------------

def test_artifact_round_trip(tmp_path):
    net = QNetwork(state_dim=3, hidden_dim=8)
    path = tmp_path / "artifact.pt"
    save_artifact(str(path), net, hyperparams={"lr": 1e-3}, seed=42)

    loaded = load_artifact(str(path))
    net2 = QNetwork(state_dim=3, hidden_dim=8)
    net2.load_state_dict(loaded["model_state"])

    x = torch.randn(1, 3)
    with torch.no_grad():
        assert torch.allclose(net(x), net2(x))
    assert loaded["hyperparams"] == {"lr": 1e-3}
    assert loaded["seed"] == 42


# --- small end-to-end training smoke test -------------------------------------

def test_train_dqn_smoke_runs_and_returns_valid_policy():
    train_env = make_env(seed=1)

    def eval_env_factory():
        return make_env(seed=2024)

    online, history = train_dqn(
        train_env,
        eval_env_factory,
        episodes=200,
        hidden_dim=8,
        batch_size=16,
        buffer_capacity=500,
        target_update_every=20,
        checkpoint_interval=100,
        checkpoint_eval_episodes=20,
        checkpoint_grid_n_money=5,
        seed=7,
    )

    assert len(history["episode_log"]) == 200
    assert len(history["checkpoints"]) == 2  # at episode 100 and 200

    policy_fn = make_dqn_policy(online)
    state = train_env.reset()
    action = policy_fn(state)
    assert action in (0, 1)

    for cp in history["checkpoints"]:
        assert np.isfinite(cp["value"])
        assert cp["se"] >= 0.0
        assert "state_dict" in cp


def test_select_best_checkpoint_returns_valid_confirmed_choice():
    train_env = make_env(seed=1)

    def eval_env_factory():
        return make_env(seed=2024)

    _, history = train_dqn(
        train_env,
        eval_env_factory,
        episodes=300,
        hidden_dim=8,
        batch_size=16,
        buffer_capacity=500,
        target_update_every=20,
        checkpoint_interval=100,
        checkpoint_eval_episodes=20,
        checkpoint_grid_n_money=5,
        seed=7,
    )

    best_state_dict, selection = select_best_checkpoint(history, hidden_dim=8,
                                                          eval_env_factory=eval_env_factory,
                                                          confirm_episodes=50)

    assert selection["selected"] in selection["candidates"]
    for name, c in selection["candidates"].items():
        assert np.isfinite(c["confirmed_value"])
        assert c["confirmed_se"] >= 0.0

    # the returned state_dict must actually load into a fresh network of the same shape
    net = QNetwork(hidden_dim=8)
    net.load_state_dict(best_state_dict)
    x = torch.zeros((1, 3), dtype=torch.float32)
    assert net(x).shape == (1, 2)

    # selecting among a single candidate (no distinct peak) must not crash
    single = {"checkpoints": [history["checkpoints"][-1]]}
    _, selection_single = select_best_checkpoint(single, hidden_dim=8,
                                                   eval_env_factory=eval_env_factory,
                                                   confirm_episodes=50)
    assert selection_single["selected"] == "final"
    assert list(selection_single["candidates"].keys()) == ["final"]


def test_exploring_starts_are_interleaved_and_decay_not_a_block_schedule():
    """Regression test for the deep-ITM Q-value drift bug: exploring starts
    must never drop to *zero* for an extended late stretch of training --
    that's what let the replay buffer stop receiving fresh deep-ITM
    transitions under the old block warmup-then-refine schedule (see the
    module docstring in train.py). The rate is expected to *decay* (more
    coverage early, more precision-focused t=0 starts late, per
    exploring_start_schedule), so this checks the direction and a nonzero
    late-training floor, not a flat range like the pre-decay version of this
    test did."""
    train_env = make_env(seed=1)

    def eval_env_factory():
        return make_env(seed=2024)

    _, history = train_dqn(
        train_env,
        eval_env_factory,
        episodes=400,
        hidden_dim=8,
        batch_size=16,
        buffer_capacity=500,
        target_update_every=20,
        checkpoint_interval=400,
        checkpoint_eval_episodes=10,
        checkpoint_grid_n_money=5,
        seed=7,
    )
    first_half = [e["exploring_start"] for e in history["episode_log"][:200]]
    second_half = [e["exploring_start"] for e in history["episode_log"][200:]]
    first_rate = sum(first_half) / len(first_half)
    second_rate = sum(second_half) / len(second_half)

    assert first_rate > second_rate  # decays, as designed
    assert second_rate > 0.0  # but never hits a zero-for-the-rest-of-training block
    last_50 = [e["exploring_start"] for e in history["episode_log"][-50:]]
    assert sum(last_50) > 0  # explicitly: even the tail still gets some exploring starts


def test_exploring_start_schedule_decays_and_floors():
    values = [exploring_start_schedule(e, 1000, start_fraction=0.7, end_fraction=0.1, decay_target_fraction=0.8)
              for e in range(1000)]
    assert values[0] == pytest.approx(0.7)
    assert values[-1] == pytest.approx(0.1, abs=1e-6)
    assert all(values[i] >= values[i + 1] - 1e-9 for i in range(len(values) - 1))  # monotonic decay


def test_lr_schedule_decays_and_floors():
    values = [lr_schedule(e, 1000, lr_start=1e-3, lr_end=1e-4) for e in range(1000)]
    assert values[0] == pytest.approx(1e-3)
    assert values[-1] == pytest.approx(1e-4, rel=1e-3)
    assert all(values[i] >= values[i + 1] - 1e-12 for i in range(len(values) - 1))


def test_set_seeds_reproducible_numpy_and_python_random():
    import random
    set_seeds(123)
    a = [random.random() for _ in range(5)]
    b = np.random.rand(5)
    set_seeds(123)
    a2 = [random.random() for _ in range(5)]
    b2 = np.random.rand(5)
    assert a == a2
    assert np.array_equal(b, b2)
