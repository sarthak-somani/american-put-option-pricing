"""Double DQN training loop: epsilon schedule, replay buffer, target network,
interleaved exploring starts, and periodic convergence checkpoints.

Reuses ../week7-rl-formulation/evaluate.py's `run_policy` for checkpoint
evaluation and ../week7-rl-formulation/policies.py's MONEY_MIN/MONEY_MAX for
exploring-start sampling range, via the same sys.path pattern Week 7's own
pipeline.py uses to reach ../binomial-tree.
"""

import os
import random
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week7-rl-formulation"))
from evaluate import run_policy  # noqa: E402
from policies import MONEY_MIN, MONEY_MAX  # noqa: E402

from dqn import QNetwork, ReplayBuffer, compute_dqn_loss, expand_state
from dqn_policies import make_dqn_policy, network_exercise_region

CHECKPOINT_EVAL_SEED = 2024  # fixed across checkpoints: isolates training progress
                              # from Monte Carlo noise by evaluating the same
                              # sequence of price paths at every checkpoint.


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # PyTorch's default intra-op thread pool adds dispatch/sync overhead that
    # dominates actual compute time for a network this tiny (3->64->64->2);
    # single-threaded execution measured ~1.8x faster end to end on this CPU.
    torch.set_num_threads(1)


def epsilon_schedule(episode, episodes, epsilon_start=1.0, epsilon_min=0.05, decay_target_fraction=0.8):
    """Exponential decay whose base is solved so epsilon hits epsilon_min at
    exactly decay_target_fraction * episodes, instead of a hardcoded decay
    constant that silently goes stale if episode count changes. E.g. for
    episodes=20000 and decay_target_fraction=0.8 (floor reached at episode
    16000), this solves to decay ~= 0.999813.
    """
    decay = (epsilon_min / epsilon_start) ** (1.0 / (decay_target_fraction * episodes))
    return max(epsilon_min, epsilon_start * (decay ** episode))


def lr_schedule(episode, episodes, lr_start=1e-3, lr_end=1e-4, decay_target_fraction=0.8):
    """Exponential learning-rate decay, same solved-base construction as
    epsilon_schedule. Motivation: with a constant lr, late-training gradient
    steps are exactly as large as early ones, so the policy keeps getting
    shoved around by ordinary replay noise even after it has found roughly
    the right region -- a plausible explanation for the checkpointed value
    *declining* over training instead of settling. Decaying lr lets later
    updates refine rather than relocate.
    """
    decay = (lr_end / lr_start) ** (1.0 / (decay_target_fraction * episodes))
    return max(lr_end, lr_start * (decay ** episode))


def exploring_start_schedule(episode, episodes, start_fraction=0.7, end_fraction=0.1, decay_target_fraction=0.8):
    """Linearly decay the exploring-start probability from start_fraction
    (early -- broad coverage of the whole (time, moneyness) grid, which is
    what keeps deep-ITM Q-values from drifting the way they did under a
    block warmup-then-refine schedule) to end_fraction (late -- most episodes
    anchor at contract inception, concentrating final training on precisely
    the region a real t=0 rollout, and evaluation, actually lives in).

    This is the same coverage-vs-precision trade-off
    ../week7-rl-formulation/q_learning.py's tabular learner resolved with a
    two-block schedule, adapted for a function approximator: a hard block
    would reintroduce the deep-ITM drift bug (see train_dqn's docstring), so
    coverage needs to persist at some rate for the entire run, but a *flat*
    rate spends the same fraction of the late-training budget on rare
    regions as the early-training budget, diluting the signal that would
    otherwise sharpen the boundary's exact location where it matters.
    """
    target_episode = decay_target_fraction * episodes
    progress = min(1.0, episode / target_episode) if target_episode > 0 else 1.0
    return start_fraction + (end_fraction - start_fraction) * progress


def train_dqn(
    env,
    eval_env_factory,
    episodes=20_000,
    hidden_dim=64,
    lr_start=1e-3,
    lr_end=1e-4,
    batch_size=128,
    buffer_capacity=50_000,
    target_update_every=250,
    epsilon_start=1.0,
    epsilon_min=0.05,
    decay_target_fraction=0.8,
    grad_clip_norm=5.0,
    exploring_starts=True,
    exploring_start_fraction_start=0.7,
    exploring_start_fraction_end=0.1,
    seed=0,
    checkpoint_interval=1000,
    checkpoint_eval_episodes=300,
    checkpoint_grid_n_money=30,
    device=None,
):
    """Train a Double DQN policy against `env`.

    `device` defaults to CUDA if available, else CPU. Note this environment's
    per-episode rollout is inherently sequential (one Python-level env.step()
    at a time, batch size 1 during greedy action selection) and the gradient
    batch itself is tiny (128 x 3 -> 64 -> 64 -> 2) -- GPU kernel-launch and
    host/device sync overhead per call can rival or exceed the actual compute,
    so a GPU is not guaranteed to be faster here than single-threaded CPU
    unless multiple environments are vectorized into one large batch (not
    implemented). Device support is provided so it can be measured rather
    than assumed.

    Interleaved, decaying exploring starts: independently for *each* episode,
    with probability `exploring_start_schedule(episode, ...)` (see that
    function) it starts from a state drawn uniformly over (time, moneyness)
    via `env.reset(step_count=..., spot=...)`; otherwise it starts at
    contract inception. A pure t=0-only schedule under-visits extreme-
    moneyness states within a 50-step random walk from S0=K, so some
    exploring starts are needed throughout -- but a *flat* rate for the
    entire run over-spends late-training budget on those same rare states
    instead of sharpening the boundary's precise location. Decaying the rate
    (high early, low late) keeps both: early broad coverage prevents the
    deep-ITM Q-drift described next, late concentration refines the boundary
    where a real rollout actually lives.

    This is deliberately *not* Week 7's tabular learner's warmup-then-refine
    schedule (all exploring starts in a first block, then all t=0 starts,
    with a fixed split point). A table has one independent, permanent memory
    cell per state, so front-loading coverage works: once a cell is written
    it stays written. A neural net sampling from a finite replay buffer has
    neither property. Measured directly: with a block schedule, deep-ITM
    transitions only enter the buffer during the warmup block, and by
    20,000 episodes the buffer (capacity 50,000) has long since cycled past
    them -- worse, the refine block's t=0-starting policy exercises quickly
    once even mildly ITM, so it rarely re-generates fresh deep-ITM
    transitions on its own either. The result was a confidently *wrong*
    Q-function at deep ITM (favoring hold over an obviously-larger immediate
    payoff) even though the same states looked fine in a mid-training
    checkpoint -- the shared network weights drifted on deep-ITM inputs
    while training continued on the near-the-money states the refine block
    actually visits, and nothing in the data stream corrected it. A
    continuous decay keeps a nonzero trickle of fresh deep-ITM transitions
    arriving for the entire run rather than letting it drop to zero for the
    whole back half.

    Learning-rate decay (`lr_schedule`): applied once per episode. Motivated
    by an earlier run where the checkpointed greedy-policy value *declined*
    over training instead of settling -- a constant lr lets ordinary replay
    noise keep reshuffling an already-reasonable policy indefinitely; decay
    lets late updates refine rather than relocate.

    Returns (online_network, history) where history["episode_log"] has one
    dict per training episode (epsilon, lr, exploring_start_p, terminal
    reward, loss ema, max|Q|) and history["checkpoints"] has one dict every
    `checkpoint_interval` episodes (discounted value + SE from a small
    greedy-policy eval, a coarse exercise-region grid, and a CPU state_dict
    of the network at that point) for the value-convergence curve, boundary
    snapshots, and best-checkpoint selection (`online_network` is simply the
    final episode's network; the best-performing checkpoint by evaluated
    value is not necessarily the last one -- see pipeline.py's
    `select_best_checkpoint`).
    """
    set_seeds(seed)
    explore_rng = np.random.default_rng(seed + 1)
    buffer_rng = np.random.default_rng(seed + 2)
    start_rng = np.random.default_rng(seed + 3)

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    online = QNetwork(hidden_dim=hidden_dim).to(device)
    target = QNetwork(hidden_dim=hidden_dim).to(device)
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(online.parameters(), lr=lr_start)
    buffer = ReplayBuffer(capacity=buffer_capacity)

    updates = 0
    loss_ema = None
    max_q_last = None
    loss_ema_beta = 0.98

    episode_log = []
    checkpoints = []

    for episode in range(episodes):
        epsilon = epsilon_schedule(episode, episodes, epsilon_start, epsilon_min, decay_target_fraction)
        current_lr = lr_schedule(episode, episodes, lr_start, lr_end, decay_target_fraction)
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        exploring_start_p = exploring_start_schedule(
            episode, episodes, exploring_start_fraction_start, exploring_start_fraction_end, decay_target_fraction
        )

        is_exploring_start = exploring_starts and start_rng.random() < exploring_start_p
        if is_exploring_start:
            start_step = int(start_rng.integers(0, env.steps))
            start_moneyness = start_rng.uniform(MONEY_MIN, MONEY_MAX)
            state = env.reset(step_count=start_step, spot=start_moneyness * env.K)
        else:
            state = env.reset()

        done = False
        info = {"reason": "expiry", "step": 0}
        reward = 0.0

        while not done:
            if explore_rng.random() < epsilon:
                action = int(explore_rng.integers(0, 2))
            else:
                online.eval()
                with torch.no_grad():
                    x = torch.tensor(expand_state(state), dtype=torch.float32, device=device).unsqueeze(0)
                    action = int(torch.argmax(online(x), dim=1).item())

            next_state, reward, done, info = env.step(action)
            buffer.push(expand_state(state), action, reward, expand_state(next_state), done)
            state = next_state

            if len(buffer) >= batch_size:
                online.train()
                batch = buffer.sample(batch_size, buffer_rng)
                loss, q_selected = compute_dqn_loss(online, target, batch, env.discount)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(online.parameters(), grad_clip_norm)
                optimizer.step()
                updates += 1

                loss_val = float(loss.item())
                loss_ema = loss_val if loss_ema is None else loss_ema_beta * loss_ema + (1 - loss_ema_beta) * loss_val
                max_q_last = float(q_selected.abs().max().item())

                if updates % target_update_every == 0:
                    target.load_state_dict(online.state_dict())

        episode_log.append({
            "episode": episode,
            "epsilon": epsilon,
            "lr": current_lr,
            "exploring_start_p": exploring_start_p,
            "exploring_start": is_exploring_start,
            "reason": info["reason"],
            "step": info["step"],
            "reward": reward,
            "discounted_reward": reward * (env.discount ** info["step"]),
            "loss_ema": loss_ema,
            "max_q": max_q_last,
            "updates": updates,
        })

        is_last = episode == episodes - 1
        if (episode + 1) % checkpoint_interval == 0 or is_last:
            online.eval()
            policy_fn = make_dqn_policy(online)
            eval_env = eval_env_factory()
            eval_result = run_policy(eval_env, policy_fn, episodes=checkpoint_eval_episodes, seed=CHECKPOINT_EVAL_SEED)
            tf, mg, grid = network_exercise_region(online, env.steps, n_money=checkpoint_grid_n_money)
            checkpoints.append({
                "episode": episode + 1,
                "value": eval_result["discounted_payoff_mean"],
                "se": eval_result["discounted_payoff_std"] / np.sqrt(checkpoint_eval_episodes),
                "exercise_rate": eval_result["exercise_rate"],
                "grid_time_fractions": tf,
                "grid_moneyness": mg,
                "grid_exercise": grid,
                # CPU state_dict so the caller can reconstruct and re-evaluate any
                # checkpoint's weights later -- e.g. to confirm a best-by-checkpoint
                # candidate on a much larger, independent evaluation before trusting
                # it, since `checkpoint_eval_episodes` alone is a small, noisy sample.
                "state_dict": {k: v.detach().cpu().clone() for k, v in online.state_dict().items()},
            })

    return online, {"episode_log": episode_log, "checkpoints": checkpoints}
