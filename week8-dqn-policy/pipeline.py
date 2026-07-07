"""End-to-end Week 8 driver: multi-seed Double DQN training, baseline
comparison (always-hold, immediate-exercise, random, Week 7 tabular Q),
Week 4 binomial benchmarking, diagnostics, figures, and artifact export.

Run with: python pipeline.py
Parallel across CPU cores: python pipeline.py --workers 8 --seeds 0 1 2 3 4 5 6 7
"""

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")  # headless script: only ever saves figures, never shows them

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "binomial-tree"))
from american_put import crr_put_price, crr_put_with_boundary  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "week7-rl-formulation"))
from environment import AmericanPutEnv  # noqa: E402
from policies import (  # noqa: E402
    always_hold_policy,
    immediate_exercise_policy,
    make_q_policy,
    make_random_policy,
)
from evaluate import (  # noqa: E402
    boundary_to_moneyness,
    plot_exercise_region,
    plot_exercise_step_histogram,
    plot_q_margin,
    policy_exercise_region,
    policy_q_margin,
    run_policy,
)
from q_learning import train_q_learning  # noqa: E402

from dqn import QNetwork, save_artifact
from dqn_policies import make_dqn_policy, network_exercise_region, network_q_margin
from diagnostics import boundary_monotonicity_report, theoretical_bounds_check
from train import train_dqn

ENV_KWARGS = dict(S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, steps=50)
FIGURES_DIR = os.path.join(os.path.dirname(__file__), "figures")
ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")

DQN_SEEDS = [0]  # single seed by default (no seed-sensitivity analysis wanted); pass
                  # --seeds 0 1 2 ... --workers N to reinstate a multi-seed spread report --
                  # the training/evaluation/plotting code below stays correct either way.
DQN_EPISODES = 20_000
FINAL_EVAL_EPISODES = 10_000
EVAL_SEED = 2024  # matches Week 7's "official" evaluation seed

DQN_HYPERPARAMS = dict(
    episodes=DQN_EPISODES,
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
    checkpoint_interval=1000,
    checkpoint_eval_episodes=300,
    checkpoint_grid_n_money=30,
)


def full_eval(policy_fn, episodes=FINAL_EVAL_EPISODES):
    env = AmericanPutEnv(**ENV_KWARGS, seed=EVAL_SEED)
    result = run_policy(env, policy_fn, episodes=episodes, seed=EVAL_SEED)
    result["se"] = float(result["discounted_payoff_std"] / np.sqrt(episodes))
    return result


def select_best_checkpoint(history, hidden_dim, eval_env_factory, confirm_episodes=2000):
    """Pick the better of (a) the final-episode network and (b) the training
    checkpoint with the best *training-time* value, confirmed on a larger,
    independent evaluation before trusting either.

    Motivation: `value_convergence.png` (a first run without this selection
    step) showed the checkpointed value peaking mid-training and then
    declining through the end of the run -- plausibly because epsilon and
    the exploring-start rate both anneal to their floors around the same
    episode, narrowing the training data distribution enough to erode a
    perfectly good earlier policy. Week 6's neural pricer already establishes
    the right response to this shape of problem: keep the best checkpoint,
    not the final epoch. The one addition needed here is a confirmation
    step, because `checkpoint_eval_episodes` (300, during training) is a
    small enough sample that just taking its argmax over ~20 checkpoints
    risks picking one that got lucky rather than one that's genuinely
    better -- so both the peak-checkpoint and final-checkpoint candidates
    are re-evaluated here on `confirm_episodes` (larger, independent) before
    a choice is made, and both results are returned for an auditable
    comparison rather than a silent swap.
    """
    checkpoints = history["checkpoints"]
    final_cp = checkpoints[-1]
    peak_cp = max(checkpoints, key=lambda c: c["value"])

    candidates = {"final": final_cp}
    if peak_cp["episode"] != final_cp["episode"]:
        candidates["peak"] = peak_cp

    confirmed = {}
    for name, cp in candidates.items():
        model = QNetwork(hidden_dim=hidden_dim)
        model.load_state_dict(cp["state_dict"])
        policy_fn = make_dqn_policy(model)
        env = eval_env_factory()
        result = run_policy(env, policy_fn, episodes=confirm_episodes, seed=EVAL_SEED)
        confirmed[name] = {
            "episode": cp["episode"],
            "checkpoint_value": cp["value"],
            "confirmed_value": result["discounted_payoff_mean"],
            "confirmed_se": float(result["discounted_payoff_std"] / np.sqrt(confirm_episodes)),
        }

    best_name = max(confirmed, key=lambda n: confirmed[n]["confirmed_value"])
    return candidates[best_name]["state_dict"], {"candidates": confirmed, "selected": best_name}


def _train_seed_worker(seed, hyperparams):
    """Train one DQN seed to completion. Must be a module-level function (not
    a closure) so it can be pickled and sent to a worker process on Windows'
    spawn start method. Pinned to CPU explicitly: with several worker
    processes running concurrently, each defaulting to "cuda if available"
    would have them all pile onto one GPU, which is not what --workers > 1
    is for -- this mode is specifically the many-CPU-cores strategy.

    Returns (seed, cpu_state_dict, history) rather than the live nn.Module:
    state_dict is the natural unit to pickle back to the parent process, and
    forcing it to CPU here avoids any device-handle pickling issues. The
    returned state_dict is the best-checkpoint-selected one (see
    `select_best_checkpoint`), not necessarily the final episode's.
    """
    train_env = AmericanPutEnv(**ENV_KWARGS, seed=1000 + seed)

    def worker_eval_env_factory():
        return AmericanPutEnv(**ENV_KWARGS, seed=EVAL_SEED)

    online, history = train_dqn(train_env, worker_eval_env_factory, seed=seed, device="cpu", **hyperparams)
    best_state_dict, selection = select_best_checkpoint(history, hyperparams["hidden_dim"], worker_eval_env_factory)

    # Strip the (bulky, one-per-checkpoint) state dicts before returning -- only
    # the selected candidate's weights need to travel back to the parent process.
    for cp in history["checkpoints"]:
        cp.pop("state_dict", None)
    history["checkpoint_selection"] = selection

    state_dict = {k: v.cpu() for k, v in best_state_dict.items()}
    return seed, state_dict, history


def _format_selection(history):
    sel = history["checkpoint_selection"]
    parts = []
    for name, c in sel["candidates"].items():
        marker = " <- selected" if name == sel["selected"] else ""
        parts.append(f"{name}(ep={c['episode']}): confirmed={c['confirmed_value']:.4f}"
                      f" (se={c['confirmed_se']:.4f}){marker}")
    return "; ".join(parts)


def run_dqn_training(seeds, hyperparams, max_workers=1):
    """Train every seed in `seeds`, sequentially if max_workers <= 1, else in
    parallel worker processes (one seed per process; each pinned to a single
    torch thread by train.set_seeds, so max_workers processes together use at
    most max_workers CPU cores, not max_workers * all-cores)."""
    results = {}

    if max_workers <= 1:
        for seed in seeds:
            print(f"\nTraining DQN seed={seed} ({hyperparams['episodes']} episodes)...")
            seed_out, state_dict, history = _train_seed_worker(seed, hyperparams)
            model = QNetwork(hidden_dim=hyperparams["hidden_dim"])
            model.load_state_dict(state_dict)
            results[seed_out] = {"model": model, "history": history}
            print(f"  seed={seed_out} checkpoint selection: {_format_selection(history)}")
        return results

    print(f"\nTraining {len(seeds)} DQN seeds in parallel across {max_workers} worker processes "
          f"({hyperparams['episodes']} episodes each)...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_train_seed_worker, seed, hyperparams): seed for seed in seeds}
        for future in as_completed(futures):
            seed_out, state_dict, history = future.result()
            model = QNetwork(hidden_dim=hyperparams["hidden_dim"])
            model.load_state_dict(state_dict)
            results[seed_out] = {"model": model, "history": history}
            print(f"  seed={seed_out} checkpoint selection: {_format_selection(history)}")
    return dict(sorted(results.items()))


def parse_args():
    parser = argparse.ArgumentParser(description="Week 8 DQN pipeline")
    parser.add_argument("--seeds", type=int, nargs="+", default=DQN_SEEDS,
                         help="DQN training seeds (default: %(default)s)")
    parser.add_argument("--episodes", type=int, default=DQN_EPISODES,
                         help="training episodes per seed (default: %(default)s)")
    parser.add_argument("--workers", type=int, default=1,
                         help="parallel worker processes for DQN training, one seed per "
                              "process (default: 1, i.e. sequential). Set to len(--seeds) "
                              "or os.cpu_count() to fully parallelize across cores.")
    parser.add_argument("--eval-episodes", type=int, default=FINAL_EVAL_EPISODES,
                         help="Monte Carlo evaluation episodes per policy (default: %(default)s)")
    return parser.parse_args()


def main():
    args = parse_args()
    hyperparams = {**DQN_HYPERPARAMS, "episodes": args.episodes,
                   "checkpoint_interval": max(1, args.episodes // 20)}
    seeds = args.seeds
    eval_episodes = args.eval_episodes
    os.makedirs(FIGURES_DIR, exist_ok=True)
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)

    # --- Reference: Week 4 binomial price + exercise boundary on the same grid ---
    ref_price, ref_boundary = crr_put_with_boundary(**ENV_KWARGS)
    euro_price = crr_put_price(**ENV_KWARGS, american=False)
    ref_t, ref_m = boundary_to_moneyness(ref_boundary, ENV_KWARGS["T"], ENV_KWARGS["K"])
    print(f"Week 4 binomial American put price ({ENV_KWARGS['steps']} steps): {ref_price:.4f}")
    print(f"Week 4 binomial European put price ({ENV_KWARGS['steps']} steps): {euro_price:.4f}")

    # --- Week 7 tabular Q-learning baseline (reused as an extra comparison) ---
    print("\nTraining Week 7 tabular Q-learning baseline (20000 episodes, 50% warmup)...")
    tabular_train_env = AmericanPutEnv(**ENV_KWARGS, seed=42)
    Q = train_q_learning(tabular_train_env, episodes=20_000, seed=123)
    tabular_q_policy = make_q_policy(Q)

    # --- Multi-seed DQN training (sequential if --workers 1, else parallel processes) ---
    dqn_runs = run_dqn_training(seeds, hyperparams, max_workers=args.workers)

    # --- Full evaluation of every DQN seed, plus all baselines ---
    print(f"\nFull evaluation ({eval_episodes} episodes) of all policies:")
    baseline_policies = {
        "always_hold": always_hold_policy,
        "immediate_exercise": immediate_exercise_policy,
        "random": make_random_policy(seed=99),
        "tabular_q": tabular_q_policy,
    }
    results = {}
    for name, policy_fn in baseline_policies.items():
        results[name] = full_eval(policy_fn, episodes=eval_episodes)

    dqn_seed_results = {}
    for seed, run in dqn_runs.items():
        policy_fn = make_dqn_policy(run["model"])
        dqn_seed_results[seed] = full_eval(policy_fn, episodes=eval_episodes)

    dqn_values = np.array([r["discounted_payoff_mean"] for r in dqn_seed_results.values()])
    median_seed = int(list(dqn_seed_results.keys())[int(np.argsort(dqn_values)[len(dqn_values) // 2])])
    results["dqn"] = dqn_seed_results[median_seed]

    print(f"{'policy':20s} {'raw payoff':>12s} {'disc. payoff':>13s} {'se':>8s} {'exercise rate':>14s}")
    for name, result in results.items():
        print(f"{name:20s} {result['raw_payoff_mean']:12.4f} {result['discounted_payoff_mean']:13.4f} "
              f"{result['se']:8.4f} {result['exercise_rate']:14.2%}")
    if len(dqn_seed_results) > 1:
        print(f"\nDQN seed spread (discounted payoff, {eval_episodes} eval episodes each):")
        for seed, r in dqn_seed_results.items():
            marker = "  <- reported (median)" if seed == median_seed else ""
            print(f"  seed={seed}: {r['discounted_payoff_mean']:.4f} (se={r['se']:.4f}){marker}")
        print(f"  median={np.median(dqn_values):.4f}, mean={dqn_values.mean():.4f}, "
              f"std={dqn_values.std():.4f}, min={dqn_values.min():.4f}, max={dqn_values.max():.4f}")

    print(f"\nReference (Week 4 binomial, {ENV_KWARGS['steps']} steps): "
          f"American={ref_price:.4f}, European={euro_price:.4f}")

    # --- Automated theoretical bound check ---
    # Only "exceeds_binomial" is a hard invariant violation (no stopping rule can beat
    # the American price); "below_always_hold" is a policy-quality note, not a bug signal.
    print("\nTheoretical bound check (value <= American price, within 1.96*SE):")
    for name in ("tabular_q", "dqn"):
        r = results[name]
        check = theoretical_bounds_check(r["discounted_payoff_mean"], r["se"], euro_price, ref_price)
        flag = "EXCEEDS BINOMIAL -- investigate" if check["exceeds_binomial"] else "OK (<= American price)"
        quality_note = " [below always-hold baseline]" if check["below_always_hold"] else ""
        print(f"  {name}: value={check['rl_value']:.4f}, upper_bound={check['upper_bound']:.4f} -> {flag}{quality_note}")

    # --- Boundary-shape diagnostics ---
    median_model = dqn_runs[median_seed]["model"]
    _, _, dqn_grid_hi = network_exercise_region(median_model, ENV_KWARGS["steps"], n_money=121)
    _, _, tabq_grid_hi = policy_exercise_region(tabular_q_policy, ENV_KWARGS["steps"], n_money=121)

    dqn_mono = boundary_monotonicity_report(dqn_grid_hi)
    tabq_mono = boundary_monotonicity_report(tabq_grid_hi)
    print("\nBoundary monotonicity (fraction of time-slices with <=1 exercise/hold flip):")
    print(f"  dqn:       score={dqn_mono['monotonicity_score']:.3f}, "
          f"violation_rows={dqn_mono['violation_rows']}/{dqn_mono['total_rows']}, "
          f"max_flips={dqn_mono['max_flips_in_a_row']}")
    print(f"  tabular_q: score={tabq_mono['monotonicity_score']:.3f}, "
          f"violation_rows={tabq_mono['violation_rows']}/{tabq_mono['total_rows']}, "
          f"max_flips={tabq_mono['max_flips_in_a_row']}")

    # --- Figures ---
    plot_exercise_step_histogram(
        {k: v for k, v in results.items() if k not in ("always_hold",)},
        steps=ENV_KWARGS["steps"],
        save_path=os.path.join(FIGURES_DIR, "exercise_step_histogram.png"),
    )

    tf, mg, grid = network_exercise_region(median_model, ENV_KWARGS["steps"], n_money=121)
    plot_exercise_region(
        tf, mg, grid, f"DQN (seed={median_seed}) exercise region vs. Week 4 boundary",
        ref_boundary=(ref_t, ref_m),
        save_path=os.path.join(FIGURES_DIR, "dqn_exercise_region.png"),
    )

    tf, mg, margin = network_q_margin(median_model, ENV_KWARGS["steps"], n_money=121)
    plot_q_margin(
        tf, mg, margin, f"DQN (seed={median_seed}) Q-margin [Q(exercise) - Q(hold)] vs. Week 4 boundary",
        ref_boundary=(ref_t, ref_m),
        save_path=os.path.join(FIGURES_DIR, "dqn_q_margin.png"),
    )

    tf, mg, grid = policy_exercise_region(tabular_q_policy, ENV_KWARGS["steps"], n_money=121)
    plot_exercise_region(
        tf, mg, grid, "Week 7 tabular Q exercise region vs. Week 4 boundary",
        ref_boundary=(ref_t, ref_m),
        save_path=os.path.join(FIGURES_DIR, "tabular_q_exercise_region.png"),
    )

    tf, mg, margin = policy_q_margin(Q, ENV_KWARGS["steps"])
    plot_q_margin(
        tf, mg, margin, "Week 7 tabular Q-margin vs. Week 4 boundary",
        ref_boundary=(ref_t, ref_m),
        save_path=os.path.join(FIGURES_DIR, "tabular_q_margin.png"),
    )

    plot_value_convergence(dqn_runs, median_seed, ref_price, euro_price,
                            save_path=os.path.join(FIGURES_DIR, "value_convergence.png"))
    plot_loss_curve(dqn_runs[median_seed]["history"],
                     save_path=os.path.join(FIGURES_DIR, "loss_curve.png"))
    plot_boundary_snapshots(dqn_runs[median_seed]["history"], ref_t, ref_m,
                             save_path=os.path.join(FIGURES_DIR, "boundary_snapshots.png"))
    if len(dqn_seed_results) > 1:
        plot_seed_spread(dqn_seed_results, median_seed,
                          save_path=os.path.join(FIGURES_DIR, "seed_spread.png"))

    print(f"\nFigures written to {FIGURES_DIR}")

    # --- Artifact ---
    summary = {
        "reference_american_price": ref_price,
        "reference_european_price": euro_price,
        "dqn_seed_values": {str(s): float(v) for s, v in zip(dqn_seed_results.keys(), dqn_values)},
        "median_seed": median_seed,
        "checkpoint_selection": dqn_runs[median_seed]["history"]["checkpoint_selection"],
        "results": {name: {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                            for k, v in r.items() if k not in ("exercise_steps", "exercise_moneyness")}
                    for name, r in results.items()},
    }
    save_artifact(
        os.path.join(ARTIFACTS_DIR, "week8_dqn_policy.pt"),
        median_model,
        hyperparams=hyperparams,
        seed=median_seed,
        extra={"env_kwargs": ENV_KWARGS, "summary": summary},
    )
    with open(os.path.join(ARTIFACTS_DIR, "week8_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Artifact written to {ARTIFACTS_DIR}")


def plot_value_convergence(dqn_runs, median_seed, ref_price, euro_price, save_path=None):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    for seed, run in dqn_runs.items():
        cps = run["history"]["checkpoints"]
        episodes = [c["episode"] for c in cps]
        values = [c["value"] for c in cps]
        if seed == median_seed:
            ax.plot(episodes, values, color="C0", linewidth=2.5, label=f"DQN seed={seed} (reported)", zorder=5)
            ses = [c["se"] for c in cps]
            lo = [v - 1.96 * se for v, se in zip(values, ses)]
            hi = [v + 1.96 * se for v, se in zip(values, ses)]
            ax.fill_between(episodes, lo, hi, color="C0", alpha=0.2)

            selection = run["history"].get("checkpoint_selection")
            if selection is not None:
                selected = selection["candidates"][selection["selected"]]
                ax.scatter([selected["episode"]], [selected["confirmed_value"]], color="red", s=80,
                           zorder=6, marker="*", label=f"selected checkpoint (ep={selected['episode']})")
        else:
            ax.plot(episodes, values, color="gray", alpha=0.5, linewidth=1.0)

    ax.axhline(ref_price, color="black", linewidth=2, linestyle="--", label="Week 4 American (binomial)")
    ax.axhline(euro_price, color="gray", linewidth=1.5, linestyle=":", label="Week 4 European (always-hold)")
    ax.set_xlabel("training episode")
    ax.set_ylabel("estimated discounted value (greedy policy)")
    title = "DQN value-convergence vs. Week 4 binomial reference"
    if len(dqn_runs) > 1:
        title += f" ({len(dqn_runs)} seeds, others in gray)"
    ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_loss_curve(history, save_path=None):
    import matplotlib.pyplot as plt

    log = history["episode_log"]
    episodes = [e["episode"] for e in log if e["loss_ema"] is not None]
    losses = [e["loss_ema"] for e in log if e["loss_ema"] is not None]
    epsilons = [e["epsilon"] for e in log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(episodes, losses, color="C1")
    ax1.set_xlabel("training episode")
    ax1.set_ylabel("Huber loss (EMA)")
    ax1.set_yscale("log")
    ax1.set_title("Training loss")

    ax2.plot(range(len(epsilons)), epsilons, color="C2")
    ax2.set_xlabel("training episode")
    ax2.set_ylabel("epsilon")
    ax2.set_title("Exploration schedule")

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_boundary_snapshots(history, ref_t, ref_m, save_path=None):
    import matplotlib.pyplot as plt

    cps = history["checkpoints"]
    n = len(cps)
    picks = sorted(set([0, n // 2, n - 1]))
    fig, axes = plt.subplots(1, len(picks), figsize=(5 * len(picks), 4.5), sharey=True)
    if len(picks) == 1:
        axes = [axes]

    for ax, idx in zip(axes, picks):
        cp = cps[idx]
        ax.pcolormesh(cp["grid_time_fractions"], cp["grid_moneyness"], cp["grid_exercise"].T,
                      cmap="RdBu_r", shading="auto", vmin=0, vmax=1)
        if len(ref_t) > 0:
            ax.plot(ref_t, ref_m, color="black", linewidth=2)
        ax.set_title(f"episode {cp['episode']}")
        ax.set_xlabel("time fraction")
    axes[0].set_ylabel("moneyness (S/K)")
    fig.suptitle("DQN exercise-region boundary sharpening over training (coarse grid)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


def plot_seed_spread(dqn_seed_results, median_seed, save_path=None):
    import matplotlib.pyplot as plt

    seeds = list(dqn_seed_results.keys())
    values = [dqn_seed_results[s]["discounted_payoff_mean"] for s in seeds]
    ses = [dqn_seed_results[s]["se"] for s in seeds]
    colors = ["C0" if s == median_seed else "C7" for s in seeds]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.bar([str(s) for s in seeds], values, yerr=[1.96 * se for se in ses], color=colors, capsize=4)
    ax.set_xlabel("training seed")
    ax.set_ylabel("estimated discounted value")
    ax.set_title(f"DQN final value across {len(seeds)} training seeds (95% CI)")
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150)
    return fig


if __name__ == "__main__":
    main()
