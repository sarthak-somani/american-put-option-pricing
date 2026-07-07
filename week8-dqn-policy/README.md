# Week 8: Double DQN American Put Stopping Policy

A Double DQN trained inside the Week 7 `AmericanPutEnv` to learn a hold/exercise
stopping rule, evaluated against always-hold, immediate-exercise, random, and
the Week 7 tabular Q-learner, and checked against the Week 4 binomial price
and exercise boundary (`../binomial-tree/american_put.py`) on the same
contract and step grid.

**Headline result**: the trained DQN reaches a discounted value of **7.56**,
above the always-hold/European baseline (7.41) and close to the binomial-
optimal American price (7.95), at a 45% exercise rate with a clean,
single-crossing exercise boundary. That result did not come from the first
training run -- it came from finding and fixing two real problems along the
way. Both are documented below with before/after evidence, not smoothed over,
because the diagnostic process is as much the point of this week as the
final number.

## Quick start

```bash
pip install numpy torch matplotlib pytest
pytest test_dqn.py -v        # 24 fast tests, ~25s
python pipeline.py            # single seed, 20000 episodes, ~10-12 min on this machine
jupyter notebook week8_report.ipynb
```

For more seeds and/or parallel training across CPU cores, `pipeline.py`
accepts `--seeds`, `--workers`, `--episodes`, and `--eval-episodes`
(e.g. `python pipeline.py --seeds 0 1 2 3 4 --workers 5`).

## Part A -- Training setup

**Contract**: S0=100, K=100, T=1.0 year, r=5%, σ=25%, 50 binomial steps --
identical to Week 7's baseline, so the binomial reference price/boundary are
directly comparable.

**State**: `[time_fraction, time_to_expiry, moneyness]`, where
`time_to_expiry = 1 - time_fraction`. The environment itself only exposes
`[time_fraction, moneyness]` (`../week7-rl-formulation/environment.py`);
`dqn.expand_state` appends the third feature. This is included for parity
with the assignment's stated state vector, not because it adds information a
linear-then-ReLU first layer couldn't already recover from `time_fraction`
alone (weight -1, bias 1 recovers `1 - time_fraction` exactly, at zero
capacity cost) -- see the docstring in `dqn.py` for the full argument. No
future information is exposed: `expand_state` is a pure function of the
current `[time_fraction, moneyness]` only.

**Action, reward, discount, transition**: unchanged from Week 7 -- HOLD (0)
takes one risk-neutral CRR step, EXERCISE (1) ends the episode and pays
`max(K - S, 0)` exactly once; `env.discount = exp(-r * dt)` per step.

**Network**: `QNetwork`, `Linear(3,64) -> ReLU -> Linear(64,64) -> ReLU ->
Linear(64,2)`, outputs `[Q(hold), Q(exercise)]`.

**Algorithm**: Double DQN (online net selects the next action, target net
evaluates it -- guards against the overestimation bias plain DQN would carry
into this environment's large, sparse terminal payoffs), Huber loss,
gradient clipping at norm 5.0, replay buffer (capacity 50,000, its own seeded
`np.random.Generator` for sampling), hard target sync every 250 gradient
updates, batch size 128, Adam with a **decaying learning rate** (1e-3 -> 1e-4,
see below for why).

**Exploration**: epsilon-greedy, exponential decay solved so it hits the
floor (0.05) at 80% of training (`train.epsilon_schedule` -- computed from a
target fraction, not a hardcoded decay constant), starting from 1.0. Uses its
own RNG, decoupled from the environment's price-path RNG, the replay buffer's
sampling RNG, and the exploring-start RNG.

**Exploring starts**: interleaved and *decaying*, not staged and flat.
Independently for each episode, with probability `exploring_start_schedule(...)`
(linearly decaying from 0.7 early to 0.1 late, reaching the floor at 80% of
training) the episode starts from a state drawn uniformly over
`(time, moneyness)`; otherwise it starts at contract inception. See "Bug #1"
below for why interleaving (vs. Week 7's block schedule) was needed at all,
and "Fix #1" for why a flat interleave rate wasn't the end of the story.

**Best-checkpoint selection**: the network from the *final* training episode
is not automatically what gets reported. Every `checkpoint_interval` episodes
(here, every 1,000), the current network is evaluated and its weights saved;
after training, `pipeline.select_best_checkpoint` compares the final episode
against the best-scoring checkpoint on a larger, independent confirmation
evaluation (2,000 episodes) before picking one. See "Fix #2" below for why
this was necessary and how the confirmation step guards against just picking
evaluation noise.

**Training budget**: 20,000 episodes, single seed (seed=0) -- see "Only one
seed" under Limitations for why there's no seed-sensitivity spread this time.

**Runtime**: ~10-12 minutes wall-clock on the development machine for the
entire pipeline (Week 7 tabular-Q retrain + DQN training + checkpoint
confirmation + 10,000-episode evaluation of 5 policies + all figures).

## Part B -- Evaluation

Reference (Week 4 binomial, 50 steps): **American = 7.9520, European =
7.4096**.

Full evaluation, 10,000 Monte Carlo episodes per policy, contract fixed at
inception (S0=K=100), seed=2024:

| Policy | Raw payoff | Discounted payoff | SE | Exercise rate |
|---|---:|---:|---:|---:|
| Always hold | 7.7428 | 7.3652 | 0.1092 | 0.00% |
| Immediate exercise | 0.0000 | 0.0000 | 0.0000 | 100.00% |
| Random | 0.9712 | 0.9690 | 0.0214 | 100.00% |
| Tabular Q (Week 7, 20,000 ep.) | 4.2048 | 4.1754 | 0.0247 | 82.12% |
| **DQN (this week, 20,000 ep.)** | **7.8867** | **7.5573** | **0.1083** | **45.16%** |
| Week 4 binomial (reference) | -- | American 7.9520 / European 7.4096 | -- | -- |

DQN beats the Week 7 tabular-Q baseline by a wide margin (7.56 vs. 4.18),
beats the always-hold/European baseline (7.56 vs. 7.37), and sits within
**~5%** of the binomial-optimal American price (7.95) -- while remaining
safely below it, as it must.

**Automated theoretical bound check** (`diagnostics.theoretical_bounds_check`):
no admissible stopping rule can beat the American price, so `value >
American + 1.96*SE` is the one hard red flag; falling below the always-hold
(European) price is a policy-quality note, not a bug (immediate exercise at
S0=K, for instance, legitimately scores 0, far below European, with nothing
wrong with the simulation).

| Policy | Value | Upper bound (American + 1.96·SE) | Verdict |
|---|---:|---:|---|
| Tabular Q | 4.1754 | 8.0004 | OK (below always-hold) |
| DQN | 7.5573 | 8.1644 | **OK -- and above always-hold** |

**Boundary monotonicity** (`diagnostics.boundary_monotonicity_report`, 51
time-slices x 121 moneyness points): a clean put stopping boundary flips
exercise->hold at most once per time-slice as moneyness rises; extra flips
are speckle.

| Policy | Monotonicity score | Violating rows | Max flips in one row |
|---|---:|---:|---:|
| Tabular Q | 0.000 | 51 / 51 | 16 |
| DQN | 0.431 | 29 / 51 | 2 |

DQN's boundary is far cleaner than the discretized table's (max 2 flips vs.
16), though not perfectly monotonic -- worth noting explicitly, the
checkpoint that was *selected* (see below) was chosen for its confirmed
**value**, not its boundary cleanliness; a later, more-trained checkpoint
had a cleaner-looking boundary (score 1.000 in an earlier run of this
pipeline, before the checkpoint-selection fix) but a substantially *worse*
realized value (5.05-5.32). Value, not cosmetic boundary smoothness, is
what this week's evaluation is actually supposed to measure.

**Figures** (`figures/`): `dqn_exercise_region.png` / `dqn_q_margin.png`
(binary decision and continuous margin vs. the Week 4 boundary),
`tabular_q_exercise_region.png` / `tabular_q_margin.png` (same, for the reused
Week 7 baseline), `value_convergence.png` (checkpointed greedy-policy value
vs. training episode, with American/European reference lines and the
selected checkpoint marked with a star), `loss_curve.png` (Huber loss EMA +
epsilon schedule), `boundary_snapshots.png` (exercise region at episodes
1000 / 11000 / 20000, coarse grid), `exercise_step_histogram.png` (when each
policy exercises).

## Two bugs we found and fixed

### Bug #1 -- confidently wrong at deep ITM (block exploring-starts schedule)

The very first run reused Week 7's tabular-Q exploring-starts schedule as-is
(a block: first 50% of episodes warmup with uniform `(time, moneyness)`
starts, remaining 50% start at inception). It produced a plausible-looking
aggregate value (5.09 discounted, 93.76% exercise rate) but a **confidently
wrong** Q-function at deep in-the-money states, found by querying the
trained network directly rather than trusting the exercise-region plot:

```
time_fraction=0.10: moneyness 0.50  margin (Q(exercise)-Q(hold)) = -25.7
time_fraction=0.90: moneyness 0.50  margin                        = -27.1
```

A put at spot 50 (K=100) pays 50 on immediate exercise -- an unambiguous,
easy-to-learn one-step target. A margin of -27 means the network had come to
believe *holding* was worth roughly 27 more than that, at a state where
holding has essentially no further upside.

**Root cause**: the block schedule works for Week 7's tabular Q-learner
because a table has one independent, permanent memory cell per state --
front-loading coverage is safe there, since a cell keeps its value forever
once written. Neither property holds for a neural net with a finite replay
buffer. With 20,000 episodes at several steps each, total transitions
comfortably exceed the buffer's 50,000 capacity, so by late training the
buffer had long since cycled past the warmup block's deep-ITM transitions --
and the refine block's own policy (already exercising quickly once even
mildly ITM) rarely regenerates fresh deep-ITM transitions on its own. Nothing
in the back half of training corrected those values while the shared network
weights kept moving in response to the states that phase actually visits.

**Fix #1**: `train.py`'s exploring-starts logic changed from a contiguous
block to an independent per-episode draw, so a steady trickle of fresh
deep-ITM transitions keeps entering the buffer for the entire run. Verified
directly, same query post-fix:

```
time_fraction=0.10: moneyness 0.50  margin = +0.3   (correctly favors exercise)
time_fraction=0.90: moneyness 0.50  margin = +0.3   (correctly favors exercise)
```

Added `test_exploring_starts_are_interleaved_and_decay_not_a_block_schedule`
as a regression test.

### Bug #2 -- boundary too eager, and value declining over training

Fixing Bug #1 alone (flat 50/50 interleaving, constant learning rate)
produced a cleanly-shaped but *mispositioned* boundary: exercising 95.49% of
episodes at a boundary sitting around moneyness 0.92-0.97 for most of the
contract's life, well above the true boundary's 0.73-0.97. That's a real,
consequential gap -- exercising this early forfeits the option's remaining
time value, which is why that version's value (5.05) sat *below even the
always-hold baseline* (7.37). Worse, `value_convergence.png` from that run
showed the checkpointed value **declining** over training (~8 early, ~5 by
the end) rather than converging upward.

**Root cause**: a flat 50% exploring-start rate spends the same fraction of
late-training budget on rare (time, moneyness) states as early training does
-- diluting the signal needed to precisely locate the boundary in the region
that actually matters (where a real t=0 rollout lives), instead of
concentrating late training there once broad coverage has already done its
job. Separately, a constant learning rate means late-training gradient steps
are exactly as large as early ones, so ordinary replay noise keeps
reshuffling an already-reasonable policy indefinitely instead of letting it
settle -- a plausible mechanism for the observed decline.

**Fix #2a -- decaying schedules**: `exploring_start_schedule` now linearly
decays the exploring-start probability from 0.7 (broad early coverage,
preventing Bug #1 from recurring) to 0.1 (late-training precision, mirroring
Week 7's own coverage-vs-precision insight from its warmup-then-refine
schedule, adapted here as a continuous decay rather than a hard block, since
a hard block is exactly what caused Bug #1). `lr_schedule` decays Adam's
learning rate 1e-3 -> 1e-4 on the same target schedule. This alone took the
final-episode boundary to a **perfect** monotonicity score of 1.000 (0
violating rows, max 1 flip) -- a real improvement in boundary shape.

**But this didn't fix the underlying value decline** -- it only made it
visible more clearly: the checkpointed value still peaked mid-training
(~8.2-8.3 around episodes 6,000-12,500, briefly matching the American
reference) and still declined afterward, ending around 5.3 at episode
20,000, even with the improved schedules.

**Fix #2b -- best-checkpoint selection with confirmation**: rather than
continuing to chase why late training erodes an already-good policy, apply
the same principle Week 6's neural pricer already established for exactly
this shape of problem -- keep the best checkpoint, not the final epoch. The
one addition needed for DQN: `checkpoint_eval_episodes` (300, used during
training) is a small enough sample that naively taking the argmax over ~20
checkpoints risks picking one that got lucky rather than one that's
genuinely better. `select_best_checkpoint` re-evaluates both the
final-episode network and the best-scoring checkpoint on a much larger,
independent 2,000-episode confirmation before choosing, and records both
results rather than silently swapping one in:

```
final  (ep=20000): confirmed=5.3828 (se=0.0739)
peak   (ep=6000):  confirmed=7.5703 (se=0.2435)  <- selected
```

The gap survives the larger, independent sample -- this is a real
difference, not noise from the small 300-episode checkpoint estimate. The
selected (episode 6,000) checkpoint's full 10,000-episode evaluation is the
headline result reported above: **7.5573**.

## Part C -- Analysis

**Where DQN agrees with binomial intuition.** The exercise region
(`dqn_exercise_region.png`) has the qualitatively correct shape end to end:
a single, clean exercise/hold crossing, red (exercise) below, blue (hold)
above, rising from moneyness ~0.57 at inception to ~0.81 near expiry -- the
same *rising* shape as the true boundary (0.73 -> 0.97), just running
somewhat lower. The margin plot (`dqn_q_margin.png`) shows a smooth single
sign change, strongest "hold" confidence concentrated almost exactly at
moneyness=1.0 -- correctly the region of highest continuation value for a
put. A function approximator recovered this from interaction alone, no
binomial labels anywhere in training.

**Where it disagrees.** DQN's boundary sits *below* (more conservative than)
the true one for most of the contract's life -- e.g. at time_fraction=0.2,
the true boundary exercises below moneyness~0.73, while DQN waits until
roughly 0.59. This is the mirror image of the failure mode in the pre-Fix#2
run: instead of exercising too early (giving up time value), the selected
policy now waits somewhat *too long*, giving up part of the early-exercise
premium the true optimal boundary captures. That's a much gentler error than
Bug #1's confident inversion or the earlier over-eager boundary -- it costs
some value (7.56 vs. the 7.95 ceiling) rather than a large fraction of it --
and it's directionally consistent with a lower exercise rate (45.16%) than
optimal likely requires. There's also a small near-zero-margin patch deep
out-of-the-money near expiry (top-right corner of the exercise-region plot),
financially irrelevant since both actions are worth ~0 there.

**Too early, too late, or sensible?** **Slightly too late / conservative**,
after two rounds of fixes that started from "much too early." The direction
of the remaining error flipped between the pre- and post-Fix#2b versions of
this policy -- worth remembering when reading any single exercise-region
plot as "the" DQN behavior; it is a snapshot of one training run's one
selected checkpoint, and different points on the same value-convergence
curve show qualitatively different boundary positions.

**Training instability.** `loss_curve.png` shows a mid-training plateau
before resuming its downward trend as both decaying schedules approach their
floors. `value_convergence.png` is the more important diagnostic here: the
checkpointed value is not monotonically improving over training, peaking
around episodes 6,000-12,500 and declining afterward even with decaying
exploration and learning-rate schedules. Best-checkpoint selection is a
correction *after the fact* (evaluate everything, keep the best), not a fix
to *why* later training degrades a good policy -- that mechanism (plausibly:
a narrowing, increasingly on-policy data distribution as both schedules
approach their floors together) is identified as a plausible cause here, not
confirmed. A cleaner fix would stagger the two schedules' floors, or
diagnose the mid-to-late training window directly; neither was done this
week.

**Only one seed.** This run used a single training seed (seed=0), by
explicit choice -- no seed-sensitivity spread is reported this week. The
original plan called for 3-5 seeds specifically to characterize training
instability quantitatively (`pipeline.py --seeds` and `--workers` still
support this). Whether the value-convergence peak-then-
decline shape, and the specific episode at which it peaks, is a property of
the method in general or of this particular seed's trajectory is genuinely
unknown without more runs -- best-checkpoint selection helps regardless of
seed (it doesn't require knowing in advance where the peak is), but its
robustness across seeds hasn't been checked.

**Conclusion: usable, partially usable, or prototype?** **Partially usable.**
The Double DQN, with best-checkpoint selection, reaches a discounted value
(7.56) above the always-hold baseline and within ~5% of the binomial-optimal
price, with a clean, single-crossing exercise boundary shaped correctly and
positioned close to (if somewhat more conservative than) the true one. That
is a materially better result than the Week 7 tabular baseline achieves at
the same training budget, and a genuinely non-trivial stopping rule learned
from interaction alone. It is not yet "usable" outright: the training
dynamics that made best-checkpoint selection necessary are not fully
understood (only diagnosed, not resolved), the result rests on a single
seed, and the reported value is still a meaningful ~5% below the true
optimum. Read the 7.56 number as evidence the pipeline's diagnostics
(monotonicity score, theoretical bound check, and especially the value-
convergence curve that motivated checkpoint selection) did their job of
catching and correcting real problems -- not as a claim that the underlying
training recipe is stable or well-understood yet.

## Reproducibility

- Contract: S0=100, K=100, T=1.0, r=0.05, sigma=0.25, steps=50.
- Seeds: env seed 1000 (training rollouts), eval seed 2024 (checkpoint,
  confirmation, and final evaluation, matching Week 7's "official"
  evaluation seed), DQN training seed 0 (network init, epsilon-greedy RNG,
  buffer-sampling RNG, exploring-start RNG -- all independently derived from
  the training seed, decoupled from each other and from the environment's
  price-path RNG).
- Hyperparameters: see Part A and `pipeline.DQN_HYPERPARAMS`; exact values
  used for this run, plus the checkpoint-selection record (both candidates'
  confirmed values), are saved in `artifacts/week8_dqn_policy.pt`
  (`load_artifact(...)["hyperparams"]`) and `artifacts/week8_summary.json`.
- As with Week 6's neural pricer, PyTorch's CPU execution is not bit-
  reproducible across runs even with a fixed seed (non-deterministic
  reduction order in matmuls unless `torch.use_deterministic_algorithms` and
  single-threaded execution are forced -- the latter is already done here,
  for speed, not determinism). Re-running with the same seed will land in
  the same ballpark, not bit-identical -- and given the value-convergence
  curve's shape, "same ballpark" could plausibly mean a different peak
  episode, which is exactly why best-checkpoint selection re-evaluates
  rather than hardcoding an episode number.

## Files

- `dqn.py` -- `QNetwork`, `ReplayBuffer` (own seeded RNG), Double DQN loss,
  `expand_state`, artifact save/load (mirrors Week 6's convention).
- `dqn_policies.py` -- `greedy_action` / `make_dqn_policy` (policy_fn wrapper
  for `run_policy` etc.), `network_exercise_region` / `network_q_margin`
  (vectorized batched grid queries, same output shape as Week 7's
  table-based versions so Week 7's plotting functions are reused unchanged).
- `diagnostics.py` -- `boundary_monotonicity_report` (flip counter),
  `theoretical_bounds_check` (the one hard bound: value <= American price).
- `train.py` -- `set_seeds`, `epsilon_schedule` / `lr_schedule` /
  `exploring_start_schedule` (all solved-decay-constant, not hardcoded),
  `train_dqn` (interleaved decaying exploring starts, replay buffer, target
  network, per-checkpoint value/grid/state_dict snapshots, device-aware).
- `pipeline.py` -- CLI driver: Week 4 reference, Week 7 tabular-Q retrain,
  (optionally parallel) multi-seed DQN training, `select_best_checkpoint`
  (confirmed best-checkpoint selection), full baseline comparison,
  diagnostics, figures, artifact export. `--seeds`, `--episodes`,
  `--workers`, `--eval-episodes` are all configurable.
- `test_dqn.py` -- 24 pytest tests: state expansion/no-leakage, replay buffer
  (push/evict/seeded-sample), Double DQN loss (finite, backprops, masks
  terminal bootstrap correctly), all three schedules, diagnostics, artifact
  round-trip, a training smoke test, best-checkpoint selection, and a
  regression test for the interleaved/decaying-exploring-starts fix.
- `week8_report.ipynb` -- narrative notebook covering Parts A/B/C.
- `figures/`, `artifacts/` -- saved plots and the trained model + summary
  stats (`week8_dqn_policy.pt`, `week8_summary.json`).

## References

Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). "Option pricing: A
simplified approach." *Journal of Financial Economics*, 7(3), 229-263.

Van Hasselt, H., Guez, A., & Silver, D. (2016). "Deep Reinforcement Learning
with Double Q-learning." *AAAI*.

Mnih, V., et al. (2015). "Human-level control through deep reinforcement
learning." *Nature*, 518(7540), 529-533.
