# Week 7: American Put as an RL Exercise/Hold Problem

Formulates American put exercise as a Markov Decision Process, implements a Gym-style
environment, tests it against the pitfalls the instructional material calls out
(leakage, double-counted reward, missing terminal handling), and compares baseline and
tabular-Q-learning exercise policies against the Week 4 binomial boundary
(`../binomial-tree/american_put.py`).

**This week's deliverable is the formulation, not a fully trained agent.** The tabular
Q-learning prototype is a toy used to verify the environment and learning loop end to
end; Week 8 trains a stronger policy.

## Part A — MDP Definition

| Component | Definition |
|---|---|
| **State** `s` | `[time_fraction, moneyness] = [step / steps, S_t / K]` |
| **Actions** `a` | `0 = hold`, `1 = exercise` |
| **Transition** `P` | Hold: one risk-neutral CRR binomial step, `S ← S·u` w.p. `p`, else `S ← S·d`. Exercise: episode ends. |
| **Reward** `R` | `0.0` on every hold step; `max(K − S_t, 0)` exactly once, at exercise or forced expiry-exercise. Never paid twice. |
| **Discount** `γ` | `exp(−r·Δt)` per step, matching the finance present-value convention. |
| **Terminal condition** | Exercise, or `step == steps` (option is automatically exercised if in the money). |

**Why the state doesn't leak future information:** `[time_fraction, moneyness]` is a
function only of the current step and current spot — both already realized when the
agent decides. The transition model (`environment.py`) draws the *next* spot only
inside `step()` after the action is chosen, so no future draw is available to the
policy at decision time. The risk-neutral probability `p` is used for the transition
(not physical drift), because we want simulated returns to be pricing-consistent with
the Week 4 tree, not a forecast of real-world price behavior.

## Part B — Environment

`environment.py` implements `AmericanPutEnv` with `reset()` / `step(action)`:

- `done` flag blocks any `step()` call after exercise or expiry (`RuntimeError`).
- Reward is `0.0` on hold, non-negative and paid once on exercise/expiry.
- Invalid actions raise `ValueError`.

Five sample episodes under a random hold/exercise policy (`S0=K=100, T=1, r=5%, σ=25%,
steps=50`, seed=11), plus one always-hold episode shown for contrast:

```
episode 0: reason=exercise, step=2,  reward=0.0000, final moneyness=1.0733
episode 1: reason=exercise, step=1,  reward=3.4738, final moneyness=0.9653
episode 2: reason=exercise, step=0,  reward=0.0000, final moneyness=1.0000
episode 3: reason=exercise, step=0,  reward=0.0000, final moneyness=1.0000
episode 4: reason=exercise, step=4,  reward=0.0000, final moneyness=1.0733
(always-hold, for contrast): reason=expiry, step=50, reward=0.0000, final moneyness=1.1519
```

A pure random policy exercises almost immediately in expectation (mean ~2 steps for a
fair coin), so it rarely survives long enough to demonstrate expiry termination —
hence the always-hold contrast line, which confirms expiry-side termination works too.

`test_environment.py` has 19 tests covering both required invariants (non-negative
payoff, cannot-step-after-done) plus additional coverage: no-leakage state shape,
zero reward on every hold step, correct expiry payoff/termination, invalid-action
rejection, random-policy always terminates within `steps+1` actions, discounted value
never exceeds raw payoff, non-uniform state-discretization bin behavior, exploring-start
`reset()` (valid and invalid arguments), the warmup-then-refine schedule (fraction=0.0
matches exploring_starts=False; a moderate episode budget achieves full table
coverage), and Q-learning smoke tests (finite table, valid greedy action, exploring
starts reach far-from-the-money states).

```bash
pytest test_environment.py -v   # 19 passed
```

## Part C — Policy Comparison

Baseline: `S0=K=100, T=1, r=5%, σ=25%`, 50 binomial steps, evaluated over 2000 episodes
(seed=2024). Reference is the Week 4 CRR binomial price on the same 50-step grid. The
Q-learning prototype trains with **non-uniform moneyness bins** (fine across
[0.7, 1.0], coarse outside) and a **warmup-then-refine exploring-starts schedule**
(20,000 episodes: the first 50% start from a state drawn uniformly over the whole
`(time, moneyness)` grid, the remaining 50% start at contract inception) — see below
for why it's split this way rather than pure exploring starts.

| Policy | Avg. raw payoff | Avg. discounted payoff | Exercise rate |
|---|---:|---:|---:|
| Always hold | 7.6946 | 7.3194 | 0.00% |
| Immediate exercise | 0.0000 | 0.0000 | 100.00% |
| Random | 0.9703 | 0.9682 | 100.00% |
| Tabular Q-learned (20,000 episodes, 50% warmup) | 4.2034 | 4.1720 | 83.35% |
| **Week 4 binomial (reference)** | — | **American: 7.9520**, European: 7.4096 | — |

`run_policy` in `evaluate.py` reports *both* raw and discount-adjusted mean payoff.
This matters: raw payoff overstates value because it ignores the time cost of money —
e.g. always-hold's raw mean (7.69) sits above its discounted mean (7.32), and the
discounted figure is the one directly comparable to a binomial/Black-Scholes price
(it lands close to the European reference of 7.41, as it should, since always-hold
is exactly the European exercise rule).

### Exploring starts: what they fixed, what they didn't, and the warmup-then-refine fix

The first version of this prototype always started training episodes at contract
inception (`t=0, S=S0`). Checked directly: after 5000 episodes, **400 of the 600**
`(time, moneyness)` cells had never been visited by either action, because a 50-step
random walk from `S0=K=100` rarely wanders into extreme moneyness within one episode.
Those cells stayed at their zero initialization, so the greedy policy silently
defaulted to "hold" there regardless of whether that was sensible.

Pure exploring starts (`exploring_starts=True`, resetting every episode to a state
drawn uniformly over `(time, moneyness)` via `env.reset(step_count=..., spot=...)`)
fixed coverage completely — **0 of 600** cells unvisited — but did *not* improve
average discounted payoff at a fixed 5000-episode budget (3.33 vs. 3.65 for the
original t=0-only version), and increasing to 20,000 episodes made **no difference at
all** (3.33, byte-identical). The reason: evaluation is greedy and deterministic given
`Q`, walks one fixed random path, and only depends on the `argmax` at cells that path
visits — those settled early, so additional training just kept moving Q magnitudes in
cells outside that path. Meanwhile the *non*-exploring version, given the same 20,000
episodes, reached 4.52 by concentrating its entire budget on the region that matters —
but still left 375/600 cells unvisited even at 4x the episodes (400/600 at 5,000),
confirming t=0-only starts can't reach full coverage just by training longer. Exploring starts buys
*coverage*; it doesn't buy *sample efficiency* where you need it, for free.

**Fix:** `train_q_learning`'s `exploring_start_fraction` (default `0.5`) splits the
schedule instead of trading one property for the other. The first half of episodes use
exploring starts (covers the whole table); the remaining half reset normally at `t=0`
(concentrates refinement on the region the contract's actual rollout — and this
evaluation — lives in). At 20,000 episodes with a 50/50 split: **0 of 600 cells
unvisited** (full coverage retained) *and* discounted payoff of **4.17** — recovering
most of the pure-refinement ceiling (4.52) while keeping the coverage guarantee
non-exploring-only can never provide at any episode budget. Verified further: unlike
pure exploring starts, this schedule keeps improving with more episodes and does not
saturate — 50,000 episodes (25,000 warmup / 25,000 refine) reaches 5.52 discounted
payoff, still with full coverage. A function approximator (Week 8) is still the more
principled long-run answer, since it generalizes from a visited state to a nearby
unvisited one instead of requiring every cell to be independently sampled at all — but
warmup-then-refine shows that even within the tabular prototype, the coverage/precision
trade-off isn't fixed once you're willing to spend training episodes non-uniformly
across the *episode* dimension too, not just the *state* dimension.

### Exercise-region plot vs. Q-margin plot

`figures/` has three exercise-region-style plots, and comparing the first two makes a
point about diagnostics, not just about the policy:

- `random_policy_exercise_region.png` — pure speckle, no structure. This is the
  "should not exercise randomly" debugging baseline from the instructional material,
  confirmed visually.
- `q_policy_exercise_region.png` — the learned policy's *binary* `argmax` decision.
  With exploring starts, this plot looks **noisier** than the very first prototype did,
  not cleaner — a coarse cluster near the true boundary, but heavily speckled well
  below it too.
- `q_policy_margin.png` — the same policy's `Q(exercise) − Q(hold)` **margin**, plotted
  continuously instead of thresholded to 0/1. This tells the real story: the margin is
  strongly positive (favors exercise) at low moneyness and grows monotonically as
  moneyness drops further, is small/near-zero in a band roughly straddling the true
  Week 4 boundary, and turns negative (favors hold) well above the money, especially
  near expiry (bottom-right of the plot). That is qualitatively the correct shape.

The apparent contradiction resolves once you see both: the binary plot speckles
because many cells near the true boundary have a *small* margin, so tiny numerical
noise flips their `argmax` sign independently of their neighbors — the underlying value
function is smoother and more sensible than a thresholded view can show. This is
exactly why a margin plot, not just a region plot, is worth having: a 0/1 decision can
look far noisier than the value function actually is.

**Reflection:** among the four, **always-hold** is still closest to true American put
value by average payoff (within ~7% of the exact binomial American price, discounted
mean matching the European price almost exactly — correct for a policy that never
exercises early). The **tabular Q-learned** policy is the only one that exercises
*selectively* based on moneyness and time, and with the warmup-then-refine schedule
its margin function has the right qualitative shape across the *entire* state space
(full coverage) while being noticeably closer to always-hold's payoff (4.17 vs. 7.32,
versus 3.33 before the schedule change) — and, unlike pure exploring starts, it
continues to close that gap with more training (5.52 at 50,000 episodes) rather than
stalling. It still hasn't converged precisely enough to beat always-hold outright at a
reasonable episode budget, and its region plot is still visibly speckled near the
boundary even though its margin plot shows the right underlying shape. That gap —
right shape and full coverage, imprecise value, still improving with more (but not
unlimited) budget — is the natural motivation for Week 8: replace the table with a
function approximator that gets both full coverage and sample efficiency at once,
instead of needing a hand-tuned episode-scheduling trick to approximate that trade-off
within a table.

## Files

- `environment.py` — `AmericanPutEnv`: reset/step (supports exploring starts via
  optional `step_count`/`spot` args), risk-neutral CRR transition, reward timing,
  discount factor.
- `policies.py` — `always_hold_policy`, `immediate_exercise_policy`,
  `make_random_policy`, `discretize_state` (non-uniform moneyness bins, fine across
  [0.7, 1.0]), `make_q_policy`.
- `q_learning.py` — `train_q_learning`: epsilon-greedy tabular Q-learning with a
  warmup-then-refine exploring-starts schedule by default (`exploring_start_fraction`,
  default 0.5; own RNG, decoupled from the environment's price-path RNG).
- `evaluate.py` — `run_policy` (Monte Carlo: raw + discounted payoff, exercise rate,
  timing), `policy_exercise_region` (deterministic 0/1 grid query), `policy_q_margin`
  (deterministic `Q(exercise) - Q(hold)` grid query), `boundary_to_moneyness`,
  plotting helpers.
- `pipeline.py` — end-to-end CLI: reference binomial price/boundary, sample episodes,
  Q-learning training (20,000 episodes), policy comparison table, figure generation.
- `test_environment.py` — 19 pytest invariant tests.
- `week7_report.ipynb` — narrative notebook covering Parts A/B/C.
- `figures/` — exercise-step histogram, Q-policy/random-policy exercise-region plots,
  and the Q-margin plot (each overlaid with the Week 4 binomial boundary where
  applicable).

## Run it

```bash
pip install numpy matplotlib pytest
pytest test_environment.py -v
python pipeline.py
jupyter notebook week7_report.ipynb
```

All randomness is seeded (`pipeline.py`: env seed 2024 for evaluation, Q-learning
train seed 42/123, sample-episode seed 11) and documented inline.

## References

Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). "Option pricing: A simplified
approach." *Journal of Financial Economics*, 7(3), 229–263.
