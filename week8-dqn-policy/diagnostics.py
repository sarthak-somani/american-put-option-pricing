"""Quantitative sanity checks for a learned exercise policy.

Turns two qualitative "does this look right" questions from Week 7's report
into numbers:

- Is the exercise region a clean stopping boundary (put value is
  non-increasing in spot, so at fixed time the greedy decision should flip
  from exercise to hold at most once as moneyness rises), or is it speckled?
- Does the policy's estimated value respect the one hard bound implied by
  arbitrage: no admissible stopping rule can beat the American price, since
  that price is defined as the supremum over all stopping times. There is
  no equivalent hard *lower* bound -- a legitimate (just badly-chosen)
  stopping rule can fall well below the European (always-hold) price. For
  example, immediate exercise at S0=K=100 pays exactly 0, far below the
  European price, with nothing wrong with the simulation. So "below
  always-hold" is only a policy-quality signal, not evidence of a bug,
  and is reported separately from the one real red flag.
"""

import numpy as np


def boundary_monotonicity_report(exercise_grid):
    """Count decision flips per time-row of a 0/1 exercise grid.

    `exercise_grid[i, j]` is assumed ordered by ascending moneyness along
    axis 1. A clean put stopping boundary flips 1 -> 0 at most once per row
    (exercise at low moneyness, hold at high moneyness); any additional
    flips are noise ("speckle") rather than a genuine second boundary.
    """
    flips_per_row = (np.diff(exercise_grid, axis=1) != 0).sum(axis=1)
    violation_rows = int((flips_per_row > 1).sum())
    total_rows = exercise_grid.shape[0]
    return {
        "flips_per_row": flips_per_row,
        "violation_rows": violation_rows,
        "total_rows": total_rows,
        "monotonicity_score": float(1.0 - violation_rows / total_rows) if total_rows else float("nan"),
        "max_flips_in_a_row": int(flips_per_row.max()) if len(flips_per_row) else 0,
    }


def theoretical_bounds_check(rl_value, rl_se, european_price, american_price, z=1.96):
    """Check a policy's estimated value against the one hard bound (value <=
    American price, within a z * SE allowance for Monte Carlo noise) and
    report the always-hold comparison separately as a quality note only.

    `exceeds_binomial=True` is the real red flag: investigate for reward
    leakage, double-counted payoff, or a broken discount before treating a
    high number as good news. `below_always_hold=True` is not a bug signal
    by itself -- it just means this policy underperforms the trivial
    never-exercise-early baseline, which any sufficiently bad (but valid)
    stopping rule can do.
    """
    upper_bound = american_price + z * rl_se
    return {
        "rl_value": rl_value,
        "rl_se": rl_se,
        "upper_bound": upper_bound,
        "exceeds_binomial": bool(rl_value > upper_bound),
        "always_hold_price": european_price,
        "below_always_hold": bool(rl_value < european_price - z * rl_se),
    }
