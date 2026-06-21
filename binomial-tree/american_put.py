"""Cox-Ross-Rubinstein (CRR) binomial tree option pricer for European and American puts."""

import math
import numpy as np


def crr_put_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    steps: int,
    american: bool = True,
) -> float:
    """Price a put option with the Cox-Ross-Rubinstein binomial tree.

    Units:
      S0, K  : same currency units
      T      : years, e.g. 90 trading days -> 90 / 252
      r      : continuously compounded annual rate, e.g. 0.06
      sigma  : annual volatility as a decimal, e.g. 0.25
      steps  : positive integer tree depth
      american : if True, prices American put with early exercise;
                 if False, prices European put.

    Returns:
      float : option price
    """
    # Input validation
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive")
    if T <= 0:
        return max(K - S0, 0.0)
    if sigma <= 0:
        raise ValueError("sigma must be positive for the CRR model")
    if int(steps) != steps or steps < 1:
        raise ValueError("steps must be a positive integer")

    steps = int(steps)
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp(r * dt)
    p = (growth - d) / (u - d)
    disc = math.exp(-r * dt)

    if not (0.0 < p < 1.0):
        raise ValueError("Invalid risk-neutral probability; check inputs")

    # Terminal layer: j up moves and steps-j down moves.
    j = np.arange(steps + 1)
    stock = S0 * (u ** j) * (d ** (steps - j))
    value = np.maximum(K - stock, 0.0)

    # Roll the option values back to today via backward induction.
    for i in range(steps - 1, -1, -1):
        value = disc * (p * value[1 : i + 2] + (1.0 - p) * value[0 : i + 1])

        if american:
            j = np.arange(i + 1)
            stock = S0 * (u ** j) * (d ** (i - j))
            exercise = np.maximum(K - stock, 0.0)
            value = np.maximum(value, exercise)

    return float(value[0])


def crr_put_with_boundary(
    S0: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    steps: int,
) -> tuple:
    """Price an American put and return the early-exercise boundary.

    Returns:
      (price, boundary) where boundary is a list of (time, stock_price) tuples
      representing the highest stock price at each time step where exercise
      is optimal. Below this boundary, immediate exercise dominates.
    """
    # Input validation (same as crr_put_price)
    if S0 <= 0 or K <= 0:
        raise ValueError("S0 and K must be positive")
    if T <= 0:
        return max(K - S0, 0.0), []
    if sigma <= 0:
        raise ValueError("sigma must be positive for the CRR model")
    if int(steps) != steps or steps < 1:
        raise ValueError("steps must be a positive integer")

    steps = int(steps)
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    growth = math.exp(r * dt)
    p = (growth - d) / (u - d)
    disc = math.exp(-r * dt)

    if not (0.0 < p < 1.0):
        raise ValueError("Invalid risk-neutral probability; check inputs")

    # Terminal layer
    j = np.arange(steps + 1)
    stock = S0 * (u ** j) * (d ** (steps - j))
    value = np.maximum(K - stock, 0.0)
    boundary = []

    # Backward induction with boundary tracking
    for i in range(steps - 1, -1, -1):
        continuation = disc * (p * value[1 : i + 2] + (1.0 - p) * value[0 : i + 1])
        j = np.arange(i + 1)
        stock = S0 * (u ** j) * (d ** (i - j))
        exercise = np.maximum(K - stock, 0.0)
        exercise_now = exercise > continuation + 1e-10

        if np.any(exercise_now):
            boundary_stock = float(np.max(stock[exercise_now]))
            boundary.append((i * dt, boundary_stock))

        value = np.maximum(continuation, exercise)

    boundary.reverse()
    return float(value[0]), boundary


def convergence_table(S0=100, K=100, T=1.0, r=0.05, sigma=0.25):
    """Generate a convergence table showing prices at increasing step counts."""
    rows = []
    for steps in [25, 50, 100, 200, 500, 1000]:
        price = crr_put_price(S0, K, T, r, sigma, steps, american=True)
        rows.append((steps, price))
    return rows


def price_grid(K=100, r=0.05, sigma=0.25, steps=300):
    """Generate a 2D grid of American put prices over spot and maturity.

    Returns:
      (spots, maturities, prices) where prices[i,j] is the price for
      maturity maturities[i] and spot spots[j].
    """
    spots = np.linspace(60, 140, 41)
    maturities = np.linspace(0.05, 2.0, 40)
    prices = np.zeros((len(maturities), len(spots)))

    for i, T in enumerate(maturities):
        for j, S0 in enumerate(spots):
            prices[i, j] = crr_put_price(S0, K, T, r, sigma, steps, american=True)

    return spots, maturities, prices
