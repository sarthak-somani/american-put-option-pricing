"""Sanity tests for the CRR American put pricer."""

import pytest
from american_put import crr_put_price, convergence_table


def test_american_put_not_less_than_european():
    """American put should always be worth at least as much as European."""
    args = dict(S0=100, K=105, T=1.0, r=0.05, sigma=0.25, steps=500)
    euro = crr_put_price(**args, american=False)
    amer = crr_put_price(**args, american=True)
    assert amer >= euro, f"American {amer} < European {euro}"


def test_put_value_falls_as_spot_rises():
    """Put value should decrease as spot price increases, all else fixed."""
    low_spot = crr_put_price(80, 100, 1.0, 0.05, 0.25, 500, american=True)
    high_spot = crr_put_price(120, 100, 1.0, 0.05, 0.25, 500, american=True)
    assert low_spot > high_spot, f"Low spot {low_spot} not > high spot {high_spot}"


def test_more_volatility_is_not_cheaper():
    """Put value should increase as volatility increases, all else fixed."""
    low_vol = crr_put_price(100, 100, 1.0, 0.05, 0.15, 500, american=True)
    high_vol = crr_put_price(100, 100, 1.0, 0.05, 0.35, 500, american=True)
    assert high_vol >= low_vol, f"High vol {high_vol} < low vol {low_vol}"


def test_deep_otm_put_is_cheap():
    """Far out-of-the-money put should have small value."""
    price = crr_put_price(200, 100, 1.0, 0.05, 0.25, 500, american=True)
    assert price < 0.1, f"Deep OTM put {price} should be near zero"


def test_deep_itm_american_put_near_intrinsic():
    """Deep in-the-money American put should be near K - S0."""
    S0, K = 50, 100
    price = crr_put_price(S0, K, 1.0, 0.05, 0.25, 500, american=True)
    intrinsic = K - S0
    assert price >= intrinsic - 0.01, f"Price {price} < intrinsic {intrinsic}"


def test_american_put_never_negative():
    """Option prices should never be negative."""
    for S0 in [50, 100, 150]:
        for sigma in [0.15, 0.25, 0.35]:
            price = crr_put_price(S0, 100, 1.0, 0.05, sigma, 500, american=True)
            assert price >= 0, f"Negative price {price} for S0={S0}, sigma={sigma}"


def test_convergence_is_increasing():
    """Convergence table should be populated and increasing step count."""
    table = convergence_table()
    assert len(table) == 6, f"Expected 6 rows, got {len(table)}"
    step_counts = [steps for steps, _ in table]
    assert step_counts == [25, 50, 100, 200, 500, 1000]
    prices = [price for _, price in table]
    assert all(p > 0 for p in prices), f"Some prices not positive: {prices}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
