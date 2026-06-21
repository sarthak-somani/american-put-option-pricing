# Week 4: American Put Baseline Pricer

A Cox-Ross-Rubinstein (CRR) binomial tree implementation for pricing European and American puts. Includes convergence analysis, price surface visualization, and early-exercise boundary mapping.

## Quick Start

### Setup
```bash
pip install numpy matplotlib
```

### Run the Report
```bash
jupyter notebook week4_report.ipynb
```

This generates:
- Convergence table (prices at 25, 50, 100, 200, 500, 1000 steps)
- 3D price surface plot (spot vs. maturity)
- Early-exercise boundary plot
- Sanity test results

### Use the Pricer in Your Code
```python
from american_put import crr_put_price

# Price an American put
price = crr_put_price(
    S0=100,           # Spot price
    K=100,            # Strike price
    T=1.0,            # Time to maturity (years)
    r=0.05,           # Risk-free rate
    sigma=0.25,       # Volatility
    steps=500,        # Tree depth
    american=True     # American or European
)
print(f"${price:.6f}")
```

## Module Reference

### `crr_put_price(S0, K, T, r, sigma, steps, american=True) → float`
Prices a put option using the CRR binomial tree.

**Parameters:**
- `S0, K`: Spot and strike prices (same currency)
- `T`: Time to maturity in years (e.g., 90 days = 90/252)
- `r`: Annual continuously compounded rate (decimal, e.g., 0.05 not 5)
- `sigma`: Annual volatility (decimal, e.g., 0.25 not 25)
- `steps`: Tree depth (positive integer, 500 recommended)
- `american`: If True, prices American option (early exercise); if False, European

**Returns:** Option price as float

### `crr_put_with_boundary(S0, K, T, r, sigma, steps) → (float, list)`
Prices an American put and returns the early-exercise boundary.

**Returns:** 
- `price`: American put price
- `boundary`: List of (time, spot_price) tuples marking where exercise is optimal

### `price_grid(K, r, sigma, steps) → (array, array, array)`
Generates a 2D grid of American put prices over spot [60, 140] and maturity [0.05, 2.0].

**Returns:** `(spots, maturities, prices)` where `prices[i,j]` is the price at `maturities[i], spots[j]`

### `convergence_table(S0, K, T, r, sigma) → list`
Returns American put prices for step counts [25, 50, 100, 200, 500, 1000].

## Implementation Notes

**Units are strict:**
- `T=0.25` means 0.25 years (not 90 days—use 90/252 ≈ 0.357)
- `r=0.05` means 5% annual (not 5)
- `sigma=0.25` means 25% annual (not 25)

**CRR Parameters:**
- $\Delta t = T / \text{steps}$
- $u = e^{\sigma\sqrt{\Delta t}}$
- $d = 1/u$
- $p = \frac{e^{r\Delta t} - d}{u - d}$

**Backward Induction:**
1. Terminal payoff: $\max(K - S, 0)$
2. Roll back: $V = e^{-r\Delta t}[p \cdot V_{up} + (1-p) \cdot V_{down}]$
3. American: $V = \max(V_{continuation}, K - S)$ at each node

## Tests

Run the test suite:
```bash
pytest test_american_put.py -v
```

Tests verify:
- ✓ American ≥ European
- ✓ Value decreases as spot rises
- ✓ Value increases with volatility
- ✓ Deep OTM puts → 0
- ✓ Deep ITM American puts → K - S0
- ✓ No negative prices

## Files

- `american_put.py` — Core CRR implementation (no plotting)
- `test_american_put.py` — Sanity tests
- `week4_report.ipynb` — Report with plots and analysis
- `figures/` — Saved plots (price surface, exercise boundary)
- `README.md` — This file

## Key Results

**Baseline parameters:** S₀=100, K=100, T=1, r=5%, σ=25%, steps=500

- European put: $10.45
- American put: $10.80
- Early-exercise premium: $0.35 (3.3%)

**Convergence:** Prices stabilize by 500 steps. Change from 500→1000 steps < $0.001.

**Exercise boundary:** Below the boundary, immediate exercise is optimal. The boundary tightens toward the strike as maturity approaches, reflecting reduced time value.

## References

Cox, J. C., Ross, S. A., & Rubinstein, M. (1979). "Option pricing: A simplified approach." *Journal of Financial Economics*, 7(3), 229–263.
