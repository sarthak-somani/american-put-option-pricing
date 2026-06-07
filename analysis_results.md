# Black-Scholes Option Pricing Analysis
## NIFTY 50 Option Chain - June 9, 2026 Expiry

**Analysis Date:** June 05, 2026  
**Report Generated:** 2026-06-07 12:42:12

---

## Section 1: Parameters

| Parameter | Value |
|-----------|-------|
| Spot Price (S) | 23,366.70 |
| Selected Strike (K) | 23,350.00 |
| Distance from ATM | 16.70 points |
| Snapshot Date | 2026-06-05 |
| Expiry Date | 2026-06-09 |
| Time to Maturity (T) | 4 days (0.010959 years) |
| Risk-Free Rate (r) | 6.50% |

---

## Section 2: Volatility Comparison

### Historical vs. Market Implied Volatility

| Metric | Value | Percentage |
|--------|-------|-----------|
| **30-Day Historical Volatility (σ_hist)** | 0.127239 | 12.72% |
| **Market Call IV (σ_mkt)** | 0.142400 | 14.24% |
| **Difference (σ_mkt - σ_hist)** | 0.015161 | +11.92% |

**Interpretation:** The market is pricing in volatility approximately **11.92% higher** than the realized 30-day historical volatility. This suggests market expectations for elevated price movements near the expiry.

---

## Section 3: Black-Scholes Mathematical Framework

### Intermediate Calculations

The Black-Scholes model uses the following intermediate variables:

$$d_1 = \frac{\ln(S/K) + (r + \frac{\sigma^2}{2})T}{\sigma\sqrt{T}}$$

$$d_2 = d_1 - \sigma\sqrt{T}$$

**Calculated Values:**

| Variable | Formula | Value |
|----------|---------|-------|
| **d₁** | As above | 0.103198 |
| **d₂** | d₁ - σ√T | 0.088291 |
| **N(d₁)** | CDF of Standard Normal(d₁) | 0.541097 |
| **N(d₂)** | CDF of Standard Normal(d₂) | 0.535177 |

### Black-Scholes Option Prices

**Call Option:**

$$C = S \cdot N(d_1) - K \cdot e^{-rT} \cdot N(d_2)$$
$$C = 23366.70 \times 0.541097 - 23350.00 \times e^{-0.065\times0.010959} \times 0.535177$$
$$C = \boxed{156.16}$$

**Put Option:**

$$P = K \cdot e^{-rT} \cdot N(-d_2) - S \cdot N(-d_1)$$
$$P = 23350.00 \times e^{-0.065\times0.010959} \times N(-0.088291) - 23366.70 \times N(-0.103198)$$
$$P = \boxed{122.84}$$

---

## Section 4: Market vs. Theoretical Prices

### Comparison Table

| Option Type | Theoretical Price | Market LTP | Absolute Error | Percentage Error |
|-------------|-------------------|------------|-----------------|------------------|
| **Call** | 156.16 | 161.00 | -4.84 | -3.00% |
| **Put** | 122.84 | 105.70 | 17.14 | +16.21% |

**Analysis:**
- The **Call option** theoretical price is **4.84 points lower** than market LTP, suggesting the market is valuing calls slightly higher than Black-Scholes (possibility of upside bias or skew effects).
- The **Put option** theoretical price is **17.14 points higher** than market LTP, indicating the market is underpricing puts relative to the model.

---

## Section 5: Greeks & Price Sensitivity

### Option Greeks (Call Option)

The Greeks represent the sensitivity of the option price to various factors:

| Greek | Symbol | Value | Interpretation |
|-------|--------|-------|-----------------|
| **Delta** | Δ | 0.541097 | Price increases 0.5411 for each 1-point move in spot |
| **Gamma** | Γ | 0.00113922 | Delta changes by 0.00113922 for each 1-point move |
| **Vega** | ν | 9.706852 | Price changes 9.7069 per 1% change in volatility |
| **Theta** | θ | -19.501997/day | Price decays 19.5020 per day (time decay) |

### Scenario Analysis: +1% Spot, +1% Volatility

**Scenario Setup:**
- Current Spot Price: 23,366.70
- New Spot Price (+1%): 23,600.37 (change: +233.67 points)
- Current Volatility: 14.24%
- New Volatility: 15.24% (change: +1.00%)

**Price Change Breakdown (Greeks Approximation):**

$$\Delta C \approx \Delta \times \Delta S + \frac{1}{2} \times \Gamma \times (\Delta S)^2 + \nu \times \Delta\sigma$$

| Component | Amount | Impact |
|-----------|--------|--------|
| Delta Impact | +126.44 | +233.67 points × 0.541097 |
| Gamma Impact | +31.10 | 0.5 × 0.00113922 × (233.67)² |
| Vega Impact | +0.10 | 9.706852 × 0.0100 |
| **Total Change (Approx)** | **+157.63** | **+100.94%** |

**Price Projection:**

| Metric | Value |
|--------|-------|
| Current Theoretical Call Price | 156.16 |
| Estimated New Price (Greeks Approx) | 313.80 |
| Exact New Price (Recalculation) | 319.30 |
| Greeks Approximation Error | 5.51 (1.72%) |

**Conclusion:** In this scenario, the call option is expected to gain approximately **157.63 points (or 100.94%)** in value. The Greeks approximation is highly accurate (within 1.72% error), validating the model's sensitivity measures.

---

## Appendix: Python Implementation

### Complete Analysis Script

```python
import pandas as pd
import numpy as np
from scipy.stats import norm
import yfinance as yf
from datetime import datetime, timedelta
import csv
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# BLACK-SCHOLES OPTION PRICING FRAMEWORK
# ============================================================================

def black_scholes_call(S, K, T, r, sigma):
    """Calculate Black-Scholes call price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price, d1, d2

def black_scholes_put(S, K, T, r, sigma):
    """Calculate Black-Scholes put price"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    return put_price, d1, d2

def calculate_greeks(S, K, T, r, sigma, option_type='call'):
    """Calculate option Greeks"""
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    # Delta
    if option_type == 'call':
        delta = norm.cdf(d1)
    else:
        delta = norm.cdf(d1) - 1

    # Gamma (same for both call and put)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))

    # Vega (same for both, per 1% change in volatility)
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100  # Per 1% change

    # Theta (per day, so divide by 365)
    if option_type == 'call':
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) - 
                 r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    else:
        theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) + 
                 r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365

    return {'delta': delta, 'gamma': gamma, 'vega': vega, 'theta': theta}

# ============================================================================
# PARAMETERS
# ============================================================================

S = 23366.7                    # Spot Price
K = 23350.0                    # Strike Price
T = 0.010959               # Time to Maturity
r = 0.065                    # Risk-Free Rate
sigma = 0.142400       # Volatility (Market IV)

# ============================================================================
# BLACK-SCHOLES PRICING
# ============================================================================

call_price, d1, d2 = black_scholes_call(S, K, T, r, sigma)
put_price, _, _ = black_scholes_put(S, K, T, r, sigma)

print(f"Call Price: {call_price:,.2f}")
print(f"Put Price: {put_price:,.2f}")
print(f"d1: {d1:.6f}")
print(f"d2: {d2:.6f}")

# ============================================================================
# GREEKS CALCULATION
# ============================================================================

call_greeks = calculate_greeks(S, K, T, r, sigma, option_type='call')
print(f"\nCall Delta: {call_greeks['delta']:.6f}")
print(f"Call Gamma: {call_greeks['gamma']:.8f}")
print(f"Call Vega: {call_greeks['vega']:.6f}")
print(f"Call Theta: {call_greeks['theta']:.6f}")

# ============================================================================
# SCENARIO ANALYSIS
# ============================================================================

spot_change_pct = 0.01
vol_change_abs = 0.01
S_new = S * (1 + spot_change_pct)
sigma_new = sigma + vol_change_abs

call_price_new, _, _ = black_scholes_call(S_new, K, T, r, sigma_new)
print(f"\nNew Call Price (S +1%, σ +1%): {call_price_new:,.2f}")
print(f"Change: {call_price_new - call_price:+,.2f}")
```

---

## Summary

This analysis demonstrates the application of the Black-Scholes model to value NIFTY 50 options expiring on June 9, 2026. Key findings:

1. **Strike Selection:** The ATM strike (23,350) is 16.70 points below the spot price of 23,366.70.

2. **Volatility Regime:** Market IV (14.24%) is 11.92% higher than historical volatility (12.72%), indicating elevated market expectations.

3. **Model Pricing vs. Market:**
   - Theoretical Call: 156.16 vs Market: 161.00
   - Theoretical Put: 122.84 vs Market: 105.70

4. **Greeks Sensitivity:** The call option exhibits moderate Delta (0.5411), positive Gamma (0.00113922), and significant negative Theta (-19.50/day).

5. **Scenario Outcome:** A +1% move in spot combined with +1% volatility increase would increase the call value by approximately 157.63 points (100.94%).

---

*Analysis completed on 2026-06-07 at 12:42:12 using Python Black-Scholes Framework*
