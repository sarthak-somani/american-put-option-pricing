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