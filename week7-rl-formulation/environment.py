"""Gym-style hold/exercise environment for the American put optimal-stopping MDP.

State      : [time_fraction, moneyness] = [step / steps, spot / K]
Actions    : HOLD (0) advances one risk-neutral binomial step; EXERCISE (1) stops
             the episode and pays max(K - spot, 0).
Transition : risk-neutral CRR up/down move (matches binomial-tree/american_put.py),
             not physical drift -- rewards must be pricing-consistent.
Reward     : paid exactly once, at the stopping step (exercise, or forced exercise
             at expiry). Holding always returns reward 0.0.
Discount   : self.discount = exp(-r * dt) is the *per-step* discount factor. The
             environment does not discount rewards itself -- callers accumulate
             gamma**t over elapsed steps when they need a present-value estimate.
"""

import math
import numpy as np


class AmericanPutEnv:
    HOLD = 0
    EXERCISE = 1

    def __init__(self, S0=100.0, K=100.0, T=1.0, r=0.05, sigma=0.25, steps=50, seed=42):
        if S0 <= 0 or K <= 0:
            raise ValueError("S0 and K must be positive")
        if T <= 0:
            raise ValueError("T must be positive")
        if sigma <= 0:
            raise ValueError("sigma must be positive")
        if int(steps) != steps or steps < 1:
            raise ValueError("steps must be a positive integer")

        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma
        self.steps = int(steps)

        self.dt = T / self.steps
        self.u = math.exp(sigma * math.sqrt(self.dt))
        self.d = 1.0 / self.u
        self.p = (math.exp(r * self.dt) - self.d) / (self.u - self.d)
        self.discount = math.exp(-r * self.dt)

        if not (0.0 < self.p < 1.0):
            raise ValueError("Invalid risk-neutral probability; check inputs")

        self.rng = np.random.default_rng(seed)
        self.step_count = 0
        self.spot = self.S0
        self.done = False

    def _state(self):
        return np.array([self.step_count / self.steps, self.spot / self.K], dtype=np.float32)

    def reset(self, seed=None, step_count=None, spot=None):
        """Start a new episode. Pass `seed` only to force a reproducible price path;
        omit it to keep drawing from the running RNG stream (independent episodes
        across repeated resets, as needed for Monte Carlo policy evaluation).

        Pass `step_count`/`spot` to start from an arbitrary state instead of the
        contract's actual inception (t=0, S=S0) -- this is an "exploring start,"
        used by training loops (e.g. q_learning.train_q_learning) to inject
        experience directly into states a real t=0 rollout would rarely reach.
        Evaluation should not use this: real episodes always begin at inception.
        """
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        if step_count is None:
            step_count = 0
        if spot is None:
            spot = self.S0
        if not (0 <= step_count < self.steps):
            raise ValueError("step_count must be in [0, steps)")
        if spot <= 0:
            raise ValueError("spot must be positive")

        self.step_count = step_count
        self.spot = spot
        self.done = False
        return self._state()

    def step(self, action):
        if self.done:
            raise RuntimeError("Episode is already done. Call reset().")
        if action not in (self.HOLD, self.EXERCISE):
            raise ValueError("action must be 0=hold or 1=exercise")

        if action == self.EXERCISE:
            payoff = max(self.K - self.spot, 0.0)
            self.done = True
            return self._state(), payoff, True, {"reason": "exercise", "step": self.step_count}

        # Hold: advance one risk-neutral binomial step. No reward for holding.
        if self.rng.random() < self.p:
            self.spot *= self.u
        else:
            self.spot *= self.d
        self.step_count += 1

        if self.step_count >= self.steps:
            self.done = True
            terminal_payoff = max(self.K - self.spot, 0.0)
            return self._state(), terminal_payoff, True, {"reason": "expiry", "step": self.step_count}

        return self._state(), 0.0, False, {"reason": "hold", "step": self.step_count}
