# Resume Pointers — American Put Option Pricing

Drafted with the *IIT Bombay Career Cell Master Resume Pointer Guide* framework:
STAR-Impact 4-part equation (Power Verb + Method/Tooling + Context + Quantified
Result), 15–24 words per line, 2–4 bolded tokens (metrics/tools only), no verb
repetition, no trailing periods. Target format: **Software / AI-ML / Data Science
(dense)** — lead with tooling and measurable performance.

Every number below is sourced from the repo's own READMEs, `analysis_results.md`,
and artifacts — nothing is invented. Swap in the exact tokens you want bolded to
match your template's line width.

---

## Recommended resume block (6 pointers, one project)

**American Put Option Pricing — Classical Models → Deep Reinforcement Learning** *(Python, PyTorch, NumPy)*

- Engineered a **5-stage** option-pricing pipeline advancing from Black-Scholes and CRR binomial trees to a **Double DQN** early-exercise policy
- Priced live **NIFTY 50** options via Black-Scholes with the full Greeks suite, matching market call prices to **within 3%** on a 4-day expiry
- Architected a Cox-Ross-Rubinstein binomial pricer converging under **$0.001** by 500 steps and quantifying a **3.3%** early-exercise premium over European puts
- Distilled the binomial tree pricer into a **PyTorch** MLP over **12,000** synthetic contracts, reaching 0.06 test MAE at 0.4% of the mean price
- Formulated early exercise as a **Markov Decision Process** in a leakage-free Gym-style environment hardened by **19** invariant tests
- Trained a Double DQN hold/exercise policy from interaction alone to a **7.56** discounted value, landing **within 5%** of the binomial optimum

> **Verbs used (all unique):** Engineered · Priced · Architected · Distilled · Formulated · Trained.
> If you add the debugging pointer below, it introduces *Diagnosed* — still no repeats.

**Optional 7th pointer (signals debugging depth for research/quant roles):**

- Diagnosed and corrected a deep-ITM **Q-value inversion** plus late-training value decay via interleaved exploring starts and confirmed best-checkpoint selection

---

## Alternate candidates per module

Pick the phrasing that best fits your line width and the role. Each set follows the
guide's three-candidate logic: **A** = impact-first inversion, **B** = methodology/depth,
**C** = executive/strategic synthesis.

### Black-Scholes analytics (NIFTY 50 option chain)

- **A —** Matched Black-Scholes theoretical call prices to **within 3%** of live NIFTY 50 market quotes, computing Delta, Gamma, Vega, and Theta on a 4-day expiry
- **B —** Modeled NIFTY 50 option value in Python via Black-Scholes, benchmarking **14.24%** implied against **12.72%** 30-day historical volatility from live `yfinance` data
- **C —** Validated a Greeks-based sensitivity approximation **within 1.72%** of exact repricing under a joint +1% spot and +1% volatility shock scenario

### CRR binomial tree pricer (baseline)

- **A —** Quantified a **3.3%** early-exercise premium ($10.80 vs $10.45) by pricing American and European puts on a Cox-Ross-Rubinstein binomial tree
- **B —** Architected a CRR binomial pricer with backward-induction early-exercise logic, converging under **$0.001** by 500 steps across a 25–1000 step sweep
- **C —** Mapped the early-exercise boundary and 3D price surface, validating six no-arbitrage bounds through a **7-test** sanity suite

### Neural network pricer (supervised imitation)

- **A —** Achieved **0.06** test MAE (0.4% of mean price) with a PyTorch MLP imitating a binomial pricer across 12,000 labeled contracts
- **B —** Trained a 5→128→128→1 **PyTorch** MLP under best-checkpoint selection, enforcing monotonicity, non-negativity, and intrinsic-value finance sanity checks
- **C —** Built a leakage-safe 80/10/10 pipeline standardized on training data only, exposing highest error near the early-exercise boundary via moneyness-bucketed MAE

### RL formulation + tabular Q-learning (Week 7)

- **A —** Lifted state-space coverage from **200 to 600** cells with a warmup-then-refine exploring-starts schedule, tripling tabular-Q discounted payoff to 4.17
- **B —** Formulated American-put exercise as an **MDP** with risk-neutral CRR transitions and present-value discounting, verified by 19 leakage and reward-timing tests
- **C —** Engineered a Gym-style `AmericanPutEnv` and diagnostic margin plots that exposed value-function structure a binary exercise-region plot could not

### Double DQN stopping policy (Week 8)

- **A —** Reached a **7.56** discounted value — above the European baseline and **within 5%** of the binomial optimum — with a Double DQN learned label-free
- **B —** Implemented **Double DQN** with Huber loss, a 50k replay buffer, target-network sync, and decaying epsilon/learning-rate/exploring-start schedules
- **C —** Isolated two training failures via direct Q-margin queries and a value-convergence curve, recovering optimal value through confirmed best-checkpoint selection

---

## One-line project summary (for a header or LinkedIn)

- Built an end-to-end American put pricing suite — Black-Scholes, binomial trees, supervised deep learning, and reinforcement learning — validated by **57+** automated tests

## Technical Skills line (contribution)

**Quant/ML:** Black-Scholes & Greeks, Cox-Ross-Rubinstein binomial trees, PyTorch, Double DQN / tabular Q-learning, Markov Decision Processes, NumPy, Monte Carlo evaluation

---

## Self-correction checklist (applied)

| Rule | Status |
|---|---|
| Power verb, unique per section | ✓ Engineered, Priced, Architected, Distilled, Formulated, Trained, Diagnosed |
| Tense consistency (completed → past) | ✓ all past tense |
| 15–24 words per pointer | ✓ every line in range |
| 2–4 bolded tokens (metrics/tools only) | ✓ numbers and named frameworks only |
| Trailing period stripped | ✓ none |
| Anti-vagueness ("various", "worked on", "successfully") | ✓ replaced with exact counts |

**Choose-your-own-density note:** the recommended block bolds one metric or tool per
line. If your template runs a metric bold *and* a tool bold on the same line pushes
past 4 bolded words, keep the metric — hard numbers route the recruiter's eye first.
