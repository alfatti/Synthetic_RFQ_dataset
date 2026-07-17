# Synthetic IG-Corp RFQ Bandit Dataset

Logged-bandit data for off-policy learning / evaluation with large action
spaces (POTEC-style two-stage OPL). The data-generating process follows the
dealer market-making world of Bergault-Evangelista-Gueant-Vieira (closed-form
approximations for multi-asset market making): RFQ arrivals with logistic fill
probability in the quoted spread, client tiering, inventory-skewed quoting,
and a factor risk model — mapped onto a one-step contextual bandit.

## The bandit problem

- **Context x**: client, bond (CUSIP), RFQ terms (side/size/#dealers in comp),
  market proxies, desk inventory summaries. One row per RFQ.
- **Action a**: the quoted spread, on a 500-point grid (0.25 .. 125 bps,
  0.25 bp ticks). `action_space.parquet` maps actions to spreads and provides
  a **true clustering** (25 contiguous 5-bp bands) plus noisy/coarser/finer
  candidate clusterings for cluster-robustness ablations.
- **Outcome** (two classes): `CLIENT-TRADED` iff the quote beats the best
  competing dealer ("the cover"), else `CLIENT-TRADED-AWAY`. Overall hit rate
  is calibrated to ~5%, strongly heterogeneous across bonds/clients.
- **Reward** (bps of notional): `(spread - markout - cost) * 1{TRADED}` —
  spread captured net of adverse selection. The markout is the 30-minute
  unhedged mid move against the dealer; it is reported on **all** rows
  (mid moves are public) while it only enters PnL on wins. Rewards on wins
  can be negative. Use `outcome` if you want the binary-reward variant
  (degenerate: optimum is always the tightest quote — documented, not
  recommended).
- **Logging policy pi0**: a trader heuristic built from observables plus
  inventory skew and trader effects, softly mixed toward the true optimum
  (`logging_quality`), discretised via a softmax with round-tick stickiness
  and a **hard support window** of +/-60 ticks around the trader's center —
  i.e. deliberate support deficiency. `pscore` is the exact propensity of the
  logged action. The full pmf for any row is reconstructable with
  `generator.logging_pmf(center_bps, cfg)` using `center_bps` from
  `evaluation/*_truth.parquet`.

## Counterfactual doubles

The cover and the markout are drawn **once per RFQ**, so outcomes under any
alternative quote are deterministic given the logged latent state:
tighter-than-cover wins, wider loses, same markout. 
`rfq_counterfactuals.parquet` materialises K=8 stratified alternative quotes
per training RFQ with their outcomes, rewards, propensities, and exact
`true_q_alt`. Coupling is monotone by construction: if a tighter quote lost,
every wider quote also lost.

## Files

| file | contents |
|---|---|
| `rfq_events.parquet` | 1M logged training RFQs (x, a, pscore, outcome, reward, markout) |
| `rfq_events_test.parquet` | 200k held-out RFQs, same schema (`rfq_id` offset by 10M) |
| `rfq_counterfactuals.parquet` | 8 counterfactual quotes per training RFQ |
| `action_space.parquet` | grid, true bands, noisy(10/30%)/coarse/fine clusterings |
| `bonds.parquet` / `clients.parquet` | slow-moving reference data (observables only) |
| `market_daily.parquet` | daily market proxies (`vix_like`, `cdx_like`) |
| `evaluation/{train,test}_truth.parquet` | per-RFQ sufficient stats `(m_c, s_c, mu_delta, a1)` for **exact** q(x, .), plus `center_bps`, `true_q_logged`, `oracle_action`, `oracle_q` |
| `answers/` | sealed hidden structure — don't open before the exercise |
| `validation_report.txt`, `figures/` | audit of the shipped generation |

## Exact evaluation (no Monte Carlo needed)

```python
from rfq_synth.config import GenConfig
from rfq_synth.ground_truth import true_q, policy_value
import pandas as pd

cfg = GenConfig()
truth = pd.read_parquet("out/evaluation/test_truth.parquet")

def my_policy_probs(lo, hi):
    # return (hi-lo, 500) action-probability rows for test contexts [lo:hi)
    ...

V = policy_value(my_policy_probs, truth, cfg.grid_bps)   # bps per RFQ
```

`q(x, delta) = (delta - cost - mu_delta) * sigma(-(delta-m_c)/s_c)
- a1 * s_c * (u*sigma(-u) + softplus(-u))` — closed form; see
`ground_truth.py` for the derivation. `oracle_q` gives the per-row optimum;
V(pi0) is `true_q_logged.mean()`.

## POTEC / OBP mapping

- `context` = feature columns of `rfq_events*.parquet` (join bonds/clients as
  desired), `action` = `action`, `reward` = `reward_bps`, `pscore` = `pscore`.
- Cluster map `phi(a)` = `band_true` (or a noisy variant) in
  `action_space.parquet`; |A| = 500, |C| = 25.
- Regression baselines: fit q on logged data; two-stage: learn the band
  policy from cluster-level effects, then a within-band scorer. Local
  correctness is attainable: q is smooth within bands (the report quantifies
  within- vs cross-band variation).

## Caveats (read before drawing conclusions)

1. **One-step counterfactuals.** The cover does not react to your policy:
   no competitive response, no client-relationship dynamics, no inventory
   feedback loop. The oracle exploits this freely (its headroom vs pi0 is
   large by construction — the 5% hit constraint pins pi0 to be conservative).
2. **i.i.d.-style contexts.** Inventory summaries are sampled, not simulated
   through fills; day effects exist but rows are exchangeable within the DGP.
3. **`evaluation/*_truth.parquet` is the oracle.** It exposes per-row
   sufficient statistics; anyone who reads it can reverse-engineer parts of
   the hidden structure. Keep it out of model-training pipelines; it exists
   for exact policy evaluation. Hidden labels remain in `answers/`.
4. Markout is observable on losses too (public mids), which is realistic and
   intentionally useful; strict-bandit purists should restrict features to
   pre-quote columns and reward.

## Regenerate / rescale

```bash
python -m rfq_synth.generator          # defaults: 1M train / 200k test
```

or in Python: `generate(GenConfig(n_train=..., n_bonds=..., seed=...))`.
Everything (scale, hit-rate target, support window, stickiness, coupling
strength, logging quality, K) is a `GenConfig` field. Generation is fully
deterministic given `seed`.
