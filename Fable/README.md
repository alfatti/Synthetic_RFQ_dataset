# rfqsim — Synthetic Electronic Credit Desk RFQ Generator

Simulates an IG corporate-bond e-trading desk: **5k clients, 30k CUSIPs,
~100k RFQs/day**, four outcomes (CLIENT-TRADED ~5–7%, CLIENT-TRADED-AWAY,
CLIENT-CANCELLED, EXPIRED), built around the liquidity-dynamics framework of
Bergault & Guéant (2024), *Liquidity Dynamics in RFQ Markets and Impact on
Pricing*.

## Embedded "Easter eggs" (recoverable structure)

1. **Sector-level bidimensional MMPP** (2 intensity states/side, exchangeable
   generator, Appendix A.1) driving RFQ arrival regimes.
2. **Per-CUSIP betas** composed (not drawn): issuer Zipf × size^1.3 ×
   age-decay (on-the-run effect, ~120-day half-life) × benchmark-tenor boost
   × lognoise — gives Gini ≈ 0.84, top-10% ≈ 74% of flow, majority of CUSIPs
   silent on any given day, and a *nonstationary* concentration ranking via a
   live primary-issuance calendar.
3. **Price drift κ·(λᵃ−λᵇ)**: composite mids = issuer Brownian factor
   (weekly grid + intra-week bridge → continuous across parallel workers)
   + exact piecewise-linear MMPP imbalance integral + CUSIP dispersion.
4. **Auction outcome model**: trade-at-all is logistic in the *best* quote
   (α=−0.7, β=3.1 per the paper); we win iff our quote beats the closed-form
   best-of-(k−1) cover. Composition yields the 5–7% hit rate without
   hand-tuning, plus heterogeneity by tier, size, k-in-comp, and regime skew.
5. **Client attribution layer**: P(client | bond bucket, side, week) with
   mandates, type affinities (HF↔new issues, insurer↔long end), weekly
   log-OU activity drift, and per-client intent noise that caps achievable
   model AUC realistically. Attribution is multinomial thinning, so the MMPP
   count process is exactly preserved.

## Run

```bash
pip install numpy scipy pyarrow polars            # + cupy-cuda12x on the GPU box
python run.py --out ./rfq_out                     # full year, auto multi-GPU
python run.py --smoke --no-gpu --out ./smoke      # 30s CPU sanity check
python run.py --seeds 16 --out ./ensemble         # benchmark ensemble
```

Output: Parquet partitioned by `week=NNN`, zstd. One year ≈ 25M rows ≈ 1.6 GB.

## Using the 4×H200 / 176-CPU / 740GB box

- **Parallel axis = time.** Weeks shard round-robin across one process per
  GPU (`RFQSIM_DEVICE` pins CuPy device). Chunks are independent because the
  MMPP chains + weekly issuer-factor grid are precomputed on the master and
  intra-week prices are Brownian *bridges* between grid points — paths stay
  continuous regardless of which worker generated which week.
- **Hot path** (Poisson scatter, searchsorted sampling, segmented cumsums,
  outcome RNG) runs on CuPy; cold path (universe, CTMC, weekly CDF tables)
  stays NumPy — it is tiny and transfer latency would dominate.
- **Honest sizing:** one year is a single-GPU-seconds workload measured at
  ~150–175k rows/s/worker on CPU alone; H200s push the generation to where
  zstd Parquet writing is the bottleneck (hence compression level 3 and
  per-worker writes). The hardware's real leverage is `--seeds N`:
  independent years for model benchmarking, embarrassingly parallel across
  the 4 GPUs, with the 176 cores absorbing compression and the validation
  suite (Polars). 740 GB RAM means even a 10-year × 16-seed ensemble never
  touches disk mid-generation.
- To split a single year across all 4 GPUs anyway: it already does
  (weeks round-robin). To parallelize the validation suite, point
  `validate.py` at the partitioned dataset — Polars scan_parquet uses all
  cores.

## Validation suite (run automatically by run.py)

| metric | target | measured (60d run) |
|---|---|---|
| hit rate (CLIENT-TRADED) | 5–7% | 5.5% |
| class split | TA > hit, CXL/EXP majority | 28.0 / 31.7 / 34.8% |
| CUSIP Gini | >0.84 | 0.842 |
| top-10% CUSIP flow share | ~70%+ | 73.8% |
| daily zero-RFQ CUSIP fraction | majority | 56% |
| client Gini | ~0.8+ | 0.87 |
| S-curve fill vs distance | monotone ↓ | yes |
| hit rate by tier (0/1/2) | monotone ↓ | 5.8 / 4.4 / 3.5% |
| imbalance → same-day drift corr | > 0 | +0.20 |

## Module map

```
rfqsim/backend.py      CuPy/NumPy switch, device pinning
rfqsim/config.py       every parameter, dataclass-based
rfqsim/mmpp.py         exchangeable 4-state CTMC + imbalance integral
rfqsim/universe.py     issuers, composed CUSIP weights, primary calendar,
                       clients/mandates/OU drift, weekly attribution CDFs
rfqsim/engine.py       GPU hot path: scatter, sample, price, outcomes
rfqsim/orchestrate.py  multi-GPU week sharding, Parquet writes, ensembles
rfqsim/validate.py     Easter-egg recovery checks (Polars)
run.py                 CLI
```

## Schema

`rfq_id, timestamp, date, sector, issuer_id, cusip_id, client_id,
client_type, client_tier, side (BUY/SELL), size, k_dealers, composite_bid,
composite_ask, composite_mid, our_quote, cover_price (TRADED-AWAY only,
MarketAxess-style disclosure), mmpp_state, status`.

## Notes / extension hooks

- Rating-migration and index-rebalance event shocks are the next affinity
  layer (discussed in design); slot into `universe.client_bucket_cdfs` and
  the side-tilt as multiplicative bumps.
- The EM-recovery test for the MMPP (full closed-loop check from the paper's
  Section 2.2) is the natural next test module: run the EM on a generated
  sector's (t, side) stream and assert λ¹, λ², Q within tolerance.
