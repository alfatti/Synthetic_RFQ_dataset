# Synthetic RFQ Dataset — OTC HY Corporate Bond Market

End-to-end simulation of a Request-For-Quote (RFQ) dataset for a
high-yield corporate bond dealer desk. Designed to support training
of security recommender models and contextual bandit policies.

## Repository Structure

```
rfq_simulation/
├── rfq_sim/
│   ├── core/
│   │   ├── config.py          # All simulation parameters in one place
│   │   ├── calendar.py        # US business day calendar, trading hours
│   │   ├── bonds.py           # Bond universe, latent factors, similarity matrix
│   │   ├── clients.py         # Client universe, archetypes, static params
│   │   ├── mmpp.py            # Bidimensional MMPP (Bergault & Gueant 2024)
│   │   ├── price_process.py   # Factor model, GARCH, MMPP drift, jumps
│   │   ├── inventory.py       # Trader inventory state and hedging
│   │   ├── rfq_arrivals.py    # RFQ generation, sizes, lifetimes, program trades
│   │   ├── quoting.py         # Trader quoting model
│   │   └── outcomes.py        # WIN/LOSS/CANCELLED/EXPIRED logistic model
│   ├── utils/
│   │   ├── io.py              # Save/load parquet, ground truth separation
│   │   └── diagnostics.py     # Hit rate checks, data validation
│   └── simulator.py           # Main simulation loop
├── scripts/
│   ├── run_simulation.py      # CLI entry point
│   └── validate_dataset.py    # Post-hoc sanity checks
├── notebooks/
│   ├── 01_universe_inspection.ipynb
│   ├── 02_price_process.ipynb
│   ├── 03_rfq_flow_analysis.ipynb
│   ├── 04_outcome_model.ipynb
│   └── 05_easter_egg_audit.ipynb
├── tests/
│   └── test_core.py
├── data/
│   ├── raw/                   # Full output including ground truth
│   └── processed/             # Observable only (what models see)
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
python scripts/run_simulation.py --seed 42 --n-bonds 100 --n-clients 150
```

## Design Reference

See `docs/synthetic_rfq_whitepaper.tex` for the full mathematical
specification. Key design choices:

- Bergault & Gueant (2024) MMPP calibration for sector liquidity
- Client-bond affinity via latent factor dot product
- 5-7% unconditional hit rate
- Multinomial outcomes: WIN / LOSS / CANCELLED / EXPIRED
- US business day calendar, Jan 3 -- Jun 30 2023
- Hidden signal Easter eggs for recommender model benchmarking
