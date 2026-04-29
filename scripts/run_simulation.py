"""
scripts/run_simulation.py
--------------------------
CLI entry point.

Usage:
    python scripts/run_simulation.py
    python scripts/run_simulation.py --seed 42 --n-bonds 100 --n-clients 150
    python scripts/run_simulation.py --seed 7 --output-dir data/run_7

All outputs land in --output-dir:
    processed/  — observable.parquet, bond_metadata.parquet, client_metadata.parquet
    raw/        — ground_truth.parquet, *_gt.parquet, similarity.npy, affinity.npy
    config.json — full config for reproducibility
"""

import argparse, sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from rfq_sim.core.config import SimConfig
from rfq_sim.simulator import RFQSimulator
from rfq_sim.utils.io import save_simulation
from rfq_sim.utils.diagnostics import run_all_checks


def parse_args():
    p = argparse.ArgumentParser(description="Run the synthetic RFQ simulation")
    p.add_argument("--seed",       type=int,   default=42)
    p.add_argument("--n-bonds",    type=int,   default=100)
    p.add_argument("--n-issuers",  type=int,   default=30)
    p.add_argument("--n-clients",  type=int,   default=150)
    p.add_argument("--output-dir", type=str,   default="data")
    p.add_argument("--no-validate", action="store_true",
                   help="Skip post-simulation validation")
    return p.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("Synthetic RFQ Simulation — OTC HY Corporate Bonds")
    print("=" * 60)
    print(f"Seed {args.seed} | {args.n_bonds} bonds | {args.n_clients} clients")

    cfg = SimConfig(seed=args.seed)
    cfg.bonds.n_bonds     = args.n_bonds
    cfg.bonds.n_issuers   = args.n_issuers
    cfg.clients.n_clients = args.n_clients

    t0 = time.time()
    sim = RFQSimulator(cfg)
    obs_df, gt_df = sim.run()
    print(f"\nSimulation complete in {time.time()-t0:.1f}s")

    save_simulation(args.output_dir, obs_df, gt_df, sim.bonds, sim.clients, cfg)

    if not args.no_validate:
        print("\nValidation:")
        run_all_checks(obs_df, gt_df)


if __name__ == "__main__":
    main()
