"""
scripts/validate_dataset.py
----------------------------
Post-hoc dataset validation — run after a simulation to confirm all
Easter egg signals are detectable before handing off to model teams.

Usage:
    python scripts/validate_dataset.py --data-dir data/
"""

import argparse, sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from rfq_sim.utils.io import load_all
from rfq_sim.utils.diagnostics import run_all_checks


def check_affinity_arrival(data):
    obs = data["observable"]
    aff = data["affinity"]
    counts = obs.groupby(["client_id","bond_id"]).size().reset_index(name="n")
    counts["affinity"] = counts.apply(
        lambda r: float(aff[int(r.client_id), int(r.bond_id)]), axis=1
    )
    c, p = spearmanr(counts["n"], counts["affinity"])
    print(f"[EE1] Affinity vs RFQ count  Spearman={c:.4f}  p={p:.4f}  "
          f"[{'OK' if c > 0.03 else 'WEAK'}]")


def check_program_fraction(data):
    obs = data["observable"]
    frac = obs["client_in_program"].mean()
    print(f"[EE3] Program-state fraction: {frac:.2%}  "
          f"[{'OK' if 0.02 < frac < 0.60 else 'CHECK CONFIG'}]")


def check_month_end(data):
    obs = data["observable"].copy()
    obs["date"]  = pd.to_datetime(obs["timestamp"]).dt.date
    obs["month"] = pd.to_datetime(obs["timestamp"]).dt.to_period("M")
    daily = obs.groupby("date").size().reset_index(name="n")
    daily["date"]  = pd.to_datetime(daily["date"])
    daily["month"] = daily["date"].dt.to_period("M")
    me = set()
    for _, g in daily.groupby("month"):
        for d in g.sort_values("date")["date"].iloc[-2:]:
            me.add(d)
    daily["is_me"] = daily["date"].isin(me)
    norm_avg = daily[~daily["is_me"]]["n"].mean()
    me_avg   = daily[ daily["is_me"]]["n"].mean()
    ratio    = me_avg / max(norm_avg, 1)
    print(f"[EE9] Month-end boost: {ratio:.3f}x  [{'OK' if ratio > 1.05 else 'WEAK'}]")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    args = p.parse_args()

    print(f"Loading from {args.data_dir}...")
    data = load_all(args.data_dir)
    obs, gt = data["observable"], data["ground_truth"]
    print(f"{len(obs):,} RFQs  |  {obs['timestamp'].min()[:10]} → {obs['timestamp'].max()[:10]}")

    run_all_checks(obs, gt)

    print("\n=== Easter Egg Signal Audit ===")
    check_affinity_arrival(data)
    check_program_fraction(data)
    check_month_end(data)
    print("Done.")


if __name__ == "__main__":
    main()
