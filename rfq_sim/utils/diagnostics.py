"""utils/diagnostics.py — post-simulation validation and Easter egg checks."""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def check_hit_rate(obs, lo=0.03, hi=0.12):
    hr = (obs["outcome"] == "WIN").mean()
    status = "OK" if lo <= hr <= hi else f"WARNING — outside [{lo:.0%}, {hi:.0%}]"
    print(f"Hit rate: {hr:.2%}  {status}")
    return hr


def outcome_distribution(obs):
    dist = obs["outcome"].value_counts(normalize=True).round(3)
    print("Outcome distribution:\n", dist.to_string())
    return dist


def cancel_rho_correlation(obs, gt):
    m = obs.merge(gt[["rfq_id", "rho_k"]], on="rfq_id")
    c, p = spearmanr(m["rho_k"], (m["outcome"] == "CANCELLED").astype(float))
    status = "PRESENT" if c > 0.02 else "WEAK"
    print(f"Cancel–rho_k Spearman: {c:.4f}  p={p:.4f}  [{status}]")
    return c


def affinity_hit_rate_by_quartile(obs, gt):
    m = obs.merge(gt[["rfq_id", "affinity_kn"]], on="rfq_id")
    m["q"] = pd.qcut(m["affinity_kn"], q=4, labels=["Q1","Q2","Q3","Q4"])
    r = m.groupby("q", observed=True)["outcome"].apply(
        lambda x: (x == "WIN").mean()
    ).round(4)
    print("Hit rate by affinity quartile (Q1→Q4 should increase):\n", r.to_string())
    return r


def check_train_test_split(obs):
    ts = pd.to_datetime(obs["timestamp"])
    tr = ts[obs["split"] == "train"]
    te = ts[obs["split"] == "test"]
    ok = (len(tr) > 0 and len(te) > 0 and tr.max() < te.min())
    print(f"Train/test split: {'OK' if ok else 'LEAKAGE'} "
          f"(train ends {tr.max().date()}, test starts {te.min().date()})")
    return ok


def month_end_boost(obs):
    obs = obs.copy()
    obs["date"]  = pd.to_datetime(obs["timestamp"]).dt.date
    obs["month"] = pd.to_datetime(obs["timestamp"]).dt.to_period("M")
    daily = obs.groupby("date").size().reset_index(name="n")
    daily["date"]  = pd.to_datetime(daily["date"])
    daily["month"] = daily["date"].dt.to_period("M")

    me_days = set()
    for _, g in daily.groupby("month"):
        for d in g.sort_values("date")["date"].iloc[-2:]:
            me_days.add(d)
    daily["is_me"] = daily["date"].isin(me_days)
    norm = daily[~daily["is_me"]]["n"].mean()
    me   = daily[daily["is_me"]]["n"].mean()
    boost = me / max(norm, 1)
    print(f"Month-end boost: {boost:.3f}x  (target ~1.30x)")
    return boost


def run_all_checks(obs, gt):
    print("=" * 50)
    print("DATASET VALIDATION")
    print("=" * 50)
    check_hit_rate(obs)
    outcome_distribution(obs)
    cancel_rho_correlation(obs, gt)
    affinity_hit_rate_by_quartile(obs, gt)
    check_train_test_split(obs)
    month_end_boost(obs)
    print("=" * 50)
