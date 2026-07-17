"""Validation / audit for a generated dataset. Writes a text report + PNGs.

Usage: python -m rfq_synth.validate /path/to/out
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .config import GenConfig
from .generator import logging_pmf
from .ground_truth import true_q, win_prob


def main(od):
    cfg = GenConfig(**{k: v for k, v in json.load(
        open(f"{od}/answers/config_dump.json")).items()
        if k in GenConfig.__dataclass_fields__})
    g = cfg.grid_bps
    ev = pd.read_parquet(f"{od}/rfq_events.parquet")
    tru = pd.read_parquet(f"{od}/evaluation/train_truth.parquet")
    hid = pd.read_parquet(f"{od}/answers/hidden_train.parquet")
    cf = pd.read_parquet(f"{od}/rfq_counterfactuals.parquet")
    fig_dir = os.path.join(od, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    rep = []
    win = ev.outcome.values == "CLIENT-TRADED"

    rep.append(f"rows train={len(ev):,}  counterfactuals={len(cf):,}")
    rep.append(f"hit rate = {win.mean():.4f} (target {cfg.target_hit_rate})")

    dec = pd.qcut(ev.liq_score, 10, labels=False, duplicates="drop")
    ladder = pd.Series(win).groupby(dec).mean()
    rep.append("hit by liquidity decile (illiquid->liquid): "
               + np.array2string(ladder.values.round(3)))

    m = ev[["rfq_id", "spread_bps", "markout_bps", "reward_bps", "pscore",
            "action", "size_mm", "liq_score"]].merge(hid, on="rfq_id")
    tier_hit = pd.Series(win).groupby(m.client_tier.values).mean()
    rep.append("hit by hidden tier 1..5: " + np.array2string(tier_hit.values.round(3)))

    rep.append(f"winner's curse: E[markout|win]={m.markout_bps[win].mean():.2f} "
               f"vs E[markout]={m.markout_bps.mean():.2f} bps")
    w = m[win]
    terc = w.groupby(pd.qcut(w.spread_bps, 3, labels=False))["markout_bps"].mean()
    rep.append("E[markout|win] by spread tercile: " + np.array2string(terc.values.round(2)))

    # propensity calibration: empirical action freq vs mean pi0 on a sample
    samp = np.random.default_rng(0).choice(len(ev), 40_000, replace=False)
    pmf = logging_pmf(tru.center_bps.values[samp], cfg)
    emp = np.bincount(ev.action.values[samp], minlength=cfg.n_actions) / len(samp)
    corr = np.corrcoef(pmf.mean(0), emp)[0, 1]
    rep.append(f"pi0 calibration corr(mean pmf, empirical freq) = {corr:.4f}")
    sup = (pmf > 0).mean()
    rep.append(f"support deficiency: {1 - sup:.1%} of (context, action) pairs unsupported")
    rep.append(f"pscore: min={ev.pscore.min():.2e} p50={ev.pscore.median():.3f} "
               f"max={ev.pscore.max():.3f}")

    # demand-curve recovery per tier: empirical win vs analytic
    rep.append("demand-curve check (per-tier logistic fit on logged data):")
    for t in range(1, 6):
        sel = m.client_tier.values == t
        x, y = ev.spread_bps.values[sel], win[sel].astype(float)
        b = pd.qcut(x, 12, labels=False, duplicates="drop")
        curve = pd.DataFrame({"b": b, "y": y, "x": x}).groupby("b").mean()
        rep.append(f"  tier {t}: hit@tight-bin={curve.y.iloc[0]:.2f} "
                   f"hit@wide-bin={curve.y.iloc[-1]:.3f} n={sel.sum():,}")

    # ground truth: analytic q vs Monte Carlo from counterfactual doubles
    j = cf.merge(tru[["rfq_id", "m_c", "s_c", "mu_delta", "a1"]], on="rfq_id")
    key = j.groupby("alt_action")
    sub = j[j.alt_action.isin(key.size().nlargest(5).index)]
    mc = sub.groupby("alt_action").agg(emp=("alt_reward_bps", "mean"))
    an = []
    for a in mc.index:
        s = sub[sub.alt_action == a]
        an.append(true_q(s.m_c.values, s.s_c.values, s.mu_delta.values,
                         s.a1.values, np.array([g[a]]))[:, 0].mean())
    mc["analytic"] = an
    rep.append("MC(counterfactual rewards) vs analytic q, top-5 alt actions:\n"
               + mc.round(3).to_string())

    # value accounting
    v0_emp, v0_true = ev.reward_bps.mean(), tru.true_q_logged.mean()
    rep.append(f"V(pi0): empirical={v0_emp:.3f} exact={v0_true:.3f} bps/RFQ")
    rep.append(f"V(oracle) = {tru.oracle_q.mean():.3f} bps/RFQ "
               f"(headroom x{tru.oracle_q.mean() / v0_true:.1f})")

    # local correctness sanity: within-band q spread vs cross-band
    samp2 = np.random.default_rng(1).choice(len(tru), 20_000, replace=False)
    q = true_q(tru.m_c.values[samp2], tru.s_c.values[samp2],
               tru.mu_delta.values[samp2], tru.a1.values[samp2], g)
    per = cfg.n_actions // cfg.n_bands
    qb = q.reshape(len(samp2), cfg.n_bands, per)
    within = qb.std(axis=2).mean()
    cross = qb.mean(axis=2).std(axis=1).mean()
    rep.append(f"g/h structure: mean within-band q std={within:.3f}, "
               f"cross-band std of band means={cross:.3f} (ratio {cross/within:.1f}x)")

    # figures
    fig, ax = plt.subplots(2, 2, figsize=(11, 8))
    ladder.plot(kind="bar", ax=ax[0, 0], title="hit rate by liquidity decile")
    cnt = ev.groupby("cusip").size().sort_values(ascending=False).values
    ax[0, 1].loglog(np.arange(1, len(cnt) + 1), cnt)
    ax[0, 1].set_title("RFQs per CUSIP (Zipf flow)")
    mkt = pd.read_parquet(f"{od}/market_daily.parquet")
    reg = pd.read_parquet(f"{od}/answers/regime.parquet")
    ax[1, 0].plot(mkt.day, mkt.vix_like, lw=0.8)
    ax[1, 0].fill_between(mkt.day, 0, 1, where=reg.regime_stressed.values > 0,
                          transform=ax[1, 0].get_xaxis_transform(), alpha=0.2, color="r")
    ax[1, 0].set_title("vix_like proxy vs hidden regime (red)")
    row = tru.iloc[7]
    ax[1, 1].plot(g, true_q([row.m_c], [row.s_c], [row.mu_delta], [row.a1], g)[0])
    ax[1, 1].axvline(g[int(row.oracle_action)], color="g", ls="--", label="oracle")
    ax[1, 1].axvline(row.center_bps, color="k", ls=":", label="pi0 center")
    ax[1, 1].set_title("example true q(x, spread)"); ax[1, 1].legend()
    fig.tight_layout(); fig.savefig(f"{fig_dir}/validation.png", dpi=110)

    fig2, ax2 = plt.subplots(figsize=(7, 4))
    for t, c in zip(range(1, 6), plt.cm.viridis(np.linspace(0, 1, 5))):
        sel = m.client_tier.values == t
        b = pd.qcut(ev.spread_bps.values[sel], 15, labels=False, duplicates="drop")
        d = pd.DataFrame({"b": b, "x": ev.spread_bps.values[sel],
                          "y": win[sel]}).groupby("b").mean()
        ax2.plot(d.x, d.y, color=c, label=f"tier {t}")
    ax2.set_xlim(0, 40); ax2.set_xlabel("quoted spread (bps)")
    ax2.set_ylabel("P(CLIENT-TRADED)"); ax2.legend()
    ax2.set_title("empirical demand curves by hidden tier")
    fig2.tight_layout(); fig2.savefig(f"{fig_dir}/demand_curves.png", dpi=110)

    txt = "\n".join(rep)
    open(f"{od}/validation_report.txt", "w").write(txt)
    print(txt)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else GenConfig().out_dir)
