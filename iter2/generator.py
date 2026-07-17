"""Synthetic IG-corp RFQ dataset generator (POTEC-ready logged bandit data).

World model per the Bergault-Evangelista-Gueant-Vieira setup, mapped to the
locked two-outcome design:

  context x  : client, bond (CUSIP), market state, RFQ terms, book exposures
  action a   : quoted spread on a 500-tick grid (0.25 .. 125 bps)
  logging pi0: closed-form-style trader quote (base + inventory skew + biases),
               softmax-discretised with round-tick stickiness and a hard
               support window (deliberate support deficiency); exact pscores
  latent     : cover dbar ~ Logistic(m_c, s_c) drawn ONCE per RFQ ->
               counterfactual doubles are deterministic for any quote
  markout    : Delta = mu_delta + a1*(dbar - m_c) + eps  (winner's curse)
  outcome    : CLIENT-TRADED iff spread < dbar, else CLIENT-TRADED-AWAY
  reward     : (spread - Delta - cost) * 1{TRADED}, in bps of notional

Hidden Easter-egg structure: 5 client tiers + continuous sharpness, 2-state
market regime (HMM, proxies only), issuer/sector factor co-movement in
markouts, trader fixed effects (inventory-blind trader, Friday-widener),
round-tick support deficiency, size concavity, liquidity/age decay (Zipf flow).
"""
import json
import os

import numpy as np
import pandas as pd

from .config import GenConfig
from .ground_truth import true_q, win_prob, _sigmoid

SECTORS = ["Financials", "Energy", "Utilities", "Telecom", "Healthcare",
           "Technology", "ConsumerStaples", "ConsumerDisc", "Industrials",
           "Materials", "REITs"]
RATINGS = ["AAA", "AA", "A+", "A-", "BBB+", "BBB-"]
RATING_P = [0.03, 0.10, 0.17, 0.25, 0.28, 0.17]
CLIENT_TYPES = ["AssetManager", "HedgeFund", "Insurer", "Pension", "Bank", "PrivateBank"]
REGIONS = ["AMER", "EMEA", "APAC"]
SIZE_EDGES = [0.0, 0.25, 1.0, 5.0, np.inf]
SIZE_LABELS = ["micro", "round", "block", "mega"]
B36 = np.array(list("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"))  # no I/O, CUSIP-style


# --------------------------------------------------------------------------
# universe builders
# --------------------------------------------------------------------------
def _cusip_check_digit(body8: str) -> str:
    total = 0
    for i, ch in enumerate(body8):
        v = int(ch) if ch.isdigit() else ord(ch) - ord("A") + 10
        if i % 2 == 1:
            v *= 2
        total += v // 10 + v % 10
    return str((10 - total % 10) % 10)


def build_bonds(cfg: GenConfig, rng) -> pd.DataFrame:
    nb, ni = cfg.n_bonds, cfg.n_issuers
    issuer_sector = rng.integers(0, cfg.n_sectors, ni)
    issuer_sz = rng.pareto(1.1, ni) + 1.0
    issuer_of = rng.choice(ni, nb, p=issuer_sz / issuer_sz.sum())
    sector = issuer_sector[issuer_of]

    rating = rng.choice(len(RATINGS), nb, p=RATING_P)
    maturity = np.clip(rng.gamma(2.2, 4.5, nb), 1.0, 30.0)
    coupon = np.clip(2.0 + 0.45 * rating + 0.09 * maturity + rng.normal(0, 0.5, nb), 0.5, 9.0)
    age_days = np.clip(rng.exponential(900, nb), 5, 9000)
    amt_out = np.clip(rng.lognormal(6.2, 0.75, nb), 100, 6000)  # $mm
    index_member = ((amt_out >= 300) & (maturity >= 1.0)
                    & (rng.random(nb) < 0.85)).astype(np.int8)

    z = (0.90 * np.log(amt_out / 100) - 0.55 * np.log1p(age_days / 365)
         + 0.85 * index_member - 0.18 * rating + rng.normal(0, 0.55, nb))
    liq = pd.Series(z).rank(pct=True).values  # liquidity percentile in (0,1]

    # CUSIPs: sector letter + issuer body (shared within issuer) + issue + check
    iss_body = np.array(["".join(rng.choice(B36, 5)) for _ in range(ni)])
    sec_char = np.array([chr(ord("A") + s) for s in issuer_sector])
    issue_no = np.zeros(nb, dtype=int)
    order = np.argsort(issuer_of, kind="stable")
    counts = np.zeros(ni, dtype=int)
    for idx in order:
        issue_no[idx] = counts[issuer_of[idx]]
        counts[issuer_of[idx]] += 1
    cusips = []
    for b in range(nb):
        body = sec_char[issuer_of[b]] + iss_body[issuer_of[b]] \
            + B36[(issue_no[b] // 34) % 34] + B36[issue_no[b] % 34]
        cusips.append(body + _cusip_check_digit(body))

    dur = np.clip(maturity * (1 - np.exp(-0.07 * maturity)) / (0.07 * maturity + 1e-9) * 0.8,
                  0.8, 18.0)
    return pd.DataFrame({
        "cusip": cusips, "issuer_id": issuer_of, "sector": [SECTORS[s] for s in sector],
        "sector_id": sector.astype(np.int8), "rating": [RATINGS[r] for r in rating],
        "rating_id": rating.astype(np.int8), "coupon": coupon.round(3),
        "maturity_yrs": maturity.round(2), "duration": dur.round(2),
        "age_days": age_days.round(0), "amt_out_mm": amt_out.round(1),
        "index_member": index_member, "liq_score": liq.round(4),
    })


def build_clients(cfg: GenConfig, rng):
    nc = cfg.n_clients
    tier = rng.choice(5, nc, p=cfg.tier_probs)  # hidden, 0..4
    sharp_mean = np.array(cfg.tier_sharpness_bps)[tier]
    sharpness = rng.gamma(3.0, sharp_mean / 3.0)          # hidden, bps
    side_bias = rng.normal(0, 0.5, nc)                    # hidden-ish
    act_mult = np.array([0.7, 0.9, 1.1, 1.5, 2.5])[tier]
    activity = rng.lognormal(0, 1.1, nc) * act_mult       # hidden weights
    ctype_p = np.array([
        [0.42, 0.02, 0.22, 0.18, 0.10, 0.06],
        [0.40, 0.05, 0.22, 0.15, 0.11, 0.07],
        [0.38, 0.12, 0.18, 0.12, 0.13, 0.07],
        [0.30, 0.30, 0.10, 0.06, 0.18, 0.06],
        [0.15, 0.62, 0.04, 0.02, 0.15, 0.02]])
    ctype = np.array([rng.choice(6, p=ctype_p[t]) for t in tier])
    coarse = np.array([0, 0, 1, 1, 2])[tier]
    flip = rng.random(nc) < cfg.marketing_tier_noise
    mkt_tier = np.clip(coarse + flip * rng.choice([-1, 1], nc), 0, 2)
    obs = pd.DataFrame({
        "client_id": np.arange(nc, dtype=np.int32),
        "client_type": [CLIENT_TYPES[c] for c in ctype],
        "region": rng.choice(REGIONS, nc, p=[0.55, 0.33, 0.12]),
        "aum_bucket": rng.choice(["<1bn", "1-10bn", "10-100bn", ">100bn"], nc,
                                 p=[0.25, 0.4, 0.27, 0.08]),
        "marketing_tier": (mkt_tier + 1).astype(np.int8),
    })
    hid = pd.DataFrame({"client_id": obs.client_id, "true_tier": (tier + 1).astype(np.int8),
                        "sharpness_bps": sharpness.round(4), "side_bias": side_bias.round(4),
                        "activity_w": activity.round(4)})
    return obs, hid


def build_market(cfg: GenConfig, rng):
    nd = cfg.n_days
    regime = np.zeros(nd, dtype=np.int8)
    for t in range(1, nd):
        p = 0.02 if regime[t - 1] == 0 else 0.92  # P(stressed_t=1 | prev)
        regime[t] = rng.random() < (p if regime[t - 1] else 0.02)
    def ar1(phi, sig):
        x = np.zeros(nd)
        for t in range(1, nd):
            x[t] = phi * x[t - 1] + rng.normal(0, sig)
        return x
    vix = 12 + 9 * regime + ar1(0.85, 1.0)
    cdx = 55 + 35 * regime + ar1(0.90, 2.5)
    dates = pd.bdate_range(end="2026-07-10", periods=nd)
    fac_sig = np.concatenate([np.full(6, 3.0), np.full(8, 4.0), [2.5]])  # daily bps
    F = rng.normal(0, 1, (nd, cfg.n_factors)) * fac_sig * (1 + 0.8 * regime[:, None])
    mkt = pd.DataFrame({"day": np.arange(nd, dtype=np.int16), "date": dates,
                        "dow": dates.dayofweek.astype(np.int8),
                        "vix_like": vix.round(3), "cdx_like": cdx.round(3)})
    return mkt, regime, F


def bond_loadings(cfg: GenConfig, bonds: pd.DataFrame, rng):
    nb = len(bonds)
    B = np.zeros((nb, cfg.n_factors), dtype=np.float32)
    kr = np.clip((bonds.duration.values / 3.0).astype(int), 0, 5)
    B[np.arange(nb), kr] = 0.8 + 0.15 * rng.random(nb)
    B[np.arange(nb), np.minimum(kr + 1, 5)] += 0.35
    sec8 = bonds.sector_id.values % 8
    B[np.arange(nb), 6 + sec8] = 0.6 + 0.5 * (1 - bonds.liq_score.values)
    B[:, 14] = 0.5 + 0.9 * (1 - bonds.liq_score.values)  # market/DTS beta
    return B


# --------------------------------------------------------------------------
# RFQ assembly and sufficient statistics
# --------------------------------------------------------------------------
def assemble_rfqs(cfg, n, rng, bonds, cli_obs, cli_hid, mkt, regime):
    nb = len(bonds)
    bw = np.exp(7.0 * bonds.liq_score.values) * (1 + 4 * np.exp(-bonds.age_days.values / 90))
    bond_ix = rng.choice(nb, n, p=bw / bw.sum())
    cw = cli_hid.activity_w.values
    client = rng.choice(cfg.n_clients, n, p=cw / cw.sum()).astype(np.int32)

    day_w = 1 + 0.35 * regime
    day = rng.choice(cfg.n_days, n, p=day_w / day_w.sum()).astype(np.int16)
    minute = (390 * rng.beta(0.7, 0.7, n)).astype(np.int16)

    size = np.clip(rng.gamma(1.8, 2.0 / 1.8, n), 0.1, 25.0)
    n_dealers = rng.choice(np.arange(2, 9), n, p=[0.08, 0.22, 0.28, 0.20, 0.12, 0.07, 0.03])
    buy = (rng.random(n) < _sigmoid(cli_hid.side_bias.values[client])).astype(np.int8)

    sec = bonds.sector_id.values[bond_ix]
    u = rng.random(n)
    trader = np.where(u < 0.85, sec, np.where(u < 0.95, 11, rng.integers(0, 11, n)))
    trader = trader.astype(np.int8)

    own_pos = np.where(rng.random(n) < 0.15, rng.normal(0, 1.5, n), 0.0)  # $mm
    sector_dts = np.zeros(n, dtype=np.float32)
    key = trader.astype(int) * cfg.n_days + day.astype(int)
    td_expo = rng.normal(0, 1.6, cfg.n_traders * cfg.n_days)
    sector_dts = td_expo[key].astype(np.float32)

    df = pd.DataFrame({
        "bond_ix": bond_ix.astype(np.int32), "client_id": client, "day": day,
        "minute": minute, "size_mm": size.astype(np.float32),
        "n_dealers": n_dealers.astype(np.int8), "client_buys": buy,
        "trader_id": trader, "own_pos_mm": own_pos.astype(np.float32),
        "sector_dts": sector_dts,
    })
    df["size_bucket"] = pd.cut(df.size_mm, SIZE_EDGES, labels=SIZE_LABELS)
    return df


def sufficient_stats(cfg, rf, bonds, cli_hid, regime):
    liq = bonds.liq_score.values[rf.bond_ix.values]
    ill = 1.0 - liq
    idx_mem = bonds.index_member.values[rf.bond_ix.values]
    tier = cli_hid.true_tier.values[rf.client_id.values].astype(int)      # 1..5
    tau = cli_hid.sharpness_bps.values[rf.client_id.values]
    stressed = regime[rf.day.values].astype(float)
    z = rf.size_mm.values.astype(float)
    nd = rf.n_dealers.values.astype(float)

    mu_delta = tau * (0.55 + 0.45 * np.log1p(z)) * (1 + 0.6 * stressed)

    m_c = (3.5 + 50.0 * ill ** 1.8
           + (1.2 + 2.8 * ill) * np.log1p(z)
           - 1.8 * np.log(nd - 0.5) - 1.5 * (nd >= 6)
           + cfg.kappa_level * mu_delta
           + stressed * (2.5 + 4.5 * ill)
           + 4.0 * ((tier >= 4) & (stressed > 0))
           - 1.5 * ((tier == 1) & (idx_mem == 1))
           + 2.5 * ((z > 5) & (ill > 0.5)))
    m_c = np.clip(m_c, 1.0, 110.0)
    s_c = np.clip(0.8 + 3.4 * ill ** 1.3 + 0.9 * stressed + 0.30 * np.log1p(z), 0.6, 8.0)
    return m_c, s_c, mu_delta, liq, stressed, tier


# --------------------------------------------------------------------------
# logging policy
# --------------------------------------------------------------------------
def heuristic_quote(cfg, rf, bonds, cli_obs, mkt, rng):
    """Trader base quote from OBSERVABLES only (no hidden tier/regime/tau)."""
    liq = bonds.liq_score.values[rf.bond_ix.values]
    ill = 1.0 - liq
    z = rf.size_mm.values.astype(float)
    nd = rf.n_dealers.values.astype(float)
    mt = cli_obs.marketing_tier.values[rf.client_id.values].astype(int) - 1  # 0..2
    vix = mkt.vix_like.values[rf.day.values]
    dow = mkt.dow.values[rf.day.values]
    stress_read = np.clip((vix - 12.0) / 9.0, 0.0, 1.4)  # noisy regime proxy

    h = (3.5 + 50.0 * ill ** 1.8
         + (1.2 + 2.8 * ill) * np.log1p(z)
         - 1.8 * np.log(nd - 0.5)
         + np.array([0.0, 1.0, 2.5])[mt]
         + stress_read * (2.5 + 4.5 * ill))
    # premium is charged in *dispersion units*, steeper on illiquids -- this
    # produces the realistic hit ladder (liquid ~12-15% -> illiquid ~1-2%).
    # k0 multiplying prem_unit is calibrated so the overall hit rate = 5%.
    s_hat = 0.8 + 3.4 * ill ** 1.3 + 0.30 * np.log1p(z)  # observable proxy of s_c
    prem_unit = s_hat * (1.0 + 1.1 * ill)

    # inventory skew (Gueant signature) -- trader 7 (index 6) is inventory-blind
    expo = 0.6 * np.sign(rf.own_pos_mm.values) * np.minimum(np.abs(rf.own_pos_mm.values), 3) / 3 \
        + 0.4 * np.tanh(rf.sector_dts.values / 2.0)
    ss = np.where(rf.client_buys.values == 1, 1.0, -1.0)
    rr = expo * ss > 0
    skew = np.where(rr, -1.2, 1.0) * np.abs(expo)
    skew[rf.trader_id.values == 6] = 0.0

    tr_off = rng.normal(0, 0.6, cfg.n_traders)
    h = h + skew + tr_off[rf.trader_id.values]
    h = h + 2.0 * ((rf.trader_id.values == 2) & (dow == 4))      # Friday widener
    h = h + rng.normal(0, cfg.heuristic_noise_bps, len(rf))
    return h, prem_unit, tr_off


def _logit_bonus(cfg):
    a = np.arange(1, cfg.n_actions + 1)
    return (cfg.stick_bonus_1bp * (a % 4 == 0)
            + cfg.stick_bonus_5bp * (a % 20 == 0)).astype(np.float64)


def logging_pmf(center_bps, cfg):
    """Reconstruct exact pi0 rows from stored centers. center: (n,) -> (n, A)."""
    g = cfg.grid_bps
    c = np.asarray(center_bps, dtype=np.float64)[:, None]
    logit = -0.5 * ((g[None, :] - c) / cfg.sigma_quote_bps) ** 2 + _logit_bonus(cfg)[None, :]
    logit[np.abs(g[None, :] - c) > cfg.support_halfwidth_ticks * cfg.tick_bps] = -np.inf
    logit -= logit.max(axis=1, keepdims=True)
    p = np.exp(logit)
    return p / p.sum(axis=1, keepdims=True)


def calibrate_offset(cfg, h_raw, prem, oracle_d, m_c, s_c, rng, pilot=60_000):
    from scipy.optimize import brentq
    n = min(pilot, len(h_raw))
    sl = slice(0, n)
    g = cfg.grid_bps

    def hit(k0):
        c = np.clip((1 - cfg.logging_quality) * (h_raw[sl] + k0 * prem[sl])
                    + cfg.logging_quality * oracle_d[sl], g[0], g[-1])
        acc, step = 0.0, 20_000
        for lo in range(0, n, step):
            hi = min(lo + step, n)
            p = logging_pmf(c[lo:hi], cfg)
            pw = win_prob(m_c[sl][lo:hi], s_c[sl][lo:hi], g)
            acc += float((p * pw).sum())
        return acc / n - cfg.target_hit_rate

    return brentq(hit, 0.1, 12.0, xtol=1e-3)


# --------------------------------------------------------------------------
# main per-split pipeline
# --------------------------------------------------------------------------
def run_split(cfg, n, rng, universe, split, offset=None, make_cf=True):
    bonds, cli_obs, cli_hid, mkt, regime, F, B = universe
    g = cfg.grid_bps
    rf = assemble_rfqs(cfg, n, rng, bonds, cli_obs, cli_hid, mkt, regime)
    m_c, s_c, mu_d, liq, stressed, tier = sufficient_stats(cfg, rf, bonds, cli_hid, regime)
    a1 = np.full(n, cfg.kappa_toxic)

    # latent draws (once per RFQ -> coherent counterfactual doubles)
    u = rng.random(n)
    dbar = m_c + s_c * np.log(u / (1 - u))  # cover ~ Logistic(m_c, s_c)
    fac_move = (B[rf.bond_ix.values] * F[rf.day.values]).sum(axis=1) * 0.277  # 30-min scale
    eps = fac_move + rng.normal(0, 0.8 * (1 + 1.0 * stressed), n)
    delta_mkout = mu_d + cfg.kappa_toxic * (dbar - m_c) + eps

    # oracle (needed for logging-quality mix), chunked
    oracle_a = np.empty(n, dtype=np.int32)
    for lo in range(0, n, cfg.chunk):
        hi = min(lo + cfg.chunk, n)
        q = true_q(m_c[lo:hi], s_c[lo:hi], mu_d[lo:hi], a1[lo:hi], g, cfg.cost_bps)
        oracle_a[lo:hi] = np.argmax(q, axis=1)
    oracle_d = g[oracle_a]

    h_raw, prem, tr_off = heuristic_quote(cfg, rf, bonds, cli_obs, mkt, rng)
    if offset is None:
        offset = calibrate_offset(cfg, h_raw, prem, oracle_d, m_c, s_c, rng)
    center = np.clip((1 - cfg.logging_quality) * (h_raw + offset * prem)
                     + cfg.logging_quality * oracle_d, g[0], g[-1])

    # sample actions + pscores + counterfactuals, chunked
    act = np.empty(n, dtype=np.int32)
    pscore = np.empty(n, dtype=np.float64)
    tq_logged = np.empty(n, dtype=np.float32)
    oq = np.empty(n, dtype=np.float32)
    K = cfg.k_counterfactuals
    cf_a = np.empty((n, K), dtype=np.int32) if make_cf else None
    cf_p = np.empty((n, K), dtype=np.float32) if make_cf else None
    cf_q = np.empty((n, K), dtype=np.float32) if make_cf else None
    qtl = np.linspace(0.06, 0.97, K)
    for lo in range(0, n, cfg.chunk):
        hi = min(lo + cfg.chunk, n)
        p = logging_pmf(center[lo:hi], cfg)
        gum = -np.log(-np.log(rng.random(p.shape)))
        with np.errstate(divide="ignore"):
            a = np.argmax(np.log(p) + gum, axis=1)
        act[lo:hi] = a
        rows = np.arange(hi - lo)
        pscore[lo:hi] = p[rows, a]
        q = true_q(m_c[lo:hi], s_c[lo:hi], mu_d[lo:hi], a1[lo:hi], g, cfg.cost_bps)
        tq_logged[lo:hi] = q[rows, a]
        oq[lo:hi] = q[rows, oracle_a[lo:hi]]
        if make_cf:
            wlo = np.clip(np.searchsorted(g, center[lo:hi] - cfg.support_halfwidth_ticks
                                          * cfg.tick_bps), 0, cfg.n_actions - 1)
            whi = np.clip(np.searchsorted(g, center[lo:hi] + cfg.support_halfwidth_ticks
                                          * cfg.tick_bps) - 1, 0, cfg.n_actions - 1)
            span = (whi - wlo).astype(np.float64)
            alts = (wlo[:, None] + np.round(qtl[None, :] * span[:, None])).astype(np.int32)
            clash = alts == a[:, None]
            alts[clash] = np.clip(alts[clash] + 1, 0, cfg.n_actions - 1)
            cf_a[lo:hi] = alts
            cf_p[lo:hi] = p[rows[:, None], alts]
            cf_q[lo:hi] = q[rows[:, None], alts]

    spread = g[act]
    win = (spread < dbar)
    reward = np.where(win, spread - delta_mkout - cfg.cost_bps, 0.0)
    if cfg.binary_reward:
        reward = win.astype(float)

    dates = mkt.date.values[rf.day.values]
    ev = pd.DataFrame({
        "rfq_id": np.arange(n, dtype=np.int64) if split == "train"
        else np.arange(n, dtype=np.int64) + 10_000_000,
        "date": dates, "day": rf.day.values, "minute": rf.minute.values,
        "dow": mkt.dow.values[rf.day.values], "trader_id": rf.trader_id.values,
        "client_id": rf.client_id.values,
        "cusip": bonds.cusip.values[rf.bond_ix.values],
        "sector": bonds.sector.values[rf.bond_ix.values],
        "rating": bonds.rating.values[rf.bond_ix.values],
        "liq_score": liq.astype(np.float32),
        "age_days": bonds.age_days.values[rf.bond_ix.values].astype(np.float32),
        "index_member": bonds.index_member.values[rf.bond_ix.values],
        "duration": bonds.duration.values[rf.bond_ix.values].astype(np.float32),
        "side": np.where(rf.client_buys.values == 1, "CLIENT-BUY", "CLIENT-SELL"),
        "size_mm": rf.size_mm.values, "size_bucket": rf.size_bucket.values,
        "n_dealers": rf.n_dealers.values,
        "own_pos_mm": rf.own_pos_mm.values, "sector_dts": rf.sector_dts.values,
        "vix_like": mkt.vix_like.values[rf.day.values].astype(np.float32),
        "cdx_like": mkt.cdx_like.values[rf.day.values].astype(np.float32),
        "action": act, "spread_bps": spread.astype(np.float32),
        "pscore": pscore,
        "outcome": np.where(win, "CLIENT-TRADED", "CLIENT-TRADED-AWAY"),
        "reward_bps": reward.astype(np.float32),
        "markout_bps": delta_mkout.astype(np.float32),
    })
    if cfg.reveal_cover_on_win:
        ev["cover_bps"] = np.where(win, dbar, np.nan).astype(np.float32)

    truth = pd.DataFrame({
        "rfq_id": ev.rfq_id, "m_c": m_c.astype(np.float32),
        "s_c": s_c.astype(np.float32), "mu_delta": mu_d.astype(np.float32),
        "a1": a1.astype(np.float32), "center_bps": center.astype(np.float32),
        "true_q_logged": tq_logged, "oracle_action": oracle_a,
        "oracle_q": oq,
    })
    hidden = pd.DataFrame({
        "rfq_id": ev.rfq_id,
        "client_tier": cli_hid.true_tier.values[rf.client_id.values],
        "sharpness_bps": cli_hid.sharpness_bps.values[rf.client_id.values].astype(np.float32),
        "regime_stressed": stressed.astype(np.int8),
        "cover_dbar_bps": dbar.astype(np.float32),
        "markout_eps_bps": eps.astype(np.float32),
        "factor_move_bps": fac_move.astype(np.float32),
    })

    cf = None
    if make_cf:
        rid = np.repeat(ev.rfq_id.values, K)
        af = cf_a.ravel()
        sp = g[af].astype(np.float32)
        winf = sp < np.repeat(dbar, K)
        rew = np.where(winf, sp - np.repeat(delta_mkout, K) - cfg.cost_bps, 0.0)
        cf = pd.DataFrame({
            "rfq_id": rid, "alt_action": af, "alt_spread_bps": sp,
            "alt_pscore": cf_p.ravel(),
            "alt_outcome": np.where(winf, "CLIENT-TRADED", "CLIENT-TRADED-AWAY"),
            "alt_reward_bps": rew.astype(np.float32),
            "true_q_alt": cf_q.ravel(),
        })
    return ev, truth, hidden, cf, offset, tr_off


def build_action_space(cfg, rng):
    a = np.arange(cfg.n_actions)
    band = cfg.band_of_action
    df = pd.DataFrame({"action": a, "spread_bps": cfg.grid_bps.astype(np.float32),
                       "band_true": band})
    for lvl in cfg.cluster_noise_levels:
        b = band.copy()
        flip = rng.random(cfg.n_actions) < lvl
        b[flip] = rng.integers(0, cfg.n_bands, flip.sum())
        df[f"band_noisy_{int(lvl*100)}"] = b
    df["band_coarse10"] = (a // (cfg.n_actions // 10)).astype(np.int16)
    df["band_fine50"] = (a // (cfg.n_actions // 50)).astype(np.int16)
    return df


def generate(cfg: GenConfig = None):
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(cfg.seed)
    od = cfg.out_dir
    for d in ["", "evaluation", "answers"]:
        os.makedirs(os.path.join(od, d), exist_ok=True)

    bonds = build_bonds(cfg, rng)
    cli_obs, cli_hid = build_clients(cfg, rng)
    mkt, regime, F = build_market(cfg, rng)
    B = bond_loadings(cfg, bonds, rng)
    universe = (bonds, cli_obs, cli_hid, mkt, regime, F, B)

    ev_tr, tru_tr, hid_tr, cf_tr, offset, tr_off = run_split(
        cfg, cfg.n_train, rng, universe, "train", None, make_cf=True)
    ev_te, tru_te, hid_te, _, _, _ = run_split(
        cfg, cfg.n_test, rng, universe, "test", offset, make_cf=False)

    kw = dict(index=False, compression="zstd")
    ev_tr.to_parquet(f"{od}/rfq_events.parquet", **kw)
    ev_te.to_parquet(f"{od}/rfq_events_test.parquet", **kw)
    cf_tr.to_parquet(f"{od}/rfq_counterfactuals.parquet", **kw)
    build_action_space(cfg, rng).to_parquet(f"{od}/action_space.parquet", **kw)
    bonds.to_parquet(f"{od}/bonds.parquet", **kw)
    cli_obs.to_parquet(f"{od}/clients.parquet", **kw)
    mkt.to_parquet(f"{od}/market_daily.parquet", **kw)
    tru_tr.to_parquet(f"{od}/evaluation/train_truth.parquet", **kw)
    tru_te.to_parquet(f"{od}/evaluation/test_truth.parquet", **kw)
    hid_tr.to_parquet(f"{od}/answers/hidden_train.parquet", **kw)
    hid_te.to_parquet(f"{od}/answers/hidden_test.parquet", **kw)
    cli_hid.to_parquet(f"{od}/answers/clients_hidden.parquet", **kw)
    pd.DataFrame({"day": mkt.day, "regime_stressed": regime}).to_parquet(
        f"{od}/answers/regime.parquet", **kw)
    pd.DataFrame({"trader_id": range(cfg.n_traders), "base_offset_bps": tr_off,
                  "inventory_blind": [i == 6 for i in range(cfg.n_traders)],
                  "friday_widener": [i == 2 for i in range(cfg.n_traders)],
                  }).to_csv(f"{od}/answers/trader_bias.csv", index=False)
    with open(f"{od}/answers/config_dump.json", "w") as f:
        json.dump({**cfg.__dict__, "calibrated_k0": float(offset)}, f, indent=2)
    return offset


if __name__ == "__main__":
    off = generate()
    print(f"done; calibrated offset = {off:.3f} bps")
