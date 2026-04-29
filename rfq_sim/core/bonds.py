"""
bonds.py
--------
Bond universe initialisation.

I build N bonds grouped under n_issuers issuers (~3 bonds each spanning
Short / Medium / Long duration).  Each bond gets:

  Observable:   issuer_id, sector, rating, duration_bucket, liquidity_tier
  Latent:       v_n ∈ ℝ³  (sector / duration-quality / liquidity loadings)
  Price params: β_n (factor loadings), κ_n (MMPP drift sensitivity), σ_n (idio vol)
  MMPP share:   β^{n,bid}, β^{n,ask}  (Dirichlet share of sector flow)
  Structural:   Σ_nn' similarity matrix

The issuer grouping is the "issuer curve Easter egg": models trained on
client interaction data should eventually learn that same-issuer bonds
share trading patterns, even when the issuer label isn't a direct feature.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from collections import defaultdict
from typing import List, Dict
from rfq_sim.core.config import BondConfig


@dataclass
class Bond:
    bond_id:          int
    issuer_id:        int
    sector:           str
    rating:           str
    duration_bucket:  str
    liquidity_tier:   int    # 1=on-the-run, 2=active, 3=illiquid

    # Observable feature vector [issuer_idx, sector_idx, rating_idx, dur_idx, tier]
    x_obs: np.ndarray

    # Latent factor vector v_n ∈ ℝ³  — never exposed to models
    v_n: np.ndarray

    # Factor model loadings β_n ∈ ℝ^p (p=3: rates / HY-spread / sector)
    beta_factor: np.ndarray

    # MMPP micro-price drift sensitivity κ_n  (Bergault & Gueant Table 2 range)
    kappa: float

    # Base daily idio vol σ_n (price points / √trading-day)
    sigma: float

    # Baseline bid-ask spread (price points)
    baseline_spread: float

    # Outstanding notional ($MM) — caps RFQ size
    outstanding_mm: float

    # Initial mid price
    price0: float

    # Dirichlet share of sector MMPP flow (per side)
    beta_mmpp_bid: float = 0.0
    beta_mmpp_ask: float = 0.0


class BondUniverse:
    """
    Generates and holds the full bond universe.

    Construction order:
      1. Assign observable features (issuer, sector, rating, duration, tier)
      2. Draw latent factors v_n  correlated with but ≠ observables
      3. Draw price-process parameters
      4. Compute similarity matrix Σ_nn' from observables
      5. Draw Dirichlet MMPP β coefficients within each sector
    """

    def __init__(self, cfg: BondConfig, rng: np.random.Generator):
        self.cfg  = cfg
        self.rng  = rng
        self.bonds: List[Bond] = []
        self.similarity_matrix: np.ndarray = np.array([])
        self._build()

    # ------------------------------------------------------------------
    # Top-level builder
    # ------------------------------------------------------------------

    def _build(self):
        self._assign_features()
        self._draw_latent_factors()
        self._draw_price_params()
        self._compute_similarity()
        self._draw_mmpp_betas()

    # ------------------------------------------------------------------
    # Step 1: observable features
    # ------------------------------------------------------------------

    def _assign_features(self):
        """
        Assign each bond to an issuer, sector, rating, duration bucket,
        and liquidity tier.  Issuers have one bond per duration bucket so
        the same issuer appears at Short / Medium / Long — that's what
        creates the issuer curve structure.
        """
        cfg = self.cfg
        sectors   = cfg.sectors
        ratings   = cfg.ratings
        durations = cfg.duration_buckets

        # Each issuer maps to a fixed sector and rating
        issuer_sector = self.rng.choice(sectors, size=cfg.n_issuers, replace=True)
        issuer_rating = self.rng.choice(ratings, size=cfg.n_issuers, replace=True)

        bonds_per_issuer = cfg.n_bonds // cfg.n_issuers
        remainder        = cfg.n_bonds  - bonds_per_issuer * cfg.n_issuers

        bonds = []
        bond_id = 0

        for iid in range(cfg.n_issuers):
            n_this = bonds_per_issuer + (1 if iid < remainder else 0)

            # Prefer spreading across durations; fall back to random if n_this < 3
            if n_this >= len(durations):
                dur_draws = list(durations)[:n_this]
            else:
                dur_draws = self.rng.choice(durations, size=n_this,
                                            replace=False).tolist()

            for dur in dur_draws:
                # Liquidity tier: shorter maturities tend to be more liquid
                if dur == "Short":
                    tier_p = [0.60, 0.30, 0.10]
                elif dur == "Medium":
                    tier_p = [0.30, 0.50, 0.20]
                else:
                    tier_p = [0.10, 0.40, 0.50]
                tier = int(self.rng.choice(cfg.liquidity_tiers, p=tier_p))

                sec   = issuer_sector[iid]
                rat   = issuer_rating[iid]
                s_idx = sectors.index(sec)
                r_idx = ratings.index(rat)
                d_idx = durations.index(dur)

                x_obs = np.array([float(iid), float(s_idx), float(r_idx),
                                   float(d_idx), float(tier)], dtype=np.float32)

                # Rating discount from par: BB near par, CCC discounted
                rating_disc = {"BB": 0.0, "B": -5.0, "CCC": -15.0}[rat]
                price0 = float(
                    np.clip(
                        100.0 + rating_disc + self.rng.uniform(-5, 5),
                        cfg.initial_price_lo,
                        cfg.initial_price_hi,
                    )
                )

                outstanding = (
                    cfg.outstanding_by_tier[tier]
                    * float(0.5 + self.rng.random())
                )

                bonds.append(Bond(
                    bond_id=bond_id, issuer_id=int(iid),
                    sector=sec, rating=rat, duration_bucket=dur,
                    liquidity_tier=tier, x_obs=x_obs,
                    v_n=np.zeros(cfg.latent_dim, dtype=np.float32),
                    beta_factor=np.zeros(3, dtype=np.float32),
                    kappa=0.0, sigma=0.0,
                    baseline_spread=cfg.baseline_spread[tier],
                    outstanding_mm=outstanding,
                    price0=price0,
                ))
                bond_id += 1

        self.bonds = bonds  # Exact count guaranteed by construction

    # ------------------------------------------------------------------
    # Step 2: latent factors v_n
    # ------------------------------------------------------------------

    def _draw_latent_factors(self):
        """
        v_n is derived from observables (so models have a path to recover it)
        but has issuer-level and bond-level noise (so it can't be read off
        directly).  Bonds from the same issuer share an issuer-level offset,
        which plants the issuer-curve Easter egg in the latent space.
        """
        cfg = self.cfg
        sectors   = cfg.sectors
        durations = cfg.duration_buckets

        # Pre-draw one latent offset per issuer (shared across all its bonds)
        issuer_offsets: Dict[int, np.ndarray] = {}

        for bond in self.bonds:
            if bond.issuer_id not in issuer_offsets:
                issuer_offsets[bond.issuer_id] = self.rng.normal(
                    0.0, 0.15, size=cfg.latent_dim
                ).astype(np.float32)

            # Observable-derived mean for v_n
            s_pos = sectors.index(bond.sector)   / max(len(sectors)   - 1, 1)
            d_pos = durations.index(bond.duration_bucket) / max(len(durations) - 1, 1)
            t_pos = 1.0 - (bond.liquidity_tier - 1) / 2.0  # Tier 1 → high liquidity

            mean_v = np.array([s_pos, d_pos, t_pos], dtype=np.float32)

            bond_noise = self.rng.normal(0.0, 0.10, size=cfg.latent_dim).astype(np.float32)
            v_n = mean_v + issuer_offsets[bond.issuer_id] + bond_noise
            bond.v_n = np.clip(v_n, 0.01, None)

    # ------------------------------------------------------------------
    # Step 3: price-process parameters
    # ------------------------------------------------------------------

    def _draw_price_params(self):
        """
        β_n, κ_n, σ_n are drawn at initialisation and held fixed.
        Liquid bonds are more κ_n-sensitive (they respond to flow imbalance)
        and have lower σ_n (less idio vol).
        """
        cfg = self.cfg
        for bond in self.bonds:
            # Factor loadings: all bonds load on rates and HY spread;
            # sector loading varies
            rates_load  = float(self.rng.uniform(0.30, 0.90))
            spread_load = {"BB": 0.40, "B": 0.70, "CCC": 1.20}[bond.rating]
            spread_load += float(self.rng.normal(0, 0.15))
            sector_load = float(self.rng.uniform(0.10, 0.50))
            bond.beta_factor = np.array(
                [rates_load, max(0.01, spread_load), sector_load],
                dtype=np.float32,
            )

            # MMPP drift sensitivity: liquid bonds more responsive
            tier_scale = {1: 1.0, 2: 0.60, 3: 0.30}[bond.liquidity_tier]
            bond.kappa = float(
                cfg.kappa_lo + (cfg.kappa_hi - cfg.kappa_lo)
                * tier_scale * self.rng.random()
            )

            # Idio vol: illiquid and lower-rated bonds are jumpier
            r_scale = {"BB": 0.50, "B": 0.80, "CCC": 1.20}[bond.rating]
            t_scale = {1: 0.80, 2: 1.00, 3: 1.30}[bond.liquidity_tier]
            bond.sigma = float(np.clip(
                (cfg.idio_vol_lo + cfg.idio_vol_hi) / 2.0
                * r_scale * t_scale
                + self.rng.uniform(-0.05, 0.05),
                cfg.idio_vol_lo, cfg.idio_vol_hi,
            ))

    # ------------------------------------------------------------------
    # Step 4: similarity matrix
    # ------------------------------------------------------------------

    def _compute_similarity(self):
        """
        Σ_nn' = w_issuer·1[same issuer]
              + w_sector·1[same sector]
              + w_rating·1[same rating]
              + w_duration·1[same duration bucket]
        Diagonal = 1.

        This is the ground-truth bond–bond structure that a recommender
        model should learn to recover from co-occurrence patterns alone.
        """
        cfg = self.cfg
        N = len(self.bonds)
        S = np.zeros((N, N), dtype=np.float32)

        for i, bi in enumerate(self.bonds):
            for j, bj in enumerate(self.bonds):
                if i == j:
                    S[i, j] = 1.0
                    continue
                s = 0.0
                if bi.issuer_id       == bj.issuer_id:       s += cfg.sim_w_issuer
                if bi.sector          == bj.sector:           s += cfg.sim_w_sector
                if bi.rating          == bj.rating:           s += cfg.sim_w_rating
                if bi.duration_bucket == bj.duration_bucket:  s += cfg.sim_w_duration
                S[i, j] = s

        self.similarity_matrix = S

    # ------------------------------------------------------------------
    # Step 5: Dirichlet MMPP β coefficients
    # ------------------------------------------------------------------

    def _draw_mmpp_betas(self):
        """
        Within each sector, bonds share the sector-level MMPP flow.
        β^{n,bid} is bond n's share of the sector bid intensity; similarly for ask.
        I draw from a Dirichlet so the shares sum to 1.
        Liquid (tier-1) bonds get a larger Dirichlet concentration parameter.
        """
        sector_groups: Dict[str, List[Bond]] = defaultdict(list)
        for b in self.bonds:
            sector_groups[b.sector].append(b)

        for bonds_s in sector_groups.values():
            conc = np.array([
                {1: 3.0, 2: 1.5, 3: 0.5}[b.liquidity_tier]
                for b in bonds_s
            ])
            betas_bid = self.rng.dirichlet(conc).astype(np.float64)
            betas_ask = self.rng.dirichlet(conc).astype(np.float64)
            for b, bb, ba in zip(bonds_s, betas_bid, betas_ask):
                b.beta_mmpp_bid = float(bb)
                b.beta_mmpp_ask = float(ba)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.bonds)

    def __getitem__(self, idx: int) -> Bond:
        return self.bonds[idx]

    def to_dataframe(self) -> pd.DataFrame:
        """Observable bond metadata — what models are allowed to see."""
        return pd.DataFrame([{
            "bond_id":         b.bond_id,
            "issuer_id":       b.issuer_id,
            "sector":          b.sector,
            "rating":          b.rating,
            "duration_bucket": b.duration_bucket,
            "liquidity_tier":  b.liquidity_tier,
            "baseline_spread": b.baseline_spread,
            "outstanding_mm":  round(b.outstanding_mm, 2),
            "initial_price":   round(b.price0, 4),
        } for b in self.bonds])

    def to_ground_truth_dataframe(self) -> pd.DataFrame:
        """Full bond params including latent factors — evaluation only."""
        rows = []
        for b in self.bonds:
            row = {"bond_id": b.bond_id, "kappa": b.kappa, "sigma": b.sigma,
                   "beta_mmpp_bid": b.beta_mmpp_bid, "beta_mmpp_ask": b.beta_mmpp_ask}
            for d in range(self.cfg.latent_dim):
                row[f"v_{d}"] = float(b.v_n[d])
            for f in range(3):
                row[f"beta_factor_{f}"] = float(b.beta_factor[f])
            rows.append(row)
        return pd.DataFrame(rows)
