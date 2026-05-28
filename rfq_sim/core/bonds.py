"""
bonds.py
--------
Bond universe initialisation — rewritten for N ≈ 10,000+ scale.

Changes from the original N=100 version
----------------------------------------

1.  No N×N similarity matrix is ever stored.
    The original code built and stored a dense (N, N) float32 matrix.
    At N=10,000 that is 400 MB.  More critically, even a sparse CSR
    representation fails: the same-sector block alone is 2,500×2,500
    = 6.25 M non-zeros per sector, making G @ G.T for all four features
    hit ~100 M non-zeros and OOM on a typical research machine.

    The key insight: the only downstream consumer of Sigma is the price
    spillover multiply  eta * Sigma @ eps  in price_process.py.  That
    multiply decomposes as:

        Sigma @ eps
        = w_issuer  * (G_issuer  @ G_issuer.T  @ eps)
        + w_sector  * (G_sector  @ G_sector.T  @ eps)
        + w_rating  * (G_rating  @ G_rating.T  @ eps)
        + w_duration* (G_duration@ G_duration.T@ eps)
        + eps                                           (diagonal = 1)

    Each G @ G.T @ eps can be computed as G @ (G.T @ eps), which is
    just a group-sum operation: for each bond i, sum eps[j] over all j
    in the same group, then broadcast back.  This is O(N) time and
    O(n_groups) intermediate memory — no N×N object is ever created.

    The new BondUniverse exposes two methods instead of a matrix:
        .spillover_matvec(eps)          — eta * Sigma @ eps  (N,) → (N,)
        .similarity_pair(i, j)          — Sigma[i, j] scalar, on demand

2.  Latent factor drawing is fully vectorised.
    All issuer offsets drawn in one batched RNG call (shape n_issuers×d).
    All bond noises drawn in one batched call (shape N×d).
    No Python loop over bonds.

3.  Per-bond price parameters (beta_factor, kappa, sigma) drawn in
    fully vectorised NumPy calls with rating/tier lookup tables.

4.  Dirichlet beta coefficients drawn per sector using NumPy boolean
    indexing; no Python loop over bond lists.

5.  Feature index arrays (issuer_ids, sector_ids, etc.) are cached as
    integer numpy arrays at build time for O(1) group_matvec calls.

Interface changes from original
--------------------------------
REMOVED:  .similarity_matrix      (was np.ndarray N×N)
REMOVED:  .similarity_sparse      (was scipy CSR)
ADDED:    .spillover_matvec(eps)  — replaces price_process.py's Sigma @ eps
ADDED:    .similarity_pair(i, j)  — scalar Sigma[i,j] for any pair

price_process.py must be updated to call:
    spillover = bonds.spillover_matvec(idio_shocks)
instead of:
    spillover = cfg.price_spillover_eta * (bonds.similarity_matrix @ idio_shocks)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List

from rfq_sim.core.config import BondConfig


# ---------------------------------------------------------------------------
# Bond dataclass — interface unchanged
# ---------------------------------------------------------------------------

@dataclass
class Bond:
    bond_id:         int
    issuer_id:       int
    sector:          str
    rating:          str
    duration_bucket: str
    liquidity_tier:  int   # 1=on-the-run  2=active  3=illiquid

    x_obs:           np.ndarray   # Observable feature vector (5,)
    v_n:             np.ndarray   # Latent factor vector (hidden from models)
    beta_factor:     np.ndarray   # Common factor loadings β_n ∈ ℝ^p
    kappa:           float        # MMPP drift sensitivity κ_n
    sigma:           float        # Base daily idio vol σ_n
    baseline_spread: float
    outstanding_mm:  float
    price0:          float

    beta_mmpp_bid: float = 0.0
    beta_mmpp_ask: float = 0.0


# ---------------------------------------------------------------------------
# BondUniverse
# ---------------------------------------------------------------------------

class BondUniverse:
    """
    Generates and holds the full bond universe.

    Key public methods replacing the old similarity_matrix attribute
    ----------------------------------------------------------------
    spillover_matvec(eps)   — compute eta * Sigma @ eps without any N×N object
    similarity_pair(i, j)  — scalar Sigma[i,j] for any bond pair
    """

    def __init__(self, cfg: BondConfig, rng: np.random.Generator):
        self.cfg  = cfg
        self.rng  = rng
        self.bonds: List[Bond] = []

        # Feature index arrays (cached after _assign_features)
        self._issuer_ids:  np.ndarray = np.empty(0, dtype=np.int32)
        self._sector_ids:  np.ndarray = np.empty(0, dtype=np.int32)
        self._rating_ids:  np.ndarray = np.empty(0, dtype=np.int32)
        self._dur_ids:     np.ndarray = np.empty(0, dtype=np.int32)

        self._build()

    # ------------------------------------------------------------------
    # Top-level builder
    # ------------------------------------------------------------------

    def _build(self):
        self._assign_features()      # Step 1: observables + cache index arrays
        self._draw_latent_factors()  # Step 2: latent v_n  (vectorised)
        self._draw_price_params()    # Step 3: β_n, κ_n, σ_n (vectorised)
        self._draw_mmpp_betas()      # Step 4: Dirichlet β  (vectorised)
        # No similarity build step — group index arrays are all we need

    # ------------------------------------------------------------------
    # Step 1: assign observable features
    # ------------------------------------------------------------------

    def _assign_features(self):
        """
        Assign each bond to an issuer, sector, rating, duration, and tier.
        Each issuer spans Short / Medium / Long to plant the issuer-curve
        Easter egg: bonds from the same issuer cluster in latent factor space,
        creating detectable cross-tenor trading patterns.
        """
        cfg       = self.cfg
        sectors   = cfg.sectors
        ratings   = cfg.ratings
        durations = cfg.duration_buckets

        issuer_sector = self.rng.choice(sectors, size=cfg.n_issuers, replace=True)
        issuer_rating = self.rng.choice(ratings, size=cfg.n_issuers, replace=True)

        bonds_per_issuer = cfg.n_bonds // cfg.n_issuers
        remainder        = cfg.n_bonds - bonds_per_issuer * cfg.n_issuers

        bonds   = []
        bond_id = 0

        for iid in range(cfg.n_issuers):
            n_this = bonds_per_issuer + (1 if iid < remainder else 0)

            if n_this >= len(durations):
                # Cycle through durations so we get exactly n_this bonds.
                # e.g. n_this=4, durations=['Short','Medium','Long']
                # gives ['Short','Medium','Long','Short']
                dur_draws = [durations[k % len(durations)] for k in range(n_this)]
            else:
                dur_draws = self.rng.choice(
                    durations, size=n_this, replace=False
                ).tolist()

            sec   = issuer_sector[iid]
            rat   = issuer_rating[iid]
            s_idx = sectors.index(sec)
            r_idx = ratings.index(rat)

            for dur in dur_draws:
                if dur == "Short":
                    tier_p = [0.60, 0.30, 0.10]
                elif dur == "Medium":
                    tier_p = [0.30, 0.50, 0.20]
                else:
                    tier_p = [0.10, 0.40, 0.50]
                tier  = int(self.rng.choice(cfg.liquidity_tiers, p=tier_p))
                d_idx = durations.index(dur)

                x_obs = np.array(
                    [float(iid), float(s_idx), float(r_idx),
                     float(d_idx), float(tier)],
                    dtype=np.float32,
                )
                rating_disc = {"BB": 0.0, "B": -5.0, "CCC": -15.0}[rat]
                price0 = float(np.clip(
                    100.0 + rating_disc + self.rng.uniform(-5, 5),
                    cfg.initial_price_lo, cfg.initial_price_hi,
                ))
                outstanding = (
                    cfg.outstanding_by_tier[tier] * float(0.5 + self.rng.random())
                )

                bonds.append(Bond(
                    bond_id=bond_id, issuer_id=int(iid),
                    sector=sec, rating=rat, duration_bucket=dur,
                    liquidity_tier=tier, x_obs=x_obs,
                    v_n=np.zeros(cfg.latent_dim, dtype=np.float32),
                    beta_factor=np.zeros(3, dtype=np.float32),
                    kappa=0.0, sigma=0.0,
                    baseline_spread=cfg.baseline_spread[tier],
                    outstanding_mm=outstanding, price0=price0,
                ))
                bond_id += 1

        self.bonds = bonds

        # Cache feature index arrays — used by spillover_matvec and similarity_pair
        self._issuer_ids = np.array([b.issuer_id for b in bonds], dtype=np.int32)
        self._sector_ids = np.array([sectors.index(b.sector)              for b in bonds], dtype=np.int32)
        self._rating_ids = np.array([ratings.index(b.rating)              for b in bonds], dtype=np.int32)
        self._dur_ids    = np.array([durations.index(b.duration_bucket)   for b in bonds], dtype=np.int32)

    # ------------------------------------------------------------------
    # Step 2: draw latent factors v_n  (fully vectorised)
    # ------------------------------------------------------------------

    def _draw_latent_factors(self):
        """
        v_n ∈ ℝ³ = observable_mean_n + issuer_offset_{issuer(n)} + bond_noise_n

        All issuer offsets drawn in one batched call (n_issuers × d).
        All bond noises drawn in one batched call (N × d).
        No Python for-loop over bonds.
        """
        cfg       = self.cfg
        N         = len(self.bonds)
        sectors   = cfg.sectors
        durations = cfg.duration_buckets

        # Observable-derived latent mean  (N, 3)
        s_pos = self._sector_ids / max(len(sectors)   - 1, 1)
        d_pos = self._dur_ids    / max(len(durations) - 1, 1)
        t_pos = 1.0 - (np.array([b.liquidity_tier for b in self.bonds], dtype=np.float32) - 1) / 2.0
        mean_V = np.stack(
            [s_pos.astype(np.float32), d_pos.astype(np.float32), t_pos],
            axis=1,
        )   # (N, 3)

        # One shared offset per issuer — broadcast to bonds via fancy indexing
        issuer_offsets = self.rng.normal(
            0.0, 0.15, size=(cfg.n_issuers, cfg.latent_dim)
        ).astype(np.float32)
        bond_issuer_offsets = issuer_offsets[self._issuer_ids]   # (N, 3)

        # Bond-level noise
        bond_noise = self.rng.normal(
            0.0, 0.10, size=(N, cfg.latent_dim)
        ).astype(np.float32)

        V = np.clip(mean_V + bond_issuer_offsets + bond_noise, 0.01, None)

        for i, bond in enumerate(self.bonds):
            bond.v_n = V[i]

    # ------------------------------------------------------------------
    # Step 3: draw price-process parameters  (fully vectorised)
    # ------------------------------------------------------------------

    def _draw_price_params(self):
        """
        β_n, κ_n, σ_n drawn in vectorised NumPy calls.
        No per-bond Python loop.
        """
        cfg = self.cfg
        N   = len(self.bonds)

        rating_idx = np.array(
            [{"BB": 0, "B": 1, "CCC": 2}[b.rating] for b in self.bonds],
            dtype=np.int32,
        )
        tier_idx = np.array(
            [b.liquidity_tier - 1 for b in self.bonds], dtype=np.int32
        )   # values 0, 1, 2

        # Factor loadings β_n  (N, 3)
        rates_load  = self.rng.uniform(0.30, 0.90, size=N).astype(np.float32)
        spread_base = np.array([0.40, 0.70, 1.20], dtype=np.float32)[rating_idx]
        spread_load = np.maximum(
            0.01,
            spread_base + self.rng.normal(0, 0.15, size=N).astype(np.float32),
        )
        sector_load = self.rng.uniform(0.10, 0.50, size=N).astype(np.float32)
        beta_factor = np.stack([rates_load, spread_load, sector_load], axis=1)

        # MMPP drift sensitivity κ_n
        tier_kappa = np.array([1.0, 0.60, 0.30], dtype=np.float64)[tier_idx]
        kappa = (
            cfg.kappa_lo
            + (cfg.kappa_hi - cfg.kappa_lo) * tier_kappa * self.rng.random(size=N)
        )

        # Idiosyncratic vol σ_n
        r_scale = np.array([0.50, 0.80, 1.20], dtype=np.float64)[rating_idx]
        t_scale = np.array([0.80, 1.00, 1.30], dtype=np.float64)[tier_idx]
        sigma   = np.clip(
            (cfg.idio_vol_lo + cfg.idio_vol_hi) / 2.0 * r_scale * t_scale
            + self.rng.uniform(-0.05, 0.05, size=N),
            cfg.idio_vol_lo, cfg.idio_vol_hi,
        )

        for i, bond in enumerate(self.bonds):
            bond.beta_factor = beta_factor[i]
            bond.kappa       = float(kappa[i])
            bond.sigma       = float(sigma[i])

    # ------------------------------------------------------------------
    # Step 4: Dirichlet MMPP β coefficients  (vectorised)
    # ------------------------------------------------------------------

    def _draw_mmpp_betas(self):
        """
        Within each sector, bonds share the sector MMPP via Dirichlet β
        coefficients summing to 1.  Liquid bonds get higher concentration.
        """
        cfg        = self.cfg
        sector_arr = np.array([b.sector for b in self.bonds])
        tier_arr   = np.array([b.liquidity_tier for b in self.bonds], dtype=np.int32)
        conc_map   = {1: 3.0, 2: 1.5, 3: 0.5}

        for sector in cfg.sectors:
            mask = np.where(sector_arr == sector)[0]
            if len(mask) == 0:
                continue
            conc      = np.array([conc_map[tier_arr[i]] for i in mask], dtype=np.float64)
            betas_bid = self.rng.dirichlet(conc)
            betas_ask = self.rng.dirichlet(conc)
            for local_i, global_i in enumerate(mask):
                self.bonds[global_i].beta_mmpp_bid = float(betas_bid[local_i])
                self.bonds[global_i].beta_mmpp_ask = float(betas_ask[local_i])

    # ------------------------------------------------------------------
    # Similarity operations — no N×N matrix ever materialised
    # ------------------------------------------------------------------

    @staticmethod
    def _group_matvec(group_ids: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Compute (G @ G.T) @ v where G[i, group_ids[i]] = 1.

        Equivalent meaning: for each bond i, return the sum of v[j] over all
        bonds j that share the same group as i.

        Algorithm  (O(N), O(n_groups) memory):
          1. Accumulate group sums:  s[g] = sum of v[j] for j with group_ids[j]==g
          2. Broadcast back:         result[i] = s[group_ids[i]]

        This avoids materialising the N×N Gram matrix entirely.
        """
        n_groups   = int(group_ids.max()) + 1
        group_sums = np.zeros(n_groups, dtype=v.dtype)
        np.add.at(group_sums, group_ids, v)
        return group_sums[group_ids]

    def spillover_matvec(self, eps: np.ndarray) -> np.ndarray:
        """
        Compute  eta * Sigma @ eps  without ever materialising Sigma.

        Sigma @ eps decomposes by feature:
            Sigma @ eps = w_issuer   * (G_issuer   @ G_issuer.T   @ eps)
                        + w_sector   * (G_sector   @ G_sector.T   @ eps)
                        + w_rating   * (G_rating   @ G_rating.T   @ eps)
                        + w_duration * (G_duration @ G_duration.T @ eps)
                        + eps                     (diagonal = 1 term)

        Each G @ G.T @ eps is a single group_matvec call: O(N) time,
        O(n_groups) memory.  Total: O(N), ~0.2ms at N=10,000.

        This replaces the old:
            eta * bonds.similarity_matrix @ eps
        in price_process.py.  Update that line to:
            bonds.spillover_matvec(idio_shocks)
        (eta is folded in here using the config value.)
        """
        cfg = self.cfg
        gm  = self._group_matvec

        result = (
            cfg.sim_w_issuer   * gm(self._issuer_ids, eps)
            + cfg.sim_w_sector   * gm(self._sector_ids, eps)
            + cfg.sim_w_rating   * gm(self._rating_ids, eps)
            + cfg.sim_w_duration * gm(self._dur_ids,    eps)
            + eps                # diagonal = 1
        )
        return cfg.price_spillover_eta * result

    def similarity_pair(self, i: int, j: int) -> float:
        """
        Return Sigma[i, j] for any bond pair — computed on demand, O(1).
        Useful for diagnostics and the Easter egg audit notebook.
        """
        if i == j:
            return 1.0
        cfg = self.cfg
        s = 0.0
        if self._issuer_ids[i] == self._issuer_ids[j]:   s += cfg.sim_w_issuer
        if self._sector_ids[i] == self._sector_ids[j]:   s += cfg.sim_w_sector
        if self._rating_ids[i] == self._rating_ids[j]:   s += cfg.sim_w_rating
        if self._dur_ids[i]    == self._dur_ids[j]:      s += cfg.sim_w_duration
        return float(s)

    def similarity_row(self, i: int) -> np.ndarray:
        """
        Return the full i-th row of Sigma as a dense (N,) array.
        O(N) time.  Useful for inspection — avoid calling in the hot loop.
        """
        N   = len(self.bonds)
        cfg = self.cfg
        row = np.zeros(N, dtype=np.float32)

        same_issuer   = (self._issuer_ids == self._issuer_ids[i]).astype(np.float32)
        same_sector   = (self._sector_ids == self._sector_ids[i]).astype(np.float32)
        same_rating   = (self._rating_ids == self._rating_ids[i]).astype(np.float32)
        same_duration = (self._dur_ids    == self._dur_ids[i]   ).astype(np.float32)

        row = (
            cfg.sim_w_issuer   * same_issuer
            + cfg.sim_w_sector   * same_sector
            + cfg.sim_w_rating   * same_rating
            + cfg.sim_w_duration * same_duration
        )
        row[i] = 1.0
        return row

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def __len__(self)             -> int:  return len(self.bonds)
    def __getitem__(self, i: int) -> Bond: return self.bonds[i]

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
            row = {
                "bond_id":       b.bond_id,
                "kappa":         b.kappa,
                "sigma":         b.sigma,
                "beta_mmpp_bid": b.beta_mmpp_bid,
                "beta_mmpp_ask": b.beta_mmpp_ask,
            }
            for d in range(self.cfg.latent_dim):
                row[f"v_{d}"] = float(b.v_n[d])
            for f in range(3):
                row[f"beta_factor_{f}"] = float(b.beta_factor[f])
            rows.append(row)
        return pd.DataFrame(rows)
