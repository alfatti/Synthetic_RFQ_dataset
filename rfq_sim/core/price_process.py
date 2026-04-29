"""
price_process.py
----------------
Mid-price and bid-ask spread dynamics for all N bonds.

Four layers stacked on top of each other:
  1. Common factor VAR(1)        — drives correlated moves across bonds
  2. GARCH(1,1) idio variance    — vol clustering
  3. Cross-bond spillover η·Σ·ε  — similar bonds partially share idio shocks
  4. MMPP micro-price drift      — imbalance between buy/sell flow drifts mid

Spread δ⁰_{n,t} widens with vol, inventory pressure, and illiquidity tier.
The normalised quote δ/δ⁰ is what enters the logistic demand curve.

All updates are discrete-time steps of size dt (in seconds).
For the simulation's short timesteps (seconds to minutes) the discretisation
error is negligible.
"""

from __future__ import annotations
import numpy as np
from typing import Dict
from rfq_sim.core.config import BondConfig
from rfq_sim.core.bonds import BondUniverse

_DAY_S = 36_000.0   # Seconds per trading day


class PriceProcess:
    """
    Mutable price state for all N bonds.  Updated by the simulator's main loop.
    """

    def __init__(
        self,
        cfg:   BondConfig,
        bonds: BondUniverse,
        rng:   np.random.Generator,
    ):
        self.cfg   = cfg
        self.bonds = bonds
        self.rng   = rng
        self.N     = len(bonds)

        # Current mid prices — evolve throughout the simulation
        self.mid_prices = np.array(
            [b.price0 for b in bonds.bonds], dtype=np.float64
        )

        # GARCH conditional variance h_t for each bond (initialised to σ²)
        self.garch_h = np.array(
            [b.sigma ** 2 for b in bonds.bonds], dtype=np.float64
        )

        # Common factor state (p=3 factors, VAR(1))
        self.factors = self.rng.normal(0.0, 0.10, size=cfg.n_common_factors)

        # Current bid-ask spreads (initialised to baseline, then dynamic)
        self.spreads = np.array(
            [b.baseline_spread for b in bonds.bonds], dtype=np.float64
        )

        # Pre-cache factor loading matrix B ∈ ℝ^{N×p}
        self._B = np.stack(
            [b.beta_factor for b in bonds.bonds], axis=0
        )  # (N, p)

        # Pre-cache kappa vector
        self._kappa = np.array(
            [b.kappa for b in bonds.bonds], dtype=np.float64
        )

    def step(
        self,
        dt_s:            float,                    # Time step in seconds
        mmpp_imbalances: Dict[str, float],         # sector → λ_ask − λ_bid (day⁻¹)
        inventory:       np.ndarray,               # Current inventory per bond
        h_t:             float,                    # Intraday calendar multiplier
    ):
        """
        Advance all prices by dt_s seconds.

        Δ S_n = β_n · ΔF  +  η · Σ · ε  +  κ_n · imbalance · dt  +  jump
        h_t^2 is updated via GARCH(1,1).
        δ⁰_n is updated from the new h_t, inventory, and tier.
        """
        dt_days = dt_s / _DAY_S

        # 1. Common factor VAR(1) step
        #    ΔF_t = (ρ−1)·F_{t−1}·dt + σ_F·√dt·ε_F
        self.factors = (
            self.cfg.factor_ar_coeff * self.factors
            + self.cfg.factor_daily_vol * np.sqrt(dt_days)
            * self.rng.standard_normal(self.cfg.n_common_factors)
        )

        # 2. GARCH(1,1) variance update
        #    h² ← ω + α·h²(proxy for ε²) + β·h²
        self.garch_h = np.clip(
            self.cfg.garch_omega
            + (self.cfg.garch_alpha + self.cfg.garch_beta) * self.garch_h,
            1e-6, 5.0,
        )

        # 3. Idio shocks scaled by current GARCH vol and √dt
        idio_std    = np.sqrt(self.garch_h * dt_days)
        idio_shocks = self.rng.standard_normal(self.N) * idio_std

        # 4. Cross-bond spillover: η · Σ · ε
        spillover = (
            self.cfg.price_spillover_eta
            * (self.bonds.similarity_matrix @ idio_shocks)
        )

        # 5. MMPP micro-price drift: κ_n · (λ_ask − λ_bid) · dt
        #    Sell pressure (positive imbalance) → price drifts downward
        imb_by_bond = np.array([
            mmpp_imbalances.get(b.sector, 0.0) for b in self.bonds.bonds
        ])
        drift = -self._kappa * imb_by_bond * dt_days

        # 6. Common-factor contribution: B · ΔF · dt_days
        factor_contrib = (self._B @ self.factors) * dt_days

        # 7. Aggregate price move
        self.mid_prices += idio_shocks + spillover + drift + factor_contrib

        # 8. Sector-level jump process (double-exponential / Laplace jumps)
        for sector in self.cfg.sectors:
            p_jump = self.cfg.jump_intensity_per_day * dt_days
            if self.rng.random() < p_jump:
                mag = float(self.rng.laplace(0.0, self.cfg.jump_scale))
                for i, b in enumerate(self.bonds.bonds):
                    if b.sector == sector:
                        # All bonds in the sector get the same directional jump
                        # but scaled by a small uniform factor for heterogeneity
                        self.mid_prices[i] += mag * (0.5 + 0.5 * self.rng.random())

        # 9. Update bid-ask spreads
        self._update_spreads(inventory, h_t)

    def _update_spreads(self, inventory: np.ndarray, h_t: float):
        """
        δ⁰_{n,t} = δ̄_n · (1 + φ_σ·σ_{n,t} + φ_I·|I_n| + φ_tier·(tier-1))

        Spread widens with current vol, inventory exposure, and illiquidity.
        The 1/h_t term makes spreads slightly wider during thin hours.
        """
        cfg = self.cfg
        for i, b in enumerate(self.bonds.bonds):
            vol_factor  = cfg.spread_phi_sigma * float(np.sqrt(self.garch_h[i]))
            inv_factor  = cfg.spread_phi_inv   * abs(float(inventory[i]))
            tier_factor = cfg.spread_phi_tier  * (b.liquidity_tier - 1)
            time_factor = 0.10 / max(float(h_t), 0.05)

            self.spreads[i] = max(
                0.05,
                b.baseline_spread * (1.0 + vol_factor + inv_factor
                                     + tier_factor + time_factor)
            )

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def mid(self, n: int) -> float:
        return float(self.mid_prices[n])

    def spread(self, n: int) -> float:
        return float(self.spreads[n])

    def vol(self, n: int) -> float:
        """Current daily idio vol estimate √h_t."""
        return float(np.sqrt(self.garch_h[n]))
