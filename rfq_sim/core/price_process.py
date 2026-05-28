"""
price_process.py
----------------
Mid-price and bid-ask spread dynamics for all N bonds.

Four layers:
  1. Common factor VAR(1)         — correlated moves across all bonds
  2. GARCH(1,1) idio variance     — vol clustering, correctly implemented
  3. Cross-bond spillover η·Σ·ε  — similar bonds partially share idio shocks
  4. MMPP micro-price drift       — buy/sell flow imbalance tilts the mid

The previous version had two bugs that together suppressed all diffusive
structure and made prices look piecewise-linear:

  Bug 1 — GARCH used h_t as a proxy for ε_t² in the α term.
           Real GARCH: h_{t+1} = ω + α·ε_t² + β·h_t
           Bad GARCH:  h_{t+1} ≈ (ω + (α+β)·h_t)  ← always near its mean
           With α=0.10 and β=0.84, the bad version gave h_{t+1} ≈ 0.94·h_t + 0.05,
           which converges instantly to h*=0.83 and never moves.  There was
           no vol clustering, no GARCH effect at all.

  Bug 2 — Factor contribution was multiplied by dt_days a second time.
           self.factors already evolves on a per-step basis; applying dt_days
           again made the factor term negligible and broke the intended
           correlation structure across bonds.

Both are fixed here.  The key GARCH change: we save the realised idio
shock ε_t each step and use ε_t² in the next step's variance update.
"""

from __future__ import annotations
import numpy as np
from typing import Dict
from rfq_sim.core.config import BondConfig
from rfq_sim.core.bonds import BondUniverse

# One trading day = 10 hours = 36,000 seconds
_DAY_S = 36_000.0


class PriceProcess:
    """
    Mutable price state for all N bonds.  Called by the simulator's main
    loop on every clock advance.
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

        # ── State variables ────────────────────────────────────────────

        # Mid prices, initialised from bond.price0
        self.mid_prices = np.array(
            [b.price0 for b in bonds.bonds], dtype=np.float64
        )

        # GARCH conditional variance h_t² per bond.
        # Initialised at the unconditional variance σ_n².
        self.garch_h = np.array(
            [b.sigma ** 2 for b in bonds.bonds], dtype=np.float64
        )

        # Last realised idio shock ε_{t-1} per bond — needed for GARCH update.
        # Initialised at zero (treated as one quiet step before burn-in).
        self._last_idio = np.zeros(self.N, dtype=np.float64)

        # Common factor state (p=3): rates, HY-spread, sector.
        # Initialised near zero; burns in quickly via the VAR dynamics.
        self.factors = self.rng.normal(0.0, 0.05, size=cfg.n_common_factors)

        # Current bid-ask spreads, initialised to each bond's baseline
        self.spreads = np.array(
            [b.baseline_spread for b in bonds.bonds], dtype=np.float64
        )

        # ── Cached matrices (computed once) ───────────────────────────

        # Factor loading matrix B ∈ ℝ^{N×p}: B[n] = bond n's β_factor vector
        self._B = np.stack(
            [b.beta_factor for b in bonds.bonds], axis=0
        ).astype(np.float64)   # shape (N, p)

        # MMPP drift sensitivity κ_n per bond
        self._kappa = np.array(
            [b.kappa for b in bonds.bonds], dtype=np.float64
        )   # shape (N,)

    # ------------------------------------------------------------------
    # Main price step
    # ------------------------------------------------------------------

    def step(
        self,
        dt_s:             float,                  # Time elapsed in seconds
        mmpp_imbalances:  Dict[str, float],        # sector → λ_ask − λ_bid (day⁻¹)
        inventory:        np.ndarray,              # Current inventory per bond (N,)
        h_t:              float,                   # Intraday calendar multiplier
    ):
        """
        Advance all N bond prices by dt_s seconds.

        ΔS_n = β_n·ΔF·√dt  +  ε_n  +  η·Σ·ε  −  κ_n·imbalance·dt  +  jump

        where ε_n ~ N(0, √(h_t²·dt)) is drawn fresh each step, and h_t² is
        updated via proper GARCH(1,1) using the previous step's ε².
        """
        dt_days = dt_s / _DAY_S

        # ── 1. Common factor VAR(1) step ──────────────────────────────
        # F_t = ρ·F_{t-1} + σ_F·√dt·z,   z ~ N(0, I)
        # The √dt scaling is correct: over one full day dt=1, the factor
        # shock has std = factor_daily_vol, consistent with daily vol units.
        factor_shock = (
            self.cfg.factor_daily_vol
            * np.sqrt(dt_days)
            * self.rng.standard_normal(self.cfg.n_common_factors)
        )
        self.factors = self.cfg.factor_ar_coeff * self.factors + factor_shock

        # ── 2. Factor contribution to price ───────────────────────────
        # ΔS_n^{factor} = B_n · ΔF
        # B_n is the bond's loading on each factor.  We do NOT multiply by
        # dt_days again — the √dt is already in factor_shock above.
        factor_contrib = self._B @ factor_shock   # shape (N,)

        # ── 3. GARCH(1,1) variance update ─────────────────────────────
        # h_{t}² = ω + α·ε_{t-1}² + β·h_{t-1}²
        # This is the correct GARCH recursion.  _last_idio holds ε_{t-1}.
        self.garch_h = (
            self.cfg.garch_omega
            + self.cfg.garch_alpha * self._last_idio ** 2
            + self.cfg.garch_beta  * self.garch_h
        )
        # Clip to a sensible range to prevent numerical blow-up
        self.garch_h = np.clip(self.garch_h, 1e-6, 25.0)

        # ── 4. Idiosyncratic shocks ε_n ~ N(0, σ_{GARCH}·√dt) ────────
        # GARCH vol is in daily units, so we scale by √dt_days.
        idio_std    = np.sqrt(self.garch_h * dt_days)   # shape (N,)
        idio_shocks = self.rng.standard_normal(self.N) * idio_std

        # Save for the next step's GARCH update
        self._last_idio = idio_shocks.copy()

        # ── 5. Cross-bond spillover η·Σ·ε ─────────────────────────────
        # Similar bonds partially share each other's idiosyncratic shocks.
        # Σ is the observable similarity matrix (symmetric, diag=1).
        spillover = (
            self.cfg.price_spillover_eta
            * (self.bonds.similarity_matrix @ idio_shocks)
        )

        # ── 6. MMPP micro-price drift ──────────────────────────────────
        # dS_n = −κ_n·(λ_ask − λ_bid)·dt
        # Sell pressure (λ_ask > λ_bid, positive imbalance) → price drifts down.
        imb_per_bond = np.array([
            mmpp_imbalances.get(b.sector, 0.0) for b in self.bonds.bonds
        ])
        drift = -self._kappa * imb_per_bond * dt_days   # shape (N,)

        # ── 7. Total price move ────────────────────────────────────────
        self.mid_prices += factor_contrib + idio_shocks + spillover + drift

        # ── 8. Sector-level jump process ──────────────────────────────
        # Rare Laplace-distributed jumps, correlated within sectors.
        # Probability of a jump in this sector during this step = λ_jump·dt.
        for sector in self.cfg.sectors:
            if self.rng.random() < self.cfg.jump_intensity_per_day * dt_days:
                mag = float(self.rng.laplace(0.0, self.cfg.jump_scale))
                for i, b in enumerate(self.bonds.bonds):
                    if b.sector == sector:
                        # Each bond in the sector gets the jump, scaled by a
                        # small uniform factor to add cross-sectional heterogeneity
                        self.mid_prices[i] += mag * (0.5 + 0.5 * self.rng.random())

        # ── 9. Update bid-ask spreads ──────────────────────────────────
        self._update_spreads(inventory, h_t)

    # ------------------------------------------------------------------
    # Spread dynamics
    # ------------------------------------------------------------------

    def _update_spreads(self, inventory: np.ndarray, h_t: float):
        """
        δ⁰_{n,t} = δ̄_n · (1 + φ_σ·√h_t² + φ_I·|I_n| + φ_tier·(tier-1))

        Spread widens with current realised vol, inventory pressure, and
        illiquidity tier.  The 1/h_t time-of-day term makes spreads
        slightly wider during thin early-morning and late-afternoon hours.
        """
        cfg = self.cfg
        for i, b in enumerate(self.bonds.bonds):
            vol_factor  = cfg.spread_phi_sigma * float(np.sqrt(self.garch_h[i]))
            inv_factor  = cfg.spread_phi_inv   * abs(float(inventory[i]))
            tier_factor = cfg.spread_phi_tier  * (b.liquidity_tier - 1)
            # Thin hours → h_t small → 1/h_t large → wider spread
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
        """Current annualised-daily vol estimate: √h_t for bond n."""
        return float(np.sqrt(self.garch_h[n]))
