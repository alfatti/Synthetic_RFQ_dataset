"""
inventory.py
------------
Trader inventory — the most important state variable in the system.

I track per-bond notional exposure and aggregate DV01.  Both feed
back into the quoting model (via inventory skew) and the spread model
(via the φ_I term).

Stochastic hedging fires with intensity ∝ |I_n|² and partially
mean-reverts the position, preventing unrealistic accumulation.
"""

from __future__ import annotations
import numpy as np
from rfq_sim.core.config import BondConfig
from rfq_sim.core.bonds import BondUniverse

# DV01 per $1MM notional per 1bp move, by duration bucket
_DV01 = {"Short": 300.0, "Medium": 600.0, "Long": 900.0}


class InventoryManager:
    """
    Per-bond inventory in $MM notional.

    Convention:
      client SELLS to dealer  → dealer inventory increases  (+)
      client BUYS  from dealer → dealer inventory decreases (−)
    """

    def __init__(self, cfg: BondConfig, bonds: BondUniverse):
        self.cfg    = cfg
        self.bonds  = bonds
        self.N      = len(bonds)

        # Per-bond inventory in $MM notional — starts flat, builds via burn-in
        self.I = np.zeros(self.N, dtype=np.float64)

        # Soft inventory limit = 5 % of outstanding for each bond
        self._limits = np.array(
            [b.outstanding_mm * 0.05 for b in bonds.bonds]
        )

    def fill(self, bond_id: int, client_side: str, size_mm: float):
        """Update inventory after a WIN.
        client_side = 'sell' → dealer buys (goes long +)
        client_side = 'buy'  → dealer sells (goes short −)
        """
        self.I[bond_id] += (+1.0 if client_side == "sell" else -1.0) * size_mm

    def try_hedge(self, bond_id: int, rng: np.random.Generator,
                  rho_I: float = 0.50) -> bool:
        """
        Stochastic partial mean-reversion of inventory.
        Intensity ∝ (|I| / limit)² so the further from zero the more
        urgently the trader hedges.  Returns True if a hedge fired.
        """
        lim = self._limits[bond_id]
        if lim <= 0.0:
            return False
        frac = abs(self.I[bond_id]) / lim
        if rng.random() < 0.001 * frac ** 2:
            self.I[bond_id] *= rho_I
            return True
        return False

    def at_limit(self, bond_id: int, direction: int) -> bool:
        """True if adding `direction`-signed position would breach the soft limit."""
        return abs(self.I[bond_id] + direction) > self._limits[bond_id]

    @property
    def portfolio_dv01(self) -> float:
        """Signed aggregate DV01 across all positions ($)."""
        total = 0.0
        for i, b in enumerate(self.bonds.bonds):
            total += self.I[i] * _DV01.get(b.duration_bucket, 500.0)
        return total

    def skew(self, bond_id: int, sigma_n: float) -> float:
        """
        Inventory skew for the quoting model:
          skew = α_inv · I_{n,t} · σ_{n,t}

        Positive inventory (long) → skew > 0 → tighter offer (in the quoting
        model this reduces the ask-side δ, so we sell more aggressively).
        """
        alpha = self.cfg.spread_phi_inv
        return float(alpha * self.I[bond_id] * sigma_n)
