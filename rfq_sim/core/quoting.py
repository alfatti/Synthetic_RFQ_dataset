"""
quoting.py
----------
Ground-truth quoting policy.

The trader quotes:
  δ_{n,t} = ½ δ⁰_{n,t}
            − α_inv · I_{n,t} · σ_{n,t} · dir
            + α_info · ρ̂_k
            + ε_q

where dir = +1 on client buy (dealer sells), −1 on client sell (dealer buys).
The inventory skew reduces δ when the dealer wants to offload a long position.
The adverse-selection term widens quotes for suspected informed clients.

A learned policy that consistently beats this on risk-adjusted spread revenue
has demonstrably found real signal in the data.
"""

from __future__ import annotations
import numpy as np
from rfq_sim.core.config import QuotingConfig
from rfq_sim.core.clients import Client
from rfq_sim.core.bonds import Bond


class QuotingModel:
    def __init__(self, cfg: QuotingConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

    def quote(
        self,
        bond:    Bond,
        client:  Client,
        side:    str,      # 'buy' or 'sell' from client's perspective
        spread:  float,    # δ⁰_{n,t}
        vol:     float,    # Current idio vol σ_{n,t}
        inv:     float,    # I_{n,t}
        rng:     np.random.Generator | None = None,
    ) -> float:
        """
        Return the trader's half-spread quote δ_{n,t} > 0.

        Inventory skew convention:
          Client BUY  → dealer sells short → dir = +1
          Client SELL → dealer buys long   → dir = -1
          When dealer is long (inv > 0) and client is buying (dir = +1),
          inv*dir > 0 so inv_skew > 0 and we subtract it → tighter ask
          (we want to unload the position).
        """
        _rng = rng or self.rng
        cfg  = self.cfg

        base = 0.5 * spread

        # Inventory skew: positive when dealer wants to trade in client's direction
        direction = +1.0 if side == "buy" else -1.0
        inv_skew  = cfg.alpha_inventory * float(inv) * float(vol) * direction

        # Adverse-selection premium using trader's running ρ̂_k estimate
        info_skew = cfg.alpha_info * client.rho_hat

        noise = float(_rng.normal(0.0, cfg.quote_noise_std))

        delta = base - inv_skew + info_skew + noise
        return float(max(0.01, delta))
