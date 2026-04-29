"""
rfq_arrivals.py
---------------
RFQ arrival process.

Each client k has a base arrival rate λ_k^base (RFQs / trading day).
That gets split across bonds proportionally to the client's affinity-
derived softmax probabilities p^{k,n}.  Finally, the MMPP sector
intensity scales the rate up or down based on current liquidity regime.

Effective rate for client k on bond n, side s, at time t:

  λ_{k,n,s}(t) = λ_k^base  ×  p^{k,n}  ×  (λ_n^s / λ̄_n^s)  ×  h(t)

where  λ̄_n^s  is the long-run average intensity for bond n on side s
(computed from the MMPP stationary distribution × β^{n,s}), and
λ_n^s is the current instantaneous value.  The ratio keeps the average
arrival rate equal to λ_k^base regardless of which MMPP state we're in.

The aggregate arrival process across ALL (k, n, s) triples is a
superposition of Poisson processes, itself Poisson with rate
  Λ(t) = Σ_{k,n,s} λ_{k,n,s}(t)

I draw the next inter-arrival time ~ Exp(Λ(t)) then sample which
(k, n, s) triple fired proportionally to their individual rates.
This is the standard thinning / superposition algorithm — exact for
Poisson processes.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from typing import Tuple, Dict, List

from rfq_sim.core.config import ClientConfig, BondConfig
from rfq_sim.core.clients import ClientUniverse
from rfq_sim.core.bonds import BondUniverse
from rfq_sim.core.mmpp import MMPPEngine
from rfq_sim.core.calendar import TradingCalendar, SESSION_SECONDS


@dataclass
class RFQEvent:
    """Everything needed for the outcome model and the observation log."""
    rfq_id:    int
    client_id: int
    bond_id:   int
    timestamp: datetime
    side:      str     # 'buy' or 'sell' (client perspective)
    size_mm:   float

    # Market state at arrival
    mid_price:   float
    spread:      float   # δ⁰_{n,t}
    garch_vol:   float
    mmpp_state:  int     # 0-3

    # Quote
    delta:       float   # Trader's half-spread δ_{n,t}

    # RFQ lifetime (seconds)
    lifetime_s:  float

    # Context
    inventory:   float   # I_{n,t} at arrival
    n_competing: int
    in_program:  bool

    # True probabilities (stored in ground truth, not in observable)
    p_win:    float
    p_cancel: float
    p_expire: float


class RFQArrivalProcess:
    """
    Generates the stream of RFQ events.

    Design notes on the rate computation
    ------------------------------------
    λ_k^base is in RFQs / trading-day.
    The MMPP intensities λ_n^{bid/ask} are also in day⁻¹.
    I normalise by the long-run average so the mean arrival rate
    doesn't explode in high-MMPP states:

        ratio_{n,s}(t) = λ_n^s(t) / λ̄_n^s

    where λ̄_n^s = β^{n,s} × (π_lo·λ_lo + π_hi·λ_hi + π_lo·λ_hi + π_hi·λ_lo) / 2
    (roughly the stationary mean, split evenly across bid/ask).

    The aggregate rate Λ is in RFQs / trading-day.
    I convert to per-second by dividing by SESSION_SECONDS (36 000 s).
    """

    def __init__(
        self,
        cli_cfg:  ClientConfig,
        bond_cfg: BondConfig,
        clients:  ClientUniverse,
        bonds:    BondUniverse,
        mmpp:     MMPPEngine,
        calendar: TradingCalendar,
        rng:      np.random.Generator,
    ):
        self.cli_cfg  = cli_cfg
        self.bond_cfg = bond_cfg
        self.clients  = clients
        self.bonds    = bonds
        self.mmpp     = mmpp
        self.calendar = calendar
        self.rng      = rng

        self.K = len(clients)
        self.N = len(bonds)

        # RFQ counter
        self._counter = 0

        # Pre-compute long-run average MMPP intensity per sector (day⁻¹)
        # Used as the normaliser so the ratio doesn't change the mean rate.
        self._lam_avg: Dict[str, float] = {}
        for sector, proc in mmpp.processes.items():
            pi     = proc._pi
            lam_lo = proc.lambda_lo
            lam_hi = proc.lambda_hi
            # Stationary mean of λ_bid (and independently λ_ask by symmetry)
            mean_bid = pi[0]*lam_lo + pi[1]*lam_lo + pi[2]*lam_hi + pi[3]*lam_hi
            mean_ask = pi[0]*lam_lo + pi[1]*lam_hi + pi[2]*lam_lo + pi[3]*lam_hi
            self._lam_avg[sector] = float((mean_bid + mean_ask) / 2.0)

        # Active flow-spillover boosts: {(k, n): expiry_sim_seconds}
        self._spillovers: Dict[Tuple[int, int], float] = {}

    # ------------------------------------------------------------------
    # Core rate computation
    # ------------------------------------------------------------------

    def _rate_matrix(self, h_t: float) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute (lam_buy, lam_sell) matrices of shape (K, N) in day⁻¹.

        lam_buy[k, n]  = rate at which client k sends a BUY RFQ on bond n
        lam_sell[k, n] = rate at which client k sends a SELL RFQ on bond n

        The key relationship:
          rate_{k,n,s} = base_k × p_{k,n} × ratio_{n,s}(t) × h_t

        where ratio_{n,s}(t) = β^{n,s} × λ_s(t) / λ̄_s.
        """
        K, N = self.K, self.N

        # Client base rates, shape (K,)
        base = np.array([c.base_arrival_rate for c in self.clients.clients])

        # Bond-selection softmax probs, shape (K, N)
        P = self.clients.affinity.softmax_probs

        # MMPP ratio for each bond, shape (N,) per side
        ratio_bid = np.zeros(N)
        ratio_ask = np.zeros(N)
        for i, b in enumerate(self.bonds.bonds):
            lam_avg = max(self._lam_avg[b.sector], 1e-6)
            lam_bid, lam_ask = self.mmpp.intensities(b.sector)
            ratio_bid[i] = b.beta_mmpp_bid * lam_bid / lam_avg
            ratio_ask[i] = b.beta_mmpp_ask * lam_ask / lam_avg

        # Broadcast: (K,1) × (K,N) × (1,N) × scalar = (K,N)
        lam_buy  = base[:, None] * P * ratio_bid[None, :] * h_t
        lam_sell = base[:, None] * P * ratio_ask[None, :] * h_t

        # Program-state boost: multiply by intensity multiplier on cluster bonds
        mult = self.cli_cfg.program_intensity_mult
        for k, c in enumerate(self.clients.clients):
            if c.in_program:
                for n in c.prog_bonds:
                    if n < N:
                        if c.prog_direction == +1:
                            lam_buy[k, n]  *= mult
                        else:
                            lam_sell[k, n] *= mult

        # Flow-spillover boosts from recently observed RFQs
        for (k, n), exp_t in list(self._spillovers.items()):
            if h_t > 0:   # placeholder — real check is in main loop
                mu = self.cli_cfg.spillover_strength
                lam_buy[k, n]  *= (1.0 + mu)
                lam_sell[k, n] *= (1.0 + mu)

        return lam_buy, lam_sell

    # ------------------------------------------------------------------
    # Next-arrival sampling
    # ------------------------------------------------------------------

    def next_arrival(
        self,
        h_t:   float,
        t_sim: float,   # Current sim time in seconds (for spillover expiry)
    ) -> Tuple[float, int, int, str]:
        """
        Draw (dt_seconds, client_id, bond_id, side) for the next RFQ.

        Returns dt in seconds.  This is the inter-arrival time for a
        Poisson process with aggregate rate Λ (day⁻¹), converted to seconds.
        """
        # Expire stale spillovers
        self._spillovers = {
            key: exp for key, exp in self._spillovers.items()
            if exp > t_sim
        }

        lam_buy, lam_sell = self._rate_matrix(h_t)

        # Flatten into one vector [buy_k0n0, buy_k0n1, ..., sell_k0n0, ...]
        flat = np.concatenate([lam_buy.ravel(), lam_sell.ravel()])
        total_day = float(flat.sum())

        if total_day <= 0.0:
            return float(SESSION_SECONDS), 0, 0, "buy"   # Dead session fallback

        # Inter-arrival in seconds: Exp(total_day / SESSION_SECONDS)
        rate_per_second = total_day / SESSION_SECONDS
        dt = float(self.rng.exponential(1.0 / rate_per_second))

        # Cap to 2 trading days to prevent overflow on very low-activity windows
        dt = min(dt, 2 * SESSION_SECONDS)

        # Sample which (k, n, side) fired
        probs = flat / total_day
        idx   = int(self.rng.choice(len(probs), p=probs))
        KN    = self.K * self.N
        if idx < KN:
            side = "buy"
            k = idx // self.N
            n = idx  % self.N
        else:
            idx2 = idx - KN
            side = "sell"
            k = idx2 // self.N
            n = idx2  % self.N

        return dt, int(k), int(n), side

    # ------------------------------------------------------------------
    # RFQ event construction
    # ------------------------------------------------------------------

    def build_event(
        self,
        client_id:  int,
        bond_id:    int,
        side:       str,
        timestamp:  datetime,
        mid_price:  float,
        spread:     float,
        garch_vol:  float,
        mmpp_state: int,
        inventory:  float,
        delta:      float,
        p_win:      float,
        p_cancel:   float,
        p_expire:   float,
        t_sim:      float,   # Current sim seconds (for spillover registration)
    ) -> RFQEvent:
        """
        Package all context into an RFQEvent and register flow spillovers.
        """
        client = self.clients[client_id]
        bond   = self.bonds[bond_id]
        cfg    = self.cli_cfg

        # Override side if in program state
        if client.in_program and client.prog_direction != 0:
            side = "buy" if client.prog_direction == +1 else "sell"

        # Clip size: log-normal, rounded to lot size, capped at fraction of outstanding
        raw_size = float(np.exp(self.rng.normal(client.size_mu, client.size_sigma)))
        lot      = cfg.lot_size_mm
        size_mm  = max(lot, round(raw_size / lot) * lot)
        size_mm  = min(size_mm, bond.outstanding_mm * cfg.max_notional_fraction)

        # Number of competing dealers
        n_comp = int(self.rng.poisson(
            client.auction_aggressiveness * cfg.mean_competing_dealers
        )) + 1

        # RFQ lifetime: log-normal around 3 minutes, bounded 30 s to 10 min
        lifetime_s = float(np.clip(
            np.exp(self.rng.normal(np.log(180.0), 0.60)),
            30.0, 600.0,
        ))

        # Register flow spillover for similar bonds
        spill_window = float(self.rng.uniform(
            cfg.spillover_window_min * 60.0,
            cfg.spillover_window_max * 60.0,
        ))
        expiry = t_sim + spill_window
        for n2, sim_val in enumerate(self.bonds.similarity_matrix[bond_id]):
            if n2 != bond_id and sim_val > 0.10:
                self._spillovers[(client_id, n2)] = expiry

        self._counter += 1
        return RFQEvent(
            rfq_id=self._counter,
            client_id=client_id,
            bond_id=bond_id,
            timestamp=timestamp,
            side=side,
            size_mm=size_mm,
            mid_price=mid_price,
            spread=spread,
            garch_vol=garch_vol,
            mmpp_state=mmpp_state,
            delta=delta,
            lifetime_s=lifetime_s,
            inventory=inventory,
            n_competing=n_comp,
            in_program=client.in_program,
            p_win=p_win,
            p_cancel=p_cancel,
            p_expire=p_expire,
        )
