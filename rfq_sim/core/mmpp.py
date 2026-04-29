"""
mmpp.py
-------
Bidimensional Markov-Modulated Poisson Process (MMPP).

One MMPP per sector.  State space = {λ₁, λ₂}² for the (bid, ask)
intensity pair, giving four joint states:
  0: (lo, lo)   balanced low liquidity
  1: (lo, hi)   sell pressure
  2: (hi, lo)   buy pressure
  3: (hi, hi)   balanced high liquidity

Parameters taken directly from Table 1 of Bergault & Gueant (2024).

The MMPP state drives two things:
  1. RFQ arrival rates for every bond in the sector
  2. Price drift: dS = σ dW − κ·(λ_ask − λ_bid)·dt  (micro-price mechanism)

I simulate the CTMC using the standard competing-exponentials approach:
  sojourn time in state i ~ Exp(|Q[i,i]|)
  destination drawn proportional to off-diagonal row i of Q
"""

from __future__ import annotations
import numpy as np
from typing import Dict, Tuple
from rfq_sim.core.config import MMPPConfig

# State labels in lexicographic order
STATE_NAMES = ["lo_lo", "lo_hi", "hi_lo", "hi_hi"]

# Trading day length in seconds (10 h × 3600 s/h)
_DAY_S = 36_000.0


class SectorMMPP:
    """MMPP for a single sector."""

    def __init__(self, sector: str, params: dict, rng: np.random.Generator):
        self.sector   = sector
        self.rng      = rng
        self.lambda_lo = float(params["lambda_lo"])   # day⁻¹
        self.lambda_hi = float(params["lambda_hi"])   # day⁻¹

        # Intensity for each state:  state 0=(lo,lo), 1=(lo,hi), 2=(hi,lo), 3=(hi,hi)
        self._lam_bid = np.array(
            [self.lambda_lo, self.lambda_lo, self.lambda_hi, self.lambda_hi]
        )
        self._lam_ask = np.array(
            [self.lambda_lo, self.lambda_hi, self.lambda_lo, self.lambda_hi]
        )

        # Generator matrix Q (4×4), units day⁻¹
        self.Q = np.array(params["Q"], dtype=np.float64)

        # Rate out of each state = -diagonal of Q
        self._exit_rates = -np.diag(self.Q)   # all positive

        # Stationary distribution (for initial state draw)
        self._pi = self._stationary()

        # Initialise in a state drawn from the stationary distribution
        self.state = int(self.rng.choice(4, p=self._pi))

    def _stationary(self) -> np.ndarray:
        """Solve πQ = 0, Σπ = 1."""
        A = self.Q.T.copy()
        A[-1, :] = 1.0
        b = np.zeros(4); b[-1] = 1.0
        try:
            pi = np.linalg.solve(A, b)
            pi = np.abs(pi); pi /= pi.sum()
        except np.linalg.LinAlgError:
            pi = np.ones(4) / 4.0
        return pi

    @property
    def lambda_bid(self) -> float:
        return float(self._lam_bid[self.state])

    @property
    def lambda_ask(self) -> float:
        return float(self._lam_ask[self.state])

    @property
    def imbalance(self) -> float:
        """λ_ask − λ_bid  (day⁻¹).  Feeds the micro-price drift."""
        return self.lambda_ask - self.lambda_bid

    @property
    def state_name(self) -> str:
        return STATE_NAMES[self.state]

    def sojourn_seconds(self) -> float:
        """
        Draw sojourn time in the current state (in seconds).
        Exp(exit_rate_per_day) converted to seconds.
        Capped at 5 trading days to avoid degenerate draws.
        """
        rate_day = self._exit_rates[self.state]
        if rate_day <= 0.0:
            return 5 * _DAY_S
        rate_sec = rate_day / _DAY_S
        s = self.rng.exponential(1.0 / rate_sec)
        return float(min(s, 5 * _DAY_S))   # cap at 5 trading days

    def transition(self):
        """Move to next state, sampled from the off-diagonal row."""
        row = self.Q[self.state].copy()
        row[self.state] = 0.0
        total = row.sum()
        if total <= 0.0:
            return
        self.state = int(self.rng.choice(4, p=row / total))


class MMPPEngine:
    """
    Manages all sector MMPPs together.

    The simulator queries this engine for:
      - Current bid/ask intensities per sector
      - When the next MMPP transition fires (so it can interleave with RFQ arrivals)
      - The λ_ask − λ_bid imbalance for the price drift
    """

    def __init__(self, cfg: MMPPConfig, rng: np.random.Generator):
        self.cfg = cfg
        self.rng = rng

        self.processes: Dict[str, SectorMMPP] = {
            sector: SectorMMPP(sector, params, rng)
            for sector, params in cfg.sector_params.items()
        }

        # Next transition time for each sector, as absolute simulation seconds
        # from t=0 (burn-in open).  Initialised in reset_clocks().
        self._next_t: Dict[str, float] = {}

    def reset_clocks(self, t0: float = 0.0):
        """
        (Re)initialise all next-transition clocks.
        Call once at the start of the simulation with the burn-in open time.
        """
        for sector, proc in self.processes.items():
            self._next_t[sector] = t0 + proc.sojourn_seconds()

    def next_event(self) -> Tuple[float, str]:
        """Return (time, sector) of the soonest MMPP transition."""
        sector = min(self._next_t, key=self._next_t.__getitem__)
        return self._next_t[sector], sector

    def fire(self, sector: str, t_now: float):
        """Execute transition for `sector` at time t_now, then reschedule."""
        self.processes[sector].transition()
        self._next_t[sector] = t_now + self.processes[sector].sojourn_seconds()

    # Convenience accessors
    def intensities(self, sector: str) -> Tuple[float, float]:
        """(λ_bid, λ_ask) in day⁻¹."""
        p = self.processes[sector]
        return p.lambda_bid, p.lambda_ask

    def imbalance(self, sector: str) -> float:
        return self.processes[sector].imbalance

    def state(self, sector: str) -> int:
        return self.processes[sector].state

    def all_states(self) -> Dict[str, int]:
        return {s: p.state for s, p in self.processes.items()}
