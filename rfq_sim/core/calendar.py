"""
calendar.py
-----------
US HY bond market calendar utilities.

The simulation clock only ticks during real trading hours (07:00–17:00 ET)
on real business days.  Everything else — nights, weekends, Federal Reserve
holidays — gets skipped entirely, exactly as real TRACE data looks.

Key responsibility: given a datetime that might fall in dead time, return the
next valid trading second.  This is the only place in the codebase that knows
about the calendar; all other modules just call advance_clock().
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta, date, time as dtime
from typing import List, Set
from rfq_sim.core.config import CalendarConfig

# Session length in seconds (10 hours × 3600 s/hr)
SESSION_SECONDS = 10 * 3600   # 36,000 s


class TradingCalendar:
    """
    Manages the simulation clock and all calendar-related utilities.
    """

    def __init__(self, cfg: CalendarConfig):
        self.cfg = cfg

        # Pre-build a set of holiday dates for O(1) lookup
        self.holiday_set: Set[date] = {
            pd.Timestamp(d).date() for d in cfg.holidays
        }

        # Sorted list of every valid trading day in the full simulation window
        # (burn-in through dataset end)
        self._all_days: List[date] = self._build_day_list(
            cfg.burnin_start, cfg.dataset_end
        )
        self._day_set: Set[date] = set(self._all_days)

        # Subset: dataset days only (after burn-in)
        ds = pd.Timestamp(cfg.dataset_start).date()
        self.dataset_days  = [d for d in self._all_days if d >= ds]
        self.burnin_days   = [d for d in self._all_days if d <  ds]

        te = pd.Timestamp(cfg.train_end).date()
        self.train_days = [d for d in self.dataset_days if d <= te]
        self.test_days  = [d for d in self.dataset_days if d >  te]

        # Last 2 business days of each month get the γ = 1.30 intensity boost
        self._month_end_set: Set[date] = self._build_month_end_set()

        # Session open/close time objects
        self._open  = dtime(cfg.session_open_hour,  0, 0)
        self._close = dtime(cfg.session_close_hour, 0, 0)

    # ------------------------------------------------------------------
    # Building the day list
    # ------------------------------------------------------------------

    def _build_day_list(self, start: str, end: str) -> List[date]:
        """Mon–Fri business days that are not Federal Reserve holidays."""
        bdays = pd.bdate_range(start=start, end=end)
        return [d.date() for d in bdays if d.date() not in self.holiday_set]

    def _build_month_end_set(self) -> Set[date]:
        """
        Flag the last 2 business days of each calendar month within the
        full simulation window.  These receive the γ=1.30 activity boost.
        """
        from itertools import groupby
        month_ends: Set[date] = set()
        keyfn = lambda d: (d.year, d.month)
        for _, group in groupby(self._all_days, key=keyfn):
            days = sorted(group)
            for d in days[-2:]:
                month_ends.add(d)
        return month_ends

    # ------------------------------------------------------------------
    # Clock utilities
    # ------------------------------------------------------------------

    def is_trading_second(self, ts: datetime) -> bool:
        """True iff ts falls within a valid trading session."""
        return (
            ts.date() in self._day_set
            and self._open <= ts.time() < self._close
        )

    def is_month_end(self, d: date) -> bool:
        return d in self._month_end_set

    def session_open_dt(self, d: date) -> datetime:
        """Return the opening datetime for day d."""
        return datetime.combine(d, self._open)

    def advance_clock(self, ts: datetime) -> datetime:
        """
        If ts is inside a valid trading session, return ts unchanged.
        Otherwise snap forward to the next session open.

        This is the only way the simulator moves the clock forward — we
        never increment seconds manually and then hope it's still in session.
        """
        # Already in session?
        if self.is_trading_second(ts):
            return ts

        # Build candidate: same day open, or next valid day's open
        d = ts.date()
        t = ts.time()

        if d in self._day_set and t < self._open:
            # Before today's open — snap to today's open
            return datetime.combine(d, self._open)

        # After close or on a non-trading day — find next trading day
        candidate = d + timedelta(days=1)
        for _ in range(30):   # Walk up to 30 days forward (handles end-of-window)
            if candidate in self._day_set:
                return datetime.combine(candidate, self._open)
            candidate += timedelta(days=1)

        # Past the end of the simulation window — return a sentinel far in the future
        # The simulator's while-loop will exit because this exceeds end_dt
        return datetime.combine(candidate, self._open)

    def add_trading_seconds(self, ts: datetime, seconds: float) -> datetime:
        """
        Add `seconds` of real wall-clock time to ts, then snap to the next
        valid trading second.  Used to advance the event clock.

        Note: this does NOT skip overnight time — it adds physical seconds.
        The overnight gap is handled by advance_clock() snapping to open.
        """
        if seconds > 864_000:   # Cap at 10 days of real seconds to avoid overflow
            seconds = 864_000
        candidate = ts + timedelta(seconds=float(seconds))
        return self.advance_clock(candidate)

    # ------------------------------------------------------------------
    # Intraday multiplier h(t)
    # ------------------------------------------------------------------

    def session_fraction(self, ts: datetime) -> float:
        """τ ∈ [0, 1]: fraction of the trading session elapsed at ts."""
        open_dt = datetime.combine(ts.date(), self._open)
        elapsed = max(0.0, (ts - open_dt).total_seconds())
        return min(elapsed / SESSION_SECONDS, 1.0)

    def h(self, ts: datetime) -> float:
        """
        Intraday multiplier h(τ) ∈ [h_min, 1].

        h(τ) = h_min + (1 − h_min)·sin²(πτ)·(1 − α_close·1[τ > τ_close])

        This gives a smooth sin² envelope peaking around midday (τ=0.5)
        with extra suppression in the last 10 % of the session.
        """
        cfg = self.cfg
        tau = self.session_fraction(ts)
        val = cfg.h_min + (1.0 - cfg.h_min) * (np.sin(np.pi * tau) ** 2)
        if tau > cfg.tau_close:
            val *= (1.0 - cfg.alpha_close)
        return float(np.clip(val, cfg.h_min, 1.0))

    def calendar_multiplier(self, ts: datetime) -> float:
        """Combined multiplier = h(t) × γ_month_end (if applicable)."""
        mult = self.h(ts)
        if self.is_month_end(ts.date()):
            mult *= self.cfg.month_end_gamma
        return mult

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def n_dataset_days(self) -> int:
        return len(self.dataset_days)

    @property
    def n_burnin_days(self) -> int:
        return len(self.burnin_days)

    @property
    def n_train_days(self) -> int:
        return len(self.train_days)

    @property
    def n_test_days(self) -> int:
        return len(self.test_days)
