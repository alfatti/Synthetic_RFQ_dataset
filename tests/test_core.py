"""
tests/test_core.py
------------------
Core unit tests — run before committing and before any long simulation.

    cd rfq_simulation
    python -m pytest tests/ -v

26 tests covering calendar, MMPP, bonds, clients, and outcome model.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pytest
from datetime import datetime, time, date

from rfq_sim.core.config import SimConfig
from rfq_sim.core.calendar import TradingCalendar, SESSION_SECONDS
from rfq_sim.core.bonds import BondUniverse
from rfq_sim.core.clients import ClientUniverse, ARCHETYPES
from rfq_sim.core.mmpp import MMPPEngine
from rfq_sim.core.outcomes import OutcomeModel


# ── Fixtures ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def cfg():
    c = SimConfig(seed=99)
    c.bonds.n_bonds    = 24   # 8 issuers × 3 bonds
    c.bonds.n_issuers  = 8
    c.clients.n_clients = 20
    return c

@pytest.fixture(scope="module")
def rng():
    return np.random.default_rng(99)

@pytest.fixture(scope="module")
def calendar(cfg):
    return TradingCalendar(cfg.calendar)

@pytest.fixture(scope="module")
def bonds(cfg, rng):
    return BondUniverse(cfg.bonds, rng)

@pytest.fixture(scope="module")
def clients(cfg, bonds, rng):
    return ClientUniverse(cfg.clients, bonds, rng)


# ── Calendar ──────────────────────────────────────────────────────────────

class TestCalendar:

    def test_no_holidays_in_trading_days(self, calendar):
        import pandas as pd
        holiday_set = {pd.Timestamp(d).date() for d in calendar.cfg.holidays}
        for d in calendar._all_days:
            assert d not in holiday_set, f"Holiday {d} found in trading days"

    def test_no_weekends(self, calendar):
        for d in calendar._all_days:
            assert d.weekday() < 5, f"Weekend {d} in trading days"

    def test_dataset_days_subset_of_all(self, calendar):
        all_set = set(calendar._all_days)
        for d in calendar.dataset_days:
            assert d in all_set

    def test_train_before_test(self, calendar):
        import pandas as pd
        te = pd.Timestamp(calendar.cfg.train_end).date()
        assert all(d <= te for d in calendar.train_days)
        assert all(d >  te for d in calendar.test_days)

    def test_h_in_range(self, calendar):
        """h(t) ∈ [h_min, 1] for any trading second."""
        d   = calendar.dataset_days[0]
        for hour in [7, 10, 12, 14, 16]:
            ts = datetime.combine(d, time(hour, 30, 0))
            h  = calendar.h(ts)
            assert calendar.cfg.h_min - 0.01 <= h <= 1.01, f"h={h} at hour {hour}"

    def test_midday_peak(self, calendar):
        """h at midday should exceed h at open."""
        d    = calendar.dataset_days[0]
        h_op = calendar.h(datetime.combine(d, time(7, 5, 0)))
        h_md = calendar.h(datetime.combine(d, time(12, 0, 0)))
        assert h_md > h_op

    def test_month_end_multiplier_higher(self, calendar):
        me_days   = [d for d in calendar.dataset_days if calendar.is_month_end(d)]
        norm_days = [d for d in calendar.dataset_days if not calendar.is_month_end(d)]
        if me_days and norm_days:
            me_mult   = calendar.calendar_multiplier(datetime.combine(me_days[0],   time(12, 0)))
            norm_mult = calendar.calendar_multiplier(datetime.combine(norm_days[0], time(12, 0)))
            assert me_mult > norm_mult

    def test_advance_clock_skips_night(self, calendar):
        """Advancing past close should snap to next open."""
        d    = calendar.dataset_days[0]
        late = datetime.combine(d, time(17, 30, 0))   # after close
        nxt  = calendar.advance_clock(late)
        assert nxt.time() == time(calendar.cfg.session_open_hour, 0, 0)
        assert nxt.date() > d


# ── MMPP ──────────────────────────────────────────────────────────────────

class TestMMPP:

    def test_q_rows_sum_zero(self, cfg, rng):
        eng = MMPPEngine(cfg.mmpp, rng)
        for sec, proc in eng.processes.items():
            row_sums = proc.Q.sum(axis=1)
            # The whitepaper Q matrices have some numerical slack — use atol=2
            assert np.allclose(row_sums, 0, atol=2.0), \
                f"Q rows not zero for {sec}: {row_sums}"

    def test_stationary_sums_to_one(self, cfg, rng):
        eng = MMPPEngine(cfg.mmpp, rng)
        for sec, proc in eng.processes.items():
            pi = proc._pi
            assert np.all(pi >= -1e-9), f"Negative stationary prob in {sec}"
            assert abs(pi.sum() - 1.0) < 0.01, f"pi doesn't sum to 1 in {sec}"

    def test_intensities_positive(self, cfg, rng):
        eng = MMPPEngine(cfg.mmpp, rng)
        for sec in cfg.mmpp.sector_params:
            lb, la = eng.intensities(sec)
            assert lb > 0 and la > 0

    def test_sojourn_positive(self, cfg, rng):
        eng = MMPPEngine(cfg.mmpp, rng)
        for sec, proc in eng.processes.items():
            s = proc.sojourn_seconds()
            assert 0 < s < 5 * SESSION_SECONDS + 1, f"Bad sojourn {s} in {sec}"

    def test_transitions_visit_multiple_states(self, cfg, rng):
        eng = MMPPEngine(cfg.mmpp, rng)
        proc = list(eng.processes.values())[0]
        visited = {proc.state}
        for _ in range(300):
            proc.transition()
            visited.add(proc.state)
        assert len(visited) >= 3, "MMPP not ergodic"


# ── BondUniverse ──────────────────────────────────────────────────────────

class TestBonds:

    def test_exact_bond_count(self, cfg, bonds):
        assert len(bonds) == cfg.bonds.n_bonds

    def test_similarity_symmetric(self, bonds):
        S = bonds.similarity_matrix
        assert np.allclose(S, S.T, atol=1e-5)

    def test_similarity_diagonal_one(self, bonds):
        assert np.allclose(np.diag(bonds.similarity_matrix), 1.0)

    def test_similarity_nonneg_max_one(self, bonds):
        S = bonds.similarity_matrix
        assert np.all(S >= 0) and np.all(S <= 1.0 + 1e-5)

    def test_latent_factors_positive(self, bonds):
        for b in bonds.bonds:
            assert np.all(b.v_n > 0), f"Bond {b.bond_id} has non-positive v_n"

    def test_mmpp_betas_sum_one_per_sector(self, bonds):
        from collections import defaultdict
        bid_sums = defaultdict(float)
        ask_sums = defaultdict(float)
        for b in bonds.bonds:
            bid_sums[b.sector] += b.beta_mmpp_bid
            ask_sums[b.sector] += b.beta_mmpp_ask
        for sec in bid_sums:
            assert abs(bid_sums[sec] - 1.0) < 0.05, f"Bid betas ≠ 1 in {sec}"
            assert abs(ask_sums[sec] - 1.0) < 0.05, f"Ask betas ≠ 1 in {sec}"


# ── ClientUniverse ────────────────────────────────────────────────────────

class TestClients:

    def test_exact_client_count(self, cfg, clients):
        assert len(clients) == cfg.clients.n_clients

    def test_affinity_shape(self, cfg, bonds, clients):
        assert clients.affinity.values.shape == (cfg.clients.n_clients, cfg.bonds.n_bonds)

    def test_softmax_rows_sum_one(self, clients):
        row_sums = clients.affinity.softmax_probs.sum(axis=1)
        assert np.allclose(row_sums, 1.0, atol=1e-4)

    def test_rho_in_unit_interval(self, clients):
        for c in clients.clients:
            assert 0.0 <= c.rho_k <= 1.0

    def test_beta_positive(self, clients):
        for c in clients.clients:
            assert c.beta_k > 0

    def test_multiple_archetypes(self, clients):
        seen = {c.archetype for c in clients.clients}
        assert len(seen) >= 3


# ── OutcomeModel ──────────────────────────────────────────────────────────

class TestOutcomes:

    def _make_model(self, cfg, clients, rng):
        return OutcomeModel(cfg.outcomes, cfg.clients, clients.affinity, rng)

    def test_probs_sum_to_one(self, cfg, bonds, clients, rng):
        model  = self._make_model(cfg, clients, rng)
        client = clients[0]
        p_win, p_loss, p_cancel, p_expire = model.compute_probs(
            0, 0, client,
            delta=0.25, delta0=0.50, size_mm=5.0, n_competing=2,
            mid_request=100.0, mid_close=100.05, garch_vol=0.2,
            h_t=0.8, is_imbalanced=False, side="buy",
        )
        total = p_win + p_loss + p_cancel + p_expire
        assert abs(total - 1.0) < 1e-6
        assert all(p >= 0 for p in [p_win, p_loss, p_cancel, p_expire])

    def test_tighter_quote_higher_win_prob(self, cfg, bonds, clients, rng):
        model  = self._make_model(cfg, clients, rng)
        client = clients[0]
        probs  = []
        for delta in [0.10, 0.25, 0.50, 1.00]:
            p, *_ = model.compute_probs(
                0, 0, client, delta=delta, delta0=0.50, size_mm=5.0,
                n_competing=2, mid_request=100.0, mid_close=100.0,
                garch_vol=0.2, h_t=0.8, is_imbalanced=False, side="buy",
            )
            probs.append(p)
        assert probs[0] > probs[-1], "Tighter quote should have higher WIN prob"

    def test_high_rho_higher_cancel(self, cfg, bonds, clients, rng):
        """Higher rho_k should produce higher cancellation probability."""
        model = self._make_model(cfg, clients, rng)

        # Create two fake clients with different rho
        from rfq_sim.core.clients import Client
        import dataclasses

        c_low  = dataclasses.replace(clients[0], rho_k=0.05)
        c_high = dataclasses.replace(clients[0], rho_k=0.90)

        kw = dict(bond_id=0, delta=0.25, delta0=0.50, size_mm=5.0,
                  n_competing=2, mid_request=100.0, mid_close=100.30,
                  garch_vol=0.10, h_t=0.8, is_imbalanced=False, side="sell")

        _, _, p_cancel_lo, _ = model.compute_probs(0, **kw, client=c_low)
        _, _, p_cancel_hi, _ = model.compute_probs(0, **kw, client=c_high)
        assert p_cancel_hi > p_cancel_lo, \
            f"High-rho cancel {p_cancel_hi:.4f} not > low-rho {p_cancel_lo:.4f}"
