"""
simulator.py
------------
Main event-driven simulation loop.

The clock advances in two kinds of steps:
  1. MMPP transition  — sector liquidity regime changes
  2. RFQ arrival      — a client sends an RFQ to the trader

Whichever comes first wins.  Between events the price process is stepped
by the elapsed time dt.

After the burn-in period (Dec 15 – Jan 2) the simulator starts writing
rows to two output buffers:
  observable_df  — everything a model is allowed to see
  ground_truth_df — latent variables for evaluation only
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import List, Tuple
from tqdm import tqdm

from rfq_sim.core.config import SimConfig
from rfq_sim.core.calendar import TradingCalendar, SESSION_SECONDS
from rfq_sim.core.bonds import BondUniverse
from rfq_sim.core.clients import ClientUniverse
from rfq_sim.core.mmpp import MMPPEngine
from rfq_sim.core.price_process import PriceProcess
from rfq_sim.core.inventory import InventoryManager
from rfq_sim.core.rfq_arrivals import RFQArrivalProcess
from rfq_sim.core.quoting import QuotingModel
from rfq_sim.core.outcomes import OutcomeModel


class RFQSimulator:
    """
    End-to-end synthetic RFQ market simulation.

    Usage:
        cfg = SimConfig(seed=42)
        sim = RFQSimulator(cfg)
        obs_df, gt_df = sim.run()
    """

    def __init__(self, cfg: SimConfig):
        self.cfg = cfg
        self.rng = np.random.default_rng(cfg.seed)

        print("Initialising calendar...")
        self.cal = TradingCalendar(cfg.calendar)

        print(f"Building bond universe  ({cfg.bonds.n_bonds} bonds, "
              f"{cfg.bonds.n_issuers} issuers)...")
        self.bonds = BondUniverse(cfg.bonds, self._child_rng())

        print(f"Building client universe ({cfg.clients.n_clients} clients)...")
        self.clients = ClientUniverse(cfg.clients, self.bonds, self._child_rng())

        print("Initialising MMPP processes...")
        self.mmpp = MMPPEngine(cfg.mmpp, self._child_rng())

        print("Initialising price process...")
        self.prices = PriceProcess(cfg.bonds, self.bonds, self._child_rng())

        print("Initialising inventory manager...")
        self.inv = InventoryManager(cfg.bonds, self.bonds)

        print("Building RFQ arrival process...")
        self.arrivals = RFQArrivalProcess(
            cfg.clients, cfg.bonds,
            self.clients, self.bonds, self.mmpp,
            self.cal, self._child_rng(),
        )

        print("Initialising quoting model...")
        self.quoting = QuotingModel(cfg.quoting, self._child_rng())

        print("Initialising outcome model...")
        self.outcomes = OutcomeModel(
            cfg.outcomes, cfg.clients,
            self.clients.affinity, self._child_rng(),
        )

        # Output buffers
        self._obs: List[dict] = []
        self._gt:  List[dict] = []

    def _child_rng(self) -> np.random.Generator:
        return np.random.default_rng(int(self.rng.integers(0, 2**31)))

    # ------------------------------------------------------------------
    # Simulation time helpers
    # ------------------------------------------------------------------

    def _dt_to_sim_seconds(self, dt: datetime) -> float:
        """Convert a datetime to simulation seconds since burn-in open."""
        return (dt - self._burnin_open).total_seconds()

    # ------------------------------------------------------------------
    # Main run
    # ------------------------------------------------------------------

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        cal = self.cal
        cfg = self.cfg

        # Simulation epoch — burn-in open datetime
        burnin_open_date = pd.Timestamp(cfg.calendar.burnin_start).date()
        self._burnin_open = cal.session_open_dt(
            cal.advance_clock(
                datetime.combine(burnin_open_date,
                                 datetime.min.time())
            ).date()
        )

        # Record-start datetime (first valid second of dataset_start day)
        record_from_date = pd.Timestamp(cfg.calendar.dataset_start).date()
        record_from = cal.advance_clock(
            datetime.combine(record_from_date, datetime.min.time())
        )

        # Dataset end datetime
        end_date = pd.Timestamp(cfg.calendar.dataset_end).date()
        end_dt   = datetime.combine(end_date,
                                    __import__("datetime").time(
                                        cfg.calendar.session_close_hour, 0, 0))

        print(f"\nRunning simulation...")
        print(f"  Burn-in: {self._burnin_open.date()} → {record_from.date()}")
        print(f"  Dataset: {record_from.date()} → {end_date}")
        print(f"  Bonds:   {len(self.bonds)}, Clients: {len(self.clients)}")

        # Initialise MMPP clocks at t=0 (burn-in open)
        self.mmpp.reset_clocks(t0=0.0)

        current_dt  = self._burnin_open
        recording   = False
        last_day    = None
        rfq_count   = 0

        total_days = cal.n_burnin_days + cal.n_dataset_days
        pbar = tqdm(total=total_days, desc="Trading days", unit="day")

        while current_dt < end_dt:
            t_sim = self._dt_to_sim_seconds(current_dt)
            today = current_dt.date()

            # ── Daily bookkeeping ──────────────────────────────────────
            if today != last_day and cal.is_trading_second(current_dt):
                last_day = today
                pbar.update(1)

                # Step program states and try entry for all clients
                for c in self.clients.clients:
                    self.clients.step_day(c, self.rng)

            # ── Start recording? ───────────────────────────────────────
            if not recording and current_dt >= record_from:
                recording = True
                print(f"\n  Burn-in done. Recording from {current_dt.date()}")

            # ── Calendar multiplier ────────────────────────────────────
            h_t = cal.calendar_multiplier(current_dt)

            # ── Next MMPP transition ───────────────────────────────────
            mmpp_t_sim, mmpp_sector = self.mmpp.next_event()
            mmpp_dt = self._burnin_open + timedelta(seconds=mmpp_t_sim)
            mmpp_dt = cal.advance_clock(mmpp_dt)

            # ── Next RFQ arrival ───────────────────────────────────────
            dt_rfq, k, n, side = self.arrivals.next_arrival(h_t, t_sim)
            rfq_dt = cal.add_trading_seconds(current_dt, dt_rfq)

            # ── Which event is first? ──────────────────────────────────
            if mmpp_dt <= rfq_dt:
                # ── MMPP transition ────────────────────────────────────
                advance_to = mmpp_dt

                dt_s = (advance_to - current_dt).total_seconds()
                if dt_s > 0:
                    self._step_prices(dt_s, advance_to, h_t)

                t_advance = self._dt_to_sim_seconds(advance_to)
                self.mmpp.fire(mmpp_sector, t_advance)
                current_dt = advance_to

            else:
                # ── RFQ arrival ────────────────────────────────────────
                advance_to = rfq_dt

                dt_s = (advance_to - current_dt).total_seconds()
                if dt_s > 0:
                    self._step_prices(dt_s, advance_to, h_t)

                current_dt = advance_to

                # Fetch current market state
                bond   = self.bonds[n]
                client = self.clients[k]
                mid    = self.prices.mid(n)
                spread = self.prices.spread(n)
                vol    = self.prices.vol(n)
                inv_n  = float(self.inv.I[n])
                mstate = self.mmpp.state(bond.sector)

                # Compute trader's quote
                delta = self.quoting.quote(
                    bond, client, side, spread, vol, inv_n,
                    rng=self._child_rng(),
                )

                # Simulate RFQ lifetime to get mid at close
                lifetime_s = float(np.clip(
                    np.exp(self.rng.normal(np.log(180.0), 0.60)),
                    30.0, 600.0,
                ))
                close_vol   = vol * np.sqrt(lifetime_s / SESSION_SECONDS)
                mid_close   = mid + float(self.rng.normal(0.0, close_vol))

                # Mid 30-min after fill (for adverse selection)
                pt_vol    = vol * np.sqrt(1800.0 / SESSION_SECONDS)
                mid_30min = mid_close + float(self.rng.normal(0.0, pt_vol))

                # h(t) at close time (approximate — same day)
                h_close = float(cal.h(advance_to))

                # Is this sector in an imbalanced MMPP state?
                imbalanced = mstate in (1, 2)

                # True outcome probabilities
                n_comp = max(1, int(self.rng.poisson(
                    client.auction_aggressiveness * self.cfg.clients.mean_competing_dealers
                )) + 1)
                size_mm = float(np.clip(
                    round(np.exp(self.rng.normal(client.size_mu, client.size_sigma))),
                    self.cfg.clients.lot_size_mm,
                    bond.outstanding_mm * self.cfg.clients.max_notional_fraction,
                ))

                p_win, p_loss, p_cancel, p_expire = self.outcomes.compute_probs(
                    k, n, client,
                    delta, spread, size_mm, n_comp,
                    mid, mid_close, vol, h_close, imbalanced, side,
                )

                # Resolve outcome
                outcome_rec = self.outcomes.resolve(
                    rfq_id=self.arrivals._counter + 1,
                    client_id=k, bond_id=n, client=client,
                    delta=delta, delta0=spread, size_mm=size_mm,
                    n_competing=n_comp, mid_request=mid, mid_close=mid_close,
                    mid_30min=mid_30min, garch_vol=vol, h_t=h_close,
                    is_imbalanced=imbalanced, side=side,
                )
                self.arrivals._counter += 1

                # Update inventory on fill
                if outcome_rec.outcome == "WIN":
                    self.inv.fill(n, side, size_mm)

                # Stochastic hedge
                self.inv.try_hedge(n, self.rng)

                # Record if past burn-in
                if recording:
                    rfq_count += 1
                    self._record(
                        rfq_id=outcome_rec.rfq_id,
                        client_id=k, bond_id=n, side=side,
                        timestamp=current_dt, size_mm=size_mm,
                        mid=mid, spread=spread, delta=delta,
                        lifetime_s=lifetime_s, mid_close=outcome_rec.mid_at_close,
                        outcome=outcome_rec.outcome, inv_n=inv_n,
                        post_trade_mid=outcome_rec.post_trade_mid,
                        bond=bond, client=client, mmpp_state=mstate,
                        in_program=client.in_program, n_comp=n_comp,
                        p_win=outcome_rec.p_win, p_cancel=outcome_rec.p_cancel,
                        p_expire=outcome_rec.p_expire,
                        rho_k=client.rho_k, alpha_k=client.alpha_k,
                        beta_k=client.beta_k,
                        affinity=float(self.clients.affinity.values[k, n]),
                        informed=outcome_rec.informed_move,
                    )
                    if rfq_count % 2000 == 0:
                        print(f"\r  {rfq_count:,} RFQs recorded...", end="", flush=True)

        pbar.close()
        print(f"\n  Done. {rfq_count:,} RFQs recorded.")

        obs_df = pd.DataFrame(self._obs)
        gt_df  = pd.DataFrame(self._gt)

        # Add train/test split label
        if len(obs_df):
            train_end = pd.Timestamp(cfg.calendar.train_end)
            obs_df["split"] = np.where(
                pd.to_datetime(obs_df["timestamp"]) <= train_end,
                "train", "test",
            )

        self._print_summary(obs_df)
        return obs_df, gt_df

    # ------------------------------------------------------------------
    # Price stepping helper
    # ------------------------------------------------------------------

    def _step_prices(self, dt_s: float, at_dt: datetime, h_t: float):
        imbalances = {
            s: self.mmpp.imbalance(s)
            for s in self.cfg.mmpp.sector_params
        }
        self.prices.step(dt_s, imbalances, self.inv.I, h_t)

    # ------------------------------------------------------------------
    # Record helpers
    # ------------------------------------------------------------------

    def _record(self, *, rfq_id, client_id, bond_id, side, timestamp,
                size_mm, mid, spread, delta, lifetime_s, mid_close,
                outcome, inv_n, post_trade_mid, bond, client, mmpp_state,
                in_program, n_comp, p_win, p_cancel, p_expire,
                rho_k, alpha_k, beta_k, affinity, informed):

        self._obs.append({
            "rfq_id":               rfq_id,
            "timestamp":            timestamp.isoformat(),
            "client_id":            client_id,
            "bond_id":              bond_id,
            "side":                 side,
            "size_mm":              round(size_mm, 2),
            "mid_at_request":       round(mid, 4),
            "spread_at_request":    round(spread, 4),
            "quote_delta":          round(delta, 4),
            "rfq_lifetime_s":       round(lifetime_s, 1),
            "mid_at_close":         round(mid_close, 4),
            "price_move_during":    round(mid_close - mid, 4),
            "outcome":              outcome,
            "inventory_at_request": round(inv_n, 2),
            "post_trade_move":      round(post_trade_mid - mid_close, 4)
                                    if outcome == "WIN" and not np.isnan(post_trade_mid)
                                    else None,
            "bond_sector":          bond.sector,
            "bond_rating":          bond.rating,
            "bond_duration":        bond.duration_bucket,
            "bond_tier":            bond.liquidity_tier,
            "bond_issuer":          bond.issuer_id,
            "client_in_program":    in_program,
            "n_competing":          n_comp,
            "mmpp_state":           mmpp_state,
        })

        self._gt.append({
            "rfq_id":       rfq_id,
            "p_star_win":   round(p_win,    4),
            "p_cancel":     round(p_cancel, 4),
            "p_expire":     round(p_expire, 4),
            "rho_k":        round(rho_k,    4),
            "alpha_k":      round(alpha_k,  4),
            "beta_k":       round(beta_k,   4),
            "archetype":    client.archetype,
            "affinity_kn":  round(affinity, 4),
            "informed_move": informed,
        })

    def _print_summary(self, obs_df: pd.DataFrame):
        if len(obs_df) == 0:
            print("  WARNING: no RFQs recorded — check burn-in period.")
            return
        print("\n--- Summary ---")
        print(f"Total RFQs : {len(obs_df):,}")
        if "split" in obs_df.columns:
            print(obs_df["split"].value_counts().to_string())
        print("\nOutcome distribution:")
        print(obs_df["outcome"].value_counts(normalize=True).round(3).to_string())
        hit = (obs_df["outcome"] == "WIN").mean()
        print(f"\nHit rate   : {hit:.2%}  (target 5–7 %)")
        print("-" * 30)
