"""
outcomes.py
-----------
Outcome model: WIN / LOSS / CANCELLED / EXPIRED.

The demand curve follows Bergault & Gueant (2024): a logistic function
of the normalised spread δ/δ⁰.  I layer affinity-modulated parameters
on top so that high-affinity client–bond pairs have higher hit rates
even at the same normalised spread.

Cancellation embeds the key adverse-selection Easter egg: clients with
high ρ_k cancel more when prices move adversely during the RFQ window.
A model that learns to predict cancellations from observable features
(price move during RFQ, client cancel history) is implicitly recovering
the latent ρ_k signal.
"""

from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from scipy.special import expit   # numerically stable sigmoid

from rfq_sim.core.config import OutcomeConfig, ClientConfig
from rfq_sim.core.clients import Client, AffinityMatrix


@dataclass
class OutcomeRecord:
    rfq_id:        int
    outcome:       str     # WIN / LOSS / CANCELLED / EXPIRED
    mid_at_close:  float   # Mid price when the RFQ lifetime expired

    # Ground truth probabilities — stored for calibration evaluation
    p_win:    float
    p_cancel: float
    p_expire: float

    # Adverse-selection flag (WIN only)
    informed_move:  bool
    post_trade_mid: float   # Mid 30-min after fill (WIN only; NaN otherwise)


class OutcomeModel:
    """
    Resolves each RFQ event to a labelled outcome.

    The true probabilities are computed first and stored in the ground-truth
    table.  The actual outcome is then a single multinomial draw.
    """

    def __init__(
        self,
        cfg:      OutcomeConfig,
        cli_cfg:  ClientConfig,
        affinity: AffinityMatrix,
        rng:      np.random.Generator,
    ):
        self.cfg      = cfg
        self.cli_cfg  = cli_cfg
        self.affinity = affinity
        self.rng      = rng

    def compute_probs(
        self,
        client_id:    int,
        bond_id:      int,
        client:       Client,
        delta:        float,   # Trader's half-spread quote
        delta0:       float,   # Current market spread
        size_mm:      float,
        n_competing:  int,
        mid_request:  float,
        mid_close:    float,   # Mid at RFQ expiry
        garch_vol:    float,
        h_t:          float,   # Intraday intensity multiplier at close time
        is_imbalanced: bool,   # True when MMPP state is lo_hi or hi_lo
        side:         str,
    ):
        """
        Return (p_win, p_loss, p_cancel, p_expire).

        Core WIN logit:
          η = α_k − φ·a_kn + β_k·(1 − ψ·a_kn)·(δ/δ⁰)
              + γ_size·log(size/5) + γ_comp·n_competing
          p_win = σ(−η)

        CANCEL logit:
          ζ = ζ₀ + ρ_k · scale · |ΔS_intra| / σ_n
        (Informed clients cancel more when the price moves against them.)

        EXPIRE logit:
          ξ = ξ₀ + expiry_propensity + (1 − h_t)·0.5
        """
        cfg_c = self.cli_cfg

        # Client–bond affinity a_kn
        a = float(self.affinity.values[client_id, bond_id])

        # Affinity-modulated demand curve parameters
        alpha_adj = client.alpha_k - cfg_c.phi_affinity_alpha * a
        beta_adj  = max(0.10, client.beta_k * (1.0 - cfg_c.psi_affinity_beta * a))

        # Normalised spread — the Bergault & Gueant S-curve input
        norm_spread = float(delta) / max(float(delta0), 1e-4)

        # Size and competition terms
        size_term = self.cfg.gamma_size * float(np.log(max(size_mm, 0.5) / 5.0))
        comp_term = self.cfg.gamma_comp * float(n_competing)

        eta   = alpha_adj + beta_adj * norm_spread + size_term + comp_term
        p_win = float(expit(-eta))

        # Cancellation: rho_k × normalised intra-RFQ price move
        price_move_norm = abs(mid_close - mid_request) / max(garch_vol * 0.1, 1e-4)
        cancel_logit    = (
            self.cfg.cancel_base_logit
            + self.cfg.cancel_rho_scale * client.rho_k * price_move_norm
        )
        if is_imbalanced and side == "sell":
            cancel_logit += self.cfg.cancel_riskon_shift
        p_cancel_raw = float(expit(cancel_logit))

        # Expiry: higher near close, higher for high-expiry-propensity clients
        expire_logit = (
            self.cfg.expire_base_logit
            + client.expiry_propensity
            + (1.0 - h_t) * 0.50
        )
        p_expire_raw = float(expit(expire_logit))

        # Normalised multinomial vector [p_win, p_loss, p_cancel, p_expire]
        p_not_win  = 1.0 - p_win
        p_cancel   = p_not_win * p_cancel_raw
        p_expire   = p_not_win * p_expire_raw * (1.0 - p_cancel_raw)
        p_loss     = max(0.0, 1.0 - p_win - p_cancel - p_expire)

        total = p_win + p_loss + p_cancel + p_expire
        return (p_win/total, p_loss/total, p_cancel/total, p_expire/total)

    def resolve(
        self,
        rfq_id:       int,
        client_id:    int,
        bond_id:      int,
        client:       Client,
        delta:        float,
        delta0:       float,
        size_mm:      float,
        n_competing:  int,
        mid_request:  float,
        mid_close:    float,
        mid_30min:    float,   # Used for post-trade adverse move
        garch_vol:    float,
        h_t:          float,
        is_imbalanced: bool,
        side:         str,
    ) -> OutcomeRecord:
        """
        Compute true probabilities, draw the outcome, then update the
        client's running ρ̂ estimate on fills.
        """
        p_win, p_loss, p_cancel, p_expire = self.compute_probs(
            client_id, bond_id, client, delta, delta0, size_mm,
            n_competing, mid_request, mid_close, garch_vol,
            h_t, is_imbalanced, side,
        )

        # Single multinomial draw
        u = self.rng.random()
        if   u < p_win:
            outcome = "WIN"
        elif u < p_win + p_loss:
            outcome = "LOSS"
        elif u < p_win + p_loss + p_cancel:
            outcome = "CANCELLED"
        else:
            outcome = "EXPIRED"

        # Post-trade adverse selection (WIN only)
        informed_move = False
        if outcome == "WIN":
            if self.rng.random() < client.rho_k:
                informed_move = True
                # Add adverse drift to post-trade mid
                adv_dir = -1.0 if side == "buy" else +1.0
                adv_mag = float(self.rng.normal(
                    self.cfg.adverse_move_mean,
                    self.cfg.adverse_move_std,
                ))
                mid_30min = float(mid_30min) + adv_dir * adv_mag

            # Update trader's running ρ̂ estimate via exponential smoothing
            client.rho_hat = (
                0.95 * client.rho_hat
                + 0.05 * (1.0 if informed_move else 0.0)
            )

        return OutcomeRecord(
            rfq_id=rfq_id,
            outcome=outcome,
            mid_at_close=float(mid_close),
            p_win=p_win,
            p_cancel=p_cancel,
            p_expire=p_expire,
            informed_move=informed_move,
            post_trade_mid=float(mid_30min) if outcome == "WIN" else float("nan"),
        )
