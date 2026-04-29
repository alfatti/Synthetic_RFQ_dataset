"""
config.py
---------
Single source of truth for every number in the simulation.

I want everything in one place so that reproducing any run is just a
matter of serialising this object alongside the parquet output. No
magic numbers buried in modules — if it's a parameter it lives here.

The MMPP intensity values (lambda1, lambda2, Q matrices) come directly
from Bergault & Gueant (2024) Table 1. Everything else is a design
choice; I've flagged those explicitly.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import dataclasses


# ---------------------------------------------------------------------------
# Calendar
# ---------------------------------------------------------------------------

@dataclass
class CalendarConfig:
    """US HY bond market calendar.  Trading hours = 07:00–17:00 ET,
    following Bergault & Gueant (2024) exactly."""

    burnin_start:  str = "2022-12-15"   # Start the clock here; discard output
    dataset_start: str = "2023-01-03"   # First day we actually record
    dataset_end:   str = "2023-06-30"   # Last day (inclusive)

    # Temporal out-of-time train/test split
    train_end:  str = "2023-03-31"
    test_start: str = "2023-04-03"

    session_open_hour:  int = 7    # 07:00 ET
    session_close_hour: int = 17   # 17:00 ET

    # US Federal Reserve holidays inside the simulation window.
    # The event clock skips these entirely — no RFQs, no price moves.
    holidays: List[str] = field(default_factory=lambda: [
        "2022-12-26",   # Christmas observed  (burn-in period)
        "2023-01-02",   # New Year's observed  (burn-in period)
        "2023-01-16",   # MLK Day
        "2023-02-20",   # Presidents Day
        "2023-04-07",   # Good Friday
        "2023-05-29",   # Memorial Day
        "2023-06-19",   # Juneteenth
    ])

    # Intraday multiplier h(t) = h_min + (1-h_min)*sin²(πτ)*(1 - α_close·1[τ>τ_close])
    h_min:       float = 0.30   # Intensity floor at open/close relative to midday peak
    alpha_close: float = 0.40   # Extra suppression in the final 10 % of the session
    tau_close:   float = 0.90   # Fraction of session where close suppression kicks in

    # Month-end: last 2 business days of each month get this multiplier on top of h(t)
    month_end_gamma: float = 1.30


# ---------------------------------------------------------------------------
# Bond universe
# ---------------------------------------------------------------------------

@dataclass
class BondConfig:
    """Bond universe.  ~3 bonds per issuer to plant the issuer-curve Easter egg."""

    n_bonds:   int = 100
    n_issuers: int = 30     # floor(n_bonds / 3) keeps ~3 bonds per issuer

    sectors:          List[str] = field(default_factory=lambda:
                                   ["Energy", "Financials", "Consumer", "TMT"])
    ratings:          List[str] = field(default_factory=lambda: ["BB", "B", "CCC"])
    duration_buckets: List[str] = field(default_factory=lambda:
                                   ["Short", "Medium", "Long"])
    liquidity_tiers:  List[int] = field(default_factory=lambda: [1, 2, 3])

    # Latent factor dimension (sector loading / duration-quality / liquidity)
    latent_dim: int = 3

    # Similarity matrix weights Σ_nn'
    sim_w_issuer:   float = 0.50
    sim_w_sector:   float = 0.25
    sim_w_rating:   float = 0.15
    sim_w_duration: float = 0.10

    # Factor model: n_common_factors orthogonal factors driving correlated moves
    n_common_factors:    int   = 3
    factor_ar_coeff:     float = 0.95    # VAR(1) persistence
    factor_daily_vol:    float = 0.30    # Std dev of factor shock per trading day
    idio_vol_lo:         float = 0.10    # Lower bound of per-bond daily idio vol
    idio_vol_hi:         float = 0.50    # Upper bound
    price_spillover_eta: float = 0.15    # Cross-bond spillover coefficient η
    kappa_lo:            float = 0.01    # MMPP drift sensitivity range (Table 2)
    kappa_hi:            float = 2.83

    # GARCH(1,1) for idio variance dynamics
    garch_omega: float = 0.05
    garch_alpha: float = 0.10
    garch_beta:  float = 0.84   # omega + alpha + beta < 1 for stationarity

    # Sector-level jump process (double-exponential jumps)
    jump_intensity_per_day: float = 0.02   # Jumps per sector per trading day
    jump_scale:             float = 0.50   # Laplace scale parameter

    # Dynamic bid-ask spread coefficients
    spread_phi_sigma: float = 0.20   # Spread widens with vol
    spread_phi_inv:   float = 0.05   # Spread widens with |inventory|
    spread_phi_tier:  float = 0.30   # Illiquid tier premium

    # Baseline spreads per tier (price points, i.e. cents per $100 face)
    baseline_spread: Dict[int, float] = field(default_factory=lambda:
                                         {1: 0.25, 2: 0.50, 3: 1.00})

    # Outstanding notional by liquidity tier (in $MM); used for clip size cap
    outstanding_by_tier: Dict[int, float] = field(default_factory=lambda:
                                              {1: 750.0, 2: 400.0, 3: 150.0})

    # Initial price is drawn from a uniform range centred around par
    initial_price_lo: float = 85.0
    initial_price_hi: float = 105.0


# ---------------------------------------------------------------------------
# Client universe
# ---------------------------------------------------------------------------

@dataclass
class ClientConfig:
    """
    Client heterogeneity parameters.

    I have 4 archetypes: sector_specialist, duration_trader,
    liquidity_seeker, generalist.  Each archetype has a different
    centroid in latent u_k space AND a different distribution over
    alpha_k, beta_k, rho_k.  That multi-dimensional correlation is
    what makes the dataset interesting for collaborative filtering.
    """

    n_clients:  int   = 150
    latent_dim: int   = 3

    # Archetype mixing probabilities (must sum to 1)
    archetype_weights: List[float] = field(default_factory=lambda:
                                     [0.30, 0.25, 0.20, 0.25])
    # Order: sector_specialist, duration_trader, liquidity_seeker, generalist

    # Archetype centroids in latent factor space
    archetype_u_means: List[List[float]] = field(default_factory=lambda: [
        [1.5, 0.2, 0.2],   # sector_specialist — concentrated on v_n^(1)
        [0.2, 1.5, 0.2],   # duration_trader  — concentrated on v_n^(2)
        [0.2, 0.2, 1.5],   # liquidity_seeker — concentrated on v_n^(3)
        [0.5, 0.5, 0.5],   # generalist       — diffuse
    ])
    archetype_u_std: float = 0.30   # Noise around archetype centroid

    # Demand curve intercept α_k by archetype.
    # Lower (more negative) → more willing to accept a quote at fair value.
    # Calibrated so the unconditional hit rate lands in 5–7 %.
    # Mean alpha ~ 1.74 is calibrated to produce ~6% unconditional hit rate.
    # High-affinity client-bond pairs win more often (alpha is reduced by
    # phi * affinity), which pulls the aggregate up, so the baseline needs
    # to be set conservatively high.
    archetype_alpha: Dict[str, float] = field(default_factory=lambda: {
        "sector_specialist":  1.25,   # Most willing — relationship client
        "duration_trader":    1.95,   # Arms-length — moderate
        "liquidity_seeker":   2.95,   # Tough to win — shops aggressively
        "generalist":         2.15,   # Broad but not deeply committed
    })
    alpha_noise_std: float = 0.40

    # Price sensitivity β_k — truncated normal, mean from Bergault & Gueant (2024)
    beta_mean: float = 3.10
    beta_std:  float = 0.80

    # Informativeness ρ_k — logit-normal; sector specialists and liquidity
    # seekers are more informed (higher ρ).  This is the ground truth for
    # the adverse-selection Easter egg.
    archetype_rho_logit: Dict[str, float] = field(default_factory=lambda: {
        "sector_specialist": 1.00,
        "duration_trader":   0.50,
        "liquidity_seeker":  0.80,
        "generalist":       -0.50,
    })
    rho_noise_std: float = 0.30

    # Affinity modulation of demand curve
    # η = α_k - φ·a_kn + β_k·(1 - ψ·a_kn)·(δ/δ⁰) + ...
    phi_affinity_alpha: float = 0.40   # High affinity lowers the intercept
    psi_affinity_beta:  float = 0.15   # High affinity reduces price sensitivity

    # Softmax temperature for bond selection p^{k,n} = softmax(a_kn / τ)
    softmax_temperature: float = 1.00

    # Clip size: log-normal parameters by archetype (log $MM)
    archetype_size_mu: Dict[str, float] = field(default_factory=lambda: {
        "sector_specialist": 1.80,   # exp(1.80) ≈ 6 MM
        "duration_trader":   1.60,
        "liquidity_seeker":  2.30,   # exp(2.30) ≈ 10 MM — large clips
        "generalist":        1.20,
    })
    archetype_size_sigma: Dict[str, float] = field(default_factory=lambda: {
        "sector_specialist": 0.50,
        "duration_trader":   0.60,
        "liquidity_seeker":  0.70,
        "generalist":        0.80,
    })
    lot_size_mm:              float = 1.0    # Round to nearest $1 MM
    max_notional_fraction:    float = 0.08   # Cap at 8 % of outstanding

    # Auction: number of competing dealers is Poisson(aggressiveness * mean_dealers)
    auction_aggressiveness_mean: float = 0.60
    auction_aggressiveness_std:  float = 0.20
    mean_competing_dealers:      float = 2.50

    # Expiry propensity
    expiry_propensity_mean: float = 0.10
    expiry_propensity_std:  float = 0.05

    # Program trading
    program_entry_rate:    float = 0.04    # Per-day probability of entering a program
    program_duration_mean: int   = 3       # Geometric mean duration in trading days
    program_intensity_mult: float = 3.0    # Arrival rate multiplier during program
    program_alpha_boost:    float = -0.50  # Shift in α_k during program (more urgent)

    # Base RFQ arrival rate in RFQs per trading day (before affinity weighting)
    # A "trading day" here is a notional 36,000-second session.
    # With K=150 clients and mean 4 RFQs/day each, that is 600 total RFQs/day
    # across N=100 bonds → ~5 RFQs/day per bond → realistic HY activity.
    base_arrival_mean: float = 4.0
    base_arrival_std:  float = 2.0

    # Flow spillover: when bond n is queried, boost arrival rates for similar bonds
    spillover_strength:   float = 0.30    # Multiplier on Σ_nn' for the boost
    spillover_window_min: int   = 30      # Minimum duration of boost (minutes)
    spillover_window_max: int   = 60      # Maximum duration


# ---------------------------------------------------------------------------
# MMPP — calibrated directly from Bergault & Gueant (2024) Table 1
# ---------------------------------------------------------------------------

@dataclass
class MMPPConfig:
    """
    Bidimensional MMPP per sector.  State space = {low, high}² for the
    (bid, ask) intensity pair, giving 4 joint states.

    Q matrices are taken directly from Table 1 of the paper.  I'm
    keeping all four sectors so the dataset has heterogeneous liquidity
    regimes — Energy/Consumer are more liquid than TMT.
    """

    # All rates are in units of day⁻¹ (trading days, 10-hour sessions)
    sector_params: Dict[str, Dict] = field(default_factory=lambda: {
        "Energy": {
            "lambda_lo": 10.83,   # Low-intensity state  (Sector 1 analogue)
            "lambda_hi": 73.03,   # High-intensity state
            # 4×4 generator Q in state order: (lo,lo),(lo,hi),(hi,lo),(hi,hi)
            "Q": [
                [-14.01,  4.37,  4.37,  5.27],
                [ 19.32, -60.91, 12.54, 29.05],
                [ 19.32,  12.54, -60.91, 29.05],
                [ 23.67,  15.00, 15.00, -53.67],
            ],
        },
        "Financials": {
            "lambda_lo":  8.44,
            "lambda_hi": 58.28,
            "Q": [
                [ -4.55,  1.00,  1.00,  2.55],
                [ 18.53, -28.31,  0.13,  9.65],
                [ 18.53,   0.13, -28.31,  9.65],
                [ 14.77,  16.73, 16.73, -48.23],
            ],
        },
        "Consumer": {
            "lambda_lo": 15.73,
            "lambda_hi": 81.78,
            "Q": [
                [ -9.98,  2.79,  2.79,  4.40],
                [ 20.53, -23.73,  0.02,  3.18],
                [ 20.53,   0.02, -23.73,  3.18],
                [  9.87,   4.17,  4.17, -18.21],
            ],
        },
        "TMT": {
            "lambda_lo":  7.33,   # Sector 4 analogue — illiquid / slow
            "lambda_hi": 28.32,
            "Q": [
                [-1.67,  0.48,  0.48,  0.71],
                [ 1.92, -2.02,  0.00,  0.10],
                [ 1.92,  0.00, -2.02,  0.10],
                [ 0.84,  0.11,  0.11, -1.06],
            ],
        },
    })


# ---------------------------------------------------------------------------
# Quoting model
# ---------------------------------------------------------------------------

@dataclass
class QuotingConfig:
    """
    Ground-truth quoting policy.
    δ = ½ δ⁰ − α_inv · I · σ · dir + α_info · ρ̂_k + ε
    A policy that beats this has found genuine signal.
    """
    alpha_inventory: float = 0.10   # Inventory skew coefficient
    alpha_info:      float = 0.20   # Adverse-selection widening
    quote_noise_std: float = 0.02   # Jitter so the policy isn't perfectly deterministic


# ---------------------------------------------------------------------------
# Outcome model
# ---------------------------------------------------------------------------

@dataclass
class OutcomeConfig:
    """
    Multinomial outcome: WIN / LOSS / CANCELLED / EXPIRED.
    Calibrate α_bar = +1.2 to hit 5–7 % unconditional WIN rate
    when β̄ = 3.1 and normalised spread δ/δ⁰ is near 1.
    """
    # Size and competition penalties on the logit
    gamma_size: float = 0.20    # log(size / size0) coefficient
    gamma_comp: float = 0.15    # n_competing_dealers coefficient

    # Cancellation model
    cancel_base_logit:    float = -2.00   # Low baseline cancel rate
    cancel_rho_scale:     float =  2.50   # rho_k amplifies cancel rate
    cancel_riskon_shift:  float =  0.80   # Extra cancel in risk-off on sell side

    # Expiry hazard
    expire_base_logit: float = -2.50

    # Post-trade window for adverse move observation (seconds)
    post_trade_window_s: int = 1800   # 30 minutes

    # Adverse move distribution parameters (price points)
    adverse_move_mean: float = 0.20
    adverse_move_std:  float = 0.10


# ---------------------------------------------------------------------------
# Master config
# ---------------------------------------------------------------------------

@dataclass
class SimConfig:
    """
    Master config object — pass this through the whole simulation.
    Serialize to JSON alongside each output directory so any run is
    fully reproducible from the directory alone.
    """
    seed:     int           = 42
    calendar: CalendarConfig = field(default_factory=CalendarConfig)
    bonds:    BondConfig     = field(default_factory=BondConfig)
    clients:  ClientConfig   = field(default_factory=ClientConfig)
    mmpp:     MMPPConfig     = field(default_factory=MMPPConfig)
    quoting:  QuotingConfig  = field(default_factory=QuotingConfig)
    outcomes: OutcomeConfig  = field(default_factory=OutcomeConfig)

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)
