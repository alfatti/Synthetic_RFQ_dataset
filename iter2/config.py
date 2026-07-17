"""Configuration for the synthetic IG-corp RFQ bandit dataset.

All knobs for the locked design live here. Defaults reflect the agreed spec:
40k CUSIPs, 5k clients, |A|=500 spread ticks, |C|=25 bands, two outcomes
(CLIENT-TRADED / CLIENT-TRADED-AWAY) at ~5% hit rate, reward = spread
captured minus adverse-selection markout on wins, cover<->markout coupling
(winner's curse), factor risk model, hidden tiers/regime/trader-bias eggs.
"""
from dataclasses import dataclass, field


@dataclass
class GenConfig:
    seed: int = 20260716

    # ---- scale ----
    n_train: int = 1_000_000
    n_test: int = 200_000
    n_bonds: int = 40_000
    n_issuers: int = 1_200
    n_clients: int = 5_000
    n_traders: int = 12
    n_days: int = 250            # ~1 trading year
    n_sectors: int = 11
    n_factors: int = 15          # 6 key-rate + 8 sector-ish + 1 market

    # ---- action space ----
    n_actions: int = 500
    tick_bps: float = 0.25       # grid: 0.25 .. 125.0 bps quoted spread
    n_bands: int = 25            # true clusters: contiguous 5-bp bands

    # ---- outcomes / reward ----
    target_hit_rate: float = 0.05
    kappa_toxic: float = 0.35    # a1: cover<->markout coupling (winner's curse)
    kappa_level: float = 0.80    # covers back off in *level* vs expected toxicity
    cost_bps: float = 0.0        # optional fixed per-ticket cost (paper's c^{i,n})
    reveal_cover_on_win: bool = False  # Option-A censored observable (off by default)
    binary_reward: bool = False  # degenerate toggle: r = 1{TRADED}

    # ---- logging policy ----
    logging_quality: float = 0.15    # weight on oracle argmax vs trader heuristic
    sigma_quote_bps: float = 1.25    # dispersion of softmax around center
    support_halfwidth_ticks: int = 60  # hard support window (+/- 15 bps) -> deficiency
    stick_bonus_1bp: float = 0.55    # round-tick logit bonus (whole bps)
    stick_bonus_5bp: float = 0.65    # extra bonus on 5-bp super-round ticks
    heuristic_noise_bps: float = 1.5
    illiq_offset_bps: float = 2.2    # illiquidity tilt, in cover-dispersion units

    # ---- counterfactual doubles ----
    k_counterfactuals: int = 8

    # ---- hidden-structure knobs ----
    tier_probs: tuple = (0.35, 0.30, 0.20, 0.10, 0.05)
    tier_sharpness_bps: tuple = (0.2, 0.5, 1.0, 2.0, 4.0)  # mean adverse selection
    p_stay_regime: float = 0.97
    marketing_tier_noise: float = 0.25
    cluster_noise_levels: tuple = (0.10, 0.30)

    # ---- io ----
    out_dir: str = "/home/claude/out"
    chunk: int = 50_000

    @property
    def grid_bps(self):
        import numpy as np
        return (np.arange(self.n_actions, dtype=np.float64) + 1.0) * self.tick_bps

    @property
    def band_of_action(self):
        import numpy as np
        per = self.n_actions // self.n_bands
        return (np.arange(self.n_actions) // per).astype(np.int16)
