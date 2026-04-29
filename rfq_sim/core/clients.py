"""
clients.py
----------
Client universe initialisation.

K clients drawn from 4 archetypes.  Each client gets:

  Latent:   u_k ∈ ℝ³  — preference vector (hidden from models)
  Static:   α_k, β_k, ρ_k, clip-size parameters, auction aggressiveness
  Dynamic:  program-trading state, running ρ̂_k estimate

Affinity  a_kn = u_k · v_n  drives both which bonds the client tends
to request AND how willing they are to accept a quote.  That dual role
is the core collaborative-filtering Easter egg — a model that discovers
it will simultaneously improve arrival prediction and hit-rate prediction.
"""

from __future__ import annotations
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Optional
from scipy.special import expit   # numerically stable sigmoid

from rfq_sim.core.config import ClientConfig
from rfq_sim.core.bonds import BondUniverse

ARCHETYPES = ["sector_specialist", "duration_trader",
              "liquidity_seeker", "generalist"]


@dataclass
class Client:
    client_id: int
    archetype: str

    # Latent preference vector — NEVER exposed to models
    u_k: np.ndarray

    # Demand-curve parameters
    alpha_k: float   # Logit intercept (lower → more willing to trade)
    beta_k:  float   # Price sensitivity (must be > 0)

    # Adverse-selection probability  ρ_k ∈ [0, 1]
    rho_k: float

    # Clip-size log-normal parameters (log $MM)
    size_mu:    float
    size_sigma: float

    # Auction and expiry
    auction_aggressiveness: float   # Controls number of competing dealers
    expiry_propensity:      float   # Baseline expiry hazard

    # Arrival rate: RFQs per trading day, before affinity weighting
    base_arrival_rate: float

    # Program-trade state — mutated in-place by the simulator
    in_program:      bool       = False
    prog_days_left:  int        = 0
    prog_direction:  int        = 0     # +1 buy, -1 sell
    prog_bonds:      List[int]  = field(default_factory=list)

    # Trader's running estimate of ρ_k (starts at 0.10, updated after each fill)
    rho_hat: float = 0.10


@dataclass
class AffinityMatrix:
    """Pre-computed a_kn = u_k · v_n and softmax bond-selection probabilities."""
    values:        np.ndarray   # (K, N) raw dot products
    softmax_probs: np.ndarray   # (K, N) after softmax with temperature τ


class ClientUniverse:
    """
    Generates all K clients and pre-computes the affinity matrix.
    """

    def __init__(
        self,
        cfg:   ClientConfig,
        bonds: BondUniverse,
        rng:   np.random.Generator,
    ):
        self.cfg   = cfg
        self.bonds = bonds
        self.rng   = rng
        self.clients: List[Client] = []
        self.affinity: Optional[AffinityMatrix] = None
        self._build()

    def _build(self):
        self._draw_clients()
        self._compute_affinity()

    # ------------------------------------------------------------------
    # Client generation
    # ------------------------------------------------------------------

    def _draw_clients(self):
        cfg = self.cfg
        K   = cfg.n_clients

        arch_labels = self.rng.choice(ARCHETYPES, size=K, p=cfg.archetype_weights)

        arch_u_means = {
            arch: np.array(mean)
            for arch, mean in zip(ARCHETYPES, cfg.archetype_u_means)
        }

        for k, arch in enumerate(arch_labels):
            # 1. Latent preference vector
            u_k = self.rng.normal(arch_u_means[arch], cfg.archetype_u_std)
            u_k = np.clip(u_k, 0.01, None).astype(np.float32)

            # 2. Demand-curve intercept α_k
            alpha_k = float(self.rng.normal(
                cfg.archetype_alpha[arch], cfg.alpha_noise_std
            ))

            # 3. Price sensitivity β_k — truncated normal, always > 0
            beta_k = -1.0
            while beta_k <= 0.0:
                beta_k = float(self.rng.normal(cfg.beta_mean, cfg.beta_std))

            # 4. Informativeness ρ_k via logit-normal
            rho_logit = (cfg.archetype_rho_logit[arch]
                         + float(self.rng.normal(0.0, cfg.rho_noise_std)))
            rho_k = float(expit(rho_logit))

            # 5. Clip-size distribution
            size_mu    = cfg.archetype_size_mu[arch]
            size_sigma = cfg.archetype_size_sigma[arch]

            # 6. Auction aggressiveness ∈ (0, 1)
            agg = float(np.clip(
                self.rng.normal(cfg.auction_aggressiveness_mean,
                                cfg.auction_aggressiveness_std),
                0.01, 0.99,
            ))

            # 7. Expiry propensity > 0
            exp_p = float(np.clip(
                self.rng.normal(cfg.expiry_propensity_mean,
                                cfg.expiry_propensity_std),
                0.01, 0.50,
            ))

            # 8. Base arrival rate > 0 (RFQs / trading day)
            arr = float(max(0.5,
                self.rng.normal(cfg.base_arrival_mean, cfg.base_arrival_std)
            ))

            self.clients.append(Client(
                client_id=k, archetype=arch, u_k=u_k,
                alpha_k=alpha_k, beta_k=beta_k, rho_k=rho_k,
                size_mu=size_mu, size_sigma=size_sigma,
                auction_aggressiveness=agg, expiry_propensity=exp_p,
                base_arrival_rate=arr,
            ))

    # ------------------------------------------------------------------
    # Affinity matrix
    # ------------------------------------------------------------------

    def _compute_affinity(self):
        """
        A[k, n] = u_k · v_n

        Then softmax over bonds per client with temperature τ to get
        bond-selection probabilities p^{k,n}.  Low τ → each client
        concentrates on their specialty bonds (more structured data).
        """
        cfg = self.cfg
        U   = np.stack([c.u_k for c in self.clients], axis=0)   # (K, d)
        V   = np.stack([b.v_n for b in self.bonds.bonds], axis=0)  # (N, d)
        A   = (U @ V.T).astype(np.float32)   # (K, N)

        logits = A / cfg.softmax_temperature
        logits -= logits.max(axis=1, keepdims=True)   # numerical stability
        exp_l   = np.exp(logits)
        probs   = (exp_l / exp_l.sum(axis=1, keepdims=True)).astype(np.float32)

        self.affinity = AffinityMatrix(values=A, softmax_probs=probs)

    # ------------------------------------------------------------------
    # Program-trade helpers
    # ------------------------------------------------------------------

    def try_enter_program(
        self, client: Client, rng: np.random.Generator
    ) -> bool:
        """
        Once per trading day, roll to see if client k enters a program state.
        During a program the client fires RFQs on a fixed cluster of similar
        bonds in one direction — that's the serial-correlation Easter egg.
        """
        if client.in_program:
            return False
        if rng.random() > self.cfg.program_entry_rate:
            return False

        duration  = max(1, int(rng.geometric(1.0 / self.cfg.program_duration_mean)))
        direction = int(rng.choice([-1, +1]))

        # Pick the top-affinity bond cluster for this client
        affinities = self.affinity.values[client.client_id]          # (N,)
        n_cluster  = max(2, len(self.bonds) // 5)
        cluster    = list(np.argsort(affinities)[::-1][:n_cluster])

        client.in_program     = True
        client.prog_days_left = duration
        client.prog_direction = direction
        client.prog_bonds     = cluster
        return True

    def step_day(self, client: Client, rng: np.random.Generator):
        """Call at end of each trading day to decrement program counter."""
        if client.in_program:
            client.prog_days_left -= 1
            if client.prog_days_left <= 0:
                client.in_program     = False
                client.prog_days_left = 0
                client.prog_direction = 0
                client.prog_bonds     = []
        # Also try to enter a fresh program state
        self.try_enter_program(client, rng)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    def __len__(self)  -> int:     return len(self.clients)
    def __getitem__(self, i: int): return self.clients[i]

    def to_dataframe(self) -> pd.DataFrame:
        """Observable client metadata."""
        return pd.DataFrame([{
            "client_id":             c.client_id,
            "archetype":             c.archetype,
            "base_arrival_rate":     round(c.base_arrival_rate, 3),
            "auction_aggressiveness": round(c.auction_aggressiveness, 3),
            "expiry_propensity":     round(c.expiry_propensity, 3),
        } for c in self.clients])

    def to_ground_truth_dataframe(self) -> pd.DataFrame:
        """Full client params including latent variables — evaluation only."""
        rows = []
        for c in self.clients:
            row = {
                "client_id": c.client_id,
                "archetype": c.archetype,
                "alpha_k":   round(c.alpha_k, 4),
                "beta_k":    round(c.beta_k,  4),
                "rho_k":     round(c.rho_k,   4),
                "size_mu":   round(c.size_mu,  4),
                "size_sigma": round(c.size_sigma, 4),
            }
            for d in range(self.cfg.latent_dim):
                row[f"u_{d}"] = round(float(c.u_k[d]), 4)
            rows.append(row)
        return pd.DataFrame(rows)
