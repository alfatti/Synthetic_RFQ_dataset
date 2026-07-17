"""Exact ground-truth expected reward and policy-value utilities.

Per-RFQ sufficient statistics (m_c, s_c, mu_delta, a1) fully determine the
expected reward over the whole action grid, so the full q(x,.) matrix never
needs to be materialised on disk.

Model (per RFQ, conditional on context x and hidden state):
  cover     dbar  ~ Logistic(m_c, s_c)              best competing spread
  markout   Delta = mu_delta + a1*(dbar - m_c) + eps,  E[eps|x] = 0
  outcome   TRADED iff quoted spread delta < dbar
  reward    r = (delta - Delta - cost) * 1{TRADED}   (bps of notional)

Closed form, with u = (delta - m_c)/s_c and sig(t) = 1/(1+e^-t):
  P(win)              = sig(-u)
  E[(dbar-m_c) 1{win}] = s_c * ( u*sig(-u) + softplus(-u) )
  q(x, delta) = (delta - cost - mu_delta) * sig(-u)
                - a1 * s_c * ( u*sig(-u) + softplus(-u) )

The second term is the winner's-curse penalty: winning against a wide cover
means the rest of the street backed off, i.e. the flow was toxic.
"""
import numpy as np


def _sigmoid(x):
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def _softplus(x):
    return np.logaddexp(0.0, x)


def true_q(m_c, s_c, mu_delta, a1, grid_bps, cost_bps=0.0):
    """Exact E[reward | x, action] for every action on the grid.

    Args broadcast: m_c, s_c, mu_delta, a1 are (n,) arrays (or scalars),
    grid_bps is (A,). Returns (n, A) float64 (cast down by caller if wanted).
    """
    m = np.atleast_1d(np.asarray(m_c, dtype=np.float64))[:, None]
    s = np.atleast_1d(np.asarray(s_c, dtype=np.float64))[:, None]
    mu = np.atleast_1d(np.asarray(mu_delta, dtype=np.float64))[:, None]
    a1 = np.asarray(a1, dtype=np.float64)
    a1 = a1[:, None] if a1.ndim == 1 else a1
    g = np.asarray(grid_bps, dtype=np.float64)[None, :]

    u = (g - m) / s
    p_win = _sigmoid(-u)
    curse = s * (u * p_win + _softplus(-u))
    return (g - cost_bps - mu) * p_win - a1 * curse


def win_prob(m_c, s_c, grid_bps):
    m = np.atleast_1d(np.asarray(m_c, dtype=np.float64))[:, None]
    s = np.atleast_1d(np.asarray(s_c, dtype=np.float64))[:, None]
    g = np.asarray(grid_bps, dtype=np.float64)[None, :]
    return _sigmoid(-(g - m) / s)


def oracle(truth_df, grid_bps, cost_bps=0.0, chunk=50_000):
    """Best action index and its value per row of a sufficient-stats frame."""
    n = len(truth_df)
    best_a = np.empty(n, dtype=np.int32)
    best_q = np.empty(n, dtype=np.float64)
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        q = true_q(truth_df["m_c"].values[lo:hi], truth_df["s_c"].values[lo:hi],
                   truth_df["mu_delta"].values[lo:hi], truth_df["a1"].values[lo:hi],
                   grid_bps, cost_bps)
        best_a[lo:hi] = np.argmax(q, axis=1)
        best_q[lo:hi] = q[np.arange(hi - lo), best_a[lo:hi]]
    return best_a, best_q


def policy_value(action_probs_fn, truth_df, grid_bps, cost_bps=0.0, chunk=50_000):
    """Exact V(pi) = mean_x sum_a pi(a|x) q(x,a).

    action_probs_fn(lo, hi) must return an (hi-lo, A) matrix of action
    probabilities for rows [lo, hi) of truth_df (rows aligned with the
    evaluation contexts). Deterministic policies: return one-hot rows.
    """
    n, total = len(truth_df), 0.0
    for lo in range(0, n, chunk):
        hi = min(lo + chunk, n)
        q = true_q(truth_df["m_c"].values[lo:hi], truth_df["s_c"].values[lo:hi],
                   truth_df["mu_delta"].values[lo:hi], truth_df["a1"].values[lo:hi],
                   grid_bps, cost_bps)
        p = np.asarray(action_probs_fn(lo, hi), dtype=np.float64)
        total += float((p * q).sum())
    return total / n


def logged_policy_value(truth_df, grid_bps, pscore_matrix_fn, cost_bps=0.0,
                        chunk=50_000):
    """V(pi0) when the full logging pmf can be reconstructed (see generator)."""
    return policy_value(pscore_matrix_fn, truth_df, grid_bps, cost_bps, chunk)
