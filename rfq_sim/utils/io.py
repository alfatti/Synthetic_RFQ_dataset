"""utils/io.py — save and load simulation outputs.

Updated for the N=10k bonds.py rewrite: BondUniverse no longer holds a
dense similarity_matrix.  Instead we persist the four feature index arrays
(issuer / sector / rating / duration), which are all that's needed to
reconstruct any similarity value on demand via:
    bonds.similarity_pair(i, j)
    bonds.similarity_row(i)
    bonds.spillover_matvec(eps)

Any value Sigma[i,j] = w_issuer*1[issuer_i==issuer_j] + ... can be rebuilt
from these arrays plus the weights in config.json, so nothing is lost by
not storing the full matrix.
"""

import os
import json
import numpy as np
import pandas as pd

from rfq_sim.core.config import SimConfig
from rfq_sim.core.bonds import BondUniverse
from rfq_sim.core.clients import ClientUniverse


def save_simulation(output_dir, obs_df, gt_df, bonds, clients, cfg):
    os.makedirs(os.path.join(output_dir, "processed"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "raw"),       exist_ok=True)

    # ── Main datasets ────────────────────────────────────────────────
    obs_df.to_parquet(
        os.path.join(output_dir, "processed", "observable.parquet"), index=False)
    gt_df.to_parquet(
        os.path.join(output_dir, "raw", "ground_truth.parquet"), index=False)

    # ── Bond metadata ────────────────────────────────────────────────
    bonds.to_dataframe().to_parquet(
        os.path.join(output_dir, "processed", "bond_metadata.parquet"), index=False)
    bonds.to_ground_truth_dataframe().to_parquet(
        os.path.join(output_dir, "raw", "bond_gt.parquet"), index=False)

    # ── Client metadata ──────────────────────────────────────────────
    clients.to_dataframe().to_parquet(
        os.path.join(output_dir, "processed", "client_metadata.parquet"), index=False)
    clients.to_ground_truth_dataframe().to_parquet(
        os.path.join(output_dir, "raw", "client_gt.parquet"), index=False)

    # ── Similarity structure ─────────────────────────────────────────
    # The dense N×N similarity matrix is no longer stored (it is 400 MB at
    # N=10k and not even truly sparse — the same-sector block alone is huge).
    # Instead we persist the four feature index arrays.  Together with the
    # sim_w_* weights in config.json these fully determine Sigma[i,j] for any
    # pair, and BondUniverse.spillover_matvec reconstructs Sigma @ eps in O(N).
    np.save(os.path.join(output_dir, "raw", "issuer_ids.npy"), bonds._issuer_ids)
    np.save(os.path.join(output_dir, "raw", "sector_ids.npy"), bonds._sector_ids)
    np.save(os.path.join(output_dir, "raw", "rating_ids.npy"), bonds._rating_ids)
    np.save(os.path.join(output_dir, "raw", "dur_ids.npy"),    bonds._dur_ids)

    # ── Affinity matrix ──────────────────────────────────────────────
    # Affinity is (K, N) — at K=1500, N=10k that is 60 MB float32, still fine
    # to store densely.  If this becomes a problem at larger scale, switch to
    # saving U (K×d) and V (N×d) and reconstruct A = U @ V.T on load.
    np.save(os.path.join(output_dir, "raw", "affinity.npy"), clients.affinity.values)

    # ── Config for reproducibility ───────────────────────────────────
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2, default=str)

    print(f"Outputs saved to {output_dir}/")


def load_all(output_dir):
    return {
        "observable":   pd.read_parquet(
            os.path.join(output_dir, "processed", "observable.parquet")),
        "ground_truth": pd.read_parquet(
            os.path.join(output_dir, "raw", "ground_truth.parquet")),
        "bond_meta":    pd.read_parquet(
            os.path.join(output_dir, "processed", "bond_metadata.parquet")),
        "bond_gt":      pd.read_parquet(
            os.path.join(output_dir, "raw", "bond_gt.parquet")),
        "client_meta":  pd.read_parquet(
            os.path.join(output_dir, "processed", "client_metadata.parquet")),
        "client_gt":    pd.read_parquet(
            os.path.join(output_dir, "raw", "client_gt.parquet")),

        # Similarity feature index arrays (replace the old dense matrix).
        # Reconstruct any Sigma[i,j] with the helper below, or use the
        # weights from config.json directly.
        "issuer_ids":   np.load(os.path.join(output_dir, "raw", "issuer_ids.npy")),
        "sector_ids":   np.load(os.path.join(output_dir, "raw", "sector_ids.npy")),
        "rating_ids":   np.load(os.path.join(output_dir, "raw", "rating_ids.npy")),
        "dur_ids":      np.load(os.path.join(output_dir, "raw", "dur_ids.npy")),

        "affinity":     np.load(os.path.join(output_dir, "raw", "affinity.npy")),
    }


def similarity_pair_from_arrays(data, i, j, cfg):
    """
    Reconstruct Sigma[i, j] from the loaded index arrays without a BondUniverse.

    Useful in notebooks / evaluation scripts that only load the parquet+npy
    outputs and don't instantiate the full simulation objects.

    Parameters
    ----------
    data : dict          output of load_all()
    i, j : int           bond indices
    cfg  : SimConfig     for the sim_w_* weights (or pass any object with
                         .bonds.sim_w_issuer etc.)
    """
    if i == j:
        return 1.0
    bcfg = cfg.bonds
    s = 0.0
    if data["issuer_ids"][i] == data["issuer_ids"][j]: s += bcfg.sim_w_issuer
    if data["sector_ids"][i] == data["sector_ids"][j]: s += bcfg.sim_w_sector
    if data["rating_ids"][i] == data["rating_ids"][j]: s += bcfg.sim_w_rating
    if data["dur_ids"][i]    == data["dur_ids"][j]:    s += bcfg.sim_w_duration
    return float(s)
