"""utils/io.py — save and load simulation outputs."""

import os, json
import numpy as np
import pandas as pd
from rfq_sim.core.config import SimConfig
from rfq_sim.core.bonds import BondUniverse
from rfq_sim.core.clients import ClientUniverse


def save_simulation(output_dir, obs_df, gt_df, bonds, clients, cfg):
    os.makedirs(os.path.join(output_dir, "processed"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "raw"),       exist_ok=True)

    obs_df.to_parquet(os.path.join(output_dir, "processed", "observable.parquet"), index=False)
    gt_df.to_parquet( os.path.join(output_dir, "raw",       "ground_truth.parquet"), index=False)

    bonds.to_dataframe().to_parquet(
        os.path.join(output_dir, "processed", "bond_metadata.parquet"), index=False)
    bonds.to_ground_truth_dataframe().to_parquet(
        os.path.join(output_dir, "raw", "bond_gt.parquet"), index=False)

    clients.to_dataframe().to_parquet(
        os.path.join(output_dir, "processed", "client_metadata.parquet"), index=False)
    clients.to_ground_truth_dataframe().to_parquet(
        os.path.join(output_dir, "raw", "client_gt.parquet"), index=False)

    np.save(os.path.join(output_dir, "raw", "similarity.npy"), bonds.similarity_matrix)
    np.save(os.path.join(output_dir, "raw", "affinity.npy"),   clients.affinity.values)

    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(cfg.to_dict(), f, indent=2, default=str)

    print(f"Outputs saved to {output_dir}/")


def load_all(output_dir):
    return {
        "observable":  pd.read_parquet(os.path.join(output_dir, "processed", "observable.parquet")),
        "ground_truth": pd.read_parquet(os.path.join(output_dir, "raw", "ground_truth.parquet")),
        "bond_meta":   pd.read_parquet(os.path.join(output_dir, "processed", "bond_metadata.parquet")),
        "bond_gt":     pd.read_parquet(os.path.join(output_dir, "raw", "bond_gt.parquet")),
        "client_meta": pd.read_parquet(os.path.join(output_dir, "processed", "client_metadata.parquet")),
        "client_gt":   pd.read_parquet(os.path.join(output_dir, "raw", "client_gt.parquet")),
        "similarity":  np.load(os.path.join(output_dir, "raw", "similarity.npy")),
        "affinity":    np.load(os.path.join(output_dir, "raw", "affinity.npy")),
    }
