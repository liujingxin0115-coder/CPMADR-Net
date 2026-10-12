"""Quick checks after ChemBERTa scatter-by-ingredient_id fix (Plan A)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from baselines.common.data import BASE_PATH, load_tcm_full_data  # noqa: E402


def main() -> None:
    emb = np.load(os.path.join(BASE_PATH, "BERT/ChemBERTa_output/ingredient_chemberta1.npy"))
    ids_df = pd.read_csv(os.path.join(BASE_PATH, "BERT/ChemBERTa_output/ingredient_ids1.csv"))
    ing_df = pd.read_csv(os.path.join(BASE_PATH, "herb_ingredient/ingredient_node1.csv"))
    n_ing = int(ing_df["ingredient_id"].max()) + 1

    data, _ = load_tcm_full_data()
    feat = data["ingredient"].x.numpy()

    ok_shape = feat.shape == (n_ing, emb.shape[1])
    ok_id12_zero = np.allclose(feat[12], 0.0)
    ok_id13 = np.allclose(feat[13], emb[12])
    zero_rows = int((data["ingredient"].x.abs().sum(dim=1) == 0).sum().item())
    ok_zeros = zero_rows == 2917

    hi_df = pd.read_csv(os.path.join(BASE_PATH, "herb_ingredient/edge_herb_ingredient1.csv"))
    hi_kept = int(((hi_df["ingredient_id"] >= 0) & (hi_df["ingredient_id"] < n_ing)).sum())
    hi_total = len(hi_df)
    hi_high = int((hi_df["ingredient_id"] >= 12123).sum())

    print("=== ChemBERTa loading validation ===")
    print(f"BASE_PATH: {BASE_PATH}")
    print(f"n_ing (ingredient_node1 max+1): {n_ing}")
    print(f"emb shape: {emb.shape}, ids rows: {len(ids_df)}")
    print(f"feat shape: {feat.shape} -> {'PASS' if ok_shape else 'FAIL'}")
    print(f"feat[12] all zero (id=12 no SMILES) -> {'PASS' if ok_id12_zero else 'FAIL'}")
    print(f"feat[13] == emb[12] -> {'PASS' if ok_id13 else 'FAIL'}")
    print(f"zero-vector rows: {zero_rows} (expect 2917) -> {'PASS' if ok_zeros else 'FAIL'}")
    print(f"herb-ingredient edges kept: {hi_kept}/{hi_total} (id>=12123 in raw CSV: {hi_high})")
    print(f"data num_ingredient / x.size(0): {data['ingredient'].x.size(0)}")

    checks = [ok_shape, ok_id12_zero, ok_id13, ok_zeros]
    if not all(checks):
        raise SystemExit(1)
    print("All checks passed.")


if __name__ == "__main__":
    main()
