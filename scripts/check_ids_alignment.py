"""Verify non-consecutive ingredient_ids1.csv vs scatter loading logic."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
BASE = ROOT

ids_df = pd.read_csv(os.path.join(BASE, "BERT/ChemBERTa_output/ingredient_ids1.csv"))
ing_df = pd.read_csv(os.path.join(BASE, "herb_ingredient/ingredient_node1.csv"))
emb = np.load(os.path.join(BASE, "BERT/ChemBERTa_output/ingredient_chemberta1.npy"))

ids = ids_df["ingredient_id"].astype(int).tolist()
n_ing = int(ing_df["ingredient_id"].max()) + 1
ids1_set = set(ids)

print("=== scale ===")
print(f"ingredient_node1 rows: {len(ing_df)}, max id: {ing_df['ingredient_id'].max()}, n_ing: {n_ing}")
print(f"ingredient_ids1 rows: {len(ids_df)}, emb rows: {len(emb)}")
print(f"ids1 rows == emb rows: {len(ids_df) == len(emb)}")

print("\n=== non-consecutive ids in ids1 ===")
sorted_ids = sorted(ids)
gap_missing = []
for i in range(1, len(sorted_ids)):
    if sorted_ids[i] - sorted_ids[i - 1] > 1:
        for missing in range(sorted_ids[i - 1] + 1, sorted_ids[i]):
            gap_missing.append(missing)
print(f"gaps inside ids1 (ids skipped because no SMILES etc.): {len(gap_missing)}")
print(f"first gaps: {gap_missing[:15]}")

no_smiles = ing_df[ing_df["SMILES_std"].isna() | (ing_df["SMILES_std"].astype(str).str.strip() == "")]
no_smiles_ids = set(no_smiles["ingredient_id"].astype(int))

print("\n=== vs ingredient_node1 ===")
print(f"no SMILES in node table: {len(no_smiles_ids)}")
print(f"ids in ids1: {len(ids1_set)}")
print(f"no-SMILES ids not in ids1: {len(no_smiles_ids - ids1_set)}")
print(f"ids1 ids missing from node table: {len(ids1_set - set(ing_df['ingredient_id']))}")

with_smiles = ing_df[~ing_df["ingredient_id"].isin(no_smiles_ids)]
missing_emb = set(with_smiles["ingredient_id"].astype(int)) - ids1_set
print(f"has SMILES but missing from ids1: {len(missing_emb)}")
if missing_emb:
    print(f"  examples: {sorted(missing_emb)[:10]}")

print(f"duplicate ids in ids1: {ids_df['ingredient_id'].duplicated().sum()}")
out_of_range = [i for i in ids if i < 0 or i >= n_ing]
print(f"ids out of [0, {n_ing}): {len(out_of_range)}")

# scatter simulation (same as data.py)
feat = np.zeros((n_ing, emb.shape[1]), dtype=np.float32)
for row, ing_id in enumerate(ids_df["ingredient_id"].astype(int)):
    ing_id = int(ing_id)
    if 0 <= ing_id < n_ing:
        feat[ing_id] = emb[row]

print("\n=== scatter checks ===")
print(f"id 12 in ids1: {12 in ids1_set} (expect False, no SMILES)")
print(f"feat[12] all zero: {np.allclose(feat[12], 0)}")
print(f"feat[13] == emb[12]: {np.allclose(feat[13], emb[12])}")

mismatch = sum(
    1
    for row, ing_id in enumerate(ids_df["ingredient_id"].astype(int))
    if not np.allclose(feat[int(ing_id)], emb[row])
)
print(f"row->id scatter roundtrip OK: {mismatch == 0} (mismatches={mismatch})")

zero_rows = int((np.abs(feat).sum(axis=1) == 0).sum())
expected_zeros = n_ing - len(ids1_set)
print(f"zero rows: {zero_rows}, expected: {expected_zeros} (= n_ing - |ids1|)")

non_ids1 = set(range(n_ing)) - ids1_set
bad = sum(1 for i in non_ids1 if np.abs(feat[i]).sum() > 0)
print(f"non-ids1 slots non-zero: {bad} (expect 0)")

# compare with load_tcm_full_data
sys.path.insert(0, ROOT)
from baselines.common.data import load_tcm_full_data  # noqa: E402

data, _ = load_tcm_full_data()
loaded = data["ingredient"].x.numpy()
print(f"\n=== matches load_tcm_full_data ===")
print(f"max abs diff vs loader: {np.max(np.abs(loaded - feat)):.6e}")
