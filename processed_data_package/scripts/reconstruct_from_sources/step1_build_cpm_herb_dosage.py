"""
Step 1: Parse CPM prescriptions and standardize herb dosages (in grams) from
extracted China Medical Information Platform / Chinese Pharmacopoeia records.

Input CSV format:
    药品编号, 中成药, 不良反应, 成分1..成分24, 成分1_剂量..成分24_剂量
Outputs:
    medicine_node1.csv, herb_node1.csv, edge_medicine_herb1.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build CPM nodes, Herb nodes, and dosage-weighted CPM-Herb edges.")
    p.add_argument("--input_csv", type=str, default="HerbData/处理中成药成分分离/成分量分离.csv")
    p.add_argument("--out_dir", type=str, default="HerbData/处理中成药成分分离")
    p.add_argument("--max_herb_cols", type=int, default=24)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    in_path = Path(args.input_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(in_path)

    # 1. CPM nodes (0 .. N_cpm - 1)
    med_names = df["中成药"].astype(str).str.strip().unique()
    medicine_df = pd.DataFrame({
        "medicine_name": med_names,
        "medicine_id": range(len(med_names)),
    })
    medicine_df.to_csv(out_dir / "medicine_node1.csv", index=False, encoding="utf-8-sig")

    # 2. Herb nodes (0 .. N_herb - 1)
    herb_cols = [f"成分{i}" for i in range(1, args.max_herb_cols + 1) if f"成分{i}" in df.columns]
    all_herbs = pd.Series(df[herb_cols].values.flatten()).dropna().astype(str).str.strip()
    all_herbs = all_herbs[all_herbs != ""].unique()
    herb_df = pd.DataFrame({
        "herb_name": all_herbs,
        "herb_id": range(len(all_herbs)),
    })
    herb_df.to_csv(out_dir / "herb_node1.csv", index=False, encoding="utf-8-sig")

    med_map = dict(zip(medicine_df["medicine_name"], medicine_df["medicine_id"]))
    herb_map = dict(zip(herb_df["herb_name"], herb_df["herb_id"]))

    # 3. CPM-Herb prescription edges with mass dosage in grams (g)
    edges = []
    for _, row in df.iterrows():
        med_id = med_map[str(row["中成药"]).strip()]
        for i in range(1, args.max_herb_cols + 1):
            h_col, d_col = f"成分{i}", f"成分{i}_剂量"
            if h_col not in df.columns:
                continue
            herb_val = row[h_col]
            if pd.notna(herb_val) and str(herb_val).strip():
                herb_id = herb_map[str(herb_val).strip()]
                dosage_val = 0.0
                if d_col in df.columns and pd.notna(row[d_col]):
                    m = re.search(r"[-+]?\d*\.\d+|\d+", str(row[d_col]))
                    if m:
                        dosage_val = float(m.group())
                edges.append({
                    "medicine_id": med_id,
                    "herb_id": herb_id,
                    "dosage": dosage_val,
                })

    edge_df = pd.DataFrame(edges)
    edge_df.to_csv(out_dir / "edge_medicine_herb1.csv", index=False, encoding="utf-8-sig")
    print(f"[Step 1] Saved {len(medicine_df)} CPMs, {len(herb_df)} herbs, {len(edge_df)} CPM-Herb edges.")


if __name__ == "__main__":
    main()
