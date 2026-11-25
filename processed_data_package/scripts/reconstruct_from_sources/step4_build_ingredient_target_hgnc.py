"""
Step 4: Build HGNC-standardized Target nodes (9,492) and Ingredient-Target edges (104,633)
from HERB 2.0 / ITCM / TCMSP curated associations.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build target_node1.csv and edge_ingredient_target1.csv.")
    p.add_argument("--ing_nodes_csv", type=str, default="药材-成分（1）/ingredient_node1.csv")
    p.add_argument("--ing_target_xlsx", type=str, default="成分-靶点（2）/数据/成分-靶点（总）.xlsx")
    p.add_argument("--out_dir", type=str, default="成分-靶点（2）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_gene = pd.read_excel(args.ing_target_xlsx)
    df_ing = pd.read_csv(args.ing_nodes_csv)

    unique_genes = df_gene["Gene"].dropna().astype(str).str.strip().unique()
    target_node_df = pd.DataFrame({
        "target_id": range(len(unique_genes)),
        "target_name": unique_genes,
    })
    target_node_df.to_csv(out_dir / "target_node1.csv", index=False, encoding="utf-8-sig")

    ingred_map = dict(zip(df_ing["Ingredient_Name"].astype(str).str.strip(), df_ing["ingredient_id"].astype(int)))
    target_map = dict(zip(target_node_df["target_name"], target_node_df["target_id"]))

    edges = []
    for _, row in df_gene.iterrows():
        iname = str(row["Ingredient_Name"]).strip()
        gname = str(row["Gene"]).strip() if pd.notna(row["Gene"]) else ""
        if iname in ingred_map and gname in target_map:
            edges.append({
                "ingredient_id": ingred_map[iname],
                "target_id": target_map[gname],
            })

    edge_it_df = pd.DataFrame(edges)
    edge_it_df.to_csv(out_dir / "edge_ingredient_target1.csv", index=False, encoding="utf-8-sig")
    print(f"[Step 4] Saved {len(target_node_df)} HGNC targets and {len(edge_it_df)} Ingredient-Target edges.")


if __name__ == "__main__":
    main()
