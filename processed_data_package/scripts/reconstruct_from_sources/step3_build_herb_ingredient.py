"""
Step 3: Build chemical ingredient nodes (15,040) and Herb-Ingredient edges (36,102)
aligned to the master herb dictionary from Step 1.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build ingredient_node1.csv and edge_herb_ingredient1.csv.")
    p.add_argument("--herb_nodes_csv", type=str, default="HerbData/处理中成药成分分离/herb_node1.csv")
    p.add_argument(
        "--herb_ing_xlsx",
        type=str,
        default="HerbData/我的数据/标准化smiles/药材-成分（总）_标准化.xlsx",
    )
    p.add_argument("--out_dir", type=str, default="药材-成分（1）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    master_herb_df = pd.read_csv(args.herb_nodes_csv)
    herb_name_to_id = dict(
        zip(master_herb_df["herb_name"].astype(str).str.strip(), master_herb_df["herb_id"].astype(int))
    )

    df_new = pd.read_excel(args.herb_ing_xlsx)
    ingred_df = (
        df_new[["Ingredient_Name", "SMILES_std"]]
        .drop_duplicates("Ingredient_Name")
        .reset_index(drop=True)
    )
    ingred_df["ingredient_id"] = range(len(ingred_df))
    ingred_df.to_csv(out_dir / "ingredient_node1.csv", index=False, encoding="utf-8-sig")

    ingred_name_to_id = dict(zip(ingred_df["Ingredient_Name"], ingred_df["ingredient_id"]))
    edges = []
    for _, row in df_new.iterrows():
        h_name = str(row["Herb_name"]).strip()
        i_name = row["Ingredient_Name"]
        if h_name in herb_name_to_id and i_name in ingred_name_to_id:
            edges.append({
                "herb_id": herb_name_to_id[h_name],
                "ingredient_id": ingred_name_to_id[i_name],
            })

    edge_df = pd.DataFrame(edges)
    edge_df.to_csv(out_dir / "edge_herb_ingredient1.csv", index=False, encoding="utf-8-sig")
    print(f"[Step 3] Saved {len(ingred_df)} ingredients and {len(edge_df)} Herb-Ingredient edges.")


if __name__ == "__main__":
    main()
