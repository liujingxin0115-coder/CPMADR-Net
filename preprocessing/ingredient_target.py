"""Build target nodes and ingredient-target edges."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Ingredient-target Excel file")
    parser.add_argument("--ingredient-nodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    mapping = pd.read_excel(args.input)
    ingredients = pd.read_csv(args.ingredient_nodes)
    genes = mapping["Gene"].dropna().drop_duplicates().reset_index(drop=True)
    targets = pd.DataFrame({"target_id": range(len(genes)), "target_name": genes})
    targets.to_csv(args.output_dir / "target_node1.csv", index=False, encoding="utf-8-sig")

    ingredient_ids = dict(zip(ingredients["Ingredient_Name"].astype(str).str.strip(), ingredients["ingredient_id"]))
    target_ids = dict(zip(targets["target_name"], targets["target_id"]))
    edges = pd.DataFrame({
        "ingredient_id": mapping["Ingredient_Name"].astype(str).str.strip().map(ingredient_ids),
        "target_id": mapping["Gene"].map(target_ids),
    }).dropna().astype(int).drop_duplicates()
    edges.to_csv(args.output_dir / "edge_ingredient_target1.csv", index=False)
    print(f"Wrote {len(targets)} targets and {len(edges)} edges to {args.output_dir}")


if __name__ == "__main__":
    main()
