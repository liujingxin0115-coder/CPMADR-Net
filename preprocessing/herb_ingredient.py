"""Build ingredient nodes and herb-ingredient edges from standardized SMILES."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
from tqdm.auto import tqdm


def smiles_to_fp(smiles: object) -> list[int]:
    try:
        mol = Chem.MolFromSmiles(str(smiles))
        if mol is not None:
            return list(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048))
    except (TypeError, ValueError):
        pass
    return [0] * 2048


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--herb-nodes", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True, help="Standardized herb-ingredient Excel file")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    herbs = pd.read_csv(args.herb_nodes)
    source = pd.read_excel(args.input)
    herb_name_to_id = dict(zip(herbs["herb_name"], herbs["herb_id"]))
    new_names = [name for name in source["Herb_name"].dropna().unique() if name not in herb_name_to_id]
    start_id = int(herbs["herb_id"].max()) + 1 if len(herbs) else 0
    for offset, name in enumerate(new_names):
        herb_name_to_id[name] = start_id + offset
    if new_names:
        additions = pd.DataFrame({"herb_name": new_names, "herb_id": [herb_name_to_id[n] for n in new_names]})
        herbs = pd.concat([herbs, additions], ignore_index=True)
    herbs.to_csv(args.output_dir / "herb_node1.csv", index=False, encoding="utf-8-sig")

    ingredients = source[["Ingredient_Name", "SMILES_std"]].drop_duplicates("Ingredient_Name").reset_index(drop=True)
    ingredients["ingredient_id"] = range(len(ingredients))
    tqdm.pandas(desc="Morgan fingerprints")
    fps = ingredients["SMILES_std"].progress_apply(smiles_to_fp)
    ingredients.to_csv(args.output_dir / "ingredient_node1.csv", index=False, encoding="utf-8-sig")
    np.save(args.output_dir / "ingredient_features.npy", np.asarray(fps.tolist(), dtype=np.float32))

    ingredient_ids = dict(zip(ingredients["Ingredient_Name"], ingredients["ingredient_id"]))
    edges = pd.DataFrame({
        "herb_id": source["Herb_name"].map(herb_name_to_id),
        "ingredient_id": source["Ingredient_Name"].map(ingredient_ids),
    }).dropna().astype(int).drop_duplicates()
    edges.to_csv(args.output_dir / "edge_herb_ingredient1.csv", index=False)
    print(f"Wrote {len(ingredients)} ingredients and {len(edges)} edges to {args.output_dir}")


if __name__ == "__main__":
    main()
