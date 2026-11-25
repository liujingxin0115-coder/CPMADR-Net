"""
Step 2: Standardize raw chemical SMILES strings using RDKit MolStandardize
(Normalizer -> Reionizer -> LargestFragmentChooser -> SanitizeMol -> Canonical Isomeric SMILES).
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


normalizer = rdMolStandardize.Normalizer()
reionizer = rdMolStandardize.Reionizer()
largest_fragment_chooser = rdMolStandardize.LargestFragmentChooser()


def standardize_smiles(smiles: str, include_stereo: bool = True) -> str | None:
    if not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        mol = normalizer.normalize(mol)
        mol = reionizer.reionize(mol)
        mol = largest_fragment_chooser.choose(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=include_stereo)
    except Exception:
        logging.debug("Failed to standardize SMILES: %r", smiles)
        return None


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Standardize ingredient SMILES via RDKit.")
    p.add_argument("--input_xlsx", type=str, default="HerbData/我的数据/药材-成分（总）.xlsx")
    p.add_argument("--output_xlsx", type=str, default="HerbData/我的数据/标准化smiles/药材-成分（总）_标准化.xlsx")
    p.add_argument("--smiles_col", type=str, default="SMILES")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output_xlsx)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(args.input_xlsx)
    std_col = f"{args.smiles_col}_std"
    df[std_col] = df[args.smiles_col].apply(standardize_smiles)
    df["smiles_valid"] = df[std_col].notna()

    valid_df = df[df["smiles_valid"]].drop_duplicates(subset=[std_col]).copy()
    failed_df = df[~df["smiles_valid"]].copy()

    with pd.ExcelWriter(out_path, engine="openpyxl") as writer:
        df.to_excel(writer, sheet_name="with_flags_all", index=False)
        valid_df.to_excel(writer, sheet_name="standardized_unique", index=False)
        if not failed_df.empty:
            failed_df.to_excel(writer, sheet_name="failed_smiles", index=False)

    print(
        f"[Step 2] Total rows={len(df)} | Valid standardized={int(df['smiles_valid'].sum())} "
        f"| Unique structures={len(valid_df)}"
    )


if __name__ == "__main__":
    main()
