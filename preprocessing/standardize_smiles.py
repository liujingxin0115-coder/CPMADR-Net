"""Standardize and validate SMILES in an Excel table."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize

NORMALIZER = rdMolStandardize.Normalizer()
REIONIZER = rdMolStandardize.Reionizer()
FRAGMENT_CHOOSER = rdMolStandardize.LargestFragmentChooser()


def standardize_smiles(value: object, include_stereo: bool = True) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(value)
        if mol is None:
            return None
        mol = NORMALIZER.normalize(mol)
        mol = REIONIZER.reionize(mol)
        mol = FRAGMENT_CHOOSER.choose(mol)
        Chem.SanitizeMol(mol)
        return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=include_stereo)
    except (RuntimeError, ValueError):
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--column", default="SMILES")
    parser.add_argument("--no-stereo", action="store_true")
    args = parser.parse_args()

    frame = pd.read_excel(args.input)
    if args.column not in frame.columns:
        raise ValueError(f"Missing column {args.column!r}; available columns: {list(frame.columns)}")
    output_column = f"{args.column}_std"
    frame[output_column] = frame[args.column].apply(lambda value: standardize_smiles(value, not args.no_stereo))
    frame["smiles_valid"] = frame[output_column].notna()
    valid = frame[frame["smiles_valid"]].drop_duplicates(subset=[output_column])
    failed = frame[~frame["smiles_valid"]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(args.output, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="with_flags_all", index=False)
        valid.to_excel(writer, sheet_name="standardized_unique", index=False)
        failed.to_excel(writer, sheet_name="failed_smiles", index=False)
    print(f"Standardized {len(valid)} unique molecules; {len(failed)} rows failed")


if __name__ == "__main__":
    main()
