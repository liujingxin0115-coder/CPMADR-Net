"""
Step 6: Generate initial 768-d node embeddings using public Hugging Face checkpoints:
  - Chemical ingredients (SMILES_std): seyonec/ChemBERTa-zinc-base-v1
  - Herbs, Targets, and ADRs (English names): microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext
"""

from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer


def encode_texts(
    texts: list[str],
    model_name_or_path: str,
    device: torch.device,
    max_length: int,
    batch_size: int,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModel.from_pretrained(model_name_or_path).to(device)
    model.eval()

    chunks = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Encoding with {model_name_or_path}"):
        batch = texts[i : i + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            cls_emb = model(**enc).last_hidden_state[:, 0, :]
        chunks.append(cls_emb.cpu().numpy().astype(np.float32))
    return np.concatenate(chunks, axis=0)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Encode ChemBERTa and BiomedBERT node features.")
    p.add_argument("--chemberta_model", type=str, default="seyonec/ChemBERTa-zinc-base-v1")
    p.add_argument(
        "--biomedbert_model",
        type=str,
        default="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
    )
    p.add_argument("--ing_csv", type=str, default="data/nodes/ingredient_nodes_15040.csv")
    p.add_argument("--herb_csv", type=str, default="data/nodes/herb_nodes_383.csv")
    p.add_argument("--target_csv", type=str, default="data/nodes/target_nodes_9492.csv")
    p.add_argument("--adr_csv", type=str, default="data/nodes/adr_nodes_171.csv")
    p.add_argument("--out_root", type=str, default="BERT")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_root = Path(args.out_root)
    chem_dir = out_root / "ChemBERTa_output"
    bio_dir = out_root / "BiomedBERT_output"
    chem_dir.mkdir(parents=True, exist_ok=True)
    bio_dir.mkdir(parents=True, exist_ok=True)

    # 1. Ingredient SMILES -> ChemBERTa (valid SMILES encoded; missing SMILES zero-filled at load time)
    ing_df = pd.read_csv(args.ing_csv)
    valid_ing = ing_df[ing_df["smiles_rdkit_std"].notna() & (ing_df["smiles_rdkit_std"].astype(str).str.strip() != "")].copy()
    chem_emb = encode_texts(
        valid_ing["smiles_rdkit_std"].astype(str).tolist(),
        args.chemberta_model,
        device=device,
        max_length=128,
        batch_size=64,
    )
    np.save(chem_dir / "ingredient_chemberta1.npy", chem_emb, allow_pickle=False)
    pd.DataFrame({"ingredient_id": valid_ing["ingredient_id"].astype(int)}).to_csv(
        chem_dir / "ingredient_ids1.csv", index=False, encoding="utf-8-sig"
    )

    # 2. Herb English + Latin names -> BiomedBERT
    herb_df = pd.read_csv(args.herb_csv)
    herb_texts = [
        ". ".join(
            [str(row[c]).strip() for c in ("herb_name_en", "herb_latin_name") if pd.notna(row[c]) and str(row[c]).strip()]
        )
        for _, row in herb_df.iterrows()
    ]
    herb_emb = encode_texts(herb_texts, args.biomedbert_model, device=device, max_length=64, batch_size=32)
    np.save(bio_dir / "herb_emb_biomedbert1.npy", herb_emb, allow_pickle=False)
    pd.DataFrame({"herb_id": herb_df["herb_id"].astype(int)}).to_csv(
        bio_dir / "herb_ids1.csv", index=False, encoding="utf-8-sig"
    )

    # 3. Target HGNC symbols -> BiomedBERT
    targ_df = pd.read_csv(args.target_csv)
    targ_emb = encode_texts(
        targ_df["hgnc_symbol"].astype(str).tolist(),
        args.biomedbert_model,
        device=device,
        max_length=64,
        batch_size=64,
    )
    np.save(bio_dir / "target_emb_biomedbert1.npy", targ_emb, allow_pickle=False)
    pd.DataFrame({"target_id": targ_df["target_id"].astype(int)}).to_csv(
        bio_dir / "target_ids1.csv", index=False, encoding="utf-8-sig"
    )

    # 4. ADR English MedDRA PTs -> BiomedBERT
    adr_df = pd.read_csv(args.adr_csv)
    adr_emb = encode_texts(
        adr_df["meddra_pt_en"].astype(str).tolist(),
        args.biomedbert_model,
        device=device,
        max_length=64,
        batch_size=32,
    )
    np.save(bio_dir / "reaction_emb_biomedbert1.npy", adr_emb, allow_pickle=False)
    pd.DataFrame({"reaction_id": adr_df["reaction_id"].astype(int)}).to_csv(
        bio_dir / "reaction_ids1.csv", index=False, encoding="utf-8-sig"
    )

    print("[Step 6] All ChemBERTa and BiomedBERT embeddings generated.")


if __name__ == "__main__":
    main()
