from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm

# ===== 1. 加载本地 ChemBERTa 模型 =====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
model_path = PROJECT_ROOT / "BERT" / "ChemBERTa"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
model.eval()

def encode_smiles_csv(
    csv_path: str,
    id_col: str,
    smiles_col: str,
    out_npy_path: str,
    out_id_path: str,
    max_length: int = 128,
    batch_size: int = 64,
):
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=[smiles_col])

    smiles_list = df[smiles_col].astype(str).tolist()
    ids = df[id_col].tolist()

    all_embs = []

    for i in tqdm(range(0, len(smiles_list), batch_size), desc=f"Encoding {csv_path}"):
        batch_smiles = smiles_list[i:i + batch_size]

        enc = tokenizer(
            batch_smiles,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            # 取 [CLS] 向量作为整个分子的表示
            cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, hidden_size)

        all_embs.append(cls_emb.cpu().numpy())

    all_embs = np.concatenate(all_embs, axis=0)  # (N_compound, hidden_size)
    Path(out_npy_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_id_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy_path, all_embs, allow_pickle=False)

    pd.DataFrame({id_col: ids}).to_csv(out_id_path, index=False, encoding="utf-8-sig")

    print(f"Saved embeddings to {out_npy_path}, shape = {all_embs.shape}")
    print(f"Saved ids to {out_id_path}")


if __name__ == "__main__":
    BASE_DIR = PROJECT_ROOT

    compound_csv = BASE_DIR / "herb_ingredient" / "ingredient_node1.csv"

    encode_smiles_csv(
        csv_path=str(compound_csv),
        id_col="ingredient_id",          # 成分 id 列名（看你的 csv 里实际叫什么）
        smiles_col="SMILES_std",         # SMILES 列名（看你的 csv 里实际叫什么）
        out_npy_path=str(BASE_DIR / "BERT" / "ChemBERTa_output" / "ingredient_chemberta1.npy"),
        out_id_path=str(BASE_DIR / "BERT" / "ChemBERTa_output" / "ingredient_ids1.csv"),
    )