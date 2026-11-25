from transformers import AutoTokenizer, AutoModel
import torch
import pandas as pd
from pathlib import Path
import numpy as np
from tqdm import tqdm

# ===== 1. 加载本地 BiomedBERT =====
PROJECT_ROOT = Path(__file__).resolve().parents[1]
model_path = PROJECT_ROOT / "BERT" / "BiomedBERT"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModel.from_pretrained(model_path, local_files_only=True).to(device)
model.eval()

def encode_csv_nodes(
    csv_path: str,
    id_col: str,
    text_cols: list,
    out_npy_path: str,
    out_id_path: str,
    max_length: int = 64,
    batch_size: int = 32,
):
    """
    csv_path: 节点 csv 路径
    id_col:   节点 id 列名（如 herb_id / target_id / adr_id）
    text_cols: 用来拼接成一句英文的列名列表
    out_npy_path: 保存向量的 .npy 文件
    out_id_path:  保存 id 顺序的 .csv（方便之后对齐）
    """
    df = pd.read_csv(csv_path)

    # 构造文本（把多个列拼在一起，中间用 ". " 连接）
    def build_text(row):
        parts = []
        for c in text_cols:
            if c in row and pd.notna(row[c]):
                parts.append(str(row[c]))
        return ". ".join(parts).strip()

    texts = df.apply(build_text, axis=1).tolist()
    ids = df[id_col].tolist()

    all_embs = []
    for i in tqdm(range(0, len(texts), batch_size), desc=f"Encoding {csv_path}"):
        batch_texts = texts[i:i + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        enc = {k: v.to(device) for k, v in enc.items()}

        with torch.no_grad():
            outputs = model(**enc)
            cls_emb = outputs.last_hidden_state[:, 0, :]  # (batch, 768)

        all_embs.append(cls_emb.cpu().numpy())

    all_embs = np.concatenate(all_embs, axis=0)  # (N, 768)
    Path(out_npy_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_id_path).parent.mkdir(parents=True, exist_ok=True)
    np.save(out_npy_path, all_embs, allow_pickle=False)

    # 记录 id 顺序，便于之后加载对齐
    pd.DataFrame({id_col: ids}).to_csv(out_id_path, index=False, encoding="utf-8-sig")

    print(f"Saved embeddings to {out_npy_path}, shape = {all_embs.shape}")
    print(f"Saved ids to {out_id_path}")

if __name__ == "__main__":
    base_dir = PROJECT_ROOT

    # ========== 1）中药 herb 节点 ==========
    herb_csv = base_dir / "HerbData" / "medicine_herb" / "herb_english_match" / "herb_node_with_TCM_info1.csv"
    encode_csv_nodes(
        csv_path=str(herb_csv),
        id_col="herb_id",
        text_cols=["TCM_name_en", "Herb_latin_name"],  # 有需要可以加更多列
        out_npy_path=str(base_dir / r"BERT\BiomedBERT_output\herb_emb_biomedbert1.npy"),
        out_id_path=str(base_dir / r"BERT\BiomedBERT_output\herb_ids1.csv"),
    )

    # ========== 2）靶点 target 节点 ==========
    target_csv = base_dir / "ingredient_target" / "target_node1.csv"
    encode_csv_nodes(
        csv_path=str(target_csv),
        id_col="target_id",               # ← 把这里改成你实际的 id 列名
        text_cols=["target_name_en"],     # ← 把这里改成 target 的英文列，如 gene_symbol / uniprot_name 等
        out_npy_path=str(base_dir / r"BERT\BiomedBERT_output\target_emb_biomedbert1.npy"),
        out_id_path=str(base_dir / r"BERT\BiomedBERT_output\target_ids1.csv"),
    )

    # ========== 3）ADR nodes ==========
    adr_csv = base_dir / "adr" / "adr_node1.csv"
    encode_csv_nodes(
        csv_path=str(adr_csv),
        id_col="adr_id",
        text_cols=["adr_en"],
        out_npy_path=str(base_dir / "BERT" / "BiomedBERT_output" / "adr_emb_biomedbert1.npy"),
        out_id_path=str(base_dir / "BERT" / "BiomedBERT_output" / "adr_ids1.csv"),
    )