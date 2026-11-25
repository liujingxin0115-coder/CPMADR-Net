"""
Step 5: Build MedDRA PT-standardized ADR nodes (171), CPM-ADR positive edges (2,289),
and Target-ADR edges (4,356), remapping 8-digit MedDRA PT codes to 0-indexed reaction_id (0..170).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build reaction_node1.csv, edge_medicine_reaction1.csv, and edge_target_reaction1.csv.")
    p.add_argument("--med_nodes_csv", type=str, default="HerbData/处理中成药成分分离/medicine_node1.csv")
    p.add_argument("--target_nodes_csv", type=str, default="成分-靶点（2）/target_node1.csv")
    p.add_argument("--adr_target_xlsx", type=str, default="不良反应（3）/数据/标注结果_全部靶点(映射结果).xlsx")
    p.add_argument("--cpm_adr_xlsx", type=str, default="不良反应（3）/数据/adr处理后的数据.xlsx")
    p.add_argument("--out_dir", type=str, default="不良反应（3）")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_med = pd.read_csv(args.med_nodes_csv)
    df_target = pd.read_csv(args.target_nodes_csv)
    med_name_to_id = dict(zip(df_med["medicine_name"].astype(str).str.strip(), df_med["medicine_id"].astype(int)))
    target_name_to_id = dict(zip(df_target["target_name"].astype(str).str.strip(), df_target["target_id"].astype(int)))

    # 1. ADR nodes standardized to MedDRA Preferred Terms (PTs)
    df_reac_info = pd.read_excel(args.adr_target_xlsx)
    reac_df = (
        df_reac_info[["编号", "名称", "英文名称", "SOC中文"]]
        .drop_duplicates("编号")
        .reset_index(drop=True)
        .copy()
    )
    reac_df.columns = ["original_id", "reaction_name", "reaction_en", "soc_cn"]
    reac_df["original_id"] = reac_df["original_id"].astype(int)
    reac_id_map = {old_code: new_id for new_id, old_code in enumerate(reac_df["original_id"].tolist())}
    reac_df["reaction_id"] = reac_df["original_id"].map(reac_id_map)
    reac_df = reac_df[["reaction_id", "reaction_name", "reaction_en", "soc_cn", "original_id"]]
    reac_df.to_csv(out_dir / "reaction_node1.csv", index=False, encoding="utf-8-sig")

    # 2. CPM-ADR positive edges (deduplicated per CPM at MedDRA PT level -> 2,289 edges)
    df_med_reac_raw = pd.read_excel(args.cpm_adr_xlsx)
    mr_edges = []
    seen_mr = set()
    for _, row in df_med_reac_raw.iterrows():
        med_name = str(row["中药名（药品名称）"]).strip()
        if med_name not in med_name_to_id:
            continue
        m_id = med_name_to_id[med_name]
        codes_raw = str(row["不良反应"]).replace("、", ",").replace("；", ",").replace(";", ",")
        for tok in codes_raw.split(","):
            tok = tok.strip()
            if not tok or tok == "nan":
                continue
            try:
                pt_code = int(float(tok))
            except ValueError:
                continue
            if pt_code in reac_id_map:
                r_id = reac_id_map[pt_code]
                if (m_id, r_id) not in seen_mr:
                    seen_mr.add((m_id, r_id))
                    mr_edges.append({"medicine_id": m_id, "reaction_id": r_id})

    edge_mr_df = pd.DataFrame(mr_edges)
    edge_mr_df.to_csv(out_dir / "edge_medicine_reaction1.csv", index=False, encoding="utf-8-sig")

    # 3. Target-ADR edges from Open Targets mapped via HGNC gene symbol and MedDRA PT code (4,356 edges)
    tr_edges = []
    missing_targets = set()
    for _, row in df_reac_info.iterrows():
        pt_code = int(row["编号"])
        if pt_code not in reac_id_map:
            continue
        r_id = reac_id_map[pt_code]
        targets_raw = str(row["匹配到的targetId"])
        if pd.notna(row["匹配到的targetId"]) and targets_raw != "nan" and targets_raw.strip():
            for t_tok in targets_raw.replace(";", ",").replace("；", ",").replace("、", ",").split(","):
                t_name = t_tok.strip()
                if not t_name:
                    continue
                if t_name in target_name_to_id:
                    tr_edges.append({"target_id": target_name_to_id[t_name], "reaction_id": r_id})
                else:
                    missing_targets.add(t_name)

    edge_tr_df = pd.DataFrame(tr_edges)
    edge_tr_df.to_csv(out_dir / "edge_target_reaction1.csv", index=False, encoding="utf-8-sig")
    if missing_targets:
        pd.DataFrame({"missing_target": sorted(missing_targets)}).to_csv(
            out_dir / "missing_targets_in_adr1.csv", index=False, encoding="utf-8-sig"
        )

    print(
        f"[Step 5] Saved {len(reac_df)} MedDRA PT nodes, {len(edge_mr_df)} CPM-ADR edges, "
        f"{len(edge_tr_df)} Target-ADR edges."
    )


if __name__ == "__main__":
    main()
