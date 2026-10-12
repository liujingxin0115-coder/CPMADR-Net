"""Build ADR nodes, medicine-ADR edges, and target-ADR edges."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def _split_ids(value: object) -> list[str]:
    return str(value).replace("\u3001", ",").replace("\uff1b", ",").replace(";", ",").split(",")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adr-target-input", type=Path, required=True)
    parser.add_argument("--medicine-adr-input", type=Path, required=True)
    parser.add_argument("--medicine-nodes", type=Path, required=True)
    parser.add_argument("--target-nodes", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    medicines = pd.read_csv(args.medicine_nodes)
    targets = pd.read_csv(args.target_nodes)
    adr_info = pd.read_excel(args.adr_target_input)
    med_adr = pd.read_excel(args.medicine_adr_input)
    medicine_ids = dict(zip(medicines["medicine_name"].astype(str).str.strip(), medicines["medicine_id"]))
    target_ids = dict(zip(targets["target_name"].astype(str).str.strip(), targets["target_id"]))

    reactions = adr_info[["\u7f16\u53f7", "\u540d\u79f0", "\u82f1\u6587\u540d\u79f0", "SOC\u4e2d\u6587"]].copy()
    reactions.columns = ["adr_id", "adr_name", "adr_en", "soc_cn"]
    reactions.drop_duplicates("adr_id").to_csv(args.output_dir / "adr_node1.csv", index=False, encoding="utf-8-sig")

    mr_rows = []
    for _, row in med_adr.iterrows():
        medicine_id = medicine_ids.get(str(row["\u4e2d\u836f\u540d\uff08\u836f\u54c1\u540d\u79f0\uff09"]).strip())
        if medicine_id is None:
            continue
        for raw_id in _split_ids(row["\u4e0d\u826f\u53cd\u5e94"]):
            try:
                mr_rows.append((int(medicine_id), int(float(raw_id.strip()))))
            except (TypeError, ValueError):
                continue
    pd.DataFrame(mr_rows, columns=["medicine_id", "adr_id"]).drop_duplicates().to_csv(
        args.output_dir / "edge_medicine_adr1.csv", index=False
    )

    tr_rows, missing = [], set()
    for _, row in adr_info.iterrows():
        for name in _split_ids(row["\u5339\u914d\u5230\u7684targetId"]):
            name = name.strip()
            if not name or name == "nan":
                continue
            if name in target_ids:
                tr_rows.append((int(target_ids[name]), int(row["\u7f16\u53f7"])))
            else:
                missing.add(name)
    pd.DataFrame(tr_rows, columns=["target_id", "adr_id"]).drop_duplicates().to_csv(
        args.output_dir / "edge_target_adr1.csv", index=False
    )
    if missing:
        pd.DataFrame({"missing_target": sorted(missing)}).to_csv(args.output_dir / "missing_targets_in_adr1.csv", index=False)
    print(f"Wrote ADR graph tables to {args.output_dir}; unmatched targets: {len(missing)}")


if __name__ == "__main__":
    main()
