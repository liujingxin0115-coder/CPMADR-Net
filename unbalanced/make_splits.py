import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np
import pandas as pd

DEFAULT_BASE_DIR = str(Path(__file__).resolve().parents[1])
DEFAULT_MR_REL = "adr/edge_medicine_adr1.csv"
DEFAULT_MED_REL = "HerbData/medicine_herb/medicine_node1.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build tail-aware 5-fold splits with balanced test-edge difficulty."
    )
    parser.add_argument("--base-dir", type=str, default=DEFAULT_BASE_DIR, help="Project root directory.")
    parser.add_argument(
        "--mr-path",
        type=str,
        default=DEFAULT_MR_REL,
        help="Path (relative to base-dir) of medicine-reaction edge csv.",
    )
    parser.add_argument(
        "--medicine-path",
        type=str,
        default=DEFAULT_MED_REL,
        help="Path (relative to base-dir) of medicine node csv.",
    )
    parser.add_argument("--n-splits", type=int, default=5, help="Number of folds.")
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Validation ratio inside each train+val pool.",
    )
    parser.add_argument(
        "--tail-max-freq",
        type=int,
        default=3,
        help="Tail definition: reaction frequency <= tail-max-freq.",
    )
    parser.add_argument(
        "--tail-rotate-max-freq",
        type=int,
        default=3,
        help="Weak tail rotation is applied to tail reactions with frequency <= this value (usually 3).",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Random seed.")
    parser.add_argument(
        "--n-restarts",
        type=int,
        default=24,
        help="How many random restarts for test-fold assignment. Best objective is kept.",
    )
    parser.add_argument("--w-label", type=float, default=1.0, help="Weight for non-tail label balance.")
    parser.add_argument("--w-edge", type=float, default=0.35, help="Weight for edge-count balance.")
    parser.add_argument("--w-size", type=float, default=0.20, help="Weight for medicine-count balance.")
    parser.add_argument(
        "--w-tail-rotate",
        type=float,
        default=1.25,
        help="Weight for weak tail rotation objective (freq=2/3 tails spread over folds).",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=r"unbalanced/splits",
        help="Output directory (relative to base-dir).",
    )
    return parser.parse_args()


def build_multi_labels(mr_df: pd.DataFrame, medicine_ids: np.ndarray) -> Dict[int, Set[int]]:
    labels_by_med: Dict[int, Set[int]] = {int(mid): set() for mid in medicine_ids.tolist()}
    for row in mr_df.itertuples(index=False):
        labels_by_med[int(row.medicine_id)].add(int(row.adr_id))
    return labels_by_med


def build_edges_by_med(mr_df: pd.DataFrame, medicine_ids: np.ndarray) -> Dict[int, int]:
    by_med = mr_df.groupby("medicine_id").size().to_dict()
    return {int(mid): int(by_med.get(int(mid), 0)) for mid in medicine_ids.tolist()}


def _cv(values: np.ndarray) -> float:
    if values.size == 0:
        return 0.0
    mean = float(values.mean())
    if mean <= 0:
        return 0.0
    return float(values.std(ddof=0) / mean)


def _build_test_fold_caps(n_items: int, n_splits: int) -> List[int]:
    base = n_items // n_splits
    rem = n_items % n_splits
    return [base + (1 if i < rem else 0) for i in range(n_splits)]


def assign_test_folds_greedy(
    medicine_ids: List[int],
    labels_by_med: Dict[int, Set[int]],
    edges_by_med: Dict[int, int],
    reaction_freq_map: Dict[int, int],
    strat_labels: Set[int],
    tail_labels: Set[int],
    n_splits: int,
    seed: int,
    tail_rotate_max_freq: int,
    w_label: float,
    w_edge: float,
    w_size: float,
    w_tail_rotate: float,
) -> Tuple[Dict[int, int], Dict]:
    rng = np.random.default_rng(seed)
    med_ids = list(int(x) for x in medicine_ids)
    rng.shuffle(med_ids)

    rotate_tail_labels = {
        int(r)
        for r in tail_labels
        if 2 <= int(reaction_freq_map.get(int(r), 0)) <= int(tail_rotate_max_freq)
    }
    tail_target_folds = {
        int(r): int(min(n_splits, reaction_freq_map[int(r)]))
        for r in rotate_tail_labels
    }

    med_non_tail = {m: list(labels_by_med[m] & strat_labels) for m in med_ids}
    med_rotate_tail = {m: list(labels_by_med[m] & rotate_tail_labels) for m in med_ids}
    med_ids.sort(
        key=lambda m: (
            len(med_rotate_tail[m]),
            len(med_non_tail[m]),
            int(edges_by_med.get(m, 0)),
        ),
        reverse=True,
    )

    fold_caps = _build_test_fold_caps(len(med_ids), n_splits)
    fold_sizes = [0] * n_splits
    fold_edges = [0] * n_splits
    fold_label_counts: List[Dict[int, int]] = [defaultdict(int) for _ in range(n_splits)]
    tail_support_folds: Dict[int, Set[int]] = {int(r): set() for r in rotate_tail_labels}
    assign: Dict[int, int] = {}

    size_target = float(len(med_ids)) / float(n_splits)
    edge_target = float(sum(edges_by_med.get(m, 0) for m in med_ids)) / float(n_splits)
    label_target = {int(r): float(reaction_freq_map[int(r)]) / float(n_splits) for r in strat_labels}

    for med in med_ids:
        choices: List[Tuple[float, float, int]] = []
        med_edges = int(edges_by_med.get(med, 0))
        for fold in range(n_splits):
            if fold_sizes[fold] >= fold_caps[fold]:
                continue

            score = 0.0
            score += w_size * (
                ((fold_sizes[fold] + 1) - size_target) ** 2 - (fold_sizes[fold] - size_target) ** 2
            )
            score += w_edge * (
                ((fold_edges[fold] + med_edges) - edge_target) ** 2 - (fold_edges[fold] - edge_target) ** 2
            )

            for rid in med_non_tail[med]:
                cur = float(fold_label_counts[fold].get(rid, 0))
                tar = label_target[rid]
                score += w_label * (((cur + 1.0) - tar) ** 2 - (cur - tar) ** 2)

            for rid in med_rotate_tail[med]:
                supported = tail_support_folds[rid]
                cur_k = len(supported)
                new_k = cur_k if fold in supported else (cur_k + 1)
                desired = tail_target_folds[rid]
                before = float(max(0, desired - cur_k) ** 2)
                after = float(max(0, desired - new_k) ** 2)
                score += w_tail_rotate * (after - before)

            choices.append((score, float(rng.random()), fold))

        if not choices:
            fold = int(np.argmin(np.asarray(fold_sizes)))
        else:
            choices.sort(key=lambda x: (x[0], x[1]))
            fold = int(choices[0][2])

        assign[med] = fold
        fold_sizes[fold] += 1
        fold_edges[fold] += med_edges
        for rid in med_non_tail[med]:
            fold_label_counts[fold][rid] += 1
        for rid in med_rotate_tail[med]:
            tail_support_folds[rid].add(fold)

    size_dev = float(sum((sz - size_target) ** 2 for sz in fold_sizes))
    edge_dev = float(sum((ed - edge_target) ** 2 for ed in fold_edges))
    label_dev = 0.0
    for rid in strat_labels:
        tar = label_target[rid]
        for fold in range(n_splits):
            label_dev += (float(fold_label_counts[fold].get(rid, 0)) - tar) ** 2
    tail_rotate_dev = 0.0
    for rid, desired in tail_target_folds.items():
        k = len(tail_support_folds[rid])
        tail_rotate_dev += float(max(0, desired - k) ** 2)

    objective = w_size * size_dev + w_edge * edge_dev + w_label * label_dev + w_tail_rotate * tail_rotate_dev
    meta = {
        "objective": float(objective),
        "size_dev": float(size_dev),
        "edge_dev": float(edge_dev),
        "label_dev": float(label_dev),
        "tail_rotate_dev": float(tail_rotate_dev),
        "fold_sizes": [int(x) for x in fold_sizes],
        "fold_edges": [int(x) for x in fold_edges],
        "tail_rotate_labels": int(len(rotate_tail_labels)),
        "tail_rotate_support_folds": {str(k): int(len(v)) for k, v in tail_support_folds.items()},
    }
    return assign, meta


def split_train_val_dual_greedy(
    trainval_ids: List[int],
    labels_by_med: Dict[int, Set[int]],
    edges_by_med: Dict[int, int],
    strat_labels: Set[int],
    val_ratio: float,
    seed: int,
    w_label: float,
    w_edge: float,
    w_size: float,
) -> Tuple[List[int], List[int]]:
    rng = np.random.default_rng(seed)
    ids = list(int(x) for x in trainval_ids)
    rng.shuffle(ids)
    ids.sort(
        key=lambda m: (
            len(labels_by_med[m] & strat_labels),
            int(edges_by_med.get(m, 0)),
        ),
        reverse=True,
    )

    n_val = max(1, int(round(len(ids) * val_ratio)))
    n_val = min(n_val, len(ids) - 1)
    n_train = len(ids) - n_val

    size_target = {"val": float(n_val), "train": float(n_train)}
    edge_total = float(sum(edges_by_med.get(m, 0) for m in ids))
    edge_target = {
        "val": float(edge_total * val_ratio),
        "train": float(edge_total - edge_total * val_ratio),
    }

    label_freq_pool = {
        int(lbl): int(sum(1 for m in ids if lbl in labels_by_med[m]))
        for lbl in strat_labels
    }
    label_target = {
        "val": {int(lbl): float(cnt * val_ratio) for lbl, cnt in label_freq_pool.items()},
        "train": {int(lbl): float(cnt - cnt * val_ratio) for lbl, cnt in label_freq_pool.items()},
    }

    side_size = {"val": 0, "train": 0}
    side_edges = {"val": 0, "train": 0}
    side_counts = {"val": defaultdict(int), "train": defaultdict(int)}

    val_set: Set[int] = set()

    for idx, med in enumerate(ids):
        remaining = len(ids) - idx
        need_val = n_val - side_size["val"]
        need_train = n_train - side_size["train"]

        if need_val >= remaining:
            side = "val"
        elif need_train >= remaining:
            side = "train"
        elif side_size["val"] >= n_val:
            side = "train"
        elif side_size["train"] >= n_train:
            side = "val"
        else:
            med_labels = list(labels_by_med[med] & strat_labels)
            med_edges = int(edges_by_med.get(med, 0))

            def delta_for(which: str) -> float:
                sc = 0.0
                sc += w_size * (
                    ((side_size[which] + 1) - size_target[which]) ** 2 - (side_size[which] - size_target[which]) ** 2
                )
                sc += w_edge * (
                    ((side_edges[which] + med_edges) - edge_target[which]) ** 2
                    - (side_edges[which] - edge_target[which]) ** 2
                )
                for rid in med_labels:
                    cur = float(side_counts[which].get(rid, 0))
                    tar = label_target[which][rid]
                    sc += w_label * (((cur + 1.0) - tar) ** 2 - (cur - tar) ** 2)
                return float(sc)

            score_val = delta_for("val")
            score_train = delta_for("train")
            if score_val < score_train:
                side = "val"
            elif score_train < score_val:
                side = "train"
            else:
                side = "val" if float(rng.random()) < 0.5 else "train"

        side_size[side] += 1
        side_edges[side] += int(edges_by_med.get(med, 0))
        for rid in (labels_by_med[med] & strat_labels):
            side_counts[side][rid] += 1
        if side == "val":
            val_set.add(med)

    val_ids = sorted(int(m) for m in val_set)
    train_ids = sorted(int(m) for m in ids if m not in val_set)
    return train_ids, val_ids


def build_diagnostics(
    mr_df: pd.DataFrame,
    folds: List[Dict[str, List[int]]],
    reaction_freq: pd.Series,
    tail_max_freq: int,
) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    fold_rows = []
    coverage_rows = []

    all_reactions = sorted(int(x) for x in reaction_freq.index.tolist())
    tail_reactions = {int(r) for r, c in reaction_freq.items() if int(c) <= int(tail_max_freq)}
    non_tail_reactions = set(all_reactions) - tail_reactions

    for i, split in enumerate(folds):
        train_set = set(int(x) for x in split["train"])
        val_set = set(int(x) for x in split["val"])
        test_set = set(int(x) for x in split["test"])

        mr_train = mr_df[mr_df["medicine_id"].isin(train_set)]
        mr_val = mr_df[mr_df["medicine_id"].isin(val_set)]
        mr_test = mr_df[mr_df["medicine_id"].isin(test_set)]

        train_r = set(int(x) for x in mr_train["adr_id"].unique().tolist())
        val_r = set(int(x) for x in mr_val["adr_id"].unique().tolist())
        test_r = set(int(x) for x in mr_test["adr_id"].unique().tolist())
        tail_train_cov = len(train_r & tail_reactions)
        tail_test_cov = len(test_r & tail_reactions)
        nontail_missing_train = len(non_tail_reactions - train_r)
        nontail_missing_val = len(non_tail_reactions - val_r)
        nontail_missing_test = len(non_tail_reactions - test_r)

        fold_rows.append(
            {
                "fold": i,
                "n_train_medicine": len(train_set),
                "n_val_medicine": len(val_set),
                "n_test_medicine": len(test_set),
                "n_train_edges": len(mr_train),
                "n_val_edges": len(mr_val),
                "n_test_edges": len(mr_test),
                "tail_reaction_covered_in_train": int(tail_train_cov),
                "tail_reaction_covered_in_test": int(tail_test_cov),
                "nontail_missing_in_train": int(nontail_missing_train),
                "nontail_missing_in_val": int(nontail_missing_val),
                "nontail_missing_in_test": int(nontail_missing_test),
            }
        )

        for rid in all_reactions:
            coverage_rows.append(
                {
                    "fold": i,
                    "adr_id": rid,
                    "global_freq_medicine": int(reaction_freq.loc[rid]),
                    "is_tail": int(reaction_freq.loc[rid] <= tail_max_freq),
                    "train_support_medicine": int(
                        mr_train[mr_train["adr_id"] == rid]["medicine_id"].nunique()
                    ),
                    "val_support_medicine": int(
                        mr_val[mr_val["adr_id"] == rid]["medicine_id"].nunique()
                    ),
                    "test_support_medicine": int(
                        mr_test[mr_test["adr_id"] == rid]["medicine_id"].nunique()
                    ),
                }
            )

    fold_df = pd.DataFrame(fold_rows)
    cov_df = pd.DataFrame(coverage_rows)

    def _tail_presence(freq_value: int) -> Dict[str, int]:
        ids = [int(r) for r, c in reaction_freq.items() if int(c) == int(freq_value)]
        if not ids:
            return {}
        grp = (
            cov_df[cov_df["adr_id"].isin(ids)]
            .groupby("adr_id")["test_support_medicine"]
            .apply(lambda x: int((x > 0).sum()))
            .to_dict()
        )
        return {str(k): int(v) for k, v in grp.items()}

    freq2_test_presence = _tail_presence(2)
    freq3_test_presence = _tail_presence(3)

    edge_stats = {}
    for col in ("n_train_edges", "n_val_edges", "n_test_edges"):
        arr = fold_df[col].to_numpy(dtype=float)
        edge_stats[col] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=0)),
            "cv": _cv(arr),
            "min": int(arr.min()) if arr.size else 0,
            "max": int(arr.max()) if arr.size else 0,
        }

    meta = {
        "tail_max_freq": int(tail_max_freq),
        "n_tail_reactions": int((reaction_freq <= tail_max_freq).sum()),
        "n_freq2_reactions": int((reaction_freq == 2).sum()),
        "n_freq3_reactions": int((reaction_freq == 3).sum()),
        "freq2_test_presence_folds": freq2_test_presence,
        "freq3_test_presence_folds": freq3_test_presence,
        "edge_balance": edge_stats,
    }
    return fold_df, cov_df, meta


def main() -> None:
    args = parse_args()
    base_dir = Path(args.base_dir)
    mr_path = base_dir / args.mr_path
    medicine_path = base_dir / args.medicine_path
    out_dir = base_dir / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mr_df = pd.read_csv(mr_path)
    medicine_df = pd.read_csv(medicine_path)

    medicine_ids = medicine_df["medicine_id"].astype(int).unique()
    labels_by_med = build_multi_labels(mr_df, medicine_ids)
    edges_by_med = build_edges_by_med(mr_df, medicine_ids)
    reaction_freq = mr_df.groupby("adr_id")["medicine_id"].nunique().sort_index()
    reaction_freq_map = {int(r): int(c) for r, c in reaction_freq.items()}

    tail_labels = {int(r) for r, c in reaction_freq.items() if int(c) <= int(args.tail_max_freq)}
    strat_labels = set(int(r) for r in reaction_freq.index.tolist()) - tail_labels

    best_assign = None
    best_meta = None
    for i in range(max(1, int(args.n_restarts))):
        seed_i = int(args.seed) + i * 1009
        assign_i, meta_i = assign_test_folds_greedy(
            medicine_ids=medicine_ids.tolist(),
            labels_by_med=labels_by_med,
            edges_by_med=edges_by_med,
            reaction_freq_map=reaction_freq_map,
            strat_labels=strat_labels,
            tail_labels=tail_labels,
            n_splits=int(args.n_splits),
            seed=seed_i,
            tail_rotate_max_freq=int(args.tail_rotate_max_freq),
            w_label=float(args.w_label),
            w_edge=float(args.w_edge),
            w_size=float(args.w_size),
            w_tail_rotate=float(args.w_tail_rotate),
        )
        if best_meta is None or float(meta_i["objective"]) < float(best_meta["objective"]):
            best_assign = assign_i
            best_meta = dict(meta_i)
            best_meta["restart_index"] = int(i)
            best_meta["restart_seed"] = int(seed_i)

    assert best_assign is not None
    assert best_meta is not None

    folds: List[Dict[str, List[int]]] = []
    for fold_i in range(args.n_splits):
        test_ids = sorted([m for m, f in best_assign.items() if int(f) == int(fold_i)])
        trainval_ids = sorted([m for m, f in best_assign.items() if int(f) != int(fold_i)])

        train_ids, val_ids = split_train_val_dual_greedy(
            trainval_ids=trainval_ids,
            labels_by_med=labels_by_med,
            edges_by_med=edges_by_med,
            strat_labels=strat_labels,
            val_ratio=float(args.val_ratio),
            seed=int(args.seed) + int(fold_i) + 13,
            w_label=float(args.w_label),
            w_edge=float(args.w_edge),
            w_size=float(args.w_size),
        )

        fold_obj = {"train": sorted(train_ids), "val": sorted(val_ids), "test": test_ids}
        folds.append(fold_obj)

        fold_dir = out_dir / f"fold_{fold_i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"medicine_id": fold_obj["train"]}).to_csv(fold_dir / "train_medicine_ids.csv", index=False)
        pd.DataFrame({"medicine_id": fold_obj["val"]}).to_csv(fold_dir / "val_medicine_ids.csv", index=False)
        pd.DataFrame({"medicine_id": fold_obj["test"]}).to_csv(fold_dir / "test_medicine_ids.csv", index=False)

    fold_df, cov_df, diag_meta = build_diagnostics(
        mr_df=mr_df,
        folds=folds,
        reaction_freq=reaction_freq,
        tail_max_freq=int(args.tail_max_freq),
    )
    fold_df.to_csv(out_dir / "fold_stats.csv", index=False)
    cov_df.to_csv(out_dir / "adr_fold_coverage.csv", index=False)

    summary = {
        "script": "make_splits",
        "config": {
            "n_splits": int(args.n_splits),
            "val_ratio": float(args.val_ratio),
            "tail_max_freq": int(args.tail_max_freq),
            "tail_rotate_max_freq": int(args.tail_rotate_max_freq),
            "seed": int(args.seed),
            "n_restarts": int(args.n_restarts),
            "weights": {
                "w_label": float(args.w_label),
                "w_edge": float(args.w_edge),
                "w_size": float(args.w_size),
                "w_tail_rotate": float(args.w_tail_rotate),
            },
        },
        "dataset": {
            "n_medicine": int(len(medicine_ids)),
            "n_reaction": int(reaction_freq.shape[0]),
            "n_edges": int(len(mr_df)),
            "n_tail_reaction": int((reaction_freq <= args.tail_max_freq).sum()),
            "n_stratified_reaction": int((reaction_freq > args.tail_max_freq).sum()),
        },
        "best_test_assignment": best_meta,
        "diagnostics": diag_meta,
    }
    with open(out_dir / "split_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print("Done. Split files written to:", out_dir)
    print("Best test assignment objective:", f"{best_meta['objective']:.6f}")
    print("Check:", out_dir / "fold_stats.csv")
    print("Check:", out_dir / "adr_fold_coverage.csv")
    print("Check:", out_dir / "split_summary.json")


if __name__ == "__main__":
    main()
