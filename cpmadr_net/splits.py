"""Fixed-split sampling, evaluation, and stratified-training utilities."""
from __future__ import annotations

import copy
import os
import random
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from torch_geometric.loader import LinkNeighborLoader

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

from baselines.common.data import DEVICE, NUM_WORKERS, TARGET_ETYPE
from baselines.common.models import UniversalBaselineAllEdge
from .model import DoseAwareHSPESARGCNv13PathReadout
from .training import (
    aggregate_seed_summary, calc_metrics, compute_bce_pos_weight, find_best_threshold,
    paired_significance_tests, print_comparison, print_significance_table, resolve_pos_weight,
)

LABEL_BASE = "RGCN"
LABEL_V13_PATH = "DoseAwareHSP_ESA_PathReadout_v13"
DEFAULT_MR_PATH = "adr/edge_medicine_adr1.csv"

def get_ml_data_safe_for_medicine_set(
    pos_ei: torch.Tensor,
    neg_ratio: float,
    allowed_meds: List[int],
    num_reaction: int,
    all_pos_set: set,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_pos = pos_ei.size(1)
    num_neg = int(num_pos * float(neg_ratio))
    if num_pos == 0:
        raise ValueError("Positive edge set is empty for current split.")
    allowed_meds = [int(x) for x in allowed_meds]
    if not allowed_meds:
        raise ValueError("allowed_meds is empty for current split.")

    # Sample unique negatives (without replacement) to avoid duplicated test rows.
    allowed_med_set = set(allowed_meds)
    n_total_pairs = int(len(allowed_meds) * int(num_reaction))
    n_pos_in_allowed = int(sum(1 for m, _ in all_pos_set if int(m) in allowed_med_set))
    n_neg_available = n_total_pairs - n_pos_in_allowed
    if num_neg > n_neg_available:
        raise ValueError(
            f"Not enough unique negatives in allowed medicine set: requested={num_neg}, "
            f"available={n_neg_available}."
        )

    neg_list: List[List[int]] = []
    neg_seen = set()
    max_trials = max(1000, num_neg * 20)
    trials = 0
    while len(neg_list) < num_neg and trials < max_trials:
        m = random.choice(allowed_meds)
        r = random.randint(0, num_reaction - 1)
        pair = (m, r)
        if pair not in all_pos_set and pair not in neg_seen:
            neg_seen.add(pair)
            neg_list.append([m, r])
        trials += 1

    if len(neg_list) < num_neg:
        remaining = num_neg - len(neg_list)
        extra_candidates = []
        for m in allowed_meds:
            for r in range(num_reaction):
                pair = (m, r)
                if pair not in all_pos_set and pair not in neg_seen:
                    extra_candidates.append([m, r])
        if remaining > len(extra_candidates):
            raise ValueError(
                f"Not enough unique negatives after fallback: requested_remaining={remaining}, "
                f"available_remaining={len(extra_candidates)}."
            )
        random.shuffle(extra_candidates)
        neg_list.extend(extra_candidates[:remaining])

    neg_ei = torch.tensor(neg_list, dtype=torch.long).t()
    idx = torch.cat([pos_ei, neg_ei], dim=1)
    lab = torch.cat([torch.ones(num_pos), torch.zeros(num_neg)], dim=0)
    return idx, lab


@torch.no_grad()
def collect_probs(model, loader):
    model.eval()
    y_true, y_prob = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch)
        y_prob.extend(torch.sigmoid(logits).cpu().numpy())
        y_true.extend(batch[TARGET_ETYPE].edge_label.cpu().numpy())
    return np.array(y_true), np.array(y_prob)


@torch.no_grad()
def collect_probs_with_pairs(model, loader):
    model.eval()
    y_true, y_prob, med_ids, reac_ids = [], [], [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        logits = model(batch)
        probs = torch.sigmoid(logits).detach().cpu().numpy()
        labels = batch[TARGET_ETYPE].edge_label.detach().cpu().numpy()
        edge_idx = batch[TARGET_ETYPE].edge_label_index
        med_local = edge_idx[0]
        reac_local = edge_idx[1]
        med_global = batch["medicine"].n_id[med_local].detach().cpu().numpy()
        reac_global = batch["reaction"].n_id[reac_local].detach().cpu().numpy()
        y_prob.extend(probs.tolist())
        y_true.extend(labels.tolist())
        med_ids.extend(med_global.tolist())
        reac_ids.extend(reac_global.tolist())
    return (
        np.array(y_true),
        np.array(y_prob),
        np.array(med_ids, dtype=np.int64),
        np.array(reac_ids, dtype=np.int64),
    )


def build_tail_rows_for_fold(
    fold: int,
    te_true: np.ndarray,
    te_prob: np.ndarray,
    te_med: np.ndarray,
    te_reac: np.ndarray,
    fixed_threshold: float,
    best_thr: float,
    tail_reactions: set,
) -> List[Dict]:
    tail_mask = np.array([int(x) in tail_reactions for x in te_reac], dtype=bool)
    if not tail_mask.any():
        return []
    te_pred_a = (te_prob >= fixed_threshold).astype(int)
    te_pred_b = (te_prob >= best_thr).astype(int)
    rows: List[Dict] = []
    for j in np.where(tail_mask)[0]:
        rows.append(
            {
                "fold": fold,
                "medicine_id": int(te_med[j]),
                "reaction_id": int(te_reac[j]),
                "y_true": int(te_true[j]),
                "y_prob": float(te_prob[j]),
                "pred_thr0p5": int(te_pred_a[j]),
                "pred_valtuned": int(te_pred_b[j]),
                "best_thr_valtuned": float(best_thr),
            }
        )
    return rows


def build_all_rows_for_fold(
    fold: int,
    te_true: np.ndarray,
    te_prob: np.ndarray,
    te_med: np.ndarray,
    te_reac: np.ndarray,
    fixed_threshold: float,
    best_thr: float,
    reaction_freq_map: Dict[int, int],
) -> List[Dict]:
    te_pred_a = (te_prob >= fixed_threshold).astype(int)
    te_pred_b = (te_prob >= best_thr).astype(int)
    rows: List[Dict] = []
    for j in range(len(te_true)):
        rid = int(te_reac[j])
        rows.append(
            {
                "fold": fold,
                "medicine_id": int(te_med[j]),
                "reaction_id": rid,
                "reaction_freq_medicine": int(reaction_freq_map.get(rid, 0)),
                "y_true": int(te_true[j]),
                "y_prob": float(te_prob[j]),
                "pred_thr0p5": int(te_pred_a[j]),
                "pred_valtuned": int(te_pred_b[j]),
                "best_thr_valtuned": float(best_thr),
            }
        )
    return rows


def assign_bucket(freq: int, tail_max_freq: int, medium_max_freq: int) -> str:
    if freq <= tail_max_freq:
        return "tail"
    if freq <= medium_max_freq:
        return "medium"
    return "head"


def compute_bucket_metrics(df_pred: pd.DataFrame) -> pd.DataFrame:
    if df_pred.empty:
        return pd.DataFrame()
    rows = []
    for (model, seed, fold, bucket), grp in df_pred.groupby(["model", "seed", "fold", "bucket"]):
        y_true = grp["y_true"].to_numpy(dtype=np.int64)
        y_prob = grp["y_prob"].to_numpy(dtype=np.float64)
        y_pred_a = grp["pred_thr0p5"].to_numpy(dtype=np.int64)
        y_pred_b = grp["pred_valtuned"].to_numpy(dtype=np.int64)

        pr_auc = float("nan")
        roc_auc = float("nan")
        if len(np.unique(y_true)) >= 2:
            pr_auc = float(average_precision_score(y_true, y_prob))
            roc_auc = float(roc_auc_score(y_true, y_prob))
        mcc_a = (
            float(matthews_corrcoef(y_true, y_pred_a))
            if len(np.unique(np.concatenate([y_true, y_pred_a]))) >= 2
            else 0.0
        )
        mcc_b = (
            float(matthews_corrcoef(y_true, y_pred_b))
            if len(np.unique(np.concatenate([y_true, y_pred_b]))) >= 2
            else 0.0
        )

        rows.append(
            {
                "model": model,
                "seed": int(seed),
                "fold": int(fold),
                "bucket": bucket,
                "n_rows": int(len(grp)),
                "n_pos": int((y_true == 1).sum()),
                "roc_auc": roc_auc,
                "pr_auc": pr_auc,
                "acc_thr0p5": float(accuracy_score(y_true, y_pred_a)),
                "f1_thr0p5": float(f1_score(y_true, y_pred_a, zero_division=0)),
                "precision_thr0p5": float(precision_score(y_true, y_pred_a, zero_division=0)),
                "recall_thr0p5": float(recall_score(y_true, y_pred_a, zero_division=0)),
                "mcc_thr0p5": mcc_a,
                "acc_valtuned": float(accuracy_score(y_true, y_pred_b)),
                "f1_valtuned": float(f1_score(y_true, y_pred_b, zero_division=0)),
                "precision_valtuned": float(precision_score(y_true, y_pred_b, zero_division=0)),
                "recall_valtuned": float(recall_score(y_true, y_pred_b, zero_division=0)),
                "mcc_valtuned": mcc_b,
            }
        )
    return pd.DataFrame(rows)


def make_loaders(train_graph, tr_idx, tr_y, val_idx, val_y, te_idx, te_y, batch_size: int):
    tr_loader = LinkNeighborLoader(
        train_graph,
        edge_label_index=(TARGET_ETYPE, tr_idx),
        edge_label=tr_y,
        num_neighbors=[15, 10],
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    val_loader = LinkNeighborLoader(
        train_graph,
        edge_label_index=(TARGET_ETYPE, val_idx),
        edge_label=val_y,
        num_neighbors=[15, 10],
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
    )
    te_loader = LinkNeighborLoader(
        train_graph,
        edge_label_index=(TARGET_ETYPE, te_idx),
        edge_label=te_y,
        num_neighbors=[15, 10],
        batch_size=batch_size,
        num_workers=NUM_WORKERS,
    )
    return tr_loader, val_loader, te_loader


def load_fixed_medicine_splits(split_root_abs: str, n_splits: int) -> List[Dict[str, List[int]]]:
    out: List[Dict[str, List[int]]] = []
    for fold in range(n_splits):
        fold_dir = os.path.join(split_root_abs, f"fold_{fold}")
        tr = pd.read_csv(os.path.join(fold_dir, "train_medicine_ids.csv"))["medicine_id"].astype(int).tolist()
        va = pd.read_csv(os.path.join(fold_dir, "val_medicine_ids.csv"))["medicine_id"].astype(int).tolist()
        te = pd.read_csv(os.path.join(fold_dir, "test_medicine_ids.csv"))["medicine_id"].astype(int).tolist()
        out.append({"train": tr, "val": va, "test": te})
    return out


def build_fold_pos_edges_by_medicine(pos_edges: torch.Tensor, med_ids: List[int]) -> torch.Tensor:
    med_set = set(int(x) for x in med_ids)
    mask = torch.tensor([int(m.item()) in med_set for m in pos_edges[0]], dtype=torch.bool)
    return pos_edges[:, mask]


def _sample_neg_for_medicine_set(
    num_neg: int,
    allowed_meds: List[int],
    num_reaction: int,
    all_pos_set: set,
    unique_neg: bool,
) -> List[List[int]]:
    if num_neg <= 0:
        return []
    if unique_neg:
        allowed_med_set = set(allowed_meds)
        candidates: List[int] = []
        for m in allowed_meds:
            for r in range(num_reaction):
                if (m, r) not in all_pos_set:
                    candidates.append(m * int(num_reaction) + r)
        if num_neg > len(candidates):
            n_total_pairs = int(len(allowed_meds) * int(num_reaction))
            n_pos_in_allowed = int(sum(1 for m, _ in all_pos_set if int(m) in allowed_med_set))
            n_neg_available = n_total_pairs - n_pos_in_allowed
            raise ValueError(
                f"Not enough unique negatives in allowed medicine set: requested={num_neg}, "
                f"available={len(candidates)}, theoretical_available={n_neg_available}."
            )
        chosen = random.sample(candidates, num_neg)
        return [[cid // int(num_reaction), cid % int(num_reaction)] for cid in chosen]

    # with replacement
    neg_list: List[List[int]] = []
    while len(neg_list) < num_neg:
        m = random.choice(allowed_meds)
        r = random.randint(0, int(num_reaction) - 1)
        if (m, r) not in all_pos_set:
            neg_list.append([m, r])
    return neg_list


def _build_ml_edges_for_medicine_set(
    pos_ei: torch.Tensor,
    neg_ratio: float,
    allowed_meds: List[int],
    num_reaction: int,
    all_pos_set: set,
    unique_neg: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_pos = int(pos_ei.size(1))
    num_neg = int(num_pos * float(neg_ratio))
    if num_pos == 0:
        raise ValueError("Positive edge set is empty for current split.")
    allowed_meds = [int(x) for x in allowed_meds]
    if not allowed_meds:
        raise ValueError("allowed_meds is empty for current split.")

    neg_list = _sample_neg_for_medicine_set(
        num_neg=num_neg,
        allowed_meds=allowed_meds,
        num_reaction=int(num_reaction),
        all_pos_set=all_pos_set,
        unique_neg=unique_neg,
    )
    neg_ei = (
        torch.tensor(neg_list, dtype=torch.long).t().contiguous()
        if num_neg > 0
        else torch.empty((2, 0), dtype=torch.long)
    )

    idx = torch.cat([pos_ei, neg_ei], dim=1)
    lab = torch.cat([torch.ones(num_pos), torch.zeros(num_neg)], dim=0)
    return idx, lab


def _prepare_tail_df_for_plot(tail_df: pd.DataFrame) -> pd.DataFrame:
    if tail_df.empty:
        return tail_df.copy()
    need_cols = ["model", "seed", "fold", "y_true", "y_prob"]
    miss = [c for c in need_cols if c not in tail_df.columns]
    if miss:
        raise KeyError(f"tail_df missing required columns for plotting: {miss}")
    d = tail_df[need_cols].copy()
    d["seed"] = pd.to_numeric(d["seed"], errors="coerce")
    d["fold"] = pd.to_numeric(d["fold"], errors="coerce")
    d["y_true"] = pd.to_numeric(d["y_true"], errors="coerce")
    d["y_prob"] = pd.to_numeric(d["y_prob"], errors="coerce")
    d = d.dropna(subset=["seed", "fold", "y_true", "y_prob"]).copy()
    d["seed"] = d["seed"].astype(int)
    d["fold"] = d["fold"].astype(int)
    d["y_true"] = d["y_true"].astype(int)
    return d


def _plot_tail_pr_curve(tail_df: pd.DataFrame, out_root: str) -> List[str]:
    if plt is None:
        print("[WARN] matplotlib unavailable, skip tail PR curve export.")
        return []
    d = _prepare_tail_df_for_plot(tail_df)
    if d.empty:
        print("[WARN] Empty tail dataframe, skip tail PR curve export.")
        return []

    curve_rows: List[Dict] = []
    fig, ax = plt.subplots(figsize=(7.5, 6.0))
    plotted = 0
    for model, grp in d.groupby("model", sort=True):
        y_true = grp["y_true"].to_numpy(dtype=np.int64)
        y_prob = grp["y_prob"].to_numpy(dtype=np.float64)
        if y_true.size == 0 or len(np.unique(y_true)) < 2:
            print(f"[WARN] {model}: tail labels have <2 classes, skip PR curve.")
            continue
        precision, recall, _ = precision_recall_curve(y_true, y_prob)
        ap = float(average_precision_score(y_true, y_prob))
        order = np.argsort(recall)
        ax.plot(recall[order], precision[order], linewidth=2.0, label=f"{model} (AP={ap:.4f})")
        plotted += 1
        for i in range(len(precision)):
            curve_rows.append(
                {
                    "model": model,
                    "recall": float(recall[i]),
                    "precision": float(precision[i]),
                    "ap": ap,
                    "n_rows": int(y_true.size),
                    "n_pos": int((y_true == 1).sum()),
                }
            )

    if plotted == 0:
        plt.close(fig)
        print("[WARN] No valid PR curve to plot for tail.")
        return []

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Tail PR Curve")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.02)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best", frameon=True)
    fig.tight_layout()

    png_path = os.path.join(out_root, "tail_pr_curve_fixed_tail_splits_v13_path.png")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    csv_path = os.path.join(out_root, "tail_pr_curve_points_fixed_tail_splits_v13_path.csv")
    pd.DataFrame(curve_rows).to_csv(csv_path, index=False)
    return [png_path, csv_path]


def _plot_tail_topk_curves(tail_df: pd.DataFrame, out_root: str, max_k: int, k_step: int) -> List[str]:
    if plt is None:
        print("[WARN] matplotlib unavailable, skip tail Top-K curve export.")
        return []
    d = _prepare_tail_df_for_plot(tail_df)
    if d.empty:
        print("[WARN] Empty tail dataframe, skip tail Top-K curve export.")
        return []

    if k_step <= 0:
        k_step = 1
    max_k = max(1, int(max_k))

    unit_sizes = d.groupby(["model", "seed", "fold"]).size()
    if unit_sizes.empty:
        print("[WARN] No tail units found for Top-K plotting.")
        return []
    max_k_eff = min(max_k, int(unit_sizes.min()))
    if max_k_eff <= 0:
        print("[WARN] max_k_eff <= 0, skip tail Top-K plotting.")
        return []

    ks = list(range(1, max_k_eff + 1, int(k_step)))
    if ks[-1] != max_k_eff:
        ks.append(max_k_eff)

    unit_rows: List[Dict] = []
    for (model, seed, fold), grp in d.groupby(["model", "seed", "fold"], sort=True):
        s = grp.sort_values("y_prob", ascending=False)
        y = s["y_true"].to_numpy(dtype=np.int64)
        n = int(y.size)
        total_pos = int((y == 1).sum())
        if n == 0 or total_pos == 0:
            continue
        cum_tp = np.cumsum(y)
        for k in ks:
            if k > n:
                continue
            tp = int(cum_tp[k - 1])
            unit_rows.append(
                {
                    "model": model,
                    "seed": int(seed),
                    "fold": int(fold),
                    "k": int(k),
                    "precision_at_k": float(tp / k),
                    "recall_at_k": float(tp / total_pos),
                    "tp_at_k": tp,
                    "total_pos": total_pos,
                    "n_rows": n,
                }
            )

    if not unit_rows:
        print("[WARN] No valid Top-K points for tail plotting.")
        return []

    unit_df = pd.DataFrame(unit_rows)
    summary_df = (
        unit_df.groupby(["model", "k"], as_index=False)
        .agg(
            precision_at_k_mean=("precision_at_k", "mean"),
            precision_at_k_std=("precision_at_k", "std"),
            recall_at_k_mean=("recall_at_k", "mean"),
            recall_at_k_std=("recall_at_k", "std"),
            n_units=("precision_at_k", "count"),
        )
        .sort_values(["model", "k"])
    )
    for c in ["precision_at_k_std", "recall_at_k_std"]:
        summary_df[c] = summary_df[c].fillna(0.0)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharex=True)
    ax_p, ax_r = axes
    for model, grp in summary_df.groupby("model", sort=True):
        x = grp["k"].to_numpy(dtype=np.int64)
        p_mean = grp["precision_at_k_mean"].to_numpy(dtype=np.float64)
        p_std = grp["precision_at_k_std"].to_numpy(dtype=np.float64)
        r_mean = grp["recall_at_k_mean"].to_numpy(dtype=np.float64)
        r_std = grp["recall_at_k_std"].to_numpy(dtype=np.float64)
        p_lo = np.clip(p_mean - p_std, 0.0, 1.0)
        p_hi = np.clip(p_mean + p_std, 0.0, 1.0)
        r_lo = np.clip(r_mean - r_std, 0.0, 1.0)
        r_hi = np.clip(r_mean + r_std, 0.0, 1.0)

        ax_p.plot(x, p_mean, marker="o", linewidth=1.8, label=model)
        ax_p.fill_between(x, p_lo, p_hi, alpha=0.15)
        ax_r.plot(x, r_mean, marker="o", linewidth=1.8, label=model)
        ax_r.fill_between(x, r_lo, r_hi, alpha=0.15)

    ax_p.set_title("Tail Precision@K")
    ax_p.set_ylabel("Precision")
    ax_p.set_xlabel("K")
    ax_p.set_ylim(0.0, 1.02)
    ax_p.grid(True, linestyle="--", alpha=0.4)

    ax_r.set_title("Tail Recall@K")
    ax_r.set_ylabel("Recall")
    ax_r.set_xlabel("K")
    ax_r.set_ylim(0.0, 1.02)
    ax_r.grid(True, linestyle="--", alpha=0.4)

    handles, labels = ax_p.get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)))
    fig.suptitle("Tail Top-K Curves (mean +/- std over seed x fold units)", y=1.02)
    fig.tight_layout()

    png_path = os.path.join(out_root, "tail_topk_curves_fixed_tail_splits_v13_path.png")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    summary_csv = os.path.join(out_root, "tail_topk_curve_summary_fixed_tail_splits_v13_path.csv")
    unit_csv = os.path.join(out_root, "tail_topk_curve_per_unit_fixed_tail_splits_v13_path.csv")
    summary_df.to_csv(summary_csv, index=False)
    unit_df.to_csv(unit_csv, index=False)
    return [png_path, summary_csv, unit_csv]


def train_one_cycle_fixed_splits_v13_path_unique_eval_neg(
    model_name: str,
    model_builder,
    data_full,
    med_feat,
    reac_feat,
    target_feat,
    herb_feat,
    seed: int,
    lr: float,
    weight_decay: float,
    max_epochs: int,
    patience: int,
    eval_every: int,
    batch_size: int,
    fixed_threshold: float,
    fixed_splits: List[Dict[str, List[int]]],
    use_imbalanced_bce: bool,
    compute_pos_weight_fn,
    pos_weight_mode: str,
    pos_weight_scale: float,
    pos_weight_max: float,
    tail_reactions: set,
    reaction_freq_map: Dict[int, int],
    train_neg_unique: bool = False,
    val_neg_unique: bool = True,
    test_neg_unique: bool = True,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    seed_everything(seed)
    pos_edges = data_full[TARGET_ETYPE].edge_index
    all_pos_set = set(zip(pos_edges[0].tolist(), pos_edges[1].tolist()))
    n_reac = int(data_full["reaction"].num_nodes)

    fold_metrics = []
    tail_rows: List[Dict] = []
    all_test_rows: List[Dict] = []
    for fold, split in enumerate(fixed_splits):
        train_pos = build_fold_pos_edges_by_medicine(pos_edges, split["train"])
        val_pos = build_fold_pos_edges_by_medicine(pos_edges, split["val"])
        test_pos = build_fold_pos_edges_by_medicine(pos_edges, split["test"])
        if train_pos.size(1) == 0 or val_pos.size(1) == 0 or test_pos.size(1) == 0:
            raise ValueError(f"fold_{fold} has empty positive edges in train/val/test.")

        train_graph = copy.deepcopy(data_full)
        del train_graph[TARGET_ETYPE]
        train_graph[TARGET_ETYPE].edge_index = train_pos
        train_graph = T.ToUndirected()(train_graph)

        tr_idx, tr_y = _build_ml_edges_for_medicine_set(
            train_pos, 1.0, split["train"], n_reac, all_pos_set, unique_neg=train_neg_unique
        )
        val_idx, val_y = _build_ml_edges_for_medicine_set(
            val_pos, 1.0, split["val"], n_reac, all_pos_set, unique_neg=val_neg_unique
        )
        te_idx, te_y = _build_ml_edges_for_medicine_set(
            test_pos, 1.0, split["test"], n_reac, all_pos_set, unique_neg=test_neg_unique
        )
        tr_loader, val_loader, te_loader = make_loaders(
            train_graph, tr_idx, tr_y, val_idx, val_y, te_idx, te_y, batch_size=batch_size
        )

        pos_w = None
        if use_imbalanced_bce and compute_pos_weight_fn is not None:
            n_med_active = len(split["train"])
            n_pos_sup = int(train_pos.size(1))
            base_pos_w = compute_pos_weight_fn(n_med_active, n_reac, n_pos_sup, max_w=pos_weight_max).to(DEVICE)
            pos_w = resolve_pos_weight(base_pos_w, mode=pos_weight_mode, scale=pos_weight_scale)

        model = model_builder(train_graph, med_feat, reac_feat, target_feat, herb_feat).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_v_auc = -1.0
        best_res: Dict[str, float] = {}
        best_tail_rows_fold: List[Dict] = []
        best_all_rows_fold: List[Dict] = []
        early_stop_counter = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            for batch in tr_loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                logits = model(batch)
                loss = F.binary_cross_entropy_with_logits(
                    logits,
                    batch[TARGET_ETYPE].edge_label.float(),
                    pos_weight=pos_w,
                )
                loss.backward()
                optimizer.step()

            if epoch % eval_every != 0:
                continue

            val_true, val_prob = collect_probs(model, val_loader)
            v_auc, _, _, _, _, _, _ = calc_metrics(val_true, val_prob, threshold=fixed_threshold)
            if v_auc > best_v_auc:
                best_v_auc = v_auc
                early_stop_counter = 0
                best_thr = find_best_threshold(val_true, val_prob)
                tr_true, tr_prob = collect_probs(model, tr_loader)
                te_true, te_prob, te_med, te_reac = collect_probs_with_pairs(model, te_loader)
                tr_auc, _, _, _, _, _, _ = calc_metrics(tr_true, tr_prob, threshold=fixed_threshold)
                te_auc, te_pr, _, _, _, _, _ = calc_metrics(te_true, te_prob, threshold=fixed_threshold)
                _, _, te_acc_a, te_f1_a, te_mcc_a, te_prec_a, te_rec_a = calc_metrics(
                    te_true, te_prob, threshold=fixed_threshold
                )
                _, _, te_acc_b, te_f1_b, te_mcc_b, te_prec_b, te_rec_b = calc_metrics(
                    te_true, te_prob, threshold=best_thr
                )
                best_res = {
                    "tr_auc": tr_auc,
                    "v_auc": v_auc,
                    "te_auc": te_auc,
                    "te_pr": te_pr,
                    "te_f1_thr0p5": te_f1_a,
                    "te_acc_thr0p5": te_acc_a,
                    "te_mcc_thr0p5": te_mcc_a,
                    "te_precision_thr0p5": te_prec_a,
                    "te_recall_thr0p5": te_rec_a,
                    "best_thr_valtuned": best_thr,
                    "te_f1_valtuned": te_f1_b,
                    "te_acc_valtuned": te_acc_b,
                    "te_mcc_valtuned": te_mcc_b,
                    "te_precision_valtuned": te_prec_b,
                    "te_recall_valtuned": te_rec_b,
                    "n_train_pos_edges": int(train_pos.size(1)),
                    "n_val_pos_edges": int(val_pos.size(1)),
                    "n_test_pos_edges": int(test_pos.size(1)),
                }
                best_tail_rows_fold = build_tail_rows_for_fold(
                    fold=fold,
                    te_true=te_true,
                    te_prob=te_prob,
                    te_med=te_med,
                    te_reac=te_reac,
                    fixed_threshold=fixed_threshold,
                    best_thr=best_thr,
                    tail_reactions=tail_reactions,
                )
                best_all_rows_fold = build_all_rows_for_fold(
                    fold=fold,
                    te_true=te_true,
                    te_prob=te_prob,
                    te_med=te_med,
                    te_reac=te_reac,
                    fixed_threshold=fixed_threshold,
                    best_thr=best_thr,
                    reaction_freq_map=reaction_freq_map,
                )
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                break

        fold_metrics.append(best_res)
        tail_rows.extend(best_tail_rows_fold)
        all_test_rows.extend(best_all_rows_fold)
        print(
            f"[{model_name}][seed={seed}] fold {fold + 1}/{len(fixed_splits)} "
            f"best_v_auc={best_v_auc:.4f} te_auc={best_res['te_auc']:.4f} | "
            f"train_unique={train_neg_unique} val_unique={val_neg_unique} test_unique={test_neg_unique}"
        )

    return pd.DataFrame(fold_metrics), pd.DataFrame(tail_rows), pd.DataFrame(all_test_rows)
