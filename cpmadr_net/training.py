"""Shared training, evaluation, and significance utilities for CPMADR-Net."""
from __future__ import annotations

import random
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
import torch
from sklearn.metrics import (
    accuracy_score, average_precision_score, f1_score, matthews_corrcoef,
    precision_score, recall_score, roc_auc_score,
)
from torch_geometric.loader import LinkNeighborLoader

from baselines.common.data import DEVICE, NUM_WORKERS, TARGET_ETYPE

IMPROVED_MODEL_LABEL = "DoseAwareRGCN_v02"

def compute_bce_pos_weight(
    num_medicine: int,
    num_reaction: int,
    num_train_pos_edges: int,
    min_w: float = 1.0,
    max_w: float = 50.0,
    dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """
    Scalar pos_weight for F.binary_cross_entropy_with_logits(..., pos_weight=...).
    Uses (#possible pairs - #train positives) / #train positives, clamped to [min_w, max_w].
    """
    num_possible = max(1, int(num_medicine) * int(num_reaction))
    n_pos = max(1, int(num_train_pos_edges))
    num_neg_theory = max(1.0, float(num_possible - n_pos))
    w = num_neg_theory / float(n_pos)
    w = max(float(min_w), min(float(max_w), w))
    t = torch.tensor([w], dtype=dtype or torch.float32)
    return t


def resolve_pos_weight(
    base_pos_w: torch.Tensor | None,
    mode: str,
    scale: float,
) -> torch.Tensor | None:
    if base_pos_w is None:
        return None
    w = float(base_pos_w.item())
    if mode == "none":
        return None
    if mode == "sqrt":
        w = float(np.sqrt(max(1e-12, w)))
    elif mode == "linear":
        w = w
    else:
        raise ValueError(f"Unsupported pos_weight mode: {mode}")
    w = max(1.0, w * float(scale))
    return torch.tensor([w], dtype=base_pos_w.dtype, device=base_pos_w.device)


def get_ml_data_safe(ei: torch.Tensor, ratio: float, num_m: int, num_r: int, pos_set) -> Tuple[torch.Tensor, torch.Tensor]:
    num_pos = ei.size(1)
    num_neg = int(num_pos * ratio)
    neg_list = []
    while len(neg_list) < num_neg:
        m, r = random.randint(0, num_m - 1), random.randint(0, num_r - 1)
        if (m, r) not in pos_set:
            neg_list.append([m, r])
    idx = torch.cat([ei, torch.tensor(neg_list).t()], dim=1)
    lab = torch.cat([torch.ones(num_pos), torch.zeros(num_neg)], dim=0)
    return idx, lab


def calc_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5):
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0
    return (
        roc_auc_score(y_true, y_prob),
        average_precision_score(y_true, y_prob),
        accuracy_score(y_true, y_pred),
        f1_score(y_true, y_pred),
        matthews_corrcoef(y_true, y_pred),
        precision_score(y_true, y_pred, zero_division=0),
        recall_score(y_true, y_pred, zero_division=0),
    )


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


def find_best_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    cand = np.linspace(0.2, 0.8, 61)
    best_t, best_mcc = 0.5, -1.0
    for t in cand:
        _, _, _, _, mcc, _, _ = calc_metrics(y_true, y_prob, threshold=float(t))
        if mcc > best_mcc:
            best_mcc = mcc
            best_t = float(t)
    return best_t


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


def aggregate_seed_summary(df: pd.DataFrame, seeds: List[int], metrics: List[str]) -> Dict[str, Dict[str, float]]:
    out = {}
    for m in metrics:
        if m not in df.columns:
            continue
        seed_means = df.groupby("seed")[m].mean()
        out[m] = {
            "mean": float(seed_means.mean()),
            "std": float(seed_means.std()) if len(seeds) > 1 else 0.0,
        }
    return out


def print_comparison(title: str, name_a: str, sum_a: Dict, name_b: str, sum_b: Dict, metrics: List[str]) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"{'metric':<28} | {name_a:<22} | {name_b:<22}")
    print("-" * 72)
    for m in metrics:
        if m not in sum_a and m not in sum_b:
            continue
        a = f"{sum_a[m]['mean']:.4f} ± {sum_a[m]['std']:.4f}" if m in sum_a else "—"
        b = f"{sum_b[m]['mean']:.4f} ± {sum_b[m]['std']:.4f}" if m in sum_b else "—"
        print(f"{m:<28} | {a:<22} | {b:<22}")


def paired_significance_tests(
    df: pd.DataFrame,
    metrics: List[str],
    baseline_name: str = "RGCN",
    improved_name: str = IMPROVED_MODEL_LABEL,
    protocol_label: str = "",
) -> pd.DataFrame:
    d = df.copy()
    d["fold"] = d.groupby(["seed", "model"]).cumcount()
    bl = d[d["model"] == baseline_name].copy()
    im = d[d["model"] == improved_name].copy()
    merge_cols = ["seed", "fold"]
    merged = bl[merge_cols + metrics].merge(
        im[merge_cols + metrics],
        on=merge_cols,
        suffixes=("_baseline", "_improved"),
    )
    rows = []
    n_pairs = len(merged)
    for m in metrics:
        bcol, icol = f"{m}_baseline", f"{m}_improved"
        x_b = merged[bcol].to_numpy(dtype=np.float64)
        x_i = merged[icol].to_numpy(dtype=np.float64)
        diff = x_i - x_b
        mean_diff = float(np.mean(diff))
        std_diff = float(np.std(diff, ddof=1)) if n_pairs > 1 else 0.0
        cohens_dz = float(mean_diff / std_diff) if std_diff > 1e-12 else float("nan")

        if n_pairs < 2:
            rows.append(
                {
                    "protocol": protocol_label,
                    "metric": m,
                    "n_pairs": n_pairs,
                    "mean_diff_improved_minus_baseline": mean_diff,
                    "cohens_dz": cohens_dz,
                    "ttest_statistic": float("nan"),
                    "ttest_pvalue_two_sided": float("nan"),
                    "wilcoxon_statistic": float("nan"),
                    "wilcoxon_pvalue_two_sided": float("nan"),
                }
            )
            continue

        t_stat, p_t = stats.ttest_rel(x_i, x_b)
        p_w = float("nan")
        w_stat = float("nan")
        try:
            if np.allclose(diff, 0.0):
                p_w = 1.0
                w_stat = 0.0
            else:
                try:
                    w_res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided", mode="auto")
                except TypeError:
                    w_res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                w_stat = float(w_res.statistic)
                p_w = float(w_res.pvalue)
        except ValueError:
            pass

        rows.append(
            {
                "protocol": protocol_label,
                "metric": m,
                "n_pairs": n_pairs,
                "mean_diff_improved_minus_baseline": mean_diff,
                "cohens_dz": cohens_dz,
                "ttest_statistic": float(t_stat),
                "ttest_pvalue_two_sided": float(p_t),
                "wilcoxon_statistic": w_stat,
                "wilcoxon_pvalue_two_sided": p_w,
            }
        )
    return pd.DataFrame(rows)


def print_significance_table(sig_df: pd.DataFrame, title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)
    print(f"{'metric':<28} | {'meanΔ':>10} | {'dz':>8} | {'p_t':>10} | {'p_w':>10}")
    print("-" * 72)
    for _, r in sig_df.iterrows():
        print(
            f"{r['metric']:<28} | {r['mean_diff_improved_minus_baseline']:>10.4f} | "
            f"{r['cohens_dz']:>8.3f} | {r['ttest_pvalue_two_sided']:>10.4g} | {r['wilcoxon_pvalue_two_sided']:>10.4g}"
        )
