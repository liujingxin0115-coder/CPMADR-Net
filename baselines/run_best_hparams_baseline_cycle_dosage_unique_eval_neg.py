"""
Run `run_baseline_cycle` with dosage data and best hyperparameters, but use
unique negatives on validation/test by default.

Differences vs `run_best_hparams_baseline_cycle_dosage.py`:
  - train negatives: with replacement by default
  - val/test negatives: without replacement by default
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from torch_geometric.loader import LinkNeighborLoader
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baselines.common.data import (  # noqa: E402
    DEVICE,
    NUM_WORKERS,
    TARGET_ETYPE,
    load_tcm_full_data,
    prepare_unified_features,
    seed_everything,
)
from baselines.common.metrics import calculate_metrics, evaluate_model  # noqa: E402
from baselines.common.models import UniversalBaselineAllEdge  # noqa: E402
from baselines.run_no_dosage_best_baselines import (  # noqa: E402
    build_model_args_from_best,
    load_best_hparams,
)


def mean_std_across_folds(seed_dfs: List[pd.DataFrame], col: str) -> Tuple[float, float]:
    vals = np.concatenate([df[col].to_numpy(dtype=float) for df in seed_dfs])
    m = float(np.mean(vals))
    if len(vals) <= 1:
        return m, 0.0
    return m, float(np.std(vals, ddof=1))


def format_mean_pm_std(mean: float, std: float, decimals: int = 4) -> str:
    return f"{mean:.{decimals}f}+-{std:.{decimals}f}"


def format_report_row(model_name: str, agg: Dict[str, Any]) -> str:
    return (
        f"{model_name:<12}"
        f"| {format_mean_pm_std(agg['tr_auc_mean'], agg['tr_auc_std']):<13} "
        f"| {format_mean_pm_std(agg['v_auc_mean'], agg['v_auc_std']):<13} "
        f"| {format_mean_pm_std(agg['te_auc_mean'], agg['te_auc_std']):<13} "
        f"| {format_mean_pm_std(agg['te_pr_mean'], agg['te_pr_std']):<13} "
        f"| {format_mean_pm_std(agg['te_precision_mean'], agg['te_precision_std']):<13} "
        f"| {format_mean_pm_std(agg['te_recall_mean'], agg['te_recall_std']):<13} "
        f"| {format_mean_pm_std(agg['te_acc_mean'], agg['te_acc_std']):<13} "
        f"| {format_mean_pm_std(agg['te_f1_mean'], agg['te_f1_std']):<13} "
        f"| {format_mean_pm_std(agg['te_mcc_mean'], agg['te_mcc_std'])}"
    )


def _parse_models(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def _sample_neg_with_replacement(num_neg: int, num_m: int, num_r: int, pos_set: set) -> List[List[int]]:
    neg_list: List[List[int]] = []
    while len(neg_list) < num_neg:
        m = random.randint(0, num_m - 1)
        r = random.randint(0, num_r - 1)
        if (m, r) not in pos_set:
            neg_list.append([m, r])
    return neg_list


def _sample_neg_unique(num_neg: int, num_m: int, num_r: int, pos_set: set) -> List[List[int]]:
    total = num_m * num_r
    candidate_ids = [idx for idx in range(total) if ((idx // num_r), (idx % num_r)) not in pos_set]
    if num_neg > len(candidate_ids):
        raise ValueError(
            f"Not enough unique negatives: requested={num_neg}, available={len(candidate_ids)} "
            f"(num_m={num_m}, num_r={num_r}, |pos_set|={len(pos_set)})."
        )
    chosen = random.sample(candidate_ids, num_neg)
    return [[cid // num_r, cid % num_r] for cid in chosen]


def _build_ml_edges(
    ei: torch.Tensor,
    ratio: float,
    num_m: int,
    num_r: int,
    pos_set: set,
    unique_neg: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_pos = int(ei.size(1))
    num_neg = int(num_pos * float(ratio))
    if num_pos <= 0:
        raise ValueError("Positive edge set is empty for current split.")

    if unique_neg:
        neg_list = _sample_neg_unique(num_neg, num_m, num_r, pos_set)
    else:
        neg_list = _sample_neg_with_replacement(num_neg, num_m, num_r, pos_set)

    idx = torch.cat([ei, torch.tensor(neg_list, dtype=torch.long).t()], dim=1)
    lab = torch.cat([torch.ones(num_pos), torch.zeros(num_neg)], dim=0)
    return idx, lab


def run_baseline_cycle_unique_eval_neg(
    model_name: str,
    data_full,
    med_feat,
    reac_feat,
    target_feat,
    herb_feat,
    seed: int = 42,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_channels: int = 64,
    num_conv_layers: int = 2,
    transformer_num_layers: int = 2,
    transformer_nhead: int = 4,
    max_epochs: int = 500,
    patience: int = 20,
    eval_every: int = 5,
    n_splits: int = 5,
    train_neg_unique: bool = False,
    val_neg_unique: bool = True,
    test_neg_unique: bool = True,
) -> Tuple[pd.DataFrame, List[float]]:
    seed_everything(seed)

    target_etype = TARGET_ETYPE
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pos_edges = data_full[target_etype].edge_index
    all_pos_set = set(zip(pos_edges[0].tolist(), pos_edges[1].tolist()))

    fold_metrics: List[dict] = []
    all_folds_loss: List[List[float]] = []

    med_herb_bert_agg: Optional[np.ndarray] = None
    reac_bert_np: Optional[np.ndarray] = None
    reac_target_bert_agg: Optional[np.ndarray] = None
    if model_name in ["RF", "XGBoost"]:
        num_m_nodes = int(data_full["medicine"].num_nodes)
        num_r_nodes = int(data_full["reaction"].num_nodes)
        herb_x_np = data_full["herb"].x.detach().cpu().numpy().astype(np.float32)
        reac_bert_np = data_full["reaction"].x.detach().cpu().numpy().astype(np.float32)
        target_x_np = data_full["target"].x.detach().cpu().numpy().astype(np.float32)
        mh_ei = data_full["medicine", "contains", "herb"].edge_index.cpu().numpy()
        mh_dose = data_full["medicine", "contains", "herb"].edge_weight.cpu().numpy().astype(np.float64)
        h_dim = herb_x_np.shape[1]
        acc = np.zeros((num_m_nodes, h_dim), dtype=np.float64)
        wsum = np.zeros(num_m_nodes, dtype=np.float64)
        for k in range(mh_ei.shape[1]):
            m, h = int(mh_ei[0, k]), int(mh_ei[1, k])
            w = float(mh_dose[k])
            acc[m] += w * herb_x_np[h]
            wsum[m] += w
        med_herb_bert_agg = (acc / np.maximum(wsum[:, None], 1e-6)).astype(np.float32)

        tr_ei = data_full["target", "causes", "reaction"].edge_index.cpu().numpy()
        t_dim = target_x_np.shape[1]
        r_acc = np.zeros((num_r_nodes, t_dim), dtype=np.float64)
        r_cnt = np.zeros(num_r_nodes, dtype=np.int64)
        for k in range(tr_ei.shape[1]):
            t, r = int(tr_ei[0, k]), int(tr_ei[1, k])
            r_acc[r] += target_x_np[t]
            r_cnt[r] += 1
        reac_target_bert_agg = np.zeros((num_r_nodes, t_dim), dtype=np.float32)
        mask = r_cnt > 0
        reac_target_bert_agg[mask] = (r_acc[mask] / r_cnt[mask].astype(np.float64)[:, None]).astype(np.float32)

    num_m = int(data_full["medicine"].num_nodes)
    num_r = int(data_full["reaction"].num_nodes)

    for fold, (tv_idx, test_idx) in enumerate(kf.split(np.arange(pos_edges.size(1)))):
        test_ei, tv_ei = pos_edges[:, test_idx], pos_edges[:, tv_idx]
        perm = np.random.permutation(tv_ei.size(1))
        tv_ei = tv_ei[:, perm]
        val_size = max(1, int(tv_ei.size(1) * 0.1))
        val_ei, sup_ei = tv_ei[:, :val_size], tv_ei[:, val_size:]

        train_graph = copy.deepcopy(data_full)
        del train_graph[target_etype]
        train_graph[target_etype].edge_index = sup_ei
        train_graph = T.ToUndirected()(train_graph)

        m_deg = torch.zeros(data_full["medicine"].num_nodes).index_add_(0, sup_ei[0], torch.ones(sup_ei.size(1)))
        r_deg = torch.zeros(data_full["reaction"].num_nodes).index_add_(0, sup_ei[1], torch.ones(sup_ei.size(1)))

        def extract_ml_features(ei: torch.Tensor) -> np.ndarray:
            m_idx, r_idx = ei[0].numpy().astype(np.int64), ei[1].numpy().astype(np.int64)
            if model_name in ["RF", "XGBoost"]:
                assert (
                    med_herb_bert_agg is not None
                    and reac_bert_np is not None
                    and reac_target_bert_agg is not None
                )
                return np.hstack(
                    [
                        med_feat[m_idx].detach().cpu().numpy(),
                        med_herb_bert_agg[m_idx],
                        reac_feat[r_idx].detach().cpu().numpy(),
                        reac_bert_np[r_idx],
                        reac_target_bert_agg[r_idx],
                        np.log1p(m_deg[m_idx].numpy().reshape(-1, 1)),
                        np.log1p(r_deg[r_idx].numpy().reshape(-1, 1)),
                    ]
                )
            return np.hstack(
                [
                    med_feat[m_idx].detach().cpu().numpy(),
                    reac_feat[r_idx].detach().cpu().numpy(),
                    np.log1p(m_deg[m_idx].numpy().reshape(-1, 1)),
                    np.log1p(r_deg[r_idx].numpy().reshape(-1, 1)),
                ]
            )

        tr_idx, tr_y = _build_ml_edges(sup_ei, 1.0, num_m, num_r, all_pos_set, unique_neg=train_neg_unique)
        val_idx, val_y = _build_ml_edges(val_ei, 1.0, num_m, num_r, all_pos_set, unique_neg=val_neg_unique)
        te_idx, te_y = _build_ml_edges(test_ei, 1.0, num_m, num_r, all_pos_set, unique_neg=test_neg_unique)

        if model_name in ["RF", "XGBoost"]:
            X_tr, Y_tr = extract_ml_features(tr_idx), tr_y.numpy()
            if model_name == "RF":
                clf = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    n_jobs=-1,
                    random_state=seed,
                )
                clf.fit(X_tr, Y_tr)
                all_tree_preds = np.array([tree.predict_proba(X_tr)[:, 1] for tree in clf.estimators_])
                raw_losses = [
                    log_loss(Y_tr, p)
                    for p in (
                        np.cumsum(all_tree_preds, axis=0)
                        / np.arange(1, len(clf.estimators_) + 1)[:, None]
                    )
                ]
            else:
                X_val, Y_val = extract_ml_features(val_idx), val_y.numpy()
                xgb_kw = dict(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    tree_method="hist",
                    device="cpu",
                    random_state=seed,
                )
                if patience > 0:
                    xgb_kw["early_stopping_rounds"] = int(patience)
                clf = XGBClassifier(**xgb_kw)
                clf.fit(X_tr, Y_tr, eval_set=[(X_val, Y_val)], verbose=False)
                raw_losses = clf.evals_result()["validation_0"]["logloss"]

            epoch_losses = [raw_losses[i] for i in np.linspace(0, len(raw_losses) - 1, 50).astype(int)]
            y_prob_te = clf.predict_proba(extract_ml_features(te_idx))[:, 1]

            tr_auc, _, _, _, _, _, _ = calculate_metrics(Y_tr, clf.predict_proba(X_tr)[:, 1])
            v_auc, _, _, _, _, _, _ = calculate_metrics(val_y, clf.predict_proba(extract_ml_features(val_idx))[:, 1])
            te_auc, te_pr, te_acc, te_f1, te_mcc, te_precision, te_recall = calculate_metrics(te_y, y_prob_te)

            fold_metrics.append(
                {
                    "tr_auc": tr_auc,
                    "v_auc": v_auc,
                    "te_auc": te_auc,
                    "te_pr": te_pr,
                    "te_acc": te_acc,
                    "te_f1": te_f1,
                    "te_mcc": te_mcc,
                    "te_precision": te_precision,
                    "te_recall": te_recall,
                }
            )
            all_folds_loss.append(epoch_losses)
            continue

        tr_loader = LinkNeighborLoader(
            train_graph,
            edge_label_index=(target_etype, tr_idx),
            edge_label=tr_y,
            num_neighbors=[15, 10],
            batch_size=512,
            shuffle=True,
            num_workers=NUM_WORKERS,
        )
        val_loader = LinkNeighborLoader(
            train_graph,
            edge_label_index=(target_etype, val_idx),
            edge_label=val_y,
            num_neighbors=[15, 10],
            batch_size=512,
            num_workers=NUM_WORKERS,
        )
        te_loader = LinkNeighborLoader(
            train_graph,
            edge_label_index=(target_etype, te_idx),
            edge_label=te_y,
            num_neighbors=[15, 10],
            batch_size=512,
            num_workers=NUM_WORKERS,
        )

        model = UniversalBaselineAllEdge(
            model_name,
            hidden_channels,
            train_graph.metadata(),
            train_graph,
            med_feat,
            reac_feat,
            target_feat,
            herb_feat,
            num_conv_layers=num_conv_layers,
            transformer_num_layers=transformer_num_layers,
            transformer_nhead=transformer_nhead,
        ).to(DEVICE)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_v_auc = -1.0
        best_res = {}
        this_fold_loss: List[float] = []
        early_stop_counter = 0

        for epoch in range(1, max_epochs + 1):
            model.train()
            total_loss = 0.0
            for batch in tr_loader:
                batch = batch.to(DEVICE)
                optimizer.zero_grad()
                logits = model(batch)
                loss = F.binary_cross_entropy_with_logits(logits, batch[target_etype].edge_label.float())
                loss.backward()
                optimizer.step()
                total_loss += loss.item()

            this_fold_loss.append(total_loss / max(1, len(tr_loader)))
            if epoch == 1 or epoch % 50 == 0:
                print(
                    f"[{model_name}][seed={seed}] fold {fold + 1}/{n_splits} epoch {epoch}/{max_epochs} "
                    f"train_loss={this_fold_loss[-1]:.4f}, best_v_auc={best_v_auc:.4f}"
                )

            if epoch % eval_every == 0:
                v_auc, _, _, _, _, _, _ = evaluate_model(model, val_loader, target_etype=target_etype)
                if v_auc > best_v_auc:
                    best_v_auc = v_auc
                    early_stop_counter = 0
                    te_auc, te_pr, te_acc, te_f1, te_mcc, te_precision, te_recall = evaluate_model(
                        model, te_loader, target_etype=target_etype
                    )
                    best_res = {
                        "tr_auc": evaluate_model(model, tr_loader, target_etype=target_etype)[0],
                        "v_auc": v_auc,
                        "te_auc": te_auc,
                        "te_pr": te_pr,
                        "te_acc": te_acc,
                        "te_f1": te_f1,
                        "te_mcc": te_mcc,
                        "te_precision": te_precision,
                        "te_recall": te_recall,
                    }
                else:
                    early_stop_counter += 1

                if early_stop_counter >= patience:
                    break

        if not best_res:
            te_auc, te_pr, te_acc, te_f1, te_mcc, te_precision, te_recall = evaluate_model(
                model, te_loader, target_etype=target_etype
            )
            best_res = {
                "tr_auc": evaluate_model(model, tr_loader, target_etype=target_etype)[0],
                "v_auc": best_v_auc,
                "te_auc": te_auc,
                "te_pr": te_pr,
                "te_acc": te_acc,
                "te_f1": te_f1,
                "te_mcc": te_mcc,
                "te_precision": te_precision,
                "te_recall": te_recall,
            }

        fold_metrics.append(best_res)
        all_folds_loss.append(this_fold_loss)

    max_epochs_seen = max(len(fl) for fl in all_folds_loss) if all_folds_loss else 0
    avg_loss_curve: List[float] = []
    for i in range(max_epochs_seen):
        active = [fl[i] for fl in all_folds_loss if i < len(fl)]
        avg_loss_curve.append(sum(active) / max(1, len(active)))

    return pd.DataFrame(fold_metrics), avg_loss_curve


def run_once_per_seed(
    model_name: str,
    data_full,
    med_feat,
    reac_feat,
    target_feat,
    herb_feat,
    lr: float,
    weight_decay: float,
    hidden_channels: int,
    num_conv_layers: int,
    transformer_num_layers: int,
    transformer_nhead: int,
    seeds: List[int],
    max_epochs: int,
    patience: int,
    eval_every: int,
    n_splits: int,
    out_dir: str,
    train_neg_unique: bool,
    val_neg_unique: bool,
    test_neg_unique: bool,
) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    seed_dfs: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    for s in seeds:
        seed_everything(s)
        print(
            f"[RUN] {model_name} use_dosage=True seed={s} lr={lr} wd={weight_decay} "
            f"max_epochs={max_epochs} patience={patience} eval_every={eval_every} n_splits={n_splits} "
            f"train_unique={train_neg_unique} val_unique={val_neg_unique} test_unique={test_neg_unique}"
        )
        df, loss_curve = run_baseline_cycle_unique_eval_neg(
            model_name=model_name,
            data_full=data_full,
            med_feat=med_feat,
            reac_feat=reac_feat,
            target_feat=target_feat,
            herb_feat=herb_feat,
            seed=s,
            lr=lr,
            weight_decay=weight_decay,
            hidden_channels=hidden_channels,
            num_conv_layers=num_conv_layers,
            transformer_num_layers=transformer_num_layers,
            transformer_nhead=transformer_nhead,
            max_epochs=max_epochs,
            patience=patience,
            eval_every=eval_every,
            n_splits=n_splits,
            train_neg_unique=train_neg_unique,
            val_neg_unique=val_neg_unique,
            test_neg_unique=test_neg_unique,
        )
        df = df.copy()
        df["seed"] = s
        seed_dfs.append(df)

        per_seed_csv = os.path.join(out_dir, f"{model_name}_seed{s}_folds.csv")
        df.to_csv(per_seed_csv, index=False, encoding="utf-8-sig")

        summary_rows.append(
            {
                "seed": s,
                "tr_auc": float(df["tr_auc"].mean()),
                "v_auc": float(df["v_auc"].mean()),
                "te_auc": float(df["te_auc"].mean()),
                "te_pr": float(df["te_pr"].mean()),
                "te_precision": float(df["te_precision"].mean()),
                "te_recall": float(df["te_recall"].mean()),
                "te_acc": float(df["te_acc"].mean()),
                "te_f1": float(df["te_f1"].mean()),
                "te_mcc": float(df["te_mcc"].mean()),
                "max_loss_points": int(len(loss_curve)),
            }
        )

    summary_csv = os.path.join(out_dir, f"{model_name}_seed_summary.csv")
    pd.DataFrame(summary_rows).to_csv(summary_csv, index=False, encoding="utf-8-sig")

    def _ms(col: str) -> Tuple[float, float]:
        return mean_std_across_folds(seed_dfs, col)

    tr_m, tr_s = _ms("tr_auc")
    v_m, v_s = _ms("v_auc")
    te_auc_m, te_auc_s = _ms("te_auc")
    te_pr_m, te_pr_s = _ms("te_pr")
    te_p_m, te_p_s = _ms("te_precision")
    te_r_m, te_r_s = _ms("te_recall")
    te_acc_m, te_acc_s = _ms("te_acc")
    te_f1_m, te_f1_s = _ms("te_f1")
    te_mcc_m, te_mcc_s = _ms("te_mcc")

    agg = {
        "model": model_name,
        "use_dosage": True,
        "lr": lr,
        "weight_decay": weight_decay,
        "hidden_channels": hidden_channels,
        "num_conv_layers": num_conv_layers,
        "transformer_num_layers": transformer_num_layers,
        "transformer_nhead": transformer_nhead,
        "seeds": seeds,
        "max_epochs": max_epochs,
        "patience": patience,
        "eval_every": eval_every,
        "n_splits": n_splits,
        "train_neg_unique": train_neg_unique,
        "val_neg_unique": val_neg_unique,
        "test_neg_unique": test_neg_unique,
        "tr_auc_mean": tr_m,
        "tr_auc_std": tr_s,
        "v_auc_mean": v_m,
        "v_auc_std": v_s,
        "te_auc_mean": te_auc_m,
        "te_auc_std": te_auc_s,
        "te_pr_mean": te_pr_m,
        "te_pr_std": te_pr_s,
        "te_precision_mean": te_p_m,
        "te_precision_std": te_p_s,
        "te_recall_mean": te_r_m,
        "te_recall_std": te_r_s,
        "te_acc_mean": te_acc_m,
        "te_acc_std": te_acc_s,
        "te_f1_mean": te_f1_m,
        "te_f1_std": te_f1_s,
        "te_mcc_mean": te_mcc_m,
        "te_mcc_std": te_mcc_s,
        "tr_auc_meanpm_std": format_mean_pm_std(tr_m, tr_s),
        "v_auc_meanpm_std": format_mean_pm_std(v_m, v_s),
        "te_auc_meanpm_std": format_mean_pm_std(te_auc_m, te_auc_s),
        "te_pr_meanpm_std": format_mean_pm_std(te_pr_m, te_pr_s),
        "te_precision_meanpm_std": format_mean_pm_std(te_p_m, te_p_s),
        "te_recall_meanpm_std": format_mean_pm_std(te_r_m, te_r_s),
        "te_acc_meanpm_std": format_mean_pm_std(te_acc_m, te_acc_s),
        "te_f1_meanpm_std": format_mean_pm_std(te_f1_m, te_f1_s),
        "te_mcc_meanpm_std": format_mean_pm_std(te_mcc_m, te_mcc_s),
    }
    return seed_dfs, agg


def main() -> None:
    p = argparse.ArgumentParser(description="run_baseline_cycle + dosage + best_hparams JSON (unique eval negatives)")
    p.add_argument(
        "--best_json",
        type=str,
        default=os.path.join(PROJECT_ROOT, "configs", "best_hparams_all_models.json"),
        help="Path to best_hparams_all_models.json",
    )
    p.add_argument("--dosage_mode", type=str, default="log_share", help="Dosage preprocessing mode for load_tcm_full_data.")
    p.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model types (e.g. RGCN). Empty = all keys in JSON.",
    )
    p.add_argument(
        "--seeds",
        type=str,
        default="",
        help="Override seeds, e.g. 42 or 42,43,44. Empty = use eval_seeds from JSON per model.",
    )
    p.add_argument("--max_epochs", type=int, default=0, help="0 = use JSON max_epochs_eval.")
    p.add_argument("--patience", type=int, default=0, help="0 = use JSON patience.")
    p.add_argument("--eval_every", type=int, default=0, help="0 = use JSON eval_every.")
    p.add_argument("--n_splits", type=int, default=5, help="K-fold splits.")
    p.add_argument(
        "--train_neg_unique",
        action="store_true",
        help="Use unique negatives for train split as well (default: False).",
    )
    p.add_argument(
        "--allow_dup_val_neg",
        action="store_true",
        help="Allow duplicated negatives on validation split (default: unique).",
    )
    p.add_argument(
        "--allow_dup_test_neg",
        action="store_true",
        help="Allow duplicated negatives on test split (default: unique).",
    )
    p.add_argument("--out_dir", type=str, default="", help="Output directory.")
    p.add_argument(
        "--out_csv",
        type=str,
        default="",
        help="Alias for --out_dir (kept for compatibility).",
    )
    args = p.parse_args()

    if args.n_splits < 2:
        raise SystemExit("--n_splits must be >= 2 (sklearn KFold).")

    if args.out_dir.strip() and args.out_csv.strip():
        raise SystemExit("Use only one of --out_dir or --out_csv (they set the same output directory).")

    best_path = args.best_json
    if not os.path.isabs(best_path):
        best_path = os.path.abspath(best_path)
    best_all = load_best_hparams(best_path)

    baselines_order = [
        "LR",
        "MLP",
        "XGBoost",
        "RF",
        "GCN",
        "SAGE",
        "GIN",
        "RGCN",
        "HGT",
        "RGAT",
        "GraphEdgeDosage",
        "Transformer",
    ]

    parsed_models = _parse_models(args.models)
    if parsed_models:
        model_types = parsed_models
    else:
        model_types = sorted(
            best_all.keys(),
            key=lambda m: (baselines_order.index(m) if m in baselines_order else 999, m),
        )

    seeds_override: List[int] | None = None
    if args.seeds.strip():
        seeds_override = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]

    out_dir_arg = (args.out_dir or args.out_csv or "").strip()
    if out_dir_arg:
        out_dir = os.path.abspath(out_dir_arg)
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(PROJECT_ROOT, "baselines", "grid_search_results", f"baseline_cycle_dosage_unique_eval_neg_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    train_neg_unique = bool(args.train_neg_unique)
    val_neg_unique = not bool(args.allow_dup_val_neg)
    test_neg_unique = not bool(args.allow_dup_test_neg)

    print(f"[INFO] Loading data: use_dosage=True, dosage_mode={args.dosage_mode}")
    print(
        f"[INFO] Negative sampling policy: train_unique={train_neg_unique} "
        f"val_unique={val_neg_unique} test_unique={test_neg_unique}"
    )
    data_full, _ = load_tcm_full_data(use_dosage=True, dosage_mode=args.dosage_mode)
    med_feat, reac_feat, target_feat, herb_feat = prepare_unified_features(data_full)

    run_results: List[Dict[str, Any]] = []

    for model_name in model_types:
        if model_name not in best_all:
            raise KeyError(f"Model '{model_name}' not in {best_path}")

        (
            lr,
            weight_decay,
            hidden_channels,
            num_conv_layers,
            transformer_num_layers,
            transformer_nhead,
            seeds_json,
            max_epochs,
            patience,
            eval_every,
        ) = build_model_args_from_best(model_name, best_all)

        seeds = seeds_override if seeds_override is not None else seeds_json
        if not seeds:
            raise SystemExit(f"{model_name}: empty seed list (check JSON eval_seeds or --seeds).")

        if args.max_epochs > 0:
            max_epochs = int(args.max_epochs)
        if args.patience > 0:
            patience = int(args.patience)
        if args.eval_every > 0:
            eval_every = int(args.eval_every)
        if eval_every < 1:
            raise SystemExit(f"{model_name}: eval_every must be >= 1 (got {eval_every}).")
        if max_epochs < 1:
            raise SystemExit(f"{model_name}: max_epochs must be >= 1 (got {max_epochs}).")

        model_out = os.path.join(out_dir, model_name)
        os.makedirs(model_out, exist_ok=True)

        _, agg = run_once_per_seed(
            model_name=model_name,
            data_full=data_full,
            med_feat=med_feat,
            reac_feat=reac_feat,
            target_feat=target_feat,
            herb_feat=herb_feat,
            lr=lr,
            weight_decay=weight_decay,
            hidden_channels=hidden_channels,
            num_conv_layers=num_conv_layers,
            transformer_num_layers=transformer_num_layers,
            transformer_nhead=transformer_nhead,
            seeds=seeds,
            max_epochs=max_epochs,
            patience=patience,
            eval_every=eval_every,
            n_splits=args.n_splits,
            out_dir=model_out,
            train_neg_unique=train_neg_unique,
            val_neg_unique=val_neg_unique,
            test_neg_unique=test_neg_unique,
        )
        run_results.append(agg)

    meta = {
        "best_json": best_path,
        "dosage_mode": args.dosage_mode,
        "models": model_types,
        "n_splits": args.n_splits,
        "seeds_cli_override": seeds_override,
        "max_epochs_override": args.max_epochs or None,
        "patience_override": args.patience or None,
        "eval_every_override": args.eval_every or None,
        "negative_sampling": {
            "train_neg_unique": train_neg_unique,
            "val_neg_unique": val_neg_unique,
            "test_neg_unique": test_neg_unique,
        },
        "out_dir": out_dir,
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    results_csv = os.path.join(out_dir, "all_models_summary.csv")
    pd.DataFrame(run_results).to_csv(results_csv, index=False, encoding="utf-8-sig")

    ordered = [m for m in baselines_order if m in model_types]

    print("\n" + "=" * 12 + " Dosage + run_baseline_cycle (best JSON, unique eval neg) " + "=" * 12)
    header = (
        f"{'Model':<12}"
        f"| {'Tr-AUC':<15} "
        f"| {'Val-AUC':<15} "
        f"| {'Te-AUC':<15} "
        f"| {'Te-PR':<15} "
        f"| {'Te-Prec':<15} "
        f"| {'Te-Rec':<15} "
        f"| {'Te-ACC':<15} "
        f"| {'Te-F1':<15} "
        f"| {'Te-MCC':<15}"
    )
    print(header)
    print("-" * len(header))
    agg_by_model = {r["model"]: r for r in run_results}
    for m in ordered:
        print(format_report_row(m, agg_by_model[m]))

    print(f"\n[DONE] Results saved to: {out_dir}")


if __name__ == "__main__":
    main()

