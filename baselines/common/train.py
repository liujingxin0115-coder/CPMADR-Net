import copy
import random
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.metrics import log_loss
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.loader import LinkNeighborLoader
from xgboost import XGBClassifier

from .data import DEVICE, NUM_WORKERS, TARGET_ETYPE, seed_everything
from .metrics import calculate_metrics, evaluate_model
from .models import UniversalBaselineAllEdge


def get_ml_data_safe(ei: torch.Tensor, ratio: float, num_m: int, num_r: int, pos_set) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Negative sampling for link prediction:
    - Keep all provided positive edges in `ei`
    - Sample `num_neg = int(num_pos * ratio)` negative edges not present in pos_set
    """
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


def run_baseline_cycle(
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
) -> Tuple[pd.DataFrame, List[float]]:
    """
    Train one baseline model_type and evaluate with:
    - K-fold split (default 5)
    - Each fold splits tv into val/sup with 9:1 after shuffling
    - Early stopping based on val AUC
    """
    seed_everything(seed)

    target_etype = TARGET_ETYPE
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    pos_edges = data_full[target_etype].edge_index
    all_pos_set = set(zip(pos_edges[0].tolist(), pos_edges[1].tolist()))

    fold_metrics: List[dict] = []
    all_folds_loss: List[List[float]] = []

    # RF / XGBoost: use the same raw signals as UniversalBaselineAllEdge node inputs (before GNN):
    # - medicine: med_feat + dosage-weighted mean of herb BiomedBERT (herb BERT only enters via herb nodes in GNN)
    # - reaction: reac_feat (mean of target_feat on target->reaction) + reaction BiomedBERT
    #             + mean of target BiomedBERT over the same target->reaction neighbors (matches target node BERT in GNN)
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

        # Same aggregation scope as reac_feat in prepare_unified_features: target --causes--> reaction
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

    for fold, (tv_idx, test_idx) in enumerate(kf.split(np.arange(pos_edges.size(1)))):
        test_ei, tv_ei = pos_edges[:, test_idx], pos_edges[:, tv_idx]

        # Shuffle within fold to make val/sup splitting reproducible
        perm = np.random.permutation(tv_ei.size(1))
        tv_ei = tv_ei[:, perm]
        val_size = max(1, int(tv_ei.size(1) * 0.1))
        val_ei, sup_ei = tv_ei[:, :val_size], tv_ei[:, val_size:]

        train_graph = copy.deepcopy(data_full)
        del train_graph[target_etype]
        train_graph[target_etype].edge_index = sup_ei
        train_graph = T.ToUndirected()(train_graph)

        # Degree used for LR/ML baselines' ML feature construction (kept for parity).
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
            # unused for non-tree models, but keep a consistent signature if ever called
            return np.hstack(
                [
                    med_feat[m_idx].detach().cpu().numpy(),
                    reac_feat[r_idx].detach().cpu().numpy(),
                    np.log1p(m_deg[m_idx].numpy().reshape(-1, 1)),
                    np.log1p(r_deg[r_idx].numpy().reshape(-1, 1)),
                ]
            )

        tr_idx, tr_y = get_ml_data_safe(
            sup_ei, 1, data_full["medicine"].num_nodes, data_full["reaction"].num_nodes, all_pos_set
        )
        val_idx, val_y = get_ml_data_safe(
            val_ei, 1, data_full["medicine"].num_nodes, data_full["reaction"].num_nodes, all_pos_set
        )
        te_idx, te_y = get_ml_data_safe(
            test_ei, 1, data_full["medicine"].num_nodes, data_full["reaction"].num_nodes, all_pos_set
        )

        # Tree-based baselines (XGBoost / RF) use sklearn/xgboost and ML features.
        if model_name in ["RF", "XGBoost"]:
            X_tr, Y_tr = extract_ml_features(tr_idx), tr_y.numpy()
            if model_name == "RF":
                clf = RandomForestClassifier(
                    n_estimators=300, max_depth=12, n_jobs=-1, random_state=seed
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
                # XGBoost 2.1+: early_stopping_rounds belongs on the estimator, not fit().
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

        # Safety: if best_res never updated (should be rare)
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

