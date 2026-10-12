import argparse
import copy
import json
import os
import sys
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from torch_geometric.loader import LinkNeighborLoader
from xgboost import XGBClassifier

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Default: prefer GPU when available.
# You can override with GNN02_DEVICE=cpu if needed.
os.environ.setdefault("GNN02_DEVICE", "cuda")

from baselines.common.data import DEVICE, TARGET_ETYPE, load_tcm_full_data, prepare_unified_features, seed_everything
from baselines.common.metrics import calculate_metrics, evaluate_model
from baselines.common.models import UniversalBaselineAllEdge
from baselines.common.train import get_ml_data_safe


def run_one_model(
    model_type: str,
    dosage_mode: str,
    seed: int,
    n_splits: int,
    max_epochs: int,
    eval_every: int = 5,
    patience: int = 20,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    hidden_channels: int = 64,
    num_conv_layers: int = 2,
    transformer_num_layers: int = 2,
    transformer_nhead: int = 4,
) -> pd.DataFrame:
    seed_everything(seed)
    use_dosage = dosage_mode != "none"
    is_tree = model_type in {"RF", "XGBoost"}

    raw_data, _ = load_tcm_full_data(use_dosage=use_dosage, dosage_mode=dosage_mode)
    med_feat, reac_feat, target_feat, herb_feat = prepare_unified_features(raw_data)
    # Tree baselines need BERT-based node embeddings + dosage-weighted aggregation.
    # We precompute those aggregates once per (model_type, dosage_mode) run.
    if is_tree:
        num_m_nodes = int(raw_data["medicine"].num_nodes)
        num_r_nodes = int(raw_data["reaction"].num_nodes)

        herb_x = raw_data["herb"].x.detach().cpu()  # (H, dim)
        reac_bert = raw_data["reaction"].x.detach().cpu()  # (R, dim)
        target_x = raw_data["target"].x.detach().cpu()  # (T, dim)

        mh_ei = raw_data["medicine", "contains", "herb"].edge_index.detach().cpu()
        mh_dose = raw_data["medicine", "contains", "herb"].edge_weight.detach().cpu().float()

        h_dim = int(herb_x.size(1))
        med_herb_bert_agg = torch.zeros((num_m_nodes, h_dim), dtype=torch.float32)
        wsum = torch.zeros((num_m_nodes,), dtype=torch.float32)
        # Accumulate dosage-weighted herb embeddings into medicine space.
        med_herb_bert_agg.index_add_(
            0, mh_ei[0], herb_x[mh_ei[1]] * mh_dose.view(-1, 1)
        )
        wsum.index_add_(0, mh_ei[0], mh_dose)
        med_herb_bert_agg = med_herb_bert_agg / (wsum.view(-1, 1) + 1e-6)

        tr_ei = raw_data["target", "causes", "reaction"].edge_index.detach().cpu()
        t_dim = int(target_x.size(1))
        r_acc = torch.zeros((num_r_nodes, t_dim), dtype=torch.float32)
        r_cnt = torch.zeros((num_r_nodes,), dtype=torch.float32)
        r_acc.index_add_(0, tr_ei[1], target_x[tr_ei[0]])
        r_cnt.index_add_(0, tr_ei[1], torch.ones(tr_ei.size(1)))
        reac_target_bert_agg = torch.zeros((num_r_nodes, t_dim), dtype=torch.float32)
        mask = r_cnt > 0
        reac_target_bert_agg[mask] = r_acc[mask] / r_cnt[mask].view(-1, 1)

        # Capture numpy arrays for fast feature extraction.
        med_feat_np = med_feat.detach().cpu().numpy().astype(np.float32)
        reac_feat_np = reac_feat.detach().cpu().numpy().astype(np.float32)
        reac_bert_np = reac_bert.numpy().astype(np.float32)
        med_herb_bert_agg_np = med_herb_bert_agg.numpy().astype(np.float32)
        reac_target_bert_agg_np = reac_target_bert_agg.numpy().astype(np.float32)

    target_etype = TARGET_ETYPE
    pos_edges = raw_data[target_etype].edge_index
    all_pos_set = set(zip(pos_edges[0].tolist(), pos_edges[1].tolist()))

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    fold_metrics = []

    print(f"[RUN] model={model_type} dosage_mode={dosage_mode} seed={seed} folds={n_splits} epochs={max_epochs}")
    for fold, (tv_idx, test_idx) in enumerate(kf.split(np.arange(pos_edges.size(1))), start=1):
        test_ei, tv_ei = pos_edges[:, test_idx], pos_edges[:, tv_idx]
        perm = np.random.permutation(tv_ei.size(1))
        tv_ei = tv_ei[:, perm]
        val_size = max(1, int(tv_ei.size(1) * 0.1))
        val_ei, sup_ei = tv_ei[:, :val_size], tv_ei[:, val_size:]

        tr_idx, tr_y = get_ml_data_safe(
            sup_ei, 1, raw_data["medicine"].num_nodes, raw_data["reaction"].num_nodes, all_pos_set
        )
        val_idx, val_y = get_ml_data_safe(
            val_ei, 1, raw_data["medicine"].num_nodes, raw_data["reaction"].num_nodes, all_pos_set
        )
        te_idx, te_y = get_ml_data_safe(
            test_ei, 1, raw_data["medicine"].num_nodes, raw_data["reaction"].num_nodes, all_pos_set
        )

        if is_tree:
            # Degrees for (medicine, reaction) pair feature scaling.
            m_deg = torch.zeros(raw_data["medicine"].num_nodes, dtype=torch.float32).index_add_(
                0, sup_ei[0], torch.ones(sup_ei.size(1))
            )
            r_deg = torch.zeros(raw_data["reaction"].num_nodes, dtype=torch.float32).index_add_(
                0, sup_ei[1], torch.ones(sup_ei.size(1))
            )
            m_deg_np = m_deg.numpy().astype(np.float32)
            r_deg_np = r_deg.numpy().astype(np.float32)

            def extract_ml_features(ei: torch.Tensor) -> np.ndarray:
                m_idx = ei[0].detach().cpu().numpy().astype(np.int64)
                r_idx = ei[1].detach().cpu().numpy().astype(np.int64)
                return np.hstack(
                    [
                        med_feat_np[m_idx],
                        med_herb_bert_agg_np[m_idx],
                        reac_feat_np[r_idx],
                        reac_bert_np[r_idx],
                        reac_target_bert_agg_np[r_idx],
                        np.log1p(m_deg_np[m_idx].reshape(-1, 1)),
                        np.log1p(r_deg_np[r_idx].reshape(-1, 1)),
                    ]
                ).astype(np.float32)

            X_tr = extract_ml_features(tr_idx)
            Y_tr = tr_y.detach().cpu().numpy()
            X_val = extract_ml_features(val_idx)
            Y_val = val_y.detach().cpu().numpy()
            X_te = extract_ml_features(te_idx)
            Y_te = te_y.detach().cpu().numpy()

            if model_type == "RF":
                clf = RandomForestClassifier(
                    n_estimators=300,
                    max_depth=12,
                    n_jobs=-1,
                    random_state=seed,
                )
                clf.fit(X_tr, Y_tr)
            else:
                xgb_kw = dict(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    tree_method="hist",
                    device="cpu",
                    random_state=seed,
                )
                if patience and patience > 0:
                    # XGBoost early stopping to match common/train.py behavior.
                    xgb_kw["early_stopping_rounds"] = int(patience)
                clf = XGBClassifier(**xgb_kw)
                clf.fit(X_tr, Y_tr, eval_set=[(X_val, Y_val)], verbose=False)

            tr_auc, *_ = calculate_metrics(Y_tr, clf.predict_proba(X_tr)[:, 1])
            v_auc, *_ = calculate_metrics(Y_val, clf.predict_proba(X_val)[:, 1])
            te_auc, te_pr, te_acc, te_f1, te_mcc, te_precision, te_recall = calculate_metrics(
                Y_te, clf.predict_proba(X_te)[:, 1]
            )

            fold_metrics.append(
                {
                    "model": model_type,
                    "dosage_mode": dosage_mode,
                    "fold": fold,
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
            continue

        # GNN / Transformer branch (existing behavior)
        train_graph = copy.deepcopy(raw_data)
        del train_graph[target_etype]
        train_graph[target_etype].edge_index = sup_ei
        train_graph = T.ToUndirected()(train_graph)

        tr_loader = LinkNeighborLoader(
            train_graph,
            edge_label_index=(target_etype, tr_idx),
            edge_label=tr_y,
            num_neighbors=[15, 10],
            batch_size=512,
            shuffle=True,
            num_workers=0,
        )
        val_loader = LinkNeighborLoader(
            train_graph,
            edge_label_index=(target_etype, val_idx),
            edge_label=val_y,
            num_neighbors=[15, 10],
            batch_size=512,
            num_workers=0,
        )
        te_loader = LinkNeighborLoader(
            train_graph,
            edge_label_index=(target_etype, te_idx),
            edge_label=te_y,
            num_neighbors=[15, 10],
            batch_size=512,
            num_workers=0,
        )

        model = UniversalBaselineAllEdge(
            model_type=model_type,
            hidden_channels=hidden_channels,
            metadata=train_graph.metadata(),
            data_obj=train_graph,
            init_med_x=med_feat,
            init_reac_x=reac_feat,
            init_target_x=target_feat,
            init_herb_x=herb_feat,
            num_conv_layers=num_conv_layers,
            transformer_num_layers=transformer_num_layers,
            transformer_nhead=transformer_nhead,
        ).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_v_auc = -1.0
        best_res = None
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
                total_loss += float(loss.item())

            if epoch == 1 or epoch % 50 == 0:
                avg_loss = total_loss / max(1, len(tr_loader))
                print(f"[{model_type}|{dosage_mode}] fold {fold}/{n_splits} epoch {epoch}/{max_epochs} loss={avg_loss:.4f}")

            if epoch % eval_every == 0:
                v_auc, *_ = evaluate_model(model, val_loader, target_etype=target_etype)
                if v_auc > best_v_auc:
                    best_v_auc = v_auc
                    te_auc, te_pr, te_acc, te_f1, te_mcc, te_precision, te_recall = evaluate_model(
                        model, te_loader, target_etype=target_etype
                    )
                    best_res = {
                        "model": model_type,
                        "dosage_mode": dosage_mode,
                        "fold": fold,
                        "v_auc": v_auc,
                        "te_auc": te_auc,
                        "te_pr": te_pr,
                        "te_acc": te_acc,
                        "te_f1": te_f1,
                        "te_mcc": te_mcc,
                        "te_precision": te_precision,
                        "te_recall": te_recall,
                    }

        if best_res is None:
            te_auc, te_pr, te_acc, te_f1, te_mcc, te_precision, te_recall = evaluate_model(
                model, te_loader, target_etype=target_etype
            )
            best_res = {
                "model": model_type,
                "dosage_mode": dosage_mode,
                "fold": fold,
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

    return pd.DataFrame(fold_metrics)


def _parse_csv_list(text: str) -> List[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _extract_arch_params(best_arch: Dict[str, object], model_type: str) -> Tuple[int, int, int, int]:
    hidden_channels = int(best_arch.get("hidden_channels", 64))
    if model_type in {"GraphEdgeDosage", "GIN", "RGAT", "GCN", "SAGE", "RGCN", "HGT"}:
        num_conv_layers = int(best_arch.get("num_conv_layers", 2))
    else:
        num_conv_layers = 2
    transformer_num_layers = int(best_arch.get("transformer_num_layers", 2))
    transformer_nhead = int(best_arch.get("transformer_nhead", 4))
    return hidden_channels, num_conv_layers, transformer_num_layers, transformer_nhead


def _resolve_hparams(
    hp_db: Optional[Dict[str, object]],
    model_type: str,
    dosage_mode: str,
    ignore_json_dosage_mode: bool = False,
) -> Optional[Dict[str, float]]:
    """
    Try multiple key layouts for hparam JSON:
    1) hp_db[dosage_mode][model_type]
    2) hp_db[f"{model_type}__{dosage_mode}"] or hp_db[f"{model_type}|{dosage_mode}"]
    3) hp_db[model_type] with optional dosage_mode field match
    """
    if not hp_db:
        return None

    entry = None
    if dosage_mode in hp_db and isinstance(hp_db[dosage_mode], dict):
        entry = hp_db[dosage_mode].get(model_type)
    if entry is None:
        entry = hp_db.get(f"{model_type}__{dosage_mode}") or hp_db.get(f"{model_type}|{dosage_mode}")
    if entry is None and model_type in hp_db and isinstance(hp_db[model_type], dict):
        maybe = hp_db[model_type]
        if ignore_json_dosage_mode:
            entry = maybe
        elif "dosage_mode" not in maybe or str(maybe.get("dosage_mode")).lower() == dosage_mode:
            entry = maybe
    if entry is None:
        return None

    final_best = entry.get("final_best", entry)
    best_arch = final_best.get("best_arch", {})
    hc, nconv, tnl, tnh = _extract_arch_params(best_arch, model_type)
    return {
        "lr": float(final_best.get("lr", 1e-3)),
        "weight_decay": float(final_best.get("weight_decay", 1e-4)),
        "hidden_channels": hc,
        "num_conv_layers": nconv,
        "transformer_num_layers": tnl,
        "transformer_nhead": tnh,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="GraphEdgeDosage,GIN,RGAT")
    parser.add_argument("--dosage_modes", default="none,raw_log,log_share,linear_share")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=500)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_conv_layers", type=int, default=2)
    parser.add_argument("--transformer_num_layers", type=int, default=2)
    parser.add_argument("--transformer_nhead", type=int, default=4)
    parser.add_argument(
        "--best_hparams_json",
        default="",
        help="Optional JSON path containing tuned hparams per model×dosage_mode.",
    )
    parser.add_argument(
        "--ignore_json_dosage_mode",
        action="store_true",
        help="Use hp_db[model] for every dosage_mode even if JSON entry's dosage_mode differs "
        "(e.g. one grid run under log_share: same lr/arch for none vs log_share ablation).",
    )
    parser.add_argument("--out_csv", default=os.path.join("baselines", "grid_search_results", "dosage_ablation_results.csv"))
    args = parser.parse_args()

    models = _parse_csv_list(args.models)
    modes = _parse_csv_list(args.dosage_modes)
    hp_db = None
    if args.best_hparams_json:
        with open(args.best_hparams_json, "r", encoding="utf-8") as f:
            hp_db = json.load(f)
        print(f"[INFO] Loaded tuned hparams: {args.best_hparams_json}")

    all_df = []
    for model in models:
        for mode in modes:
            hp = _resolve_hparams(
                hp_db, model, mode, ignore_json_dosage_mode=args.ignore_json_dosage_mode
            )
            if hp is None:
                hp = {
                    "lr": args.lr,
                    "weight_decay": args.weight_decay,
                    "hidden_channels": args.hidden_channels,
                    "num_conv_layers": args.num_conv_layers,
                    "transformer_num_layers": args.transformer_num_layers,
                    "transformer_nhead": args.transformer_nhead,
                }
                print(f"[HP] model={model} mode={mode} using CLI/default hparams: {hp}")
            else:
                print(f"[HP] model={model} mode={mode} using tuned hparams: {hp}")

            df = run_one_model(
                model_type=model,
                dosage_mode=mode,
                seed=args.seed,
                n_splits=args.folds,
                max_epochs=args.epochs,
                patience=args.patience,
                lr=float(hp["lr"]),
                weight_decay=float(hp["weight_decay"]),
                hidden_channels=int(hp["hidden_channels"]),
                num_conv_layers=int(hp["num_conv_layers"]),
                transformer_num_layers=int(hp.get("transformer_num_layers", args.transformer_num_layers)),
                transformer_nhead=int(hp.get("transformer_nhead", args.transformer_nhead)),
            )
            all_df.append(df)

            mean_te = float(df["te_auc"].mean())
            std_te = float(df["te_auc"].std(ddof=1)) if len(df) > 1 else 0.0
            print(f"[SUMMARY] model={model} mode={mode} te_auc={mean_te:.4f}+-{std_te:.4f}")

    final_df = pd.concat(all_df, ignore_index=True)
    out_csv = args.out_csv
    out_dir = os.path.dirname(out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    final_df.to_csv(out_csv, index=False, encoding="utf-8-sig")
    print(f"[DONE] saved {out_csv}")


if __name__ == "__main__":
    main()

