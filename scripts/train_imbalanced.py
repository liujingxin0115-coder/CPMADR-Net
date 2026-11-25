"""
Compare v13 vs path-readout variants (dual protocol 13), with unique negatives
for validation/test by default.

Same as `run_rgcn_v13_vs_path_readout_dual_protocol_13_unique_eval_neg.py`, except:
  - positive : negative ratio = 1 : 10 (neg_ratio=10.0)

Key difference vs run_rgcn_v13_vs_path_readout_dual_protocol_13.py:
  - train negatives: with replacement by default (can enable unique by flag)
  - val/test negatives: without replacement by default (can disable by flags)
"""

from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch_geometric.transforms as T
from sklearn.model_selection import KFold

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

if os.environ.get("GNN02_LINK_LOADER_WORKERS", "").strip():
    os.environ["NUM_WORKERS"] = os.environ.get("GNN02_LINK_LOADER_WORKERS", "").strip()
else:
    os.environ["NUM_WORKERS"] = "0"

from baselines.common.data import (  # noqa: E402
    DEVICE,
    TARGET_ETYPE,
    load_tcm_full_data,
    prepare_unified_features,
    seed_everything,
)
from cpmadr_net.model import DoseAwareHSPESARGCNv13  # noqa: E402
from cpmadr_net.model import (  # noqa: E402
    DoseAwareHSPESARGCNv13PathReadout,
    PathReadoutMode,
)
from cpmadr_net.training import (  # noqa: E402
    compute_bce_pos_weight,
    aggregate_seed_summary,
    calc_metrics,
    collect_probs,
    find_best_threshold,
    make_loaders,
    paired_significance_tests,
    print_comparison,
    print_significance_table,
    resolve_pos_weight,
)

# num_neg = int(num_pos * NEG_POS_RATIO) -> 1 positive : NEG_POS_RATIO negatives
NEG_POS_RATIO = 10.0

VARIANT_ALIASES: Dict[str, str] = {
    "v13": "v13_full",
    "full": "v13_full",
    "path": "path_readout",
    "readout": "path_readout",
    "path_attn": "path_readout",
    "cross": "path_cross",
}

VARIANT_DESCRIPTIONS: Dict[str, str] = {
    "v13_full": "baseline v13 interaction head",
    "path_readout": "v13 encoder + learnable path ingredient attention readout",
    "path_cross": "path_readout + cross-gate on med/reac path pools",
}

PATH_SPECS: Dict[str, Tuple[PathReadoutMode, int]] = {
    "path_readout": ("attn", 32),
    "path_cross": ("attn_cross", 32),
}


def parse_args():
    p = argparse.ArgumentParser(
        description="v13 vs path-readout head (dual protocol 13, unique eval negatives, pos:neg=1:10)."
    )
    p.add_argument(
        "--variants",
        type=str,
        default="v13_full,path_readout",
        help="Comma-separated: v13_full, path_readout, path_cross",
    )
    p.add_argument("--seeds", type=str, default="42,43,44")
    p.add_argument("--dosage_mode", type=str, default="log_share")
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_channels", type=int, default=64)
    p.add_argument("--improve_layers", type=int, default=2)
    p.add_argument("--improve_dropout", type=float, default=0.25)
    p.add_argument("--improve_rel_emb_dim", type=int, default=8)
    p.add_argument("--improve_edge_drop", type=float, default=0.1)
    p.add_argument("--ffn_mult", type=float, default=2.0)
    p.add_argument("--esa_mlp_hidden", type=int, default=16)
    p.add_argument("--ffn_dropout", type=float, default=-1.0)
    p.add_argument("--head_dropout", type=float, default=-1.0)
    p.add_argument("--head_res_blocks", type=int, default=1)
    p.add_argument("--no_bimodal_film", action="store_true")
    p.add_argument("--path_max_ings", type=int, default=32, help="Cap ingredients per side per link.")
    p.add_argument("--fixed_threshold", type=float, default=0.5)
    p.add_argument("--pos_weight_mode", type=str, default="sqrt", choices=["none", "sqrt", "linear"])
    p.add_argument("--pos_weight_scale", type=float, default=1.0)
    p.add_argument("--pos_weight_max", type=float, default=10.0)
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
    p.add_argument(
        "--out_dir",
        type=str,
        default="results/RGCN_v13_vs_path_readout_dual_protocol_13_unique_eval_neg_ratio1_10",
    )
    return p.parse_args()


def _parse_variants(raw: str) -> List[str]:
    out: List[str] = []
    for token in raw.split(","):
        t = token.strip().lower()
        if not t:
            continue
        t = VARIANT_ALIASES.get(t, t)
        if t not in VARIANT_DESCRIPTIONS:
            raise ValueError(f"Unknown variant {token!r}. Choose from: {', '.join(VARIANT_DESCRIPTIONS)}")
        if t not in out:
            out.append(t)
    if not out:
        raise ValueError("No variants selected.")
    return out


def _v13_kwargs(args, ffn_drop: float, head_drop: float, use_film: bool) -> dict:
    return dict(
        hidden_channels=args.hidden_channels,
        num_conv_layers=args.improve_layers,
        dropout=args.improve_dropout,
        rel_emb_dim=args.improve_rel_emb_dim,
        edge_drop=args.improve_edge_drop,
        ffn_mult=args.ffn_mult,
        esa_mlp_hidden=args.esa_mlp_hidden,
        ffn_dropout=ffn_drop,
        use_bimodal_film=use_film,
        head_dropout=head_drop,
        head_res_blocks=args.head_res_blocks,
    )


def _make_v13_builder(kw: dict) -> Callable:
    def builder(train_graph, med_x, reac_x, targ_x, herb_x, _kw=kw):
        return DoseAwareHSPESARGCNv13(
            metadata=train_graph.metadata(),
            data_obj=train_graph,
            init_med_x=med_x,
            init_reac_x=reac_x,
            init_target_x=targ_x,
            init_herb_x=herb_x,
            **_kw,
        )

    return builder


def _make_path_builder(
    mode: PathReadoutMode,
    kw: dict,
    path_max_ings: int,
    ing_emb,
) -> Callable:
    def builder(train_graph, med_x, reac_x, targ_x, herb_x, _kw=kw, _mode=mode, _cap=path_max_ings, _ing=ing_emb):
        return DoseAwareHSPESARGCNv13PathReadout(
            metadata=train_graph.metadata(),
            data_obj=train_graph,
            init_med_x=med_x,
            init_reac_x=reac_x,
            init_target_x=targ_x,
            init_herb_x=herb_x,
            path_readout_mode=_mode,
            path_max_ings=_cap,
            ingredient_emb=_ing,
            **_kw,
        )

    return builder


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
    pos_ei: torch.Tensor,
    ratio: float,
    num_m: int,
    num_r: int,
    pos_set: set,
    unique_neg: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    num_pos = int(pos_ei.size(1))
    num_neg = int(num_pos * float(ratio))
    if num_pos <= 0:
        raise ValueError("Positive edge set is empty for current split.")

    if num_neg > 0:
        if unique_neg:
            neg_list = _sample_neg_unique(num_neg, num_m, num_r, pos_set)
        else:
            neg_list = _sample_neg_with_replacement(num_neg, num_m, num_r, pos_set)
        neg_ei = torch.tensor(neg_list, dtype=torch.long).t().contiguous()
    else:
        neg_ei = torch.empty((2, 0), dtype=torch.long)

    idx = torch.cat([pos_ei, neg_ei], dim=1)
    lab = torch.cat([torch.ones(num_pos), torch.zeros(num_neg)], dim=0)
    return idx, lab


def train_one_cycle_dual_unique_eval_neg(
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
    n_splits: int,
    batch_size: int,
    fixed_threshold: float = 0.5,
    use_imbalanced_bce: bool = False,
    compute_pos_weight_fn=None,
    pos_weight_mode: str = "sqrt",
    pos_weight_scale: float = 1.0,
    pos_weight_max: float = 10.0,
    train_neg_unique: bool = False,
    val_neg_unique: bool = True,
    test_neg_unique: bool = True,
) -> pd.DataFrame:
    seed_everything(seed)
    pos_edges = data_full[TARGET_ETYPE].edge_index
    all_pos_set = set(zip(pos_edges[0].tolist(), pos_edges[1].tolist()))
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    num_m = int(data_full["medicine"].num_nodes)
    num_r = int(data_full["reaction"].num_nodes)

    fold_metrics = []
    for fold, (tv_idx, test_idx) in enumerate(kf.split(np.arange(pos_edges.size(1)))):
        test_ei, tv_ei = pos_edges[:, test_idx], pos_edges[:, tv_idx]
        perm = np.random.permutation(tv_ei.size(1))
        tv_ei = tv_ei[:, perm]
        val_size = max(1, int(tv_ei.size(1) * 0.1))
        val_ei, sup_ei = tv_ei[:, :val_size], tv_ei[:, val_size:]

        train_graph = copy.deepcopy(data_full)
        del train_graph[TARGET_ETYPE]
        train_graph[TARGET_ETYPE].edge_index = sup_ei
        train_graph = T.ToUndirected()(train_graph)

        tr_idx, tr_y = _build_ml_edges(sup_ei, NEG_POS_RATIO, num_m, num_r, all_pos_set, unique_neg=train_neg_unique)
        val_idx, val_y = _build_ml_edges(val_ei, NEG_POS_RATIO, num_m, num_r, all_pos_set, unique_neg=val_neg_unique)
        te_idx, te_y = _build_ml_edges(test_ei, NEG_POS_RATIO, num_m, num_r, all_pos_set, unique_neg=test_neg_unique)
        tr_loader, val_loader, te_loader = make_loaders(
            train_graph, tr_idx, tr_y, val_idx, val_y, te_idx, te_y, batch_size=batch_size
        )

        pos_w = None
        if use_imbalanced_bce and compute_pos_weight_fn is not None:
            n_pos_sup = int(sup_ei.size(1))
            base_pos_w = compute_pos_weight_fn(num_m, num_r, n_pos_sup, max_w=pos_weight_max).to(DEVICE)
            pos_w = resolve_pos_weight(base_pos_w, mode=pos_weight_mode, scale=pos_weight_scale)

        model = model_builder(train_graph, med_feat, reac_feat, target_feat, herb_feat).to(DEVICE)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_v_auc = -1.0
        best_res = {}
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
                te_true, te_prob = collect_probs(model, te_loader)

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
                }
            else:
                early_stop_counter += 1

            if early_stop_counter >= patience:
                break

        if not best_res:
            val_true, val_prob = collect_probs(model, val_loader)
            best_thr = find_best_threshold(val_true, val_prob)
            tr_true, tr_prob = collect_probs(model, tr_loader)
            te_true, te_prob = collect_probs(model, te_loader)
            tr_auc, _, _, _, _, _, _ = calc_metrics(tr_true, tr_prob, threshold=fixed_threshold)
            v_auc, _, _, _, _, _, _ = calc_metrics(val_true, val_prob, threshold=fixed_threshold)
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
            }

        fold_metrics.append(best_res)
        print(
            f"[{model_name}][seed={seed}] fold {fold + 1}/{n_splits} "
            f"best_v_auc={best_v_auc:.4f} te_auc={best_res['te_auc']:.4f} | "
            f"train_unique={train_neg_unique} val_unique={val_neg_unique} test_unique={test_neg_unique}"
        )

    return pd.DataFrame(fold_metrics)


def main() -> None:
    args = parse_args()
    variants = _parse_variants(args.variants)
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    compute_pos_weight_fn = compute_bce_pos_weight

    ffn_drop = args.improve_dropout if args.ffn_dropout < 0 else args.ffn_dropout
    head_drop = args.improve_dropout if args.head_dropout < 0 else args.head_dropout
    use_film = not args.no_bimodal_film
    kw = _v13_kwargs(args, ffn_drop, head_drop, use_film)

    val_neg_unique = not bool(args.allow_dup_val_neg)
    test_neg_unique = not bool(args.allow_dup_test_neg)
    train_neg_unique = bool(args.train_neg_unique)

    print("[INFO] Variants:")
    for v in variants:
        print(f"  - {v}: {VARIANT_DESCRIPTIONS[v]}")
    print(f"[INFO] FiLM={'on' if use_film else 'off'}  path_max_ings={args.path_max_ings}")
    print(f"[INFO] Positive:negative ratio = 1:{int(NEG_POS_RATIO)} (neg_ratio={NEG_POS_RATIO})")
    print(
        f"[INFO] Negative sampling policy: train_unique={train_neg_unique} "
        f"val_unique={val_neg_unique} test_unique={test_neg_unique}"
    )
    print("[INFO] Loading data once...")
    raw_data, _ = load_tcm_full_data(use_dosage=True, dosage_mode=args.dosage_mode)
    med_feat, reac_feat, target_feat, herb_feat = prepare_unified_features(raw_data)
    ing_emb = raw_data["ingredient"].x

    builders: Dict[str, Callable] = {}
    if "v13_full" in variants:
        builders["v13_full"] = _make_v13_builder(kw)

    for variant in variants:
        if variant == "v13_full":
            continue
        mode, _ = PATH_SPECS[variant]
        builders[variant] = _make_path_builder(mode, kw, args.path_max_ings, ing_emb)

    all_rows = []
    for seed in seeds:
        print(f"\n>>> [Seed] {seed}")
        for label in variants:
            print(f"  - Running {label}...")
            df_run = train_one_cycle_dual_unique_eval_neg(
                model_name=label,
                model_builder=builders[label],
                data_full=raw_data,
                med_feat=med_feat,
                reac_feat=reac_feat,
                target_feat=target_feat,
                herb_feat=herb_feat,
                seed=seed,
                lr=args.lr,
                weight_decay=args.weight_decay,
                max_epochs=args.max_epochs,
                patience=args.patience,
                eval_every=args.eval_every,
                n_splits=args.n_splits,
                batch_size=args.batch_size,
                fixed_threshold=args.fixed_threshold,
                use_imbalanced_bce=True,
                compute_pos_weight_fn=compute_pos_weight_fn,
                pos_weight_mode=args.pos_weight_mode,
                pos_weight_scale=args.pos_weight_scale,
                pos_weight_max=args.pos_weight_max,
                train_neg_unique=train_neg_unique,
                val_neg_unique=val_neg_unique,
                test_neg_unique=test_neg_unique,
            )
            df_run["seed"] = seed
            df_run["model"] = label
            df_run["path_readout_variant"] = label
            if label in PATH_SPECS:
                df_run["path_readout_mode"] = PATH_SPECS[label][0]
            df_run["path_max_ings"] = args.path_max_ings
            df_run["train_neg_unique"] = train_neg_unique
            df_run["val_neg_unique"] = val_neg_unique
            df_run["test_neg_unique"] = test_neg_unique
            all_rows.append(df_run)

    final_df = pd.concat(all_rows, axis=0, ignore_index=True)
    final_df["fold"] = final_df.groupby(["seed", "model"]).cumcount()

    out_root = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_root, exist_ok=True)
    raw_path = os.path.join(out_root, "combined_all_seeds_raw_v13_vs_path_readout.csv")
    final_df.to_csv(raw_path, index=False)

    metrics_shared = ["tr_auc", "v_auc", "te_auc", "te_pr"]
    metrics_a = metrics_shared + [
        "te_f1_thr0p5",
        "te_acc_thr0p5",
        "te_mcc_thr0p5",
        "te_precision_thr0p5",
        "te_recall_thr0p5",
    ]
    metrics_b = metrics_shared + [
        "te_f1_valtuned",
        "te_acc_valtuned",
        "te_mcc_valtuned",
        "te_precision_valtuned",
        "te_recall_valtuned",
    ]

    baseline = "v13_full" if "v13_full" in variants else variants[0]
    summaries_a: Dict[str, dict] = {}
    summaries_b: Dict[str, dict] = {}
    for label in variants:
        sub = final_df[final_df["model"] == label]
        summaries_a[label] = aggregate_seed_summary(sub, seeds, metrics_a)
        summaries_b[label] = aggregate_seed_summary(sub, seeds, metrics_b)

    for label in variants:
        if label == baseline:
            continue
        print_comparison(
            f"PROTOCOL A vs {baseline}: fixed threshold = {args.fixed_threshold} -- {label}",
            baseline,
            summaries_a[baseline],
            label,
            summaries_a[label],
            metrics_a,
        )

    if baseline in variants and len(variants) > 1:
        for label in variants:
            if label == baseline:
                continue
            sig_a = paired_significance_tests(
                final_df,
                metrics_a,
                baseline_name=baseline,
                improved_name=label,
                protocol_label=f"fixed_{args.fixed_threshold}",
            )
            print_significance_table(sig_a, f"SIGNIFICANCE {label} vs {baseline} -- Protocol A")

    sig_rows = []
    if baseline in variants:
        for label in variants:
            if label == baseline:
                continue
            sig_a = paired_significance_tests(
                final_df,
                metrics_a,
                baseline_name=baseline,
                improved_name=label,
                protocol_label=f"fixed_{args.fixed_threshold}",
            )
            sig_b = paired_significance_tests(
                final_df,
                metrics_b,
                baseline_name=baseline,
                improved_name=label,
                protocol_label="val_tuned_mcc",
            )
            sig_a["compared_to"] = baseline
            sig_b["compared_to"] = baseline
            sig_rows.append(sig_a)
            sig_rows.append(sig_b)
    sig_combined = pd.concat(sig_rows, axis=0, ignore_index=True) if sig_rows else pd.DataFrame()
    sig_path = os.path.join(out_root, "significance_paired_v13_vs_path_readout.csv")
    if not sig_combined.empty:
        sig_combined.to_csv(sig_path, index=False)

    summary = {
        "script": "run_rgcn_v13_vs_path_readout_dual_protocol_13_unique_eval_neg_ratio1_10",
        "variants": {v: VARIANT_DESCRIPTIONS[v] for v in variants},
        "baseline_for_tables": baseline,
        "use_bimodal_film": use_film,
        "path_max_ings": args.path_max_ings,
        "seeds": seeds,
        "pos_neg_ratio": "1:10",
        "neg_pos_ratio": NEG_POS_RATIO,
        "negative_sampling": {
            "train_neg_unique": train_neg_unique,
            "val_neg_unique": val_neg_unique,
            "test_neg_unique": test_neg_unique,
        },
        "summary_protocol_A": summaries_a,
        "summary_protocol_B": summaries_b,
        "significance_paired": sig_combined.to_dict(orient="records") if not sig_combined.empty else [],
    }
    summary_path = os.path.join(out_root, "comparison_summary_v13_vs_path_readout.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n[DONE] Saved:\n  {raw_path}\n  {summary_path}")
    if not sig_combined.empty:
        print(f"  {sig_path}")


if __name__ == "__main__":
    main()
