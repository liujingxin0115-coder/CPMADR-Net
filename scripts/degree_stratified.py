"""
Run baseline RGCN vs v13 path-readout on fixed medicine-id splits,
and evaluate degree bias by medicine-degree buckets (tail/medium/head).

Example:
  python scripts/degree_stratified.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from scipy import stats

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:
    plt = None

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

if os.environ.get("GNN02_LINK_LOADER_WORKERS", "").strip():
    os.environ["NUM_WORKERS"] = os.environ.get("GNN02_LINK_LOADER_WORKERS", "").strip()
else:
    os.environ["NUM_WORKERS"] = "0"

from baselines.common.data import load_tcm_full_data, prepare_unified_features  # noqa: E402
from baselines.common.models import UniversalBaselineAllEdge  # noqa: E402
from cpmadr_net.model import DoseAwareHSPESARGCNv13PathReadout  # noqa: E402
from cpmadr_net.training import (  # noqa: E402
    compute_bce_pos_weight,
    aggregate_seed_summary,
    paired_significance_tests,
    print_comparison,
    print_significance_table,
)
from cpmadr_net.splits import compute_bucket_metrics, load_fixed_medicine_splits  # noqa: E402
from cpmadr_net.splits import (  # noqa: E402
    DEFAULT_MR_PATH,
    LABEL_BASE,
    LABEL_V13_PATH,
    train_one_cycle_fixed_splits_v13_path_unique_eval_neg,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fixed medicine-split comparison with medicine-degree bucket analysis (RGCN vs v13_path)."
    )
    p.add_argument("--seeds", type=str, default="42,43,44,45,46")
    p.add_argument("--dosage_mode", type=str, default="log_share")
    p.add_argument("--max_epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--eval_every", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--hidden_channels", type=int, default=64)
    p.add_argument("--baseline_layers", type=int, default=1)
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
    p.add_argument("--v13_path_layers", type=int, default=-1, help="If >0, override v13_path num_conv_layers only.")
    p.add_argument("--v13_path_dropout", type=float, default=-1.0, help="If >=0, override v13_path dropout only.")
    p.add_argument("--v13_path_no_bimodal_film", action="store_true", help="Disable bimodal FiLM for v13_path only.")
    p.add_argument("--path_readout_mode", type=str, default="attn", choices=["attn", "attn_cross"])
    p.add_argument("--path_max_ings", type=int, default=32, help="Cap ingredients per side per link.")
    p.add_argument(
        "--path_ing_cos_mode",
        type=str,
        default="mean_pool",
        choices=["mean_pool", "max"],
        help="Path ingredient cosine mode used in extra link features.",
    )
    p.add_argument("--fixed_threshold", type=float, default=0.5)
    p.add_argument("--rgcn_fixed_threshold", type=float, default=-1.0, help="If >=0, override fixed threshold for RGCN.")
    p.add_argument(
        "--v13_path_fixed_threshold",
        type=float,
        default=-1.0,
        help="If >=0, override fixed threshold for v13_path.",
    )
    p.add_argument("--pos_weight_mode", type=str, default="sqrt", choices=["none", "sqrt", "linear"])
    p.add_argument("--pos_weight_scale", type=float, default=1.0)
    p.add_argument("--pos_weight_max", type=float, default=10.0)
    p.add_argument("--mr_path", type=str, default=DEFAULT_MR_PATH)
    p.add_argument("--split_root", type=str, default="unbalanced/splits")
    p.add_argument("--n_splits", type=int, default=5)
    p.add_argument("--tail_max_freq_for_training", type=int, default=3, help="Only used in internal tail-row extraction.")
    p.add_argument(
        "--train_neg_unique",
        action="store_true",
        help="Use unique negatives for training split (default: duplicated allowed).",
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
        "--degree_bucket_mode",
        type=str,
        default="quantile",
        choices=["quantile", "fixed"],
        help="How to define medicine-degree buckets.",
    )
    p.add_argument("--degree_q_tail", type=float, default=1.0 / 3.0, help="Tail quantile (quantile mode).")
    p.add_argument("--degree_q_medium", type=float, default=2.0 / 3.0, help="Medium quantile (quantile mode).")
    p.add_argument("--degree_tail_max", type=float, default=8.0, help="Tail max degree (fixed mode).")
    p.add_argument("--degree_medium_max", type=float, default=20.0, help="Medium max degree (fixed mode).")
    p.add_argument("--skip_degree_plot", action="store_true", help="Disable degree AUC bar plotting.")
    p.add_argument("--out_dir", type=str, default="results/RGCN_fixed_medicine_degree_v13_path_unique_eval_neg")
    return p.parse_args()


def _build_medicine_degree_map(mr_df: pd.DataFrame, n_medicine_nodes: int) -> Dict[int, int]:
    deg = mr_df.groupby("medicine_id")["adr_id"].nunique().to_dict()
    out: Dict[int, int] = {}
    for mid in range(int(n_medicine_nodes)):
        out[int(mid)] = int(deg.get(int(mid), 0))
    return out


def _resolve_degree_bucket_thresholds(
    degree_map: Dict[int, int],
    mode: str,
    q_tail: float,
    q_medium: float,
    fixed_tail_max: float,
    fixed_medium_max: float,
) -> Tuple[float, float, Dict]:
    if mode == "fixed":
        tail_cut = float(fixed_tail_max)
        medium_cut = float(fixed_medium_max)
        if medium_cut < tail_cut:
            raise ValueError("degree_medium_max must be >= degree_tail_max.")
        return tail_cut, medium_cut, {"mode": "fixed", "tail_cut": tail_cut, "medium_cut": medium_cut}

    if not (0.0 <= q_tail <= 1.0 and 0.0 <= q_medium <= 1.0):
        raise ValueError("degree quantiles must be in [0, 1].")
    if q_medium < q_tail:
        raise ValueError("degree_q_medium must be >= degree_q_tail.")

    vals = np.asarray(list(degree_map.values()), dtype=np.float64)
    tail_cut = float(np.quantile(vals, q_tail))
    medium_cut = float(np.quantile(vals, q_medium))
    if medium_cut < tail_cut:
        medium_cut = tail_cut
    meta = {
        "mode": "quantile",
        "q_tail": float(q_tail),
        "q_medium": float(q_medium),
        "tail_cut": tail_cut,
        "medium_cut": medium_cut,
    }
    return tail_cut, medium_cut, meta


def _assign_degree_bucket(degree: int, tail_cut: float, medium_cut: float) -> str:
    if float(degree) <= float(tail_cut):
        return "tail"
    if float(degree) <= float(medium_cut):
        return "medium"
    return "head"


def _bucket_order_key(name: str) -> int:
    order = {"tail": 0, "medium": 1, "head": 2}
    return order.get(str(name), 99)


def _paired_significance_tests_by_bucket(
    bucket_df: pd.DataFrame,
    metrics: List[str],
    baseline_name: str,
    improved_name: str,
) -> pd.DataFrame:
    if bucket_df.empty:
        return pd.DataFrame()
    rows: List[Dict] = []
    for bucket in sorted(bucket_df["bucket"].unique(), key=_bucket_order_key):
        bl = bucket_df[(bucket_df["model"] == baseline_name) & (bucket_df["bucket"] == bucket)].copy()
        im = bucket_df[(bucket_df["model"] == improved_name) & (bucket_df["bucket"] == bucket)].copy()
        if bl.empty or im.empty:
            continue

        merge_cols = ["seed", "fold"]
        use_metrics = [m for m in metrics if m in bl.columns and m in im.columns]
        merged = bl[merge_cols + use_metrics].merge(im[merge_cols + use_metrics], on=merge_cols, suffixes=("_baseline", "_improved"))
        n_pairs = len(merged)
        for m in use_metrics:
            bcol, icol = f"{m}_baseline", f"{m}_improved"
            x_b = merged[bcol].to_numpy(dtype=np.float64)
            x_i = merged[icol].to_numpy(dtype=np.float64)
            diff = x_i - x_b
            mean_diff = float(np.mean(diff)) if n_pairs else float("nan")
            std_diff = float(np.std(diff, ddof=1)) if n_pairs > 1 else 0.0
            cohens_dz = float(mean_diff / std_diff) if std_diff > 1e-12 else float("nan")

            t_stat = float("nan")
            p_t = float("nan")
            w_stat = float("nan")
            p_w = float("nan")
            if n_pairs >= 2:
                t_res = stats.ttest_rel(x_i, x_b)
                t_stat = float(t_res.statistic)
                p_t = float(t_res.pvalue)
                try:
                    if np.allclose(diff, 0.0):
                        w_stat = 0.0
                        p_w = 1.0
                    else:
                        w_res = stats.wilcoxon(diff, zero_method="wilcox", alternative="two-sided")
                        w_stat = float(w_res.statistic)
                        p_w = float(w_res.pvalue)
                except ValueError:
                    pass

            rows.append(
                {
                    "bucket": bucket,
                    "metric": m,
                    "n_pairs": int(n_pairs),
                    "mean_diff_improved_minus_baseline": mean_diff,
                    "cohens_dz": cohens_dz,
                    "ttest_statistic": t_stat,
                    "ttest_pvalue_two_sided": p_t,
                    "wilcoxon_statistic": w_stat,
                    "wilcoxon_pvalue_two_sided": p_w,
                    "comparison": f"{improved_name} vs {baseline_name}",
                }
            )
    return pd.DataFrame(rows)


def _star_from_p(p: float) -> str:
    if p is None or np.isnan(p):
        return "ns"
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "ns"


def _plot_degree_auc_bar(bucket_df: pd.DataFrame, sig_df: pd.DataFrame, out_root: str) -> List[str]:
    if plt is None or bucket_df.empty:
        return []

    order = ["tail", "medium", "head"]
    labels = ["Tail\n(Low Degree)", "Medium", "Head\n(High Degree)"]
    models = [LABEL_V13_PATH, LABEL_BASE]
    colors = {LABEL_V13_PATH: "#cf7a6a", LABEL_BASE: "#7aa6c8"}

    stats_df = (
        bucket_df.groupby(["model", "bucket"])["roc_auc"]
        .agg(["mean", "std"])
        .reset_index()
    )
    fig, ax = plt.subplots(figsize=(9.0, 6.0))
    x = np.arange(len(order))
    width = 0.34

    y_max = 0.0
    for i, model in enumerate(models):
        means = []
        errs = []
        for b in order:
            row = stats_df[(stats_df["model"] == model) & (stats_df["bucket"] == b)]
            if row.empty:
                means.append(np.nan)
                errs.append(0.0)
            else:
                means.append(float(row["mean"].iloc[0]))
                errs.append(float(row["std"].fillna(0.0).iloc[0]))
        xpos = x + (i - 0.5) * width
        ax.bar(xpos, means, width=width, label=model, color=colors.get(model, None), edgecolor="black", linewidth=0.5)
        ax.errorbar(xpos, means, yerr=errs, fmt="none", ecolor="black", elinewidth=1.0, capsize=4, capthick=1.0)
        y_max = max(y_max, np.nanmax(np.asarray(means) + np.asarray(errs)))

    p_map = {}
    if not sig_df.empty:
        sub = sig_df[sig_df["metric"] == "roc_auc"]
        for _, r in sub.iterrows():
            p_map[str(r["bucket"])] = float(r["wilcoxon_pvalue_two_sided"])
    for j, b in enumerate(order):
        p = p_map.get(b, np.nan)
        star = _star_from_p(p)
        y = max(0.62, y_max + 0.015)
        ax.plot([x[j] - width / 2, x[j] + width / 2], [y, y], color="black", linewidth=1.0)
        ax.text(x[j], y + 0.003, star, ha="center", va="bottom", fontsize=10, fontstyle="italic")

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("AUC")
    ax.set_xlabel("Medicine Degree Group")
    ax.set_ylim(0.60, min(1.0, max(0.98, y_max + 0.04)))
    ax.set_title("Degree Bias Analysis (Medicine Degree Buckets)")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend(loc="upper right")
    fig.tight_layout()

    png_path = os.path.join(out_root, "degree_bias_auc_bar_medicine_degree_fixed_tail_splits_v13_path.png")
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [png_path]


def main() -> None:
    args = parse_args()
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    split_root_abs = os.path.join(PROJECT_ROOT, args.split_root)
    fixed_splits = load_fixed_medicine_splits(split_root_abs, args.n_splits)

    compute_pos_weight_fn = compute_bce_pos_weight
    use_film = not args.no_bimodal_film
    v13_path_layers = args.improve_layers if args.v13_path_layers <= 0 else int(args.v13_path_layers)
    v13_path_dropout = args.improve_dropout if args.v13_path_dropout < 0 else float(args.v13_path_dropout)
    v13_path_ffn_drop = v13_path_dropout if args.ffn_dropout < 0 else args.ffn_dropout
    v13_path_head_drop = v13_path_dropout if args.head_dropout < 0 else args.head_dropout
    v13_path_use_film = False if args.v13_path_no_bimodal_film else use_film

    train_neg_unique = bool(args.train_neg_unique)
    val_neg_unique = not bool(args.allow_dup_val_neg)
    test_neg_unique = not bool(args.allow_dup_test_neg)
    print(
        "[INFO] Negative sampling policy: "
        f"train_unique={train_neg_unique} val_unique={val_neg_unique} test_unique={test_neg_unique}"
    )

    print("[INFO] Loading data once...")
    raw_data, _ = load_tcm_full_data(use_dosage=True, dosage_mode=args.dosage_mode)
    med_feat, reac_feat, target_feat, herb_feat = prepare_unified_features(raw_data)
    ing_emb = raw_data["ingredient"].x

    mr_path_abs = os.path.join(PROJECT_ROOT, args.mr_path)
    mr_df = pd.read_csv(mr_path_abs)
    reaction_freq = mr_df.groupby("adr_id")["medicine_id"].nunique()
    reaction_freq_map = {int(r): int(c) for r, c in reaction_freq.items()}
    tail_reactions = {
        int(r) for r, c in reaction_freq.items() if int(c) <= int(args.tail_max_freq_for_training)
    }

    medicine_degree_map = _build_medicine_degree_map(mr_df, int(raw_data["medicine"].num_nodes))
    degree_tail_cut, degree_medium_cut, degree_meta = _resolve_degree_bucket_thresholds(
        degree_map=medicine_degree_map,
        mode=args.degree_bucket_mode,
        q_tail=float(args.degree_q_tail),
        q_medium=float(args.degree_q_medium),
        fixed_tail_max=float(args.degree_tail_max),
        fixed_medium_max=float(args.degree_medium_max),
    )
    print(
        "[INFO] Medicine-degree buckets: "
        f"mode={degree_meta['mode']} tail<= {degree_tail_cut:.4f} medium<= {degree_medium_cut:.4f} head> {degree_medium_cut:.4f}"
    )

    def baseline_builder(train_graph, med_x, reac_x, targ_x, herb_x):
        return UniversalBaselineAllEdge(
            model_type="RGCN",
            hidden_channels=args.hidden_channels,
            metadata=train_graph.metadata(),
            data_obj=train_graph,
            init_med_x=med_x,
            init_reac_x=reac_x,
            init_target_x=targ_x,
            init_herb_x=herb_x,
            num_conv_layers=args.baseline_layers,
            transformer_num_layers=1,
            transformer_nhead=1,
        )

    def v13_path_builder(train_graph, med_x, reac_x, targ_x, herb_x):
        return DoseAwareHSPESARGCNv13PathReadout(
            metadata=train_graph.metadata(),
            data_obj=train_graph,
            init_med_x=med_x,
            init_reac_x=reac_x,
            init_target_x=targ_x,
            init_herb_x=herb_x,
            hidden_channels=args.hidden_channels,
            num_conv_layers=v13_path_layers,
            dropout=v13_path_dropout,
            rel_emb_dim=args.improve_rel_emb_dim,
            edge_drop=args.improve_edge_drop,
            ffn_mult=args.ffn_mult,
            esa_mlp_hidden=args.esa_mlp_hidden,
            ffn_dropout=v13_path_ffn_drop,
            use_bimodal_film=v13_path_use_film,
            head_dropout=v13_path_head_drop,
            head_res_blocks=args.head_res_blocks,
            path_readout_mode=args.path_readout_mode,
            path_max_ings=args.path_max_ings,
            ingredient_emb=ing_emb,
            path_ing_cos_mode=args.path_ing_cos_mode,
        )

    models = [(LABEL_BASE, baseline_builder), (LABEL_V13_PATH, v13_path_builder)]
    threshold_by_model = {
        LABEL_BASE: float(args.rgcn_fixed_threshold) if args.rgcn_fixed_threshold >= 0 else float(args.fixed_threshold),
        LABEL_V13_PATH: float(args.v13_path_fixed_threshold)
        if args.v13_path_fixed_threshold >= 0
        else float(args.fixed_threshold),
    }

    all_rows: List[pd.DataFrame] = []
    all_test_pred_rows: List[pd.DataFrame] = []
    for i, seed in enumerate(seeds):
        print(f"\n>>> [Seed {i + 1}/{len(seeds)}] {seed}")
        for label, builder in models:
            print(f"  - Running {label}...")
            model_fixed_thr = threshold_by_model[label]
            df_run, _, df_all_test = train_one_cycle_fixed_splits_v13_path_unique_eval_neg(
                model_name=label,
                model_builder=builder,
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
                batch_size=args.batch_size,
                fixed_threshold=model_fixed_thr,
                fixed_splits=fixed_splits,
                use_imbalanced_bce=(label == LABEL_V13_PATH),
                compute_pos_weight_fn=compute_pos_weight_fn,
                pos_weight_mode=args.pos_weight_mode,
                pos_weight_scale=args.pos_weight_scale,
                pos_weight_max=args.pos_weight_max,
                tail_reactions=tail_reactions,
                reaction_freq_map=reaction_freq_map,
                train_neg_unique=train_neg_unique,
                val_neg_unique=val_neg_unique,
                test_neg_unique=test_neg_unique,
            )
            df_run["seed"] = seed
            df_run["model"] = label
            df_run["fixed_threshold_used"] = model_fixed_thr
            df_run["train_neg_unique"] = train_neg_unique
            df_run["val_neg_unique"] = val_neg_unique
            df_run["test_neg_unique"] = test_neg_unique
            all_rows.append(df_run)

            if not df_all_test.empty:
                df_all_test["seed"] = seed
                df_all_test["model"] = label
                all_test_pred_rows.append(df_all_test)

    final_df = pd.concat(all_rows, axis=0, ignore_index=True)
    final_df["fold"] = final_df.groupby(["seed", "model"]).cumcount()

    out_root = os.path.join(PROJECT_ROOT, args.out_dir)
    os.makedirs(out_root, exist_ok=True)

    raw_path = os.path.join(out_root, "combined_all_seeds_raw_fixed_tail_splits_v13_path.csv")
    final_df.to_csv(raw_path, index=False)

    bucket_df = pd.DataFrame()
    bucket_summary = pd.DataFrame()
    bucket_sig_df = pd.DataFrame()
    if all_test_pred_rows:
        all_test_df = pd.concat(all_test_pred_rows, axis=0, ignore_index=True)
        all_test_df["medicine_degree"] = all_test_df["medicine_id"].map(medicine_degree_map).fillna(0).astype(int)
        all_test_df["bucket"] = all_test_df["medicine_degree"].apply(
            lambda d: _assign_degree_bucket(int(d), degree_tail_cut, degree_medium_cut)
        )

        all_test_path = os.path.join(out_root, "all_test_predictions_medicine_degree_fixed_tail_splits_v13_path.csv")
        all_test_df.to_csv(all_test_path, index=False)

        bucket_df = compute_bucket_metrics(all_test_df)
        bucket_path = os.path.join(out_root, "bucket_metrics_medicine_degree_fixed_tail_splits_v13_path.csv")
        bucket_df.to_csv(bucket_path, index=False)

        bucket_metric_cols = [c for c in bucket_df.columns if c not in {"model", "bucket", "seed", "fold"}]
        bucket_summary = (
            bucket_df.groupby(["model", "bucket"])[bucket_metric_cols]
            .mean()
            .reset_index()
        )
        bucket_summary["bucket_order"] = bucket_summary["bucket"].map(_bucket_order_key)
        bucket_summary = bucket_summary.sort_values(["model", "bucket_order"]).drop(columns=["bucket_order"])
        bucket_sum_path = os.path.join(out_root, "bucket_metrics_summary_medicine_degree_fixed_tail_splits_v13_path.csv")
        bucket_summary.to_csv(bucket_sum_path, index=False)

        bucket_sig_metrics = [
            "roc_auc",
            "pr_auc",
            "f1_thr0p5",
            "acc_thr0p5",
            "mcc_thr0p5",
            "precision_thr0p5",
            "recall_thr0p5",
            "f1_valtuned",
            "acc_valtuned",
            "mcc_valtuned",
            "precision_valtuned",
            "recall_valtuned",
        ]
        bucket_sig_df = _paired_significance_tests_by_bucket(
            bucket_df=bucket_df,
            metrics=bucket_sig_metrics,
            baseline_name=LABEL_BASE,
            improved_name=LABEL_V13_PATH,
        )
        bucket_sig_path = os.path.join(
            out_root, "significance_by_bucket_medicine_degree_fixed_tail_splits_v13_path.csv"
        )
        bucket_sig_df.to_csv(bucket_sig_path, index=False)

        if not bool(args.skip_degree_plot):
            plot_paths = _plot_degree_auc_bar(bucket_df, bucket_sig_df, out_root)
            if plot_paths:
                print("[INFO] Degree-bias plot saved:")
                for pth in plot_paths:
                    print(f"  {pth}")

    metrics_shared = ["tr_auc", "v_auc", "te_auc", "te_pr"]
    metrics_a = metrics_shared + ["te_f1_thr0p5", "te_acc_thr0p5", "te_mcc_thr0p5", "te_precision_thr0p5", "te_recall_thr0p5"]
    metrics_b = metrics_shared + ["te_f1_valtuned", "te_acc_valtuned", "te_mcc_valtuned", "te_precision_valtuned", "te_recall_valtuned"]

    bl = final_df[final_df["model"] == LABEL_BASE]
    im = final_df[final_df["model"] == LABEL_V13_PATH]
    sum_bl_a = aggregate_seed_summary(bl, seeds, metrics_a)
    sum_im_a = aggregate_seed_summary(im, seeds, metrics_a)
    sum_bl_b = aggregate_seed_summary(bl, seeds, metrics_b)
    sum_im_b = aggregate_seed_summary(im, seeds, metrics_b)
    print_comparison(f"PROTOCOL A (Base vs {LABEL_V13_PATH})", LABEL_BASE, sum_bl_a, LABEL_V13_PATH, sum_im_a, metrics_a)
    print_comparison(f"PROTOCOL B (Base vs {LABEL_V13_PATH})", LABEL_BASE, sum_bl_b, LABEL_V13_PATH, sum_im_b, metrics_b)

    sig_a = paired_significance_tests(
        final_df,
        metrics_a,
        baseline_name=LABEL_BASE,
        improved_name=LABEL_V13_PATH,
        protocol_label=(
            f"{LABEL_V13_PATH}_vs_{LABEL_BASE}_fixed_modelwise_"
            f"{threshold_by_model[LABEL_V13_PATH]:.4f}_{threshold_by_model[LABEL_BASE]:.4f}"
        ),
    )
    sig_b = paired_significance_tests(
        final_df,
        metrics_b,
        baseline_name=LABEL_BASE,
        improved_name=LABEL_V13_PATH,
        protocol_label=f"{LABEL_V13_PATH}_vs_{LABEL_BASE}_val_tuned_mcc",
    )
    for df_sig, pview in ((sig_a, "A_fixed_modelwise"), (sig_b, "B_val_tuned_mcc")):
        df_sig["comparison"] = f"{LABEL_V13_PATH} vs {LABEL_BASE}"
        df_sig["protocol_view"] = pview
        df_sig["stat_level"] = "fold_level"
    print_significance_table(sig_a, f"SIGNIFICANCE (FOLD-LEVEL) - {LABEL_V13_PATH} vs {LABEL_BASE} (Protocol A)")
    print_significance_table(sig_b, f"SIGNIFICANCE (FOLD-LEVEL) - {LABEL_V13_PATH} vs {LABEL_BASE} (Protocol B)")

    sig_combined = pd.concat([sig_a, sig_b], axis=0, ignore_index=True)
    sig_path = os.path.join(out_root, "significance_vs_baseline_fixed_tail_splits_v13_path.csv")
    sig_combined.to_csv(sig_path, index=False)

    summary = {
        "script": "scripts/degree_stratified.py",
        "labels": {"baseline": LABEL_BASE, "v13_path": LABEL_V13_PATH},
        "seeds": seeds,
        "mr_path": args.mr_path,
        "split_root": args.split_root,
        "fixed_threshold": args.fixed_threshold,
        "fixed_threshold_by_model": threshold_by_model,
        "negative_sampling": {
            "train_neg_unique": train_neg_unique,
            "val_neg_unique": val_neg_unique,
            "test_neg_unique": test_neg_unique,
        },
        "degree_bucketing": {
            **degree_meta,
            "degree_bucket_mode": args.degree_bucket_mode,
            "n_medicine_with_degree": int(len(medicine_degree_map)),
            "bucket_counts": (
                pd.Series(list(medicine_degree_map.values()))
                .map(lambda d: _assign_degree_bucket(int(d), degree_tail_cut, degree_medium_cut))
                .value_counts()
                .to_dict()
            ),
        },
        "v13_path_training": {
            "loss_type": "bce",
            "use_imbalanced_bce": True,
            "rank_loss_weight": 0.0,
            "v13_path_layers": v13_path_layers,
            "v13_path_dropout": v13_path_dropout,
            "v13_path_use_bimodal_film": v13_path_use_film,
            "path_readout_mode": args.path_readout_mode,
            "path_max_ings": args.path_max_ings,
            "path_ing_cos_mode": args.path_ing_cos_mode,
        },
        "significance_vs_baseline": sig_combined.to_dict(orient="records"),
        "significance_by_bucket": bucket_sig_df.to_dict(orient="records") if not bucket_sig_df.empty else [],
    }
    summary_path = os.path.join(out_root, "summary_medicine_degree_fixed_tail_splits_v13_path.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(
        "\n[DONE] Saved:\n"
        f"  {raw_path}\n"
        f"  {sig_path}\n"
        f"  {summary_path}"
    )
    if not bucket_df.empty:
        print("  (medicine-degree bucket artifacts written in same out_dir)")


if __name__ == "__main__":
    main()

