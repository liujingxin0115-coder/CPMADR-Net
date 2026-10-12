"""
Run `run_baseline_cycle` (baselines/common/train.py) with **dosage** data and
hyperparameters from `best_hparams_all_models.json` (e.g. grid output under
`dosage_mode_log_share/`).

Use this when you want the same training code path as `grid_search_all` / eval,
not `run_dosage_ablation.py` (no early-stop mismatch).

Example (align with dual_protocol_13 defaults: 500 epochs cap, patience 20, eval_every 5, 5-fold):

  python baselines/run_best_hparams_baseline_cycle_dosage.py \\
    --best_json baselines/grid_search_results/dosage_mode_log_share/best_hparams_all_models.json \\
    --dosage_mode log_share \\
    --seeds 42,43,44 \\
    --max_epochs 500 --patience 20 --eval_every 5 --n_splits 5 \\
    --models RGCN

Single seed (match one dual13 run):

  python baselines/run_best_hparams_baseline_cycle_dosage.py \\
    --best_json baselines/grid_search_results/dosage_mode_log_share/best_hparams_all_models.json \\
    --seeds 42 --models RGCN
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from baselines.common.data import load_tcm_full_data, prepare_unified_features, seed_everything
from baselines.common.train import run_baseline_cycle
from baselines.run_no_dosage_best_baselines import (
    build_model_args_from_best,
    load_best_hparams,
)


def mean_std_across_folds(seed_dfs: List[pd.DataFrame], col: str) -> Tuple[float, float]:
    """Mean and sample std (ddof=1) over fold scores, pooling all seeds."""
    vals = np.concatenate([df[col].to_numpy(dtype=float) for df in seed_dfs])
    m = float(np.mean(vals))
    if len(vals) <= 1:
        return m, 0.0
    return m, float(np.std(vals, ddof=1))


def format_mean_pm_std(mean: float, std: float, decimals: int = 4) -> str:
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"


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
) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    seed_dfs: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    for s in seeds:
        seed_everything(s)
        print(
            f"[RUN] {model_name} use_dosage=True seed={s} lr={lr} wd={weight_decay} "
            f"max_epochs={max_epochs} patience={patience} eval_every={eval_every} n_splits={n_splits}"
        )
        df, loss_curve = run_baseline_cycle(
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
    p = argparse.ArgumentParser(description="run_baseline_cycle + dosage + best_hparams JSON")
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
    p.add_argument("--n_splits", type=int, default=5, help="K-fold splits (dual_protocol_13 default: 5).")
    p.add_argument("--out_dir", type=str, default="", help="Output directory; default timestamp under baselines/grid_search_results/")
    p.add_argument(
        "--out_csv",
        type=str,
        default="",
        help="Alias for --out_dir (writes a whole run directory including all_models_summary.csv, not a single csv file).",
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
        out_dir = os.path.join(PROJECT_ROOT, "baselines", "grid_search_results", f"baseline_cycle_dosage_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    print(f"[INFO] Loading data: use_dosage=True, dosage_mode={args.dosage_mode}")
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
        "out_dir": out_dir,
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    results_csv = os.path.join(out_dir, "all_models_summary.csv")
    pd.DataFrame(run_results).to_csv(results_csv, index=False, encoding="utf-8-sig")

    ordered = [m for m in baselines_order if m in model_types]

    print("\n" + "=" * 12 + " Dosage + run_baseline_cycle (best JSON) " + "=" * 12)
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
