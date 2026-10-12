import argparse
import json
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from baselines.common.data import load_tcm_full_data, prepare_unified_features, seed_everything
from baselines.common.train import run_baseline_cycle

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def _parse_models(s: str) -> List[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]


def load_best_hparams(best_json_path: str) -> Dict[str, Any]:
    with open(best_json_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_int(d: Dict[str, Any], key: str, default: int) -> int:
    if key not in d:
        return default
    try:
        return int(d[key])
    except Exception:
        return default


def _get_float(d: Dict[str, Any], key: str, default: float) -> float:
    if key not in d:
        return default
    try:
        return float(d[key])
    except Exception:
        return default


def build_model_args_from_best(
    model_name: str, best_all: Dict[str, Any]
) -> Tuple[float, float, int, int, int, int, List[int], int, int, int]:
    """
    Returns:
      lr, weight_decay,
      hidden_channels, num_conv_layers,
      transformer_num_layers, transformer_nhead,
      seeds,
      max_epochs, patience, eval_every
    """
    obj = best_all[model_name]
    final_best = obj.get("final_best", {})
    lr = _get_float(final_best, "lr", 1e-3)
    weight_decay = _get_float(final_best, "weight_decay", 1e-4)
    best_arch = final_best.get("best_arch", {}) or {}

    hidden_channels = _get_int(best_arch, "hidden_channels", 64)
    num_conv_layers = _get_int(best_arch, "num_conv_layers", 2)
    transformer_num_layers = _get_int(best_arch, "transformer_num_layers", 1)
    transformer_nhead = _get_int(best_arch, "transformer_nhead", 4)

    seeds = obj.get("seeds", {}).get("eval_seeds", None) or [42, 43, 44]
    # normalize to int list
    seeds = [int(x) for x in seeds]

    max_epochs = _get_int(obj.get("epochs", {}), "max_epochs_eval", 1000)
    patience = _get_int(obj.get("early_stop", {}), "patience", 20)
    eval_every = _get_int(obj.get("early_stop", {}), "eval_every", 5)

    return (
        lr,
        weight_decay,
        hidden_channels,
        num_conv_layers,
        transformer_num_layers,
        transformer_nhead,
        seeds,
        max_epochs,
        patience,
        eval_every,
    )


def mean_across_seeds(seed_dfs: List[pd.DataFrame], col: str) -> float:
    return float(np.mean([df[col].mean() for df in seed_dfs]))


def std_across_seeds(seed_dfs: List[pd.DataFrame], col: str) -> float:
    arr = [df[col].mean() for df in seed_dfs]
    return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0


def mean_std_across_folds(seed_dfs: List[pd.DataFrame], col: str) -> Tuple[float, float]:
    """
    Mean and sample std (ddof=1) over every fold score, pooling all seeds.

    Each row of each per-seed dataframe is one CV fold; values are concatenated
    before mean/std so ``std`` reflects fold-to-fold variability (and across seeds
    if multiple seeds are run).
    """
    vals = np.concatenate([df[col].to_numpy(dtype=float) for df in seed_dfs])
    m = float(np.mean(vals))
    if len(vals) <= 1:
        return m, 0.0
    return m, float(np.std(vals, ddof=1))


def format_mean_pm_std(mean: float, std: float, decimals: int = 4) -> str:
    return f"{mean:.{decimals}f}±{std:.{decimals}f}"


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
    out_dir: str,
) -> Tuple[List[pd.DataFrame], Dict[str, Any]]:
    """
    Runs `run_baseline_cycle` for each seed and writes per-seed CSVs.
    """
    seed_dfs: List[pd.DataFrame] = []
    summary_rows: List[Dict[str, Any]] = []

    for s in seeds:
        print(f"[RUN] {model_name} use_dosage=False seed={s} lr={lr} wd={weight_decay}")
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
        "use_dosage": False,
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


def format_report_row(
    model_name: str, agg: Dict[str, Any]
) -> str:
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Run best baseline models without dosage.")
    parser.add_argument(
        "--best_json",
        type=str,
        default=os.path.join(
            PROJECT_ROOT, "baselines", "grid_search_results", "best_hparams_all_models.json"
        ),
        help="Path to best hyperparams json.",
    )
    parser.add_argument(
        "--models",
        type=str,
        default="",
        help="Comma-separated model types, e.g. 'GCN,SAGE,Transformer'. Empty means all.",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="",
        help="Output dir. Empty means create timestamped dir under baselines.",
    )
    parser.add_argument(
        "--override_max_epochs",
        type=int,
        default=0,
        help="Override max_epochs_eval for all models (0 means use best json).",
    )
    parser.add_argument(
        "--override_patience",
        type=int,
        default=0,
        help="Override patience for all models (0 means use best json).",
    )
    parser.add_argument(
        "--override_eval_every",
        type=int,
        default=0,
        help="Override eval_every for all models (0 means use best json).",
    )
    parser.add_argument("--seed", type=int, default=60, help="Global seed for reproducibility.")
    args = parser.parse_args()

    best_json_path = args.best_json
    if not os.path.isabs(best_json_path):
        # Script may be run from repo root; normalize relative paths.
        best_json_path = os.path.abspath(best_json_path)

    best_all = load_best_hparams(best_json_path)
    model_types = _parse_models(args.models) or list(best_all.keys())

    if args.out_dir:
        out_dir = args.out_dir
    else:
        ts = time.strftime("%Y%m%d_%H%M%S")
        out_dir = os.path.join(PROJECT_ROOT, "baselines", f"no_dosage_best_run_{ts}")

    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    seed_everything(args.seed)

    print(f"[INFO] Loading data ONCE: use_dosage=False")
    data_full, _ = load_tcm_full_data(use_dosage=False)
    med_feat, reac_feat, target_feat, herb_feat = prepare_unified_features(data_full)

    run_results: List[Dict[str, Any]] = []
    seed_dfs_by_model: Dict[str, List[pd.DataFrame]] = {}

    for model_name in model_types:
        if model_name not in best_all:
            raise KeyError(f"Model '{model_name}' not found in best json.")

        (
            lr,
            weight_decay,
            hidden_channels,
            num_conv_layers,
            transformer_num_layers,
            transformer_nhead,
            seeds,
            max_epochs,
            patience,
            eval_every,
        ) = build_model_args_from_best(model_name, best_all)

        if args.override_max_epochs and args.override_max_epochs > 0:
            max_epochs = int(args.override_max_epochs)
        if args.override_patience and args.override_patience > 0:
            patience = int(args.override_patience)
        if args.override_eval_every and args.override_eval_every > 0:
            eval_every = int(args.override_eval_every)

        model_out_dir = os.path.join(out_dir, model_name)
        os.makedirs(model_out_dir, exist_ok=True)

        seed_dfs, agg = run_once_per_seed(
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
            out_dir=model_out_dir,
        )

        run_results.append(agg)
        seed_dfs_by_model[model_name] = seed_dfs

    results_csv = os.path.join(out_dir, "all_models_summary.csv")
    pd.DataFrame(run_results).to_csv(results_csv, index=False, encoding="utf-8-sig")

    # Print report similar to `grid_search_all.py`.
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
    ordered = [m for m in baselines_order if m in model_types]

    print("\n" + "=" * 12 + " No-dosage Best Baselines " + "=" * 12)
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

