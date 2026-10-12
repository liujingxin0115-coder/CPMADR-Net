import argparse
import os

import numpy as np
import pandas as pd


def _paired_p_value(x: np.ndarray, y: np.ndarray) -> float:
    """
    Paired test with scipy if available; otherwise returns nan.
    """
    try:
        from scipy import stats

        _, p = stats.ttest_rel(x, y)
        return float(p)
    except Exception:
        return float("nan")


def _bh_fdr_adjust(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR correction; NaN stays NaN."""
    pvals = np.asarray(pvals, dtype=float)
    out = np.full_like(pvals, np.nan, dtype=float)
    valid_mask = np.isfinite(pvals)
    if not np.any(valid_mask):
        return out

    pv = pvals[valid_mask]
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order]

    adj_ranked = np.empty(m, dtype=float)
    prev = 1.0
    for i in range(m - 1, -1, -1):
        rank = i + 1
        val = ranked[i] * m / rank
        prev = min(prev, val)
        adj_ranked[i] = min(prev, 1.0)

    adj = np.empty(m, dtype=float)
    adj[order] = adj_ranked
    out[valid_mask] = adj
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--in_csv",
        default=os.path.join("baselines", "grid_search_results", "dosage_ablation_results.csv"),
    )
    parser.add_argument(
        "--out_csv",
        default=os.path.join("baselines", "grid_search_results", "dosage_ablation_summary.csv"),
    )
    parser.add_argument("--ref_mode", default="log_share")
    parser.add_argument(
        "--filter_modes",
        default="",
        help="Comma-separated dosage_mode to keep, e.g. 'none,log_share'. Empty = all.",
    )
    args = parser.parse_args()

    df = pd.read_csv(args.in_csv)
    if args.filter_modes.strip():
        keep = {x.strip() for x in args.filter_modes.split(",") if x.strip()}
        df = df[df["dosage_mode"].astype(str).isin(keep)].copy()
        if df.empty:
            raise ValueError(f"No rows left after --filter_modes={args.filter_modes!r}")
    required = {"model", "dosage_mode", "fold", "te_auc", "v_auc"}
    miss = required - set(df.columns)
    if miss:
        raise ValueError(f"Missing required columns: {sorted(miss)}")

    rows = []
    for (model, mode), g in df.groupby(["model", "dosage_mode"], as_index=False):
        row = {
            "model": model,
            "dosage_mode": mode,
            "n_folds": int(g["fold"].nunique()),
            "te_auc_mean": float(g["te_auc"].mean()),
            "te_auc_std": float(g["te_auc"].std(ddof=1)) if len(g) > 1 else 0.0,
            "v_auc_mean": float(g["v_auc"].mean()),
            "v_auc_std": float(g["v_auc"].std(ddof=1)) if len(g) > 1 else 0.0,
            "te_pr_mean": float(g["te_pr"].mean()) if "te_pr" in g else float("nan"),
            "te_f1_mean": float(g["te_f1"].mean()) if "te_f1" in g else float("nan"),
            "te_mcc_mean": float(g["te_mcc"].mean()) if "te_mcc" in g else float("nan"),
        }
        rows.append(row)

    summary = pd.DataFrame(rows).sort_values(["model", "dosage_mode"]).reset_index(drop=True)

    # Paired significance against ref_mode per model, using fold-aligned te_auc.
    pvals = []
    for _, row in summary.iterrows():
        model, mode = row["model"], row["dosage_mode"]
        if mode == args.ref_mode:
            pvals.append(np.nan)
            continue
        ref = df[(df["model"] == model) & (df["dosage_mode"] == args.ref_mode)][["fold", "te_auc"]].rename(
            columns={"te_auc": "te_auc_ref"}
        )
        cur = df[(df["model"] == model) & (df["dosage_mode"] == mode)][["fold", "te_auc"]].rename(
            columns={"te_auc": "te_auc_cur"}
        )
        merged = pd.merge(ref, cur, on="fold", how="inner")
        if merged.empty:
            pvals.append(np.nan)
        else:
            pvals.append(_paired_p_value(merged["te_auc_ref"].to_numpy(), merged["te_auc_cur"].to_numpy()))

    summary[f"p_vs_{args.ref_mode}"] = pvals
    summary[f"p_vs_{args.ref_mode}_bh_fdr"] = _bh_fdr_adjust(
        summary[f"p_vs_{args.ref_mode}"].to_numpy()
    )

    out_dir = os.path.dirname(args.out_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    summary.to_csv(args.out_csv, index=False, encoding="utf-8-sig")

    print(f"[DONE] summary saved: {args.out_csv}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()

