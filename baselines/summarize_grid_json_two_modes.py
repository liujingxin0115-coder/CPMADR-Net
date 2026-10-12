"""
Summarize grid_search results for exactly two dosage modes from two
`best_hparams_all_models.json` files (e.g. dosage_mode_log_share vs dosage_mode_none).

This matches the common workflow: run grid_search_all.py twice with different
DOSAGE_MODE, each producing .../dosage_mode_<mode>/best_hparams_all_models.json

Output: wide table per model with te_auc mean±std for each mode and deltas.
Note: paired p-values require fold- or seed-level rows; JSON only has aggregate
eval_te_auc_mean/std across eval seeds, so p_vs_* is left empty (use
summarize_dosage_ablation.py on fold-level CSV if you have it).
"""

import argparse
import json
import os
import sys

import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


def _load(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--json_log_share",
        required=True,
        help="Path to best_hparams_all_models.json for log_share run",
    )
    parser.add_argument(
        "--json_none",
        required=True,
        help="Path to best_hparams_all_models.json for none run",
    )
    parser.add_argument(
        "--out_csv",
        default=os.path.join(
            "baselines", "grid_search_results", "summary_log_share_vs_none.csv"
        ),
    )
    args = parser.parse_args()

    js_log = _load(args.json_log_share)
    js_none = _load(args.json_none)

    models = sorted(set(js_log.keys()) | set(js_none.keys()))
    rows = []
    for m in models:
        e_log = (js_log.get(m) or {}).get("eval", {})
        e_none = (js_none.get(m) or {}).get("eval", {})
        te_l = float(e_log.get("eval_te_auc_mean", float("nan")))
        te_l_std = float(e_log.get("eval_te_auc_std", float("nan")))
        te_n = float(e_none.get("eval_te_auc_mean", float("nan")))
        te_n_std = float(e_none.get("eval_te_auc_std", float("nan")))
        v_l = float(e_log.get("eval_v_auc_mean", float("nan")))
        v_n = float(e_none.get("eval_v_auc_mean", float("nan")))

        rows.append(
            {
                "model": m,
                "log_share_te_auc_mean": te_l,
                "log_share_te_auc_std": te_l_std,
                "none_te_auc_mean": te_n,
                "none_te_auc_std": te_n_std,
                "delta_te_auc_mean_none_minus_log_share": te_n - te_l,
                "log_share_v_auc_mean": v_l,
                "none_v_auc_mean": v_n,
                "delta_v_auc_mean_none_minus_log_share": v_n - v_l,
            }
        )

    df = pd.DataFrame(rows).sort_values("model").reset_index(drop=True)
    out = args.out_csv
    d = os.path.dirname(out)
    if d:
        os.makedirs(d, exist_ok=True)
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"[DONE] wrote {out}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
