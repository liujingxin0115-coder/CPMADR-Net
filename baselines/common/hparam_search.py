import itertools
import json
import os
from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from .train import run_baseline_cycle


def _parse_float_list(s: str) -> List[float]:
    s = (s or "").strip()
    if not s:
        return []
    return [float(x) for x in s.split(",") if x.strip()]


def _parse_int_list(s: str) -> List[int]:
    s = (s or "").strip()
    if not s:
        return []
    return [int(x) for x in s.split(",") if x.strip()]


@dataclass
class SearchConfig:
    lr_candidates: List[float]
    wd_candidates: List[float]
    search_seeds: List[int]
    eval_seeds: List[int]
    max_epochs_search: int
    max_epochs_eval: int

    patience: int
    eval_every: int
    hidden_channels: int


def _evaluate_candidate_seedwise(
    model_type: str,
    raw_data,
    med_feat,
    reac_feat,
    target_feat,
    herb_feat,
    lr: float,
    wd: float,
    seeds: List[int],
    max_epochs: int,
    hidden_channels: int,
    patience: int,
    eval_every: int,
) -> Tuple[List[float], List[float]]:
    """
    Returns lists of per-seed means:
      v_auc_means_per_seed, te_auc_means_per_seed
    """
    v_means: List[float] = []
    te_means: List[float] = []
    for s in seeds:
        df, _ = run_baseline_cycle(
            model_name=model_type,
            data_full=raw_data,
            med_feat=med_feat,
            reac_feat=reac_feat,
            target_feat=target_feat,
            herb_feat=herb_feat,
            seed=s,
            lr=lr,
            weight_decay=wd,
            hidden_channels=hidden_channels,
            max_epochs=max_epochs,
            patience=patience,
            eval_every=eval_every,
        )
        v_means.append(float(df["v_auc"].mean()))
        te_means.append(float(df["te_auc"].mean()))

    return v_means, te_means


def grid_search_model(
    model_type: str,
    raw_data,
    med_feat,
    reac_feat,
    target_feat,
    herb_feat,
    cfg_module,
    out_dir: str,
    search_cfg: SearchConfig,
) -> Dict[str, object]:
    """
    Grid search on (lr, weight_decay) for a single model_type.
    Select best by mean validation AUC across search seeds.
    """
    os.makedirs(out_dir, exist_ok=True)

    results_rows: List[Dict[str, object]] = []
    best = None  # dict row

    lr_candidates = search_cfg.lr_candidates
    wd_candidates = search_cfg.wd_candidates

    print(f"\n===== Grid search for {model_type} =====")
    print(
        f"[INFO] candidates={len(lr_candidates) * len(wd_candidates)} "
        f"SEARCH_SEEDS={search_cfg.search_seeds} EVAL_SEEDS={search_cfg.eval_seeds} "
        f"MAX_EPOCHS_SEARCH={search_cfg.max_epochs_search} MAX_EPOCHS_EVAL={search_cfg.max_epochs_eval}"
    )

    for lr, wd in itertools.product(lr_candidates, wd_candidates):
        print(f"[CAND] {model_type} lr={lr} wd={wd} ...")

        v_list, te_list = _evaluate_candidate_seedwise(
            model_type=model_type,
            raw_data=raw_data,
            med_feat=med_feat,
            reac_feat=reac_feat,
            target_feat=target_feat,
            herb_feat=herb_feat,
            lr=lr,
            wd=wd,
            seeds=search_cfg.search_seeds,
            max_epochs=search_cfg.max_epochs_search,
            hidden_channels=search_cfg.hidden_channels,
            patience=search_cfg.patience,
            eval_every=search_cfg.eval_every,
        )

        v_mean = float(np.mean(v_list))
        te_mean = float(np.mean(te_list))
        v_std_val = float(np.std(v_list, ddof=1)) if len(v_list) > 1 else 0.0
        te_std_val = float(np.std(te_list, ddof=1)) if len(te_list) > 1 else 0.0

        print(
            f"[CAND_RES] {model_type} lr={lr} wd={wd} "
            f"v_auc={v_mean:.4f}±{v_std_val:.4f} te_auc={te_mean:.4f}±{te_std_val:.4f}"
        )

        row = {
            "model": model_type,
            "lr": lr,
            "weight_decay": wd,
            "search_v_auc_mean": v_mean,
            "search_v_auc_std": v_std_val,
            "search_te_auc_mean": te_mean,
            "search_te_auc_std": te_std_val,
        }
        results_rows.append(row)

        if best is None or v_mean > best["search_v_auc_mean"]:
            best = row

    assert best is not None

    # Final evaluation with best params
    best_lr = float(best["lr"])
    best_wd = float(best["weight_decay"])
    print(f"[BEST] {model_type}: lr={best_lr} wd={best_wd} search_v_auc={best['search_v_auc_mean']}")

    v_eval_list, te_eval_list = _evaluate_candidate_seedwise(
        model_type=model_type,
        raw_data=raw_data,
        med_feat=med_feat,
        reac_feat=reac_feat,
        target_feat=target_feat,
        herb_feat=herb_feat,
        lr=best_lr,
        wd=best_wd,
        seeds=search_cfg.eval_seeds,
        max_epochs=search_cfg.max_epochs_eval,
        hidden_channels=search_cfg.hidden_channels,
        patience=search_cfg.patience,
        eval_every=search_cfg.eval_every,
    )

    v_mean_eval = float(np.mean(v_eval_list))
    te_mean_eval = float(np.mean(te_eval_list))
    te_std_eval = float(np.std(te_eval_list, ddof=1)) if len(te_eval_list) > 1 else 0.0

    best_eval_res = {
        "model": model_type,
        "lr": best_lr,
        "weight_decay": best_wd,
        "eval_v_auc_mean": v_mean_eval,
        "eval_te_auc_mean": te_mean_eval,
        "eval_te_auc_std": te_std_eval,
    }
    print(
        f"[EVAL_RES] {model_type}: eval_v_auc={v_mean_eval:.4f}, eval_te_auc={te_mean_eval:.4f}±{te_std_eval:.4f}"
    )

    # Save outputs
    csv_path = os.path.join(out_dir, f"hparam_search_{model_type}.csv")
    pd.DataFrame(results_rows).to_csv(csv_path, index=False, encoding="utf-8-sig")

    json_path = os.path.join(out_dir, f"best_hparams_{model_type}.json")
    payload = {
        "model": model_type,
        "selection_metric": "val_v_auc",
        "selection_phase": "stage1",
        "best": {
            "lr": best_lr,
            "weight_decay": best_wd,
            "search_v_auc_mean": best["search_v_auc_mean"],
        },
        "eval": best_eval_res,
        "search_config": {
            "lr_candidates": lr_candidates,
            "wd_candidates": wd_candidates,
            "search_seeds": search_cfg.search_seeds,
            "eval_seeds": search_cfg.eval_seeds,
            "max_epochs_search": search_cfg.max_epochs_search,
            "max_epochs_eval": search_cfg.max_epochs_eval,
            "patience": search_cfg.patience,
            "eval_every": search_cfg.eval_every,
            "hidden_channels": search_cfg.hidden_channels,
        },
    }
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    return {"csv": csv_path, "json": json_path, "best_eval": best_eval_res}

