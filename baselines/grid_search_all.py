import importlib
import os
import sys
from typing import Dict


# Allow "direct run" (python baselines/grid_search_all.py) to import baselines package.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)


from baselines.common.data import seed_everything, load_tcm_full_data, prepare_unified_features
from baselines.common.hparam_search import SearchConfig, grid_search_model


def _get_search_candidates():
    # default: match your earlier hparam search script
    lr_str = os.environ.get("LR_CANDIDATES", "3e-4,1e-3,3e-3")
    wd_str = os.environ.get("WD_CANDIDATES", "1e-4")
    from baselines.common.hparam_search import _parse_float_list, _parse_int_list

    lr_candidates = _parse_float_list(lr_str)
    wd_candidates = _parse_float_list(wd_str)
    return lr_candidates, wd_candidates, lr_str, wd_str


def main() -> None:
    seed_everything(60)

    default_model_types = [
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
    # e.g. MODEL_TYPES="RGAT,GraphEdgeDosage" to only search some models.
    model_types_env = os.environ.get("MODEL_TYPES", "").strip()
    model_types = [m.strip() for m in model_types_env.split(",") if m.strip()] if model_types_env else default_model_types

    lr_candidates, wd_candidates, lr_str, wd_str = _get_search_candidates()
    search_seeds = None
    eval_seeds = None

    from baselines.common.hparam_search import _parse_int_list

    if os.environ.get("SEARCH_SEEDS"):
        search_seeds = _parse_int_list(os.environ["SEARCH_SEEDS"])
    if os.environ.get("EVAL_SEEDS"):
        eval_seeds = _parse_int_list(os.environ["EVAL_SEEDS"])

    # Architecture candidates for stage-2 (after choosing best lr/wd).
    # Keep small defaults for CPU feasibility; adjust via env vars if needed.
    hidden_channels_candidates = _parse_int_list(os.environ.get("HC_CANDIDATES", "32,64"))
    num_conv_layers_candidates = _parse_int_list(os.environ.get("CONV_LAYERS_CANDIDATES", "1,2"))
    transformer_num_layers_candidates = _parse_int_list(
        os.environ.get("TRANSFORMER_LAYERS_CANDIDATES", "1,2")
    )
    transformer_nhead_candidates = _parse_int_list(os.environ.get("TRANSFORMER_NHEAD_CANDIDATES", "2,4"))

    max_epochs_search = int(os.environ.get("MAX_EPOCHS_SEARCH", "200"))
    max_epochs_eval = int(os.environ.get("MAX_EPOCHS_EVAL", "500"))

    # Dosage mode for data loading / weighting:
    # - none, raw_log, log_share, linear_share
    dosage_mode = os.environ.get("DOSAGE_MODE", "log_share").strip().lower()
    if dosage_mode not in {"none", "raw_log", "log_share", "linear_share"}:
        raise ValueError(
            f"Unsupported DOSAGE_MODE={dosage_mode!r}. "
            "Use one of: none, raw_log, log_share, linear_share."
        )

    # Common save dir (mode-scoped to prevent result overwrite across dosage settings)
    out_root_base = os.environ.get(
        "GRID_SEARCH_OUT_DIR",
        os.path.join(PROJECT_ROOT, "baselines", "grid_search_results"),
    )
    out_root = os.path.join(out_root_base, f"dosage_mode_{dosage_mode}")
    os.makedirs(out_root, exist_ok=True)

    # Load data ONCE (all your baseline configs set USE_DOSAGE=True).
    first_cfg = importlib.import_module(f"baselines.{model_types[0]}.config")
    use_dosage = bool(getattr(first_cfg, "USE_DOSAGE", True))
    print(f"[INFO] Loading data once: use_dosage={use_dosage}, dosage_mode={dosage_mode}")
    raw_data, _ = load_tcm_full_data(use_dosage=use_dosage, dosage_mode=dosage_mode)
    med_feat_unif, reac_feat_unif, target_feat_unif, herb_feat_unif = prepare_unified_features(raw_data)

    all_best: Dict[str, object] = {}
    # Store per-seed DataFrames for final report printing.
    # all_res_for_report[model_type] = [df_seed1, df_seed2, ...]
    all_res_for_report: Dict[str, list] = {}

    for model_type in model_types:
        cfg_module = importlib.import_module(f"baselines.{model_type}.config")
        seeds_default = list(getattr(cfg_module, "SEEDS", [42, 43, 44]))

        patience = int(getattr(cfg_module, "PATIENCE", 20))
        eval_every = int(getattr(cfg_module, "EVAL_EVERY", 5))
        hidden_channels = int(getattr(cfg_module, "HIDDEN_CHANNELS", 64))
        # fallback to cfg MAX_EPOCHS if user doesn't override eval
        cfg_max_epochs_eval = int(getattr(cfg_module, "MAX_EPOCHS", 500))
        max_eval = max_epochs_eval if max_epochs_eval is not None else cfg_max_epochs_eval

        cur_search_seeds = search_seeds if search_seeds else seeds_default
        cur_eval_seeds = eval_seeds if eval_seeds else seeds_default

        # -------- Stage 1: lr/wd grid search --------
        # Use a fixed hidden_channels for stage-1 so lr/wd search cost stays manageable.
        stage1_hidden_channels = hidden_channels
        if hidden_channels_candidates and stage1_hidden_channels not in hidden_channels_candidates:
            stage1_hidden_channels = hidden_channels_candidates[0]

        search_cfg = SearchConfig(
            lr_candidates=lr_candidates,
            wd_candidates=wd_candidates,
            search_seeds=cur_search_seeds,
            eval_seeds=cur_eval_seeds,
            max_epochs_search=max_epochs_search,
            max_epochs_eval=max_eval,
            patience=patience,
            eval_every=eval_every,
            hidden_channels=stage1_hidden_channels,
        )

        model_out_dir_stage1 = os.path.join(out_root, model_type, "stage1")
        res1 = grid_search_model(
            model_type=model_type,
            raw_data=raw_data,
            med_feat=med_feat_unif,
            reac_feat=reac_feat_unif,
            target_feat=target_feat_unif,
            herb_feat=herb_feat_unif,
            cfg_module=cfg_module,
            out_dir=model_out_dir_stage1,
            search_cfg=search_cfg,
        )

        best_lr = float(res1["best_eval"]["lr"])
        best_wd = float(res1["best_eval"]["weight_decay"])

        # -------- Stage 2: architecture grid search --------
        print(f"\n===== Stage 2 architecture search for {model_type} (lr={best_lr}, wd={best_wd}) =====")
        import numpy as np
        import pandas as pd

        stage2_out_dir = os.path.join(out_root, model_type, "stage2")
        os.makedirs(stage2_out_dir, exist_ok=True)

        arch_rows = []
        best_arch = None

        # Candidate spaces per model.
        if model_type == "Transformer":
            arch_space = []
            for hc in hidden_channels_candidates:
                for nl in transformer_num_layers_candidates:
                    for nh in transformer_nhead_candidates:
                        if hc % nh != 0:
                            continue
                        arch_space.append((hc, nl, nh))

            if not arch_space:
                raise ValueError("Transformer architecture candidate space is empty. Check candidates.")

            def eval_arch(hc, nl, nh):
                v_list, te_list = [], []
                for s in cur_search_seeds:
                    df, _ = __import__("baselines.common.train", fromlist=["run_baseline_cycle"]).run_baseline_cycle(
                        model_name=model_type,
                        data_full=raw_data,
                        med_feat=med_feat_unif,
                        reac_feat=reac_feat_unif,
                        target_feat=target_feat_unif,
                        herb_feat=herb_feat_unif,
                        seed=s,
                        lr=best_lr,
                        weight_decay=best_wd,
                        hidden_channels=hc,
                        num_conv_layers=num_conv_layers_candidates[0],
                        transformer_num_layers=nl,
                        transformer_nhead=nh,
                        max_epochs=max_epochs_search,
                        patience=patience,
                        eval_every=eval_every,
                    )
                    v_list.append(float(df["v_auc"].mean()))
                    te_list.append(float(df["te_auc"].mean()))
                return float(np.mean(v_list)), float(np.mean(te_list))

            for hc, nl, nh in arch_space:
                print(f"[ARCH_CAND] Transformer hc={hc} trans_layers={nl} nhead={nh} ...")
                v_mean, te_mean = eval_arch(hc, nl, nh)
                row = {"model": model_type, "hidden_channels": hc, "transformer_num_layers": nl, "transformer_nhead": nh,
                       "search_v_auc_mean": v_mean, "search_te_auc_mean": te_mean}
                arch_rows.append(row)
                if best_arch is None or v_mean > best_arch["search_v_auc_mean"]:
                    best_arch = row

        elif model_type in ["LR", "MLP"]:
            arch_space = hidden_channels_candidates
            for hc in arch_space:
                print(f"[ARCH_CAND] {model_type} hidden_channels={hc} ...")
                v_list, te_list = [], []
                from baselines.common.train import run_baseline_cycle

                for s in cur_search_seeds:
                    df, _ = run_baseline_cycle(
                        model_name=model_type,
                        data_full=raw_data,
                        med_feat=med_feat_unif,
                        reac_feat=reac_feat_unif,
                        target_feat=target_feat_unif,
                        herb_feat=herb_feat_unif,
                        seed=s,
                        lr=best_lr,
                        weight_decay=best_wd,
                        hidden_channels=hc,
                        num_conv_layers=num_conv_layers_candidates[0],
                        transformer_num_layers=transformer_num_layers_candidates[0],
                        transformer_nhead=transformer_nhead_candidates[0],
                        max_epochs=max_epochs_search,
                        patience=patience,
                        eval_every=eval_every,
                    )
                    v_list.append(float(df["v_auc"].mean()))
                    te_list.append(float(df["te_auc"].mean()))
                v_mean = float(np.mean(v_list))
                te_mean = float(np.mean(te_list))
                row = {"model": model_type, "hidden_channels": hc, "search_v_auc_mean": v_mean, "search_te_auc_mean": te_mean}
                arch_rows.append(row)
                if best_arch is None or v_mean > best_arch["search_v_auc_mean"]:
                    best_arch = row

        elif model_type in ["XGBoost", "RF"]:
            # Tree baselines ignore architecture hyper-parameters (hidden_channels/conv layers).
            best_arch = {"model": model_type, "arch_skipped": True}
            arch_rows = []

        else:
            # GNN-like models: tune hidden_channels and number of conv layers.
            from baselines.common.train import run_baseline_cycle

            for hc in hidden_channels_candidates:
                for nconv in num_conv_layers_candidates:
                    print(f"[ARCH_CAND] {model_type} hc={hc} num_conv_layers={nconv} ...")
                    v_list, te_list = [], []
                    for s in cur_search_seeds:
                        df, _ = run_baseline_cycle(
                            model_name=model_type,
                            data_full=raw_data,
                            med_feat=med_feat_unif,
                            reac_feat=reac_feat_unif,
                            target_feat=target_feat_unif,
                            herb_feat=herb_feat_unif,
                            seed=s,
                            lr=best_lr,
                            weight_decay=best_wd,
                            hidden_channels=hc,
                            num_conv_layers=nconv,
                            transformer_num_layers=transformer_num_layers_candidates[0],
                            transformer_nhead=transformer_nhead_candidates[0],
                            max_epochs=max_epochs_search,
                            patience=patience,
                            eval_every=eval_every,
                        )
                        v_list.append(float(df["v_auc"].mean()))
                        te_list.append(float(df["te_auc"].mean()))
                    v_mean = float(np.mean(v_list))
                    te_mean = float(np.mean(te_list))
                    row = {"model": model_type, "hidden_channels": hc, "num_conv_layers": nconv,
                           "search_v_auc_mean": v_mean, "search_te_auc_mean": te_mean}
                    arch_rows.append(row)
                    if best_arch is None or v_mean > best_arch["search_v_auc_mean"]:
                        best_arch = row

        if best_arch is None:
            raise RuntimeError(f"Stage 2 failed to find best architecture for model {model_type}")

        # Stage-2 final eval with best arch.
        from baselines.common.train import run_baseline_cycle
        v_eval_list, te_eval_list = [], []
        seed_dfs = []
        for s in cur_eval_seeds:
            if model_type == "Transformer":
                hc = int(best_arch["hidden_channels"])
                nl = int(best_arch["transformer_num_layers"])
                nh = int(best_arch["transformer_nhead"])
                df, _ = run_baseline_cycle(
                    model_name=model_type,
                    data_full=raw_data,
                    med_feat=med_feat_unif,
                    reac_feat=reac_feat_unif,
                    target_feat=target_feat_unif,
                    herb_feat=herb_feat_unif,
                    seed=s,
                    lr=best_lr,
                    weight_decay=best_wd,
                    hidden_channels=hc,
                    num_conv_layers=num_conv_layers_candidates[0],
                    transformer_num_layers=nl,
                    transformer_nhead=nh,
                    max_epochs=max_epochs_eval,
                    patience=patience,
                    eval_every=eval_every,
                )
            elif model_type in ["LR", "MLP"]:
                hc = int(best_arch["hidden_channels"])
                df, _ = run_baseline_cycle(
                    model_name=model_type,
                    data_full=raw_data,
                    med_feat=med_feat_unif,
                    reac_feat=reac_feat_unif,
                    target_feat=target_feat_unif,
                    herb_feat=herb_feat_unif,
                    seed=s,
                    lr=best_lr,
                    weight_decay=best_wd,
                    hidden_channels=hc,
                    num_conv_layers=num_conv_layers_candidates[0],
                    transformer_num_layers=transformer_num_layers_candidates[0],
                    transformer_nhead=transformer_nhead_candidates[0],
                    max_epochs=max_epochs_eval,
                    patience=patience,
                    eval_every=eval_every,
                )
            elif model_type in ["XGBoost", "RF"]:
                df, _ = run_baseline_cycle(
                    model_name=model_type,
                    data_full=raw_data,
                    med_feat=med_feat_unif,
                    reac_feat=reac_feat_unif,
                    target_feat=target_feat_unif,
                    herb_feat=herb_feat_unif,
                    seed=s,
                    lr=best_lr,
                    weight_decay=best_wd,
                    hidden_channels=hidden_channels_candidates[0],
                    num_conv_layers=num_conv_layers_candidates[0],
                    transformer_num_layers=transformer_num_layers_candidates[0],
                    transformer_nhead=transformer_nhead_candidates[0],
                    max_epochs=max_epochs_eval,
                    patience=patience,
                    eval_every=eval_every,
                )
            else:
                hc = int(best_arch["hidden_channels"])
                nconv = int(best_arch["num_conv_layers"])
                df, _ = run_baseline_cycle(
                    model_name=model_type,
                    data_full=raw_data,
                    med_feat=med_feat_unif,
                    reac_feat=reac_feat_unif,
                    target_feat=target_feat_unif,
                    herb_feat=herb_feat_unif,
                    seed=s,
                    lr=best_lr,
                    weight_decay=best_wd,
                    hidden_channels=hc,
                    num_conv_layers=nconv,
                    transformer_num_layers=transformer_num_layers_candidates[0],
                    transformer_nhead=transformer_nhead_candidates[0],
                    max_epochs=max_epochs_eval,
                    patience=patience,
                    eval_every=eval_every,
                )

            seed_dfs.append(df)
            v_eval_list.append(float(df["v_auc"].mean()))
            te_eval_list.append(float(df["te_auc"].mean()))

        v_mean_eval = float(np.mean(v_eval_list))
        te_mean_eval = float(np.mean(te_eval_list))
        te_std_eval = float(np.std(te_eval_list, ddof=1)) if len(te_eval_list) > 1 else 0.0

        best_arch_eval_res = {
            "model": model_type,
            "dosage_mode": dosage_mode,
            "final_best": {
                "lr": best_lr,
                "weight_decay": best_wd,
                "best_arch": best_arch,
            },
            "selection_metric": "val_v_auc",
            "selection_phase": "stage2",
            "early_stop": {
                "patience": patience,
                "eval_every": eval_every,
            },
            "epochs": {
                "max_epochs_search": max_epochs_search,
                "max_epochs_eval": max_epochs_eval,
            },
            "seeds": {
                "search_seeds": cur_search_seeds,
                "eval_seeds": cur_eval_seeds,
            },
            "eval": {
                "eval_v_auc_mean": v_mean_eval,
                "eval_te_auc_mean": te_mean_eval,
                "eval_te_auc_std": te_std_eval,
            },
        }

        # Save stage2 outputs
        csv_path_stage2 = os.path.join(stage2_out_dir, f"arch_search_{model_type}.csv")
        if arch_rows:
            pd.DataFrame(arch_rows).to_csv(csv_path_stage2, index=False, encoding="utf-8-sig")
        json_path_stage2 = os.path.join(stage2_out_dir, f"best_arch_{model_type}.json")
        import json

        with open(json_path_stage2, "w", encoding="utf-8") as f:
            json.dump(best_arch_eval_res, f, ensure_ascii=False, indent=2)

        all_best[model_type] = best_arch_eval_res
        all_res_for_report[model_type] = seed_dfs

    best_json_all = os.path.join(out_root, "best_hparams_all_models.json")
    import json

    with open(best_json_all, "w", encoding="utf-8") as f:
        json.dump(all_best, f, ensure_ascii=False, indent=2)

    # ===== Final report printing (replicate screenshot style) =====
    try:
        from scipy import stats
        import numpy as np

        def mean_across_seeds(model_name: str, col: str) -> float:
            return float(np.mean([df[col].mean() for df in all_res_for_report[model_name]]))

        def std_across_seeds(model_name: str, col: str) -> float:
            arr = [df[col].mean() for df in all_res_for_report[model_name]]
            return float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

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

        print("\n" + "=" * 12 + " Model Comparison " + "=" * 12)
        header = (
            f"{'Model':<12}"
            f"| {'Tr-AUC':<8} "
            f"| {'Val-AUC':<8} "
            f"| {'Te-AUC (±Std)':<14} "
            f"| {'Te-PR':<7} "
            f"| {'Te-Precision':<14} "
            f"| {'Te-Recall':<9} "
            f"| {'Te-ACC':<7} "
            f"| {'Te-F1':<7} "
            f"| {'Te-MCC':<7}"
        )
        print(header)
        print("-" * len(header))

        for b in baselines_order:
            if b not in all_res_for_report:
                continue
            tr_auc = mean_across_seeds(b, "tr_auc")
            v_auc = mean_across_seeds(b, "v_auc")
            te_auc_mean = mean_across_seeds(b, "te_auc")
            te_auc_std = std_across_seeds(b, "te_auc")
            te_pr = mean_across_seeds(b, "te_pr")
            te_precision = mean_across_seeds(b, "te_precision")
            te_recall = mean_across_seeds(b, "te_recall")
            te_acc = mean_across_seeds(b, "te_acc")
            te_f1 = mean_across_seeds(b, "te_f1")
            te_mcc = mean_across_seeds(b, "te_mcc")

            print(
                f"{b:<12}"
                f"| {tr_auc:.4f} "
                f"| {v_auc:.4f} "
                f"| {te_auc_mean:.4f}±{te_auc_std:.4f} "
                f"| {te_pr:.4f} "
                f"| {te_precision:.4f} "
                f"| {te_recall:.4f} "
                f"| {te_acc:.4f} "
                f"| {te_f1:.4f} "
                f"| {te_mcc:.4f}"
            )

        # Significance analysis (GNN only)
        gnn_models = ["GCN", "SAGE", "GIN", "RGCN", "HGT", "RGAT", "GraphEdgeDosage"]
        gnn_models = [m for m in gnn_models if m in all_res_for_report]
        if gnn_models:
            te_means = {m: mean_across_seeds(m, "te_auc") for m in gnn_models}
            ref_model = max(te_means, key=te_means.get)

            print("\n" + "=" * 10 + " Significance Analysis " + "=" * 10)
            print(f"[INFO] Ref Model (GNN only): {ref_model}")
            print(f"{'Model':<12} | {'Te-AUC':<8} | {'p-value':<10} | {'Sig.'}")
            print("-" * 55)

            ref_arr = [df["te_auc"].mean() for df in all_res_for_report[ref_model]]
            for m in gnn_models:
                mean_m = mean_across_seeds(m, "te_auc")
                if m == ref_model:
                    print(f"{m:<12} | {mean_m:.4f} | {'-':<10} | {'Ref'}")
                else:
                    arr_m = [df["te_auc"].mean() for df in all_res_for_report[m]]
                    _, p_val = stats.ttest_rel(ref_arr, arr_m)
                    sig = "*" if p_val < 0.05 else ""
                    print(f"{m:<12} | {mean_m:.4f} | {p_val:.4f}     | {sig}")
    except Exception as e:
        print(f"[WARN] Failed to print final report: {e}")

    print(f"\n[DONE] Grid search finished. Outputs saved to: {out_root}")
    print(f"[DONE] Best evals saved: {best_json_all}")


if __name__ == "__main__":
    main()

