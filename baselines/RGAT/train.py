import os
import sys

# Allow "direct run" (python baselines/RGAT/train.py) to import baselines package.
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import numpy as np

from baselines.common.data import DEVICE, load_tcm_full_data, prepare_unified_features, seed_everything
from baselines.common.train import run_baseline_cycle

import baselines.RGAT.config as cfg


def main() -> None:
    seed_everything(60)
    raw_data, _ = load_tcm_full_data(use_dosage=cfg.USE_DOSAGE)
    med_feat_unif, reac_feat_unif, target_feat_unif, herb_feat_unif = prepare_unified_features(raw_data)

    print(f"[INFO] RGAT DEVICE={DEVICE} BASE_PATH can be set via GNN02_BASE_PATH")

    dfs = []
    for s in cfg.SEEDS:
        df, _ = run_baseline_cycle(
            cfg.MODEL_TYPE,
            raw_data,
            med_feat_unif,
            reac_feat_unif,
            target_feat_unif,
            herb_feat_unif,
            seed=s,
            lr=cfg.LR,
            weight_decay=cfg.WEIGHT_DECAY,
            hidden_channels=cfg.HIDDEN_CHANNELS,
            max_epochs=cfg.MAX_EPOCHS,
            patience=cfg.PATIENCE,
            eval_every=cfg.EVAL_EVERY,
        )
        dfs.append(df)
        print(f"[SEED {s}] done: val_auc={df['v_auc'].mean():.4f}, te_auc={df['te_auc'].mean():.4f}")

    te_auc_means = [df["te_auc"].mean() for df in dfs]
    v_auc_means = [df["v_auc"].mean() for df in dfs]
    print(
        f"[RESULT] RGAT val_auc={np.mean(v_auc_means):.4f} te_auc={np.mean(te_auc_means):.4f}±{np.std(te_auc_means, ddof=1) if len(te_auc_means)>1 else 0.0:.4f}"
    )


if __name__ == "__main__":
    main()

