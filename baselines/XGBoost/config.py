MODEL_TYPE = "XGBoost"

USE_DOSAGE = True

SEEDS = [42, 43, 44]
HIDDEN_CHANNELS = 64  # unused for XGBoost, kept for unified interface
LR = 1e-3  # unused for XGBoost branch
WEIGHT_DECAY = 1e-4  # unused for XGBoost branch

MAX_EPOCHS = 1000  # unused for XGBoost branch (n_estimators fixed in common/train.py)
PATIENCE = 20  # XGBoost: early_stopping_rounds on validation logloss
EVAL_EVERY = 5  # unused for XGBoost branch

