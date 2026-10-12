MODEL_TYPE = "RF"

USE_DOSAGE = True

SEEDS = [42, 43, 44]
HIDDEN_CHANNELS = 64  # unused for RF, kept for unified interface
LR = 1e-3  # unused for RF branch
WEIGHT_DECAY = 1e-4  # unused for RF branch

MAX_EPOCHS = 1000  # unused for RF branch
PATIENCE = 20  # unused for RF branch
EVAL_EVERY = 5  # unused for RF branch
