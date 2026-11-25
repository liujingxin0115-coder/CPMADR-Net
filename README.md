# CPMADR-Net

CPMADR-Net is a dose-aware heterogeneous graph model for predicting adverse
drug reactions of traditional Chinese medicines. The repository contains the
main model, experiment scripts, baseline implementations,
and ablation studies.

## Repository layout

```text
CPMADR-Net/
├── cpmadr_net/           # Model, training, and split utilities
├── scripts/              # Main experiment entry points
├── baselines/            # Baseline implementations
├── ablation/             # Path-readout ablation experiments
├── preprocessing/        # Offline table-building CLIs
├── configs/              # Best hyperparameters
├── BERT/                 # Feature-extraction scripts and tokenizer configs
├── data/graph/           # ID-only graph tables
├── unbalanced/           # Five-fold CPM-level splits
├── processed_data_package/ # Processed tables and mappings
└── supplementary_results/ # Supplementary result tables
```

## Installation

Use Python 3.10 or newer. Install a PyTorch build compatible with the local
CUDA runtime, then install the remaining dependencies:

```bash
python -m pip install -r requirements.txt
```

## Main experiments

Set the local data paths, then run commands from the repository root:

```bash
# CPMADR-Net, balanced negatives (1:1)
python scripts/train_balanced.py

# CPMADR-Net, imbalanced negatives (1:10)
python scripts/train_imbalanced.py

# Medicine-degree stratified evaluation
python scripts/degree_stratified.py

# Export path-attention evidence (use --help for case arguments)
python scripts/export_path_evidence.py --help
```

Baseline experiments:

```bash
python baselines/run_best_hparams_baseline_cycle_dosage_unique_eval_neg.py
python baselines/run_best_hparams_baseline_cycle_dosage_unique_eval_neg_ratio1_10.py
```

Ablation:

```bash
python ablation/run_path_readout_ablation.py
```

## Data

The repository includes integer-ID graph tables in `data/graph/` and tokenizer
configs under `BERT/`.

Processed study tables and mappings are in [processed_data_package](processed_data_package/README.md).
Supplementary result files are named by table number and title; see [supplementary_results](supplementary_results/README.md).

## License

Project source code is released under the MIT License. Third-party models and
datasets are governed by their respective licenses.

For further information, contact the author at spongebob0328@163.com.
