# FAFW-Conv

Research code for the FAFW-Conv manuscript. The repository contains the audited implementation and experiment entry points for multivariate forecasting, UEA time-series classification, and reconstruction-based pointwise anomaly detection.

## Repository layout

- `models/`, `layers/`, `exp/`, `data_provider/`, and `utils/`: model and task implementation.
- `scripts/`: all runnable Python entry points and launch scripts are kept together here.
- `requirements.txt`: Python dependencies.

This code is based on the official [ModernTCN implementation](https://github.com/luodhhh/ModernTCN) and retains its MIT license notice. See `LICENSE` and `NOTICE.md`.

## Installation

Use Python 3.10 or later, then install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Install a PyTorch build appropriate for your CPU/GPU environment if the default pip package is unsuitable.

## Data

Datasets are not included. Download them from their original sources and keep them outside version control. The long-term forecasting launch scripts expect the ETT CSV files under `./all_six_datasets/dataset/` when run from the repository root.

For classification, provide the UEA `.ts` files under a dataset-specific directory, for example:

```text
UEA/UWaveGestureLibrary/UWaveGestureLibrary_TRAIN.ts
UEA/UWaveGestureLibrary/UWaveGestureLibrary_TEST.ts
```

For anomaly detection, provide `train.npy`, `test.npy`, and `test_label.npy` (or equivalent CSV files) with arrays shaped `[time, features]` and point labels encoded as 0/1. SMD's per-machine text files are also supported; run each machine separately.

## Running experiments

Run commands from the repository root. Forecasting options are exposed by `scripts/run.py`; the included shell launch scripts for ETT, Weather, Flight, Exchange, ILI, Solar, Traffic, ECL, and PEMS are all in `scripts/`.

```bash
python scripts/run.py --help
bash scripts/ETTh1.sh
```

Classification and anomaly detection use the shared entry point:

```bash
python scripts/run_multitask.py --task classification --dataset UWaveGestureLibrary --data_root ./UEA --epochs 100 --batch_size 32 --seed 2021
python scripts/run_multitask.py --task anomaly --dataset SMD --data_root ./anomaly_data/SMD --window 100 --stride 50 --epochs 100 --batch_size 32 --seed 2021
```

The anomaly-detection entry point reports pointwise precision, recall, and F1 without benchmark-specific point adjustment. Threshold selection, preprocessing, data splits, seeds, and event-level evaluation must be specified when comparing with published benchmark scores.

## Relationship to the manuscript results

The manuscript distinguishes historical benchmark values attributed to an earlier model configuration from newly controlled comparisons using the fixed-order implementation in this repository. The historical values are not claimed to be reproduced by this code. The current controlled forecasting/classification analyses and the limited ETTh1 finite-lag ablation are described separately in the manuscript.

## Citation

Please cite the FAFW-Conv manuscript when using this code. For the underlying convolutional backbone, also cite:

> Donghao Luo and Xue Wang. ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis. ICLR 2024.

Official implementation: https://github.com/luodhhh/ModernTCN
