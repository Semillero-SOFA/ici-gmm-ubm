# Nyquist-WDM Scenario Classification with GMM

This project trains Gaussian Mixture Models (GMMs) to distinguish between **overlapping** and **non-overlapping** spectral scenarios in Nyquist-WDM systems, using constellation diagram data.

## 🎯 Objective

Train a GMM (Universal Background Model - UBM) using only **overlapping** scenarios, then evaluate it with **non-overlapping** scenarios. The idea is:

- **High log-likelihood** for overlapping cases.
- **Low log-likelihood** for non-overlapping cases.

## 📁 Project Structure

project-root/
│
├── src/ # All source code
│ ├── main.py # Entry point for running the full pipeline
│ ├── utils.py # Configuration, logging, and helper functions
│ ├── data_manager.py # Data loading, sampling, checkpoint and result I/O
│ └── gmm_ubm.py # GMM training and evaluation
│
├── checkpoints/ # Intermediate results for resumability
├── logs/ # Logging output
└── README.md # This file

## 🧩 Modules Overview

### `utils.py`

- Central configuration file: paths, constants, seeds.
- Logger setup for tracking progress.

### `data_manager.py`

- Reads database using **polars**.
- Extracts *spectral spacing* and *optical signal to noise ratio (OSNR)* from filenames using **regex**.
- Randomly samples a fixed number of symbols per scenario (with seed for reproducibility).
- Handles saving and loading **checkpoints** and **results**.

### `gmm_ubm.py`

- Trains a GMM using scikit-learn.
- Computes log-likelihood on validation data.
- Returns summary statistics (mean, std of log-likelihood).

### `main.py`

- Orchestrates the full training/evaluation loop:
  - Varies GMM components: 2, 4, 8, 16, 32.
  - Varies sample sizes: 1k, 5k, 10k, 20k, 50k.
  - Uses a fixed random seed for reproducibility.
- Stores checkpoints in `checkpoints/`. and logs in `logs/`.

## ⚙️ Configuration

- Database path is defined in `utils.py`.
- Logging configuration also lives in `utils.py`.
- Sample size and number of components are passed to each run as parameters.

## 🧪 Experimental Parameters

- **GMM components**: `[2, 4, 8, 16, 32]`
- **Sample sizes**: `[1000, 5000, 10000, 20000, 50000]`
- **Seed**: Customizable, default is `15`.

## 🛑 Checkpoint System

- Each combination of (n_components, sample_size, seed) is stored in the `checkpoints/` directory.
- If a result exists, it is reused without recomputation.
- Enables safe interruption/resumption of long-running experiments.

## 🚀 Run the Pipeline

From the project root:

```bash
python src/main.py
