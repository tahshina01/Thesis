# EAFL — Efficient Asynchronous Federated Learning

A paper-faithful implementation of **EAFL** from:

> *"Towards Efficient Asynchronous Federated Learning in Heterogeneous Edge Environments"*, IEEE INFOCOM 2024.

---

## Requirements

- Python 3.9 or later
- pip

---

## Installation

### 1. Clone or copy the project files

```bash
# If using git:
git clone https://github.com/tahshina01/Thesis.git
cd Thesis

# Or just place all .py files in the same directory.
```

### 2. (Recommended) Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate        # Linux / macOS
.venv\Scripts\activate           # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **GPU users:** Install PyTorch with CUDA support from [pytorch.org](https://pytorch.org/get-started/locally/) before running the above, e.g.:
> ```bash
> pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
> ```

---

## Usage

### Quick start
run the following commands

```bash
python experiment_eafl_only_local.py --dataset mnist --split t1
python experiment_eafl_only_local.py --dataset mnist --split t2
python experiment_eafl_only_local.py --dataset cifar10 --split t1
python experiment_eafl_only_local.py --dataset cifar10 --split t2
```


### Command-line arguments

| Argument | Choices | Default | Description |
|---|---|---|---|
| `--dataset` | `mnist`, `cifar10` | `mnist` | Dataset to use |
| `--split` | `t1`, `t2` | `t2` | Non-IID partition type |
| `--seed` | integer | `42` | Global random seed |
| `--rounds` | integer | *(paper preset)* | Override number of training rounds |

### Examples

```bash
# MNIST with T1 Non-IID split, 500 rounds (quick test)
python experiment_eafl_only_local.py --dataset mnist --split t1 --rounds 500

# CIFAR-10 with T2 Non-IID split, paper default rounds
python experiment_eafl_only_local.py --dataset cifar10 --split t2

# Reproducibility: different seed
python experiment_eafl_only_local.py --dataset mnist --split t2 --seed 123
```

---

## Paper Hyperparameter Presets

The experiment file ships with four ready-to-use presets matching Table VI-A of the paper:

| Preset | Dataset | Split | Clusters (N) | φ | Rounds (T) | LR |
|---|---|---|---|---|---|---|
| `MNIST_T1_CONFIG` | MNIST | T1 (ε=0.04) | 5 | 0.1 | 1600 | 0.005 |
| `MNIST_T2_CONFIG` | MNIST | T2 (L=1) | 5 | 0.1 | 1600 | 0.005 |
| `CIFAR10_T1_CONFIG` | CIFAR-10 | T1 (ε=0.04) | 15 | 0.2 | 10000 | 0.0005 |
| `CIFAR10_T2_CONFIG` | CIFAR-10 | T2 (L=1) | 15 | 0.2 | 10000 | 0.0005 |

---

## Non-IID Data Splits

| Type | Description | Key parameter |
|---|---|---|
| **T1** | ε-fraction IID + (1-ε) sort-and-partition | `t1_epsilon=0.04` (paper default) |
| **T2** | Each client receives exactly L label types | `t2_num_labels_per_client=1` (hardest setting) |

---

## Outputs

Results are written to the `results_eafl/` directory:

- **`<run_id>_rounds.csv`** — per-round metrics: accuracy, loss, staleness, simulated wall-clock time, cluster summary.
- **`<run_id>_summary.csv`** — final summary: best accuracy, best round, total time, hyperparameters.
- **`eafl_only_local.txt`** — full console log.

---

## Data Download

Datasets are downloaded automatically on first run via `torchvision` and cached in `./data/`. No manual download is needed.

---

## Device Selection

The code automatically uses CUDA if available, otherwise falls back to CPU. No configuration is required.

---

## Customising a Run

To modify hyperparameters programmatically, import and edit a config preset:

```python
import copy
from experiment_eafl_only_local import MNIST_T2_CONFIG, run

config = copy.deepcopy(MNIST_T2_CONFIG)
config.rounds       = 200          # quick test
config.num_clusters = 10           # more clusters
config.phi          = 0.2          # larger participant fraction
config.seed         = 7

results = run(config)
print(results)
```

---

## Citation

If you use this code, please cite the original paper:

```
@inproceedings{eafl2024,
  title     = {Towards Efficient Asynchronous Federated Learning in Heterogeneous Edge Environments},
  booktitle = {IEEE INFOCOM},
  year      = {2024}
}
```

## Overview

EAFL introduces two core mechanisms on top of standard federated learning:

- **GDC** (Gradient similarity-based Dynamic Clustering) — groups clients by cosine similarity of their local gradients using K-Means, re-run every `R` rounds.
- **SAA** (Staleness-aware semi-Asynchronous intra-cluster Aggregation) — each cluster selects its fastest `φ` fraction of clients and aggregates their gradients, down-weighting stale updates.
- **DSA** (Data size-aware Synchronous inter-cluster Aggregation) — the server aggregates one gradient vector per cluster, weighted by total cluster data volume.

---

## Project Structure

```
.
├── experiment_eafl_only_local.py   # Main training loop (Algorithm 1)
├── client.py                       # Edge device: local SGD + pseudo-gradient
├── server_clean.py                 # Server: GDC clustering + DSA aggregation
├── models.py                       # MNISTModel, LeNetCIFAR, SimpleCNN
├── utils.py                        # Dataset loading, Non-IID splits, SAA
├── requirements.txt
└── README.md
```

---
