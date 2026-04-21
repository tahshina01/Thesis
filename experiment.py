"""
experiment.py — EAFL vs FedAsync Comparison

Runs both algorithms on the same dataset, same partition, same test loader,
and logs results to the same CSV structure for direct comparison.

Usage
-----
# MNIST T2, compare both algorithms, 500 rounds each
python experiment.py --dataset mnist --split t2 --rounds 500

# CIFAR-10 T1
python experiment.py --dataset cifar10 --split t1 --rounds 1000

# Run only one algorithm
python experiment.py --dataset mnist --split t1 --algo eafl
python experiment.py --dataset mnist --split t1 --algo fedasync

FedAsync staleness strategies tested (matching paper Section 5.2):
  FedAsync+Const  — s(t-τ) = 1            (α_t = α, no adaptation)
  FedAsync+Poly   — s(t-τ) = (t-τ+1)^-a  (a=0.5)
  FedAsync+Hinge  — s(t-τ) = hinge fn     (a=10, b=4 for MNIST; a=10, b=2 for CIFAR)

Papers
------
EAFL   : "Towards Efficient Asynchronous Federated Learning in Heterogeneous
          Edge Environments", IEEE INFOCOM 2024.
FedAsync: "Asynchronous Federated Optimization", OPT2020 (arXiv:1903.03934v5)
"""

import copy
import csv
import json
import math
import os
import random
import sys
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from client import Client
from models import SimpleCNN, LeNetCIFAR, MNISTModel
from server_clean import Server
from utils import (
    get_dataset,
    split_non_iid_t1,
    split_non_iid_t2,
    saa_cluster_gradient,
)


# ─────────────────────────────────────────────────────────────────────────────
# Logging to file + terminal
# ─────────────────────────────────────────────────────────────────────────────

class Logger:
    def __init__(self, filename="experiment.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


# ─────────────────────────────────────────────────────────────────────────────
# Model registry
# ─────────────────────────────────────────────────────────────────────────────

MODEL_REGISTRY = {
    "MNISTModel": MNISTModel,
    "LeNetCIFAR": LeNetCIFAR,
    "SimpleCNN":  SimpleCNN,
}


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 1 — EAFL (unchanged from experiment_eafl.py)
# ═════════════════════════════════════════════════════════════════════════════

@dataclass
class EAFLConfig:
    experiment_name: str = "eafl_run"
    dataset_name:    str = "mnist"
    seed:            int = 42
    output_dir:      str = "results"

    num_clients:  int   = 100
    num_clusters: int   = 5
    phi:          float = 0.1
    r_clustering: int   = 100
    rounds:       int   = 1600
    epochs:       int   = 1
    client_lr:    float = 0.005
    server_lr:    float = 0.005

    non_iid_type:             str   = "t1"
    t1_epsilon:               float = 0.04
    t2_num_labels_per_client: int   = 1

    model_name: str            = "MNISTModel"
    model_args: Dict[str, Any] = field(default_factory=lambda: {"num_channels": 1, "img_size": 28})

    speed_slow_fraction:   float = 0.20
    speed_medium_fraction: float = 0.50
    speed_fast_fraction:   float = 0.30
    speed_slow_range:   Tuple[float, float] = (0.10, 0.20)
    speed_medium_range: Tuple[float, float] = (0.30, 0.70)
    speed_fast_range:   Tuple[float, float] = (0.80, 1.00)

    local_batch_size:                    int   = 32
    compute_time_per_batch_sec:          float = 0.02
    base_bandwidth_mb_s:                 float = 10.0
    server_aggregation_time_sec:         float = 0.01
    server_clustering_time_per_client_sec: float = 0.001
    time_jitter_scale:                   float = 0.4

    @property
    def model_class(self):
        return MODEL_REGISTRY[self.model_name]


# ── Paper presets ──────────────────────────────────────────────────────────

MNIST_T1_CONFIG = EAFLConfig(
    experiment_name="eafl_mnist_t1", dataset_name="mnist",
    num_clients=100, num_clusters=5, phi=0.1, r_clustering=100,
    rounds=1600, epochs=1, client_lr=0.005, server_lr=0.005,
    non_iid_type="t1", t1_epsilon=0.04,
    model_name="MNISTModel", model_args={"num_channels": 1, "img_size": 28},
)
MNIST_T2_CONFIG = EAFLConfig(
    experiment_name="eafl_mnist_t2", dataset_name="mnist",
    non_iid_type="t2", t2_num_labels_per_client=1,
    num_clusters=5, phi=0.1, r_clustering=100, rounds=1600,
    client_lr=0.005, server_lr=0.005,
    model_name="MNISTModel", model_args={"num_channels": 1, "img_size": 28},
)
CIFAR10_T1_CONFIG = EAFLConfig(
    experiment_name="eafl_cifar10_t1", dataset_name="cifar10",
    num_clients=100, num_clusters=15, phi=0.2, r_clustering=1000,
    rounds=10000, epochs=1, client_lr=0.0005, server_lr=0.0005,
    non_iid_type="t1", t1_epsilon=0.04,
    model_name="LeNetCIFAR", model_args={"num_channels": 3, "img_size": 32},
)
CIFAR10_T2_CONFIG = EAFLConfig(
    experiment_name="eafl_cifar10_t2", dataset_name="cifar10",
    num_clusters=15, phi=0.2, r_clustering=1000, rounds=10000,
    client_lr=0.0005, server_lr=0.0005,
    non_iid_type="t2", t2_num_labels_per_client=1,
    model_name="LeNetCIFAR", model_args={"num_channels": 3, "img_size": 32},
)


# ── EAFL helpers (identical to experiment_eafl.py) ────────────────────────

def create_clients(trainset, client_indices, device, config: EAFLConfig):
    rng = np.random.default_rng(config.seed + 100)
    clients = []
    for cid in range(len(client_indices)):
        r = rng.random()
        if r < config.speed_slow_fraction:
            lo, hi = config.speed_slow_range
        elif r < config.speed_slow_fraction + config.speed_medium_fraction:
            lo, hi = config.speed_medium_range
        else:
            lo, hi = config.speed_fast_range
        speed = float(rng.uniform(lo, hi))
        clients.append(Client(cid, trainset, client_indices[cid], device=device, system_speed=speed))
    return clients


def estimate_steps(client: Client, epochs: int, batch_size: int) -> int:
    n_batches = math.ceil(client.data_size / batch_size) if client.data_size > 0 else 1
    return max(1, n_batches * epochs)


def estimate_completion_time(client: Client, steps: int, model_size_mb: float,
                              config: EAFLConfig, rng: np.random.Generator) -> float:
    speed        = max(client.system_speed, 1e-3)
    compute_time = config.compute_time_per_batch_sec * steps / speed
    comm_time    = (2.0 * model_size_mb) / max(config.base_bandwidth_mb_s * speed, 1e-6)
    jitter       = float(rng.lognormal(mean=0.0, sigma=config.time_jitter_scale))
    return compute_time + comm_time + jitter


def model_size_mb(state_dict: dict) -> float:
    total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
    return total_bytes / (1024.0 * 1024.0)


def clone_state(sd: dict) -> dict:
    return {k: v.detach().clone() for k, v in sd.items()}


def evaluate_model(model: nn.Module, test_loader: DataLoader,
                   device: torch.device) -> Tuple[float, float]:
    model.eval()
    criterion = nn.CrossEntropyLoss()
    total, correct, loss_sum = 0, 0, 0.0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            out      = model(images)
            loss_sum += criterion(out, labels).item()
            total    += labels.size(0)
            correct  += (out.argmax(dim=1) == labels).sum().item()
    acc  = 100.0 * correct / total if total > 0 else 0.0
    loss = loss_sum / len(test_loader) if test_loader else 0.0
    return acc, loss


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 2 — FedAsync (Algorithm 1, arXiv:1903.03934v5)
# ═════════════════════════════════════════════════════════════════════════════

# ── 2a. Staleness weighting functions s(t-τ) — Section 3 / Remark 2 ────────
#   Requirements: s(0)=1, monotonically decreasing as staleness grows.

def staleness_const(staleness: int, **_) -> float:
    """s(t-τ) = 1  →  FedAsync+Const (no adaptation)."""
    return 1.0


def staleness_poly(staleness: int, a: float = 0.5, **_) -> float:
    """s_a(t-τ) = (t-τ+1)^{-a}.  Paper: a=0.5."""
    return (staleness + 1) ** (-a)


def staleness_hinge(staleness: int, a: float = 10.0, b: float = 4.0, **_) -> float:
    """
    s_{a,b}(t-τ) = 1                      if t-τ ≤ b
                 = 1/(a*(t-τ-b)+1)        otherwise
    Paper: MNIST/CIFAR a=10, b=4; LM b=2.
    """
    if staleness <= b:
        return 1.0
    return 1.0 / (a * (staleness - b) + 1.0)


# ── 2b. FedAsync hyperparameter container ───────────────────────────────────

@dataclass
class FedAsyncConfig:
    """
    FedAsync hyperparameters — Table 1 of the paper.

    Aligned with EAFL config fields so the shared setup code can feed
    the same dataset / client list into both algorithms.
    """
    experiment_name: str = "fedasync_run"
    dataset_name:    str = "mnist"
    seed:            int = 42
    output_dir:      str = "results"

    # FL system
    num_clients: int   = 100           # n  (paper: 100)
    rounds:      int   = 1600          # T  (total server updates)
    H_min:       int   = 50            # Hmin (min local iterations)
    H_max:       int   = 50            # Hmax (max local iterations — set equal for fair comparison)

    # Core hyperparameters
    # gamma / rho mirror client_lr / regularisation in EAFL notation
    gamma: float = 0.005               # γ  learning rate (matched to EAFL client_lr)
    rho:   float = 0.01                # ρ  regularisation weight; must satisfy ρ > µ
    alpha: float = 0.9                 # α  base mixing hyperparameter ∈ (0,1)

    # Staleness
    max_staleness: int = 4             # K  bounded delay (paper: 4 or 16)
    staleness_fn_name: str = "const"   # "const" | "poly" | "hinge"
    poly_a:  float = 0.5               # a for polynomial decay
    hinge_a: float = 10.0              # a for hinge
    hinge_b: float = 4.0              # b for hinge (MNIST paper value)

    # Data heterogeneity — mirrors EAFLConfig so same partition is reused
    non_iid_type:             str   = "t1"
    t1_epsilon:               float = 0.04
    t2_num_labels_per_client: int   = 1

    # Model — must match EAFL to share the same architecture
    model_name: str            = "MNISTModel"
    model_args: Dict[str, Any] = field(
        default_factory=lambda: {"num_channels": 1, "img_size": 28}
    )

    # Batch size for local training
    local_batch_size: int = 32

    @property
    def model_class(self):
        return MODEL_REGISTRY[self.model_name]

    @property
    def staleness_fn(self) -> Callable[[int], float]:
        """Resolve staleness function from name + parameters."""
        if self.staleness_fn_name == "poly":
            a = self.poly_a
            return lambda s: staleness_poly(s, a=a)
        elif self.staleness_fn_name == "hinge":
            a, b = self.hinge_a, self.hinge_b
            return lambda s: staleness_hinge(s, a=a, b=b)
        else:  # "const"
            return staleness_const

    @property
    def delta(self) -> float:
        """Imbalance ratio δ = Hmax/Hmin (Table 1)."""
        return self.H_max / max(self.H_min, 1)


# ── Paper-matched FedAsync presets ──────────────────────────────────────────

FEDASYNC_MNIST_T1_CONST = FedAsyncConfig(
    experiment_name="fedasync_mnist_t1_const",
    dataset_name="mnist", seed=42, rounds=1600,
    gamma=0.005, rho=0.01, alpha=0.9, max_staleness=4,
    staleness_fn_name="const",
    non_iid_type="t1", t1_epsilon=0.04,
    model_name="MNISTModel", model_args={"num_channels": 1, "img_size": 28},
)
FEDASYNC_MNIST_T1_POLY = FedAsyncConfig(
    experiment_name="fedasync_mnist_t1_poly",
    dataset_name="mnist", seed=42, rounds=1600,
    gamma=0.005, rho=0.01, alpha=0.9, max_staleness=4,
    staleness_fn_name="poly", poly_a=0.5,
    non_iid_type="t1", t1_epsilon=0.04,
    model_name="MNISTModel", model_args={"num_channels": 1, "img_size": 28},
)
FEDASYNC_MNIST_T1_HINGE = FedAsyncConfig(
    experiment_name="fedasync_mnist_t1_hinge",
    dataset_name="mnist", seed=42, rounds=1600,
    gamma=0.005, rho=0.01, alpha=0.9, max_staleness=4,
    staleness_fn_name="hinge", hinge_a=10.0, hinge_b=4.0,
    non_iid_type="t1", t1_epsilon=0.04,
    model_name="MNISTModel", model_args={"num_channels": 1, "img_size": 28},
)
FEDASYNC_MNIST_T2_CONST = FedAsyncConfig(
    experiment_name="fedasync_mnist_t2_const",
    dataset_name="mnist", seed=42, rounds=1600,
    gamma=0.005, rho=0.01, alpha=0.9, max_staleness=4,
    staleness_fn_name="const",
    non_iid_type="t2", t2_num_labels_per_client=1,
    model_name="MNISTModel", model_args={"num_channels": 1, "img_size": 28},
)
FEDASYNC_CIFAR10_T1_CONST = FedAsyncConfig(
    experiment_name="fedasync_cifar10_t1_const",
    dataset_name="cifar10", seed=42, rounds=10000,
    gamma=0.0005, rho=0.005, alpha=0.9, max_staleness=4,
    staleness_fn_name="const",
    non_iid_type="t1", t1_epsilon=0.04,
    model_name="LeNetCIFAR", model_args={"num_channels": 3, "img_size": 32},
    hinge_b=4.0,
)
FEDASYNC_CIFAR10_T2_CONST = FedAsyncConfig(
    experiment_name="fedasync_cifar10_t2_const",
    dataset_name="cifar10", seed=42, rounds=10000,
    gamma=0.0005, rho=0.005, alpha=0.9, max_staleness=4,
    staleness_fn_name="const",
    non_iid_type="t2", t2_num_labels_per_client=1,
    model_name="LeNetCIFAR", model_args={"num_channels": 3, "img_size": 32},
    hinge_b=4.0,
)


# ── 2c. FedAsync local update — Algorithm 1 Worker process ──────────────────

def fedasync_local_update(
    model: nn.Module,
    global_snapshot: nn.Module,      # frozen x_τ for regularisation
    dataset: torch.utils.data.Dataset,
    H: int,                          # number of local iterations
    gamma: float,                    # γ  learning rate
    rho: float,                      # ρ  regularisation weight
    device: torch.device,
    batch_size: int = 32,
) -> nn.Module:
    """
    Algorithm 1, Worker process inner loop:

        for h in [H]:
            z ~ D_i
            x_{τ,h} ← x_{τ,h-1} − γ · ∇g_{x_τ}(x_{τ,h-1}; z)

    where  g_{x_τ}(x; z) = f(x; z) + (ρ/2)||x − x_τ||²

    The gradient of the regularised objective is:
        ∇g_{x_τ}(x; z) = ∇f(x; z) + ρ·(x − x_τ)

    Implemented as: standard SGD step on f, then explicit proximal correction.
    """
    model          = model.to(device)
    global_snapshot = global_snapshot.to(device)

    loader      = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    loader_iter = iter(loader)
    optimizer   = optim.SGD(model.parameters(), lr=gamma)
    criterion   = nn.CrossEntropyLoss()

    model.train()
    for _ in range(H):
        # "Randomly sample z^i_{τ,h} ~ D^i"
        try:
            inputs, targets = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            inputs, targets = next(loader_iter)
        inputs, targets = inputs.to(device), targets.to(device)

        # Step 1: SGD on f(x; z)
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

        # Step 2: regularisation correction  x ← x − γ·ρ·(x − x_τ)
        # This is the gradient of (ρ/2)||x − x_τ||² evaluated at current x.
        with torch.no_grad():
            for p, p_ref in zip(model.parameters(), global_snapshot.parameters()):
                p.data -= gamma * rho * (p.data - p_ref.data)

    return model


# ── 2d. FedAsync server update — Algorithm 1 Updater thread ─────────────────

def fedasync_server_update(
    global_model: nn.Module,
    new_model:    nn.Module,
    alpha:        float,
    staleness:    int,
    staleness_fn: Callable[[int], float],
) -> Tuple[nn.Module, float]:
    """
    Algorithm 1, Thread Updater:

        α_t ← α × s(t − τ)           [optional adaptive α]
        x_t ← (1 − α_t)·x_{t-1} + α_t·x_new

    Returns the updated global_model and the effective α_t used.
    """
    # Adaptive mixing (paper Remark 2 / Algorithm 1 "Optional" line)
    alpha_t = alpha * staleness_fn(staleness)

    with torch.no_grad():
        for p_global, p_new in zip(global_model.parameters(), new_model.parameters()):
            p_global.data = (1.0 - alpha_t) * p_global.data + alpha_t * p_new.data

    return global_model, alpha_t


# ── 2e. Staleness simulator ───────────────────────────────────────────────────

def sample_staleness(max_staleness: int) -> int:
    """
    Paper Section 5.2:
    "simulate the asynchrony by randomly sampling the staleness (t-τ)
     from a uniform distribution"
    t − τ  ~  Uniform{0, 1, …, max_staleness}
    """
    return random.randint(0, max_staleness)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 3 — Shared CSV logger
# ═════════════════════════════════════════════════════════════════════════════

class RoundLogger:
    """
    Single CSV logger used by both EAFL and FedAsync runners.
    Columns are the union of both algorithms' metrics; unused fields are
    written as empty strings so the CSV is always machine-readable.
    """

    FIELDS = [
        "algorithm", "run_id", "round",
        # Accuracy / loss
        "accuracy", "loss",
        # EAFL-specific
        "is_clustering_round", "num_clusters",
        "total_participants", "avg_staleness", "max_staleness",
        "simulated_wall_clock_sec", "cumulative_wall_clock_sec",
        # FedAsync-specific
        "staleness_sample", "alpha_t", "worker_id",
    ]

    def __init__(self, output_dir: str, run_id: str):
        os.makedirs(output_dir, exist_ok=True)
        self.run_id = run_id
        self.path   = os.path.join(output_dir, f"{run_id}_rounds.csv")
        with open(self.path, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writeheader()

    def log(self, row: dict):
        """Write a row, filling missing keys with empty string."""
        full_row = {k: row.get(k, "") for k in self.FIELDS}
        with open(self.path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.FIELDS).writerow(full_row)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 4 — EAFL runner (from experiment_eafl.py, adapted to use RoundLogger)
# ═════════════════════════════════════════════════════════════════════════════

class EAFLRunner:
    def __init__(self, config: EAFLConfig,
                 trainset, client_indices, test_loader,
                 device: torch.device):
        self.config      = config
        self.trainset    = trainset
        self.test_loader = test_loader
        self.device      = device

        cfg = config

        # Clients (shared partition passed in — not re-created)
        self.clients = create_clients(trainset, client_indices, device, cfg)

        # Server
        self.server = Server(
            test_loader=test_loader,
            device=device,
            model_args=dict(cfg.model_args),
            model_class=cfg.model_class,
            lr=cfg.server_lr,
        )

        self.steps_by_client = {
            cid: estimate_steps(self.clients[cid], cfg.epochs, cfg.local_batch_size)
            for cid in range(cfg.num_clients)
        }
        self.model_mb = model_size_mb(self.server.global_model.state_dict())

    def run(self, logger: RoundLogger) -> dict:
        set_seed(self.config.seed)
        cfg = self.config

        last_participation_round = [-1] * cfg.num_clients
        last_received_round      = [-1] * cfg.num_clients
        MODEL_HISTORY_WINDOW     = cfg.rounds
        model_history            = {0: clone_state(self.server.global_model.state_dict())}

        cluster_members:    Optional[dict] = None
        cluster_heads:      Optional[dict] = None
        cluster_data_sizes: Optional[dict] = None

        timing_rng = np.random.default_rng(cfg.seed + 77)

        best_accuracy  = -1.0
        best_round     = 0
        final_accuracy = 0.0
        final_loss     = 0.0
        cumulative_wc  = 0.0
        process_start  = time.perf_counter()

        print(f"\n[EAFL] Starting: {cfg.rounds} rounds, N={cfg.num_clusters}, "
              f"φ={cfg.phi}, R={cfg.r_clustering}")
        print("─" * 60)

        for t in range(cfg.rounds):
            is_clustering_round = (cluster_members is None) or (t % cfg.r_clustering == 0)

            # ── Clustering round ───────────────────────────────────────────
            if is_clustering_round:
                grads_list = []
                data_sizes_list = []
                client_times = {}

                for cid in range(cfg.num_clients):
                    model_round = max(0, last_received_round[cid])
                    base_state  = model_history.get(model_round, model_history[max(model_history.keys())])
                    tmp_model   = cfg.model_class(**cfg.model_args)
                    tmp_model.load_state_dict(base_state)
                    _, grad, data_size = self.clients[cid].train(
                        tmp_model, epochs=cfg.epochs, learning_rate=cfg.client_lr
                    )
                    grads_list.append(grad)
                    data_sizes_list.append(data_size)
                    client_times[cid] = estimate_completion_time(
                        self.clients[cid], self.steps_by_client[cid],
                        self.model_mb, cfg, timing_rng
                    )

                (_, cluster_heads, cluster_data_sizes, cluster_members) = (
                    self.server.run_clustering(
                        grads_list=grads_list,
                        data_sizes_list=data_sizes_list,
                        client_ids=list(range(cfg.num_clients)),
                        n_clusters=cfg.num_clusters,
                        seed=cfg.seed + t,
                    )
                )

                cluster_updates  = []
                all_participants = set()
                staleness_values = []

                for cluster_id, members in cluster_members.items():
                    k        = max(1, int(math.ceil(cfg.phi * len(members))))
                    selected = sorted(members, key=lambda c: client_times[c])[:k]
                    all_participants.update(selected)

                    updates = []
                    for cid in selected:
                        model_round = max(0, last_received_round[cid])
                        base_state  = model_history.get(model_round, model_history[max(model_history.keys())])
                        tmp_model   = cfg.model_class(**cfg.model_args)
                        tmp_model.load_state_dict(base_state)
                        _, grad, data_size = self.clients[cid].train(
                            tmp_model, epochs=cfg.epochs, learning_rate=cfg.client_lr
                        )
                        tau = (t - last_participation_round[cid]
                               if last_participation_round[cid] >= 0 else t + 1)
                        staleness_values.append(tau)
                        updates.append({"grads": grad, "data_size": data_size,
                                        "timestamp": last_participation_round[cid]})

                    g_bar = saa_cluster_gradient(updates, t)
                    if g_bar is not None:
                        cluster_updates.append({
                            "gradient":          g_bar,
                            "cluster_data_size": cluster_data_sizes[cluster_id],
                        })

                simulated_round_time = (max(client_times[c] for c in all_participants)
                                        + cfg.server_aggregation_time_sec
                                        + cfg.server_clustering_time_per_client_sec * cfg.num_clients
                                        if all_participants else 0.0)

            # ── Normal round ───────────────────────────────────────────────
            else:
                cluster_updates      = []
                all_participants     = set()
                staleness_values     = []
                completion_times_sel = {}

                for cluster_id, members in cluster_members.items():
                    k = max(1, int(math.ceil(cfg.phi * len(members))))
                    member_times = {
                        cid: estimate_completion_time(
                            self.clients[cid], self.steps_by_client[cid],
                            self.model_mb, cfg, timing_rng
                        )
                        for cid in members
                    }
                    selected = sorted(members, key=lambda c: member_times[c])[:k]
                    all_participants.update(selected)
                    for cid in selected:
                        completion_times_sel[cid] = member_times[cid]

                    updates = []
                    for cid in selected:
                        model_round = max(0, last_received_round[cid])
                        base_state  = model_history.get(model_round, model_history[max(model_history.keys())])
                        tmp_model   = cfg.model_class(**cfg.model_args)
                        tmp_model.load_state_dict(base_state)
                        _, grad, data_size = self.clients[cid].train(
                            tmp_model, epochs=cfg.epochs, learning_rate=cfg.client_lr
                        )
                        tau = (t - last_participation_round[cid]
                               if last_participation_round[cid] >= 0 else t + 1)
                        staleness_values.append(tau)
                        updates.append({"grads": grad, "data_size": data_size,
                                        "timestamp": last_participation_round[cid]})

                    g_bar = saa_cluster_gradient(updates, t)
                    if g_bar is not None:
                        cluster_updates.append({
                            "gradient":          g_bar,
                            "cluster_data_size": cluster_data_sizes[cluster_id],
                        })

                simulated_round_time = (max(completion_times_sel.values()) + cfg.server_aggregation_time_sec
                                        if completion_times_sel else 0.0)

            # ── DSA + model delivery ───────────────────────────────────────
            if cluster_updates:
                self.server.aggregate_cluster_updates(cluster_updates)

            new_state = clone_state(self.server.global_model.state_dict())
            model_history[t] = new_state
            for cid in all_participants:
                last_participation_round[cid] = t
                last_received_round[cid]      = t

            old_key = (t + 1) - MODEL_HISTORY_WINDOW
            if old_key in model_history and old_key != 0:
                del model_history[old_key]

            # ── Evaluate ──────────────────────────────────────────────────
            final_accuracy, final_loss = self.server.evaluate()
            if final_accuracy > best_accuracy:
                best_accuracy = final_accuracy
                best_round    = t + 1

            cumulative_wc += simulated_round_time
            avg_staleness  = float(np.mean(staleness_values)) if staleness_values else 0.0
            max_staleness_val = int(max(staleness_values)) if staleness_values else 0

            logger.log({
                "algorithm":                "EAFL",
                "run_id":                   logger.run_id,
                "round":                    t + 1,
                "accuracy":                 round(final_accuracy, 4),
                "loss":                     round(final_loss, 6),
                "is_clustering_round":      int(is_clustering_round),
                "num_clusters":             len(cluster_members or {}),
                "total_participants":        len(all_participants),
                "avg_staleness":            round(avg_staleness, 3),
                "max_staleness":            max_staleness_val,
                "simulated_wall_clock_sec": round(simulated_round_time, 4),
                "cumulative_wall_clock_sec":round(cumulative_wc, 4),
            })

            if (t + 1) % 100 == 0 or t == 0:
                print(f"  [EAFL]  Round {t+1:>5}/{cfg.rounds}  "
                      f"acc={final_accuracy:.2f}%  loss={final_loss:.4f}  "
                      f"avg_stale={avg_staleness:.1f}"
                      f"{'  [CLUSTER]' if is_clustering_round else ''}")

        total_time = time.perf_counter() - process_start
        print(f"\n[EAFL] Best={best_accuracy:.2f}% @ round {best_round} | "
              f"Final={final_accuracy:.2f}% | Time={total_time/60:.1f} min")

        return dict(best_accuracy=best_accuracy, best_round=best_round,
                    final_accuracy=final_accuracy, final_loss=final_loss,
                    total_time_sec=total_time)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 5 — FedAsync runner
# ═════════════════════════════════════════════════════════════════════════════

class FedAsyncRunner:
    """
    Implements Algorithm 1 (FedAsync) for direct comparison with EAFL.

    Consumes the SAME trainset partition and test_loader that EAFLRunner uses,
    so accuracy numbers are directly comparable.

    The client datasets are Subset objects built from the same client_indices,
    so no data is duplicated or re-partitioned.
    """

    def __init__(self, config: FedAsyncConfig,
                 trainset, client_indices,
                 test_loader: DataLoader,
                 device: torch.device):
        self.cfg         = config
        self.test_loader = test_loader
        self.device      = device

        # Build Subset datasets matching the partition used by EAFL clients
        self.client_datasets = [
            Subset(trainset, client_indices[cid])
            for cid in range(config.num_clients)
        ]

        # Global model — initialised fresh (same architecture as EAFL)
        self.global_model = config.model_class(**config.model_args).to(device)

    # ── Evaluation ────────────────────────────────────────────────────────

    @torch.no_grad()
    def _evaluate(self) -> Tuple[float, float]:
        return evaluate_model(self.global_model, self.test_loader, self.device)

    # ── Main training loop ────────────────────────────────────────────────

    def run(self, logger: RoundLogger) -> dict:
        """
        Execute T global epochs of Algorithm 1.

        Each epoch:
          1. Scheduler selects one worker at random.
          2. Worker trains from (possibly stale) global snapshot using
             regularised SGD for H local iterations.
          3. Updater applies the weighted-average server update with
             adaptive α_t = α × s(t-τ).
          4. Evaluate and log every round.
        """
        set_seed(self.cfg.seed)
        cfg  = self.cfg
        sfn  = cfg.staleness_fn   # resolved staleness function

        best_accuracy  = -1.0
        best_round     = 0
        final_accuracy = 0.0
        final_loss     = 0.0
        process_start  = time.perf_counter()

        print(f"\n[FedAsync+{cfg.staleness_fn_name.capitalize()}]  "
              f"α={cfg.alpha}, K={cfg.max_staleness}, "
              f"γ={cfg.gamma}, ρ={cfg.rho}, "
              f"H={cfg.H_min}-{cfg.H_max}, T={cfg.rounds}")
        print("─" * 60)

        for t in range(1, cfg.rounds + 1):

            # ── Thread Scheduler: trigger one worker ──────────────────────
            # Paper: "Periodically trigger training tasks on some workers"
            # Simulation: uniformly random selection (1 worker per epoch)
            worker_id = random.randint(0, cfg.num_clients - 1)

            # ── Staleness simulation ───────────────────────────────────────
            # Paper Section 5.2: sample (t-τ) ~ Uniform{0,...,K}
            staleness = sample_staleness(cfg.max_staleness)

            # ── Worker receives global model (possibly stale) ─────────────
            # In simulation: we use the current global model as x_τ.
            # The staleness affects α_t (server update weight), not which
            # snapshot the worker sees — this faithfully reproduces the
            # paper's simulation methodology (Section 5.2).
            local_model      = copy.deepcopy(self.global_model)
            global_snapshot  = copy.deepcopy(self.global_model)  # frozen x_τ

            # ── Worker: sample H ∈ [H_min, H_max] and run local update ────
            H = random.randint(cfg.H_min, cfg.H_max)
            local_model = fedasync_local_update(
                model=local_model,
                global_snapshot=global_snapshot,
                dataset=self.client_datasets[worker_id],
                H=H,
                gamma=cfg.gamma,
                rho=cfg.rho,
                device=self.device,
                batch_size=cfg.local_batch_size,
            )

            # ── Thread Updater: server weighted average ────────────────────
            # x_t = (1 - α_t)·x_{t-1} + α_t·x_new
            self.global_model, alpha_t = fedasync_server_update(
                global_model=self.global_model,
                new_model=local_model,
                alpha=cfg.alpha,
                staleness=staleness,
                staleness_fn=sfn,
            )

            # ── Evaluate ──────────────────────────────────────────────────
            final_accuracy, final_loss = self._evaluate()
            if final_accuracy > best_accuracy:
                best_accuracy = final_accuracy
                best_round    = t

            # ── Log ───────────────────────────────────────────────────────
            logger.log({
                "algorithm":       f"FedAsync+{cfg.staleness_fn_name}",
                "run_id":          logger.run_id,
                "round":           t,
                "accuracy":        round(final_accuracy, 4),
                "loss":            round(final_loss, 6),
                "staleness_sample":staleness,
                "alpha_t":         round(alpha_t, 6),
                "worker_id":       worker_id,
            })

            if t % 100 == 0 or t == 1:
                print(f"  [FedAsync+{cfg.staleness_fn_name:5s}]  "
                      f"Round {t:>5}/{cfg.rounds}  "
                      f"acc={final_accuracy:.2f}%  loss={final_loss:.4f}  "
                      f"staleness={staleness}  α_t={alpha_t:.4f}")

        total_time = time.perf_counter() - process_start
        print(f"\n[FedAsync+{cfg.staleness_fn_name}] "
              f"Best={best_accuracy:.2f}% @ round {best_round} | "
              f"Final={final_accuracy:.2f}% | Time={total_time/60:.1f} min")

        return dict(best_accuracy=best_accuracy, best_round=best_round,
                    final_accuracy=final_accuracy, final_loss=final_loss,
                    total_time_sec=total_time)


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 6 — Shared setup + comparison orchestrator
# ═════════════════════════════════════════════════════════════════════════════

def _build_fedasync_configs_for_eafl(eafl_cfg: EAFLConfig,
                                      rounds_override: Optional[int] = None) -> List[FedAsyncConfig]:
    """
    Build FedAsync variants (Const / Poly / Hinge) aligned to an EAFL config:
    same dataset, same split type, same model, same number of rounds.
    This ensures a fair apple-to-apple comparison.
    """
    rounds = rounds_override if rounds_override else eafl_cfg.rounds

    # Map EAFL dataset → FedAsync hyperparameters (paper Table / Sec 5)
    if eafl_cfg.dataset_name == "mnist":
        gamma, rho, alpha, max_stale = eafl_cfg.client_lr, 0.01, 0.9, 4
        hinge_b = 4.0
    else:  # cifar10
        gamma, rho, alpha, max_stale = eafl_cfg.client_lr, 0.005, 0.9, 4
        hinge_b = 4.0

    common = dict(
        dataset_name=eafl_cfg.dataset_name,
        seed=eafl_cfg.seed,
        output_dir=eafl_cfg.output_dir,
        rounds=rounds,
        num_clients=eafl_cfg.num_clients,
        gamma=gamma,
        rho=rho,
        alpha=alpha,
        max_staleness=max_stale,
        non_iid_type=eafl_cfg.non_iid_type,
        t1_epsilon=eafl_cfg.t1_epsilon,
        t2_num_labels_per_client=eafl_cfg.t2_num_labels_per_client,
        model_name=eafl_cfg.model_name,
        model_args=dict(eafl_cfg.model_args),
        local_batch_size=eafl_cfg.local_batch_size,
    )

    return [
        FedAsyncConfig(experiment_name=f"fedasync_{eafl_cfg.dataset_name}_{eafl_cfg.non_iid_type}_const",
                       staleness_fn_name="const", **common),
        FedAsyncConfig(experiment_name=f"fedasync_{eafl_cfg.dataset_name}_{eafl_cfg.non_iid_type}_poly",
                       staleness_fn_name="poly", poly_a=0.5, **common),
        FedAsyncConfig(experiment_name=f"fedasync_{eafl_cfg.dataset_name}_{eafl_cfg.non_iid_type}_hinge",
                       staleness_fn_name="hinge", hinge_a=10.0, hinge_b=hinge_b, **common),
    ]


def run_comparison(
    eafl_config:     EAFLConfig,
    run_eafl:        bool = True,
    run_fedasync:    bool = True,
    rounds_override: Optional[int] = None,
    fedasync_variants: Optional[List[str]] = None,   # e.g. ["const"] to run only Const
) -> dict:
    """
    Full comparison runner.

    1. Loads dataset once.
    2. Partitions data once (same partition for ALL algorithms).
    3. Runs EAFL (if requested).
    4. Runs FedAsync Const/Poly/Hinge (if requested).
    5. All results written to a single shared CSV for easy plotting.

    Parameters
    ----------
    eafl_config      : EAFLConfig — defines dataset, split, model, rounds.
    run_eafl         : whether to run the EAFL algorithm.
    run_fedasync     : whether to run FedAsync variants.
    rounds_override  : override number of rounds for both algorithms (useful for quick tests).
    fedasync_variants: subset of ["const","poly","hinge"] to run. None = run all three.

    Returns
    -------
    dict mapping algorithm name → results dict (best_accuracy, final_accuracy, …)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    cfg = copy.deepcopy(eafl_config)
    if rounds_override:
        cfg.rounds = rounds_override

    # ── 1. Load dataset ───────────────────────────────────────────────────
    print(f"\nLoading {cfg.dataset_name} ...")
    trainset, testset = get_dataset(cfg.dataset_name)
    test_loader = DataLoader(testset, batch_size=256, shuffle=False)

    # ── 2. Partition data (once, shared by all algorithms) ────────────────
    print(f"Partitioning ({cfg.non_iid_type.upper()}, n={cfg.num_clients}) ...")
    set_seed(cfg.seed)   # deterministic partition
    if cfg.non_iid_type == "t1":
        client_indices = split_non_iid_t1(trainset, cfg.num_clients, epsilon=cfg.t1_epsilon)
    else:
        client_indices = split_non_iid_t2(trainset, cfg.num_clients,
                                           num_labels_per_client=cfg.t2_num_labels_per_client)

    sizes = [len(idx) for idx in client_indices]
    print(f"  Partition: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f} samples/client")

    # ── 3. Shared logger ──────────────────────────────────────────────────
    os.makedirs(cfg.output_dir, exist_ok=True)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_id = f"comparison_{cfg.dataset_name}_{cfg.non_iid_type}_seed{cfg.seed}_{ts}"
    logger = RoundLogger(cfg.output_dir, run_id)
    print(f"Results → {logger.path}")

    all_results = {}

    # ── 4. EAFL ───────────────────────────────────────────────────────────
    if run_eafl:
        print("\n" + "═" * 60)
        print(" EAFL")
        print("═" * 60)
        eafl_runner = EAFLRunner(cfg, trainset, client_indices, test_loader, device)
        all_results["EAFL"] = eafl_runner.run(logger)

    # ── 5. FedAsync variants ──────────────────────────────────────────────
    if run_fedasync:
        fa_configs = _build_fedasync_configs_for_eafl(cfg, rounds_override=cfg.rounds)
        allowed    = set(fedasync_variants or ["const", "poly", "hinge"])

        for fa_cfg in fa_configs:
            if fa_cfg.staleness_fn_name not in allowed:
                continue
            label = f"FedAsync+{fa_cfg.staleness_fn_name}"
            print("\n" + "═" * 60)
            print(f" {label}")
            print("═" * 60)
            fa_runner = FedAsyncRunner(fa_cfg, trainset, client_indices, test_loader, device)
            all_results[label] = fa_runner.run(logger)

    # ── 6. Summary ────────────────────────────────────────────────────────
    summary_path = os.path.join(cfg.output_dir, f"{run_id}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n" + "═" * 60)
    print(" COMPARISON SUMMARY")
    print("═" * 60)
    for algo, res in all_results.items():
        print(f"  {algo:<28}  best={res['best_accuracy']:.2f}% @ r{res['best_round']:>5}  "
              f"final={res['final_accuracy']:.2f}%  "
              f"time={res['total_time_sec']/60:.1f}min")
    print(f"\nRounds CSV : {logger.path}")
    print(f"Summary JSON: {summary_path}")

    return all_results


# ═════════════════════════════════════════════════════════════════════════════
#  SECTION 7 — Entry point
# ═════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    sys.stdout = Logger("experiment_log.txt")

    parser = argparse.ArgumentParser(description="EAFL vs FedAsync comparison")
    parser.add_argument("--dataset",  choices=["mnist", "cifar10"], default="mnist",
                        help="Dataset to use")
    parser.add_argument("--split",    choices=["t1", "t2"],         default="t2",
                        help="Non-IID split type")
    parser.add_argument("--seed",     type=int,  default=42)
    parser.add_argument("--rounds",   type=int,  default=None,
                        help="Override rounds (default: paper value)")
    parser.add_argument("--algo",     choices=["both", "eafl", "fedasync"], default="both",
                        help="Which algorithm(s) to run")
    parser.add_argument("--fedasync_variants", nargs="+",
                        choices=["const", "poly", "hinge"],
                        default=["const", "poly", "hinge"],
                        help="Which FedAsync staleness strategies to run")
    parser.add_argument("--output_dir", default="results",
                        help="Directory for CSV / JSON output")
    args = parser.parse_args()

    # Pick EAFL preset
    preset_map = {
        ("mnist",   "t1"): MNIST_T1_CONFIG,
        ("mnist",   "t2"): MNIST_T2_CONFIG,
        ("cifar10", "t1"): CIFAR10_T1_CONFIG,
        ("cifar10", "t2"): CIFAR10_T2_CONFIG,
    }
    config          = copy.deepcopy(preset_map[(args.dataset, args.split)])
    config.seed     = args.seed
    config.output_dir = args.output_dir
    # Stamp experiment name for sub-configs to inherit
    config.experiment_name = f"eafl_{args.dataset}_{args.split}"

    results = run_comparison(
        eafl_config=config,
        run_eafl=args.algo in ("both", "eafl"),
        run_fedasync=args.algo in ("both", "fedasync"),
        rounds_override=args.rounds,
        fedasync_variants=args.fedasync_variants,
    )

    print("\nFinal results:")
    print(json.dumps(results, indent=2))
