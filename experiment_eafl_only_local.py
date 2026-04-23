"""
experiment_eafl.py — Paper-faithful EAFL training loop

Implements Algorithm 1 from the paper end-to-end:

    for t = 0 to T:
        [Clustering round, every R iterations]
            Server sends clustering signal to ALL clients
            ALL clients perform local training → send gradients to server
            Server runs GDC (K-Means on cosine-normalised gradients)
            Server randomly assigns a cluster head per cluster
            Server broadcasts cluster assignments

        [Every round — two-stage aggregation]
            INTRA-CLUSTER (SAA, Eq. 4):
                Each cluster: fastest φ·|X_n| clients train locally
                Cluster head collects their gradients, runs SAA
                Cluster head sends ḡ_n^t to server

            INTER-CLUSTER (DSA, Eq. 5):
                Server collects one ḡ_n^t per cluster
                Server computes w^t = w^{t-1} - η · Σ_n (|D_n|/|D|) · ḡ_n^t
                Server sends w^t to all cluster heads
                Each cluster head forwards w^t to its round participants

Key design decisions
─────────────────────────────────────────────────────────
1. Intra-cluster participant selection is always speed-based top-k (fastest φ
   fraction), every round, using per-round timing estimates with meaningful
   jitter (time_jitter_scale ≥ 0.3).  The original used random.sample in
   non-clustering rounds, which is wrong.

2. Staleness τ_i tracks rounds-since-last-participation per client, per cluster.
   Non-participants see their staleness grow; it is used in SAA (Eq. 4) to
   down-weight their next contribution if they re-enter.

3. Model delivery: only clients who participated in SAA this round receive the
   new global model.  Others continue training on their last received model.
   This matches Algorithm 1 lines 29-30 ("send new global model to clients
   participating in this intra-cluster aggregation").

4. The T1 Non-IID split now uses the paper's ε-IID + sort-partition method
   (split_non_iid_t1) instead of Dirichlet-α, which is a different distribution.

5. time_jitter_scale default raised to 0.3 so speed rankings within a cluster
   vary meaningfully across rounds, preventing the same clients from being
   selected every single round.

6. Round t=0 (first clustering round): ALL clients train once to produce GDC
   gradients.  Their post-training local states are persisted.  The immediately
   following SAA stage REUSES those already-computed gradients directly — no
   second training pass is performed.  From round t=1 onward, clustering rounds
   only train all clients once (for GDC), and non-clustering rounds train only
   cluster members as usual.

7. model_history and last_received_round are ELIMINATED.  Because every client
   participates in the round-0 clustering pass, every client already holds a
   valid post-training local state in client_local_states before the very first
   SAA.  In all subsequent rounds every client trains from its own persistent
   local state regardless of whether it was selected, so there is no need to
   look up a stale global snapshot from a history dict.  Participants receive
   the new global model by having client_local_states[cid] overwritten with the
   new server state after DSA.

Paper: "Towards Efficient Asynchronous Federated Learning in Heterogeneous Edge
        Environments", IEEE INFOCOM 2024.
"""

import copy
import csv
import json
import math
import os
import sys
import random
import time
import gc
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from client import Client
from models import SimpleCNN, LeNetCIFAR, MNISTModel
from server_clean import Server
from utils import (
    get_dataset,
    split_non_iid_t1,
    split_non_iid_t2,
    saa_cluster_gradient,
)

class Logger:
    def __init__(self, filename="eafl_only_local.txt"):
        self.terminal = sys.stdout
        self.log = open(filename, "w", encoding="utf-8")
        print(f"[Logger] Writing to: {filename}")

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
    "MNISTModel":  MNISTModel,
    "LeNetCIFAR":  LeNetCIFAR,
    "SimpleCNN":   SimpleCNN,
}


# ─────────────────────────────────────────────────────────────────────────────
# Configuration dataclass
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class EAFLConfig:
    """
    All hyperparameters for an EAFL run.

    Paper-specified values are annotated inline.  Any field without a paper
    reference is a simulation-engineering choice.
    """

    # ── Experiment bookkeeping ──────────────────────────────────────────────
    experiment_name: str = "eafl_run"
    dataset_name:    str = "mnist"          # "mnist" or "cifar10"
    seed:            int = 42
    output_dir:      str = "results_eafl"

    # ── FL system ──────────────────────────────────────────────────────────
    num_clients: int   = 100               # M (paper: 100)
    num_clusters: int  = 5                 # N (paper: 5 MNIST, 15 CIFAR-10)
    phi:          float = 0.1              # φ — fraction of clients per cluster per round
                                           #      (paper: 0.1 MNIST, 0.2 CIFAR-10)
    r_clustering: int  = 100              # R — re-clustering interval (100 / 1000)
    rounds:       int  = 1600             # T — total global iterations (1600 / 10000)
    epochs:       int  = 1                 # Q — local epochs per round (paper: 1)
    client_lr:    float = 0.005           # η (paper: 0.005 MNIST, 0.0005 CIFAR-10)
    server_lr:    float = 0.005           # same as client_lr in paper

    # ── Data heterogeneity ─────────────────────────────────────────────────
    non_iid_type:            str   = "t1"  # "t1" or "t2"
    t1_epsilon:              float = 0.04  # ε (paper T1: 0.04)
    t2_num_labels_per_client: int  = 1    # L_num (paper T2: 1)

    # ── Model ──────────────────────────────────────────────────────────────
    model_name: str          = "MNISTModel"
    model_args: Dict[str, Any] = field(
        default_factory=lambda: {"num_channels": 1, "img_size": 28}
    )

    # ── System heterogeneity simulation ────────────────────────────────────
    # Speed tiers control how fast each client completes local training.
    # The paper specifies heterogeneous operation times E={e1,...,eM} but
    # does not prescribe exact values; this tiered model is standard in FL sims.
    speed_slow_fraction:   float = 0.20   # 20% slow clients
    speed_medium_fraction: float = 0.50   # 50% medium clients
    speed_fast_fraction:   float = 0.30   # 30% fast clients
    speed_slow_range:      Tuple[float, float] = (0.10, 0.20)
    speed_medium_range:    Tuple[float, float] = (0.30, 0.70)
    speed_fast_range:      Tuple[float, float] = (0.80, 1.00)

    # ── Timing simulation ──────────────────────────────────────────────────
    local_batch_size:               int   = 32
    compute_time_per_batch_sec:     float = 0.02
    base_bandwidth_mb_s:            float = 10.0
    server_aggregation_time_sec:    float = 0.01
    server_clustering_time_per_client_sec: float = 0.001

    # time_jitter_scale: exponential jitter added to each client's completion
    # time so that speed rankings within a cluster are not deterministically
    # frozen to the same ordering every round.  Must be large enough to
    # occasionally promote a slower client above a faster one.
    # Empirically, 0.3 works well; 0.05 (old default) was too small.
    time_jitter_scale: float = 0.4

    @property
    def model_class(self):
        return MODEL_REGISTRY[self.model_name]


# ─────────────────────────────────────────────────────────────────────────────
# Paper hyperparameter presets (Table / Section VI-A)
# ─────────────────────────────────────────────────────────────────────────────

MNIST_T1_CONFIG = EAFLConfig(
    experiment_name="mnist_t1",
    dataset_name="mnist",
    num_clients=100,
    num_clusters=5,
    phi=0.1,
    r_clustering=100,
    rounds=1600,
    epochs=1,
    client_lr=0.005,
    server_lr=0.005,
    non_iid_type="t1",
    t1_epsilon=0.04,
    model_name="MNISTModel",
    model_args={"num_channels": 1, "img_size": 28},
)

MNIST_T2_CONFIG = EAFLConfig(
    experiment_name="mnist_t2",
    dataset_name="mnist",
    non_iid_type="t2",
    t2_num_labels_per_client=1,
    num_clusters=5,
    phi=0.1,
    r_clustering=100,
    rounds=1600,
    client_lr=0.005,
    server_lr=0.005,
    model_name="MNISTModel",
    model_args={"num_channels": 1, "img_size": 28},
)

CIFAR10_T1_CONFIG = EAFLConfig(
    experiment_name="cifar10_t1",
    dataset_name="cifar10",
    num_clients=100,
    num_clusters=15,
    phi=0.2,
    r_clustering=1000,
    rounds=10000,
    epochs=1,
    client_lr=0.0005,
    server_lr=0.0005,
    non_iid_type="t1",
    t1_epsilon=0.04,
    model_name="LeNetCIFAR",
    model_args={"num_channels": 3, "img_size": 32},
)

CIFAR10_T2_CONFIG = EAFLConfig(
    experiment_name="cifar10_t2",
    dataset_name="cifar10",
    num_clusters=15,
    phi=0.2,
    r_clustering=1000,
    rounds=10000,
    client_lr=0.0005,
    server_lr=0.0005,
    non_iid_type="t2",
    t2_num_labels_per_client=1,
    model_name="LeNetCIFAR",
    model_args={"num_channels": 3, "img_size": 32},
)


# ─────────────────────────────────────────────────────────────────────────────
# Reproducibility
# ─────────────────────────────────────────────────────────────────────────────

def set_seed(seed: int):
    """Fix all relevant RNG states for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─────────────────────────────────────────────────────────────────────────────
# Client creation
# ─────────────────────────────────────────────────────────────────────────────

def create_clients(trainset, client_indices: list, device, config: EAFLConfig) -> list:
    """
    Instantiate Client objects with heterogeneous system speeds.

    Speed assignment uses a dedicated RNG seeded from config.seed, isolated
    from all other RNG state so that data partitioning and speed assignment
    are fully independent.
    """
    rng = np.random.default_rng(config.seed + 100)   # isolated RNG
    clients = []

    slow_hi = config.speed_slow_range[1]
    med_hi  = config.speed_medium_range[1]

    for cid in range(len(client_indices)):
        r = rng.random()
        if r < config.speed_slow_fraction:
            lo, hi = config.speed_slow_range
        elif r < config.speed_slow_fraction + config.speed_medium_fraction:
            lo, hi = config.speed_medium_range
        else:
            lo, hi = config.speed_fast_range

        speed = float(rng.uniform(lo, hi))
        clients.append(
            Client(cid, trainset, client_indices[cid], device=device, system_speed=speed)
        )

    return clients


# ─────────────────────────────────────────────────────────────────────────────
# Timing simulation helpers
# ─────────────────────────────────────────────────────────────────────────────

def estimate_steps(client: Client, epochs: int, batch_size: int) -> int:
    """Number of gradient steps client will take in one round."""
    n_batches = math.ceil(client.data_size / batch_size) if client.data_size > 0 else 1
    return max(1, n_batches * epochs)


def estimate_completion_time(client: Client, steps: int, model_size_mb: float,
                             config: EAFLConfig,
                             rng: np.random.Generator) -> float:
    """
    Simulated wall-clock time for one local training + upload round.

    compute_time = (batches × compute_time_per_batch) / speed
    comm_time    = (2 × model_size) / (bandwidth × speed)
    jitter       ~ Exponential(time_jitter_scale)

    The jitter term is critical: without it, speed rankings within a cluster
    are deterministically frozen, so the same φ·|X_n| clients win every round
    and slow clients are permanently excluded — exactly the SAFL failure mode.
    """
    speed        = max(client.system_speed, 1e-3)
    compute_time = config.compute_time_per_batch_sec * steps  / speed
    comm_time    = (2.0 * model_size_mb) / max(config.base_bandwidth_mb_s * speed, 1e-6)

    # Multiplicative LogNormal jitter — models channel/load fluctuation.
    # mean of LogNormal(0, sigma) = exp(sigma^2/2) ≈ 1.08 for sigma=0.4,
    # so expected time is slightly inflated but variance is realistic.
    jitter = float(rng.lognormal(mean=0.0, sigma=config.time_jitter_scale))

    return compute_time + comm_time + jitter


def model_size_mb(state_dict: dict) -> float:
    """Model parameter footprint in megabytes (used for bandwidth simulation)."""
    total_bytes = sum(v.numel() * v.element_size() for v in state_dict.values())
    return total_bytes / (1024.0 * 1024.0)


# ─────────────────────────────────────────────────────────────────────────────
# State-dict utilities
# ─────────────────────────────────────────────────────────────────────────────

def clone_state(sd: dict) -> dict:
    return {k: v.detach().clone() for k, v in sd.items()}


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# CSV logging
# ─────────────────────────────────────────────────────────────────────────────

class EAFLLogger:
    """Minimal CSV logger for per-round accuracy / staleness metrics."""

    ROUND_FIELDS = [
        "run_id", "round", "is_clustering_round", "num_clusters",
        "total_participants", "accuracy", "loss",
        "avg_staleness", "max_staleness",
        "simulated_wall_clock_sec", "cumulative_wall_clock_sec",
        "cluster_summary",          # JSON: {cluster_id: {head, members, selected}}
    ]
    SUMMARY_FIELDS = [
        "run_id", "dataset", "non_iid_type", "seed",
        "num_clients", "num_clusters", "phi", "r_clustering",
        "rounds", "epochs", "client_lr",
        "final_accuracy", "final_loss", "best_accuracy", "best_round",
        "total_time_sec",
    ]

    def __init__(self, output_dir: str, run_id: str):
        os.makedirs(output_dir, exist_ok=True)
        self.run_id       = run_id
        self.rounds_path  = os.path.join(output_dir, f"{run_id}_rounds.csv")
        self.summary_path = os.path.join(output_dir, f"{run_id}_summary.csv")
        self._init_file(self.rounds_path,  self.ROUND_FIELDS)
        self._init_file(self.summary_path, self.SUMMARY_FIELDS)

    @staticmethod
    def _init_file(path, fields):
        if not os.path.exists(path):
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()

    def log_round(self, row: dict):
        with open(self.rounds_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.ROUND_FIELDS).writerow(row)

    def log_summary(self, row: dict):
        with open(self.summary_path, "a", newline="") as f:
            csv.DictWriter(f, fieldnames=self.SUMMARY_FIELDS).writerow(row)


# ─────────────────────────────────────────────────────────────────────────────
# Main EAFL training loop
# ─────────────────────────────────────────────────────────────────────────────

class EAFLRunner:
    """
    Faithful implementation of Algorithm 1 from the paper.

    Structural decisions
    --------------------
    * client_local_states[cid] is the single source of truth for each client's
      current model weights.  It is initialised for ALL clients during the very
      first clustering pass (t=0), so no lazy-init or model_history lookup is
      ever needed thereafter.

    * Round t=0 clustering trains every client once; the resulting gradients are
      stored in a first_round_grads dict and passed DIRECTLY into the SAA stage
      instead of training a second time.  From t=1 onward clustering rounds
      train all clients once (GDC only) and non-clustering rounds train only the
      selected φ·|X_n| members.

    * model_history is REMOVED.  Because all clients are initialised at t=0,
      there is no need to reconstruct stale starting points from a history dict.
      Participants receive the new global model by having client_local_states[cid]
      overwritten with the new server state dict after DSA.

    * last_received_round is REMOVED for the same reason — it was only needed to
      index into model_history.

    * Cluster head role: in the paper the cluster head aggregates intra-cluster
      updates and communicates with the server.  Here we simulate this by
      having the runner (playing the role of cluster head) collect training
      results from the selected clients, run SAA, and pass one gradient per
      cluster to the server's DSA.

    * Only SAA participants receive the new model (Algorithm 1, line 30).
    """

    def __init__(self, config: EAFLConfig):
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Device: {self.device}")

    def setup(self):
        """Load data, create clients, initialise server and logger."""
        cfg = self.config

        # ── Data ───────────────────────────────────────────────────────────
        print(f"Loading {cfg.dataset_name} ...")
        trainset, testset = get_dataset(cfg.dataset_name)
        self.test_loader = DataLoader(testset, batch_size=256, shuffle=False)

        # ── Non-IID partition ──────────────────────────────────────────────
        print(f"Partitioning data ({cfg.non_iid_type.upper()}) ...")
        if cfg.non_iid_type == "t1":
            # Paper T1: ε-IID + (1-ε) sort-partition — Section VI-A
            client_indices = split_non_iid_t1(
                trainset, cfg.num_clients, epsilon=cfg.t1_epsilon
            )
        elif cfg.non_iid_type == "t2":
            # Paper T2: each client gets exactly L_num label types — Section VI-A
            client_indices = split_non_iid_t2(
                trainset, cfg.num_clients,
                num_labels_per_client=cfg.t2_num_labels_per_client
            )
        else:
            raise ValueError(f"Unknown non_iid_type: {cfg.non_iid_type}")

        # ── Clients ────────────────────────────────────────────────────────
        # print(f"\n--- Client Information ({cfg.num_clients} Total) ---")
        # for i in range(cfg.num_clients):
        #     print(f"Client {i}: {len(client_indices[i])} samples")
        # print("--------------------------------------\n")
        
        self.clients = create_clients(trainset, client_indices, self.device, cfg)
        print(f"Created {len(self.clients)} clients  "
              f"(slow={cfg.speed_slow_fraction:.0%}, "
              f"medium={cfg.speed_medium_fraction:.0%}, "
              f"fast={cfg.speed_fast_fraction:.0%})")

        # ── Server ─────────────────────────────────────────────────────────
        self.server = Server(
            test_loader=self.test_loader,
            device=self.device,
            model_args=dict(cfg.model_args),
            model_class=cfg.model_class,
            lr=cfg.server_lr,
        )

        # ── Steps per client (for timing simulation) ───────────────────────
        self.steps_by_client = {
            cid: estimate_steps(self.clients[cid], cfg.epochs, cfg.local_batch_size)
            for cid in range(cfg.num_clients)
        }
        # print(f"Steps by client: {self.steps_by_client}\n")
        self.model_mb = model_size_mb(self.server.global_model.state_dict())

        # ── Logger ─────────────────────────────────────────────────────────
        ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_id = f"{cfg.experiment_name}_seed{cfg.seed}_{ts}"
        self.logger = EAFLLogger(cfg.output_dir, run_id)
        print(f"Run id: {run_id}")

    def run(self) -> dict:
        """
        Execute Algorithm 1 for config.rounds iterations.

        Returns a summary dict with final_accuracy, best_accuracy, etc.
        """
        set_seed(self.config.seed)
        self.setup()

        cfg = self.config

        # ── State tracking ─────────────────────────────────────────────────
        # last_participation_round[cid]: round t' when client cid last
        #   participated in an intra-cluster SAA.  -1 means never participated.
        last_participation_round = [-1] * cfg.num_clients

        # GDC outputs (initialised to None; set on first clustering round)
        cluster_members:    Optional[dict] = None
        cluster_heads:      Optional[dict] = None
        cluster_data_sizes: Optional[dict] = None

        # client_local_states[cid]: the client's CURRENT local model weights.
        # Populated for ALL clients during the t=0 clustering pass so that
        # no lazy-init or model_history lookup is ever needed afterwards.
        # Non-participants keep accumulating local drift here round-over-round;
        # participants have their entry overwritten with the new global after DSA.
        client_local_states: Dict[int, dict] = {}

        # first_round_grads[cid]: gradient produced by the t=0 all-client training
        # pass.  Consumed by the SAA stage of round 0 so clients are NOT trained
        # a second time in the same round.  Cleared after round 0 is complete.
        first_round_grads: Dict[int, dict] = {}

        # One stateful RNG for all timing draws — must live outside the loop
        # so jitter advances continuously and is not reset each round.
        timing_rng = np.random.default_rng(cfg.seed + 77)

        best_accuracy      = -1.0
        best_round         = 0
        final_accuracy     = 0.0
        final_loss         = 0.0
        cumulative_wall_clock = 0.0
        process_start      = time.perf_counter()

        print(f"\nStarting EAFL: {cfg.rounds} rounds, N={cfg.num_clusters}, "
              f"φ={cfg.phi}, R={cfg.r_clustering}")
        print("─" * 60)

        for t in range(cfg.rounds):
            print(f"\nRound {t+1}/{cfg.rounds} ")
            round_start = time.perf_counter()

            # ──────────────────────────────────────────────────────────────
            # CLUSTERING ROUND (Algorithm 1)
            # Triggered at t=0 and every R rounds thereafter.
            # ALL M clients train and send gradients to the server.
            # ──────────────────────────────────────────────────────────────
            is_clustering_round = (cluster_members is None) or (t % cfg.r_clustering == 0)
            clustering_time = 0.0

            if is_clustering_round:
                # print(f"\nClustering round: {t}")
                grads_list      = []
                data_sizes_list = []
                client_times    = {}   # for selecting fastest-φ after clustering

                for cid in range(cfg.num_clients):
                    # Each client trains from its current persistent local state.
                    # At t=0, client_local_states is empty so we seed from the
                    # initial global model.  At subsequent clustering rounds the
                    # state already reflects all prior local training.
                    if cid not in client_local_states:
                        # Only possible at t=0 (very first clustering pass).
                        client_local_states[cid] = clone_state(
                            self.server.global_model.state_dict()
                        )
                        # print(f"Clustering t=0: client {cid} seeded from initial global model")

                    tmp_model = cfg.model_class(**cfg.model_args)
                    tmp_model.load_state_dict(client_local_states[cid])
                    # print(f"Clustering: client {cid} training from local state")

                    # train() returns (state_dict, pseudo_gradient, data_size)
                    trained_state, grad, data_size = self.clients[cid].train(
                        tmp_model, epochs=cfg.epochs, learning_rate=cfg.client_lr
                    )
                    del tmp_model
                    gc.collect()

                    # Persist the post-training state — client advanced during
                    # clustering regardless of whether they'll be selected in SAA.
                    client_local_states[cid] = clone_state(trained_state)

                    grads_list.append(grad)
                    data_sizes_list.append(data_size)

                    # At t=0, save grad so the SAA stage can reuse it without
                    # a second training pass.
                    if t == 0:
                        first_round_grads[cid] = {
                            "grad":      grad,
                            "data_size": data_size,
                        }

                # GDC: K-Means on cosine-normalised gradients (Section V-B)
                cluster_start = time.perf_counter()
                (_, cluster_heads, cluster_data_sizes, cluster_members) = (
                    self.server.run_clustering(
                        grads_list=grads_list,
                        data_sizes_list=data_sizes_list,
                        client_ids=list(range(cfg.num_clients)),
                        n_clusters=cfg.num_clusters,
                        seed=cfg.seed + t,  # vary seed per round for diverse heads
                    )
                )
                del grads_list, data_sizes_list  # free memory
                gc.collect()
                clustering_time = time.perf_counter() - cluster_start
            # ── SAA stage: select fastest φ·|X_n| clients per cluster ────────
            # Design principle (the core correctness fix):
            #
            # Every client carries a persistent local model state (client_local_states).
            # Between rounds, NON-participants are NOT reset to any global snapshot —
            # their local state is whatever it was after their last training step.
            # This means two clients with the same staleness τ will have diverged
            # local models (trained on different data shards from different starting
            # checkpoints), which is exactly the asynchronous semantics of the paper.
            #
            # At t=0: client_local_states was fully populated by the clustering pass
            #   above, and first_round_grads already holds each client's gradient.
            #   We reuse those gradients directly — no second training pass occurs.
            #
            # At t>0 clustering rounds: clients have already trained once (above) so
            #   their local states are current; we train them again here for SAA,
            #   advancing them one more step with the same semantics as a normal round.
            #
            # Participants receive the new global model after DSA and their local
            # state is overwritten with that new global, ready for the next round.
            cluster_updates      = []
            all_participants     = set()
            staleness_values     = []
            cluster_summary      = {}
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

                # Select fastest φ·|X_n| clients by completion time
                selected = sorted(members, key=lambda c: member_times[c])[:k]
                for cid in selected:
                    completion_times_sel[cid] = member_times[cid]

                all_participants.update(selected)

                # ── Obtain gradients for all cluster members ───────────────────
                # At t=0: reuse gradients already produced by the clustering pass —
                #   client_local_states is already advanced; no second training.
                # At all other rounds: train each member from their persistent local
                #   state, then save the post-training state back (selected or not).
                member_updates = {}
                for cid in members:
                    # if t == 0:
                    #     # Reuse the gradient captured during the clustering pass.
                    #     member_updates[cid] = first_round_grads[cid]
                    #     # client_local_states[cid] is already the post-training state.
                    #     print(f"Round 0 SAA: client {cid} reusing first-round gradient")
                    # else:
                        # client_local_states[cid] is guaranteed to exist:
                        #   - participants had it overwritten with the global after DSA
                        #   - non-participants retained their evolved local state
                        # No lazy-init or model_history lookup needed.
                    tmp_model = cfg.model_class(**cfg.model_args)
                    tmp_model.load_state_dict(client_local_states[cid])

                    trained_state, grad, data_size = self.clients[cid].train(
                        tmp_model, epochs=cfg.epochs, learning_rate=cfg.client_lr
                    )
                    del tmp_model
                    gc.collect()

                    # Persist the evolved local state (selected OR not)
                    client_local_states[cid] = clone_state(trained_state)

                    member_updates[cid] = {"grad": grad, "data_size": data_size}

                # ── Build SAA update list from SELECTED clients only ───────────
                updates = []
                for cid in selected:
                    grad      = member_updates[cid]["grad"]
                    data_size = member_updates[cid]["data_size"]

                    tau = (t - last_participation_round[cid]
                           if last_participation_round[cid] >= 0 else t + 1)
                    # print(f"Client {cid}: staleness={tau} round {t} "
                    #       f"last_participation_round={last_participation_round[cid]}")
                    staleness_values.append(tau)
                    updates.append({
                        "grads":     grad,
                        "data_size": data_size,
                        "timestamp": last_participation_round[cid],
                    })

                # SAA — Eq. 4
                # print(f"SAA for cluster {cluster_id} with {len(selected)} participants...")
                
                g_bar = saa_cluster_gradient(updates, t)
                if g_bar is not None:
                    cluster_updates.append({
                        "gradient":          g_bar,
                        # DSA uses FULL cluster data volume (Eq. 5), not
                        # just the selected participants' volume
                        "cluster_data_size": cluster_data_sizes[cluster_id],
                    })

                cluster_summary[cluster_id] = {
                    "head":     cluster_heads[cluster_id],
                    "members":  sorted(members),
                    "selected": sorted(selected),
                }

            # Simulate worst-case barrier time (slowest selected client)
            if completion_times_sel:
                barrier_time = max(completion_times_sel.values())
                if is_clustering_round:
                    server_overhead = (cfg.server_aggregation_time_sec
                                        + cfg.server_clustering_time_per_client_sec
                                        * cfg.num_clients)
                else:
                    server_overhead = cfg.server_aggregation_time_sec
                simulated_round_time = barrier_time + server_overhead
            else:
                simulated_round_time = 0.0

            # ──────────────────────────────────────────────────────────────
            # NON-CLUSTERING ROUND (Algorithm 1, lines 13-15 + client lines)
            # Use existing cluster assignments; run SAA per cluster, then DSA.
            # ──────────────────────────────────────────────────────────────
            

            # ──────────────────────────────────────────────────────────────
            # DSA — Data size-aware Synchronous Inter-cluster Aggregation (Eq. 5)
            # Server aggregates one ḡ_n per cluster and updates global model.
            # ──────────────────────────────────────────────────────────────
            if cluster_updates:
                self.server.aggregate_cluster_updates(cluster_updates)

            # ── Model delivery (Algorithm 1, lines 29-30) ─────────────────
            # New global model is sent ONLY to clients who participated in SAA
            # this round (via their cluster head).
            # Non-participants keep their locally-evolved client_local_states
            # entry — they do NOT get reset.
            new_state = clone_state(self.server.global_model.state_dict())

            for cid in all_participants:
                last_participation_round[cid] = t   # τ reset for next round
                # Overwrite local state with the fresh global model
                client_local_states[cid] = clone_state(new_state)
                # print(f"Participant {cid} updated to new global model after round {t}")

            # At t=0, first_round_grads has served its purpose — free memory.
            if t == 0:
                first_round_grads.clear()
                gc.collect()

            # ── Evaluate ──────────────────────────────────────────────────
            final_accuracy, final_loss = self.server.evaluate()
            if final_accuracy > best_accuracy:
                best_accuracy = final_accuracy
                best_round    = t + 1

            # ── Logging ───────────────────────────────────────────────────
            cumulative_wall_clock += simulated_round_time
            avg_staleness = float(np.mean(staleness_values)) if staleness_values else 0.0
            max_staleness = int(max(staleness_values))       if staleness_values else 0

            self.logger.log_round({
                "run_id":                    self.logger.run_id,
                "round":                     t + 1,
                "is_clustering_round":       int(is_clustering_round),
                "num_clusters":              len(cluster_members or {}),
                "total_participants":        len(all_participants),
                "accuracy":                  round(final_accuracy, 4),
                "loss":                      round(final_loss, 6),
                "avg_staleness":             round(avg_staleness, 3),
                "max_staleness":             max_staleness,
                "simulated_wall_clock_sec":  round(simulated_round_time, 4),
                "cumulative_wall_clock_sec": round(cumulative_wall_clock, 4),
                "cluster_summary":           json.dumps(cluster_summary),
            })

            if (t + 1) % 100 == 0 or t == 0:
                print(f"  Round {t+1:>5}/{cfg.rounds}  "
                      f"acc={final_accuracy:.2f}%  "
                      f"loss={final_loss:.4f}  "
                      f"avg_staleness={avg_staleness:.1f}  "
                      f"{'[CLUSTER]' if is_clustering_round else ''}")

        # ── Final summary ──────────────────────────────────────────────────
        total_time = time.perf_counter() - process_start
        self.logger.log_summary({
            "run_id":           self.logger.run_id,
            "dataset":          cfg.dataset_name,
            "non_iid_type":     cfg.non_iid_type,
            "seed":             cfg.seed,
            "num_clients":      cfg.num_clients,
            "num_clusters":     cfg.num_clusters,
            "phi":              cfg.phi,
            "r_clustering":     cfg.r_clustering,
            "rounds":           cfg.rounds,
            "epochs":           cfg.epochs,
            "client_lr":        cfg.client_lr,
            "final_accuracy":   round(final_accuracy, 4),
            "final_loss":       round(final_loss, 6),
            "best_accuracy":    round(best_accuracy, 4),
            "best_round":       best_round,
            "total_time_sec":   round(total_time, 2),
        })

        print("\n" + "─" * 60)
        print(f"Done.  Best accuracy: {best_accuracy:.2f}% at round {best_round}")
        print(f"Final accuracy:       {final_accuracy:.2f}%")
        print(f"Total wall time:      {total_time/60:.1f} min")

        return {
            "best_accuracy":  best_accuracy,
            "best_round":     best_round,
            "final_accuracy": final_accuracy,
            "final_loss":     final_loss,
            "total_time_sec": total_time,
            "rounds_csv":     self.logger.rounds_path,
            "summary_csv":    self.logger.summary_path,
        }


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def run(config: EAFLConfig) -> dict:
    runner = EAFLRunner(config)
    return runner.run()


if __name__ == "__main__":
    import argparse
    sys.stdout = Logger()
    parser = argparse.ArgumentParser(description="EAFL — paper-faithful implementation")
    parser.add_argument("--dataset",  choices=["mnist", "cifar10"], default="mnist")
    parser.add_argument("--split",    choices=["t1", "t2"],         default="t2")
    parser.add_argument("--seed",     type=int,                     default=42)
    parser.add_argument("--rounds",   type=int,                     default=None,
                        help="Override number of rounds (default: paper value)")
    args = parser.parse_args()

    # Pick the appropriate paper preset
    preset_map = {
        ("mnist",   "t1"): MNIST_T1_CONFIG,
        ("mnist",   "t2"): MNIST_T2_CONFIG,
        ("cifar10", "t1"): CIFAR10_T1_CONFIG,
        ("cifar10", "t2"): CIFAR10_T2_CONFIG,
    }
    config = copy.deepcopy(preset_map[(args.dataset, args.split)])
    config.seed = args.seed
    if args.rounds is not None:
        config.rounds = args.rounds
    log_filename = f"eafl_{config.dataset_name}_{config.non_iid_type}_r{config.rounds}_s{config.seed}.txt"
    sys.stdout = Logger(log_filename)

    results = run(config)
    print(json.dumps(results, indent=2))
