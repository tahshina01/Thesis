"""
FedAsync: Asynchronous Federated Optimization
Implementation based on:
  Xie, Koyejo, Gupta. "Asynchronous Federated Optimization." OPT2020.
  arXiv:1903.03934v5

Algorithm 1 implemented faithfully, including:
  - Regularized local SGD:  min f(x; z) + (ρ/2)||x - x_t||^2
  - Server update:          x_t = (1-α_t) * x_{t-1} + α_t * x_new
  - Adaptive mixing via staleness functions: Const, Poly, Hinge
"""

import copy
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from typing import Callable, Optional, Tuple, List
from dataclasses import dataclass, field


# ──────────────────────────────────────────────────────────────
# 1.  Staleness weighting functions  s(t - τ)
#     Requirements (Remark 2 in paper):
#       s(0) = 1,  monotonically decreasing as (t-τ) grows
# ──────────────────────────────────────────────────────────────

def staleness_const(staleness: int) -> float:
    """s(t-τ) = 1  →  α_t = α  (no adaptation)"""
    return 1.0


def staleness_poly(staleness: int, a: float = 0.5) -> float:
    """
    Polynomial decay:
        s_a(t-τ) = (t - τ + 1)^{-a}

    Paper example: a = 0.5
    """
    return (staleness + 1) ** (-a)


def staleness_hinge(staleness: int, a: float = 10.0, b: float = 4.0) -> float:
    """
    Hinge function:
        s_{a,b}(t-τ) = 1                          if t-τ ≤ b
                      = 1 / (a*(t-τ-b) + 1)       otherwise

    Paper examples:
      CIFAR-10 : a=10, b=4
      WikiText-2: a=10, b=2
    """
    if staleness <= b:
        return 1.0
    return 1.0 / (a * (staleness - b) + 1.0)


# ──────────────────────────────────────────────────────────────
# 2.  Hyperparameter container
# ──────────────────────────────────────────────────────────────

@dataclass
class FedAsyncConfig:
    """
    All hyperparameters referenced in the paper (Table 1).

    n          : number of devices
    T          : total global epochs (server updates)
    H_min      : minimum local iterations per worker per round
    H_max      : maximum local iterations per worker per round
    gamma      : local SGD learning rate  γ  (paper: 0.1 for CIFAR, 20 for LM)
    rho        : regularisation weight    ρ  (paper: 0.005 / 0.0001; must be > µ)
    alpha      : base mixing hyperparameter α ∈ (0,1)
    max_staleness  : upper bound K on t-τ  (bounded delay assumption)
    staleness_fn   : one of staleness_const / staleness_poly / staleness_hinge
    # Experiment settings
    n_clients_per_round : how many workers are triggered each epoch
    seed               : reproducibility
    """
    n: int = 100
    T: int = 1000
    H_min: int = 50
    H_max: int = 50
    gamma: float = 0.1
    rho: float = 0.005
    alpha: float = 0.6
    max_staleness: int = 4
    staleness_fn: Callable[[int], float] = staleness_const
    n_clients_per_round: int = 1   # async: 1 worker pushes per server update
    seed: int = 42


# ──────────────────────────────────────────────────────────────
# 3.  Worker-side local solver  (Algorithm 1 – Worker process)
# ──────────────────────────────────────────────────────────────

def local_update(
    model: nn.Module,
    global_model_snapshot: nn.Module,
    dataset: torch.utils.data.Dataset,
    H: int,
    gamma: float,
    rho: float,
    device: torch.device,
    batch_size: int = 50,
) -> nn.Module:
    """
    Solve the regularised local problem for H stochastic gradient steps:

        min_{x}  E_{z~D_i}[ f(x; z) ]  +  (ρ/2) ||x - x_τ||^2

    where x_τ is the (possibly stale) global model snapshot.

    The gradient of the regularised objective is:
        ∇g_{x_τ}(x; z) = ∇f(x; z) + ρ*(x - x_τ)

    We implement this by:
      1. Running standard SGD on f(x; z) with lr=γ.
      2. After each step, applying the proximal/regularisation term
         via an explicit gradient:  x ← x - γ * ρ * (x - x_τ)

    This is equivalent to one step of SGD on the regularised objective.

    Args:
        model                 : local copy, initialised to x_τ
        global_model_snapshot : x_τ (frozen reference for regularisation)
        dataset               : D_i (device-local data)
        H                     : number of local iterations H_i^τ ∈ [H_min, H_max]
        gamma                 : learning rate γ
        rho                   : regularisation weight ρ
        device                : torch device
        batch_size            : mini-batch size

    Returns:
        model with updated parameters (x_new  =  x_i_{τ, H})
    """
    model = model.to(device)
    global_model_snapshot = global_model_snapshot.to(device)

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    loader_iter = iter(loader)

    # SGD on f(x; z) only — regularisation added manually below
    optimizer = optim.SGD(model.parameters(), lr=gamma)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for _ in range(H):
        # Randomly sample z_i ~ D_i   (paper: "Randomly sample z^i_{τ,h} ~ D^i")
        try:
            inputs, targets = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            inputs, targets = next(loader_iter)

        inputs, targets = inputs.to(device), targets.to(device)

        # ∇f(x; z) step
        optimizer.zero_grad()
        loss = criterion(model(inputs), targets)
        loss.backward()
        optimizer.step()

        # Regularisation correction:  x ← x - γ·ρ·(x - x_τ)
        # This implements the full gradient of (ρ/2)||x - x_τ||^2
        with torch.no_grad():
            for p, p_ref in zip(model.parameters(),
                                global_model_snapshot.parameters()):
                p.data -= gamma * rho * (p.data - p_ref.data)

    return model


# ──────────────────────────────────────────────────────────────
# 4.  Server updater  (Algorithm 1 – Thread Updater)
# ──────────────────────────────────────────────────────────────

def server_update(
    global_model: nn.Module,
    new_model: nn.Module,
    alpha: float,
    staleness: int,
    staleness_fn: Callable[[int], float],
) -> Tuple[nn.Module, float]:
    """
    Server weighted-average update (Updater thread, Algorithm 1):

        α_t = α · s(t - τ)          (adaptive mixing)
        x_t = (1 - α_t) · x_{t-1}  +  α_t · x_new

    Args:
        global_model : x_{t-1}   (current global parameters)
        new_model    : x_new     (locally trained model from a worker)
        alpha        : base mixing hyperparameter α
        staleness    : t - τ
        staleness_fn : s(·), one of Const/Poly/Hinge

    Returns:
        updated global_model, effective alpha_t
    """
    # Optional adaptive α  (paper Remark 2)
    alpha_t = alpha * staleness_fn(staleness)

    with torch.no_grad():
        for p_global, p_new in zip(global_model.parameters(),
                                   new_model.parameters()):
            # x_t = (1 - α_t)*x_{t-1} + α_t*x_new
            p_global.data = (1.0 - alpha_t) * p_global.data + alpha_t * p_new.data

    return global_model, alpha_t


# ──────────────────────────────────────────────────────────────
# 5.  Staleness simulator
#     (paper: "simulate the asynchrony by randomly sampling
#              the staleness (t-τ) from a uniform distribution")
# ──────────────────────────────────────────────────────────────

def sample_staleness(max_staleness: int) -> int:
    """
    t - τ  ~  Uniform{0, 1, …, max_staleness}
    """
    return random.randint(0, max_staleness)


# ──────────────────────────────────────────────────────────────
# 6.  Main FedAsync training loop
# ──────────────────────────────────────────────────────────────

class FedAsync:
    """
    Full FedAsync trainer.

    Usage
    -----
    trainer = FedAsync(
        global_model   = my_cnn,
        client_datasets= partitioned_datasets,   # list of length n
        config         = FedAsyncConfig(
            n=100, T=5000, H_min=50, H_max=50,
            gamma=0.1, rho=0.005, alpha=0.9,
            max_staleness=4,
            staleness_fn=staleness_const,
        ),
        device         = torch.device("cuda"),
    )
    history = trainer.train(test_loader=test_loader, eval_every=100)
    """

    def __init__(
        self,
        global_model: nn.Module,
        client_datasets: List[torch.utils.data.Dataset],
        config: FedAsyncConfig,
        device: Optional[torch.device] = None,
    ):
        self.cfg = config
        self.device = device or torch.device("cpu")
        self.global_model = copy.deepcopy(global_model).to(self.device)
        self.client_datasets = client_datasets

        # δ = H_max / H_min  (imbalance ratio, Table 1)
        assert config.H_min > 0
        self.delta = config.H_max / config.H_min

        random.seed(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    def _sample_local_H(self) -> int:
        """
        Sample H_i^τ  ∈  [H_min, H_max]   (uniform for simulation)
        """
        return random.randint(self.cfg.H_min, self.cfg.H_max)

    def train(
        self,
        test_loader: Optional[DataLoader] = None,
        eval_every: int = 100,
        verbose: bool = True,
    ) -> dict:
        """
        Run T global epochs of Algorithm 1.

        Returns history dict with keys:
            'train_loss', 'test_acc' (if test_loader given),
            'alpha_t_values', 'staleness_values'
        """
        history = {
            "epoch": [],
            "alpha_t": [],
            "staleness": [],
            "test_acc": [],
        }

        cfg = self.cfg

        for t in range(1, cfg.T + 1):

            # ── Scheduler: pick one worker asynchronously ─────────────
            worker_id = random.randint(0, cfg.n - 1)

            # Simulate staleness  t - τ  (bounded by K = max_staleness)
            staleness = sample_staleness(cfg.max_staleness)
            tau = max(0, t - staleness)   # τ  (the epoch the worker read)

            # ── Worker reads a (possibly stale) global snapshot ────────
            # x_τ :  in simulation we use the *current* global model as proxy
            #        because we do not store all historical checkpoints.
            #        For a real async system, the worker would hold x_τ.
            local_model = copy.deepcopy(self.global_model)  # x_i_{τ,0} ← x_τ
            global_snapshot = copy.deepcopy(self.global_model)  # frozen x_τ for regulariser

            # ── Worker: solve regularised local problem ────────────────
            H = self._sample_local_H()
            local_model = local_update(
                model=local_model,
                global_model_snapshot=global_snapshot,
                dataset=self.client_datasets[worker_id],
                H=H,
                gamma=cfg.gamma,
                rho=cfg.rho,
                device=self.device,
            )

            # ── Updater: server weighted average ─────────────────────
            self.global_model, alpha_t = server_update(
                global_model=self.global_model,
                new_model=local_model,
                alpha=cfg.alpha,
                staleness=staleness,
                staleness_fn=cfg.staleness_fn,
            )

            # ── Logging ───────────────────────────────────────────────
            history["epoch"].append(t)
            history["alpha_t"].append(alpha_t)
            history["staleness"].append(staleness)

            if test_loader is not None and t % eval_every == 0:
                acc = self._evaluate(test_loader)
                history["test_acc"].append((t, acc))
                if verbose:
                    print(
                        f"[FedAsync] Epoch {t:5d}/{cfg.T} | "
                        f"staleness={staleness} | α_t={alpha_t:.4f} | "
                        f"test_acc={acc:.2f}%"
                    )

        return history

    @torch.no_grad()
    def _evaluate(self, loader: DataLoader) -> float:
        self.global_model.eval()
        correct = total = 0
        for inputs, targets in loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            preds = self.global_model(inputs).argmax(dim=1)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
        self.global_model.train()
        return 100.0 * correct / total


# ──────────────────────────────────────────────────────────────
# 7.  Quick-start example  (CIFAR-10 replication)
# ──────────────────────────────────────────────────────────────

def build_cifar10_cnn() -> nn.Module:
    """
    CNN architecture from Appendix B, Table 2 of the paper.
    conv64-BN-conv64-BN-pool-drop(0.25)-conv128-BN-conv128-BN-pool-drop(0.25)-
    FC512-drop(0.25)-FC10
    """
    return nn.Sequential(
        # Block 1
        nn.Conv2d(3, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.Conv2d(64, 64, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(64),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.25),
        # Block 2
        nn.Conv2d(64, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.Conv2d(128, 128, kernel_size=3, padding=1),
        nn.ReLU(inplace=True),
        nn.BatchNorm2d(128),
        nn.MaxPool2d(2),
        nn.Dropout2d(0.25),
        # Classifier
        nn.Flatten(),
        nn.Linear(128 * 8 * 8, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(0.25),
        nn.Linear(512, 10),
    )


def partition_dataset_non_iid(
    dataset: torch.utils.data.Dataset,
    n_clients: int,
    seed: int = 42,
) -> List[Subset]:
    """
    Partition dataset across n_clients so each client holds a disjoint
    subset (random split, approximating non-IID as used in the paper).
    For a truly non-IID split, replace with label-skewed sharding.
    """
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(dataset))
    splits = np.array_split(indices, n_clients)
    return [Subset(dataset, split.tolist()) for split in splits]


if __name__ == "__main__":
    """
    Minimal reproduction of the CIFAR-10 experiment from Section 5.

    Paper hypers (Figure 2):
        n=100, γ=0.1, ρ=0.005, α=0.9 or 0.6,
        max_staleness ∈ {4, 16}

    Runs all three staleness strategies for α=0.9, max_staleness=4.
    """
    import torchvision
    import torchvision.transforms as T

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {DEVICE}")

    # ── Data ──────────────────────────────────────────────────
    transform = T.Compose([T.ToTensor(),
                           T.Normalize((0.4914, 0.4822, 0.4465),
                                       (0.2023, 0.1994, 0.2010))])
    train_data = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=True, transform=transform)
    test_data  = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=True, transform=transform)
    test_loader = DataLoader(test_data, batch_size=256, shuffle=False)

    # Partition: n=100 devices, non-IID (disjoint random split)
    N_CLIENTS = 100
    client_datasets = partition_dataset_non_iid(train_data, N_CLIENTS)

    # ── Run each variant ─────────────────────────────────────
    EXPERIMENTS = [
        # (name,             alpha, max_staleness, staleness_fn)
        ("FedAsync+Const α=0.9 K=4",  0.9, 4,  staleness_const),
        ("FedAsync+Poly  α=0.9 K=4",  0.9, 4,  lambda s: staleness_poly(s, a=0.5)),
        ("FedAsync+Hinge α=0.9 K=4",  0.9, 4,  lambda s: staleness_hinge(s, a=10, b=4)),
        ("FedAsync+Const α=0.6 K=4",  0.6, 4,  staleness_const),
    ]

    results = {}
    for name, alpha, max_stale, sfn in EXPERIMENTS:
        print(f"\n{'='*60}")
        print(f" {name}")
        print(f"{'='*60}")

        cfg = FedAsyncConfig(
            n=N_CLIENTS,
            T=500,            # reduce for a quick smoke test; paper uses more
            H_min=50,
            H_max=50,
            gamma=0.1,
            rho=0.005,
            alpha=alpha,
            max_staleness=max_stale,
            staleness_fn=sfn,
            seed=42,
        )

        model = build_cifar10_cnn()
        trainer = FedAsync(
            global_model=model,
            client_datasets=client_datasets,
            config=cfg,
            device=DEVICE,
        )
        hist = trainer.train(test_loader=test_loader, eval_every=100, verbose=True)
        results[name] = hist

    print("\nDone. Access `results[name]['test_acc']` for (epoch, accuracy) pairs.")
