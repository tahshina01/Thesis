"""
utils.py — EAFL utility functions

Contents
--------
  get_dataset()         — download / load MNIST or CIFAR-10
  split_non_iid_t1()    — Non-IID type T1: ε-IID + (1-ε) sort-partition
  split_non_iid_t2()    — Non-IID type T2: each client gets exactly L_num labels
  staleness_weight()    — p_i = 1/τ_i  (Eq. 3)
  saa_cluster_gradient()— SAA: staleness-aware intra-cluster aggregation (Eq. 4)
  calc_cosine_similarity() — utility for analysing gradient directions

Paper: Section VI-A (data splits), Section V-C (SAA, Eq. 3-4).
"""

import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.data import Subset


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def get_dataset(name: str = "cifar10", root: str = "./data"):
    """
    Download and return (trainset, testset) for the requested benchmark.

    Paper Section VI-A datasets: MNIST and CIFAR-10.
    Normalisation statistics are the standard ImageNet-style values used
    in the majority of FL papers for these two benchmarks.

    Parameters
    ----------
    name : "mnist" | "cifar10"
    root : local cache directory

    Returns
    -------
    trainset, testset  (torchvision Dataset objects)
    """
    if name == "cifar10":
        # Standard CIFAR-10 channel means and standard deviations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465),
                                 (0.2023, 0.1994, 0.2010)),
        ])
        trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                                download=True, transform=transform)
        testset  = torchvision.datasets.CIFAR10(root=root, train=False,
                                                download=True, transform=transform)

    elif name == "mnist":
        # Standard MNIST normalisation (global mean / std of pixel values)
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        trainset = torchvision.datasets.MNIST(root=root, train=True,
                                              download=True, transform=transform)
        testset  = torchvision.datasets.MNIST(root=root, train=False,
                                              download=True, transform=transform)
    else:
        raise ValueError(f"Unknown dataset '{name}'. Supported: 'mnist', 'cifar10'.")

    return trainset, testset


# ─────────────────────────────────────────────────────────────────────────────
# Non-IID data partitioning — T1 (paper Section VI-A)
# ─────────────────────────────────────────────────────────────────────────────

def split_non_iid_t1(dataset, num_clients: int, epsilon: float = 0.04) -> list:
    """
    T1 Non-IID split — paper Section VI-A:
        "The first type (T1) allocates ε-proportion of the examples in an IID
         fashion and allocates the rest (1-ε)-proportion in a sort-and-partition
         fashion."

    Correct implementation:
      1. IID part (ε fraction):
         For each class, randomly select ε of its samples and distribute them
         uniformly across all clients in round-robin order.

      2. Non-IID part ((1-ε) fraction):
         Collect ALL remaining samples from ALL classes into one pool and sort
         them globally by class label.  Then partition that sorted array into
         num_clients consecutive equal-sized blocks.  Because the array is
         sorted by label, each block falls in a region dominated by at most
         1-2 class boundaries, giving each client a strongly skewed label
         distribution.  This is the correct meaning of "sort-and-partition".

    Parameters
    ----------
    dataset     : torchvision dataset with .targets attribute
    num_clients : M — total number of clients (paper: 100)
    epsilon     : ε — IID fraction (paper T1: ε=0.04, so 96% is non-IID)

    Returns
    -------
    client_indices : list of lists, client_indices[i] = sample indices for client i
    """
    assert 0.0 <= epsilon <= 1.0, "epsilon must be in [0, 1]"

    targets     = np.array(dataset.targets)
    num_classes = int(targets.max()) + 1
    all_indices = np.arange(len(targets))

    rng = np.random.default_rng(42)    # fixed seed for reproducibility

    client_indices = [[] for _ in range(num_clients)]
    non_iid_pool   = []   # accumulates remaining indices across all classes

    # ── Step 1: IID part ──────────────────────────────────────────────────────
    # For each class: randomly pick ε fraction and distribute uniformly across
    # all clients in round-robin order.  The remaining samples go into the pool.
    for c in range(num_classes):
        class_idx = all_indices[targets == c].copy()
        rng.shuffle(class_idx)

        n_iid = max(0, int(round(epsilon * len(class_idx))))

        if n_iid > 0:
            for i, sample_idx in enumerate(class_idx[:n_iid]):
                client_indices[i % num_clients].append(int(sample_idx))

        non_iid_pool.extend(class_idx[n_iid:].tolist())

    # ── Step 2: Non-IID part — sort globally by label, then partition ─────────
    # Sort the entire pool by class label so the array looks like:
    #   [all class-0 samples | all class-1 samples | ... | all class-9 samples]
    # Splitting this into num_clients equal consecutive blocks means each client
    # receives samples from at most 1-2 classes — strong Non-IID.
    non_iid_pool = np.array(non_iid_pool)
    non_iid_pool = non_iid_pool[np.argsort(targets[non_iid_pool])]

    chunks = np.array_split(non_iid_pool, num_clients)
    for i, chunk in enumerate(chunks):
        client_indices[i].extend(chunk.tolist())

    # Shuffle each client's final index list so label order does not bleed
    # into mini-batch ordering during training.
    for i in range(num_clients):
        arr = np.array(client_indices[i])
        rng.shuffle(arr)
        client_indices[i] = arr.tolist()

    return client_indices


# ─────────────────────────────────────────────────────────────────────────────
# Non-IID data partitioning — T2 (paper Section VI-A)
# ─────────────────────────────────────────────────────────────────────────────

def split_non_iid_t2(dataset, num_clients: int,
                     num_labels_per_client: int = 1) -> list:
    """
    T2 Non-IID split — paper Section VI-A:
        "In the second type (T2), each client is only assigned data samples from
         a fixed L_num kinds of labels."

    Paper-faithful implementation:
      - Partition the num_clients clients into groups, one group per class.
        Each class is assigned exactly (num_clients * L_num // num_classes)
        clients.
      - Each assigned client receives an equal share of that class's samples.
        The paper specifies only label restriction, not quantity imbalance,
        so equal splitting is the correct faithful default.
      - A single seeded RNG is used throughout for full reproducibility.

    For the paper's hardest setting (L_num=1, M=100, 10 classes):
      - clients_per_class = 100 * 1 // 10 = 10
      - Each client gets data from exactly 1 label
      - Each client gets ~1/10 of that class's samples

    Parameters
    ----------
    dataset               : torchvision dataset with .targets attribute
    num_clients           : M (paper: 100)
    num_labels_per_client : L_num (paper: 1 for hardest setting)

    Returns
    -------
    client_indices : list of lists, client_indices[i] = sample indices for client i
    """
    if hasattr(dataset, "targets"):
        targets = np.array(dataset.targets)
    elif isinstance(dataset, Subset) and hasattr(dataset.dataset, "targets"):
        targets = np.array(dataset.dataset.targets)[dataset.indices]
    else:
        raise ValueError("split_non_iid_t2: dataset must have a .targets attribute")

    num_classes = int(targets.max()) + 1
    rng = np.random.default_rng(42)   # single RNG for full reproducibility

    # clients_per_class: how many clients are assigned to each class
    clients_per_class = (num_clients * num_labels_per_client) // num_classes
    if clients_per_class <= 0:
        raise ValueError(
            f"clients_per_class = {clients_per_class} ≤ 0. "
            f"Increase num_labels_per_client or reduce num_clients."
        )

    # Build a shuffled assignment: class c → client IDs [c*cpc .. (c+1)*cpc)
    # We shuffle a list of all client IDs first so the mapping is random
    # rather than always assigning clients 0-9 to class 0, etc.
    all_client_ids = np.arange(num_clients)
    rng.shuffle(all_client_ids)

    # Detect collisions: if L_num > 1, the same client could be assigned to
    # two different classes.  Raise early rather than silently violating the
    # L_num constraint.
    total_slots = clients_per_class * num_classes
    if total_slots > num_clients:
        raise ValueError(
            f"num_labels_per_client={num_labels_per_client} causes "
            f"{total_slots} assignment slots for {num_clients} clients — "
            f"some clients would receive more than L_num label types. "
            f"Reduce num_labels_per_client or increase num_clients."
        )

    client_indices = [[] for _ in range(num_clients)]

    for c in range(num_classes):
        # Clients assigned to class c
        assigned = all_client_ids[c * clients_per_class : (c + 1) * clients_per_class]

        # All samples of class c, shuffled for random per-client allocation
        class_idx = np.where(targets == c)[0].copy()
        rng.shuffle(class_idx)

        # Equal split: np.array_split handles non-divisible sizes gracefully
        chunks = np.array_split(class_idx, clients_per_class)
        for client_id, chunk in zip(assigned, chunks):
            client_indices[client_id].extend(chunk.tolist())

    # Shuffle each client's index list so label order does not bleed into
    # mini-batch ordering during training.
    for i in range(num_clients):
        arr = np.array(client_indices[i])
        rng.shuffle(arr)
        client_indices[i] = arr.tolist()

    return client_indices


# ─────────────────────────────────────────────────────────────────────────────
# SAA building blocks
# ─────────────────────────────────────────────────────────────────────────────

def staleness_weight(current_round: int, last_round: int) -> float:
    """
    Staleness-aware weight for client i, paper Eq. 3:

        p_i = 1 / τ_i,   τ_i = t - t'

    where t is the current global round and t' is the round in which client i
    last participated in intra-cluster aggregation.

    Special case: τ_i ≤ 0 (fresh update, should not normally happen but can
    occur at round 0 when t' = -1 is initialised) → weight 1.0.

    Parameters
    ----------
    current_round : t  (global round index, 0-based)
    last_round    : t' (round index of client's last intra-cluster participation,
                        -1 if client has never participated)

    Returns
    -------
    float weight p_i ∈ (0, 1]
    """
    if last_round < 0:
        # Client has never participated: treat as maximally fresh for round 0
        tau = 1
    else:
        tau = current_round - last_round

    if tau <= 0:
        return 1.0
    return 1.0 / float(tau)


def saa_cluster_gradient(updates: list, current_round: int):
    """
    SAA — Staleness-aware semi-Asynchronous intra-cluster Aggregation (Eq. 4):

        ḡ_n^t = Σ_{i ∈ V_n^t} (|D_i| / |D_n^t|) · p_i · ∇f(w_i^t, D_i)

    where:
      V_n^t    = the fastest φ·|X_n| clients in cluster n at round t
      |D_n^t|  = Σ_{i ∈ V_n^t} |D_i|  (total data of *participants* only,
                  used here as the normaliser; note this is different from
                  the |D_n| in DSA which sums over the full cluster)
      p_i      = 1/τ_i  (staleness weight, Eq. 3)
      ∇f(...)  = pseudo-gradient returned by client.train()

    This function is called by the cluster head after it has collected
    training results from all selected intra-cluster participants.

    Parameters
    ----------
    updates : list of dicts, one per participating client, with keys:
        "grads"     — pseudo-gradient array (1-D numpy), output of client.train()
        "data_size" — |D_i|
        "timestamp" — t' (last_participation_round for this client, -1 if first)
    current_round : t — the current global round index (0-based)

    Returns
    -------
    g_bar : 1-D numpy array, the cluster-level aggregated gradient ḡ_n^t,
            or None if updates is empty or total data is zero.
    """
    if not updates:
        return None

    D_n_t = sum(u["data_size"] for u in updates)
    if D_n_t == 0:
        return None

    grad_dim = len(updates[0]["grads"])
    g_bar    = np.zeros(grad_dim, dtype=np.float64)

    for u in updates:
        # Data fraction: |D_i| / |D_n^t|
        data_frac = u["data_size"] / D_n_t

        # Staleness weight p_i = 1/τ_i  (Eq. 3)
        p_i = staleness_weight(current_round, u["timestamp"])

        # Accumulate: (|D_i|/|D_n^t|) · p_i · g_i
        g_bar += data_frac * p_i * u["grads"]
        # print(f"\n[SAA] Client last round {u['timestamp']} staleness={p_i} round {current_round} data_frac={data_frac:.4f}")

    return g_bar


# ─────────────────────────────────────────────────────────────────────────────
# Miscellaneous
# ─────────────────────────────────────────────────────────────────────────────

def calc_cosine_similarity(grad1: np.ndarray, grad2: np.ndarray) -> float:
    """
    Cosine similarity between two gradient vectors.
    Used for ad-hoc analysis / sanity checks; NOT called in the main training loop.

    Returns a float in [-1, 1].
    """
    g1 = grad1.reshape(1, -1)
    g2 = grad2.reshape(1, -1)
    return float(cosine_similarity(g1, g2)[0][0])
