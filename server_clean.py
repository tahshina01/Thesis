"""
server_clean.py — EAFL Server

Implements the two server-side mechanisms from the paper:

  1. GDC  — Gradient similarity-based Dynamic Clustering (Section V-B)
             Groups clients by cosine direction of their local gradients.
             Re-runs every R global iterations (Algorithm 1, lines 4-12).

  2. DSA  — Data size-aware Synchronous Inter-cluster Aggregation (Section V-C, Eq. 5)
             Synchronously aggregates one aggregated gradient vector per cluster,
             weighted by the TOTAL data volume of each cluster (not just participants).

The cluster head concept (Algorithm 1) is represented structurally: each cluster
has a designated head ID, and the server only accepts one aggregated gradient per
cluster (the output of SAA, computed in experiment.py after intra-cluster training).
This mirrors the paper's topology where cluster heads are the only nodes that
communicate with the server.

Paper: "Towards Efficient Asynchronous Federated Learning in Heterogeneous Edge
        Environments", IEEE INFOCOM 2024.
"""

import warnings

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

from models import SimpleCNN, get_parameters_flat, set_parameters_flat


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_gradients(grads_list: list, sample_size: int = 4096, seed: int = 42) -> np.ndarray:
    """
    Prepare client gradient vectors for cosine-distance K-Means.

    Paper Section V-B:
        "We use the cosine distance to measure the gradient direction similarity
         across clients … cos(g, g') = g·g' / (|g|·|g'|)"

    Implementation:
        L2-normalising each row and then running Euclidean K-Means is
        mathematically equivalent to cosine-distance K-Means on the unit sphere
        (cosine distance is a monotone function of Euclidean distance after
        normalisation).  This avoids a custom distance metric.

    Subsampling:
        Modern CNNs have millions of parameters.  We sub-sample `sample_size`
        coordinates uniformly at random (fixed seed for reproducibility) before
        normalising.  The paper does not specify this; it is a necessary
        engineering decision for large models.  With 4 096 dimensions the
        cosine direction is well-preserved.

    Parameters
    ----------
    grads_list  : list of 1-D numpy arrays, one per client, same length D
    sample_size : number of gradient coordinates to retain (≤ D)
    seed        : RNG seed for reproducible coordinate selection

    Returns
    -------
    grads_norm : np.ndarray shape (M, sample_size), L2-normalised row-wise
    """
    if not grads_list:
        return np.empty((0, 0), dtype=np.float32)

    D      = len(grads_list[0])
    n_samp = min(sample_size, D)
    rng    = np.random.default_rng(seed)
    idx    = rng.choice(D, size=n_samp, replace=False)

    # Stack and sanitise in one pass
    G = np.array(
        [np.nan_to_num(np.asarray(g[idx], dtype=np.float32),
                       nan=0.0, posinf=0.0, neginf=0.0)
         for g in grads_list],
        dtype=np.float32,
    )

    # L2-normalise rows (sklearn does this efficiently)
    G_norm = normalize(G, norm="l2", axis=1)

    # Warn about zero-norm clients (degenerate / untrained)
    zero_rows = np.linalg.norm(G_norm, axis=1) < 1e-12
    if np.any(zero_rows):
        warnings.warn(
            f"_preprocess_gradients: {zero_rows.sum()} client(s) have near-zero "
            "gradient norm — they will be treated as cluster outliers.",
            RuntimeWarning, stacklevel=2,
        )
        G_norm[zero_rows] = 0.0     # leave as zero; K-Means will still assign them

    return G_norm


def _build_cluster_outputs(labels: np.ndarray, client_ids: list,
                           data_sizes_list: list) -> tuple:
    """
    Convert flat K-Means label array into the four cluster data structures
    used throughout experiment.py.

    Returns
    -------
    cluster_assignments : list[int], length = max(client_id)+1
        cluster_assignments[cid] = cluster_id for that client
    cluster_heads : dict  cluster_id -> client_id
        Paper Alg. 1 line 10: "randomly select a client as cluster head"
    cluster_data_sizes : dict  cluster_id -> int
        |D_n| = sum of data sizes of ALL members in cluster n.
        Paper Section V-C (DSA): the weight in Eq. 5 is the *full* cluster
        data volume, not the volume of the φ-fraction participants.
    cluster_members : dict  cluster_id -> list[client_id]
    """
    # Group client_ids by their label
    clusters: dict = {}
    for i, cid in enumerate(client_ids):
        lab = int(labels[i])
        clusters.setdefault(lab, []).append(cid)

    # flat lookup array (indexed by client_id)
    cluster_assignments = [0] * (max(client_ids) + 1)
    for cluster_id, members in clusters.items():
        for cid in members:
            cluster_assignments[cid] = cluster_id

    data_by_cid = dict(zip(client_ids, data_sizes_list))

    cluster_heads      = {}
    cluster_data_sizes = {}
    for cluster_id, members in clusters.items():
        # Random head selection — Algorithm 1 line 10
        cluster_heads[cluster_id] = int(np.random.choice(members))

        # |D_n| = total data of the FULL cluster (Eq. 5 denominator)
        cluster_data_sizes[cluster_id] = sum(data_by_cid[cid] for cid in members)

    return cluster_assignments, cluster_heads, cluster_data_sizes, clusters


# ─────────────────────────────────────────────────────────────────────────────
# Server
# ─────────────────────────────────────────────────────────────────────────────

class Server:
    """
    Central server in the EAFL system.

    Responsibilities (Algorithm 1):
      - Every R rounds: collect gradients from ALL clients, run GDC, broadcast
        cluster assignments and cluster head identities.
      - Every round: receive one aggregated gradient per cluster (from cluster
        heads), run DSA to update the global model, broadcast new model to
        cluster heads.

    The server never sees raw client-to-server communication in non-clustering
    rounds — only cluster-head-to-server communication.
    """

    def __init__(self, test_loader, device="cpu",
                 model_args: dict = None, model_class=SimpleCNN, lr: float = 0.005):
        self.model_class  = model_class
        self.model_args   = model_args or {}
        self.global_model = self.model_class(**self.model_args).to(device)
        self.test_loader  = test_loader
        self.device       = device
        self.lr           = lr          # η — server (global) learning rate, same as client lr in paper
        self.global_round = 0           # t — incremented by DSA after every aggregation

    # ------------------------------------------------------------------
    # GDC — Gradient similarity-based Dynamic Clustering (Section V-B)
    # ------------------------------------------------------------------

    def run_clustering(
        self,
        grads_list: list,
        data_sizes_list: list,
        client_ids: list,
        n_clusters: int,
        seed: int = 42,
    ) -> tuple:
        """
        GDC: cluster clients by cosine similarity of their gradient directions.

        Paper Section V-B:
            "We use the K-Means algorithm to cluster clients with high gradient
             similarity into the same cluster."
            Objective: minimise Σ_n Σ_{i∈X_n} distance(g_i, centroid_n)
            centroid_n = (1/|X_n|) Σ_{i∈X_n} g_i

        This runs on the server every R global rounds (Algorithm 1, lines 8-11).
        All M clients must have submitted gradients before this is called.

        Parameters
        ----------
        grads_list      : list of pseudo-gradient arrays, one per client
        data_sizes_list : list of |D_i| values, same order as grads_list
        client_ids      : list of client IDs, same order as grads_list
        n_clusters      : N — number of clusters (paper: 5 for MNIST, 15 for CIFAR-10)
        seed            : K-Means random seed for reproducibility

        Returns
        -------
        (cluster_assignments, cluster_heads, cluster_data_sizes, cluster_members)
        — see _build_cluster_outputs for field descriptions
        """
        M = len(client_ids)
        if M == 0:
            return [], {}, {}, {}

        # Degenerate case: fewer clients than requested clusters
        n_clusters = min(max(1, n_clusters), M)
        # print(f"Running GDC with M={M} clients, N={n_clusters} clusters, seed={seed}")
        # Step 1 — preprocess (subsample + L2-normalise for cosine K-Means)
        G_norm = _preprocess_gradients(grads_list, seed=seed)

        # Step 2 — K-Means (paper specifies K-Means explicitly)
        if n_clusters >= M:
            # Each client gets its own singleton cluster
            labels = np.arange(M)
        else:
            kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init="auto")
            labels = kmeans.fit_predict(G_norm)

        # Step 3 — package results
        return _build_cluster_outputs(labels, client_ids, data_sizes_list)

    # ------------------------------------------------------------------
    # DSA — Data size-aware Synchronous Inter-cluster Aggregation (Eq. 5)
    # ------------------------------------------------------------------

    def aggregate_cluster_updates(self, cluster_updates: list) -> None:
        """
        DSA: synchronous weighted aggregation of one gradient per cluster.

        Paper Eq. 5:
            w^t = w^{t-1} - η · (Σ_n |D_n| · ḡ_n^t) / |D|

        where:
          ḡ_n^t   = intra-cluster aggregated gradient from SAA (Eq. 4),
                    produced by the cluster head and sent to the server.
          |D_n|   = total data size of ALL members of cluster n (not just the
                    φ-fraction participants in this round).  This is stored in
                    cluster_data_sizes and passed in via cluster_updates.
          |D|     = Σ_n |D_n|

        "Inter-cluster aggregation is synchronous": all N clusters must submit
        before the server updates.  The caller (experiment.py) must ensure
        cluster_updates contains one entry per active cluster.

        Parameters
        ----------
        cluster_updates : list of dicts, each with keys:
            "gradient"         — ḡ_n^t as a 1-D numpy array (output of SAA)
            "cluster_data_size"— |D_n| (full cluster data volume, not participants)
        """
        if not cluster_updates:
            return

        total_data = sum(u["cluster_data_size"] for u in cluster_updates)
        if total_data <= 0:
            return

        # Eq. 5 numerator:  Σ_n (|D_n| / |D|) · ḡ_n^t
        current_params = get_parameters_flat(self.global_model)
        agg_grad = np.zeros_like(current_params)
        for u in cluster_updates:
            weight    = u["cluster_data_size"] / total_data   # |D_n| / |D|
            agg_grad += weight * u["gradient"]

        # Global model update: w^t = w^{t-1} - η · agg_grad
        new_params = current_params - self.lr * agg_grad
        set_parameters_flat(self.global_model, new_params)
        # print(f"[DSA] global round {self.global_round}: aggregated {len(cluster_updates)} cluster gradients, total data {total_data}")
        # Increment the global round counter (used for staleness tracking)
        self.global_round += 1

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> tuple:
        """
        Evaluate the global model on the held-out test set.

        Returns
        -------
        (accuracy_percent, avg_cross_entropy_loss)
        """
        print(f"[Evaluation] Evaluating global model at round {self.global_round}")
        self.global_model.eval()
        criterion = torch.nn.CrossEntropyLoss()
        total, correct, loss_sum = 0, 0, 0.0

        with torch.no_grad():
            for images, labels in self.test_loader:
                images  = images.to(self.device)
                labels  = labels.to(self.device)
                outputs = self.global_model(images)
                loss_sum += criterion(outputs, labels).item()
                _, predicted = torch.max(outputs, 1)
                total   += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / total if total > 0 else 0.0
        avg_loss = loss_sum / len(self.test_loader) if self.test_loader else 0.0
        return accuracy, avg_loss
