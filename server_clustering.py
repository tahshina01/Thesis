"""
server_clean.py — EAFL Server  (pluggable clustering edition)

Implements the two server-side mechanisms from the paper:

  1. GDC  — Gradient similarity-based Dynamic Clustering (Section V-B)
             Now supports multiple clustering backends:
               • KMeans       (original paper algorithm, default)
               • DBSCAN       (density-based; auto-discovers N, no pre-specified clusters)
               • OPTICS       (density-based with variable density; more robust than DBSCAN)
               • GaussianMixture (soft probabilistic assignment, hard argmax labels)
               • AgglomerativeHierarchical (Ward / complete / average / single linkage)
               • SpectralClustering (graph-Laplacian; captures non-convex clusters)
             Backend is selected by passing `clustering_algo` to Server.__init__
             or to run_clustering().

  2. DSA  — Data size-aware Synchronous Inter-cluster Aggregation (Section V-C, Eq. 5)
             Unchanged from the paper: weighted sum of cluster gradients by total
             cluster data volume.

Usage example (in experiment_eafl.py or wherever you instantiate Server):

    from server_clean import Server, ClusteringAlgo

    # DBSCAN variant
    server = Server(test_loader, device=device,
                    clustering_algo=ClusteringAlgo.DBSCAN,
                    clustering_kwargs={"eps": 0.3, "min_samples": 2})

    # OPTICS variant
    server = Server(test_loader, device=device,
                    clustering_algo=ClusteringAlgo.OPTICS,
                    clustering_kwargs={"min_samples": 2, "xi": 0.05})

    # Gaussian Mixture
    server = Server(test_loader, device=device,
                    clustering_algo=ClusteringAlgo.GAUSSIAN_MIXTURE,
                    clustering_kwargs={"covariance_type": "full", "max_iter": 200})

    # Agglomerative (Ward linkage by default)
    server = Server(test_loader, device=device,
                    clustering_algo=ClusteringAlgo.AGGLOMERATIVE)

    # Spectral
    server = Server(test_loader, device=device,
                    clustering_algo=ClusteringAlgo.SPECTRAL,
                    clustering_kwargs={"affinity": "rbf", "assign_labels": "kmeans"})

    # Or override per-call (overrides the server-level default for that round only):
    server.run_clustering(..., clustering_algo=ClusteringAlgo.KMEANS)

Paper: "Towards Efficient Asynchronous Federated Learning in Heterogeneous Edge
        Environments", IEEE INFOCOM 2024.
"""

from __future__ import annotations

import warnings
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.cluster import (
    KMeans,
    DBSCAN,
    OPTICS,
    AgglomerativeClustering,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import normalize

from models import SimpleCNN, get_parameters_flat, set_parameters_flat


# ─────────────────────────────────────────────────────────────────────────────
# Clustering algorithm registry
# ─────────────────────────────────────────────────────────────────────────────

class ClusteringAlgo(Enum):
    """Supported clustering backends for GDC."""
    KMEANS               = auto()   # original paper algorithm (default)
    DBSCAN               = auto()   # density-based spatial clustering
    OPTICS               = auto()   # ordering points to identify cluster structure
    GAUSSIAN_MIXTURE     = auto()   # Gaussian mixture model (EM)
    AGGLOMERATIVE        = auto()   # hierarchical agglomerative (Ward by default)
    SPECTRAL             = auto()   # spectral clustering on gradient similarity graph


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers — gradient preprocessing (shared by all algorithms)
# ─────────────────────────────────────────────────────────────────────────────

def _preprocess_gradients(
    grads_list: List[np.ndarray],
    sample_size: int = 4096,
    seed: int = 42,
) -> np.ndarray:
    """
    Prepare client gradient vectors for cosine-distance clustering.

    Paper Section V-B:
        "We use the cosine distance to measure the gradient direction similarity
         across clients … cos(g, g') = g·g' / (|g|·|g'|)"

    L2-normalising each row and running Euclidean-distance algorithms is
    mathematically equivalent to cosine-distance clustering on the unit sphere.
    This allows any standard Euclidean algorithm to serve as a cosine-distance
    backend without a custom metric.

    Parameters
    ----------
    grads_list  : list of 1-D numpy arrays, one per client
    sample_size : coordinates to retain (random subsample for large models)
    seed        : RNG seed for reproducible subsampling

    Returns
    -------
    G_norm : np.ndarray shape (M, min(sample_size, D)), L2-normalised row-wise
    """
    if not grads_list:
        return np.empty((0, 0), dtype=np.float32)

    D      = len(grads_list[0])
    n_samp = min(sample_size, D)
    rng    = np.random.default_rng(seed)
    idx    = rng.choice(D, size=n_samp, replace=False)

    G = np.array(
        [np.nan_to_num(np.asarray(g[idx], dtype=np.float32),
                       nan=0.0, posinf=0.0, neginf=0.0)
         for g in grads_list],
        dtype=np.float32,
    )

    G_norm = normalize(G, norm="l2", axis=1)

    zero_rows = np.linalg.norm(G_norm, axis=1) < 1e-12
    if np.any(zero_rows):
        warnings.warn(
            f"_preprocess_gradients: {zero_rows.sum()} client(s) have near-zero "
            "gradient norm — they will be treated as cluster outliers.",
            RuntimeWarning, stacklevel=3,
        )
        G_norm[zero_rows] = 0.0

    return G_norm


def _build_cluster_outputs(
    labels: np.ndarray,
    client_ids: List[int],
    data_sizes_list: List[int],
) -> Tuple[List[int], Dict[int, int], Dict[int, int], Dict[int, List[int]]]:
    """
    Convert a flat label array into the four cluster data structures used
    throughout experiment.py.

    Noise points (label == -1, produced by DBSCAN / OPTICS) are each assigned
    to their own singleton cluster so they still participate in federation.

    Returns
    -------
    cluster_assignments : list[int], length = max(client_id)+1
    cluster_heads       : dict  cluster_id -> client_id  (random head, Alg.1 line 10)
    cluster_data_sizes  : dict  cluster_id -> int         (Σ|D_i| for all members)
    cluster_members     : dict  cluster_id -> list[client_id]
    """
    # ── Handle DBSCAN / OPTICS noise points (label = -1) ──────────────────
    # Re-assign each noise point to a new unique singleton cluster so that
    # no client is silently dropped from the federation.
    labels = np.array(labels, dtype=int)
    if np.any(labels == -1):
        next_label = int(labels.max()) + 1
        noise_mask = labels == -1
        n_noise    = noise_mask.sum()
        labels[noise_mask] = np.arange(next_label, next_label + n_noise)
        if n_noise > 0:
            warnings.warn(
                f"_build_cluster_outputs: {n_noise} noise point(s) (label=-1) "
                "were each assigned to a new singleton cluster.",
                RuntimeWarning, stacklevel=2,
            )

    # Group client_ids by cluster label
    clusters: Dict[int, List[int]] = {}
    for i, cid in enumerate(client_ids):
        lab = int(labels[i])
        clusters.setdefault(lab, []).append(cid)

    cluster_assignments = [0] * (max(client_ids) + 1)
    for cluster_id, members in clusters.items():
        for cid in members:
            cluster_assignments[cid] = cluster_id

    data_by_cid = dict(zip(client_ids, data_sizes_list))

    cluster_heads:      Dict[int, int] = {}
    cluster_data_sizes: Dict[int, int] = {}
    for cluster_id, members in clusters.items():
        # Algorithm 1, line 10: randomly select a cluster head
        cluster_heads[cluster_id]      = int(np.random.choice(members))
        cluster_data_sizes[cluster_id] = sum(data_by_cid[cid] for cid in members)

    return cluster_assignments, cluster_heads, cluster_data_sizes, clusters


# ─────────────────────────────────────────────────────────────────────────────
# Individual clustering backends
# ─────────────────────────────────────────────────────────────────────────────

def _run_kmeans(
    G_norm: np.ndarray,
    n_clusters: int,
    seed: int,
    kwargs: Dict[str, Any],
) -> np.ndarray:
    """
    K-Means on L2-normalised gradients (≡ cosine-distance K-Means on unit sphere).

    Paper-specified algorithm. `n_clusters` is used directly.

    Extra kwargs (passed through to sklearn.cluster.KMeans):
        n_init, max_iter, tol, algorithm, …
    """
    M = G_norm.shape[0]
    k = min(max(1, n_clusters), M)
    if k >= M:
        return np.arange(M)
    kw = {"n_init": "auto", **kwargs}
    return KMeans(n_clusters=k, random_state=seed, **kw).fit_predict(G_norm)


def _run_dbscan(
    G_norm: np.ndarray,
    n_clusters: int,     # treated as a *hint*; DBSCAN auto-discovers cluster count
    seed: int,           # unused (DBSCAN is deterministic), kept for API consistency
    kwargs: Dict[str, Any],
) -> np.ndarray:
    """
    DBSCAN (Density-Based Spatial Clustering of Applications with Noise).

    Unlike K-Means, DBSCAN does NOT require you to pre-specify the number of
    clusters. Instead it discovers clusters automatically based on local density.
    Points in sparse regions are marked as noise (label -1); _build_cluster_outputs
    promotes these to singleton clusters so no client is lost.

    `n_clusters` is not used but kept in the signature for API uniformity.
    You should tune `eps` and `min_samples` via `clustering_kwargs`.

    Sensible defaults for L2-normalised gradient vectors on the unit sphere:
        eps         = 0.5   (cosine distance ~ 0.5 means ~60° angle between gradients)
        min_samples = 2     (minimum points to form a core)
        metric      = 'euclidean' (on unit sphere ≡ cosine distance)

    Extra kwargs (passed through to sklearn.cluster.DBSCAN):
        eps, min_samples, metric, algorithm, leaf_size, n_jobs, …
    """
    kw = {"eps": 0.5, "min_samples": 2, "metric": "euclidean", **kwargs}
    return DBSCAN(**kw).fit_predict(G_norm)


def _run_optics(
    G_norm: np.ndarray,
    n_clusters: int,     # treated as a hint; OPTICS auto-discovers cluster count
    seed: int,           # unused, kept for API consistency
    kwargs: Dict[str, Any],
) -> np.ndarray:
    """
    OPTICS (Ordering Points To Identify the Clustering Structure).

    A generalisation of DBSCAN that handles clusters of varying density.
    Like DBSCAN it outputs noise points (label -1).

    Sensible defaults for normalised gradient vectors:
        min_samples = 2
        xi          = 0.05  (steepness threshold for cluster boundary detection)
        metric      = 'euclidean'

    Extra kwargs (passed through to sklearn.cluster.OPTICS):
        min_samples, max_eps, metric, cluster_method, xi, predecessor_correction, …
    """
    kw = {"min_samples": 2, "xi": 0.05, "metric": "euclidean", **kwargs}
    return OPTICS(**kw).fit_predict(G_norm)


def _run_gaussian_mixture(
    G_norm: np.ndarray,
    n_clusters: int,
    seed: int,
    kwargs: Dict[str, Any],
) -> np.ndarray:
    """
    Gaussian Mixture Model (EM algorithm, hard argmax labels).

    Fits N multivariate Gaussians to the normalised gradient space. Each
    client is assigned to the component with the highest posterior probability.

    `n_clusters` → `n_components` for GaussianMixture.

    Sensible defaults:
        covariance_type = 'full'     (full covariance matrix per component)
        max_iter        = 200

    Extra kwargs (passed through to sklearn.mixture.GaussianMixture):
        covariance_type, tol, reg_covar, max_iter, n_init, init_params, …
    """
    M = G_norm.shape[0]
    k = min(max(1, n_clusters), M)
    kw = {"covariance_type": "full", "max_iter": 200, **kwargs}
    gm = GaussianMixture(n_components=k, random_state=seed, **kw)
    return gm.fit_predict(G_norm)


def _run_agglomerative(
    G_norm: np.ndarray,
    n_clusters: int,
    seed: int,           # sklearn AgglomerativeClustering has no random_state
    kwargs: Dict[str, Any],
) -> np.ndarray:
    """
    Agglomerative Hierarchical Clustering.

    Bottom-up: starts with M singleton clusters and iteratively merges the
    closest pair until N clusters remain. The merge criterion (linkage) can
    be tuned:
        ward     — minimises within-cluster variance (default; works well with
                   L2-normalised cosine-equivalent vectors)
        complete — maximum pairwise distance (produces compact, equal-size clusters)
        average  — average pairwise distance (compromise)
        single   — minimum pairwise distance (chaining-prone; usually avoid)

    `n_clusters` is used directly as the target number of clusters.

    Extra kwargs (passed through to sklearn.cluster.AgglomerativeClustering):
        linkage, metric, connectivity, compute_full_tree, …
    """
    M = G_norm.shape[0]
    k = min(max(1, n_clusters), M)
    kw = {"linkage": "ward", **kwargs}
    return AgglomerativeClustering(n_clusters=k, **kw).fit_predict(G_norm)


def _run_spectral(
    G_norm: np.ndarray,
    n_clusters: int,
    seed: int,
    kwargs: Dict[str, Any],
) -> np.ndarray:
    """
    Spectral Clustering (graph Laplacian embedding + K-Means in eigenspace).

    Builds a similarity graph on the gradient vectors (RBF kernel by default),
    computes the Laplacian eigenvectors, then clusters in that low-dimensional
    space. Captures non-convex cluster shapes that K-Means misses, which can
    be useful when gradient directions form curved manifolds.

    `n_clusters` is used directly as the target number of eigenvectors / clusters.

    Extra kwargs (passed through to sklearn.cluster.SpectralClustering):
        affinity, gamma, n_neighbors, eigen_solver, assign_labels, n_init, …

    Note: SpectralClustering can be slow for large M (> ~500 clients) because
    it requires a full M×M affinity matrix.  For large federations prefer
    KMeans or DBSCAN.
    """
    M = G_norm.shape[0]
    k = min(max(1, n_clusters), M)
    kw = {"affinity": "rbf", "assign_labels": "kmeans", "n_init": 10, **kwargs}
    return SpectralClustering(n_clusters=k, random_state=seed, **kw).fit_predict(G_norm)


# ─────────────────────────────────────────────────────────────────────────────
# Dispatch table  (ClusteringAlgo → backend function)
# ─────────────────────────────────────────────────────────────────────────────

_ALGO_DISPATCH = {
    ClusteringAlgo.KMEANS:           _run_kmeans,
    ClusteringAlgo.DBSCAN:           _run_dbscan,
    ClusteringAlgo.OPTICS:           _run_optics,
    ClusteringAlgo.GAUSSIAN_MIXTURE: _run_gaussian_mixture,
    ClusteringAlgo.AGGLOMERATIVE:    _run_agglomerative,
    ClusteringAlgo.SPECTRAL:         _run_spectral,
}


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

    Parameters
    ----------
    test_loader       : DataLoader for global evaluation.
    device            : torch device string.
    model_args        : kwargs forwarded to model_class constructor.
    model_class       : FL model class (default: SimpleCNN).
    lr                : global learning rate η.
    clustering_algo   : ClusteringAlgo enum member (default: KMEANS).
                        Can be overridden per run_clustering() call.
    clustering_kwargs : dict of extra hyperparameters forwarded to the chosen
                        clustering backend (e.g. {"eps": 0.4} for DBSCAN).
                        Can also be overridden per run_clustering() call.
    """

    def __init__(
        self,
        test_loader,
        device: str = "cpu",
        model_args: Optional[Dict[str, Any]] = None,
        model_class=SimpleCNN,
        lr: float = 0.005,
        clustering_algo: ClusteringAlgo = ClusteringAlgo.KMEANS,
        clustering_kwargs: Optional[Dict[str, Any]] = None,
    ):
        self.model_class  = model_class
        self.model_args   = model_args or {}
        self.global_model = self.model_class(**self.model_args).to(device)
        self.test_loader  = test_loader
        self.device       = device
        self.lr           = lr
        self.global_round = 0

        # Clustering configuration (server-level defaults, overridable per call)
        self.clustering_algo   = clustering_algo
        self.clustering_kwargs = clustering_kwargs or {}

    # ------------------------------------------------------------------
    # GDC — Gradient similarity-based Dynamic Clustering (Section V-B)
    # ------------------------------------------------------------------

    def run_clustering(
        self,
        grads_list: List[np.ndarray],
        data_sizes_list: List[int],
        client_ids: List[int],
        n_clusters: int,
        seed: int = 42,
        clustering_algo: Optional[ClusteringAlgo] = None,
        clustering_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[List[int], Dict[int, int], Dict[int, int], Dict[int, List[int]]]:
        """
        GDC: cluster clients by cosine similarity of their gradient directions.

        Paper Section V-B:
            "We use the K-Means algorithm to cluster clients with high gradient
             similarity into the same cluster."
        This method generalises that step to any clustering backend.

        Parameters
        ----------
        grads_list        : list of pseudo-gradient arrays (one per client)
        data_sizes_list   : list of |D_i| values, same order as grads_list
        client_ids        : list of client IDs, same order as grads_list
        n_clusters        : N — desired number of clusters.
                            Density-based methods (DBSCAN, OPTICS) treat this
                            as informational only and auto-discover cluster count.
        seed              : RNG seed (gradient subsampling + head selection +
                            clustering where applicable)
        clustering_algo   : override the server-level default for this call only.
        clustering_kwargs : override the server-level kwargs for this call only.

        Returns
        -------
        (cluster_assignments, cluster_heads, cluster_data_sizes, cluster_members)
        — see _build_cluster_outputs for field descriptions
        """
        M = len(client_ids)
        if M == 0:
            return [], {}, {}, {}

        # Resolve algo and kwargs (per-call override > server default)
        algo   = clustering_algo   if clustering_algo   is not None else self.clustering_algo
        kwargs = clustering_kwargs if clustering_kwargs is not None else self.clustering_kwargs

        n_clusters = min(max(1, n_clusters), M)
        print(
            f"Running GDC | algo={algo.name} | M={M} clients | "
            f"N={n_clusters} clusters | seed={seed}"
        )

        # Step 1 — preprocess: random subsample + L2-normalise (cosine ≡ Euclidean on sphere)
        G_norm = _preprocess_gradients(grads_list, seed=seed)

        # Step 2 — dispatch to chosen clustering backend
        backend_fn = _ALGO_DISPATCH[algo]
        try:
            labels = backend_fn(G_norm, n_clusters, seed, dict(kwargs))
        except Exception as exc:
            warnings.warn(
                f"run_clustering: {algo.name} raised '{exc}'. "
                "Falling back to KMeans.",
                RuntimeWarning, stacklevel=2,
            )
            labels = _run_kmeans(G_norm, n_clusters, seed, {})

        # Step 3 — package into experiment-compatible data structures
        return _build_cluster_outputs(labels, client_ids, data_sizes_list)

    # ------------------------------------------------------------------
    # DSA — Data size-aware Synchronous Inter-cluster Aggregation (Eq. 5)
    # ------------------------------------------------------------------

    def aggregate_cluster_updates(self, cluster_updates: List[Dict]) -> None:
        """
        DSA: synchronous weighted aggregation of one gradient per cluster.

        Paper Eq. 5:
            w^t = w^{t-1} - η · (Σ_n |D_n| · ḡ_n^t) / |D|

        Parameters
        ----------
        cluster_updates : list of dicts, each with keys:
            "gradient"          — ḡ_n^t as a 1-D numpy array (output of SAA)
            "cluster_data_size" — |D_n| (full cluster data volume)
        """
        if not cluster_updates:
            return

        total_data = sum(u["cluster_data_size"] for u in cluster_updates)
        if total_data <= 0:
            return

        current_params = get_parameters_flat(self.global_model)
        agg_grad = np.zeros_like(current_params)
        for u in cluster_updates:
            weight    = u["cluster_data_size"] / total_data
            agg_grad += weight * u["gradient"]

        new_params = current_params - self.lr * agg_grad
        set_parameters_flat(self.global_model, new_params)
        # print(
        #     f"[DSA] global round {self.global_round}: "
        #     f"aggregated {len(cluster_updates)} cluster gradient(s), "
        #     f"total data={total_data}"
        # )
        self.global_round += 1

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self) -> Tuple[float, float]:
        """
        Evaluate the global model on the held-out test set.

        Returns
        -------
        (accuracy_percent, avg_cross_entropy_loss)
        """
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
