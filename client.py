"""
client.py — EAFL Client

Represents a single edge device in the federated learning system.
Each client has:
  - A local dataset slice (Non-IID)
  - A fixed system_speed that controls how fast it completes local training
  - A train() method that returns pseudo-gradients for both clustering (GDC)
    and aggregation (SAA / DSA)

Paper reference: Section III-A (System Model), Section V-B (gradient collection),
                 Section V-C (local update, Eq. 2)
"""

import copy
import gc
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset

from models import get_parameters_flat


class Client:
    """
    Edge device in the EAFL system.

    Attributes
    ----------
    client_id : int
    dataset   : the full training set (shared reference, never mutated)
    indices   : the indices belonging to this client (Non-IID slice)
    device    : torch device
    system_speed : float in (0, 1].  1.0 = fastest possible, 0.1 = very slow.
                   Controls simulated completion time (used by the server to
                   decide which clients finish first in a given round).
    data_size : number of local training samples |D_i|
    """

    def __init__(self, client_id: int, dataset, indices: list,
                 device="cpu", system_speed: float = 1.0):
        self.client_id   = client_id
        self.dataset     = dataset
        self.indices     = indices
        self.device      = device
        self.system_speed = float(system_speed)
        self.data_size   = len(indices)

        # Build a persistent DataLoader for this client's local slice.
        # shuffle=True is important so repeated calls to train() see different
        # mini-batch orderings (helps avoid gradient collapse with small slices).
        self.train_loader = DataLoader(
            Subset(dataset, indices),
            batch_size=32,
            shuffle=True,
            drop_last=False,        # keep every sample even in the last batch
        )

    # ------------------------------------------------------------------
    # Core training method
    # ------------------------------------------------------------------

    def train(self, global_model, epochs: int = 1, learning_rate: float = 0.01):
        """
        Local SGD training (paper Eq. 2):

            w_i^t = w^{t'} - η · ∇f(w^{t'}, D_i)

        where w^{t'} is the global model the client received (possibly stale),
        η is the local learning rate, and ∇f is computed over all local batches.

        The method returns a *pseudo-gradient* g_i defined as:

            g_i = (w_before - w_after) / (η · num_steps)

        This quantity is dimensionally identical to a true gradient and is what
        the server uses in SAA (Eq. 4) and DSA (Eq. 5).

        Approximating the true gradient this way is standard in FL simulations
        because it avoids storing per-batch gradients while giving the server
        the correct gradient-scale signal.

        Parameters
        ----------
        global_model : nn.Module — the model state sent by the (cluster head's)
                        server.  It is deep-copied so the original is untouched.
        epochs       : number of local epochs Q (paper Section VI-A: Q=1)
        learning_rate: η (paper: 0.005 MNIST, 0.0005 CIFAR-10)

        Returns
        -------
        state_dict      : dict — updated model weights after local training
        pseudo_gradient : np.ndarray shape (D,) — approximated gradient vector,
                          safe to pass directly to saa_cluster_gradient()
        data_size       : int — |D_i|, used for data-weighted aggregation
        """
        # Work on a local copy; never mutate the server's model object
        model = copy.deepcopy(global_model).to(self.device)
        model.train()

        # Paper uses plain SGD (no momentum, no weight decay) — Section VI-A
        optimizer = optim.SGD(model.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()

        # Snapshot parameters BEFORE training to compute the parameter delta
        params_before = get_parameters_flat(model).copy()

        num_steps = 0
        for _ in range(epochs):
            for images, labels in self.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(images)
                loss    = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                num_steps += 1
                del loss, outputs, images, labels  # free memory

        params_after = get_parameters_flat(model)

        # Pseudo-gradient: how much did parameters move, normalised back to
        # gradient scale so the server's update rule  w ← w - η·g  applies correctly.
        #   param_delta = params_before - params_after   (SGD moves opposite to grad)
        #   num_steps * lr = total step size applied
        # => pseudo_grad ≈ (1/num_steps) Σ ∇f_batch
        num_steps = max(1, num_steps)
        scale     = max(learning_rate, 1e-12)   # guard against zero division

        params_before-=params_after
        params_before/=scale
    
        pseudo_gradient = params_before

        # Sanitise: NaN/Inf can appear if a client's slice is tiny or degenerate
        pseudo_gradient = np.nan_to_num(
            pseudo_gradient, nan=0.0, posinf=0.0, neginf=0.0
        )
        state=model.state_dict()
        del model, optimizer, criterion, params_before, params_after  # free memory
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        return state, pseudo_gradient, self.data_size
