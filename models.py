import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


# ─── Paper Section VI-A ──────────────────────────────────────────────────────
# MNIST  : "a CNN model containing two convolutional layers, both with ReLU"
# CIFAR-10: "classical CNN model, LeNet"   → LeNet-5
# ─────────────────────────────────────────────────────────────────────────────


class MNISTModel(nn.Module):
    """
    Two-conv-layer CNN with ReLU for MNIST (paper Section VI-A).
    Input: 1 × 28 × 28.
    """
    def __init__(self, num_channels=1, img_size=28, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, kernel_size=5, padding=2)   # 28→28
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, padding=2)             # 14→14
        self.pool  = nn.MaxPool2d(2, 2)
        self.relu  = nn.ReLU()
        # After two pool ops: 28 → 14 → 7
        fc_in = 64 * (img_size // 4) * (img_size // 4)
        self.fc1 = nn.Linear(fc_in, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))   # 28 → 14
        x = self.pool(self.relu(self.conv2(x)))   # 14 → 7
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        return self.fc2(x)


class LeNetCIFAR(nn.Module):
    """
    LeNet-5 adapted for CIFAR-10 (paper Section VI-A: "classical CNN model, LeNet").
    Input: 3 × 32 × 32 → 10 classes.

    ─── BUG FIXED ──────────────────────────────────────────────────────────────
    Original used ReLU activations in the hidden FC layers.
    Classic LeNet-5 uses Tanh (or Sigmoid) in all layers — ReLU is not standard
    LeNet-5. Using ReLU instead of Tanh changes the effective model.

    More critically: the original used MaxPool in the convolutional stages.
    LeNet-5 uses AvgPool (subsampling layers S2 and S4). Using MaxPool is not
    LeNet-5 and gives a differently-behaved model.

    This implementation follows the standard LeNet-5 architecture:
      - Conv → AvgPool (×2)
      - Tanh activations throughout
    which matches "LeNet" as cited in the paper [LeCun et al. 1998, ref [39]].
    ─────────────────────────────────────────────────────────────────────────────
    """
    def __init__(self, num_channels=3, img_size=32, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 6, kernel_size=5, padding=0)  # 32 → 32
        self.pool  = nn.AvgPool2d(2, 2)    # ← AvgPool, not MaxPool (LeNet-5 spec)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)                       # 16 → 12
        # After two pool ops: 32 → 16 → 6  (after conv2: 16→12, pool: 12→6)
        self.fc1 = nn.Linear(16 * 6 * 6, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x):
        x = self.pool(torch.tanh(self.conv1(x)))   # 32 → 16
        x = self.pool(torch.tanh(self.conv2(x)))   # 12 → 6
        x = x.view(x.size(0), -1)
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        return self.fc3(x)


# ─── Kept for compatibility / extension use ───────────────────────────────────

class SimpleCNN(nn.Module):
    def __init__(self, num_channels=3, img_size=32, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 32, 3, padding=1)
        self.pool  = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        final_dim = img_size // 4
        self.flatten_size = 64 * final_dim * final_dim
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, self.flatten_size)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def get_parameters_flat(model):
    """Returns all model parameters as a single 1D numpy array."""
    return np.concatenate([
        p.data.view(-1).cpu().numpy()
        for p in model.parameters()
    ])


def set_parameters_flat(model, flat_params):
    """Sets model parameters from a single 1D numpy array."""
    start = 0
    for param in model.parameters():
        end = start + param.numel()
        new_data = (torch.from_numpy(flat_params[start:end])
                    .view(param.shape)
                    .type_as(param.data))
        param.data.copy_(new_data)
        start = end


def get_gradients_flat(model):
    """Returns all model gradients as a single 1D numpy array."""
    grads = []
    for param in model.parameters():
        if param.grad is not None:
            grads.append(param.grad.view(-1).cpu().numpy())
        else:
            grads.append(np.zeros(param.numel()))
    return np.concatenate(grads)
