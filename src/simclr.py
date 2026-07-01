"""Two-phase amyloid-core model: SimCLR encoder + frozen-encoder classifier head.

This module is the single source of truth for the model code used by the two-phase
training protocol (``notebooks/Train_protocol.ipynb``). Both ``src/train.py`` (training)
and ``src/predict.py`` (inference) import from here so the encoder, classifier, losses
and resampling live in exactly one place.

Phase 1 (self-supervised): a :class:`SimCLREncoder` (a :class:`~src.flexible_cnn.FlexibleCNN`
backbone, no classification head) is trained with the NT-Xent contrastive loss on two
augmented views of each peptide embedding. Labels are never used in the loss.

Phase 2 (supervised): the encoder is frozen and a single ``Linear(ENC_DIM, 1)`` head
(:class:`AmyloidClassifier`) is trained with label-smoothed BCE on the class-resampled set.

The classifier checkpoint schema (``state_dict`` + ``cnn_config`` + ``enc_dim``) is the one
:func:`src.predict.load_model` rebuilds at inference time.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader, Dataset, TensorDataset

from .esm_embed import EMBED_DIM
from .flexible_cnn import FlexibleCNN, GlobalPool

# The final model's architecture (Train_protocol.ipynb, cell 15).
CNN_CONFIG = dict(
    in_channels=EMBED_DIM,   # 480
    conv_channels=(256,),
    kernel_sizes=(3, 5),
    pool="maxmean",
    batchnorm=True,
    activation="relu",
    dropout=0,
)

LABEL_SMOOTH_EPS = 0.10


def enc_dim_for(cnn_config: dict) -> int:
    """Encoder output width = last conv channels x pooling multiplier.

    ``pool='max'``/``'mean'`` -> x1; ``pool='maxmean'`` -> x2 (max + mean concatenated).
    """
    return cnn_config["conv_channels"][-1] * GlobalPool(cnn_config["pool"]).out_multiplier()


# ── Models ─────────────────────────────────────────────────────────────────────

class SimCLREncoder(nn.Module):
    """FlexibleCNN backbone (features + global pool), no head. Forward: (B,480,20) -> (B,ENC_DIM)."""

    def __init__(self, cnn_config: dict):
        super().__init__()
        _full         = FlexibleCNN(**cnn_config)
        self.features = _full.features
        self.pool     = _full.pool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.pool(self.features(x))   # (B, ENC_DIM)


class ProjectionHead(nn.Module):
    """MLP projection enc_dim -> 128 -> 64, used only during SSL then discarded."""

    def __init__(self, enc_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(enc_dim, 128), nn.ReLU(),
            nn.Linear(128, 64),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # (B, 64)


class AmyloidClassifier(nn.Module):
    """Frozen SimCLR encoder + scalar binary head. Forward: (B,480,20) -> logits (B,)."""

    def __init__(self, encoder: SimCLREncoder, enc_dim: int, freeze_encoder: bool = True):
        super().__init__()
        self.encoder = encoder
        self.head    = nn.Linear(enc_dim, 1)
        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(x)).squeeze(-1)   # (B,)


# ── SSL augmentations & dataset ────────────────────────────────────────────────

def random_mask(x: torch.Tensor, p: float = 0.10) -> torch.Tensor:
    """Zero out random position columns. x: (480, 20) -> (480, 20)."""
    mask = torch.rand(x.shape[1], device=x.device) < p
    out  = x.clone()
    out[:, mask] = 0.0
    return out


def add_noise(x: torch.Tensor, std: float = 0.05) -> torch.Tensor:
    """Add Gaussian noise. x: (480, 20) -> (480, 20)."""
    return x + torch.randn_like(x) * std


def augment(x: torch.Tensor) -> torch.Tensor:
    """Randomly apply masking, noise, or both."""
    r = torch.rand(1).item()
    if r < 0.33:
        return random_mask(x)
    elif r < 0.67:
        return add_noise(x)
    return add_noise(random_mask(x))


class SimCLRDataset(Dataset):
    """Returns two independently augmented views of the same sample. Labels ignored."""

    def __init__(self, X: np.ndarray):
        self.X = torch.from_numpy(X)   # (N, 480, 20)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        x = self.X[idx]                # (480, 20)
        return augment(x), augment(x)  # two independent views


# ── Losses, eval, loaders ──────────────────────────────────────────────────────

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """NT-Xent (InfoNCE) contrastive loss over two batches of projected views."""
    B   = z1.shape[0]
    z   = F.normalize(torch.cat([z1, z2], dim=0), dim=1)   # (2B, D)
    sim = torch.mm(z, z.T) / temperature                    # (2B, 2B)
    sim = sim.masked_fill(torch.eye(2 * B, dtype=torch.bool, device=z.device), float("-inf"))
    labels = torch.cat([
        torch.arange(B, 2 * B, device=z.device),
        torch.arange(0, B, device=z.device),
    ])
    return F.cross_entropy(sim, labels)


def bce_smooth_loss(logits: torch.Tensor, y: torch.Tensor,
                    eps: float = LABEL_SMOOTH_EPS) -> torch.Tensor:
    """BCE with label smoothing: {0,1} -> {eps/2, 1-eps/2}."""
    return F.binary_cross_entropy_with_logits(logits, y * (1 - eps) + eps / 2.0)


@torch.no_grad()
def evaluate(model, loader, device) -> dict:
    """Sigmoid-threshold-0.5 metrics (f1/auc/precision/recall) for an AmyloidClassifier."""
    model.eval()
    probs_all, true_all = [], []
    for xb, yb in loader:
        probs_all.append(torch.sigmoid(model(xb.to(device))).cpu().numpy())
        true_all.append(yb.numpy())
    probs = np.concatenate(probs_all)
    true  = np.concatenate(true_all).astype(int)
    preds = (probs >= 0.5).astype(int)
    return {
        "f1"       : f1_score(true, preds, zero_division=0),
        "auc"      : roc_auc_score(true, probs) if len(np.unique(true)) > 1 else float("nan"),
        "precision": precision_score(true, preds, zero_division=0),
        "recall"   : recall_score(true, preds, zero_division=0),
    }


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool) -> DataLoader:
    """TensorDataset loader with float32 labels (for BCE)."""
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y.astype(np.float32)))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)


# ── Resampling (protocol: oversample positives 2x + undersample negatives to 2x) ─

def resample(X_pos: np.ndarray, X_neg: np.ndarray, rng: np.random.Generator):
    """Oversample positives 2x (original + Gaussian sigma=0.05 noisy copy) and
    undersample negatives to 2x the augmented positives -> 1:2 (pos:neg). Shuffled."""
    noise     = rng.normal(0.0, 0.05, X_pos.shape).astype(np.float32)
    X_pos_aug = np.concatenate([X_pos, X_pos + noise])
    y_pos_aug = np.ones(len(X_pos_aug), dtype=np.int64)
    n_sel     = min(len(X_pos_aug) * 2, len(X_neg))
    sel       = rng.choice(len(X_neg), size=n_sel, replace=False)
    X_neg_sub = X_neg[sel]
    y_neg_sub = np.zeros(n_sel, dtype=np.int64)
    X_out     = np.concatenate([X_pos_aug, X_neg_sub])
    y_out     = np.concatenate([y_pos_aug, y_neg_sub])
    perm      = rng.permutation(len(X_out))
    return X_out[perm], y_out[perm]


def resample_split(X: np.ndarray, y: np.ndarray, rng: np.random.Generator):
    """Convenience wrapper: split (X, y) into pos/neg and :func:`resample` them."""
    return resample(X[y == 1], X[y == 0], rng)
