"""Tunable trainer for the peptide-level amyloid-core CNN.

Exposes every architecture and optimization hyperparameter on the command line so you
can sweep them yourself. The model is :class:`FlexibleCNN`; the full architecture recipe
is saved inside the checkpoint so ``src/predict.py`` can rebuild it for benchmark
inference.

Examples
--------
# baseline (2 conv layers, kernel 3)
python -m src.train_tune

# 3 conv layers, inception-style multi-kernel, mean+max pooling, a dense head
python -m src.train_tune --conv-channels 64,128,256 --kernels 3,5,7 \
    --pool maxmean --fc-hidden 64 --dropout 0.4 --weight-decay 1e-3 --epochs 80

# shallow, no batch-norm, GELU
python -m src.train_tune --conv-channels 32 --kernels 5 --no-batchnorm --activation gelu
"""

from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .data import DEFAULT_MAX_LEN, build_dataset
from .esm_embed import EMBED_DIM
from .flexible_cnn import ACTIVATIONS, FlexibleCNN

DEFAULT_POS = "New_dataset/Training/train_core_set_KA.json"
DEFAULT_NEG = "New_dataset/Training/new_negative.json"


def int_list(s: str):
    """Parse '64,128,256' -> (64, 128, 256); '' -> ()."""
    s = s.strip()
    if not s:
        return ()
    return tuple(int(x) for x in s.split(",") if x.strip())


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_prob, all_true = [], []
    for xb, yb in loader:
        prob = torch.softmax(model(xb.to(device)), dim=1)[:, 1].cpu().numpy()
        all_prob.append(prob)
        all_true.append(yb.numpy())
    prob = np.concatenate(all_prob)
    true = np.concatenate(all_true)
    pred = (prob >= 0.5).astype(np.int64)
    return {
        "acc": accuracy_score(true, pred),
        "precision": precision_score(true, pred, zero_division=0),
        "recall": recall_score(true, pred, zero_division=0),
        "f1": f1_score(true, pred, zero_division=0),
        "auc": roc_auc_score(true, prob) if len(np.unique(true)) > 1 else float("nan"),
    }


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Tunable peptide-level CNN trainer",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    # data
    d = p.add_argument_group("data")
    d.add_argument("--pos", default=DEFAULT_POS)
    d.add_argument("--neg", default=DEFAULT_NEG)
    d.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN, help="pad/truncate length")
    d.add_argument("--val-frac", type=float, default=0.2)

    # architecture
    a = p.add_argument_group("architecture")
    a.add_argument("--conv-channels", type=int_list, default="128,128",
                   help="channels per conv layer; #entries = #conv layers, e.g. 64,128,256")
    a.add_argument("--kernels", type=int_list, default="3",
                   help="kernel size(s); multiple => parallel multi-kernel block, e.g. 3,5,7")
    a.add_argument("--pool", choices=["max", "mean", "maxmean"], default="max",
                   help="global pooling over the length dimension")
    a.add_argument("--fc-hidden", type=int_list, default="",
                   help="extra dense layer widths before the output, e.g. 64 or 128,32")
    a.add_argument("--dropout", type=float, default=0.3)
    a.add_argument("--no-batchnorm", dest="batchnorm", action="store_false",
                   help="disable BatchNorm")
    a.add_argument("--activation", choices=list(ACTIVATIONS), default="relu")

    # optimization
    o = p.add_argument_group("optimization")
    o.add_argument("--epochs", type=int, default=60)
    o.add_argument("--lr", type=float, default=1e-3)
    o.add_argument("--batch-size", type=int, default=64)
    o.add_argument("--weight-decay", type=float, default=1e-4, help="L2 regularization")
    o.add_argument("--patience", type=int, default=0,
                   help="early-stop after N epochs without val-F1 improvement (0 = off)")
    o.add_argument("--no-class-weight", dest="class_weight", action="store_false",
                   help="disable inverse-frequency class weighting in the loss")
    o.add_argument("--seed", type=int, default=42)
    o.add_argument("--out", default="models/residue_cnn.pt")
    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("Building dataset (ESM2 embeddings, cached)...")
    X, y = build_dataset(args.pos, args.neg, max_len=args.max_len)
    print(f"  X={X.shape} y={y.shape} positives={int(y.sum())} negatives={int((y == 0).sum())}")

    idx = np.arange(len(y))
    itr, iv = train_test_split(idx, test_size=args.val_frac, stratify=y, random_state=args.seed)
    print(f"  train={len(itr)} val={len(iv)}")

    Xtr, Xv = X[itr], X[iv]
    ytr, yv = y[itr], y[iv]
    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xtr), torch.from_numpy(ytr)),
        batch_size=args.batch_size, shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.from_numpy(Xv), torch.from_numpy(yv)),
        batch_size=256, shuffle=False,
    )

    if args.class_weight:
        n_pos = int(ytr.sum())
        n_neg = len(ytr) - n_pos
        n_total = n_pos + n_neg
        weight = torch.tensor(
            [n_total / (2 * n_neg), n_total / (2 * n_pos)], dtype=torch.float32, device=device
        )
        print(f"class weights (neg, pos): {weight.tolist()}")
    else:
        weight = None

    model = FlexibleCNN(
        in_channels=EMBED_DIM,
        conv_channels=args.conv_channels,
        kernel_sizes=args.kernels,
        pool=args.pool,
        fc_hidden=args.fc_hidden,
        dropout=args.dropout,
        batchnorm=args.batchnorm,
        activation=args.activation,
    ).to(device)
    print(f"model config: {model.config}")
    print(f"trainable parameters: {model.num_parameters():,}\n")

    criterion = nn.CrossEntropyLoss(weight=weight)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_f1 = -1.0
    best_metrics = None
    best_state = None
    epochs_since_best = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(yb)
            n_seen += len(yb)
        train_loss = running / max(n_seen, 1)
        m = evaluate(model, val_loader, device)
        print(
            f"epoch {epoch:3d} | loss {train_loss:.4f} | val acc {m['acc']:.3f} "
            f"P {m['precision']:.3f} R {m['recall']:.3f} F1 {m['f1']:.3f} "
            f"AUC {m['auc']:.3f}"
        )
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_metrics = m
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            epochs_since_best = 0
        else:
            epochs_since_best += 1
            if args.patience and epochs_since_best >= args.patience:
                print(f"early stopping at epoch {epoch} (no val-F1 gain for {args.patience} epochs)")
                break

    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.config,
            "arch": "cnn",
            "features": "esm2",
            "max_len": args.max_len,
            "val_metrics": best_metrics,
        },
        args.out,
    )
    print(f"\nbest val F1 {best_metrics['f1']:.3f} (AUC {best_metrics['auc']:.3f}) -> saved {args.out}")


if __name__ == "__main__":
    main()
