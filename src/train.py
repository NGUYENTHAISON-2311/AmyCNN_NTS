"""Two-phase training tool for the amyloid-core model (reproduces Train_protocol.ipynb).

Phase 1 — SimCLR self-supervised encoder on the full 70% train split (labels unused).
Phase 2 — freeze the encoder, train a Linear(ENC_DIM, 1) classifier head on the
class-resampled 70% (5-fold CV for diagnostics, then a final model validated on the
30% hold-out), keeping the best-val-F1 weights.

The final checkpoint is ``src/predict.py`` compatible, so benchmark inference runs on it
directly. All model/loss/resampling code lives in :mod:`src.simclr`.

Examples
--------
# Full protocol with defaults -> models/simclr_encoder.pt + models/amyloid_classifier.pt
python -m src.train

# Reuse an existing SSL encoder (skip Phase 1) and only train the classifier head
python -m src.train --encoder models/simclr_encoder.pt

# Fast smoke run, no cross-validation
python -m src.train --ssl-epochs 2 --p2-epochs 2 --kfold 0 \
    --encoder-out /tmp/enc.pt --out /tmp/clf.pt
"""

from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data import DEFAULT_MAX_LEN, build_dataset
from .simclr import (
    CNN_CONFIG,
    LABEL_SMOOTH_EPS,
    AmyloidClassifier,
    ProjectionHead,
    SimCLRDataset,
    SimCLREncoder,
    bce_smooth_loss,
    enc_dim_for,
    evaluate,
    make_loader,
    nt_xent_loss,
    resample,
    resample_split,
)
from torch.utils.data import DataLoader

DEFAULT_POS = "New_dataset/Training/train_core_set_KA.json"
DEFAULT_NEG = "New_dataset/Training/new_negative.json"


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


# ── Phase 1: SimCLR encoder ────────────────────────────────────────────────────

def train_encoder(X_tr, cnn_config, enc_dim, device, args) -> dict:
    """Train the SimCLR encoder on the full train split; return its checkpoint dict."""
    encoder   = SimCLREncoder(cnn_config).to(device)
    proj_head = ProjectionHead(enc_dim).to(device)
    loader    = DataLoader(SimCLRDataset(X_tr), batch_size=args.ssl_batch,
                           shuffle=True, drop_last=True)
    print(f"  SSL batches/epoch: {len(loader)}  (train set, drop_last)")

    opt = torch.optim.Adam(list(encoder.parameters()) + list(proj_head.parameters()),
                           lr=args.ssl_lr, weight_decay=1e-4)

    for epoch in range(1, args.ssl_epochs + 1):
        encoder.train(); proj_head.train()
        total = 0.0
        for x1, x2 in loader:
            x1, x2 = x1.to(device), x2.to(device)
            loss = nt_xent_loss(proj_head(encoder(x1)), proj_head(encoder(x2)))
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        if epoch == 1 or epoch % 10 == 0 or epoch == args.ssl_epochs:
            print(f"  SSL [{epoch:3d}/{args.ssl_epochs}]  NT-Xent: {total/len(loader):.4f}")

    return {"encoder_state_dict": encoder.state_dict(),
            "cnn_config": cnn_config, "enc_dim": enc_dim}


# ── Phase 2: classifier head ───────────────────────────────────────────────────

def _fresh_classifier(ckpt_ssl, device) -> AmyloidClassifier:
    enc = SimCLREncoder(ckpt_ssl["cnn_config"]).to(device)
    enc.load_state_dict(ckpt_ssl["encoder_state_dict"])
    return AmyloidClassifier(enc, enc_dim=ckpt_ssl["enc_dim"], freeze_encoder=True).to(device)


def _train_head(clf, train_loader, val_loader, device, args):
    """Train a (frozen-encoder) classifier head; return (history, best_val_f1, best_state)."""
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, clf.parameters()),
                           lr=args.p2_lr, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="max", factor=0.5, patience=5, min_lr=1e-5)

    history = {"train_loss": [], "val_f1": [], "val_auc": [], "lr": []}
    best_f1, best_state = -1.0, None
    for epoch in range(1, args.p2_epochs + 1):
        clf.train()
        total = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = bce_smooth_loss(clf(xb), yb)
            opt.zero_grad(); loss.backward(); opt.step()
            total += loss.item()
        m = evaluate(clf, val_loader, device)
        history["train_loss"].append(total / len(train_loader))
        history["val_f1"].append(m["f1"])
        history["val_auc"].append(m["auc"])
        history["lr"].append(opt.param_groups[0]["lr"])
        sched.step(m["f1"])
        if m["f1"] > best_f1:
            best_f1    = m["f1"]
            best_state = copy.deepcopy({k: v.cpu() for k, v in clf.state_dict().items()})
    return history, best_f1, best_state


def cross_validate(X_tr, y_tr, ckpt_ssl, device, args):
    """StratifiedKFold on the train split; resample train fold only. Returns fold summaries."""
    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=args.seed)
    fold_metrics, fold_histories = [], []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_tr, y_tr), 1):
        print(f"\n  Fold {fold}/{args.kfold}  (train={len(tr_idx)}, val={len(val_idx)})")
        fold_rng = np.random.default_rng(args.seed + fold)
        X_f_res, y_f_res = resample_split(X_tr[tr_idx], y_tr[tr_idx], fold_rng)

        clf = _fresh_classifier(ckpt_ssl, device)
        tr_ldr  = make_loader(X_f_res, y_f_res, args.p2_batch, shuffle=True)
        val_ldr = make_loader(X_tr[val_idx], y_tr[val_idx], 256, shuffle=False)
        hist, best_f1, _ = _train_head(clf, tr_ldr, val_ldr, device, args)

        fold_metrics.append({"f1": best_f1, "auc": float(np.nanmax(hist["val_auc"]))})
        fold_histories.append(hist)
        print(f"    best val F1 {best_f1:.3f}  |  best val AUC {fold_metrics[-1]['auc']:.3f}")
    return fold_metrics, fold_histories


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Two-phase SimCLR + classifier trainer (Train_protocol.ipynb)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--pos", default=DEFAULT_POS)
    p.add_argument("--neg", default=DEFAULT_NEG)
    p.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    p.add_argument("--val-frac", type=float, default=0.30, help="held-out validation fraction")
    # Phase 1
    p.add_argument("--ssl-epochs", type=int, default=100)
    p.add_argument("--ssl-lr", type=float, default=1e-3)
    p.add_argument("--ssl-batch", type=int, default=16)
    p.add_argument("--encoder", default=None,
                   help="reuse an existing SSL encoder checkpoint and skip Phase 1")
    p.add_argument("--encoder-out", default="models/simclr_encoder.pt")
    # Phase 2
    p.add_argument("--p2-epochs", type=int, default=30)
    p.add_argument("--p2-lr", type=float, default=1e-3)
    p.add_argument("--p2-batch", type=int, default=32)
    p.add_argument("--kfold", type=int, default=5, help="StratifiedKFold folds (0 disables CV)")
    p.add_argument("--out", default="models/amyloid_classifier.pt")
    p.add_argument("--seed", type=int, default=42)
    return p


def main() -> None:
    args = build_argparser().parse_args()
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print("Building dataset (ESM2 embeddings, cached)...")
    X, y = build_dataset(args.pos, args.neg, max_len=args.max_len)
    n_pos, n_neg = int(y.sum()), int((y == 0).sum())
    print(f"  X={X.shape}  positives={n_pos}  negatives={n_neg}  ratio={n_neg/max(n_pos,1):.1f}:1")

    idx_tr, idx_val = train_test_split(
        np.arange(len(y)), test_size=args.val_frac, stratify=y, random_state=args.seed)
    X_tr, y_tr = X[idx_tr], y[idx_tr]
    X_ho, y_ho = X[idx_val], y[idx_val]
    print(f"  train={len(y_tr)} (pos={int(y_tr.sum())})  "
          f"hold-out={len(y_ho)} (pos={int(y_ho.sum())})")

    cnn_config = dict(CNN_CONFIG)
    cnn_config["in_channels"] = X.shape[1]
    enc_dim = enc_dim_for(cnn_config)
    print(f"  ENC_DIM={enc_dim}")

    # ── Phase 1 ────────────────────────────────────────────────────────────────
    if args.encoder:
        print(f"\nPhase 1: reusing encoder {args.encoder} (Phase 1 skipped)")
        ckpt_ssl = torch.load(args.encoder, map_location=device)
    else:
        print("\nPhase 1: SimCLR self-supervised encoder ...")
        ckpt_ssl = train_encoder(X_tr, cnn_config, enc_dim, device, args)
        os.makedirs(os.path.dirname(args.encoder_out) or ".", exist_ok=True)
        torch.save(ckpt_ssl, args.encoder_out)
        print(f"  encoder saved -> {args.encoder_out}")

    # ── Phase 2: K-fold CV (diagnostic) ────────────────────────────────────────
    cv_mean_f1 = cv_std_f1 = float("nan")
    fold_histories = []
    if args.kfold and args.kfold > 1:
        print(f"\nPhase 2: {args.kfold}-fold cross-validation ...")
        fold_metrics, fold_histories = cross_validate(X_tr, y_tr, ckpt_ssl, device, args)
        cv_mean_f1 = float(np.mean([m["f1"] for m in fold_metrics]))
        cv_std_f1  = float(np.std([m["f1"] for m in fold_metrics]))
        cv_mean_auc = float(np.mean([m["auc"] for m in fold_metrics]))
        cv_std_auc  = float(np.std([m["auc"] for m in fold_metrics]))
        print(f"\n  {args.kfold}-Fold CV   F1 {cv_mean_f1:.3f} +/- {cv_std_f1:.3f}   "
              f"AUC {cv_mean_auc:.3f} +/- {cv_std_auc:.3f}")

    # ── Phase 2: final model on all resampled train, validated on hold-out ─────
    print("\nPhase 2: final model on full resampled train set ...")
    rng = np.random.default_rng(args.seed)
    X_res, y_res = resample_split(X_tr, y_tr, rng)
    classifier = _fresh_classifier(ckpt_ssl, device)
    train_loader = make_loader(X_res, y_res, args.p2_batch, shuffle=True)
    ho_loader    = make_loader(X_ho, y_ho, 256, shuffle=False)
    history, best_f1, best_state = _train_head(classifier, train_loader, ho_loader, device, args)
    classifier.load_state_dict(best_state)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    torch.save({
        "state_dict"      : classifier.state_dict(),
        "cnn_config"      : ckpt_ssl["cnn_config"],
        "enc_dim"         : ckpt_ssl["enc_dim"],
        "max_len"         : args.max_len,
        "label_smooth_eps": LABEL_SMOOTH_EPS,
        "best_val_f1"     : best_f1,
        "cv_mean_f1"      : cv_mean_f1,
        "cv_std_f1"       : cv_std_f1,
        "history"         : history,
        "fold_histories"  : fold_histories,
    }, args.out)
    print(f"\nFinal hold-out F1: {best_f1:.3f}  ->  saved {args.out}")


if __name__ == "__main__":
    main()
