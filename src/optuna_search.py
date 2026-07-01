"""Optuna hyperparameter search for the peptide-level amyloid-core CNN.

Searches the full architecture + optimization space of :class:`FlexibleCNN`:

  architecture : layer layout (2 conv, or 1 conv + 1 fc), channels per layer,
                 kernel size(s) / multi-kernel, global pooling, dense-head width,
                 dropout, batch-norm, activation
  optimization : learning rate, batch size, weight decay (L2)

The search is **single-objective**: each candidate is trained on a fixed stratified split
(the ESM2 embeddings are built once and reused) and scored on one validation metric —
either **F1** or **AUROC** — controlled by ``--metric``.  A TPE sampler proposes configs;
a :class:`~optuna.pruners.MedianPruner` prunes unpromising trials based on the per-epoch
metric value.  The best trial is retrained and saved.  Hyperparameter importances (fANOVA)
are printed after the search.

After the search the chosen config is retrained and saved as a normal checkpoint, so
``src/predict.py`` can run the live-ESM2 benchmark on it directly.

Examples
--------
python -m src.optuna_search --metric f1 --n-trials 60 --epochs 40
python -m src.optuna_search --metric auroc --n-trials 200 --storage sqlite:///optuna.db \\
    --study-name amyloid --out models/best.pt
"""

from __future__ import annotations

import argparse
import copy
import os

import numpy as np
import optuna
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from .data import DEFAULT_MAX_LEN, build_dataset
from .esm_embed import EMBED_DIM
from .flexible_cnn import ACTIVATIONS, FlexibleCNN
from .train_tune import DEFAULT_NEG, DEFAULT_POS, evaluate, int_list, set_seed

# Populated once in main() and reused by every trial.
_DEVICE = None
_XTR = _YTR = _XV = _YV = None
_CLASS_WEIGHT = None
_EPOCHS = 40
_PATIENCE = 10
_SEED = 42
_METRIC = "f1"          # "f1" or "auroc" — set by --metric

CHANNEL_CHOICES = [32, 64, 128, 256]
KERNEL_CHOICES = [ "3","5", "7", "9"]
FC_CHOICES = [32, 64, 128]
BATCH_CHOICES = [32, 64, 128]
LAYOUT_CHOICES = [ "1conv_1fc", "2conv"]


def layout_counts(layout: str):
    return {"2conv": (2, 0), "1conv_1fc": (1, 1), "1conv": (1, 0)}[layout]


def _metric_value(m: dict) -> float:
    """Extract the scalar we are optimising from an evaluate() metrics dict."""
    if _METRIC == "auroc":
        v = m["auc"]
        return float(v) if not np.isnan(v) else 0.0
    return float(m["f1"])


def decode_params(params: dict):
    """Turn a flat Optuna params dict into model kwargs + optim settings."""
    n_conv, n_fc = layout_counts(params["layout"])
    conv_channels = tuple(params[f"ch_l{i}"] for i in range(n_conv))
    kernels = int_list(params["kernels"])
    fc_hidden = tuple(params[f"fc_l{i}"] for i in range(n_fc))
    model_kwargs = dict(
        in_channels=EMBED_DIM,
        conv_channels=conv_channels,
        kernel_sizes=kernels,
        pool=params["pool"],
        fc_hidden=fc_hidden,
        dropout=params["dropout"],
        batchnorm=params["batchnorm"],
        activation=params["activation"],
    )
    return model_kwargs, params["lr"], params["batch_size"], params["weight_decay"]


def fit_model(model_kwargs, lr, batch_size, weight_decay, epochs, patience=None,
              trial: "optuna.Trial | None" = None):
    """Train one config; return (best_metrics_dict, best_state_dict_on_cpu).

    The best epoch maximises the chosen metric (``_METRIC``).  When ``trial`` is given,
    the per-epoch metric value is reported so MedianPruner can prune early.
    """
    train_loader = DataLoader(TensorDataset(_XTR, _YTR), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(_XV, _YV), batch_size=256, shuffle=False)

    model = FlexibleCNN(**model_kwargs).to(_DEVICE)
    criterion = nn.CrossEntropyLoss(weight=_CLASS_WEIGHT)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    pat = patience if patience is not None else _PATIENCE
    best_score = -1e9
    best_metrics = None
    best_state = None
    epochs_since_best = 0
    for epoch in range(1, epochs + 1):
        model.train()
        for xb, yb in train_loader:
            xb, yb = xb.to(_DEVICE), yb.to(_DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
        m = evaluate(model, val_loader, _DEVICE)
        score = _metric_value(m)
        if trial is not None:
            trial.report(score, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        if score > best_score:
            best_score = score
            best_metrics = m
            best_state = copy.deepcopy({k: v.cpu() for k, v in model.state_dict().items()})
            epochs_since_best = 0
        else:
            epochs_since_best += 1
        if pat and epochs_since_best >= pat:
            break
    return best_metrics, best_state


def objective(trial: optuna.Trial) -> float:
    layout = trial.suggest_categorical("layout", LAYOUT_CHOICES)
    n_conv, n_fc = layout_counts(layout)
    for i in range(n_conv):
        trial.suggest_categorical(f"ch_l{i}", CHANNEL_CHOICES)
    for i in range(n_fc):
        trial.suggest_categorical(f"fc_l{i}", FC_CHOICES)
    trial.suggest_categorical("kernels", KERNEL_CHOICES)
    trial.suggest_categorical("pool", ["max", "mean", "maxmean"])
    trial.suggest_float("dropout", 0.1, 0.6)
    trial.suggest_categorical("batchnorm", [True, False])
    trial.suggest_categorical("activation", list(ACTIVATIONS))
    trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    trial.suggest_categorical("batch_size", BATCH_CHOICES)
    trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)

    model_kwargs, lr, batch_size, weight_decay = decode_params(trial.params)
    best_metrics, _ = fit_model(model_kwargs, lr, batch_size, weight_decay, _EPOCHS, trial=trial)
    return _metric_value(best_metrics)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for the peptide-level CNN",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--metric", choices=["f1", "auroc"], default="f1",
                        help="validation metric to maximise during the search")
    parser.add_argument("--pos", default=DEFAULT_POS)
    parser.add_argument("--neg", default=DEFAULT_NEG)
    parser.add_argument("--max-len", type=int, default=DEFAULT_MAX_LEN)
    parser.add_argument("--val-frac", type=float, default=0.2)
    parser.add_argument("--n-trials", type=int, default=50)
    parser.add_argument("--timeout", type=float, default=None, help="search time budget (seconds)")
    parser.add_argument("--epochs", type=int, default=40, help="max epochs per trial")
    parser.add_argument("--final-epochs", type=int, default=None,
                        help="max epochs to retrain the best config (default: --epochs)")
    parser.add_argument("--patience", type=int, default=10,
                        help="early-stop after N epochs without improvement (0 = off)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--study-name", default=None)
    parser.add_argument("--storage", default=None, help="e.g. sqlite:///optuna.db to persist/resume")
    parser.add_argument("--out", default="models/best.pt")
    args = parser.parse_args()

    global _DEVICE, _XTR, _YTR, _XV, _YV, _CLASS_WEIGHT, _EPOCHS, _PATIENCE, _SEED, _METRIC
    set_seed(args.seed)
    _DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _EPOCHS = args.epochs
    _PATIENCE = args.patience
    _SEED = args.seed
    _METRIC = args.metric
    metric_label = "AUROC" if _METRIC == "auroc" else "F1"
    print(f"device: {_DEVICE} | optimising val {metric_label} (single-objective) "
          f"| early-stop patience {args.patience}")

    print("Building dataset (ESM2 embeddings, cached)...")
    X, y = build_dataset(args.pos, args.neg, max_len=args.max_len)
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=args.val_frac, stratify=y,
                                         random_state=args.seed)
    _XTR, _YTR = torch.from_numpy(Xtr), torch.from_numpy(ytr)
    _XV, _YV = torch.from_numpy(Xv), torch.from_numpy(yv)
    n_pos = int(ytr.sum())
    n_neg = len(ytr) - n_pos
    n_total = n_pos + n_neg
    _CLASS_WEIGHT = torch.tensor(
        [n_total / (2 * n_neg), n_total / (2 * n_pos)], dtype=torch.float32, device=_DEVICE
    )
    print(f"  X={X.shape} train={len(ytr)} val={len(yv)} positives(train)={n_pos}")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    # MedianPruner: prune a trial whose per-epoch metric is below the median of completed
    # trials at the same step (5-trial warm-up, 5-epoch burn-in, then checks every epoch).
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=5, interval_steps=1)
    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=bool(args.storage),
    )
    study.optimize(objective, n_trials=args.n_trials, timeout=args.timeout, show_progress_bar=False)

    print("\n=== Search complete ===")
    completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    pruned    = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    print(f"  trials total: {len(study.trials)} | completed: {len(completed)} "
          f"| pruned: {len(pruned)}")

    chosen = study.best_trial
    print(f"  best trial: #{chosen.number}  {metric_label} = {chosen.value:.4f}")
    print("  best params:")
    for k, v in chosen.params.items():
        print(f"    {k}: {v}")

    # --- Hyperparameter importance (fANOVA) ---
    if len(completed) >= 4:
        print(f"\n=== Hyperparameter importances  [{metric_label}] ===")
        try:
            imp = optuna.importance.get_param_importances(
                study,
                evaluator=optuna.importance.FanovaImportanceEvaluator(),
            )
            for param, importance in sorted(imp.items(), key=lambda x: x[1], reverse=True):
                bar = "#" * int(importance * 30)
                print(f"  {param:<20s} {importance:.4f}  {bar}")
        except Exception as exc:
            print(f"  importance skipped: {exc}")
    else:
        print("\n(hyperparameter importance skipped: need >=4 completed trials)")

    # Retrain the chosen config and save a predict-compatible checkpoint.
    final_epochs = args.final_epochs or args.epochs
    model_kwargs, lr, batch_size, weight_decay = decode_params(chosen.params)
    print(f"\nRetraining best config (max {final_epochs} epochs, patience {args.patience}) "
          f"-> {args.out}")
    best_metrics, best_state = fit_model(
        model_kwargs, lr, batch_size, weight_decay, final_epochs, patience=args.patience
    )
    model = FlexibleCNN(**model_kwargs)
    model.load_state_dict(best_state)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    auc_final = best_metrics["auc"] if not np.isnan(best_metrics["auc"]) else 0.0
    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": model.config,
            "arch": "cnn",
            "features": "esm2",
            "max_len": args.max_len,
            "val_metrics": {"auroc": auc_final, "f1": best_metrics["f1"]},
            "best_params": chosen.params,
            "search_metric": _METRIC,
        },
        args.out,
    )
    print(f"saved retrained model (val AUROC {auc_final:.4f}, F1 {best_metrics['f1']:.4f}) "
          f"to {args.out}")
    print(f"\nEvaluate on the benchmark with:\n  python -m src.predict --checkpoint {args.out} --k 7")


if __name__ == "__main__":
    main()
