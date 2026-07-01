# Amyloid Core CNN

A peptide-level predictor of **amyloid core** regions built on frozen **ESM2-35M** protein
embeddings and a configurable 1D CNN. The final model is trained with a **two-phase protocol**:
self-supervised contrastive pre-training (SimCLR) of the CNN encoder, followed by a supervised
classifier head — a design chosen to cope with the severe class imbalance (~1:10.5) and the
small number of positive peptides.

At inference the peptide model is applied to every residue of a full protein via a stride-1
`2k+1` sliding window, producing a per-residue core probability; proteins with a long enough
run of high-probability residues are labelled **AMYLOID**.

---

## How it works

### Features — ESM2-35M embeddings
Each peptide is embedded once by the frozen `esm2_t12_35M_UR50D` model into per-residue
representations of shape `(L, 480)`, then zero-padded to a fixed `(480, max_len=20)` tensor
(channels-first). Embeddings are cached on disk keyed by sequence, so re-runs and benchmark
evaluation reuse them for free. See [`src/esm_embed.py`](src/esm_embed.py) and
[`src/data.py`](src/data.py).

### Model — `FlexibleCNN`
A fully configurable multi-kernel 1D CNN ([`src/flexible_cnn.py`](src/flexible_cnn.py)): the
number/width of conv layers, kernel size(s) (multiple sizes → parallel inception-style
branches), global pooling (`max` / `mean` / `maxmean`), optional dense head, batch-norm,
activation and dropout are all constructor arguments. The final model uses a single conv block
with `conv_channels=(256,)`, `kernel_sizes=(3, 5)` and `maxmean` pooling → a 512-d encoding.

### Two-phase training
The two-phase model code lives in one place — [`src/simclr.py`](src/simclr.py) — and is shared
by both training and inference.

- **Phase 1 — SimCLR (self-supervised).** A `SimCLREncoder` (the `FlexibleCNN` backbone + global
  pool, no classification head) is trained with the NT-Xent contrastive loss on two augmented
  views (random position masking / Gaussian noise) of each peptide embedding. Labels are never
  used in the loss, so the encoder sees all data regardless of class balance.
- **Phase 2 — classifier head (supervised).** The encoder is frozen and a single
  `Linear(512 → 1)` head (`AmyloidClassifier`) is trained with label-smoothed BCE on a
  class-resampled set (oversample positives 2× with noise + undersample negatives to a 1:2
  ratio). 5-fold cross-validation is run for diagnostics, then the final model is retrained on
  the full resampled training split and validated on a 30% stratified hold-out.

The reference protocol is in [`notebooks/Train_protocol.ipynb`](notebooks/Train_protocol.ipynb);
[`src/train.py`](src/train.py) reproduces it as a command-line tool.

---

## Project layout

```
src/
  esm_embed.py        Frozen ESM2-35M per-residue embeddings + on-disk cache
  data.py             build_dataset(): peptides -> padded (N, 480, 20) tensors
  flexible_cnn.py     Configurable multi-kernel 1D CNN
  simclr.py           Shared two-phase model: encoder, classifier, losses, resampling
  train.py            Two-phase training CLI (reproduces Train_protocol.ipynb)
  predict.py          Full-protein sliding-window scoring + benchmark evaluation
  train_tune.py       Single-config CNN trainer with full CLI hyperparameters
  optuna_search.py    Optuna architecture/hyperparameter search
  model.py            Legacy ResidueCNN (kept for old checkpoints)
notebooks/
  Train_protocol.ipynb      The final model's two-phase training protocol
  analyze_chosen_model.ipynb Benchmark metrics + model-size analysis of the final model
New_dataset/
  Training/           Positive core peptides + negative peptides (JSON)
  Benchmark/          Full-protein classification benchmark (JSON)
models/
  chosen_model/Final_model.pt   Trained two-phase checkpoint
results/              Benchmark residue-score CSVs
```

---

## Setup

Requires Python 3.12.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

The first embedding call downloads the ESM2-35M weights and populates `cache/` (git-ignored).

---

## Usage

### Train the final model (two-phase protocol)

```bash
# Full protocol with defaults -> models/simclr_encoder.pt + models/amyloid_classifier.pt
python -m src.train

# Reuse an existing SSL encoder (skip Phase 1), only train the classifier head
python -m src.train --encoder models/simclr_encoder.pt

# Quick run without cross-validation
python -m src.train --kfold 0 --out models/amyloid_classifier.pt
```

Key flags: `--ssl-epochs/-lr/-batch` (Phase 1), `--p2-epochs/-lr/-batch` (Phase 2),
`--kfold N` (0 disables CV), `--val-frac`, `--encoder` / `--encoder-out`, `--out`, `--seed`.
Run `python -m src.train --help` for the full list.

### Evaluate on the benchmark (full-protein sliding window)

```bash
python -m src.predict --checkpoint models/chosen_model/Final_model.pt --k 7
```

This scores every residue of each benchmark protein (ESM2 computed live per window) and reports
residue-level metrics plus protein-level AMYLOID vs NONAMYLOID classification. Flags:
`--threshold` (per-residue probability cutoff), `--min-core-len` (consecutive residues above the
threshold required to call a protein AMYLOID).

### Hyperparameter search (optional)

```bash
python -m src.optuna_search --metric f1 --n-trials 60 --epochs 40
```

---

## Checkpoint formats

`src/predict.py::load_model` auto-detects three checkpoint types by their keys:

| Keys present | Model | Loss / output |
|---|---|---|
| `enc_dim`, `cnn_config` | `AmyloidClassifier` (two-phase) | BCE, `(B,)` logit → sigmoid |
| `config` | `FlexibleCNN` | cross-entropy, `(B, 2)` → softmax |
| `in_channels` | legacy `ResidueCNN` | cross-entropy |

In all cases the usable output is one probability per window: **P(this 2k+1 window is an amyloid
core)**.
