"""Build peptide-level training samples from frozen ESM2 embeddings.

Each training peptide is embedded whole by ESM2 into a per-residue matrix ``(L, 480)``
and zero-padded (right) to a fixed length (default 20, the max peptide length) so every
input is ``(480, max_len)`` and batches cleanly. The label is the peptide-level label:
every positive peptide is an amyloid core, every negative peptide is not.

There is **no windowing at training time** — the ``2k+1`` sliding window is applied only
during benchmark inference (see ``src/predict.py``).
"""

from __future__ import annotations

import json

import numpy as np

from .esm_embed import EMBED_DIM, embed_batch

# Max training-peptide length is 20; pad all inputs to this fixed width.
DEFAULT_MAX_LEN = 20


def load_peptides(path: str):
    """Load a training JSON file -> list of (core_id, sequence)."""
    with open(path) as fh:
        data = json.load(fh)
    return [(d["core_id"], d["sequence"]) for d in data]


def pad_to(emb: np.ndarray, max_len: int) -> np.ndarray:
    """Zero-pad/truncate a ``(L, D)`` embedding to channels-first ``(D, max_len)``."""
    L, D = emb.shape
    out = np.zeros((D, max_len), dtype=np.float32)
    n = min(L, max_len)
    out[:, :n] = emb[:n].T
    return out


def build_dataset(pos_path: str, neg_path: str, max_len: int = DEFAULT_MAX_LEN, use_cache: bool = True):
    """Build the peptide-level dataset from positive and negative JSON files.

    Returns ``(X, y)``:
      X: ``(N, 480, max_len)`` float32 padded embeddings (one row per peptide)
      y: ``(N,)`` int64 labels (1 = amyloid core peptide, 0 = negative)
    """
    pos = load_peptides(pos_path)
    neg = load_peptides(neg_path)
    seqs = [s for _, s in pos] + [s for _, s in neg]
    labels = [1] * len(pos) + [0] * len(neg)

    embs = embed_batch(seqs, use_cache=use_cache)
    X = np.stack([pad_to(e.numpy().astype(np.float32), max_len) for e in embs])
    y = np.array(labels, dtype=np.int64)
    return X, y


def _self_test() -> None:
    from .esm_embed import embed_sequence

    seq = "ACDEFGHIK"
    emb = embed_sequence(seq).numpy()
    assert emb.shape == (len(seq), EMBED_DIM), emb.shape

    padded = pad_to(emb, DEFAULT_MAX_LEN)
    assert padded.shape == (EMBED_DIM, DEFAULT_MAX_LEN), padded.shape
    np.testing.assert_allclose(padded[:, : len(seq)], emb.T, rtol=1e-5)
    assert np.all(padded[:, len(seq):] == 0.0), "tail not zero-padded"
    print(f"padded shape {padded.shape} OK; residues + zero-pad asserts passed")


if __name__ == "__main__":
    _self_test()
