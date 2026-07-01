"""Frozen ESM2-35M per-residue embeddings with on-disk caching.

ESM2-35M (`esm2_t12_35M_UR50D`) is used purely as a fixed feature extractor: each
sequence is run through the model once and the last-layer (layer 12) per-residue
representations are returned, shape ``(L, 480)``. Embeddings are cached to disk keyed
by sequence so re-runs and benchmark evaluation reuse them for free.
"""

from __future__ import annotations

import argparse
import hashlib
import os

import torch

# ESM2-35M: 12 transformer layers, 480-dim embeddings.
ESM2_MODEL_NAME = "esm2_t12_35M_UR50D"
ESM2_REPR_LAYER = 12
EMBED_DIM = 480

_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache", "esm2_35M")

# Lazily-initialised singletons (the model is heavy to load).
_model = None
_batch_converter = None
_device = None


def _get_model():
    """Load and cache the frozen ESM2-35M model, alphabet and batch converter."""
    global _model, _batch_converter, _device
    if _model is None:
        import esm  # imported lazily so `--help` etc. don't pay the import cost

        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, alphabet = esm.pretrained.esm2_t12_35M_UR50D()
        model = model.to(_device).eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _model = model
        _batch_converter = alphabet.get_batch_converter()
    return _model, _batch_converter, _device


def _cache_path(sequence: str) -> str:
    digest = hashlib.sha1(sequence.encode("utf-8")).hexdigest()
    return os.path.join(_CACHE_DIR, f"{digest}.pt")


@torch.no_grad()
def embed_sequence(sequence: str, use_cache: bool = True) -> torch.Tensor:
    """Return last-layer per-residue ESM2 embeddings for ``sequence``.

    Shape ``(L, 480)`` where ``L == len(sequence)`` (BOS/EOS tokens dropped).
    """
    if use_cache:
        path = _cache_path(sequence)
        if os.path.exists(path):
            return torch.load(path, map_location="cpu")

    model, batch_converter, device = _get_model()
    _, _, tokens = batch_converter([("seq", sequence)])
    tokens = tokens.to(device)
    out = model(tokens, repr_layers=[ESM2_REPR_LAYER], return_contacts=False)
    # Drop BOS (index 0) and EOS (index len+1); keep the L residue positions.
    reps = out["representations"][ESM2_REPR_LAYER][0, 1 : len(sequence) + 1].cpu()

    if use_cache:
        os.makedirs(_CACHE_DIR, exist_ok=True)
        torch.save(reps, _cache_path(sequence))
    return reps


def embed_many(sequences, use_cache: bool = True):
    """Embed a list of sequences, returning a list of ``(L, 480)`` tensors."""
    return [embed_sequence(s, use_cache=use_cache) for s in sequences]


@torch.no_grad()
def embed_batch(sequences, use_cache: bool = False, batch_size: int = 64):
    """Embed many sequences in batches, returning a list of ``(L, 480)`` tensors.

    Used for benchmark inference where ESM2 is computed **live** (``use_cache=False``)
    over the sliding-window subsequences. With ``use_cache=True`` it doubles as a fast
    bulk embedder for the training peptides.
    """
    results = [None] * len(sequences)
    todo = []
    if use_cache:
        for idx, s in enumerate(sequences):
            path = _cache_path(s)
            if os.path.exists(path):
                results[idx] = torch.load(path, map_location="cpu")
            else:
                todo.append(idx)
    else:
        todo = list(range(len(sequences)))

    if todo:
        model, batch_converter, device = _get_model()
        for b in range(0, len(todo), batch_size):
            idxs = todo[b : b + batch_size]
            data = [(str(i), sequences[i]) for i in idxs]
            _, _, tokens = batch_converter(data)
            tokens = tokens.to(device)
            out = model(tokens, repr_layers=[ESM2_REPR_LAYER], return_contacts=False)
            reps = out["representations"][ESM2_REPR_LAYER].cpu()
            for j, i in enumerate(idxs):
                L = len(sequences[i])
                r = reps[j, 1 : L + 1]
                results[i] = r
                if use_cache:
                    os.makedirs(_CACHE_DIR, exist_ok=True)
                    torch.save(r, _cache_path(sequences[i]))
    return results


def _self_test() -> None:
    seq = "LSIGTGRAML"
    reps = embed_sequence(seq)
    assert reps.shape == (len(seq), EMBED_DIM), reps.shape
    print(f"embedded {seq!r} -> shape {tuple(reps.shape)}")
    # Second call should hit the cache and match.
    reps2 = embed_sequence(seq)
    assert torch.allclose(reps, reps2)
    print(f"cache hit OK at {_cache_path(seq)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ESM2-35M embedding utility")
    parser.add_argument("--self-test", action="store_true", help="embed one peptide and verify caching")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
    else:
        parser.print_help()
