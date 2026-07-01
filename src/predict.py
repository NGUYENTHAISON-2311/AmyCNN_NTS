"""Full-protein scoring and benchmark evaluation.

The peptide-level CNN is applied to each full protein with a stride-1 ``2k+1`` sliding
window: for every residue *i*, the subsequence ``seq[max(0,i-k):i+k+1]`` (clipped at the
protein ends) is embedded **live** by ESM2 (no cache), padded to the model's ``max_len``,
and classified; the resulting probability is attributed to the center residue *i*.

Two evaluations are reported:

* residue-level: predicted per-residue core probability vs the ground-truth mask built
  from each protein's ``matched_core_regions``;
* protein-level: AMYLOID / NONAMYLOID classification, where a protein is AMYLOID if it
  contains a run of >= ``min_core_len`` consecutive residues above ``threshold``.
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .data import DEFAULT_MAX_LEN, pad_to
from .esm_embed import embed_batch
from .model import ResidueCNN
from .simclr import AmyloidClassifier, SimCLREncoder

DEFAULT_BENCHMARK = "New_dataset/Benchmark/new_representative_classification_benchmark_KA.json"


# ── Model loading ─────────────────────────────────────────────────────────────

def load_model(checkpoint: str, device):
    ckpt = torch.load(checkpoint, map_location=device, weights_only=False)
    if "enc_dim" in ckpt:
        # Two-phase SimCLR checkpoint (AmyloidClassifier) — see src/simclr.py
        encoder = SimCLREncoder(ckpt["cnn_config"])
        model   = AmyloidClassifier(encoder, ckpt["enc_dim"])
    elif "config" in ckpt:
        from .flexible_cnn import FlexibleCNN
        model = FlexibleCNN.from_config(ckpt["config"])
    else:
        # Legacy ResidueCNN checkpoint
        model = ResidueCNN(in_channels=ckpt["in_channels"], dropout=ckpt.get("dropout", 0.3))
    model.load_state_dict(ckpt["state_dict"])
    model = model.to(device).eval()
    max_len = ckpt.get("max_len", DEFAULT_MAX_LEN)
    return model, max_len


# ── Scoring ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def score_protein(seq: str, model, k: int, max_len: int, device, batch_size: int = 256) -> np.ndarray:
    """Per-residue core probability via centered ``2k+1`` windows (clipped at protein ends).

    For every residue *i*, the window ``seq[max(0,i-k):i+k+1]`` is embedded live by ESM2,
    zero-padded to ``(480, max_len)`` channels-first, and classified by the CNN. Returns a
    vector of length ``len(seq)``.

    Handles both checkpoint formats automatically:
    - ``_AmyloidClassifier`` (SimCLR) outputs ``(B,)`` logits → sigmoid
    - ``FlexibleCNN`` outputs ``(B, 2)`` logits → softmax[:, 1]
    """
    L = len(seq)
    windows = [seq[max(0, i - k):i + k + 1] for i in range(L)]
    embs = embed_batch(windows, use_cache=False, batch_size=batch_size)
    X = np.stack([pad_to(e.numpy().astype(np.float32), max_len) for e in embs])
    probs = []
    for start in range(0, len(X), batch_size):
        xb  = torch.from_numpy(X[start:start + batch_size]).to(device)
        out = model(xb)
        p   = (torch.sigmoid(out) if out.dim() == 1
               else torch.softmax(out, dim=1)[:, 1]).cpu().numpy()
        probs.append(p)
    return np.concatenate(probs)


def core_mask(protein: dict) -> np.ndarray:
    """Ground-truth per-residue core mask from matched_core_regions (end-exclusive)."""
    L = len(protein["Sequence"])
    mask = np.zeros(L, dtype=np.int64)
    for r in protein["matched_core_regions"]:
        mask[r["start"]: r["end"]] = 1
    return mask


def longest_positive_run(binary: np.ndarray) -> int:
    best = cur = 0
    for v in binary:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def main() -> None:
    parser = argparse.ArgumentParser(description="Full-protein scoring + benchmark eval")
    parser.add_argument("--checkpoint", default="models/residue_cnn.pt")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--k", type=int, default=7, help="window half-width (window = 2k+1)")
    parser.add_argument("--threshold", type=float, default=0.5, help="per-residue core prob threshold")
    parser.add_argument("--min-core-len", type=int, default=5, help="min consecutive run => AMYLOID")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, max_len = load_model(args.checkpoint, device)
    if 2 * args.k + 1 > max_len:
        print(f"warning: window {2 * args.k + 1} > model max_len {max_len}; windows will be truncated")
    print(f"loaded {args.checkpoint} (max_len={max_len}, k={args.k}) on {device}; ESM2 computed live")

    with open(args.benchmark) as fh:
        proteins = json.load(fh)

    res_true, res_prob = [], []
    prot_true, prot_pred = [], []
    for p in proteins:
        prob = score_protein(p["Sequence"], model, args.k, max_len, device)
        mask = core_mask(p)
        res_true.append(mask)
        res_prob.append(prob)

        run = longest_positive_run(prob >= args.threshold)
        prot_pred.append(1 if run >= args.min_core_len else 0)
        prot_true.append(1 if p["LABEL"] == "AMYLOID" else 0)

    # --- residue-level metrics ---
    rt = np.concatenate(res_true)
    rp = np.concatenate(res_prob)
    rpred = (rp >= args.threshold).astype(np.int64)
    print("\n=== Residue-level (all benchmark proteins) ===")
    print(f"  residues: {len(rt)} | core: {int(rt.sum())} | non-core: {int((rt == 0).sum())}")
    print(f"  precision {precision_score(rt, rpred, zero_division=0):.3f}")
    print(f"  recall    {recall_score(rt, rpred, zero_division=0):.3f}")
    print(f"  F1        {f1_score(rt, rpred, zero_division=0):.3f}")
    if len(np.unique(rt)) > 1:
        print(f"  ROC-AUC   {roc_auc_score(rt, rp):.3f}")

    # --- protein-level classification ---
    pt = np.array(prot_true)
    pp = np.array(prot_pred)
    print("\n=== Protein-level classification (AMYLOID vs NONAMYLOID) ===")
    print(f"  accuracy  {(pt == pp).mean():.3f}  ({int((pt == pp).sum())}/{len(pt)})")
    print(f"  precision {precision_score(pt, pp, zero_division=0):.3f}")
    print(f"  recall    {recall_score(pt, pp, zero_division=0):.3f}")
    print(f"  F1        {f1_score(pt, pp, zero_division=0):.3f}")
    cm = confusion_matrix(pt, pp, labels=[0, 1])
    print("  confusion matrix [rows=true NON/AMY, cols=pred NON/AMY]:")
    print(f"    {cm.tolist()}")


if __name__ == "__main__":
    main()
