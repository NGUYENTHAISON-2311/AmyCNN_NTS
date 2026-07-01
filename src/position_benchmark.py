"""Positional (per-residue) benchmark on the positive-only protein set.

Unlike the classification benchmark (is a protein amyloid?), this evaluates *where* the
amyloid cores are. Every protein here is AMYLOID, so protein-level accuracy is
meaningless; instead we score each residue with the live-ESM2 ``2k+1`` sliding window
and compare the predicted core positions against the ground-truth ``matched_core_regions``.

Reported metrics:
* residue-level (pooled over all proteins): precision / recall / F1 / ROC-AUC / PR-AUC
* per-protein residue F1 and ROC-AUC (macro view)
* region detection: a true core region counts as detected if at least ``overlap_frac``
  of its residues are predicted positive — this is the localization recall.

Optionally writes the predicted core regions per protein to a JSON file.

Example
-------
python -m src.position_benchmark --checkpoint models/best.pt --k 7 \
    --threshold 0.5 --min-core-len 5 --out predicted_positions.json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from .predict import core_mask, load_model, score_protein

DEFAULT_BENCHMARK = "New_dataset/Benchmark/new_representative_positive_benchmark_KA.json"


def extract_runs(binary: np.ndarray, min_len: int):
    """Return [start, end) spans of consecutive 1s with length >= min_len."""
    runs = []
    start = None
    for i, v in enumerate(binary):
        if v and start is None:
            start = i
        elif not v and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    if start is not None and len(binary) - start >= min_len:
        runs.append((start, len(binary)))
    return runs


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Positional (per-residue) benchmark on the positive protein set",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--benchmark", default=DEFAULT_BENCHMARK)
    parser.add_argument("--k", type=int, default=7, help="window half-width (window = 2k+1)")
    parser.add_argument("--threshold", type=float, default=0.5, help="per-residue core prob threshold")
    parser.add_argument("--min-core-len", type=int, default=5,
                        help="min consecutive positives to call a predicted core region")
    parser.add_argument("--overlap-frac", type=float, default=0.5,
                        help="fraction of a true region's residues that must be predicted "
                             "positive for it to count as detected")
    parser.add_argument("--out", default=None, help="optional JSON to save predicted regions")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, max_len = load_model(args.checkpoint, device)
    print(f"loaded {args.checkpoint} (max_len={max_len}, k={args.k}) on {device}; ESM2 computed live")

    with open(args.benchmark) as fh:
        proteins = json.load(fh)

    all_true, all_prob = [], []
    n_regions = n_detected = 0
    rows = []
    predictions = []
    for p in proteins:
        seq = p["Sequence"]
        prob = score_protein(seq, model, args.k, max_len, device)
        gt = core_mask(p)
        pred_mask = (prob >= args.threshold).astype(np.int64)

        all_true.append(gt)
        all_prob.append(prob)

        # Region-level detection (localization recall).
        for r in p["matched_core_regions"]:
            n_regions += 1
            seg = pred_mask[r["start"] : r["end"]]
            if seg.size and seg.mean() >= args.overlap_frac:
                n_detected += 1

        pred_runs = extract_runs(pred_mask, args.min_core_len)

        # Per-protein residue metrics.
        f1 = f1_score(gt, pred_mask, zero_division=0)
        auc = roc_auc_score(gt, prob) if len(np.unique(gt)) > 1 else float("nan")
        rows.append((p["ID"], len(seq), len(p["matched_core_regions"]), len(pred_runs), f1, auc))

        predictions.append({
            "ID": p["ID"],
            "length": len(seq),
            "true_regions": [{"start": r["start"], "end": r["end"]} for r in p["matched_core_regions"]],
            "predicted_regions": [
                {"start": s, "end": e, "mean_prob": round(float(prob[s:e].mean()), 4)}
                for s, e in pred_runs
            ],
        })

    # --- per-protein table ---
    print("\n=== Per-protein positional metrics ===")
    print(f"  {'ID':<12} {'len':>5} {'#true':>6} {'#pred':>6} {'resF1':>7} {'AUC':>7}")
    for pid, L, ntrue, npred, f1, auc in rows:
        auc_s = f"{auc:.3f}" if not np.isnan(auc) else "  n/a"
        print(f"  {pid:<12} {L:>5} {ntrue:>6} {npred:>6} {f1:>7.3f} {auc_s:>7}")

    # --- aggregate residue-level ---
    rt = np.concatenate(all_true)
    rp = np.concatenate(all_prob)
    rpred = (rp >= args.threshold).astype(np.int64)
    macro_auc = np.nanmean([r[5] for r in rows])
    print("\n=== Aggregate residue-level (pooled over all proteins) ===")
    print(f"  residues: {len(rt)} | core: {int(rt.sum())} | non-core: {int((rt == 0).sum())}")
    print(f"  precision {precision_score(rt, rpred, zero_division=0):.3f}")
    print(f"  recall    {recall_score(rt, rpred, zero_division=0):.3f}")
    print(f"  F1        {f1_score(rt, rpred, zero_division=0):.3f}")
    print(f"  ROC-AUC   {roc_auc_score(rt, rp):.3f}  (pooled) | {macro_auc:.3f} (macro per-protein)")
    print(f"  PR-AUC    {average_precision_score(rt, rp):.3f}")

    # --- region detection ---
    print("\n=== Region detection (localization recall) ===")
    rate = n_detected / n_regions if n_regions else float("nan")
    print(f"  detected {n_detected}/{n_regions} true core regions "
          f"(>= {args.overlap_frac:.0%} residue overlap) = {rate:.3f}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(predictions, fh, indent=2)
        print(f"\nwrote predicted core regions -> {args.out}")


if __name__ == "__main__":
    main()
