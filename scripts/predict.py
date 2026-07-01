"""Benchmark prediction -> CSV reports (classification + positional).

Runs the trained CNN over one or more benchmark JSON files with the live-ESM2 ``2k+1``
sliding window and writes, per benchmark, three CSV files into ``--out-dir``:

  <stem>_residue_scores.csv      per-residue score for every residue of every protein
  <stem>_protein_predictions.csv per-protein amyloid/non-amyloid decision
  <stem>_metrics.csv             all aggregate metrics (residue / protein / region)

It computes whatever applies to each file:
* residue-level metrics + per-residue scores  -> always (positional task)
* protein-level classification metrics         -> only when the file has >=2 labels
* region-detection (localization) metrics      -> over proteins with core regions

Run from the project root:
    ./.venv/bin/python scripts/predict.py --checkpoint models/best.pt --k 7
    ./.venv/bin/python scripts/predict.py --benchmark New_dataset/Benchmark/new_representative_positive_benchmark_KA.json
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# Make the project's `src` package importable when run as a loose script.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src.predict import core_mask, load_model, longest_positive_run, score_protein  # noqa: E402
from src.position_benchmark import extract_runs  # noqa: E402

CLASSIFICATION_BENCHMARK = "New_dataset/Benchmark/new_representative_classification_benchmark_KA.json"
POSITIVE_BENCHMARK = "New_dataset/Benchmark/new_representative_positive_benchmark_KA.json"


def evaluate_benchmark(path, model, max_len, device, args, out_dir):
    stem = os.path.splitext(os.path.basename(path))[0]
    with open(path) as fh:
        proteins = json.load(fh)

    residue_rows = []          # per-residue scores
    protein_rows = []          # per-protein classification
    all_true, all_prob = [], []
    prot_true, prot_pred = [], []
    n_regions = n_detected = 0

    for p in proteins:
        seq = p["Sequence"]
        prob = score_protein(seq, model, args.k, max_len, device)
        gt = core_mask(p)
        pred_mask = (prob >= args.threshold).astype(np.int64)
        label = p.get("LABEL", "")
        true_class = 1 if label == "AMYLOID" else 0

        all_true.append(gt)
        all_prob.append(prob)

        # Per-residue scores.
        for i, aa in enumerate(seq):
            residue_rows.append(
                {
                    "protein_id": p.get("ID", ""),
                    "protein_label": label,
                    "position": i,
                    "residue": aa,
                    "prob_core": round(float(prob[i]), 6),
                    "pred_core": int(pred_mask[i]),
                    "true_core": int(gt[i]),
                }
            )

        # Protein-level classification: AMYLOID if a run >= min_core_len clears threshold.
        run = longest_positive_run(pred_mask)
        pred_class = 1 if run >= args.min_core_len else 0
        prot_true.append(true_class)
        prot_pred.append(pred_class)
        protein_rows.append(
            {
                "protein_id": p.get("ID", ""),
                "true_label": label,
                "true_class": true_class,
                "max_prob": round(float(prob.max()), 6),
                "longest_run": int(run),
                "n_pred_regions": len(extract_runs(pred_mask, args.min_core_len)),
                "n_true_regions": len(p.get("matched_core_regions", [])),
                "pred_class": pred_class,
                "pred_label": "AMYLOID" if pred_class else "NONAMYLOID",
                "correct": int(pred_class == true_class),
            }
        )

        # Region detection (localization recall).
        for r in p.get("matched_core_regions", []):
            n_regions += 1
            seg = pred_mask[r["start"] : r["end"]]
            if seg.size and seg.mean() >= args.overlap_frac:
                n_detected += 1

    # ---- write per-residue scores ----
    res_path = os.path.join(out_dir, f"{stem}_residue_scores.csv")
    _write_csv(res_path, residue_rows,
               ["protein_id", "protein_label", "position", "residue",
                "prob_core", "pred_core", "true_core"])

    # ---- write per-protein predictions ----
    prot_path = os.path.join(out_dir, f"{stem}_protein_predictions.csv")
    _write_csv(prot_path, protein_rows,
               ["protein_id", "true_label", "true_class", "max_prob", "longest_run",
                "n_pred_regions", "n_true_regions", "pred_class", "pred_label", "correct"])

    # ---- compute metrics ----
    rt = np.concatenate(all_true)
    rp = np.concatenate(all_prob)
    rpred = (rp >= args.threshold).astype(np.int64)
    metrics = []

    def add(scope, name, value):
        metrics.append({"scope": scope, "metric": name, "value": value})

    add("residue", "n_residues", int(len(rt)))
    add("residue", "n_core", int(rt.sum()))
    add("residue", "n_noncore", int((rt == 0).sum()))
    add("residue", "accuracy", round(accuracy_score(rt, rpred), 6))
    add("residue", "precision", round(precision_score(rt, rpred, zero_division=0), 6))
    add("residue", "recall", round(recall_score(rt, rpred, zero_division=0), 6))
    add("residue", "f1", round(f1_score(rt, rpred, zero_division=0), 6))
    if len(np.unique(rt)) > 1:
        add("residue", "roc_auc", round(roc_auc_score(rt, rp), 6))
        add("residue", "pr_auc", round(average_precision_score(rt, rp), 6))

    pt = np.array(prot_true)
    pp = np.array(prot_pred)
    if len(np.unique(pt)) > 1:
        tn, fp, fn, tp = confusion_matrix(pt, pp, labels=[0, 1]).ravel()
        add("protein", "n_proteins", int(len(pt)))
        add("protein", "accuracy", round(accuracy_score(pt, pp), 6))
        add("protein", "precision", round(precision_score(pt, pp, zero_division=0), 6))
        add("protein", "recall", round(recall_score(pt, pp, zero_division=0), 6))
        add("protein", "f1", round(f1_score(pt, pp, zero_division=0), 6))
        add("protein", "tn", int(tn))
        add("protein", "fp", int(fp))
        add("protein", "fn", int(fn))
        add("protein", "tp", int(tp))
    else:
        add("protein", "n_proteins", int(len(pt)))
        add("protein", "note", "single-class set; classification metrics skipped")

    if n_regions:
        add("region", "n_true_regions", int(n_regions))
        add("region", "n_detected", int(n_detected))
        add("region", "detection_rate", round(n_detected / n_regions, 6))

    metrics_path = os.path.join(out_dir, f"{stem}_metrics.csv")
    _write_csv(metrics_path, metrics, ["scope", "metric", "value"])

    print(f"\n[{stem}] {len(proteins)} proteins, {len(rt)} residues")
    for m in metrics:
        print(f"    {m['scope']:<8} {m['metric']:<14} {m['value']}")
    print(f"  wrote: {res_path}")
    print(f"         {prot_path}")
    print(f"         {metrics_path}")


def _write_csv(path, rows, fieldnames):
    with open(path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark prediction -> CSV (classification + positional)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--checkpoint", default="models/best.pt")
    parser.add_argument("--benchmark", nargs="+",
                        default=[CLASSIFICATION_BENCHMARK, POSITIVE_BENCHMARK],
                        help="one or more benchmark JSON files")
    parser.add_argument("--k", type=int, default=7, help="window half-width (window = 2k+1)")
    parser.add_argument("--threshold", type=float, default=0.5, help="per-residue core prob threshold")
    parser.add_argument("--min-core-len", type=int, default=5,
                        help="min consecutive positives => predicted core / AMYLOID call")
    parser.add_argument("--overlap-frac", type=float, default=0.5,
                        help="residue-overlap fraction for region detection")
    parser.add_argument("--out-dir", default="results", help="directory for CSV outputs")
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, max_len = load_model(args.checkpoint, device)
    print(f"loaded {args.checkpoint} (max_len={max_len}, k={args.k}) on {device}; ESM2 computed live")

    for path in args.benchmark:
        evaluate_benchmark(path, model, max_len, device, args, args.out_dir)


if __name__ == "__main__":
    main()
