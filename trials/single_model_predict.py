#!/usr/bin/env python3
"""
The simplest possible submission: bag one k-fold model group's 5 folds
together (no SVM, no second architecture, no blend-weight search at all).

This exists to directly test the hypothesis that a simpler, less-tuned
pipeline might generalize better than an intricately blended one -- every
blend-weight grid search you've run is itself fit to the same small
internal validation set it's evaluated on, which risks quietly overfitting
to that specific 774-806 row slice rather than the true test distribution.

Usage:
    python src/single_model_predict.py \
        --kfold_dir outputs/task_a_kfold_xlmr_large_v2 \
        --input data/binary_validation_inputs.csv \
        --out predictions/task_a_predictions_xlmr_large_v2_ALONE.csv
"""
import argparse
import csv

from utils import read_csv_dicts, find_id_key, find_comment_key, clean_text, TASK_A_LABELS, TASK_B_LABELS
from multi_ensemble import bagged_kfold_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a", "b"], default="a")
    ap.add_argument("--kfold_dir", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="predictions_single_model.csv")
    args = ap.parse_args()

    labels_list = TASK_A_LABELS if args.task == "a" else TASK_B_LABELS

    rows = read_csv_dicts(args.input)
    id_key = find_id_key(rows[0].keys())
    comment_key = find_comment_key(rows[0].keys())
    ids = [r[id_key] for r in rows]
    texts = [clean_text(r[comment_key]) for r in rows]

    print(f"Bagging {args.kfold_dir}'s 5 folds (no SVM, no blending)...")
    probs = bagged_kfold_probs(args.kfold_dir, texts)
    preds = [labels_list[i] for i in probs.argmax(axis=1)]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, p in zip(ids, preds):
            w.writerow([i, p])
    print(f"Wrote {args.out} with {len(preds)} predictions.")


if __name__ == "__main__":
    main()