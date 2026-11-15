#!/usr/bin/env python3
"""
For Task A ONLY (binary): every script so far picks whichever class has
the higher raw probability -- an implicit 0.5 threshold. But macro-F1
doesn't require that threshold to be optimal, and Task A's classes aren't
perfectly balanced (3,950 Hate vs 4,108 Non-Hate in the full corpus).
Shifting the decision threshold is a free, well-established lever that
none of your blends have used yet.

This loads your existing k-fold OOF caches (no retraining), rebuilds the
same blend, then searches thresholds on the Hate-class probability that
maximize OOF macro-F1 -- reporting whether it beats the flat 0.5 default.

Usage:
    python src/threshold_tune.py --task a \
        --train_csv data/binary_train.csv \
        --kfold_dirs outputs/task_a_kfold_muril outputs/task_a_kfold_xlmr_large_v2 \
        --input data/binary_validation_inputs.csv \
        --out predictions/task_a_predictions_threshold_tuned.csv
"""
import argparse
import csv
import json
import os

import numpy as np
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from utils import read_csv_dicts, find_id_key, find_comment_key, clean_text, TASK_A_LABELS
from train_kfold_ensemble import svm_kfold_oof_and_final
from multi_ensemble import bagged_kfold_probs, grid_search_weights


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a"], default="a", help="threshold tuning only makes sense for Task A's binary setup")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--kfold_dirs", nargs="+", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="predictions_threshold_tuned.csv")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    labels_list = TASK_A_LABELS  # ['Non-Hate', 'Hate']
    label2id = {l: i for i, l in enumerate(labels_list)}
    hate_idx = label2id["Hate"]

    rows = read_csv_dicts(args.train_csv)
    label_key = next(k for k in rows[0].keys() if k.strip().lower() == "label")
    raw_labels = [r[label_key].strip() for r in rows]
    y = np.array([label2id[l] for l in raw_labels])

    # Use the FIRST kfold_dir's cached texts as canonical, and only require
    # every other kfold_dir to match THAT (not today's utils.py). This is
    # the correct check: what matters is that the k-fold groups you're
    # combining are mutually row-aligned with each other and with y, not
    # that they match whatever clean_text() happens to produce today --
    # text-cleaning changes (e.g. a mojibake fix) don't reorder or drop
    # rows, so old k-fold groups trained under an earlier utils.py are
    # still perfectly valid to combine with each other.
    with open(os.path.join(args.kfold_dirs[0], "internal_val_texts.json"), encoding="utf-8") as f:
        texts = json.load(f)
    assert len(texts) == len(y), (
        f"{args.kfold_dirs[0]}'s cached texts ({len(texts)} rows) don't match "
        f"{args.train_csv}'s row count ({len(y)}) -- was this trained on a different file?"
    )
    for kd in args.kfold_dirs[1:]:
        with open(os.path.join(kd, "internal_val_texts.json"), encoding="utf-8") as f:
            other_texts = json.load(f)
        assert other_texts == texts, (
            f"{kd} was trained with different text-cleaning than {args.kfold_dirs[0]} -- "
            "these two k-fold groups aren't safe to combine with each other."
        )

    # The SVM is always recomputed fresh anyway, so it's fine (and better)
    # for it to use today's clean_text() -- it doesn't need to match the
    # older cached transformer text.
    svm_texts = [clean_text(r[find_comment_key(rows[0].keys())]) for r in rows]

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_indices = list(skf.split(texts, y))

    print("Loading cached OOF predictions...")
    tf_oof_list = [np.load(os.path.join(kd, "internal_val_cache.npz"))["val_probs"] for kd in args.kfold_dirs]

    print("Computing cross-validated SVM...")
    svm_oof, svm_predict_proba_final = svm_kfold_oof_and_final(svm_texts, raw_labels, labels_list, fold_indices)

    prob_matrices = tf_oof_list + [svm_oof]
    weights, blend_f1_at_argmax = grid_search_weights(prob_matrices, y, step=0.05)
    blended_oof = sum(w * pm for w, pm in zip(weights, prob_matrices))
    print(f"Blend weights: {weights} -> macro-F1 at default argmax (0.5 threshold): {blend_f1_at_argmax:.4f}")

    # search decision threshold on the Hate-class probability
    hate_probs = blended_oof[:, hate_idx]
    best_t, best_f1 = 0.5, blend_f1_at_argmax
    for t in np.arange(0.30, 0.71, 0.01):
        preds = (hate_probs >= t).astype(int)
        f1 = f1_score(y, preds, average="macro")
        if f1 > best_f1:
            best_f1, best_t = f1, t

    print(f"\nBest threshold: {best_t:.2f} -> OOF macro-F1 = {best_f1:.4f}")
    print(f"(vs default 0.5 threshold: {blend_f1_at_argmax:.4f}, gain = {best_f1 - blend_f1_at_argmax:+.4f})")

    # apply to submission file
    pred_rows = read_csv_dicts(args.input)
    id_key = find_id_key(pred_rows[0].keys())
    pred_comment_key = find_comment_key(pred_rows[0].keys())
    ids = [r[id_key] for r in pred_rows]
    sub_texts = [clean_text(r[pred_comment_key]) for r in pred_rows]

    sub_matrices = [bagged_kfold_probs(kd, sub_texts) for kd in args.kfold_dirs]
    sub_matrices.append(svm_predict_proba_final(sub_texts))
    blended_sub = sum(w * pm for w, pm in zip(weights, sub_matrices))
    hate_probs_sub = blended_sub[:, hate_idx]
    final_preds = [labels_list[hate_idx] if p >= best_t else labels_list[1 - hate_idx] for p in hate_probs_sub]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, p in zip(ids, final_preds):
            w.writerow([i, p])
    print(f"\nWrote {args.out} with {len(final_preds)} predictions.")
    print(f"Expected internal macro-F1 for this file: {best_f1:.4f}")


if __name__ == "__main__":
    main()