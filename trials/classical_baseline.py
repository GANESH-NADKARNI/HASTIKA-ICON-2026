#!/usr/bin/env python3
"""
Optional fast, CPU-only baseline: char+word TF-IDF -> Linear SVM.
Useful if you don't have a GPU, or want a quick sanity-check score before
spending time on the transformer fine-tune in train.py.

It won't beat a well-tuned transformer, but char n-grams are genuinely
decent on noisy, transliterated, code-mixed text since they're robust to
spelling variation (e.g. "yaake" / "yake" / "yaakhe").

Usage:
    python src/classical_baseline.py --task a \
        --train_csv data/binary_train.csv \
        --input data/binary_validation_inputs.csv \
        --out predictions/task_a_predictions_svm.csv

    python src/classical_baseline.py --task b \
        --train_csv data/multiclass_train.csv \
        --input data/multiclass_validation_inputs.csv \
        --out predictions/task_b_predictions_svm.csv
"""
import argparse
import csv

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import f1_score, classification_report
from sklearn.pipeline import FeatureUnion, Pipeline
from scipy.sparse import hstack

from utils import read_csv_dicts, find_id_key, find_comment_key, clean_text


def build_features():
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    return word_vec, char_vec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a", "b"], required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="predictions_svm.csv")
    ap.add_argument("--val_frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    label_col = "Label" if args.task == "a" else "Hate Category"

    rows = read_csv_dicts(args.train_csv)
    comment_key = find_comment_key(rows[0].keys())
    label_key = next(k for k in rows[0].keys() if k.strip().lower() == label_col.lower())

    texts = [clean_text(r[comment_key]) for r in rows]
    labels = [r[label_key].strip() for r in rows]

    X_train, X_val, y_train, y_val = train_test_split(
        texts, labels, test_size=args.val_frac, random_state=args.seed, stratify=labels
    )

    word_vec, char_vec = build_features()
    Xw_train = word_vec.fit_transform(X_train)
    Xc_train = char_vec.fit_transform(X_train)
    Xtr = hstack([Xw_train, Xc_train])

    clf = LinearSVC(class_weight="balanced", C=1.0, max_iter=5000)
    clf.fit(Xtr, y_train)

    Xw_val = word_vec.transform(X_val)
    Xc_val = char_vec.transform(X_val)
    Xval = hstack([Xw_val, Xc_val])
    val_preds = clf.predict(Xval)

    print("Internal-validation report:")
    print(classification_report(y_val, val_preds, digits=3))
    print("Macro-F1:", f1_score(y_val, val_preds, average="macro"))

    # Refit on ALL training data before predicting the real submission file
    Xw_all = word_vec.fit_transform(texts)
    Xc_all = char_vec.fit_transform(texts)
    Xall = hstack([Xw_all, Xc_all])
    clf_final = LinearSVC(class_weight="balanced", C=1.0, max_iter=5000)
    clf_final.fit(Xall, labels)

    pred_rows = read_csv_dicts(args.input)
    id_key = find_id_key(pred_rows[0].keys())
    pred_comment_key = find_comment_key(pred_rows[0].keys())
    pred_texts = [clean_text(r[pred_comment_key]) for r in pred_rows]
    pred_ids = [r[id_key] for r in pred_rows]

    Xw_pred = word_vec.transform(pred_texts)
    Xc_pred = char_vec.transform(pred_texts)
    Xpred = hstack([Xw_pred, Xc_pred])
    final_preds = clf_final.predict(Xpred)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, p in zip(pred_ids, final_preds):
            w.writerow([i, p])
    print(f"Wrote {args.out} with {len(final_preds)} predictions.")


if __name__ == "__main__":
    main()
