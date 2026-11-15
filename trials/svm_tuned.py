#!/usr/bin/env python3
"""
Improved classical baseline: adds hand-crafted features on top of the
existing word+char TF-IDF, and does a small hyperparameter search (SVM's
C, char n-gram range) -- all evaluated with proper k-fold cross-validation
(same StratifiedKFold pattern as train_kfold_ensemble.py) so the search
itself doesn't leak or overfit to a single split.

WHY THIS IS WORTH TRYING:
The plain SVM baseline has quietly out-performed every standalone
transformer in this project (0.80-0.81 on Task A, ~0.56-0.61 on Task B).
Two cheap levers haven't been tried yet:
  1. Hand-crafted features TF-IDF can't see directly: ALL-CAPS ratio,
     punctuation/repeated-punctuation intensity, and the ratio of Kannada
     script vs Latin script characters (code-mixing degree) -- all
     plausible hate-speech signals in a code-mixed comment.
  2. A proper (if small) hyperparameter search over C and the char n-gram
     range, instead of the single fixed config used everywhere so far.

Usage:
    python src/svm_tuned.py --task a \
        --train_csv data/binary_train.csv \
        --input data/binary_validation_inputs.csv \
        --out predictions/task_a_predictions_svm_tuned.csv

    python src/svm_tuned.py --task b \
        --train_csv data/multiclass_train.csv \
        --input data/multiclass_validation_inputs.csv \
        --out predictions/task_b_predictions_svm_tuned.csv

This runs entirely on CPU in well under a minute -- cheap to try both
tasks, and cheap to re-run if you want to widen the search grid.
"""
import argparse
import csv
import re

import numpy as np
from scipy.sparse import hstack, csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from utils import read_csv_dicts, find_id_key, find_comment_key, clean_text, TASK_A_LABELS, TASK_B_LABELS

KANNADA_RANGE = (0x0C80, 0x0CFF)


def handcrafted_features(texts):
    """A handful of hate-speech-plausible signals TF-IDF doesn't directly
    encode: shouting (caps), punctuation intensity, and code-mixing degree
    (Kannada-script vs Latin-script character ratio)."""
    feats = []
    for t in texts:
        n = max(len(t), 1)
        letters = [c for c in t if c.isalpha()]
        n_letters = max(len(letters), 1)
        caps_ratio = sum(1 for c in letters if c.isupper()) / n_letters
        excl_count = t.count("!")
        quest_count = t.count("?")
        repeated_punct = len(re.findall(r"[!?]{2,}", t))
        kannada_chars = sum(1 for c in t if KANNADA_RANGE[0] <= ord(c) <= KANNADA_RANGE[1])
        kannada_ratio = kannada_chars / n
        word_count = max(len(t.split()), 1)
        avg_word_len = sum(len(w) for w in t.split()) / word_count
        feats.append([
            caps_ratio, excl_count / n, quest_count / n, repeated_punct,
            kannada_ratio, word_count, avg_word_len,
        ])
    return np.array(feats, dtype=float)


def build_and_score(texts, y, labels_list, fold_indices, C, char_ngram_range, use_handcrafted):
    """Run one hyperparameter config through k-fold CV, return OOF macro-F1."""
    n = len(texts)
    oof_probs = np.zeros((n, len(labels_list)))

    for train_idx, val_idx in fold_indices:
        fit_texts = [texts[i] for i in train_idx]
        fit_y = [y[i] for i in train_idx]
        val_texts = [texts[i] for i in val_idx]

        word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngram_range, min_df=2, sublinear_tf=True)
        Xw = word_vec.fit_transform(fit_texts)
        Xc = char_vec.fit_transform(fit_texts)
        Xw_v = word_vec.transform(val_texts)
        Xc_v = char_vec.transform(val_texts)

        if use_handcrafted:
            hc_fit = handcrafted_features(fit_texts)
            hc_val = handcrafted_features(val_texts)
            scaler = StandardScaler()
            hc_fit = scaler.fit_transform(hc_fit)
            hc_val = scaler.transform(hc_val)
            X_fit = hstack([Xw, Xc, csr_matrix(hc_fit)])
            X_val = hstack([Xw_v, Xc_v, csr_matrix(hc_val)])
        else:
            X_fit = hstack([Xw, Xc])
            X_val = hstack([Xw_v, Xc_v])

        base = LinearSVC(class_weight="balanced", C=C, max_iter=5000)
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
        clf.fit(X_fit, fit_y)
        col_for = {c: i for i, c in enumerate(clf.classes_)}
        probs = clf.predict_proba(X_val)
        reordered = np.zeros((probs.shape[0], len(labels_list)))
        for j, lab in enumerate(labels_list):
            reordered[:, j] = probs[:, col_for[lab]]
        oof_probs[val_idx] = reordered

    y_arr = np.array([labels_list.index(l) for l in y])
    f1 = f1_score(y_arr, oof_probs.argmax(axis=1), average="macro")
    return f1, oof_probs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a", "b"], required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="predictions_svm_tuned.csv")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    labels_list = TASK_A_LABELS if args.task == "a" else TASK_B_LABELS
    label_col = "Label" if args.task == "a" else "Hate Category"

    rows = read_csv_dicts(args.train_csv)
    comment_key = find_comment_key(rows[0].keys())
    label_key = next(k for k in rows[0].keys() if k.strip().lower() == label_col.lower())
    texts = [clean_text(r[comment_key]) for r in rows]
    raw_labels = [r[label_key].strip() for r in rows]
    y_idx = np.array([labels_list.index(l) for l in raw_labels])

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_indices = list(skf.split(texts, y_idx))

    # small, cheap grid -- widen if you want to search more
    grid = []
    for C in [0.5, 1.0, 2.0]:
        for char_ngram in [(3, 5), (2, 5), (2, 6)]:
            for use_hc in [False, True]:
                grid.append((C, char_ngram, use_hc))

    print(f"Searching {len(grid)} configs with {args.n_folds}-fold CV...\n")
    best = None
    for C, char_ngram, use_hc in grid:
        f1, _ = build_and_score(texts, raw_labels, labels_list, fold_indices, C, char_ngram, use_hc)
        tag = f"C={C:<4} char_ngram={char_ngram} handcrafted={'yes' if use_hc else 'no '}"
        print(f"  {tag}  ->  macro-F1 = {f1:.4f}")
        if best is None or f1 > best[0]:
            best = (f1, C, char_ngram, use_hc)

    best_f1, best_C, best_char_ngram, best_use_hc = best
    print(f"\nBest config: C={best_C}, char_ngram={best_char_ngram}, handcrafted={best_use_hc} -> OOF macro-F1 = {best_f1:.4f}")

    # rerun the winning config once more to get its OOF probs for the classification report
    _, oof_probs = build_and_score(texts, raw_labels, labels_list, fold_indices, best_C, best_char_ngram, best_use_hc)
    print("\nPer-class OOF report for the winning config:")
    print(classification_report(y_idx, oof_probs.argmax(axis=1), target_names=labels_list, digits=3))

    # ---- fit the winning config on ALL data, predict the real submission file ----
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=best_char_ngram, min_df=2, sublinear_tf=True)
    Xw = word_vec.fit_transform(texts)
    Xc = char_vec.fit_transform(texts)

    if best_use_hc:
        hc = handcrafted_features(texts)
        scaler = StandardScaler()
        hc = scaler.fit_transform(hc)
        X_all = hstack([Xw, Xc, csr_matrix(hc)])
    else:
        X_all = hstack([Xw, Xc])

    base = LinearSVC(class_weight="balanced", C=best_C, max_iter=5000)
    final_clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    final_clf.fit(X_all, raw_labels)

    pred_rows = read_csv_dicts(args.input)
    id_key = find_id_key(pred_rows[0].keys())
    pred_comment_key = find_comment_key(pred_rows[0].keys())
    ids = [r[id_key] for r in pred_rows]
    sub_texts = [clean_text(r[pred_comment_key]) for r in pred_rows]

    Xw_s = word_vec.transform(sub_texts)
    Xc_s = char_vec.transform(sub_texts)
    if best_use_hc:
        hc_s = scaler.transform(handcrafted_features(sub_texts))
        X_sub = hstack([Xw_s, Xc_s, csr_matrix(hc_s)])
    else:
        X_sub = hstack([Xw_s, Xc_s])

    final_preds = final_clf.predict(X_sub)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, p in zip(ids, final_preds):
            w.writerow([i, p])
    print(f"\nWrote {args.out} with {len(final_preds)} predictions.")
    print(f"Expected macro-F1 for this file (OOF estimate): {best_f1:.4f}")


if __name__ == "__main__":
    main()