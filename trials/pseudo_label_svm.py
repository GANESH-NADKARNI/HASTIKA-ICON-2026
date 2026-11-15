#!/usr/bin/env python3
"""
Pseudo-labels binary_validation_inputs.csv using the current best blend
(bagged MuRIL + bagged XLM-R-large + char n-gram SVM), keeps only
high-confidence predictions, and tests -- honestly, leak-free -- whether
adding them as extra training data actually improves the SVM before you
commit to retraining any transformers on it.

Leak-free check: for each of the 5 original folds, the pseudo-labeled rows
(which come from validation_inputs.csv, disjoint from binary_train.csv) are
added ONLY to that fold's training split, never to its held-out evaluation
split. So the comparison below is a fair OOF macro-F1 on real labels,
not inflated by the model grading its own pseudo-labels.

Usage:
    python src/pseudo_label_svm.py --task a \
        --train_csv data/binary_train.csv \
        --input data/binary_validation_inputs.csv \
        --kfold_dirs outputs/task_a_kfold_muril_v2 outputs/task_a_kfold_xlmr_large_v3 \
        --weights 0.15 0.15 0.70 \
        --confidence 0.90
"""
import argparse
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from utils import read_csv_dicts, find_id_key, find_comment_key, clean_text, TASK_A_LABELS
from train_kfold_ensemble import svm_kfold_oof_and_final
from multi_ensemble import bagged_kfold_probs


def train_svm_char(fit_texts, fit_labels):
    """Char n-gram (2,5) SVM, calibrated -- the config that's consistently
    won out over word+char and handcrafted-feature variants on this data."""
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, sublinear_tf=True)
    X = vec.fit_transform(fit_texts)
    base = LinearSVC(class_weight="balanced", C=1.0, max_iter=10000)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(X, fit_labels)
    col_for = {c: i for i, c in enumerate(clf.classes_)}

    def predict_proba_for(texts):
        Xt = vec.transform(texts)
        probs = clf.predict_proba(Xt)
        reordered = np.zeros((probs.shape[0], 2))
        for j, lab in enumerate(TASK_A_LABELS):
            reordered[:, j] = probs[:, col_for[lab]]
        return reordered

    return predict_proba_for


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a"], required=True, help="pseudo-labeling script is written for the binary task only")
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--input", required=True, help="binary_validation_inputs.csv (unlabeled)")
    ap.add_argument("--kfold_dirs", nargs=2, required=True, metavar=("MURIL_DIR", "XLMR_DIR"))
    ap.add_argument("--weights", nargs=3, type=float, required=True, metavar=("W_MURIL", "W_XLMR", "W_SVM"),
                     help="blend weights in the SAME order as kfold_dirs, then SVM last -- copy from your best kfold_multi_ensemble.py run")
    ap.add_argument("--confidence", type=float, default=0.90,
                     help="keep pseudo-labels with P(Hate) > this OR P(Hate) < (1 - this)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--n_folds", type=int, default=5)
    args = ap.parse_args()

    label2id = {l: i for i, l in enumerate(TASK_A_LABELS)}

    # ---- load labeled training data ----
    rows = read_csv_dicts(args.train_csv)
    comment_key = find_comment_key(rows[0].keys())
    label_key = next(k for k in rows[0].keys() if k.strip().lower() == "label")
    texts = [clean_text(r[comment_key]) for r in rows]
    raw_labels = [r[label_key].strip() for r in rows]
    y = np.array([label2id[l] for l in raw_labels])

    # ---- load unlabeled validation inputs ----
    pred_rows = read_csv_dicts(args.input)
    id_key = find_id_key(pred_rows[0].keys())
    pred_comment_key = find_comment_key(pred_rows[0].keys())
    unlabeled_ids = [r[id_key] for r in pred_rows]
    unlabeled_texts = [clean_text(r[pred_comment_key]) for r in pred_rows]

    print(f"Loaded {len(texts)} labeled rows and {len(unlabeled_texts)} unlabeled rows.\n")

    # ---- score unlabeled rows with the current best blend ----
    print("Bagging transformer folds on unlabeled inputs...")
    muril_probs = bagged_kfold_probs(args.kfold_dirs[0], unlabeled_texts)
    xlmr_probs = bagged_kfold_probs(args.kfold_dirs[1], unlabeled_texts)

    print("Fitting SVM on all labeled data for pseudo-labeling...")
    svm_predict_proba_full = train_svm_char(texts, raw_labels)
    svm_probs = svm_predict_proba_full(unlabeled_texts)

    w_muril, w_xlmr, w_svm = args.weights
    blended = w_muril * muril_probs + w_xlmr * xlmr_probs + w_svm * svm_probs
    hate_probs = blended[:, label2id["Hate"]]

    # ---- keep only high-confidence predictions ----
    hi_thresh = args.confidence
    lo_thresh = 1.0 - args.confidence
    confident_mask = (hate_probs > hi_thresh) | (hate_probs < lo_thresh)
    n_confident = confident_mask.sum()
    print(f"\n{n_confident}/{len(unlabeled_texts)} unlabeled rows are confident "
          f"(P(Hate) > {hi_thresh:.2f} or < {lo_thresh:.2f})")

    pseudo_texts = [t for t, m in zip(unlabeled_texts, confident_mask) if m]
    pseudo_labels_str = ["Hate" if p > 0.5 else "Non-Hate" for p, m in zip(hate_probs, confident_mask) if m]
    pseudo_hate_count = sum(1 for l in pseudo_labels_str if l == "Hate")
    print(f"  -> {pseudo_hate_count} pseudo-Hate, {n_confident - pseudo_hate_count} pseudo-Non-Hate")

    if n_confident == 0:
        print("\nNo confident pseudo-labels at this threshold -- try a lower --confidence value.")
        return

    # ---- leak-free comparison: does adding these actually help? ----
    print("\n=== Leak-free 5-fold comparison: baseline SVM vs. SVM + pseudo-labels ===")
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_indices = list(skf.split(texts, y))

    baseline_oof, _ = svm_kfold_oof_and_final(texts, raw_labels, TASK_A_LABELS, fold_indices)
    baseline_f1 = f1_score(y, baseline_oof.argmax(axis=1), average="macro")
    print(f"Baseline SVM OOF macro-F1 (no pseudo-labels): {baseline_f1:.4f}")

    aug_oof = np.zeros((len(texts), 2))
    for fold_i, (train_idx, val_idx) in enumerate(fold_indices):
        fit_texts = [texts[i] for i in train_idx] + pseudo_texts
        fit_labels = [raw_labels[i] for i in train_idx] + pseudo_labels_str
        val_texts_fold = [texts[i] for i in val_idx]

        predict_proba = train_svm_char(fit_texts, fit_labels)
        aug_oof[val_idx] = predict_proba(val_texts_fold)
        print(f"  fold {fold_i}: {len(train_idx)} real + {len(pseudo_texts)} pseudo -> "
              f"scored {len(val_idx)} real held-out rows")

    aug_f1 = f1_score(y, aug_oof.argmax(axis=1), average="macro")
    print(f"\nSVM + pseudo-labels OOF macro-F1: {aug_f1:.4f}")
    print(f"Delta: {aug_f1 - baseline_f1:+.4f}")

    if aug_f1 > baseline_f1:
        print("\n=> Pseudo-labeling helped. Worth feeding these same pseudo-labeled rows "
              "into a transformer retrain (train_kfold_ensemble.py) next.")
    else:
        print("\n=> Pseudo-labeling did NOT help at this confidence threshold. "
              "Try raising --confidence (e.g. 0.95) for cleaner pseudo-labels, "
              "or it may just not be worth pursuing further for this model.")


if __name__ == "__main__":
    main()