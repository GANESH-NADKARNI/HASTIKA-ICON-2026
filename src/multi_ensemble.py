#!/usr/bin/env python3
"""
Combine MULTIPLE trained transformer checkpoints (e.g. outputs/task_b and
outputs/task_b_v3 -- different hyperparameters, so they make different
mistakes) with the calibrated SVM, and grid-search the blend weights that
maximize macro-F1 on the internal validation split.

Requires all --model_dirs to have been trained with the SAME --train_csv,
--seed and --val_frac as each other (so they share the identical internal
validation split, and there's no leakage when we compare/blend them).

Usage:
    python src/multi_ensemble.py --task b \
        --train_csv data/multiclass_train.csv \
        --model_dirs outputs/task_b outputs/task_b_v3 \
        --input data/multiclass_validation_inputs.csv \
        --out predictions/task_b_predictions_multiensemble.csv
"""
import argparse
import csv
import json
import os
from itertools import product

import numpy as np
import torch
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import read_csv_dicts, find_id_key, find_comment_key, clean_text, TASK_A_LABELS, TASK_B_LABELS


def build_svm_probs(train_csv, label_col, holdout_texts, labels_list):
    rows = read_csv_dicts(train_csv)
    comment_key = find_comment_key(rows[0].keys())
    label_key = next(k for k in rows[0].keys() if k.strip().lower() == label_col.lower())
    all_texts = [clean_text(r[comment_key]) for r in rows]
    all_labels = [r[label_key].strip() for r in rows]

    holdout_set = set(holdout_texts)
    fit_texts = [t for t in all_texts if t not in holdout_set]
    fit_labels = [l for t, l in zip(all_texts, all_labels) if t not in holdout_set]

    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    Xw = word_vec.fit_transform(fit_texts)
    Xc = char_vec.fit_transform(fit_texts)
    X = hstack([Xw, Xc])

    base = LinearSVC(class_weight="balanced", C=1.0, max_iter=5000)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(X, fit_labels)
    col_for = {c: i for i, c in enumerate(clf.classes_)}

    def predict_proba_for(texts):
        Xw_ = word_vec.transform(texts)
        Xc_ = char_vec.transform(texts)
        X_ = hstack([Xw_, Xc_])
        probs = clf.predict_proba(X_)
        reordered = np.zeros((probs.shape[0], len(labels_list)))
        for j, lab in enumerate(labels_list):
            reordered[:, j] = probs[:, col_for[lab]]
        return reordered

    return predict_proba_for


def transformer_probs(model_dir, texts, batch_size=32, max_len=160):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(device)
    model.eval()
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            enc = tokenizer(batch, truncation=True, max_length=max_len, padding=True, return_tensors="pt").to(device)
            probs = torch.softmax(model(**enc).logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
    return np.concatenate(all_probs, axis=0)


def grid_search_weights(prob_matrices, y_val, step=0.1, n_random_restarts=60, seed=42):
    """
    Find blend weights (summing to 1, each >= 0) that maximize macro-F1.
    - For 2-3 matrices: exhaustive grid search (fast, exact enough at step=0.1).
    - For 4+ matrices: exhaustive grid becomes huge, so use random Dirichlet
      sampling + local coordinate-descent refinement around the best sample.
    """
    n = len(prob_matrices)
    rng = np.random.default_rng(seed)

    def score(w):
        blended = sum(wi * pm for wi, pm in zip(w, prob_matrices))
        return f1_score(y_val, blended.argmax(axis=1), average="macro")

    if n <= 3:
        steps = np.arange(0, 1.0 + 1e-9, step)
        if n == 1:
            candidates = [(1.0,)]
        elif n == 2:
            candidates = [(w, round(1 - w, 10)) for w in steps]
        else:  # n == 3
            candidates = []
            for w1 in steps:
                for w2 in steps:
                    w3 = round(1 - w1 - w2, 10)
                    if -1e-9 <= w3 <= 1 + 1e-9:
                        candidates.append((round(w1, 10), round(w2, 10), max(0.0, w3)))
        best_w, best_f1 = None, -1
        for w in candidates:
            f1 = score(w)
            if f1 > best_f1:
                best_f1, best_w = f1, w
        return best_w, best_f1

    # n >= 4: random Dirichlet search (favors sparser mixes with alpha<1,
    # since a few strong models usually beat averaging everything equally)
    # then greedy coordinate-descent polish.
    best_w, best_f1 = None, -1
    for _ in range(n_random_restarts):
        w = rng.dirichlet(alpha=np.full(n, 0.7))
        f1 = score(w)
        if f1 > best_f1:
            best_f1, best_w = f1, w

    best_w = np.array(best_w)
    for _ in range(3):  # a few coordinate-descent sweeps
        improved = False
        for i in range(n):
            for delta in np.linspace(-0.3, 0.3, 13):
                cand = best_w.copy()
                cand[i] = max(0.0, cand[i] + delta)
                if cand.sum() == 0:
                    continue
                cand = cand / cand.sum()
                f1 = score(cand)
                if f1 > best_f1:
                    best_f1, best_w, improved = f1, cand, True
        if not improved:
            break
    return tuple(round(float(x), 4) for x in best_w), best_f1


def bagged_kfold_probs(kfold_dir, texts, max_len=160):
    """Average predictions across every fold model in kfold_dir/fold* on
    the given texts -- this is the same 'bagging' train_kfold_ensemble.py
    does at submission time, but callable on ANY text list (e.g. the
    shared 380-row internal-val split used by the single-split models),
    so a k-fold group can be compared/blended apples-to-apples with them."""
    import glob
    fold_dirs = sorted(glob.glob(os.path.join(kfold_dir, "fold*")))
    if not fold_dirs:
        raise SystemExit(f"No fold* subdirectories found under {kfold_dir}")
    all_probs = [transformer_probs(fd, texts, max_len=max_len) for fd in fold_dirs]
    return np.mean(all_probs, axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a", "b"], required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--model_dirs", nargs="+", default=[], help="single-split trained transformer output_dirs to combine")
    ap.add_argument("--kfold_dirs", nargs="+", default=[], help="k-fold base dirs (each containing fold0, fold1, ... subdirs) -- bagged and blended alongside --model_dirs")
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="predictions_multiensemble.csv")
    args = ap.parse_args()

    if not args.model_dirs and not args.kfold_dirs:
        raise SystemExit("Pass at least one of --model_dirs or --kfold_dirs")

    labels_list = TASK_A_LABELS if args.task == "a" else TASK_B_LABELS
    label_col = "Label" if args.task == "a" else "Hate Category"

    # Load cached internal-val probs for every single-split transformer --
    # all must share the same split (same train_csv/seed/val_frac).
    tf_val_probs_list = []
    y_val_ref = None
    val_texts_ref = None
    for md in args.model_dirs:
        cache = np.load(os.path.join(md, "internal_val_cache.npz"))
        with open(os.path.join(md, "internal_val_texts.json"), encoding="utf-8") as f:
            texts = json.load(f)
        if y_val_ref is None:
            y_val_ref = cache["y_val"]
            val_texts_ref = texts
        else:
            assert texts == val_texts_ref, (
                f"{md} has a different internal-val split than {args.model_dirs[0]} -- "
                "retrain with the same --train_csv/--seed/--val_frac so splits match."
            )
        tf_val_probs_list.append(cache["val_probs"])

    if val_texts_ref is None:
        raise SystemExit(
            "--kfold_dirs alone can't be scored on a comparable split -- pass at least "
            "one --model_dirs entry too, so there's a shared 380-row internal-val slice "
            "to re-score the k-fold group's bagged predictions on."
        )

    # Re-score each k-fold group's BAGGED predictions on that same shared
    # split, so it's directly comparable to the single-split models above
    # (its own internal OOF metric covers different/more rows and can't be
    # mixed in directly).
    kfold_val_probs_list = []
    for kd in args.kfold_dirs:
        print(f"Bagging {kd} folds on the shared internal-val split...")
        kfold_val_probs_list.append(bagged_kfold_probs(kd, val_texts_ref))

    print("Fitting calibrated SVM on training rows outside the internal-val split...")
    svm_predict_proba = build_svm_probs(args.train_csv, label_col, val_texts_ref, labels_list)
    svm_val_probs = svm_predict_proba(val_texts_ref)

    prob_matrices = tf_val_probs_list + kfold_val_probs_list + [svm_val_probs]
    names = (
        [os.path.basename(md.rstrip("/")) for md in args.model_dirs]
        + [os.path.basename(kd.rstrip("/")) + "(bagged)" for kd in args.kfold_dirs]
        + ["SVM"]
    )

    for name, pm in zip(names, prob_matrices):
        f1 = f1_score(y_val_ref, pm.argmax(axis=1), average="macro")
        print(f"  {name} alone: {f1:.4f}")

    weights, best_f1 = grid_search_weights(prob_matrices, y_val_ref, step=0.1)
    print(f"Best blend weights {dict(zip(names, weights))} -> internal macro-F1 = {best_f1:.4f}")

    # apply to the real submission file
    rows = read_csv_dicts(args.input)
    id_key = find_id_key(rows[0].keys())
    comment_key = find_comment_key(rows[0].keys())
    ids = [r[id_key] for r in rows]
    texts = [clean_text(r[comment_key]) for r in rows]

    sub_matrices = []
    for md in args.model_dirs:
        print(f"Scoring {md} on submission inputs...")
        sub_matrices.append(transformer_probs(md, texts))
    for kd in args.kfold_dirs:
        print(f"Scoring {kd} (bagged folds) on submission inputs...")
        sub_matrices.append(bagged_kfold_probs(kd, texts))
    print("Scoring SVM on submission inputs...")
    sub_matrices.append(svm_predict_proba(texts))

    blended_sub = sum(w * pm for w, pm in zip(weights, sub_matrices))
    final_preds = [labels_list[i] for i in blended_sub.argmax(axis=1)]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, p in zip(ids, final_preds):
            w.writerow([i, p])
    print(f"Wrote {args.out} with {len(final_preds)} predictions.")


if __name__ == "__main__":
    main()