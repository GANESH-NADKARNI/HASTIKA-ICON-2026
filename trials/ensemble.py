#!/usr/bin/env python3
"""
Blend the fine-tuned transformer with a probability-calibrated TF-IDF+SVM,
since the SVM baseline scored *better* than the raw transformer on Task B
(0.615 vs 0.45 macro-F1) -- it's clearly stronger on the small classes
(Geo-political, Violence). Averaging the two should beat either alone.

Requires that train.py has already been run for this task (so
outputs/task_x/internal_val_cache.npz and internal_val_texts.json exist) --
that's the exact held-out split we use to pick the blend weight and check
it actually helps before trusting it on the real submission.

Usage:
    python src/ensemble.py --task a \
        --train_csv data/binary_train.csv \
        --model_dir outputs/task_a \
        --input data/binary_validation_inputs.csv \
        --out predictions/task_a_predictions_ensemble.csv

    python src/ensemble.py --task b \
        --train_csv data/multiclass_train.csv \
        --model_dir outputs/task_b \
        --input data/multiclass_validation_inputs.csv \
        --out predictions/task_b_predictions_ensemble.csv
"""
import argparse
import csv
import json
import os

import numpy as np
import torch
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score
from sklearn.svm import LinearSVC
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import read_csv_dicts, find_id_key, find_comment_key, clean_text, TASK_A_LABELS, TASK_B_LABELS


def build_svm_probs(train_csv, label_col, val_texts_for_holdout, labels_list):
    """Train the SVM on ALL rows of train_csv EXCEPT the ones in
    val_texts_for_holdout (so it's not evaluated on data it saw), with
    probability calibration so we can average it with the transformer."""
    rows = read_csv_dicts(train_csv)
    comment_key = find_comment_key(rows[0].keys())
    label_key = next(k for k in rows[0].keys() if k.strip().lower() == label_col.lower())

    all_texts = [clean_text(r[comment_key]) for r in rows]
    all_labels = [r[label_key].strip() for r in rows]

    holdout_set = set(val_texts_for_holdout)
    fit_texts, fit_labels = [], []
    for t, l in zip(all_texts, all_labels):
        if t not in holdout_set:
            fit_texts.append(t)
            fit_labels.append(l)

    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    Xw = word_vec.fit_transform(fit_texts)
    Xc = char_vec.fit_transform(fit_texts)
    X = hstack([Xw, Xc])

    base = LinearSVC(class_weight="balanced", C=1.0, max_iter=5000)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(X, fit_labels)

    def predict_proba_for(texts):
        Xw_ = word_vec.transform(texts)
        Xc_ = char_vec.transform(texts)
        X_ = hstack([Xw_, Xc_])
        probs = clf.predict_proba(X_)
        # reorder columns to match labels_list order (CalibratedClassifierCV uses clf.classes_ order)
        col_for = {c: i for i, c in enumerate(clf.classes_)}
        reordered = np.zeros((probs.shape[0], len(labels_list)))
        for j, lab in enumerate(labels_list):
            reordered[:, j] = probs[:, col_for[lab]]
        return reordered

    return predict_proba_for, word_vec, char_vec


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a", "b"], required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--model_dir", required=True, help="output_dir used when training the transformer")
    ap.add_argument("--input", required=True, help="the *_validation_inputs.csv to predict on")
    ap.add_argument("--out", default="predictions_ensemble.csv")
    ap.add_argument("--weight", type=float, default=None,
                     help="transformer weight in [0,1]; SVM gets 1-weight. If omitted, auto-picks the best weight on the internal val split.")
    args = ap.parse_args()

    labels_list = TASK_A_LABELS if args.task == "a" else TASK_B_LABELS
    label_col = "Label" if args.task == "a" else "Hate Category"

    cache_path = os.path.join(args.model_dir, "internal_val_cache.npz")
    texts_path = os.path.join(args.model_dir, "internal_val_texts.json")
    if not (os.path.exists(cache_path) and os.path.exists(texts_path)):
        raise SystemExit(
            f"Missing {cache_path} / {texts_path}. Re-run the current src/train.py for this task first "
            "(the v2 version saves these automatically)."
        )

    cache = np.load(cache_path)
    tf_val_probs = cache["val_probs"]
    y_val = cache["y_val"]
    with open(texts_path, encoding="utf-8") as f:
        val_texts = json.load(f)

    print("Fitting calibrated SVM on training rows outside the internal-val split...")
    svm_predict_proba, _, _ = build_svm_probs(args.train_csv, label_col, val_texts, labels_list)
    svm_val_probs = svm_predict_proba(val_texts)

    # pick the blend weight that maximizes macro-F1 on the internal val split
    if args.weight is None:
        best_w, best_f1 = 0.5, -1
        for w in np.arange(0.0, 1.01, 0.05):
            blended = w * tf_val_probs + (1 - w) * svm_val_probs
            f1 = f1_score(y_val, blended.argmax(axis=1), average="macro")
            if f1 > best_f1:
                best_f1, best_w = f1, w
        weight = best_w
        print(f"Auto-selected transformer weight = {weight:.2f} (internal macro-F1 = {best_f1:.4f})")
    else:
        weight = args.weight
        blended = weight * tf_val_probs + (1 - weight) * svm_val_probs
        f1 = f1_score(y_val, blended.argmax(axis=1), average="macro")
        print(f"Using transformer weight = {weight:.2f} (internal macro-F1 = {f1:.4f})")

    tf_only_f1 = f1_score(y_val, tf_val_probs.argmax(axis=1), average="macro")
    svm_only_f1 = f1_score(y_val, svm_val_probs.argmax(axis=1), average="macro")
    print(f"For reference -- transformer alone: {tf_only_f1:.4f}   SVM alone: {svm_only_f1:.4f}")

    # now predict on the real submission file
    rows = read_csv_dicts(args.input)
    id_key = find_id_key(rows[0].keys())
    comment_key = find_comment_key(rows[0].keys())
    ids = [r[id_key] for r in rows]
    texts = [clean_text(r[comment_key]) for r in rows]

    print("Scoring transformer on submission inputs...")
    tf_sub_probs = transformer_probs(args.model_dir, texts)
    print("Scoring SVM on submission inputs...")
    svm_sub_probs = svm_predict_proba(texts)

    blended_sub = weight * tf_sub_probs + (1 - weight) * svm_sub_probs
    final_preds = [labels_list[i] for i in blended_sub.argmax(axis=1)]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, p in zip(ids, final_preds):
            w.writerow([i, p])
    print(f"Wrote {args.out} with {len(final_preds)} predictions.")


if __name__ == "__main__":
    main()