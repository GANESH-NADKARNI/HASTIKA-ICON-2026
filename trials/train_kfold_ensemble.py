#!/usr/bin/env python3
"""
K-fold cross-validation training + bagging, blended with a k-fold SVM.

WHY THIS MATTERS MORE THAN FURTHER HYPERPARAMETER TUNING:
So far, every internal score has come from a single 88/12 train/val split.
With Task B's ~2,779 training rows, a 12% held-out slice is only ~380 rows
-- small enough that a different random split would give a meaningfully
different internal score. That's very likely why internal validation
scores have consistently overestimated the real leaderboard score by
~0.07-0.09: it's not overfitting so much as measurement noise from a small
single split.

K-fold fixes both problems at once:
  1. Every training row gets evaluated exactly once, by a model that never
     saw it during training (out-of-fold / OOF predictions) -- using ALL
     ~2,779 rows for validation instead of ~380, for a genuinely reliable
     internal macro-F1.
  2. At prediction time, you average all N fold models together ("bagging")
     instead of trusting a single model -- this alone is usually a bigger
     accuracy lever than further LR/epoch tuning, because each fold model
     saw a slightly different 80% of the data and makes different mistakes,
     the same way your task_b vs task_b_v3 vs xlmr models did.

The SVM is also cross-validated across the SAME folds (via a manual
leave-fold-out loop, refitting the vectorizer each time -- no leakage), so
its OOF probabilities are directly comparable to the transformer's and can
be blended with the same weight-search approach as multi_ensemble.py.

Usage:
    python src/train_kfold_ensemble.py --task a \
        --train_csv data/binary_train.csv \
        --input data/binary_validation_inputs.csv \
        --output_dir outputs/task_a_kfold \
        --out predictions/task_a_predictions_kfold.csv

    python src/train_kfold_ensemble.py --task b \
        --train_csv data/multiclass_train.csv \
        --input data/multiclass_validation_inputs.csv \
        --output_dir outputs/task_b_kfold \
        --out predictions/task_b_predictions_kfold.csv \
        --n_folds 5 --model_name xlm-roberta-base

Time expectation on an RTX 4060: roughly N_FOLDS x (single-run time from
train.py) -- e.g. 5 folds x ~10 min = ~50 min for Task B, more for Task A
since it has more data per fold. Use --n_folds 3 if you want a faster,
slightly-less-robust version.
"""
import argparse
import inspect
import json
import os

import numpy as np
import torch
from scipy.sparse import hstack, csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from utils import read_csv_dicts, find_id_key, find_comment_key, clean_text, TASK_A_LABELS, TASK_B_LABELS
from train import set_seed, TextDataset, WeightedTrainer, compute_metrics, build_training_args


def train_one_fold(texts_train, y_train, texts_val, y_val, labels_list, args, fold_output_dir):
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(labels_list),
        id2label={i: l for i, l in enumerate(labels_list)},
        label2id={l: i for i, l in enumerate(labels_list)},
    )

    train_ds = TextDataset(texts_train, y_train, tokenizer, args.max_len)
    val_ds = TextDataset(texts_val, y_val, tokenizer, args.max_len)

    class_weights_arr = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(labels_list)), y=y_train
    )
    if args.oversample:
        loss_weights_arr = np.sqrt(class_weights_arr)
        sample_weights = torch.tensor([class_weights_arr[l] for l in y_train], dtype=torch.double)
    else:
        loss_weights_arr = class_weights_arr
        sample_weights = None
    class_weights = torch.tensor(loss_weights_arr, dtype=torch.float)

    training_args = build_training_args(args, len(train_ds))
    training_args.output_dir = fold_output_dir
    training_args.save_total_limit = 1

    trainer = WeightedTrainer(
        class_weights=class_weights,
        sample_weights=sample_weights,
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )
    trainer.train()

    pred_output = trainer.predict(val_ds)
    val_probs = torch.softmax(torch.tensor(pred_output.predictions), dim=-1).numpy()

    trainer.save_model(fold_output_dir)
    tokenizer.save_pretrained(fold_output_dir)

    import shutil
    import glob
    for ckpt_dir in glob.glob(os.path.join(fold_output_dir, "checkpoint-*")):
        shutil.rmtree(ckpt_dir, ignore_errors=True)

    return val_probs


def transformer_probs_from_dir(model_dir, texts, batch_size=32, max_len=160):
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


import re


def handcrafted_features(texts):
    """Same feature set validated by svm_tuned.py's grid search -- helped
    in every one of 18 configs tested, not just the winner. Kept here
    (rather than imported from svm_tuned.py) so this function has no
    dependency on which scripts happen to be present."""
    KANNADA_RANGE = (0x0C80, 0x0CFF)
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


def svm_kfold_oof_and_final(texts, labels_str, labels_list, fold_indices, C=1.0, char_ngram_range=(3, 5), use_handcrafted=False):
    """Cross-validated SVM OOF probabilities (no leakage: vectorizer +
    classifier refit per fold, evaluated only on that fold's held-out rows),
    plus a final SVM refit on ALL data for scoring the real submission file.

    NOTE on defaults: svm_tuned.py's standalone grid search found C=0.5,
    char n-grams (2,5), and hand-crafted features improved the SVM's OWN
    isolated accuracy (0.8062 vs 0.8024). But when plugged into the full
    blend here, that config made the OVERALL ensemble WORSE (0.8231 vs
    0.8290) -- likely because the new features shifted the SVM's errors
    to overlap more with the transformers' errors, reducing the
    complementary diversity blending depends on. A component improving in
    isolation does not guarantee the ensemble improves. Defaults are kept
    at the values that empirically produced your best verified blend
    (0.8290 OOF); pass use_handcrafted=True / different C or ngrams only
    if you've re-verified it helps the FULL blend, not just the SVM alone."""
    n = len(texts)
    oof_probs = np.zeros((n, len(labels_list)))

    for fold_i, (train_idx, val_idx) in enumerate(fold_indices):
        fit_texts = [texts[i] for i in train_idx]
        fit_labels = [labels_str[i] for i in train_idx]
        val_texts_fold = [texts[i] for i in val_idx]

        word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
        char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngram_range, min_df=2, sublinear_tf=True)
        Xw = word_vec.fit_transform(fit_texts)
        Xc = char_vec.fit_transform(fit_texts)

        if use_handcrafted:
            scaler = StandardScaler()
            hc_fit = scaler.fit_transform(handcrafted_features(fit_texts))
            X = hstack([Xw, Xc, csr_matrix(hc_fit)])
        else:
            X = hstack([Xw, Xc])

        base = LinearSVC(class_weight="balanced", C=C, max_iter=5000, random_state=42)
        clf = CalibratedClassifierCV(base, method="sigmoid", cv=3, n_jobs=None)
        clf.fit(X, fit_labels)
        col_for = {c: i for i, c in enumerate(clf.classes_)}

        Xw_v = word_vec.transform(val_texts_fold)
        Xc_v = char_vec.transform(val_texts_fold)
        if use_handcrafted:
            hc_val = scaler.transform(handcrafted_features(val_texts_fold))
            probs = clf.predict_proba(hstack([Xw_v, Xc_v, csr_matrix(hc_val)]))
        else:
            probs = clf.predict_proba(hstack([Xw_v, Xc_v]))
        reordered = np.zeros((probs.shape[0], len(labels_list)))
        for j, lab in enumerate(labels_list):
            reordered[:, j] = probs[:, col_for[lab]]
        oof_probs[val_idx] = reordered
        print(f"    SVM fold {fold_i}: fit on {len(train_idx)}, scored {len(val_idx)}")

    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=char_ngram_range, min_df=2, sublinear_tf=True)
    Xw = word_vec.fit_transform(texts)
    Xc = char_vec.fit_transform(texts)
    if use_handcrafted:
        scaler = StandardScaler()
        hc = scaler.fit_transform(handcrafted_features(texts))
        X = hstack([Xw, Xc, csr_matrix(hc)])
    else:
        X = hstack([Xw, Xc])
    base = LinearSVC(class_weight="balanced", C=C, max_iter=5000, random_state=42)
    final_clf = CalibratedClassifierCV(base, method="sigmoid", cv=3, n_jobs=None)
    final_clf.fit(X, labels_str)
    col_for_final = {c: i for i, c in enumerate(final_clf.classes_)}

    def predict_proba_for(new_texts):
        Xw_ = word_vec.transform(new_texts)
        Xc_ = char_vec.transform(new_texts)
        if use_handcrafted:
            hc_ = scaler.transform(handcrafted_features(new_texts))
            probs = final_clf.predict_proba(hstack([Xw_, Xc_, csr_matrix(hc_)]))
        else:
            probs = final_clf.predict_proba(hstack([Xw_, Xc_]))
        reordered = np.zeros((probs.shape[0], len(labels_list)))
        for j, lab in enumerate(labels_list):
            reordered[:, j] = probs[:, col_for_final[lab]]
        return reordered

    return oof_probs, predict_proba_for




def search_weight(tf_oof, svm_oof, y, step=0.02):
    best_w, best_f1 = 0.5, -1
    for w in np.arange(0.0, 1.0 + 1e-9, step):
        blended = w * tf_oof + (1 - w) * svm_oof
        f1 = f1_score(y, blended.argmax(axis=1), average="macro")
        if f1 > best_f1:
            best_f1, best_w = f1, w
    return best_w, best_f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a", "b"], required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--input", required=True, help="the *_validation_inputs.csv to predict on")
    ap.add_argument("--output_dir", required=True, help="base dir; fold models go in output_dir/fold0, fold1, ...")
    ap.add_argument("--out", default="predictions_kfold.csv")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--model_name", default="google/muril-base-cased")
    ap.add_argument("--epochs", type=float, default=None)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--oversample", type=int, default=1)
    ap.add_argument("--max_grad_norm", type=float, default=1.0)
    ap.add_argument("--fp16", action="store_true", default=torch.cuda.is_available())
    args = ap.parse_args()

    if args.epochs is None:
        args.epochs = 6.0 if args.task == "a" else 10.0
    if args.lr is None:
        args.lr = 2e-5 if args.task == "a" else 1e-5

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    labels_list = TASK_A_LABELS if args.task == "a" else TASK_B_LABELS
    label2id = {l: i for i, l in enumerate(labels_list)}
    label_col = "Label" if args.task == "a" else "Hate Category"

    rows = read_csv_dicts(args.train_csv)
    comment_key = find_comment_key(rows[0].keys())
    label_key = next(k for k in rows[0].keys() if k.strip().lower() == label_col.lower())
    texts = [clean_text(r[comment_key]) for r in rows]
    raw_labels = [r[label_key].strip() for r in rows]
    y = np.array([label2id[l] for l in raw_labels])

    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    fold_indices = list(skf.split(texts, y))

    n = len(texts)
    tf_oof = np.zeros((n, len(labels_list)))

    print(f"=== Training {args.n_folds} transformer folds ({args.model_name}) ===")
    for fold_i, (train_idx, val_idx) in enumerate(fold_indices):
        texts_val = [texts[i] for i in val_idx]
        y_val = [int(y[i]) for i in val_idx]
        fold_dir = os.path.join(args.output_dir, f"fold{fold_i}")

        already_done = os.path.exists(os.path.join(fold_dir, "config.json"))
        if already_done:
            print(f"\n--- Fold {fold_i+1}/{args.n_folds}: already trained, found at {fold_dir}. "
                  f"Loading it and just re-scoring its held-out rows (no retraining). ---")
            val_probs = transformer_probs_from_dir(fold_dir, texts_val, max_len=args.max_len)
        else:
            print(f"\n--- Fold {fold_i+1}/{args.n_folds}: training ---")
            texts_train = [texts[i] for i in train_idx]
            y_train = [int(y[i]) for i in train_idx]
            val_probs = train_one_fold(texts_train, y_train, texts_val, y_val, labels_list, args, fold_dir)

        tf_oof[val_idx] = val_probs
        fold_f1 = f1_score(y_val, val_probs.argmax(axis=1), average="macro")
        print(f"Fold {fold_i} macro-F1: {fold_f1:.4f}")

    tf_oof_f1 = f1_score(y, tf_oof.argmax(axis=1), average="macro")
    print(f"\n=== Transformer OOF macro-F1 (all {n} rows, low-variance estimate): {tf_oof_f1:.4f} ===")
    print(classification_report(y, tf_oof.argmax(axis=1), target_names=labels_list, digits=3))

    print("\n=== Cross-validated SVM (same folds) ===")
    svm_oof, svm_predict_proba_final = svm_kfold_oof_and_final(texts, raw_labels, labels_list, fold_indices)
    svm_oof_f1 = f1_score(y, svm_oof.argmax(axis=1), average="macro")
    print(f"SVM OOF macro-F1: {svm_oof_f1:.4f}")

    weight, blend_f1 = search_weight(tf_oof, svm_oof, y)
    print(f"\nBest blend: transformer_weight={weight:.2f} -> OOF macro-F1 = {blend_f1:.4f}")
    print(f"(vs transformer alone {tf_oof_f1:.4f}, SVM alone {svm_oof_f1:.4f})")

    np.savez(os.path.join(args.output_dir, "internal_val_cache.npz"), val_probs=tf_oof, y_val=y)
    with open(os.path.join(args.output_dir, "internal_val_texts.json"), "w", encoding="utf-8") as f:
        json.dump(texts, f, ensure_ascii=False)

    print("\n=== Scoring submission inputs ===")
    pred_rows = read_csv_dicts(args.input)
    id_key = find_id_key(pred_rows[0].keys())
    pred_comment_key = find_comment_key(pred_rows[0].keys())
    ids = [r[id_key] for r in pred_rows]
    sub_texts = [clean_text(r[pred_comment_key]) for r in pred_rows]

    fold_sub_probs = []
    for fold_i in range(args.n_folds):
        fold_dir = os.path.join(args.output_dir, f"fold{fold_i}")
        print(f"  scoring fold {fold_i}...")
        fold_sub_probs.append(transformer_probs_from_dir(fold_dir, sub_texts, max_len=args.max_len))
    tf_sub_probs = np.mean(fold_sub_probs, axis=0)

    print("  scoring final SVM...")
    svm_sub_probs = svm_predict_proba_final(sub_texts)

    blended_sub = weight * tf_sub_probs + (1 - weight) * svm_sub_probs
    final_preds = [labels_list[i] for i in blended_sub.argmax(axis=1)]

    import csv
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, p in zip(ids, final_preds):
            w.writerow([i, p])
    print(f"\nWrote {args.out} with {len(final_preds)} predictions.")
    print(f"Expected internal macro-F1 for this file (OOF estimate): {blend_f1:.4f}")


if __name__ == "__main__":
    main()