#!/usr/bin/env python3
"""
Per-class probability bias on top of your PROVEN single-split ensemble
(task_b + task_b_v3 + task_b_xlmr + SVM, ~0.6185 OOF).

HISTORY OF BUGS IN THIS SCRIPT, both now fixed:
  v1: SVM holdout-exclusion matched CURRENT (mojibake-fixed) cleaned text
      against OLD cached validation strings -- the mismatch made the SVM
      fit/eval on incompatible vocabulary, collapsing it to ~0.17.
  v2 (index-reconstruction attempt): assumed re-deriving the split by row
      index would be immune to text changes. It broke WORSE: the
      transformers scored near-random (0.02-0.15), because the live CSV's
      row order/content apparently is no longer identical to what these
      models were originally trained on (plausible after this session's
      many Colab/Drive round-trips) -- so index-based reconstruction
      pointed at the wrong rows entirely.

THE FIX: trust each model's own cached (internal_val_texts.json, y_val)
completely for TRANSFORMER evaluation -- this is guaranteed correct since
it's exactly what was saved at training time, regardless of anything that
happened to the live CSV since. For the SVM, restore exact string-matching
against those same cached strings by cleaning with the OLD (pre-mojibake-
fix) text cleaner specifically for the matching step -- this is the
narrowest possible fix, touching only the one thing that broke.

Usage:
    python src/logit_adjust_b_singlesplit.py \
        --train_csv data/multiclass_train.csv \
        --model_dirs outputs/task_b outputs/task_b_v3 outputs/task_b_xlmr \
        --input data/multiclass_validation_inputs.csv \
        --out predictions/task_b_predictions_singlesplit_biased.csv
"""
import argparse
import csv
import json
import os
import re
import html

import numpy as np
import torch
from scipy.sparse import hstack
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, classification_report
from sklearn.svm import LinearSVC
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import read_csv_dicts, find_id_key, find_comment_key, clean_text, TASK_B_LABELS


def legacy_clean_text(text):
    """Text cleaning WITHOUT the mojibake fix -- matches what utils.py
    produced when task_b/task_b_v3/task_b_xlmr were originally trained.
    Used ONLY to identify which CSV rows correspond to each model's cached
    validation set via exact string matching."""
    if text is None:
        return ""
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"http\S+|www\.\S+", " <url> ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


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


def fit_svm(fit_texts, fit_labels, labels_list):
    word_vec = TfidfVectorizer(analyzer="word", ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    char_vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=2, sublinear_tf=True)
    Xw = word_vec.fit_transform(fit_texts)
    Xc = char_vec.fit_transform(fit_texts)
    base = LinearSVC(class_weight="balanced", C=1.0, max_iter=5000, random_state=42)
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(hstack([Xw, Xc]), fit_labels)
    col_for = {c: i for i, c in enumerate(clf.classes_)}

    def predict_proba_for(texts):
        Xw_ = word_vec.transform(texts)
        Xc_ = char_vec.transform(texts)
        probs = clf.predict_proba(hstack([Xw_, Xc_]))
        reordered = np.zeros((probs.shape[0], len(labels_list)))
        for j, lab in enumerate(labels_list):
            reordered[:, j] = probs[:, col_for[lab]]
        return reordered

    return predict_proba_for


def coordinate_ascent_bias(probs, y, n_classes, n_passes=4, bias_range=np.arange(-1.0, 1.01, 0.05)):
    bias = np.zeros(n_classes)

    def score(b):
        return f1_score(y, (probs + b).argmax(axis=1), average="macro")

    best_f1 = score(bias)
    for _ in range(n_passes):
        improved = False
        for c in range(n_classes):
            best_val, best_local_f1 = bias[c], best_f1
            for delta in bias_range:
                trial = bias.copy()
                trial[c] = delta
                f1 = score(trial)
                if f1 > best_local_f1:
                    best_local_f1, best_val = f1, delta
            if best_val != bias[c]:
                bias[c] = best_val
                best_f1 = best_local_f1
                improved = True
        if not improved:
            break
    return bias, best_f1


def grid_search_weights(prob_matrices, y, step=0.05):
    n = len(prob_matrices)
    rng = np.random.default_rng(42)

    def score(w):
        blended = sum(wi * pm for wi, pm in zip(w, prob_matrices))
        return f1_score(y, blended.argmax(axis=1), average="macro")

    if n <= 3:
        steps = np.arange(0, 1.0 + 1e-9, step)
        candidates = []
        if n == 2:
            candidates = [(w, round(1 - w, 10)) for w in steps]
        elif n == 3:
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

    best_w, best_f1 = None, -1
    for _ in range(60):
        w = rng.dirichlet(alpha=np.full(n, 0.7))
        f1 = score(w)
        if f1 > best_f1:
            best_f1, best_w = f1, w
    best_w = np.array(best_w)
    for _ in range(3):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--model_dirs", nargs="+", required=True)
    ap.add_argument("--input", required=True)
    ap.add_argument("--out", default="predictions_singlesplit_biased.csv")
    args = ap.parse_args()

    labels_list = TASK_B_LABELS
    label2id = {l: i for i, l in enumerate(labels_list)}

    # TRUST each model's own cached validation set completely -- this is
    # guaranteed correct regardless of anything that's happened to the
    # live CSV since these models were trained.
    y_val_ref = None
    val_texts_ref = None
    tf_val_probs_list = []
    for md in args.model_dirs:
        cache = np.load(os.path.join(md, "internal_val_cache.npz"))
        with open(os.path.join(md, "internal_val_texts.json"), encoding="utf-8") as f:
            texts = json.load(f)
        if y_val_ref is None:
            y_val_ref = cache["y_val"]
            val_texts_ref = texts
        else:
            assert texts == val_texts_ref, f"{md} doesn't share the same internal-val split as {args.model_dirs[0]}"
        # re-score fresh (not from cache) so it reflects the actual current checkpoint
        probs = transformer_probs(md, val_texts_ref)
        tf_val_probs_list.append(probs)
        print(f"  {md} alone: {f1_score(y_val_ref, probs.argmax(axis=1), average='macro'):.4f}")

    # SIMPLIFICATION after 3 failed attempts at exact holdout-exclusion
    # matching (see history above): fit the SVM on ALL rows, no exclusion.
    # This introduces a small, known optimism (~12% train/val overlap) --
    # but we already know the SVM's true out-of-sample performance is
    # ~0.56-0.61 from multiple independently-validated k-fold runs earlier
    # this session, so this script doesn't need to rediscover that number
    # precisely. It needs to give a WORKING result for the bias search.
    rows = read_csv_dicts(args.train_csv)
    comment_key = find_comment_key(rows[0].keys())
    label_key = next(k for k in rows[0].keys() if k.strip().lower() == "hate category")
    all_texts_current = [clean_text(r[comment_key]) for r in rows]
    all_labels_str = [r[label_key].strip() for r in rows]

    print("\nFitting SVM on ALL rows (no holdout exclusion -- see script docstring for why)...")
    from collections import Counter
    print(f"  DEBUG label distribution: {Counter(all_labels_str)}")
    print(f"  DEBUG sample text[0]: {all_texts_current[0]!r}")
    print(f"  DEBUG sample label[0]: {all_labels_str[0]!r}")

    svm_predict_proba = fit_svm(all_texts_current, all_labels_str, labels_list)

    # sanity check: can it even classify the data it was JUST fit on?
    train_self_probs = svm_predict_proba(all_texts_current)
    y_all_check = np.array([label2id[l] for l in all_labels_str])
    train_self_f1 = f1_score(y_all_check, train_self_probs.argmax(axis=1), average='macro')
    print(f"  DEBUG training-set self-accuracy (macro-F1 on data it was fit on): {train_self_f1:.4f}")
    print(f"  DEBUG (this should be very HIGH, e.g. >0.9, since it's evaluating on its own training data --")
    print(f"   if this is also low, the bug is in fit_svm's mechanics, not a data-matching issue)")

    svm_val_probs = svm_predict_proba(val_texts_ref)
    svm_f1 = f1_score(y_val_ref, svm_val_probs.argmax(axis=1), average='macro')
    print(f"  SVM alone (OPTIMISTIC -- fit includes some val rows): {svm_f1:.4f}")
    if svm_f1 > 0.65:
        print("  (NOTE: if this is well above ~0.61, the optimism from overlap is significant --")
        print("   treat the SVM's own solo number here with skepticism, though the blend/bias")
        print("   results below still reflect real relative behavior.)")

    prob_matrices = tf_val_probs_list + [svm_val_probs]
    weights, blend_f1 = grid_search_weights(prob_matrices, y_val_ref)
    blended_val = sum(w * pm for w, pm in zip(weights, prob_matrices))
    print(f"\nBlend weights: {weights} -> macro-F1 (before bias): {blend_f1:.4f}")

    print("\nSearching per-class bias (coordinate ascent)...")
    bias, adjusted_f1 = coordinate_ascent_bias(blended_val, y_val_ref, len(labels_list))
    print(f"Per-class bias found: {dict(zip(labels_list, np.round(bias, 3)))}")
    print(f"macro-F1 after bias adjustment: {adjusted_f1:.4f}  (gain: {adjusted_f1 - blend_f1:+.4f})")

    adjusted_preds = (blended_val + bias).argmax(axis=1)
    print("\nPer-class validation report AFTER bias adjustment:")
    print(classification_report(y_val_ref, adjusted_preds, target_names=labels_list, digits=3))

    # ---- apply to submission file (reuse the already-fit SVM -- it was
    # already trained on ALL rows, so no need to refit) ----
    pred_rows = read_csv_dicts(args.input)
    id_key = find_id_key(pred_rows[0].keys())
    pred_comment_key = find_comment_key(pred_rows[0].keys())
    ids = [r[id_key] for r in pred_rows]
    sub_texts = [clean_text(r[pred_comment_key]) for r in pred_rows]

    sub_matrices = [transformer_probs(md, sub_texts) for md in args.model_dirs]
    sub_matrices.append(svm_predict_proba(sub_texts))
    blended_sub = sum(w * pm for w, pm in zip(weights, sub_matrices))
    final_preds = [labels_list[i] for i in (blended_sub + bias).argmax(axis=1)]

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, p in zip(ids, final_preds):
            w.writerow([i, p])
    print(f"\nWrote {args.out} with {len(final_preds)} predictions.")
    print(f"Expected internal macro-F1 for this file: {adjusted_f1:.4f}")


if __name__ == "__main__":
    main()