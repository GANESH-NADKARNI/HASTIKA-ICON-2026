"""
Regenerates the char n-gram TF-IDF + SVM out-of-fold probabilities and saves
them in the same internal_val_cache.npz format (val_probs, y_val) used by
the MuRIL / XLM-R-large runs, so all three can be blended together.

Run from repo root or from src/ -- paths are resolved relative to this file.
Output: outputs/task_a_svm/internal_val_cache.npz
"""
import html
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score

SEED = 42
N_FOLDS = 5

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
OUT_DIR = ROOT_DIR / "outputs" / "task_a_svm"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def fix_mojibake(s):
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def clean_text(t):
    t = fix_mojibake(str(t))
    t = html.unescape(t)
    t = re.sub(r"<[^>]+>", " ", t)
    t = re.sub(r"http\S+|www\.\S+", " ", t)
    t = re.sub(r"([!?.,])\1{2,}", r"\1\1", t)
    t = re.sub(r"(.)\1{3,}", r"\1\1\1", t)
    return re.sub(r"\s+", " ", t).strip()


# IMPORTANT: only binary_train.csv, same rows/order as the MuRIL and
# XLM-R-large caches -- NOT the augmented set with the extra 644 rows,
# so this lines up row-for-row with the other two caches.
train_df = pd.read_csv(DATA_DIR / "binary_train.csv").dropna(subset=["Comment", "Label"]).reset_index(drop=True)
train_df["clean"] = train_df["Comment"].apply(clean_text)

y = (train_df["Label"] == "Hate").astype(int).values
X_text = train_df["clean"].values

skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_probs = np.zeros((len(X_text), 2))

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_text, y)):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform(X_text[tr_idx])
    Xva = vec.transform(X_text[va_idx])

    base_clf = LinearSVC(C=1.0, class_weight="balanced")
    clf = CalibratedClassifierCV(base_clf, cv=3, method="sigmoid")
    clf.fit(Xtr, y[tr_idx])

    oof_probs[va_idx] = clf.predict_proba(Xva)
    fold_f1 = f1_score(y[va_idx], clf.predict(Xva), average="macro")
    print(f"fold {fold + 1}/{N_FOLDS}: macro-F1 = {fold_f1:.4f}")

overall_f1 = f1_score(y, oof_probs.argmax(axis=1), average="macro")
print(f"\nSVM OOF macro-F1 @ argmax: {overall_f1:.4f}")

np.savez(OUT_DIR / "internal_val_cache.npz", val_probs=oof_probs, y_val=y)
print(f"Saved {OUT_DIR / 'internal_val_cache.npz'}")