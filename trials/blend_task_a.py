"""
Blends out-of-fold probabilities from three already-trained models:
  - char n-gram TF-IDF + SVM      (outputs/task_a_svm)
  - MuRIL                          (outputs/task_a_kfold_muril)
  - XLM-R-large                    (outputs/task_a_kfold_xlmr_large_v2)

Verifies all three caches are aligned to the same rows (same y_val), then
grid-searches the 3-way blend weights + decision threshold to maximize
macro-F1 on the combined out-of-fold predictions.

This does NOT retrain anything -- it only reads the saved .npz caches.
Run from repo root or from src/.
"""
from pathlib import Path
import numpy as np
from sklearn.metrics import f1_score

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
OUT_DIR = ROOT_DIR / "outputs"

CACHES = {
    "svm": OUT_DIR / "task_a_svm" / "internal_val_cache.npz",
    "muril": OUT_DIR / "task_a_kfold_muril" / "internal_val_cache.npz",
    "xlmr": OUT_DIR / "task_a_kfold_xlmr_large_v2" / "internal_val_cache.npz",
}

# ---------------------------------------------------------------------------
# 1. Load + verify alignment
# ---------------------------------------------------------------------------
probs = {}
y_ref = None
for name, path in CACHES.items():
    if not path.exists():
        raise FileNotFoundError(f"Missing cache for '{name}': {path}")
    d = np.load(path)
    val_probs, y_val = d["val_probs"], d["y_val"]
    if y_ref is None:
        y_ref = y_val
    else:
        if len(y_val) != len(y_ref):
            raise ValueError(
                f"'{name}' has {len(y_val)} rows but the reference has {len(y_ref)} -- "
                "these caches are not from the same dataset/split and cannot be blended directly."
            )
        if not np.array_equal(y_val, y_ref):
            n_mismatch = np.sum(y_val != y_ref)
            raise ValueError(
                f"'{name}' y_val does not match the reference y_val "
                f"({n_mismatch}/{len(y_ref)} labels differ at the same row index). "
                "This means the fold order/shuffle differs between runs -- "
                "the row-wise blend below would be silently wrong. "
                "Re-check that both scripts used the same random_state and the same "
                "base binary_train.csv (not the augmented +644-row version)."
            )
    probs[name] = val_probs[:, 1]   # P(Hate)
    fold_f1 = f1_score(y_ref, (probs[name] > 0.5).astype(int), average="macro")
    print(f"{name:>6} OOF macro-F1 @0.5: {fold_f1:.4f}")

y = y_ref
print()

# ---------------------------------------------------------------------------
# 2. Grid search: 3-way blend weights (simplex) + decision threshold
# ---------------------------------------------------------------------------
best_score = -1
best_combo = None

weight_steps = np.arange(0.0, 1.01, 0.05)
threshold_steps = np.arange(0.30, 0.71, 0.01)

for w_svm in weight_steps:
    for w_muril in weight_steps:
        w_xlmr = 1.0 - w_svm - w_muril
        if w_xlmr < -1e-9 or w_xlmr > 1.0 + 1e-9:
            continue
        w_xlmr = max(0.0, w_xlmr)
        blend = w_svm * probs["svm"] + w_muril * probs["muril"] + w_xlmr * probs["xlmr"]
        for t in threshold_steps:
            f1 = f1_score(y, (blend > t).astype(int), average="macro")
            if f1 > best_score:
                best_score = f1
                best_combo = (w_svm, w_muril, w_xlmr, t)

w_svm, w_muril, w_xlmr, t = best_combo
print(f"Best blend: svm={w_svm:.2f}  muril={w_muril:.2f}  xlmr={w_xlmr:.2f}  threshold={t:.2f}")
print(f"Best OOF macro-F1: {best_score:.4f}")

# ---------------------------------------------------------------------------
# 3. Also report the best 2-way blends for comparison (sanity check that
#    all three models are actually pulling their weight)
# ---------------------------------------------------------------------------
print("\n--- pairwise comparisons for reference ---")
pairs = [("svm", "muril"), ("svm", "xlmr"), ("muril", "xlmr")]
for a, b in pairs:
    best_pair_score = -1
    best_pair = None
    for w in weight_steps:
        blend = w * probs[a] + (1 - w) * probs[b]
        for t in threshold_steps:
            f1 = f1_score(y, (blend > t).astype(int), average="macro")
            if f1 > best_pair_score:
                best_pair_score = f1
                best_pair = (w, t)
    print(f"{a} + {b}: best macro-F1 = {best_pair_score:.4f}  (w_{a}={best_pair[0]:.2f}, threshold={best_pair[1]:.2f})")

print(f"\n=> 3-way blend gain over best single model: "
      f"{best_score - max(f1_score(y, (probs[n] > 0.5).astype(int), average='macro') for n in probs):.4f}")

# ---------------------------------------------------------------------------
# 4. Refine with a finer local grid around the coarse optimum
# ---------------------------------------------------------------------------
print("\n--- refining around the coarse optimum ---")
refine_score = best_score
refine_combo = best_combo
fine_weight_steps = np.arange(max(0.0, w_svm - 0.1), min(1.0, w_svm + 0.1) + 0.001, 0.01)
fine_weight_steps_m = np.arange(max(0.0, w_muril - 0.1), min(1.0, w_muril + 0.1) + 0.001, 0.01)
fine_thresholds = np.arange(max(0.0, t - 0.05), min(1.0, t + 0.05) + 0.001, 0.005)

for ws in fine_weight_steps:
    for wm in fine_weight_steps_m:
        wx = 1.0 - ws - wm
        if wx < -1e-9 or wx > 1.0 + 1e-9:
            continue
        wx = max(0.0, wx)
        blend = ws * probs["svm"] + wm * probs["muril"] + wx * probs["xlmr"]
        for tt in fine_thresholds:
            f1 = f1_score(y, (blend > tt).astype(int), average="macro")
            if f1 > refine_score:
                refine_score = f1
                refine_combo = (ws, wm, wx, tt)

print(f"Refined blend: svm={refine_combo[0]:.3f}  muril={refine_combo[1]:.3f}  "
      f"xlmr={refine_combo[2]:.3f}  threshold={refine_combo[3]:.3f}")
print(f"Refined OOF macro-F1: {refine_score:.4f}")

# ---------------------------------------------------------------------------
# 5. Logistic regression stacking (nested CV, so the score is still a fair
#    out-of-fold estimate rather than optimistic in-sample fitting)
# ---------------------------------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

print("\n--- logistic regression stacker (nested 5-fold) ---")
X_stack = np.column_stack([probs["svm"], probs["muril"], probs["xlmr"]])
stack_oof = np.zeros(len(y))
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
for tr_idx, va_idx in skf.split(X_stack, y):
    meta = LogisticRegression(class_weight="balanced", max_iter=1000)
    meta.fit(X_stack[tr_idx], y[tr_idx])
    stack_oof[va_idx] = meta.predict_proba(X_stack[va_idx])[:, 1]

best_stack_score, best_stack_t = -1, None
for tt in np.arange(0.30, 0.71, 0.01):
    f1 = f1_score(y, (stack_oof > tt).astype(int), average="macro")
    if f1 > best_stack_score:
        best_stack_score, best_stack_t = f1, tt

print(f"Stacked (nested OOF) macro-F1: {best_stack_score:.4f} @ threshold={best_stack_t:.2f}")

print("\n=== SUMMARY ===")
print(f"Best single model:        {max(f1_score(y, (probs[n] > 0.5).astype(int), average='macro') for n in probs):.4f}")
print(f"Coarse 3-way blend:       {best_score:.4f}")
print(f"Refined 3-way blend:      {refine_score:.4f}")
print(f"Logistic stack (nested):  {best_stack_score:.4f}")