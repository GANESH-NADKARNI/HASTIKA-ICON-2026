"""
Task A - Binary Hate Speech Detection (Kannada-English code-mixed)
IndicBERT v2 + char n-gram SVM blend

Run this in an environment with GPU + internet access (e.g. Google Colab).
It CANNOT run as-is in a sandbox without internet access to huggingface.co.

Install first:
    pip install transformers torch scikit-learn pandas numpy accelerate

Expected files in the same folder:
    binary_train.csv               (id, Comment, Label)
    binary_validation_inputs.csv   (id, Comment)
"""

import html
import re
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL_NAME = "ai4bharat/IndicBERTv2-MLM-only"   # swap for IndicBERTv2-MLM-Back-TLM if you want to try the TLM variant
MAX_LEN = 128
N_FOLDS = 5
EPOCHS = 8
BATCH_SIZE = 16
LR = 3e-5
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

torch.manual_seed(SEED)
np.random.seed(SEED)

# Paths are resolved relative to this script's location, not the current
# working directory, so this works whether you run it from repo root or
# from inside src/. Matches a layout of: repo/data, repo/predictions, repo/src.
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT_DIR = SCRIPT_DIR.parent
DATA_DIR = ROOT_DIR / "data"
PRED_DIR = ROOT_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Cleaning: fix mojibake FIRST, then normalize
# ---------------------------------------------------------------------------
def fix_mojibake(s: str) -> str:
    """Repairs UTF-8 text that was incorrectly decoded as Latin-1 and re-saved.
    Detected in this dataset: ~16% of comments have Kannada script / emoji
    double-encoded, e.g. 'à²à³à²¹à³à²²à²¿' should be 'ಕೊಹ್ಲಿ'.
    Falls back to the original string if the round-trip isn't valid
    (i.e. the text was never mojibake in the first place).
    """
    try:
        return s.encode("latin-1").decode("utf-8")
    except (UnicodeEncodeError, UnicodeDecodeError):
        return s


def clean_text(t) -> str:
    t = fix_mojibake(str(t))
    t = html.unescape(t)                              # &quot; -> "
    t = re.sub(r"<[^>]+>", " ", t)                     # strip <br> etc.
    t = re.sub(r"http\S+|www\.\S+", " ", t)            # strip URLs
    t = re.sub(r"([!?.,])\1{2,}", r"\1\1", t)          # !!!! -> !!
    t = re.sub(r"(.)\1{3,}", r"\1\1\1", t)             # yeahhhhh -> yeahhh
    t = re.sub(r"\s+", " ", t).strip()
    return t


# ---------------------------------------------------------------------------
# 2. Load data (+ merge in Task B category labels for the auxiliary head)
# ---------------------------------------------------------------------------
train_df = pd.read_csv(DATA_DIR / "binary_train.csv").dropna(subset=["Comment", "Label"]).reset_index(drop=True)
test_df = pd.read_csv(DATA_DIR / "binary_validation_inputs.csv").reset_index(drop=True)
multi_df = pd.read_csv(DATA_DIR / "multiclass_train.csv").dropna(subset=["Comment", "Hate Category"])

# multiclass_train.csv contains ~644 Hate-labeled comments that are absent
# from binary_train.csv entirely (not mislabeled -- just missing). Recover
# them as real extra training data: ~10% more rows, all Hate, all with a
# category label for the auxiliary head.
extra_ids = set(multi_df["id"]) - set(train_df["id"])
extra_df = multi_df[multi_df["id"].isin(extra_ids)][["id", "Comment"]].copy()
extra_df["Label"] = "Hate"
print(f"Recovered {len(extra_df)} Hate rows present in multiclass_train.csv but missing from binary_train.csv")
train_df = pd.concat([train_df, extra_df], ignore_index=True)

train_df["clean"] = train_df["Comment"].apply(clean_text)
test_df["clean"] = test_df["Comment"].apply(clean_text)

# Category labels only exist for a subset of the Hate rows (matched by id).
# Non-hate rows and any hate rows missing a category get -100, which is
# nn.CrossEntropyLoss's default ignore_index -- their category loss is
# simply skipped, only the binary loss applies to them.
CATEGORIES = sorted(multi_df["Hate Category"].unique().tolist())
CAT2IDX = {c: i for i, c in enumerate(CATEGORIES)}
N_CATS = len(CATEGORIES)
print(f"Auxiliary category task: {N_CATS} classes -> {CATEGORIES}")

train_df = train_df.merge(multi_df[["id", "Hate Category"]], on="id", how="left")
train_df["cat_idx"] = train_df["Hate Category"].map(CAT2IDX)
train_df["cat_idx"] = train_df["cat_idx"].fillna(-100).astype(int)
matched = (train_df["cat_idx"] >= 0).sum()
print(f"Category labels matched to {matched}/{len(train_df)} training rows "
      f"({(train_df['Label'] == 'Hate').sum()} are Hate)")

# Global class weights for the auxiliary head (computed once, reused every fold).
cat_counts = multi_df["Hate Category"].value_counts().reindex(CATEGORIES).values
cat_weights = torch.tensor(cat_counts.sum() / (N_CATS * cat_counts), dtype=torch.float32).to(DEVICE)

y = (train_df["Label"] == "Hate").astype(int).values
y_cat = train_df["cat_idx"].values
X_text = train_df["clean"].values
X_test_text = test_df["clean"].values


# ---------------------------------------------------------------------------
# 3. IndicBERT v2 multi-task fine-tuning: shared encoder, binary + category
#    heads. The category head only gets gradient from rows that have a
#    category label; this regularizes the shared representation so the
#    binary head learns *what kind* of hate it's looking at, not just a
#    surface pattern match. (5-fold, out-of-fold + test predictions)
# ---------------------------------------------------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
AUX_WEIGHT = 0.4   # relative weight of the category loss vs. the binary loss


class HateDataset(Dataset):
    def __init__(self, texts, labels=None, cat_labels=None):
        self.texts = list(texts)
        self.labels = labels
        self.cat_labels = cat_labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = tokenizer(
            self.texts[idx],
            truncation=True,
            max_length=MAX_LEN,
            padding="max_length",
            return_tensors="pt",
        )
        item = {k: v.squeeze(0) for k, v in enc.items()}
        if self.labels is not None:
            item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.cat_labels is not None:
            item["cat_labels"] = torch.tensor(self.cat_labels[idx], dtype=torch.long)
        return item


def mean_pool(last_hidden_state, attention_mask):
    """Attention-masked mean pooling. More reliable than raw [CLS] for
    MLM-only checkpoints that were never trained with a pooling objective."""
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = torch.sum(last_hidden_state * mask, dim=1)
    counts = torch.clamp(mask.sum(dim=1), min=1e-9)
    return summed / counts


class MultiTaskClassifier(nn.Module):
    def __init__(self, model_name, n_bin=2, n_cat=N_CATS, dropout=0.3):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(model_name)
        hidden = self.encoder.config.hidden_size
        self.dropout = nn.Dropout(dropout)
        self.bin_head = nn.Linear(hidden, n_bin)
        self.cat_head = nn.Linear(hidden, n_cat)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        kwargs = {"input_ids": input_ids, "attention_mask": attention_mask}
        if token_type_ids is not None:
            kwargs["token_type_ids"] = token_type_ids
        out = self.encoder(**kwargs)
        pooled = self.dropout(mean_pool(out.last_hidden_state, attention_mask))
        return self.bin_head(pooled), self.cat_head(pooled)


def train_one_fold(train_texts, train_labels, train_cats, val_texts, val_labels, test_texts):
    model = MultiTaskClassifier(MODEL_NAME).to(DEVICE)

    train_ds = HateDataset(train_texts, train_labels, train_cats)
    val_ds = HateDataset(val_texts, val_labels)
    test_ds = HateDataset(test_texts)

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE * 2)
    test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE * 2)

    class_counts = np.bincount(train_labels)
    class_weights = torch.tensor(class_counts.sum() / (2.0 * class_counts), dtype=torch.float32).to(DEVICE)
    bin_loss_fn = nn.CrossEntropyLoss(weight=class_weights)
    cat_loss_fn = nn.CrossEntropyLoss(weight=cat_weights, ignore_index=-100)

    # Exclude bias and LayerNorm params from weight decay -- applying decay to
    # these destabilizes BERT fine-tuning, especially on small datasets.
    no_decay = ["bias", "LayerNorm.weight", "LayerNorm.bias"]
    grouped_params = [
        {
            "params": [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
            "weight_decay": 0.01,
        },
        {
            "params": [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
        },
    ]

    optimizer = torch.optim.AdamW(grouped_params, lr=LR)
    total_steps = len(train_dl) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.06 * total_steps), num_training_steps=total_steps)

    best_f1 = -1
    best_val_probs = None
    for epoch in range(EPOCHS):
        model.train()
        for batch in train_dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            labels = batch.pop("labels")
            cat_labels = batch.pop("cat_labels")
            bin_logits, cat_logits = model(**batch)
            loss = bin_loss_fn(bin_logits, labels) + AUX_WEIGHT * cat_loss_fn(cat_logits, cat_labels)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

        # eval this epoch (binary macro-F1 is the metric that matters for Task A)
        model.eval()
        val_probs = []
        with torch.no_grad():
            for batch in val_dl:
                batch = {k: v.to(DEVICE) for k, v in batch.items()}
                batch.pop("labels", None)
                bin_logits, _ = model(**batch)
                probs = torch.softmax(bin_logits, dim=1)[:, 1]
                val_probs.append(probs.cpu().numpy())
        val_probs = np.concatenate(val_probs)
        f1 = f1_score(val_labels, (val_probs > 0.5).astype(int), average="macro")
        print(f"  epoch {epoch + 1}: val macro-F1 @0.5 = {f1:.4f}")
        if f1 > best_f1:
            best_f1 = f1
            best_val_probs = val_probs
            # snapshot test predictions from the best epoch
            model.eval()
            test_probs = []
            with torch.no_grad():
                for batch in test_dl:
                    batch = {k: v.to(DEVICE) for k, v in batch.items()}
                    bin_logits, _ = model(**batch)
                    probs = torch.softmax(bin_logits, dim=1)[:, 1]
                    test_probs.append(probs.cpu().numpy())
            best_test_probs = np.concatenate(test_probs)

    return best_val_probs, best_test_probs


skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
oof_bert = np.zeros(len(X_text))
test_bert_folds = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_text, y)):
    print(f"=== IndicBERT fold {fold + 1}/{N_FOLDS} ===")
    val_probs, test_probs = train_one_fold(
        X_text[tr_idx], y[tr_idx], y_cat[tr_idx], X_text[va_idx], y[va_idx], X_test_text
    )
    oof_bert[va_idx] = val_probs
    test_bert_folds.append(test_probs)

test_bert = np.mean(test_bert_folds, axis=0)
print("IndicBERT OOF macro-F1 @0.5:", f1_score(y, (oof_bert > 0.5).astype(int), average="macro"))


# ---------------------------------------------------------------------------
# 4. Char n-gram TF-IDF + SVM (out-of-fold, calibrated for probabilities)
# ---------------------------------------------------------------------------
oof_svm = np.zeros(len(X_text))
test_svm_folds = []

for fold, (tr_idx, va_idx) in enumerate(skf.split(X_text, y)):
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 5), min_df=2, max_features=50000, sublinear_tf=True)
    Xtr = vec.fit_transform(X_text[tr_idx])
    Xva = vec.transform(X_text[va_idx])
    Xte = vec.transform(X_test_text)

    base_clf = LinearSVC(C=1.0, class_weight="balanced")
    clf = CalibratedClassifierCV(base_clf, cv=3, method="sigmoid")
    clf.fit(Xtr, y[tr_idx])

    oof_svm[va_idx] = clf.predict_proba(Xva)[:, 1]
    test_svm_folds.append(clf.predict_proba(Xte)[:, 1])

test_svm = np.mean(test_svm_folds, axis=0)
print("SVM OOF macro-F1 @0.5:", f1_score(y, (oof_svm > 0.5).astype(int), average="macro"))


# ---------------------------------------------------------------------------
# 5. Blend weight + decision threshold search (optimize macro-F1 on OOF preds)
# ---------------------------------------------------------------------------
best_score, best_w, best_t = -1, None, None
for w in np.arange(0.0, 1.01, 0.05):          # weight on BERT; (1-w) on SVM
    blend = w * oof_bert + (1 - w) * oof_svm
    for t in np.arange(0.30, 0.71, 0.01):
        f1 = f1_score(y, (blend > t).astype(int), average="macro")
        if f1 > best_score:
            best_score, best_w, best_t = f1, w, t

print(f"\nBest blend: w_bert={best_w:.2f}, w_svm={1 - best_w:.2f}, threshold={best_t:.2f}")
print(f"Best OOF macro-F1: {best_score:.4f}")

# ---------------------------------------------------------------------------
# 6. Final predictions on the validation_inputs file
# ---------------------------------------------------------------------------
final_test_probs = best_w * test_bert + (1 - best_w) * test_svm
final_preds = (final_test_probs > best_t).astype(int)

submission = pd.DataFrame({
    "id": test_df["id"],
    "Label": np.where(final_preds == 1, "Hate", "Non-Hate"),
})
out_path = PRED_DIR / "prediction_indicbert_v2.csv"
submission.to_csv(out_path, index=False)
print(f"Saved {out_path}")