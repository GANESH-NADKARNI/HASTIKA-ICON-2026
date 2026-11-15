#!/usr/bin/env python3
"""
HASTIKA @ ICON-2026 -- Transformer fine-tuning for Task A (binary hate
detection) and Task B (fine-grained target-category classification).

v2 changes vs the first version (after Task B scored 0.45 macro-F1 on the
leaderboard, below the SVM baseline's 0.615 internal score):
  - Minority-class OVERSAMPLING during training (WeightedRandomSampler),
    on top of the existing class-weighted loss -- belt and suspenders for
    Task B's small classes (Geo-political ~186, Violence ~221 rows).
  - Longer early-stopping patience (3 instead of 2), since small-data
    fine-tunes can have a noisy first couple of epochs before they find
    their footing.
  - Lower default learning rate for Task B (1e-5) -- Task B has ~half the
    data of Task A, and small-data transformer fine-tunes are prone to
    instability at a 2e-5 LR.
  - Saves the internal validation set's texts/labels/predicted-probabilities
    alongside the model, so ensemble.py can reuse the *exact same split*
    when blending with the SVM baseline (no data leakage between the two).

Usage:
    python src/train.py --task a --train_csv data/binary_train.csv --output_dir outputs/task_a
    python src/train.py --task b --train_csv data/multiclass_train.csv --output_dir outputs/task_b
"""
import argparse
import inspect
import json
import os
import random

import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    Trainer,
    TrainingArguments,
    EarlyStoppingCallback,
)

from utils import read_csv_dicts, find_comment_key, clean_text, TASK_A_LABELS, TASK_B_LABELS


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=160):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx], truncation=True, max_length=self.max_len, padding="max_length"
        )
        item = {k: torch.tensor(v) for k, v in enc.items()}
        item["labels"] = torch.tensor(self.labels[idx], dtype=torch.long)
        return item


class WeightedTrainer(Trainer):
    """Class-weighted cross-entropy loss + optional oversampled train loader."""

    def __init__(self, class_weights=None, sample_weights=None, **kwargs):
        super().__init__(**kwargs)
        self.class_weights = class_weights
        self.sample_weights = sample_weights  # per-example weights for WeightedRandomSampler

    def get_train_dataloader(self):
        if self.sample_weights is None:
            return super().get_train_dataloader()
        sampler = WeightedRandomSampler(
            weights=self.sample_weights, num_samples=len(self.sample_weights), replacement=True
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            drop_last=self.args.dataloader_drop_last,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        weight = self.class_weights.to(logits.device) if self.class_weights is not None else None
        loss_fct = torch.nn.CrossEntropyLoss(weight=weight)
        loss = loss_fct(logits.view(-1, logits.shape[-1]), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "macro_f1": f1_score(labels, preds, average="macro"),
        "accuracy": accuracy_score(labels, preds),
    }


def build_training_args(args, train_ds_len):
    """Build TrainingArguments dynamically so this works across different
    transformers versions (warmup_ratio vs warmup_steps, eval_strategy vs
    evaluation_strategy)."""
    ta_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())

    ta_kwargs = dict(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        learning_rate=args.lr,
        weight_decay=0.01,
        max_grad_norm=args.max_grad_norm,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        logging_steps=50,
        fp16=args.fp16,
        report_to=[],
        seed=args.seed,
    )

    if "warmup_ratio" in ta_params:
        ta_kwargs["warmup_ratio"] = 0.1
    elif "warmup_steps" in ta_params:
        steps_per_epoch = max(1, train_ds_len // args.batch_size)
        ta_kwargs["warmup_steps"] = int(0.1 * steps_per_epoch * args.epochs)

    if "eval_strategy" in ta_params:
        ta_kwargs["eval_strategy"] = "epoch"
    elif "evaluation_strategy" in ta_params:
        ta_kwargs["evaluation_strategy"] = "epoch"

    if "save_strategy" in ta_params:
        ta_kwargs["save_strategy"] = "epoch"

    return TrainingArguments(**ta_kwargs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a", "b"], required=True)
    ap.add_argument("--train_csv", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--model_name", default="google/muril-base-cased")
    ap.add_argument("--epochs", type=float, default=None, help="default: 6 for task a, 10 for task b")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None, help="default: 2e-5 for task a, 1e-5 for task b")
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--val_frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--oversample", type=int, default=1, help="1=on (recommended for task b), 0=off")
    ap.add_argument("--max_grad_norm", type=float, default=1.0, help="gradient clipping threshold")
    ap.add_argument("--fp16", action="store_true", default=torch.cuda.is_available())
    args = ap.parse_args()

    # task-specific defaults tuned after the first Task B run underperformed
    if args.epochs is None:
        args.epochs = 6.0 if args.task == "a" else 10.0
    if args.lr is None:
        args.lr = 2e-5 if args.task == "a" else 1e-5

    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    labels_list = TASK_A_LABELS if args.task == "a" else TASK_B_LABELS
    label2id = {l: i for i, l in enumerate(labels_list)}
    id2label = {i: l for l, i in label2id.items()}
    label_col = "Label" if args.task == "a" else "Hate Category"

    rows = read_csv_dicts(args.train_csv)
    comment_key = find_comment_key(rows[0].keys())
    label_key = next(k for k in rows[0].keys() if k.strip().lower() == label_col.lower())

    texts = [clean_text(r[comment_key]) for r in rows]
    raw_labels = [r[label_key].strip() for r in rows]
    y = [label2id[l] for l in raw_labels]

    X_train, X_val, y_train, y_val = train_test_split(
        texts, y, test_size=args.val_frac, random_state=args.seed, stratify=y
    )
    print(f"Train: {len(X_train)}  Internal-val: {len(X_val)}  Classes: {labels_list}")
    print(f"LR={args.lr}  epochs={args.epochs}  oversample={'on' if args.oversample else 'off'}  patience={args.patience}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, num_labels=len(labels_list), id2label=id2label, label2id=label2id
    )

    train_ds = TextDataset(X_train, y_train, tokenizer, args.max_len)
    val_ds = TextDataset(X_val, y_val, tokenizer, args.max_len)

    class_weights_arr = compute_class_weight(
        class_weight="balanced", classes=np.arange(len(labels_list)), y=y_train
    )

    # IMPORTANT: don't stack oversampling and loss-weighting at full strength --
    # both correct for the same imbalance, and multiplying their effects caused
    # a total mode collapse on Task B (model learned to always predict the
    # rarest/most-weighted class). Pick one primary mechanism:
    #   oversample=1 (default) -> the sampler handles rebalancing; loss weights
    #                             are softened to sqrt() so it's a mild nudge,
    #                             not a second full-strength correction.
    #   oversample=0           -> full-strength class-weighted loss only,
    #                             natural (unweighted) sampling.
    if args.oversample:
        loss_weights_arr = np.sqrt(class_weights_arr)
        sample_weights = torch.tensor(
            [class_weights_arr[label] for label in y_train], dtype=torch.double
        )
    else:
        loss_weights_arr = class_weights_arr
        sample_weights = None

    class_weights = torch.tensor(loss_weights_arr, dtype=torch.float)
    print("Class weights used in loss:", dict(zip(labels_list, class_weights.tolist())))
    if args.oversample:
        print("Sampler weights (raw, for oversampling):", dict(zip(labels_list, class_weights_arr.tolist())))

    training_args = build_training_args(args, len(train_ds))

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

    # final internal-val metrics + per-class report
    pred_output = trainer.predict(val_ds)
    val_probs = torch.softmax(torch.tensor(pred_output.predictions), dim=-1).numpy()
    val_preds = val_probs.argmax(axis=-1)
    print("Final internal-validation metrics:", pred_output.metrics)
    print(classification_report(y_val, val_preds, target_names=labels_list, digits=3))

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    # Save the internal val split + transformer's predicted probabilities,
    # so ensemble.py can blend with the SVM on the exact same held-out rows.
    np.savez(
        os.path.join(args.output_dir, "internal_val_cache.npz"),
        val_probs=val_probs,
        y_val=np.array(y_val),
    )
    with open(os.path.join(args.output_dir, "internal_val_texts.json"), "w", encoding="utf-8") as f:
        json.dump(X_val, f, ensure_ascii=False)

    print(f"Saved best model + tokenizer + internal-val cache to {args.output_dir}")


if __name__ == "__main__":
    main()