#!/usr/bin/env python3
"""
Domain-Adaptive Pretraining (DAPT): continue masked-language-model
pretraining on the shared task's own Kanglish text -- BEFORE fine-tuning
for classification.

WHY THIS IS A GENUINELY NEW LEVER (not just more ensembling):
MuRIL and XLM-R were pretrained on huge but generic multilingual corpora.
They likely saw very little authentic Kannada-English code-mixed YouTube-
comment text during that original pretraining. Right now, the ONLY
exposure either model gets to this specific text style is a handful of
fine-tuning epochs on ~5-6K labeled examples -- not much for a model to
adapt its internal representations to a genuinely different register
(code-mixing, transliteration noise, comment-style grammar).

DAPT closes that gap: it continues the model's original MLM objective
(randomly mask ~15% of tokens, predict them) directly on YOUR text --
using ALL available text, labeled AND unlabeled (the validation/test
input files have no labels but their COMMENT TEXT is free to use for
this, since we're not touching the labels). This lets the model adjust
its embeddings and attention patterns to Kanglish specifically, before
it ever sees a single Hate/Non-Hate label. This is a different mechanism
entirely from ensembling -- it can improve the ceiling of EACH individual
model, not just how well you combine models you already have.

After this finishes, use --output_dir as the --model_name in train.py or
train_kfold_ensemble.py, exactly like you'd pass "google/muril-base-cased"
or "xlm-roberta-large" -- it's a normal, loadable checkpoint.

Usage:
    python src/dapt_pretrain.py \
        --model_name xlm-roberta-large \
        --text_csvs data/binary_train.csv data/binary_validation_inputs.csv data/multiclass_train.csv data/multiclass_validation_inputs.csv \
        --output_dir outputs/xlmr_large_dapt \
        --epochs 20

Then fine-tune from the adapted checkpoint:
    python src/train_kfold_ensemble.py --task a \
        --train_csv data/binary_train.csv \
        --input data/binary_validation_inputs.csv \
        --output_dir outputs/task_a_kfold_xlmr_large_dapt \
        --out predictions/task_a_predictions_kfold_xlmrlarge_dapt.csv \
        --n_folds 5 --model_name outputs/xlmr_large_dapt --lr 5e-6 --batch_size 8

Time expectation: pretraining on ~7-8K short comments for 20 epochs is
much cheaper than the classification fine-tunes you've already run --
expect roughly 15-40 minutes depending on model size, since there's no
eval loop and no early stopping to slow things down.
"""
import argparse
import os

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForMaskedLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from utils import read_csv_dicts, find_comment_key, clean_text


class TextOnlyDataset(Dataset):
    """Tokenized text with no labels -- for MLM pretraining."""
    def __init__(self, texts, tokenizer, max_len):
        self.encodings = tokenizer(
            texts, truncation=True, max_length=max_len, padding=False
        )

    def __len__(self):
        return len(self.encodings["input_ids"])

    def __getitem__(self, idx):
        return {k: v[idx] for k, v in self.encodings.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_name", required=True, help="base checkpoint to continue pretraining, e.g. google/muril-base-cased or xlm-roberta-large")
    ap.add_argument("--text_csvs", nargs="+", required=True, help="one or more CSVs with a Comment column -- use ALL train + validation_inputs files across both tasks for maximum in-domain text")
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--epochs", type=float, default=20.0, help="MLM pretraining converges slower than classification fine-tuning; more epochs than usual is normal here")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--max_len", type=int, default=160)
    ap.add_argument("--mlm_probability", type=float, default=0.15)
    ap.add_argument("--fp16", action="store_true", default=torch.cuda.is_available())
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # Pool text from every CSV given -- dedupe since train/validation files
    # across Task A and Task B share many of the same underlying comments.
    all_texts = set()
    for path in args.text_csvs:
        rows = read_csv_dicts(path)
        comment_key = find_comment_key(rows[0].keys())
        for r in rows:
            t = clean_text(r[comment_key])
            if t:
                all_texts.add(t)
    texts = list(all_texts)
    print(f"Pooled {len(texts)} unique comments across {len(args.text_csvs)} files for DAPT.")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name)

    dataset = TextOnlyDataset(texts, tokenizer, args.max_len)
    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer, mlm=True, mlm_probability=args.mlm_probability
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        logging_steps=50,
        save_strategy="no",  # we save the final model ourselves below
        fp16=args.fp16,
        report_to=[],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"\nSaved domain-adapted checkpoint to {args.output_dir}")
    print("Use this path as --model_name in train.py / train_kfold_ensemble.py to fine-tune from it.")


if __name__ == "__main__":
    main()