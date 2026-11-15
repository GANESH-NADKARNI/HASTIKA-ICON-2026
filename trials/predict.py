#!/usr/bin/env python3
"""
Generate a HASTIKA-format predictions.csv from a fine-tuned model.

Usage:
    python src/predict.py --task a \
        --model_dir outputs/task_a \
        --input data/binary_validation_inputs.csv \
        --out predictions/task_a_predictions.csv

    python src/predict.py --task b \
        --model_dir outputs/task_b \
        --input data/multiclass_validation_inputs.csv \
        --out predictions/task_b_predictions.csv

Output matches the sample files exactly: two columns, id,label.

Pass --probs_out to also dump the raw per-class softmax probabilities to a
.npy file (aligned row-for-row with the input) -- used by ensemble.py to
blend with the SVM baseline.
"""
import argparse
import csv

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from utils import read_csv_dicts, find_id_key, find_comment_key, clean_text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a", "b"], required=True)
    ap.add_argument("--model_dir", required=True)
    ap.add_argument("--input", required=True, help="CSV with id + Comment columns")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--probs_out", default=None, help="optional .npy path to save class probabilities")
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_len", type=int, default=160)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_dir).to(device)
    model.eval()
    id2label = model.config.id2label

    rows = read_csv_dicts(args.input)
    id_key = find_id_key(rows[0].keys())
    comment_key = find_comment_key(rows[0].keys())
    if id_key is None or comment_key is None:
        raise SystemExit("Input CSV needs an 'id' column and a 'Comment' column.")

    ids = [r[id_key] for r in rows]
    texts = [clean_text(r[comment_key]) for r in rows]

    preds = []
    all_probs = []
    with torch.no_grad():
        for i in range(0, len(texts), args.batch_size):
            batch = texts[i:i + args.batch_size]
            enc = tokenizer(batch, truncation=True, max_length=args.max_len,
                             padding=True, return_tensors="pt").to(device)
            logits = model(**enc).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()
            all_probs.append(probs)
            batch_preds = probs.argmax(axis=-1).tolist()
            preds.extend(id2label[p] for p in batch_preds)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["id", "label"])
        for i, p in zip(ids, preds):
            w.writerow([i, p])
    print(f"Wrote {args.out} with {len(preds)} predictions.")

    if args.probs_out:
        np.save(args.probs_out, np.concatenate(all_probs, axis=0))
        print(f"Wrote class probabilities to {args.probs_out}")


if __name__ == "__main__":
    main()