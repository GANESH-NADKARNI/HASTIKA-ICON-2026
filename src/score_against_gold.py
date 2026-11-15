#!/usr/bin/env python3
"""
Now that validation gold labels are released, score any prediction CSV
you've already generated against REAL ground truth -- finally answering,
with actual data instead of internal estimates, which of this session's
many experiments genuinely generalizes best.

Usage:
    python src/score_against_gold.py --task a \
        --gold_csv data/binary_validation_gold.csv \
        --predictions predictions/task_a_predictions_kfold_multi.csv predictions/task_a_predictions_ensemble.csv ...

    python src/score_against_gold.py --task b \
        --gold_csv data/multiclass_validation_gold.csv \
        --predictions predictions/task_b_predictions_final.csv ...

Pass as many --predictions files as you want to compare at once; results
are printed sorted best-to-worst by macro-F1.
"""
import argparse

from sklearn.metrics import f1_score, accuracy_score, classification_report

from utils import read_csv_dicts, find_id_key, TASK_A_LABELS, TASK_B_LABELS


def load_labels(path, labels_list, label_col_candidates):
    rows = read_csv_dicts(path)
    id_key = find_id_key(rows[0].keys())
    label_key = None
    for k in rows[0].keys():
        if k.strip().lower() in label_col_candidates:
            label_key = k
            break
    if label_key is None:
        raise SystemExit(f"Could not find a label column in {path}. Columns found: {list(rows[0].keys())}")
    id_to_label = {r[id_key]: r[label_key].strip() for r in rows}
    return id_to_label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", choices=["a", "b"], required=True)
    ap.add_argument("--gold_csv", required=True, help="the released validation file WITH gold labels")
    ap.add_argument("--predictions", nargs="+", required=True, help="one or more predictions/*.csv files to score")
    args = ap.parse_args()

    labels_list = TASK_A_LABELS if args.task == "a" else TASK_B_LABELS
    label_candidates = ("label", "hate category", "category") if args.task == "b" else ("label",)
    gold = load_labels(args.gold_csv, labels_list, label_candidates)
    print(f"Loaded {len(gold)} gold-labeled rows from {args.gold_csv}\n")

    results = []
    for pred_path in args.predictions:
        pred_rows = read_csv_dicts(pred_path)
        pred_id_key = find_id_key(pred_rows[0].keys())
        pred_label_key = next(k for k in pred_rows[0].keys() if k.strip().lower() == "label")

        y_true, y_pred = [], []
        missing = 0
        for r in pred_rows:
            rid = r[pred_id_key]
            if rid not in gold:
                missing += 1
                continue
            y_true.append(gold[rid])
            y_pred.append(r[pred_label_key].strip())

        if missing:
            print(f"  WARNING: {pred_path} has {missing} ids not found in gold file (skipped)")

        macro_f1 = f1_score(y_true, y_pred, average="macro", labels=labels_list)
        acc = accuracy_score(y_true, y_pred)
        results.append((pred_path, macro_f1, acc, y_true, y_pred))

    results.sort(key=lambda r: r[1], reverse=True)

    print(f"{'Rank':<5}{'macro-F1':<10}{'accuracy':<10}File")
    print("-" * 70)
    for i, (path, f1, acc, _, _) in enumerate(results, 1):
        print(f"{i:<5}{f1:<10.4f}{acc:<10.4f}{path}")

    best_path, best_f1, best_acc, best_y_true, best_y_pred = results[0]
    print(f"\n=== Best: {best_path} (macro-F1 {best_f1:.4f}) ===")
    print(classification_report(best_y_true, best_y_pred, labels=labels_list, digits=3))


if __name__ == "__main__":
    main()