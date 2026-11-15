#!/usr/bin/env bash
# Trains both tasks, predicts, then blends each with a calibrated SVM
# (ensemble.py auto-picks the blend weight per task using the internal
# validation split saved by train.py).
set -e

# ---- Task A -- binary hate detection ----
python src/train.py --task a \
  --train_csv data/binary_train.csv \
  --output_dir outputs/task_a

python src/predict.py --task a \
  --model_dir outputs/task_a \
  --input data/binary_validation_inputs.csv \
  --out predictions/task_a_predictions.csv

python src/ensemble.py --task a \
  --train_csv data/binary_train.csv \
  --model_dir outputs/task_a \
  --input data/binary_validation_inputs.csv \
  --out predictions/task_a_predictions_ensemble.csv

# ---- Task B -- fine-grained target category ----
python src/train.py --task b \
  --train_csv data/multiclass_train.csv \
  --output_dir outputs/task_b

python src/predict.py --task b \
  --model_dir outputs/task_b \
  --input data/multiclass_validation_inputs.csv \
  --out predictions/task_b_predictions.csv

python src/ensemble.py --task b \
  --train_csv data/multiclass_train.csv \
  --model_dir outputs/task_b \
  --input data/multiclass_validation_inputs.csv \
  --out predictions/task_b_predictions_ensemble.csv

echo "Done. Compare predictions/task_*_predictions.csv (transformer-only) vs"
echo "predictions/task_*_predictions_ensemble.csv (blended) -- submit whichever"
echo "scored better internally (ensemble.py prints both scores when it runs)."