# HASTIKA @ ICON-2026, Task A: Kannada-English Hate Speech Detection

Binary hate speech detection (Hate / Non-Hate) for Kannada-English code-mixed YouTube comments.
The system blends the class probabilities of fine-tuned **MuRIL**, **XLM-RoBERTa** and a calibrated
**TF-IDF + linear SVM**, with blend weights tuned on held-out data.

## Results (macro-F1)

| Phase | Macro-F1 | Accuracy |
|---|---|---|
| Development leaderboard | 0.82 | n/a |
| Validation gold labels | 0.8162 | 0.8164 |
| Official test set | 0.7963 | 0.7965 |

Held-out macro-F1 of each component (774-row split) and the blend weight selected for the final run:

| Component | Held-out F1 | Weight |
|---|---|---|
| MuRIL (`google/muril-base-cased`) | 0.762 | 0.20 |
| XLM-RoBERTa (`xlm-roberta-base`) | 0.791 | 0.00 |
| TF-IDF SVM | 0.807 | 0.80 |

## Method

1. Light text cleaning (HTML unescape, tag strip, URL token, whitespace, mojibake repair).
2. Stratified 88/12 split of the training file.
3. Fine-tune MuRIL and XLM-RoBERTa (weighted cross-entropy, early stopping on held-out macro-F1).
4. Train a class-balanced linear SVM on word (1-2) and character (3-5) TF-IDF, then calibrate it (sigmoid, 3-fold).
5. Search blend weights on the held-out split (grid, step 0.05) and predict with the arg max of the blended probabilities.

## Repository layout

```
src/                        the scripts needed to reproduce the submitted system
  utils.py                  shared helpers (text cleaning, label lists, CSV reading)
  train.py                  fine-tune one encoder on a stratified 88/12 split
  multi_ensemble.py         blend encoders + SVM, search weights, write predictions
  score_against_gold.py     score a predictions file against gold labels
trials/                     scripts for the approaches tried (see trials/README.md)
requirements.txt
data/                       NOT included (see Data)
outputs/                    fine-tuned checkpoints (created by training, not in git)
predictions/                prediction CSVs (created by the scripts, not in git)
```

## Data

The dataset is not included and is subject to the organizers' terms. Download the HASTIKA files from the organizers' repository and put them in `data/`:

- `binary_train.csv` (columns: `id`, `Comment`, `Label`)
- `hastika_binary_validation.csv` (with gold labels)
- `hastika_binary_test.csv` (labels withheld)

Please cite the dataset paper:

```bibtex
@article{kavatagi2025hastika,
  title={HASTIKA: hate speech and target identification in Kannada-English code-mixed text},
  author={Kavatagi, Sanjana and Rachh, Rashmi},
  journal={Language Resources and Evaluation},
  volume={59}, number={3}, pages={2811--2856}, year={2025}, publisher={Springer}
}
```

## Setup

```bash
python -m venv venv
source venv/bin/activate          # Windows Git Bash: source venv/Scripts/activate
pip install -r requirements.txt
```

A GPU with 8 GB of memory is enough; each encoder trains in about 8 to 10 minutes.

## Reproduce the Task A system

1. **Fine-tune the two encoders**

```bash
python src/train.py --task a --train_csv data/binary_train.csv \
    --output_dir outputs/task_a --model_name google/muril-base-cased

python src/train.py --task a --train_csv data/binary_train.csv \
    --output_dir outputs/task_a_xlmr --model_name xlm-roberta-base
```

2. **Blend with the SVM and write test predictions**

```bash
python src/multi_ensemble.py --task a --train_csv data/binary_train.csv \
    --model_dirs outputs/task_a outputs/task_a_xlmr \
    --input data/hastika_binary_test.csv --out predictions/task_a_submission.csv
```

The script fits the calibrated SVM, searches the blend weights on the held-out split, prints each
component's score and the weights, and writes `id,label` predictions.

3. **Check against the validation gold labels**

```bash
python src/multi_ensemble.py --task a --train_csv data/binary_train.csv \
    --model_dirs outputs/task_a outputs/task_a_xlmr \
    --input data/hastika_binary_validation.csv --out predictions/task_a_validation.csv

python src/score_against_gold.py --task a \
    --gold_csv data/hastika_binary_validation.csv \
    --predictions predictions/task_a_validation.csv
```

## Approaches tried

Experiments that are not part of the submitted system (k-fold bagging, domain-adaptive pretraining,
threshold tuning, SVM tuning and others) are in [`trials/`](trials/README.md) with a short note on each outcome.

Warning: `trials/train_task_a.py` adds 644 rows from `multiclass_train.csv` that are Task A validation (324)
and test (320) comments, so it trains on evaluation data. Its scores are not valid and are not used in the
results above.

## Notes and limitations

- Validation and test sets have 806 rows each, so differences of 0.01 to 0.02 macro-F1 are within noise.
- The text cleaner gained a mojibake repair after the encoders were trained, so cached validation text from
  the earlier runs no longer matches the live pipeline. The SVM used to select the final blend weights
  appears to have partly seen the validation rows, which probably made its weight too high.
- Fix random seeds (42 in the final scripts) and keep preprocessing and data files unchanged between
  training and inference.
- The models are for research. They should assist human moderators, not replace them.

## Models on the Hugging Face Hub

| Model | Link |
|---|---|
| MuRIL fine-tuned for Task A | https://huggingface.co/Ganesh-Nadkarni/hastika-task-a-muril |
| XLM-RoBERTa-base fine-tuned for Task A | https://huggingface.co/Ganesh-Nadkarni/hastika-task-a-xlmr-base |

```python
from transformers import pipeline

clf = pipeline("text-classification", model="Ganesh-Nadkarni/hastika-task-a-muril")
print(clf("your Kanglish comment here"))
```

These are single encoders. The scores reported above are for the blend with the TF-IDF SVM, which is not on the Hub,
so each encoder alone is weaker.