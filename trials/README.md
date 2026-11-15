# Approaches tried

Scripts for experiments that are not part of the submitted Task A system (MuRIL + XLM-RoBERTa + calibrated TF-IDF SVM, see the top-level README). None of them improved the leaderboard score; they are kept to document what was tried.

Run a script from the project root with the helpers in `src/` on the path, for example:

```bash
PYTHONPATH=src python trials/threshold_tune.py --help
```

| Script | Approach | Outcome |
|---|---|---|
| `blend_task_a.py` | Blends out-of-fold probabilities from three already-trained models: 
| `check_tokenizer_coverage.py` | Quick diagnostic: how well does the IndicBERT v2 tokenizer actually cover 
| `classical_baseline.py` | TF-IDF (word 1-2 + char 3-5) + class-balanced LinearSVC baseline | Strong baseline; later the best single component (5-fold OOF 0.8024) |
| `dapt_pretrain.py` | Domain-adaptive masked-LM pretraining of XLM-R-large on pooled comments | Leaderboard 0.81, worse than 0.82 |
| `ensemble.py` | Blend of one fine-tuned encoder with the SVM on a single split | MuRIL + SVM: leaderboard 0.80 |
| `kfold_multi_ensemble.py` | Combine several k-fold model groups that share identical folds (leak-free) | OOF blend 0.8256; leaderboard 0.82 |
| `logit_adjust_b_singlesplit.py` | Per-class bias search on blended probabilities (Task B experiment) | Inconclusive; not used |
| `predict.py` | Inference with one fine-tuned encoder; can save class probabilities | Used for early submissions (MuRIL alone: leaderboard 0.78) |
| `pseudo_label_svm.py` | Pseudo-labels binary_validation_inputs.csv using the current best blend  
| `regen_svm_cache.py` | Regenerates the char n-gram TF-IDF + SVM out-of-fold probabilities and saves 
| `single_model_predict.py` | Bag the folds of one k-fold model group; no SVM, no blending (test of a simpler system)
| `svm_tuned.py` | Grid search over SVM settings: C, character n-gram range, hand-crafted features | SVM alone 0.8062 vs 0.8024 OOF, but the blend was lower (0.8231 vs 0.8290) |
| `threshold_tune.py` | Search the decision threshold on the blended Hate probability | OOF gain at most 0.0007; leaderboard 0.82 |
| `train_kfold_ensemble.py` | 5-fold training with out-of-fold probabilities and a cross-validated SVM | 5-fold MuRIL + XLM-R-large + SVM: leaderboard 0.82, no gain over the single split |
| `train_task_a.py` | Task A - Binary Hate Speech Detection (Kannada-English code-mixed)


Scores are macro-F1. OOF = out-of-fold (5-fold cross-validation on the training data); leaderboard = development-phase leaderboard.
