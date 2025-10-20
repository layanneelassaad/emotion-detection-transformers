# Multi-Label Emotion Detection with Transformers
BERT vs GPT-2 vs BART on a 5-label emotion task (Anger, Fear, Joy, Sadness, Surprise).

This project fine-tunes encoder-only (BERT), decoder-only (GPT-2), and encoder-decoder (BART) models for multi-label classification. It includes training scripts, evaluation utilities, reproducible plots, and the final report PDF.

## Background: Why these models?
BERT produces bidirectional contextual representations suited to classification. GPT-2 is decoder-only and generative but can be adapted with a classification head. BART combines a bidirectional encoder with an autoregressive decoder, often strong on sequence understanding.

## Data
Place CSVs in `data/`:
- `emotion_train.csv`: columns `text, Anger, Fear, Joy, Sadness, Surprise` with 0/1 labels
- `emotion_test.csv`: same schema for evaluation

## Quickstart
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python src/train_bert.py --train data/emotion_train.csv --dev data/emotion_test.csv --out results_bert
python src/train_bart.py --train data/emotion_train.csv --dev data/emotion_test.csv --out results_bart
python src/train_gpt.py  --train data/emotion_train.csv --dev data/emotion_test.csv --out results_gpt

python src/eval_all.py \
  --truth data/emotion_test.csv \
  --bert results_bert/dev_predictions.csv \
  --bart results_bart/dev_predictions.csv \
  --gpt  results_gpt/dev_predictions.csv \
  --out  results/metrics/summary.txt
```
## Results:

BERT achieved the best overall accuracy and macro F1, with balanced performance on minority classes. GPT-2 trailed BERT but remained competitive. BART was similar to GPT-2 across micro/macro F1. Cross-dataset transfer from larger label taxonomies can underperform without careful label mapping and calibration.


