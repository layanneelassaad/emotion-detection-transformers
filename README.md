# Multi-Label Emotion Detection with Transformers
> This repository implements and compares transformer architectures for multi-label emotion classification over five emotions, Anger, Fear, Joy, Sadness, Surprise, and evaluates generalization from a model trained on GoEmotions to a simplified 5-label taxonomy. The codebase includes training, evaluation, threshold optimization, and analysis utilities.

## Background: 
**Task:**
Multi-label emotion detection assigns one or more emotions to each text. Compared to single-label sentiment analysis, the space is nuanced and overlapping (e.g., Surprise vs. Fear), and minority classes are common—both factors that stress model calibration and thresholding.

We evaluate three transformer families under a consistent fine-tuning protocol:
- Encoder-only: `bert-base-uncased` (≈109M params) is well-suited for classification.
- Decoder-only: `gpt2` (≈124M) is typically generative, adapted here via a classification head.
- Encoder–decoder: `facebook/bart-base` (≈139M) is bidirectional encoder plus autoregressive decoder.


## Data
- `emotion_train.csv`: SemEval-style 5-label set: Anger, Fear, Joy, Sadness, Surprise. Training set ≈2.5k samples (train/val split) and a small held-out test set that we manually labeled by majority vote. We provide corpus statistics (lengths, label frequencies, frequent co-occurrence pair
- `emotion_test.csv`: GoEmotions: 27 fine-grained labels over ≈58k Reddit comments. We map fine-grained categories to the five coarse labels for transfer experiments (e.g., several fine “joy-like” labels map to Joy).

## Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

#BERT
python src/train_bert.py --train data/emotion_train.csv --dev data/emotion_test.csv --out results_bert
python src/eval/evaluate.py \
  --model runs/bert_base/final_model \
  --data  data/emotion_test.csv \
  --out   results/bert_eval

#BART
python src/train_bart.py --train data/emotion_train.csv --dev data/emotion_test.csv --out results_bart
python src/eval/evaluate.py \
  --model runs/bart_base/final_model \
  --data  data/emotion_test.csv \
  --out   results/bart_eval

#GPT
python src/train_gpt.py  --train data/emotion_train.csv --dev data/emotion_test.csv --out results_gpt
python src/eval/evaluate.py \
  --model runs/gpt2_base/final_model \
  --data  data/emotion_test.csv \
  --out   results/gpt2_eval

#Evaluate all:
python src/eval_all.py \
  --truth data/emotion_test.csv \
  --bert results_bert/dev_predictions.csv \
  --bart results_bart/dev_predictions.csv \
  --gpt  results_gpt/dev_predictions.csv \
  --out  results/metrics/summary.txt
```
## Results:


| Model       | Accuracy  | Micro F1  | Macro F1  | Most-Predicted Emotion |
| ----------- | --------- | --------- | --------- | ---------------------- |
| **BERT**    | **0.397** | **0.613** | **0.608** | Fear                   |
| **BART**    | 0.345     | 0.573     | 0.562     | Fear                   |
| **GPT-2**   | 0.362     | 0.572     | 0.545     | Fear                   |
| GoEmotions* | 0.070     | 0.257     | 0.254     | Joy                    |

> GoEmotions row corresponds to a model fine-tuned on 27 labels and mapped to our five-label schema; its poor performance is largely due to label mismatch and domain shift.

**Class-level findings:**
- Joy is consistently strong; for BERT, class accuracy reaches ~0.897, with balanced recall/F1.

- Fear poses challenges for all models (lower recall/F1), yet is also the most predicted class across BERT/BART/GPT.

- Surprise is fragile and often confused with Fear (contextual overlap)

**Error patterns:**
- Ambiguous disbelief phrases (e.g., “I can’t believe this happened!”) were often scored as Fear rather than Surprise. Fear tends to be over-predicted, with BERT showing high recall but low precision on this class.
- Very short, abrupt expressions of Anger were missed or labeled Neutral. Anger is under-predicted, especially for short sentences that lack dense lexical cues.
- Surprise shows low precision/recall and confuses with Fear/Joy in celebratory or unexpected contexts. Excited Joy sometimes drifted into Surprise.


BERT remains a strong baseline for multi-label emotion detection thanks to its bidirectional encoder and classification-friendly pretraining.

BART is competitive but slightly behind in our setting.

GPT-2 is workable with a classification head, though less robust on minority classes.

Cross-dataset transfer without careful label mapping and domain alignment performs poorly (GoEmotions-to-SemEval mapping)

**Limitations and future work:**
Hyperparameter tuning was intentionally constrained for comparability. More extensive sweeps, class-balanced sampling, and calibration could further lift minority-class performance. For transfer, retraining on an aligned label schema or using multi-task learning would likely improve results.
