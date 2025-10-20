import argparse, os, numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from transformers import GPT2Tokenizer, GPT2ForSequenceClassification, Trainer, TrainingArguments
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from src.utils import EMOTIONS, preprocess_text
from src.data import EmotionDataset

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = (torch.sigmoid(torch.tensor(logits)) > 0.5).int().numpy()
    labels = np.array(labels)
    f1 = f1_score(labels, preds, average="macro", zero_division=0)
    p = precision_score(labels, preds, average="macro", zero_division=0)
    r = recall_score(labels, preds, average="macro", zero_division=0)
    return {"f1": f1, "precision": p, "recall": r}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--bsz", type=int, default=8)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    tr = pd.read_csv(args.train)
    dv = pd.read_csv(args.dev)
    tr["text"] = tr["text"].apply(preprocess_text)
    dv["text"] = dv["text"].apply(preprocess_text)
    y_tr = tr[EMOTIONS].values
    y_dv = dv[EMOTIONS].values

    tok = GPT2Tokenizer.from_pretrained("gpt2")
    tok.pad_token = tok.eos_token
    model = GPT2ForSequenceClassification.from_pretrained("gpt2", num_labels=len(EMOTIONS))
    model.config.pad_token_id = tok.pad_token_id
    model.config.problem_type = "multi_label_classification"

    tr_texts, va_texts, tr_labels, va_labels = train_test_split(tr["text"].tolist(), y_tr, test_size=0.1, random_state=42)
    ds_tr = EmotionDataset(tr_texts, tr_labels, tok)
    ds_va = EmotionDataset(va_texts, va_labels, tok)
    ds_dv = EmotionDataset(dv["text"].tolist(), y_dv, tok)

    hfa = TrainingArguments(
        output_dir=args.out,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=args.lr,
        per_device_train_batch_size=args.bsz,
        per_device_eval_batch_size=args.bsz,
        num_train_epochs=args.epochs,
        weight_decay=0.01,
        logging_dir=f"{args.out}/logs",
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False
    )
    trainer = Trainer(model=model, args=hfa, train_dataset=ds_tr, eval_dataset=ds_va, tokenizer=tok, compute_metrics=compute_metrics)
    trainer.train()

    model.save_pretrained(f"{args.out}/final_model")
    tok.save_pretrained(f"{args.out}/final_model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()
    dl = DataLoader(ds_dv, batch_size=args.bsz)
    probs = []
    with torch.no_grad():
        for batch in dl:
            batch = {k: v.to(device) for k, v in batch.items()}
            logits = model(**batch).logits
            p = torch.sigmoid(logits).cpu().numpy()
            probs.append(p)
    P = np.vstack(probs)
    pred = (P > 0.5).astype(int)

    prob_df = pd.DataFrame(P, columns=EMOTIONS)
    prob_df.insert(0, "text", dv["text"].tolist())
    prob_df.to_csv(f"{args.out}/dev_probabilities.csv", index=False)

    out_df = pd.DataFrame(pred, columns=EMOTIONS)
    out_df.insert(0, "text", dv["text"].tolist())
    out_df.to_csv(f"{args.out}/dev_predictions.csv", index=False)

if __name__ == "__main__":
    main()
