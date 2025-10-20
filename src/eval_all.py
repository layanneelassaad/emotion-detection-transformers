import argparse, pandas as pd
from sklearn.metrics import classification_report
from src.utils import EMOTIONS

def load_csv(p):
    return pd.read_csv(p)

def rep(true_df, pred_df):
    y_true = true_df[EMOTIONS].values
    y_pred = pred_df[EMOTIONS].values
    return classification_report(y_true, y_pred, target_names=EMOTIONS, zero_division=0)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--bert", required=True)
    ap.add_argument("--bart", required=True)
    ap.add_argument("--gpt", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    t = load_csv(args.truth)
    rb = load_csv(args.bert)
    rba = load_csv(args.bart)
    rg = load_csv(args.gpt)

    txt = []
    txt.append("BERT\n")
    txt.append(rep(t, rb))
    txt.append("\nBART\n")
    txt.append(rep(t, rba))
    txt.append("\nGPT-2\n")
    txt.append(rep(t, rg))

    with open(args.out, "w") as f:
        f.write("\n".join(txt))

if __name__ == "__main__":
    main()
