import argparse, numpy as np, pandas as pd, json
from sklearn.metrics import f1_score
from src.utils import EMOTIONS

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--truth", required=True)
    ap.add_argument("--probs", required=True)
    ap.add_argument("--out_pred", required=True)
    ap.add_argument("--out_thresh", required=True)
    args = ap.parse_args()

    truth = pd.read_csv(args.truth)
    probs = pd.read_csv(args.probs)

    Y = truth[EMOTIONS].values
    P = probs[EMOTIONS].values

    best = {}
    for i, e in enumerate(EMOTIONS):
        bt = 0.5
        bf = -1.0
        for t in np.arange(0.1, 0.91, 0.01):
            pred = (P[:, i] > t).astype(int)
            f = f1_score(Y[:, i], pred, zero_division=0)
            if f > bf:
                bf = f
                bt = float(t)
        best[e] = bt

    B = np.column_stack([(P[:, i] > best[EMOTIONS[i]]).astype(int) for i in range(len(EMOTIONS))])
    out_df = pd.DataFrame(B, columns=EMOTIONS)
    out_df.insert(0, "text", probs["text"].tolist())
    out_df.to_csv(args.out_pred, index=False)

    with open(args.out_thresh, "w") as f:
        json.dump(best, f, indent=2)

if __name__ == "__main__":
    main()
