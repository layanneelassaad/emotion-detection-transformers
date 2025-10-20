import argparse, pandas as pd, itertools, matplotlib.pyplot as plt
from src.utils import EMOTIONS, preprocess_text

def compute_stats(df):
    df = df.copy()
    df["text"] = df["text"].apply(preprocess_text)
    df["input_length"] = df["text"].apply(lambda x: len(str(x).split()))
    label_counts = df[EMOTIONS].sum().to_dict()
    pairs = list(itertools.combinations(EMOTIONS, 2))
    pair_counts = {}
    for a,b in pairs:
        pair_counts[(a,b)] = int(((df[a]==1) & (df[b]==1)).sum())
    return df, label_counts, pair_counts

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--plots_dir", required=True)
    args = ap.parse_args()

    df = pd.read_csv(args.csv)
    df, label_counts, pair_counts = compute_stats(df)

    s = df["input_length"]
    s.describe().to_frame().to_csv(f"{args.plots_dir}/input_length_stats.csv")

    plt.figure()
    s.plot(kind="hist", bins=40)
    plt.xlabel("Input length (tokens)")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{args.plots_dir}/input_length_hist.png", dpi=200)

    lc = pd.DataFrame(list(label_counts.items()), columns=["Label","Count"]).sort_values("Count", ascending=False)
    plt.figure()
    plt.bar(lc["Label"], lc["Count"])
    plt.xlabel("Label")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(f"{args.plots_dir}/label_counts.png", dpi=200)

    pc = pd.DataFrame(list(pair_counts.items()), columns=["Pair","Count"]).sort_values("Count", ascending=False).head(10)
    plt.figure()
    plt.bar([f"{a}+{b}" for a,b in pc["Pair"]], pc["Count"])
    plt.xlabel("Pair")
    plt.ylabel("Count")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(f"{args.plots_dir}/top_pair_counts.png", dpi=200)

if __name__ == "__main__":
    main()
