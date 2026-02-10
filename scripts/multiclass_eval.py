from __future__ import annotations

import argparse
from glob import glob
from pathlib import Path
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt


def parse_args():
    p = argparse.ArgumentParser(description="HW02Q8: aggregate metrics and create boxplot")
    p.add_argument("--keyword", type=str, required=True, help="keyword used in metrics filenames")
    p.add_argument("--indir", type=str, default="results", help="directory containing metrics CSV files")
    p.add_argument("--outdir", type=str, default="results", help="directory to save the boxplot PDF")
    return p.parse_args()


def main():
    args = parse_args()

    indir = Path(args.indir).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    outdir.mkdir(parents=True, exist_ok=True)

    pattern = str(indir / f"metrics_{args.keyword}_*.csv")
    files = sorted(glob(pattern))

    if not files:
        raise FileNotFoundError(f"No metrics files found: {pattern}")

    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    metrics = [
        ("Accuracy", "train_accuracy", "test_accuracy"),
        ("F1 (macro)", "train_f1_macro", "test_f1_macro"),
        ("Precision (macro)", "train_precision_macro", "test_precision_macro"),
        ("Recall (macro)", "train_recall_macro", "test_recall_macro"),
    ]

    data = []
    labels = []

    for title, tr, te in metrics:
        data.append(df[tr].values)
        labels.append(f"{title}\nTrain")
        data.append(df[te].values)
        labels.append(f"{title}\nTest")

    plt.figure(figsize=(12, 6))
    plt.boxplot(data)
    plt.xticks(range(1, len(labels) + 1), labels, rotation=0)
    plt.ylabel("Score")
    plt.title(f"HW02Q8 Metrics Boxplot (keyword={args.keyword})")
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    outpath = outdir / f"boxplot_{args.keyword}_{ts}.pdf"
    plt.savefig(outpath, bbox_inches="tight")
    plt.close()

    print(f"Found {len(files)} runs")
    print("Saved:", outpath)


if __name__ == "__main__":
    main()
