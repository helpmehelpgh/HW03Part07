from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from mchnpkg.deepl import SimpleNN, ClassTrainer


DROP_COLS = [
    "Flow ID",
    "Source IP",
    "Source Port",
    "Destination IP",
    "Destination Port",
    "Protocol",
    "Timestamp",
]

# Expected 4 labels (per homework)
EXPECTED_CLASSES = [
    "Android_Adware",
    "Android_Scareware",
    "Android_SMS_Malware",
    "Benign",
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="HW02Q8 multiclass malware classification")
    p.add_argument("--data", type=str, default="data/Android_Malware.csv", help="Path to Android_Malware.csv")
    p.add_argument("--keyword", type=str, default="hw02", help="Unique keyword to tag outputs")
    p.add_argument("--eta", type=float, default=1e-3, help="Learning rate")
    p.add_argument("--epoch", type=int, default=50, help="Number of epochs")
    p.add_argument("--seed", type=int, default=0, help="Random seed")
    p.add_argument("--test_size", type=float, default=0.2, help="Test fraction (default 0.2)")
    p.add_argument("--standardize", action="store_true", help="Standardize features")
    p.add_argument("--outdir", type=str, default="results", help="Output directory for metrics CSV")
    return p.parse_args()


def find_label_column(df: pd.DataFrame) -> str:
    # Most datasets use "Label"
    if "Label" in df.columns:
        return "Label"
    # fallback: common variants
    for c in ["label", "Class", "class", "Category", "category", "type"]:
        if c in df.columns:
            return c
    # last column fallback
    return df.columns[-1]


def load_prepare_split(
    csv_path: str,
    seed: int,
    test_size: float,
    standardize: bool,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, int]:
    df = pd.read_csv(csv_path)

    label_col = find_label_column(df)

    # drop unhelpful columns if present
    drop_existing = [c for c in DROP_COLS if c in df.columns]
    df = df.drop(columns=drop_existing, errors="ignore")

    # separate X/y
    y_raw = df[label_col].astype(str)
    X = df.drop(columns=[label_col])

    # numeric features only
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0.0)

    # encode labels into 0..m-1
    le = LabelEncoder()
    y = le.fit_transform(y_raw.values)
    classes = list(le.classes_)

    # print a quick check so you see the labels in the dataset
    print("Detected classes:", classes)

    # number of classes
    m = len(classes)

    # train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X.values.astype(np.float32),
        y.astype(np.int64),
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )

    # optional standardize
    if standardize:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train).astype(np.float32)
        X_test = scaler.transform(X_test).astype(np.float32)

    # torch tensors
    X_train_t = torch.from_numpy(X_train)
    X_test_t = torch.from_numpy(X_test)
    y_train_t = torch.from_numpy(y_train)
    y_test_t = torch.from_numpy(y_test)

    return X_train_t, y_train_t, X_test_t, y_test_t, m


def ensure_dir(p: str) -> Path:
    out = Path(p).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def save_metrics_csv(
    outdir: Path,
    keyword: str,
    timestamp: str,
    train_metrics: Dict[str, Any],
    test_metrics: Dict[str, Any],
    eta: float,
    epoch: int,
    seed: int,
    standardize: bool,
) -> Path:
    row = {
        "keyword": keyword,
        "timestamp": timestamp,
        "eta": eta,
        "epoch": epoch,
        "seed": seed,
        "standardize": standardize,
        "train_accuracy": train_metrics["accuracy"],
        "train_precision_macro": train_metrics["precision_macro"],
        "train_recall_macro": train_metrics["recall_macro"],
        "train_f1_macro": train_metrics["f1_macro"],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision_macro": test_metrics["precision_macro"],
        "test_recall_macro": test_metrics["recall_macro"],
        "test_f1_macro": test_metrics["f1_macro"],
    }

    out_path = outdir / f"metrics_{keyword}_{timestamp}.csv"
    pd.DataFrame([row]).to_csv(out_path, index=False)
    return out_path


def compute_train_metrics(trainer: ClassTrainer) -> Dict[str, Any]:
    # evaluate on training set
    y_true = trainer.Y_train.detach().cpu()
    y_pred = trainer.predict(trainer.X_train.detach().cpu())
    # reuse trainer.test metric logic by calling test on train tensors
    return trainer.test(trainer.X_train.detach().cpu(), y_true)


def main() -> None:
    args = parse_args()

    # reproducibility
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    X_train, y_train, X_test, y_test, m = load_prepare_split(
        args.data, seed=args.seed, test_size=args.test_size, standardize=args.standardize
    )

    # create trainer
    trainer = ClassTrainer(
        X_train=X_train,
        Y_train=y_train,
        eta=args.eta,
        epoch=args.epoch,
        loss=nn.CrossEntropyLoss(),
        optimizer=None,  # uses default Adam in __post_init__
        model_cls=SimpleNN,
        num_classes=m,
        device=None,  # auto
    )

    # train
    trainer.train()

    # metrics
    train_metrics = compute_train_metrics(trainer)
    test_metrics = trainer.test(X_test, y_test)

    # save results
    outdir = ensure_dir(args.outdir)
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    out_csv = save_metrics_csv(
        outdir,
        args.keyword,
        timestamp,
        train_metrics,
        test_metrics,
        eta=args.eta,
        epoch=args.epoch,
        seed=args.seed,
        standardize=args.standardize,
    )

    print("Saved:", out_csv)
    print("Train:", {k: v for k, v in train_metrics.items() if k != "confusion_matrix"})
    print("Test :", {k: v for k, v in test_metrics.items() if k != "confusion_matrix"})


if __name__ == "__main__":
    main()
