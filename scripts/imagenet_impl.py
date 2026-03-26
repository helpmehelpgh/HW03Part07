from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_from_disk
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, Subset
from torchvision import transforms
from torchvision.utils import save_image

from mchnpkg.deepl.multiclass import CNNTrainer, ImageNetCNN


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ImageNet CNN training script")
    parser.add_argument("--data_dir", type=str, default="/data/CPE_487-587/imagenet-1k-arrow")
    parser.add_argument("--output_dir", type=str, default="results/imagenet_run")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--train_ratio", type=float, default=0.01)
    parser.add_argument("--val_ratio", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--gpu_id",
        type=int,
        default=None,
        help="GPU index to use, e.g. 0, 1, 2. If None, choose least-used GPU.",
    )
    return parser.parse_args()


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(gpu_id: int | None = None) -> torch.device:
    if not torch.cuda.is_available():
        return torch.device("cpu")

    if gpu_id is not None:
        return torch.device(f"cuda:{gpu_id}")

    best_gpu = 0
    best_mem = None
    for i in range(torch.cuda.device_count()):
        used_mem = torch.cuda.memory_allocated(i)
        if best_mem is None or used_mem < best_mem:
            best_mem = used_mem
            best_gpu = i
    return torch.device(f"cuda:{best_gpu}")


def get_transforms() -> Tuple[transforms.Compose, transforms.Compose]:
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, val_transform


class ImageNetTorchDataset(torch.utils.data.Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int):
        sample = self.dataset[idx]
        image = sample["image"].convert("RGB")
        label = int(sample["label"])

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def build_subset(dataset, ratio: float, seed: int):
    n_total = len(dataset)
    n_use = max(1, int(n_total * ratio))

    rng = np.random.default_rng(seed)
    indices = rng.choice(n_total, size=n_use, replace=False)

    return Subset(dataset, indices.tolist())


def save_example_images(train_loader: DataLoader, val_loader: DataLoader, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    train_images, train_labels = next(iter(train_loader))
    val_images, val_labels = next(iter(val_loader))

    save_image(train_images[0], output_dir / f"example_train_label_{int(train_labels[0])}.png")
    save_image(val_images[0], output_dir / f"example_val_label_{int(val_labels[0])}.png")


def plot_curves(history: dict, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure()
    plt.plot(history["train_loss"], label="Train Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Loss vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=300, bbox_inches="tight")
    plt.close()

    plt.figure()
    plt.plot(history["train_accuracy"], label="Train Accuracy")
    plt.plot(history["val_accuracy"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Epoch")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "accuracy_curve.png", dpi=300, bbox_inches="tight")
    plt.close()


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    device = get_device(args.gpu_id)
    print(f"Using device: {device}")

    train_transform, val_transform = get_transforms()

    print("Loading cached ImageNet dataset...")
    dataset = load_from_disk(args.data_dir)

    original_train_count = len(dataset["train"])
    original_val_count = len(dataset["validation"])

    if hasattr(dataset["train"].features["label"], "names"):
        class_names = dataset["train"].features["label"].names
        num_classes = len(class_names)
    else:
        max_label = max(dataset["train"]["label"])
        num_classes = int(max_label) + 1
        class_names = [str(i) for i in range(num_classes)]

    print(f"Original train samples: {original_train_count}")
    print(f"Original validation samples: {original_val_count}")
    print(f"Number of classes: {num_classes}")

    train_dataset_full = ImageNetTorchDataset(dataset["train"], transform=train_transform)
    val_dataset_full = ImageNetTorchDataset(dataset["validation"], transform=val_transform)

    train_dataset = build_subset(train_dataset_full, args.train_ratio, args.seed)
    val_dataset = build_subset(val_dataset_full, args.val_ratio, args.seed + 1)

    print(f"Subset train samples: {len(train_dataset)}")
    print(f"Subset validation samples: {len(val_dataset)}")

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    save_example_images(train_loader, val_loader, output_dir)

    model = ImageNetCNN(num_classes=num_classes, dropout=0.5)

    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=0.9,
        weight_decay=1e-4,
    )

    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)
    criterion = torch.nn.CrossEntropyLoss()

    trainer = CNNTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epoch=args.epochs,
        eta=args.lr,
        loss=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        print_every=10,
    )

    print("Starting training...")
    history = trainer.train()

    trainer.plot_history(save_prefix=str(output_dir / "imagenet"))
    plot_curves(history, output_dir)

    onnx_path = trainer.save(file_name=str(output_dir / "imagenet_cnn.onnx"))
    print(f"Saved ONNX model to: {onnx_path}")

    param_count = trainer.count_trainable_parameters()
    print(f"Total trainable parameters: {param_count}")

    summary_path = output_dir / "summary.txt"
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Original train samples: {original_train_count}\n")
        f.write(f"Original validation samples: {original_val_count}\n")
        f.write(f"Number of classes: {num_classes}\n")
        f.write(f"Total trainable parameters: {param_count}\n")
        f.write(f"Train subset samples: {len(train_dataset)}\n")
        f.write(f"Validation subset samples: {len(val_dataset)}\n")
        f.write(f"ONNX path: {onnx_path}\n")

    print(f"Saved summary to: {summary_path}")
    print("Done.")


if __name__ == "__main__":
    main()