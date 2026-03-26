from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, List

import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class ConvLayer(nn.Module):
    """
    Reusable convolution block:
    Conv2d -> BatchNorm2d -> ReLU -> MaxPool2d
    """

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ImageNetCNN(nn.Module):
    """
    CNN for ImageNet classification.

    Architecture:
    3 -> 64 -> 128 -> 256 -> 512 -> 512
    then global average pooling
    then FC 512->1024 -> ReLU -> Dropout -> FC -> num_classes
    """

    def __init__(self, num_classes: int, dropout: float = 0.5):
        super().__init__()

        self.block1 = ConvLayer(3, 64)
        self.block2 = ConvLayer(64, 128)
        self.block3 = ConvLayer(128, 256)
        self.block4 = ConvLayer(256, 512)
        self.block5 = ConvLayer(512, 512)

        self.global_avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()

        self.fc1 = nn.Linear(512, 1024)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=dropout)
        self.fc2 = nn.Linear(1024, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)

        x = self.global_avg_pool(x)
        x = self.flatten(x)

        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x  # logits


@dataclass
class CNNTrainer:
    model: nn.Module
    train_loader: torch.utils.data.DataLoader
    val_loader: torch.utils.data.DataLoader
    epoch: int = 10
    eta: float = 1e-2
    loss: Optional[nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    scheduler: Optional[Any] = None
    device: Optional[torch.device] = None
    print_every: int = 10

    train_loss_vector: List[float] = field(default_factory=list)
    train_accuracy_vector: List[float] = field(default_factory=list)
    val_loss_vector: List[float] = field(default_factory=list)
    val_accuracy_vector: List[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)

        if self.loss is None:
            self.loss = nn.CrossEntropyLoss()

        if self.optimizer is None:
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=self.eta,
                momentum=0.9,
                weight_decay=1e-4,
            )

    def _run_one_train_epoch(self, ep: int) -> Tuple[float, float]:
        assert self.optimizer is not None
        assert self.loss is not None

        self.model.train()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images = images.to(self.device, dtype=torch.float32)
            labels = labels.to(self.device, dtype=torch.long)

            self.optimizer.zero_grad()
            logits = self.model(images)
            loss_val = self.loss(logits, labels)
            loss_val.backward()
            self.optimizer.step()

            preds = torch.argmax(logits.detach(), dim=1)
            correct = (preds == labels).sum().item()
            total = labels.size(0)

            running_loss += loss_val.item() * total
            running_correct += correct
            running_total += total

            if (batch_idx + 1) % self.print_every == 0:
                batch_acc = correct / total
                print(
                    f"Epoch [{ep + 1}/{self.epoch}] "
                    f"Batch [{batch_idx + 1}/{len(self.train_loader)}] "
                    f"Loss: {loss_val.item():.6f} "
                    f"Acc: {batch_acc:.4f}"
                )

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total
        return epoch_loss, epoch_acc

    def _run_one_val_epoch(self) -> Tuple[float, float]:
        assert self.loss is not None

        self.model.eval()

        running_loss = 0.0
        running_correct = 0
        running_total = 0

        with torch.no_grad():
            for images, labels in self.val_loader:
                images = images.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                logits = self.model(images)
                loss_val = self.loss(logits, labels)

                preds = torch.argmax(logits, dim=1)
                correct = (preds == labels).sum().item()
                total = labels.size(0)

                running_loss += loss_val.item() * total
                running_correct += correct
                running_total += total

        epoch_loss = running_loss / running_total
        epoch_acc = running_correct / running_total
        return epoch_loss, epoch_acc

    def train(self) -> Dict[str, List[float]]:
        for ep in range(self.epoch):
            train_loss, train_acc = self._run_one_train_epoch(ep)
            val_loss, val_acc = self._run_one_val_epoch()

            self.train_loss_vector.append(train_loss)
            self.train_accuracy_vector.append(train_acc)
            self.val_loss_vector.append(val_loss)
            self.val_accuracy_vector.append(val_acc)

            if self.scheduler is not None:
                self.scheduler.step()

            print(
                f"Epoch [{ep + 1}/{self.epoch}] completed | "
                f"Train Loss: {train_loss:.6f}, Train Acc: {train_acc:.4f} | "
                f"Val Loss: {val_loss:.6f}, Val Acc: {val_acc:.4f}"
            )

        return {
            "train_loss": self.train_loss_vector,
            "train_accuracy": self.train_accuracy_vector,
            "val_loss": self.val_loss_vector,
            "val_accuracy": self.val_accuracy_vector,
        }

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            x = x.to(self.device, dtype=torch.float32)
            logits = self.model(x)
            preds = torch.argmax(logits, dim=1)
        return preds.detach().cpu()

    def evaluate_loader(self, loader: torch.utils.data.DataLoader) -> Dict[str, Any]:
        self.model.eval()

        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device, dtype=torch.float32)
                labels = labels.to(self.device, dtype=torch.long)

                logits = self.model(images)
                preds = torch.argmax(logits, dim=1)

                all_preds.append(preds.cpu())
                all_labels.append(labels.cpu())

        y_pred = torch.cat(all_preds).numpy()
        y_true = torch.cat(all_labels).numpy()
        accuracy = float((y_pred == y_true).mean())

        return {
            "accuracy": accuracy,
            "y_true": y_true,
            "y_pred": y_pred,
        }

    def save(
        self,
        file_name: str = "model.onnx",
        input_size: Tuple[int, int, int] = (3, 224, 224),
    ) -> Path:
        self.model.eval()

        out_path = Path(file_name).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        c, h, w = input_size
        dummy = torch.randn(1, c, h, w, device=self.device, dtype=torch.float32)

        torch.onnx.export(
            self.model,
            dummy,
            str(out_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["images"],
            output_names=["logits"],
            dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        )
        return out_path

    def plot_history(self, save_prefix: str = "results/imagenet") -> None:
        out_dir = Path(save_prefix).expanduser().resolve().parent
        out_dir.mkdir(parents=True, exist_ok=True)

        plt.figure()
        plt.plot(self.train_loss_vector, label="Train Loss")
        plt.plot(self.val_loss_vector, label="Val Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.title("Training and Validation Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_loss.png", dpi=300, bbox_inches="tight")
        plt.close()

        plt.figure()
        plt.plot(self.train_accuracy_vector, label="Train Accuracy")
        plt.plot(self.val_accuracy_vector, label="Val Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.title("Training and Validation Accuracy")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"{save_prefix}_accuracy.png", dpi=300, bbox_inches="tight")
        plt.close()

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)