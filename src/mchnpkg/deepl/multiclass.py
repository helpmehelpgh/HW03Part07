from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Type, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, precision_recall_fscore_support, accuracy_score


class SimpleNN(nn.Module):
    """
    in_features -> 3 -> 4 -> 5 -> num_classes (logits)
    """

    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 3)
        self.fc2 = nn.Linear(3, 4)
        self.fc3 = nn.Linear(4, 5)
        self.fc4 = nn.Linear(5, num_classes)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        return self.fc4(x)  # logits (N, num_classes)


@dataclass
class ClassTrainer:
    X_train: torch.Tensor
    Y_train: torch.Tensor
    eta: float = 1e-3
    epoch: int = 100  # HW uses "epoch"
    loss: Optional[nn.Module] = None
    optimizer: Optional[torch.optim.Optimizer] = None
    model_cls: Type[nn.Module] = SimpleNN
    num_classes: int = 4
    device: Optional[torch.device] = None

    # stored history
    loss_vector: Optional[torch.Tensor] = None
    accuracy_vector: Optional[torch.Tensor] = None

    # trained model
    model: Optional[nn.Module] = None

    def __post_init__(self) -> None:
        if self.device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.X_train = self.X_train.to(self.device).float()
        self.Y_train = self.Y_train.to(self.device).long().view(-1)

        if self.loss is None:
            self.loss = nn.CrossEntropyLoss()

        self.model = self.model_cls(in_features=int(self.X_train.shape[1]), num_classes=self.num_classes).to(self.device)

        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.eta)

        self.loss_vector = torch.zeros(self.epoch, dtype=torch.float32, device="cpu")
        self.accuracy_vector = torch.zeros(self.epoch, dtype=torch.float32, device="cpu")

    def train(self) -> Tuple[torch.Tensor, torch.Tensor]:
        assert self.model is not None and self.optimizer is not None and self.loss is not None

        self.model.train()
        for ep in range(self.epoch):
            self.optimizer.zero_grad()
            logits = self.model(self.X_train)
            loss_val = self.loss(logits, self.Y_train)
            loss_val.backward()
            self.optimizer.step()

            preds = torch.argmax(logits.detach(), dim=1)
            acc = (preds == self.Y_train).float().mean().item()

            self.loss_vector[ep] = loss_val.detach().cpu()
            self.accuracy_vector[ep] = float(acc)

        return self.loss_vector, self.accuracy_vector

    def predict(self, X: torch.Tensor) -> torch.Tensor:
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X.to(self.device).float())
            preds = torch.argmax(logits, dim=1)
        return preds.detach().cpu()

    def test(self, X_test: torch.Tensor, y_test: torch.Tensor) -> Dict[str, Any]:
        y_true = y_test.view(-1).long().cpu().numpy()
        y_pred = self.predict(X_test).cpu().numpy()

        acc = float(accuracy_score(y_true, y_pred))
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_true, y_pred, average="macro", zero_division=0
        )
        cm = confusion_matrix(y_true, y_pred)

        return {
            "accuracy": acc,
            "precision_macro": float(precision),
            "recall_macro": float(recall),
            "f1_macro": float(f1),
            "confusion_matrix": cm,
        }

    def save(self, file_name: str = "model.onnx") -> Path:
        """
        Save ONNX model to file_name.
        """
        assert self.model is not None
        self.model.eval()

        out_path = Path(file_name).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)

        dummy = torch.randn(1, int(self.X_train.shape[1]), device=self.device, dtype=torch.float32)

        torch.onnx.export(
            self.model,
            dummy,
            str(out_path),
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=["X"],
            output_names=["logits"],
            dynamic_axes={"X": {0: "batch"}, "logits": {0: "batch"}},
        )
        return out_path

    def evaluation(
        self,
        X_test: Optional[torch.Tensor] = None,
        y_test: Optional[torch.Tensor] = None,
        save_prefix: str = "results/hw02",
    ) -> Dict[str, Any]:
        """
        Plot loss/accuracy and confusion matrices. Saves PDFs under save_prefix_*.
        """
        Path("results").mkdir(exist_ok=True)

        # loss plot
        if self.loss_vector is not None:
            plt.figure()
            plt.plot(self.loss_vector.numpy())
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.title("Training Loss")
            plt.savefig(f"{save_prefix}_loss.pdf", bbox_inches="tight")
            plt.close()

        # accuracy plot
        if self.accuracy_vector is not None:
            plt.figure()
            plt.plot(self.accuracy_vector.numpy())
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.title("Training Accuracy")
            plt.savefig(f"{save_prefix}_accuracy.pdf", bbox_inches="tight")
            plt.close()

        # train confusion matrix
        train_pred = self.predict(self.X_train).numpy()
        train_true = self.Y_train.detach().cpu().numpy()
        cm_train = confusion_matrix(train_true, train_pred)

        plt.figure()
        plt.imshow(cm_train)
        plt.title("Confusion Matrix (Train)")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.colorbar()
        for (i, j), v in np.ndenumerate(cm_train):
            plt.text(j, i, str(v), ha="center", va="center")
        plt.savefig(f"{save_prefix}_cm_train.pdf", bbox_inches="tight")
        plt.close()

        out = {"train_confusion_matrix": cm_train}

        if X_test is not None and y_test is not None:
            test_metrics = self.test(X_test, y_test)
            cm_test = test_metrics["confusion_matrix"]

            plt.figure()
            plt.imshow(cm_test)
            plt.title("Confusion Matrix (Test)")
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.colorbar()
            for (i, j), v in np.ndenumerate(cm_test):
                plt.text(j, i, str(v), ha="center", va="center")
            plt.savefig(f"{save_prefix}_cm_test.pdf", bbox_inches="tight")
            plt.close()

            out["test_metrics"] = test_metrics

        return out
