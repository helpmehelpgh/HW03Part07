from __future__ import annotations
import math
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader


class _LogitNet(nn.Module):
    """Single-layer logistic regression: logits = b + X @ w."""
    def __init__(self, in_features: int):
        super().__init__()
        self.lin = nn.Linear(in_features, 1, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lin(x).squeeze(-1)  # logits


class LogisticRegressionGD:
    """
    Stochastic-gradient logistic regression (BCE-with-logits).

    Args:
        lr: learning rate for SGD.
        batch_size: mini-batch size.
        max_epochs: max passes over training data.
        weight_decay: L2 regularization on weights (not bias).
        standardize: z-score standardization on features.
        pos_weight: "auto" (N_neg/N_pos), a float, or None.
        val_split: fraction of data for validation (for early stopping).
        patience: early-stop patience on val loss.
        tol: improvement tolerance on val loss.
        seed: RNG seed.
        verbose: print training logs.

    Attributes after fit():
        hist: {"loss": [...], "val": [...]}
        n_iter_: epochs actually used (after early stop).
        coef_: learned weights (on original feature scale).
        intercept_: learned bias on original scale.
    """

    def __init__(
        self,
        lr: float = 5e-2,
        batch_size: int = 2048,
        max_epochs: int = 60,
        weight_decay: float = 1e-4,
        standardize: bool = True,
        pos_weight: Optional[float | str] = "auto",
        val_split: float = 0.10,
        patience: int = 6,
        tol: float = 1e-5,
        seed: int = 42,
        verbose: bool = True,
    ):
        self.lr = float(lr)
        self.batch_size = int(batch_size)
        self.max_epochs = int(max_epochs)
        self.weight_decay = float(weight_decay)
        self.standardize = bool(standardize)
        self.pos_weight = pos_weight
        self.val_split = float(val_split)
        self.patience = int(patience)
        self.tol = float(tol)
        self.seed = int(seed)
        self.verbose = bool(verbose)

        self.mu_: Optional[np.ndarray] = None
        self.sigma_: Optional[np.ndarray] = None
        self.model_: Optional[_LogitNet] = None
        self.hist: Dict[str, List[float]] = {"loss": [], "val": []}
        self.n_iter_: int = 0
        self.coef_: Optional[np.ndarray] = None
        self.intercept_: Optional[float] = None

    # ---------- internal ----------
    def _fit_standardize(self, X: np.ndarray) -> np.ndarray:
        self.mu_ = X.mean(axis=0)
        self.sigma_ = X.std(axis=0, ddof=0)
        self.sigma_[self.sigma_ == 0.0] = 1.0
        return (X - self.mu_) / self.sigma_

    def _apply_standardize(self, X: np.ndarray) -> np.ndarray:
        if self.mu_ is None or self.sigma_ is None:
            return X
        return (X - self.mu_) / self.sigma_

    # ---------- public ----------
    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticRegressionGD":
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        assert set(np.unique(y)).issubset({0.0, 1.0}), "y must be binary 0/1."

        Xn = self._fit_standardize(X) if self.standardize else X

        # train/val split
        N = Xn.shape[0]
        idx = rng.permutation(N)
        n_val = int(self.val_split * N)
        val_idx, tr_idx = idx[:n_val], idx[n_val:]
        Xtr, ytr = Xn[tr_idx], y[tr_idx]
        Xva, yva = Xn[val_idx], y[val_idx]

        # tensors (use as_tensor to avoid arg issues)
        Xtr_t = torch.as_tensor(Xtr, dtype=torch.float32)
        ytr_t = torch.as_tensor(ytr, dtype=torch.float32)
        Xva_t = torch.as_tensor(Xva, dtype=torch.float32)
        yva_t = torch.as_tensor(yva, dtype=torch.float32)

        train_loader = DataLoader(
            TensorDataset(Xtr_t, ytr_t), batch_size=self.batch_size, shuffle=True
        )

        # model / optimizer / loss
        d = Xn.shape[1]
        self.model_ = _LogitNet(d)

        if isinstance(self.pos_weight, (int, float)):
            pw = torch.tensor(float(self.pos_weight), dtype=torch.float32)
        elif self.pos_weight == "auto":
            n_pos = float((ytr == 1).sum())
            n_neg = float((ytr == 0).sum())
            pw = torch.tensor(n_neg / max(n_pos, 1.0), dtype=torch.float32)
        else:
            pw = None

        criterion = nn.BCEWithLogitsLoss(pos_weight=pw) if pw is not None else nn.BCEWithLogitsLoss()
        optimizer = torch.optim.SGD(
            [
                {"params": self.model_.lin.weight, "weight_decay": self.weight_decay},
                {"params": self.model_.lin.bias, "weight_decay": 0.0},
            ],
            lr=self.lr,
            momentum=0.9,
        )

        best_val = math.inf
        wait = 0
        best_state = None
        self.hist = {"loss": [], "val": []}

        for epoch in range(self.max_epochs):
            # train
            self.model_.train()
            total = 0.0
            num = 0
            for xb, yb in train_loader:
                optimizer.zero_grad()
                logits = self.model_(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
                total += float(loss.item()) * len(xb)
                num += len(xb)
            train_loss = total / max(num, 1)

            # validate
            self.model_.eval()
            with torch.no_grad():
                val_loss = float(criterion(self.model_(Xva_t), yva_t).item())

            self.hist["loss"].append(train_loss)
            self.hist["val"].append(val_loss)

            if self.verbose:
                print(f"epoch {epoch+1:03d}  loss={train_loss:.4f}  val={val_loss:.4f}")

            # early stopping
            if val_loss + self.tol < best_val:
                best_val = val_loss
                wait = 0
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
            else:
                wait += 1
                if wait >= self.patience:
                    if self.verbose:
                        print("early stopping.")
                    break

        if best_state is not None:
            self.model_.load_state_dict(best_state)

        self.n_iter_ = len(self.hist["loss"])

        # export weights on original feature scale
        W = self.model_.lin.weight.detach().cpu().numpy().reshape(-1)
        b = float(self.model_.lin.bias.detach().cpu().numpy())
        if self.standardize:
            self.coef_ = W / self.sigma_
            self.intercept_ = b - float((W * self.mu_ / self.sigma_).sum())
        else:
            self.coef_ = W
            self.intercept_ = b

        return self

    def decision_function(self, X: np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() first.")
        X = np.asarray(X, dtype=np.float32)
        Xn = self._apply_standardize(X) if self.standardize else X
        with torch.no_grad():
            logits = self.model_(torch.as_tensor(Xn, dtype=torch.float32)).cpu().numpy()
        return logits

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        z = self.decision_function(X)
        p1 = 1.0 / (1.0 + np.exp(-z))
        return np.c_[1.0 - p1, p1]

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        p = self.predict_proba(X)[:, 1]
        return (p >= threshold).astype(int)
