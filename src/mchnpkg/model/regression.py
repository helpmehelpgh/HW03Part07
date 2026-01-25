from __future__ import annotations
import io
from dataclasses import dataclass
from typing import Optional, Tuple, List, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix


def _to_tensor(x: np.ndarray, dtype=torch.float32) -> torch.Tensor:
    return torch.tensor(x, dtype=dtype)


class _LinearModel(nn.Module):
    """
    Linear model with explicit bias: y_hat = w0 + X @ w
    """
    def __init__(self, in_features: int):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(in_features))  # shape [D]
        self.b = nn.Parameter(torch.zeros(()))           # scalar bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.b + x @ self.w


@dataclass
class FitResult:
    coef_: np.ndarray
    intercept_: float
    history_: List[float]


class CauchyRegression:
    """
    Multiple linear regression trained with Cauchy loss (robust to outliers).

    Loss per sample:  (c^2 / 2) * log(1 + ((y - y_hat)/c)^2)

    Features:
      - Torch-based optimizer (Adam) with L2 weight decay (optional)
      - Optional standardization of features (mean/std) for stable training
      - Scatterplot matrix helper
      - Residual plots per feature
    """
    def __init__(
        self,
        c: float = 1.0,
        lr: float = 1e-2,
        max_epochs: int = 3000,
        weight_decay: float = 0.0,         # L2 on weights (not bias)
        standardize: bool = True,
        tol: float = 1e-8,
        patience: int = 200,
        random_state: int = 42,
        verbose: bool = False
    ):
        self.c = float(c)
        self.lr = float(lr)
        self.max_epochs = int(max_epochs)
        self.weight_decay = float(weight_decay)
        self.standardize = bool(standardize)
        self.tol = float(tol)
        self.patience = int(patience)
        self.random_state = int(random_state)
        self.verbose = bool(verbose)

        # learned / fitted stuff
        self.model_: Optional[_LinearModel] = None
        self.mu_: Optional[np.ndarray] = None
        self.sigma_: Optional[np.ndarray] = None
        self.feature_names_: Optional[List[str]] = None
        self.history_: List[float] = []

    # ---------- utilities ----------
    def _standardize_fit(self, X: np.ndarray) -> np.ndarray:
        self.mu_ = X.mean(axis=0)
        self.sigma_ = X.std(axis=0, ddof=0)
        self.sigma_[self.sigma_ == 0.0] = 1.0  # avoid divide by zero
        return (X - self.mu_) / self.sigma_

    def _standardize_apply(self, X: np.ndarray) -> np.ndarray:
        if self.mu_ is None or self.sigma_ is None:
            return X
        return (X - self.mu_) / self.sigma_

    def _cauchy_loss(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.c
        r = (y - yhat) / c
        return 0.5 * (c ** 2) * torch.log1p(r * r).mean()

    # ---------- public API ----------
    def fit(
        self,
        X: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray
    ) -> FitResult:
        """
        Train the model on full batch using Adam + Cauchy loss.

        X: array-like (N, D)  with columns [AT, V, AP, RH] in any order
        y: array-like (N,)
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = list(X.columns)
            X_np = X.to_numpy(dtype=float)
        else:
            X_np = np.asarray(X, dtype=float)
            self.feature_names_ = [f"x{i+1}" for i in range(X_np.shape[1])]

        y_np = np.asarray(y, dtype=float).reshape(-1)

        # standardize if requested
        if self.standardize:
            Xn = self._standardize_fit(X_np)
        else:
            Xn = X_np.copy()

        torch.manual_seed(self.random_state)
        self.model_ = _LinearModel(in_features=Xn.shape[1])

        # L2 via optimizer weight_decay (applies to all params); we emulate L2 on w only
        # by constructing param groups.
        opt = torch.optim.Adam([
            {"params": [self.model_.w], "weight_decay": self.weight_decay},
            {"params": [self.model_.b], "weight_decay": 0.0},
        ], lr=self.lr)

        X_t = _to_tensor(Xn)
        y_t = _to_tensor(y_np)

        self.history_.clear()
        best_loss = float("inf")
        bad_epochs = 0
        best_state: Dict[str, torch.Tensor] | None = None

        for epoch in range(self.max_epochs):
            opt.zero_grad()
            yhat = self.model_(X_t)
            loss = self._cauchy_loss(yhat, y_t)
            loss.backward()
            opt.step()

            L = float(loss.item())
            self.history_.append(L)

            if L + self.tol < best_loss:
                best_loss = L
                bad_epochs = 0
                best_state = {k: v.detach().clone() for k, v in self.model_.state_dict().items()}
            else:
                bad_epochs += 1
                if bad_epochs >= self.patience:
                    if self.verbose:
                        print(f"Early stop at epoch {epoch}, best loss={best_loss:.6f}")
                    break

        # restore best weights
        if best_state is not None:
            self.model_.load_state_dict(best_state)

        # convert standardized weights back to original feature scale for interpretability:
        # y = b_s + sum(w_s * (x - mu)/sigma)  =>  y = (b_s - sum(w_s*mu/sigma)) + sum((w_s/sigma) * x)
        w_s = self.model_.w.detach().cpu().numpy()
        b_s = float(self.model_.b.detach().cpu().numpy())

        if self.standardize:
            coef = w_s / self.sigma_
            intercept = b_s - np.sum((w_s * self.mu_) / self.sigma_)
        else:
            coef = w_s
            intercept = b_s

        return FitResult(coef_=coef, intercept_=intercept, history_=self.history_)

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.model_ is None:
            raise RuntimeError("Call fit() before predict().")
        if isinstance(X, pd.DataFrame):
            X_np = X.to_numpy(dtype=float)
        else:
            X_np = np.asarray(X, dtype=float)
        Xn = self._standardize_apply(X_np) if self.standardize else X_np
        with torch.no_grad():
            yhat = self.model_(_to_tensor(Xn)).cpu().numpy()
        return yhat

    def scatterplot_matrix(self, df: pd.DataFrame, target_col: str, figsize=(9, 9)) -> None:
        """
        Quickly draw a scatterplot matrix including the target.
        """
        cols = [c for c in df.columns if c != target_col] + [target_col]
        sm = scatter_matrix(df[cols], figsize=figsize, diagonal='hist')
        plt.tight_layout()
        plt.show()

    def residual_plots(self, X: pd.DataFrame, y: pd.Series, figsize=(12, 8)) -> None:
        """
        Plot residuals (y_hat - y) vs each feature.
        """
        yhat = self.predict(X)
        resid = yhat - np.asarray(y).reshape(-1)
        feats = list(X.columns) if isinstance(X, pd.DataFrame) else [f"x{i+1}" for i in range(X.shape[1])]

        n = len(feats)
        ncols = min(2, n)
        nrows = int(np.ceil(n / ncols))

        # single-axes plots (no specific colors/styles to keep it simple)
        idx = 0
        plt.figure(figsize=figsize)
        for j, name in enumerate(feats):
            idx += 1
            plt.subplot(nrows, ncols, idx)
            plt.scatter(np.asarray(X)[:, j], resid, s=10)
            plt.axhline(0.0, linestyle="--")
            plt.xlabel(name)
            plt.ylabel("Residual (ŷ - y)")
            plt.title(f"Residuals vs {name}")
        plt.tight_layout()
        plt.show()
