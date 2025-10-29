# pkg_name/model/regression.py
from __future__ import annotations
import math
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class LinearRegression:
    """
    A minimal PyTorch-based Linear Regression for one variable.

    Model: y = w1 * x + w0
    Loss : Mean Squared Error (MSE)

    Assignment requirements covered:
      - __init__(learning_rate, n_epochs) sets optimizer=SGD and loss=MSE
      - forward(x) computes y
      - fit(train) trains; optionally accepts test set and computes R^2 on test
      - predict(x) predicts AnnualProduction given BCR
      - analysis() creates a single figure containing:
          (i) data + fitted line,
          (ii) loss vs. epoch,
          (iii) w1 history,
          (iv) w0 history
      - stores w0, w1, and loss histories during training
    """

    def __init__(
        self,
        learning_rate: float = 1e-2,
        n_epochs: int = 1000,
        tolerance: float = 1e-6,
        seed: Optional[int] = None,
    ):
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)

        self.learning_rate = float(learning_rate)
        self.n_epochs = int(n_epochs)
        self.tolerance = float(tolerance)

        # Parameters
        self.w1 = nn.Parameter(torch.randn(1, requires_grad=True))
        self.w0 = nn.Parameter(torch.randn(1, requires_grad=True))

        # Optimizer and loss
        self.criterion = nn.MSELoss()
        self.optimizer = optim.SGD([self.w1, self.w0], lr=self.learning_rate)

        # Training artifacts
        self.loss_history: list[float] = []
        self.w1_history: list[float] = []
        self.w0_history: list[float] = []

        # Bookkeeping
        self._fitted: bool = False
        self._train_r2: Optional[float] = None
        self._test_r2: Optional[float] = None

        # Cache of the last train data (for plotting)
        self._X_train: Optional[np.ndarray] = None
        self._y_train: Optional[np.ndarray] = None

    # (b) forward
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w1 * x + self.w0

    @staticmethod
    def _to_tensor_1d(x: np.ndarray | list | torch.Tensor) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            t = x
        else:
            t = torch.tensor(x, dtype=torch.float32)
        return t.view(-1)

    @staticmethod
    def _r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true = y_true.reshape(-1)
        y_pred = y_pred.reshape(-1)
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
        return 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    # (c) fit
    def fit(
        self,
        X_train: np.ndarray | list | torch.Tensor,
        y_train: np.ndarray | list | torch.Tensor,
        X_test: Optional[np.ndarray | list | torch.Tensor] = None,
        y_test: Optional[np.ndarray | list | torch.Tensor] = None,
        verbose: bool = False,
    ) -> "LinearRegression":
        x = self._to_tensor_1d(X_train)
        y = self._to_tensor_1d(y_train)

        self._X_train = x.detach().cpu().numpy()
        self._y_train = y.detach().cpu().numpy()

        prev_loss = math.inf
        for epoch in range(1, self.n_epochs + 1):
            self.optimizer.zero_grad()
            y_pred = self.forward(x)
            loss = self.criterion(y_pred, y)
            loss.backward()
            self.optimizer.step()

            # store histories
            L = loss.item()
            self.loss_history.append(L)
            self.w1_history.append(float(self.w1.detach().cpu().numpy()))
            self.w0_history.append(float(self.w0.detach().cpu().numpy()))

            if verbose and (epoch % max(1, self.n_epochs // 10) == 0 or epoch == 1):
                print(f"[{epoch:5d}/{self.n_epochs}] loss={L:.6f} w1={self.w1.item():.6f} w0={self.w0.item():.6f}")

            # simple early stopping on loss change
            if abs(prev_loss - L) < self.tolerance:
                if verbose:
                    print(f"Converged at epoch {epoch} (Δloss < {self.tolerance:g}).")
                break
            prev_loss = L

        # compute train R^2
        with torch.no_grad():
            yhat_tr = self.forward(x).detach().cpu().numpy()
        self._train_r2 = self._r2_score(self._y_train, yhat_tr)

        # optionally compute test R^2 (assignment calls for computing R^2 on test data)
        if X_test is not None and y_test is not None:
            xt = self._to_tensor_1d(X_test)
            yt = self._to_tensor_1d(y_test)
            with torch.no_grad():
                yhat_te = self.forward(xt).detach().cpu().numpy()
            self._test_r2 = self._r2_score(yt.detach().cpu().numpy(), yhat_te)

        self._fitted = True
        return self

    # (d) predict
    def predict(self, X: np.ndarray | list | torch.Tensor) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit(...) before predict(...).")
        x = self._to_tensor_1d(X)
        with torch.no_grad():
            yhat = self.forward(x).detach().cpu().numpy()
        return yhat

    def summary(self) -> Dict[str, Any]:
        if not self._fitted:
            raise RuntimeError("Model not fitted.")
        return {
            "parameters": {"w1 (slope)": float(self.w1.item()), "w0 (intercept)": float(self.w0.item())},
            "training": {
                "epochs_run": len(self.loss_history),
                "final_loss": self.loss_history[-1] if self.loss_history else None,
                "train_R2": self._train_r2,
                "test_R2": self._test_r2,
            },
        }

# (e) analysis plots on one figure
    def analysis(self, title: Optional[str] = None, figsize: Tuple[int, int] = (12, 9)) -> plt.Figure:
        """
        Creates ONE figure with 4 subplots:
          1) scatter of training data + fitted line
          2) loss vs. epoch
          3) w1 vs. epoch
          4) w0 vs. epoch
        """
        if not self._fitted:
            raise RuntimeError("Fit the model before calling analysis().")
        if self._X_train is None or self._y_train is None:
            raise RuntimeError("Training data cache is empty.")

        X = self._X_train
        y = self._y_train

        xx = np.linspace(X.min(), X.max(), 200)
        yy = self.predict(xx)

        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.25)

        ax1 = fig.add_subplot(gs[0, 0])
        ax1.scatter(X, y, alpha=0.7, label="Data")
        ax1.plot(xx, yy, linewidth=2, label=f"Fit: y = {self.w1.item():.3f}x + {self.w0.item():.3f}")
        ax1.set_xlabel("BCR (x)")
        ax1.set_ylabel("AnnualProduction (y)")
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        ax1.set_title("Data & Fitted Line")

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(np.arange(1, len(self.loss_history) + 1), self.loss_history, linewidth=2)
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Loss (MSE)")
        ax2.grid(True, alpha=0.3)
        ax2.set_title("Loss vs. Epoch")

        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(np.arange(1, len(self.w1_history) + 1), self.w1_history, linewidth=2)
        ax3.set_xlabel("Epoch")
        ax3.set_ylabel("w1 (slope)")
        ax3.grid(True, alpha=0.3)
        ax3.set_title("w1 during Training")

        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(np.arange(1, len(self.w0_history) + 1), self.w0_history, linewidth=2)
        ax4.set_xlabel("Epoch")
        ax4.set_ylabel("w0 (intercept)")
        ax4.grid(True, alpha=0.3)
        ax4.set_title("w0 during Training")

        if title:
            fig.suptitle(title, y=1.02, fontsize=14)
        return fig

class CauchyLoss(nn.Module):
    def __init__(self, c: float = 1.0):
        super().__init__()
        self.c = float(c)
    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = (y - yhat) / self.c
        # mean over samples (and over output dim)
        return 0.5 * torch.log1p(z**2).mean()

class CauchyRegression(nn.Module):
    def __init__(self, n_features: int, c: float = 1.0):
        super().__init__()
        self.w = nn.Parameter(torch.zeros((n_features, 1), dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros((1,), dtype=torch.float32))
        self.loss_fn = CauchyLoss(c=c)
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.w + self.b

# -- tensors
Xtr_t = torch.tensor(X_train_s, dtype=torch.float32)
ytr_t = torch.tensor(y_train,   dtype=torch.float32)
Xte_t = torch.tensor(X_test_s,  dtype=torch.float32)
yte_t = torch.tensor(y_test,    dtype=torch.float32)

# -- model / optimizer
torch.manual_seed(0)
model = CauchyRegression(n_features=X_train_s.shape[1], c=1.0)
opt = torch.optim.Adam(model.parameters(), lr=0.05)

# -- train
loss_hist = []
for epoch in range(5000):
    opt.zero_grad()
    yhat = model(Xtr_t)
    loss = model.loss_fn(yhat, ytr_t)
    loss.backward()
    opt.step()
    loss_hist.append(float(loss.detach().cpu().numpy()))
    if epoch % 500 == 0:
        print(f"epoch {epoch:4d}  loss={loss_hist[-1]:.6f}")

# -- final loss (train)
final_loss = loss_hist[-1]
print("Final train Cauchy loss:", round(final_loss, 6))

# -- coefficients back on standardized X scale
b = model.b.detach().cpu().numpy().item()
w = model.w.detach().cpu().numpy().ravel()
print("Intercept (b):", b)
print("Weights on standardized features [AT, V, AP, RH]:", w)

# -- predictions, metrics
with torch.no_grad():
    yhat_tr = model(Xtr_t).cpu().numpy()
    yhat_te = model(Xte_t).cpu().numpy()

def metrics(y_true, y_pred):
    r2  = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return r2, mae, rmse

print("Train (R2, MAE, RMSE):", [round(v,4) for v in metrics(y_train, yhat_tr)])
print("Test  (R2, MAE, RMSE):", [round(v,4) for v in metrics(y_test,  yhat_te)])

# -- (optional) Loss curve figure (one plot, no styles)
plt.figure(figsize=(6,4))
plt.plot(loss_hist)
plt.xlabel("Epoch")
plt.ylabel("Train Cauchy Loss (c=1)")
plt.title("Training Loss")
plt.show()










"""
Robust and ordinary linear regression utilities.

Implements:
- CauchyLoss (robust, c=1 by default)
- CauchyRegression (PyTorch)
- OLS closed-form for comparison
- Helpers: metrics, scale conversions, bootstrap CIs, residual plots

Dependencies: numpy, pandas (optional), torch, matplotlib, scikit-learn (for metrics only)
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict

import numpy as np
import torch
import torch.nn as nn

# Metrics kept lightweight (no seaborn). Only scikit metrics are used.
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt


# ------------------------------
# Utilities
# ------------------------------
def add_intercept(X: np.ndarray) -> np.ndarray:
    """Add a column of ones as the first column."""
    return np.hstack([np.ones((X.shape[0], 1)), X])


def ols_closed_form(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    Closed-form OLS solution with intercept.

    Args:
        X: array (n, p) WITHOUT intercept column.
        y: array (n, 1)
    Returns:
        beta: array (p+1, 1) -> [b, w1, ..., wp]
    """
    Xi = add_intercept(X)
    beta = np.linalg.pinv(Xi.T @ Xi) @ (Xi.T @ y)
    return beta


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Return R2/MAE/RMSE."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
    }


def to_original_scale(
    w_std: np.ndarray, b_std: float, mu: np.ndarray, sd: np.ndarray
) -> Tuple[np.ndarray, float]:
    """
    Convert standardized-feature parameters (w_std, b_std) to original feature scale.

    Model on std scale:  y = b_std + sum_j w_std_j * (x_j - mu_j)/sd_j
    =>
    y = b_orig + sum_j w_orig_j * x_j, where
      w_orig_j = w_std_j / sd_j
      b_orig   = b_std - sum_j (w_std_j * mu_j / sd_j)
    """
    w_orig = w_std / sd
    b_orig = b_std - float((w_std * mu / sd).sum())
    return w_orig, b_orig


# ------------------------------
# Robust loss (Cauchy)
# ------------------------------
class CauchyLoss(nn.Module):
    """
    Cauchy loss (robust). For residual r = y - yhat and scale c:
        L = (c^2 / 2) * log(1 + (r/c)^2)
    Here we drop c^2 factor (constant wrt params) and use:
        0.5 * log(1 + (r/c)^2)
    """

    def __init__(self, c: float = 1.0):
        super().__init__()
        self.c = float(c)

    def forward(self, yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z = (y - yhat) / self.c
        return 0.5 * torch.log1p(z**2).mean()


# ------------------------------
# Cauchy Regression (PyTorch)
# ------------------------------
@dataclass
class CauchyRegressionConfig:
    c: float = 1.0
    lr: float = 0.05
    max_epochs: int = 5000
    tol: float = 1e-8
    optimizer: str = "adam"  # "adam" or "sgd"
    verbose: bool = False
    random_state: Optional[int] = 0


class CauchyRegression(nn.Module):
    """
    Linear model with Cauchy loss:
        yhat = b + X @ w

    Fits on standardized or raw features (your choice). If you pass standardized X,
    use `to_original_scale` to convert coefficients for reporting.
    """

    def __init__(self, n_features: int, cfg: Optional[CauchyRegressionConfig] = None):
        super().__init__()
        self.n_features = int(n_features)
        self.cfg = cfg or CauchyRegressionConfig()
        if self.cfg.random_state is not None:
            torch.manual_seed(self.cfg.random_state)

        # Parameters
        self.w = nn.Parameter(torch.zeros((self.n_features, 1), dtype=torch.float32))
        self.b = nn.Parameter(torch.zeros((1,), dtype=torch.float32))

        # Loss
        self.loss_fn = CauchyLoss(c=self.cfg.c)

        # Optimizer
        if self.cfg.optimizer.lower() == "sgd":
            self.opt = torch.optim.SGD(self.parameters(), lr=self.cfg.lr, momentum=0.9)
        else:
            self.opt = torch.optim.Adam(self.parameters(), lr=self.cfg.lr)

        # Fitted flags/holders
        self._fitted = False
        self.history_: List[float] = []

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return X @ self.w + self.b

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CauchyRegression":
        """
        Train with gradient descent.

        Args:
            X: array (n, p), features
            y: array (n, 1) or (n,)
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1, 1)

        X_t = torch.tensor(X, dtype=torch.float32)
        y_t = torch.tensor(y, dtype=torch.float32)

        prev = float("inf")
        self.history_.clear()

        for epoch in range(self.cfg.max_epochs):
            self.opt.zero_grad()
            yhat = self.forward(X_t)
            loss = self.loss_fn(yhat, y_t)
            loss.backward()
            self.opt.step()

            val = float(loss.detach().cpu().numpy())
            self.history_.append(val)

            if self.cfg.verbose and (epoch % 500 == 0 or epoch == self.cfg.max_epochs - 1):
                print(f"[CauchyRegression] epoch={epoch:5d} loss={val:.6f}")

            if abs(prev - val) < self.cfg.tol:
                break
            prev = val

        self._fitted = True
        return self

    # Properties for sklearn-like API
    @property
    def coef_(self) -> np.ndarray:
        """Weights as (p,) ndarray (standardized-scale if you trained on standardized X)."""
        return self.w.detach().cpu().numpy().ravel()

    @property
    def intercept_(self) -> float:
        """Intercept (standardized-scale if you trained on standardized X)."""
        return float(self.b.detach().cpu().numpy().ravel()[0])

    def predict(self, X: np.ndarray) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Model is not fitted. Call .fit() first.")
        X_t = torch.tensor(np.asarray(X, dtype=np.float32), dtype=torch.float32)
        with torch.no_grad():
            yhat = self.forward(X_t).cpu().numpy()
        return yhat

    def score(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        yhat = self.predict(X)
        return regression_metrics(y, yhat)


# ------------------------------
# Bootstrap Confidence Intervals
# ------------------------------
def bootstrap_cis_cauchy(
    X: np.ndarray,
    y: np.ndarray,
    mu: Optional[np.ndarray],
    sd: Optional[np.ndarray],
    cfg: Optional[CauchyRegressionConfig] = None,
    B: int = 500,
    random_state: int = 123,
) -> Dict[str, np.ndarray]:
    """
    Nonparametric bootstrap CIs for [Intercept, w1, ..., wp] on ORIGINAL scale.
    Resamples rows with replacement, refits, converts coefs to original scale.

    Args:
        X, y: training data used to fit
        mu, sd: feature means/scales used for standardization; if None, assume raw scale
        cfg: training config; reasonable default if None
        B: bootstrap reps
    Returns:
        dict with keys: 'est', 'low', 'high', 'names'
    """
    rng = np.random.default_rng(random_state)
    n, p = X.shape
    cfg = cfg or CauchyRegressionConfig()

    # fit once to get point estimate (on std scale if X is std)
    base = CauchyRegression(n_features=p, cfg=cfg)
    base.fit(X, y)
    w_std = base.coef_.copy()
    b_std = base.intercept_

    if (mu is not None) and (sd is not None):
        w_est, b_est = to_original_scale(w_std, b_std, mu, sd)
    else:
        w_est, b_est = w_std, b_std

    params = np.zeros((B, p + 1), dtype=float)
    for b in range(B):
        idx = rng.integers(0, n, size=n)
        Xb, yb = X[idx], y[idx]
        m = CauchyRegression(n_features=p, cfg=cfg)
        m.fit(Xb, yb)
        ws, bs = m.coef_, m.intercept_
        if (mu is not None) and (sd is not None):
            w_o, b_o = to_original_scale(ws, bs, mu, sd)
        else:
            w_o, b_o = ws, bs
        params[b, 0] = b_o
        params[b, 1:] = w_o

    low = np.percentile(params, 2.5, axis=0)
    high = np.percentile(params, 97.5, axis=0)
    est = np.concatenate([[b_est], w_est])
    names = ["Intercept"] + [f"w{j+1}" for j in range(p)]
    return {"est": est, "low": low, "high": high, "names": names}


# ------------------------------
# Simple plotting helpers
# ------------------------------
def plot_loss_curve(loss_history: List[float]) -> None:
    plt.figure(figsize=(6, 4))
    plt.plot(loss_history)
    plt.xlabel("Epoch")
    plt.ylabel("Train Cauchy Loss")
    plt.title("Training Loss")
    plt.show()


def residual_scatter(y_true: np.ndarray, y_pred: np.ndarray, x: np.ndarray, x_name: str) -> None:
    res = (y_pred - y_true).ravel()
    plt.figure(figsize=(6, 4))
    plt.scatter(x, res, s=8)
    plt.axhline(0.0)
    plt.xlabel(x_name)
    plt.ylabel("Residual (ŷ - y)")
    plt.title(f"Residuals vs {x_name}")
    plt.show()

