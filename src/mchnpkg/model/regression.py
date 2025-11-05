# regression.py
# Minimal regression toolkit with:
#   - LinearRegression (OLS/Ridge; already useful for other HWs)
#   - CauchyRegression (PyTorch) with ε=1 Cauchy loss  c^2/2 * log(1 + ((y - yhat)/c)^2)
#   - Statsmodels BONUS: GenericLikelihoodModel version for Cauchy MLE
#
# Dependencies: numpy, matplotlib, torch (for CauchyRegression), statsmodels (bonus, optional)

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, List, Iterable

import numpy as np
import matplotlib.pyplot as plt

# ---------- optional libs ----------
try:
    import torch
    from torch import nn
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    import pandas as pd
    from pandas.plotting import scatter_matrix
    _HAS_PANDAS = True
except Exception:
    _HAS_PANDAS = False

try:
    import statsmodels.api as sm
    from statsmodels.base.model import GenericLikelihoodModel
    _HAS_SM = True
except Exception:
    _HAS_SM = False


# ======================================================================
# Shared helpers (kept small and dependency-light)
# ======================================================================

def _add_bias(X: np.ndarray) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    return np.hstack([np.ones((X.shape[0], 1)), X])

def standardize(X: np.ndarray, mean_: Optional[np.ndarray] = None, std_: Optional[np.ndarray] = None
                ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X[:, None]
    if mean_ is None:
        mean_ = X.mean(axis=0)
    if std_ is None:
        std_ = X.std(axis=0, ddof=0)
        std_[std_ == 0.0] = 1.0
    return (X - mean_) / std_, mean_, std_

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: Optional[int] = 42
                     ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_state)
    n = len(y)
    idx = np.arange(n)
    rng.shuffle(idx)
    n_test = int(np.ceil(test_size * n))
    te = idx[:n_test]; tr = idx[n_test:]
    return X[tr], X[te], y[tr], y[te]

def mae(y_true, y_pred): y_true=np.ravel(y_true); y_pred=np.ravel(y_pred); return float(np.mean(np.abs(y_true-y_pred)))
def mse(y_true, y_pred): y_true=np.ravel(y_true); y_pred=np.ravel(y_pred); return float(np.mean((y_true-y_pred)**2))
def rmse(y_true, y_pred): return float(np.sqrt(mse(y_true, y_pred)))
def r2_score(y_true, y_pred):
    y_true=np.ravel(y_true); y_pred=np.ravel(y_pred)
    ssr=np.sum((y_true-y_pred)**2); sst=np.sum((y_true-y_true.mean())**2)
    return float(1.0 - ssr/sst) if sst>0 else 0.0


# ======================================================================
# (Kept) Simple OLS/Ridge for convenience in your repo
# ======================================================================

@dataclass
class LinearRegression:
    fit_intercept: bool = True
    ridge_lambda: float = 0.0

    coef_: Optional[np.ndarray] = field(init=False, default=None)
    intercept_: float = field(init=False, default=0.0)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LinearRegression":
        X = np.asarray(X, float)
        y = np.asarray(y, float).ravel()
        Phi = _add_bias(X) if self.fit_intercept else X
        if self.ridge_lambda > 0:
            I = np.eye(Phi.shape[1]); 
            if self.fit_intercept: I[0,0]=0.0
            w = np.linalg.solve(Phi.T@Phi + self.ridge_lambda*I, Phi.T@y)
        else:
            w = np.linalg.lstsq(Phi, y, rcond=None)[0]
        if self.fit_intercept:
            self.intercept_ = float(w[0]); self.coef_ = w[1:]
        else:
            self.intercept_ = 0.0; self.coef_ = w
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, float)
        return (self.intercept_ + X @ self.coef_) if self.fit_intercept else (X @ self.coef_)


# ======================================================================
# NEW: CauchyRegression (PyTorch)  — main deliverable
# ======================================================================

@dataclass
class CauchyRegression:
    """
    Multiple linear regression trained by minimizing the Cauchy loss:

        L(y, ŷ) = (c^2 / 2) * log(1 + ((y - ŷ)/c)^2)

    with default c=1. This is robust to outliers (less sensitive than MSE).

    Parameters
    ----------
    learning_rate : float
    max_epochs    : int
    tol           : float       Early stopping on abs(Δloss) < tol
    c             : float       Cauchy scale (homework: c=1)
    standardize_X : bool        If True, standardizes X internally and
                                returns parameters on the ORIGINAL scale.
    optimizer     : {"adam","sgd"}
    verbose       : bool
    random_state  : Optional[int]
    """
    learning_rate: float = 0.02
    max_epochs: int = 8000
    tol: float = 1e-7
    c: float = 1.0
    standardize_X: bool = False
    optimizer: str = "adam"
    verbose: bool = False
    random_state: Optional[int] = 42

    # learned parameters on ORIGINAL X scale:
    coef_: Optional[np.ndarray] = field(init=False, default=None)  # shape (p,)
    intercept_: float = field(init=False, default=0.0)
    final_loss_: float = field(init=False, default=np.nan)
    loss_history_: List[float] = field(init=False, default_factory=list)

    # internal cache
    _X_mean_: Optional[np.ndarray] = field(init=False, default=None)
    _X_std_: Optional[np.ndarray] = field(init=False, default=None)

    def _prep_X(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        X = np.asarray(X, float)
        if X.ndim == 1: X = X[:, None]
        if self.standardize_X:
            if fit:
                X, self._X_mean_, self._X_std_ = standardize(X)
            else:
                X, _, _ = standardize(X, self._X_mean_, self._X_std_)
        return X

    def _to_original_scale(self, w: np.ndarray) -> Tuple[float, np.ndarray]:
        """Convert params from standardized design (bias + weights) back to original X scale."""
        b = float(w[0]); betas = np.array(w[1:], float)
        if not self.standardize_X:
            return b, betas
        # yhat = b + sum_j beta_j * (x_j - mu_j)/std_j
        beta_orig = betas / self._X_std_
        b_orig = b - np.sum(betas * self._X_mean_ / self._X_std_)
        return float(b_orig), beta_orig

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CauchyRegression":
        assert _HAS_TORCH, "PyTorch is required for CauchyRegression."
        X = self._prep_X(X, fit=True)
        y = np.asarray(y, float).ravel()

        n, p = X.shape
        Phi = _add_bias(X)                      # [1, X]
        # Params in STANDARDIZED space:
        g = torch.Generator().manual_seed(self.random_state or 0)
        w = torch.randn(p + 1, generator=g, dtype=torch.float64) * 1e-3
        w.requires_grad_(True)

        if self.optimizer.lower() == "adam":
            opt = torch.optim.Adam([w], lr=self.learning_rate)
        else:
            opt = torch.optim.SGD([w], lr=self.learning_rate)

        c = torch.tensor(float(self.c), dtype=torch.float64)

        Phi_t = torch.from_numpy(Phi.astype(np.float64))
        y_t   = torch.from_numpy(y.astype(np.float64))

        self.loss_history_.clear()
        last = float("inf")
        for epoch in range(self.max_epochs):
            opt.zero_grad()
            yhat = Phi_t @ w
            res = (y_t - yhat) / c
            loss = (c**2 / 2.0) * torch.log1p(res**2).mean()     # mean loss
            loss.backward()
            opt.step()

            lval = float(loss.detach().cpu().item())
            self.loss_history_.append(lval)
            if self.verbose and (epoch % 1000 == 0 or epoch == self.max_epochs-1):
                print(f"[{epoch:5d}] loss={lval:.6f}")
            if abs(last - lval) < self.tol:
                break
            last = lval

        self.final_loss_ = self.loss_history_[-1]
        # Map to ORIGINAL X scale for user-facing attributes:
        b_orig, beta_orig = self._to_original_scale(w.detach().cpu().numpy())
        self.intercept_, self.coef_ = b_orig, beta_orig
        return self

    # ---------- API ----------
    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("Model not yet fitted.")
        Xstd = self._prep_X(X, fit=False)
        return self.intercept_ + Xstd @ (self.coef_ if not self.standardize_X else (self.coef_ * 1.0)) \
               if not self.standardize_X else \
               (self.intercept_ + (X - self._X_mean_) / self._X_std_ @ (self.coef_ * self._X_std_ / self._X_std_))  # noop

    def residuals(self, X: np.ndarray, y: np.ndarray) -> np.ndarray:
        y = np.asarray(y, float).ravel()
        return y - self.predict(X)

    # ---------- Diagnostics required by HW ----------
    def confint_95(self, X: np.ndarray, y: np.ndarray) -> Dict[str, np.ndarray]:
        """
        95% Wald-type CI using the observed Hessian of the MEAN loss at the optimum.
        This is an approximation but standard in practice for custom M-estimators.

        Returns: dict with keys names, estimate, lower, upper, stderr  (ORIGINAL SCALE)
        """
        assert _HAS_TORCH, "PyTorch required."
        # Rebuild standardised design Phi and optimum parameters in standardized space:
        Xs = self._prep_X(X, fit=False)
        Phi = _add_bias(Xs)
        Phi_t = torch.from_numpy(Phi.astype(np.float64))
        y_t   = torch.from_numpy(np.asarray(y, float).ravel().astype(np.float64))

        # Recompute optimum w in STANDARDIZED space from orig coefficients:
        if self.standardize_X:
            # convert back: beta_std = beta_orig * std; b_std = b_orig + sum(beta_std * mu/std?) -> derive:
            # From earlier mapping: beta_orig = beta_std/std, b_orig = b_std - sum(beta_std*mu/std)
            # => b_std = b_orig + sum(beta_std*mu/std)
            beta_std = self.coef_ * self._X_std_
            b_std = self.intercept_ + float(np.sum(beta_std * (self._X_mean_ / self._X_std_)))
            w_hat = np.r_[b_std, beta_std]
        else:
            w_hat = np.r_[self.intercept_, self.coef_]

        w_t = torch.from_numpy(w_hat.astype(np.float64))

        c = torch.tensor(float(self.c), dtype=torch.float64)

        def meanloss(wvec: torch.Tensor) -> torch.Tensor:
            yhat = Phi_t @ wvec
            res = (y_t - yhat) / c
            return (c**2 / 2.0) * torch.log1p(res**2).mean()

        # Hessian wrt parameters (observed information of mean loss)
        H = torch.autograd.functional.hessian(meanloss, w_t)
        H = H.detach().cpu().numpy()
        # Numerical guard
        H = (H + H.T) / 2.0
        # Covariance in STANDARDIZED parameterization (delta method)
        cov_std = np.linalg.pinv(H) / Phi.shape[0]  # divide by n since loss was a mean

        # Transform covariance to ORIGINAL scale (linear transformation):
        if self.standardize_X:
            # mapping: [b_orig, beta_orig] = T * [b_std, beta_std]
            p = Xs.shape[1]
            T = np.eye(p+1)
            # beta_orig = beta_std / std
            T[1:, 1:] = np.diag(1.0 / self._X_std_)
            # b_orig = b_std - sum_j (beta_std * mu_j / std_j)
            T[0, 1:] = - (self._X_mean_ / self._X_std_)
            cov = T @ cov_std @ T.T
        else:
            cov = cov_std

        se = np.sqrt(np.clip(np.diag(cov), 0, np.inf))
        est = np.r_[self.intercept_, self.coef_]
        z = 1.96
        lo = est - z * se
        hi = est + z * se
        names = ["intercept"] + [f"w{i+1}" for i in range(len(self.coef_))]
        return {"names": np.array(names, object),
                "estimate": est, "stderr": se, "lower": lo, "upper": hi}

    def plot_residuals_vs_features(self, X: np.ndarray, y: np.ndarray,
                                   feature_names: Optional[Iterable[str]] = None) -> None:
        """Creates one separate figure per feature: residual vs feature."""
        X = np.asarray(X, float)
        res = self.residuals(X, y)
        p = X.shape[1] if X.ndim > 1 else 1
        if X.ndim == 1: X = X[:, None]
        names = list(feature_names) if feature_names is not None else [f"x{i+1}" for i in range(p)]
        for j in range(p):
            plt.figure()
            plt.scatter(X[:, j], res, alpha=0.8)
            plt.axhline(0.0)
            plt.xlabel(names[j]); plt.ylabel("ŷ − y")
            plt.title(f"Residuals vs {names[j]}")

    @staticmethod
    def scatterplot_matrix(X: np.ndarray, y: np.ndarray, feature_names: List[str], target_name: str = "y") -> None:
        """Simple scatterplot matrix for Task 1 (requires pandas)."""
        assert _HAS_PANDAS, "pandas is required for scatterplot matrix."
        X = np.asarray(X, float)
        if X.ndim == 1: X = X[:, None]
        df = pd.DataFrame(X, columns=feature_names)
        df[target_name] = np.asarray(y, float).ravel()
        scatter_matrix(df, figsize=(8, 8), diagonal='hist')
        plt.suptitle("Scatterplot Matrix", y=1.02)


# ======================================================================
# BONUS: statsmodels GenericLikelihoodModel version (alternative impl)
# ======================================================================

if _HAS_SM:
    class _CauchyGLM(GenericLikelihoodModel):
        def __init__(self, endog, exog, c=1.0, **kw):
            self.c = float(c)
            super().__init__(endog, exog, **kw)

        def nloglikeobs(self, params):
            # params: [b, w1, ..., wp]
            yhat = self.exog @ params
            res = (self.endog - yhat) / self.c
            return (self.c**2 / 2.0) * np.log1p(res**2)

        def loglike(self, params):
            return -np.sum(self.nloglikeobs(params))

    def fit_cauchy_sm(X: np.ndarray, y: np.ndarray, c: float = 1.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fits Cauchy regression via statsmodels GenericLikelihoodModel.
        Returns (params, conf_int_95) where params = [intercept, w1..wp].
        """
        X = np.asarray(X, float); y = np.asarray(y, float).ravel()
        exog = _add_bias(X)
        mod = _CauchyGLM(y, exog, c=c)
        res = mod.fit(method='bfgs', disp=False)
        params = res.params
        ci = res.conf_int(alpha=0.05)  # 95%
        return params, ci
else:
    def fit_cauchy_sm(*args, **kwargs):
        raise RuntimeError("statsmodels is not available; cannot run the BONUS part.")


# ======================================================================
# (Optional) light plotting sugar for quick residual panel (not required)
# ======================================================================

def residual_panel(model: CauchyRegression, X: np.ndarray, y: np.ndarray,
                   feature_names: Optional[List[str]] = None) -> None:
    """Convenience wrapper to draw residuals vs each feature in a 2x2 grid if p==4."""
    X = np.asarray(X, float)
    if X.ndim == 1: X = X[:, None]
    p = X.shape[1]
    names = feature_names or [f"x{i+1}" for i in range(p)]
    res = model.residuals(X, y)

    if p == 4:
        fig, axs = plt.subplots(2, 2, figsize=(8, 6))
        axs = axs.ravel()
        for j in range(4):
            axs[j].scatter(X[:, j], res, alpha=0.8)
            axs[j].axhline(0.0)
            axs[j].set_xlabel(names[j]); axs[j].set_ylabel("ŷ − y")
            axs[j].set_title(f"Residuals vs {names[j]}")
        fig.tight_layout()
    else:
        # fallback: one-by-one
        for j in range(p):
            plt.figure()
            plt.scatter(X[:, j], res, alpha=0.8)
            plt.axhline(0.0)
            plt.xlabel(names[j]); plt.ylabel("ŷ − y")
            plt.title(f"Residuals vs {names[j]}")

