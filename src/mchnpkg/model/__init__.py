# __init__.py
from .regression import (
    LinearRegression,
    CauchyRegression,
    fit_cauchy_sm,          # BONUS helper (statsmodels) – raises if statsmodels missing
    train_test_split,
    standardize,
    mae, mse, rmse, r2_score,
    residual_panel,
)

__all__ = [
    "LinearRegression",
    "CauchyRegression",
    "fit_cauchy_sm",
    "train_test_split",
    "standardize",
    "mae", "mse", "rmse", "r2_score",
    "residual_panel",
]
