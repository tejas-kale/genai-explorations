"""Loess and harmonic smoothing of the daily climatological normal, plus the
5-fold cross-validation (split *by year*) that re-derives Germany's own optimal
loess ``span`` and harmonic ``K``.

The intellectual core of the analysis lives here: a daily normal must be smooth
enough to suppress sampling noise (only ~30 baseline values per calendar day)
yet flexible enough to follow the true seasonal march. Rather than copy the
Spanish study's choices (span 0.16, K=4), we let leave-years-out cross-validation
choose both for the German annual cycle.

Pure, unit-tested helpers:
* :func:`harmonic_design`  - the sin/cos design matrix.
* :func:`year_folds`       - K-fold splits where whole years move together.

Fitting / CV:
* :func:`fit_harmonic` / :func:`predict_harmonic` (statsmodels OLS).
* :func:`fit_loess` / :func:`predict_loess` (skmisc.loess, which can predict).
* :func:`cross_validate_span` / :func:`cross_validate_K` and their selectors.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
from skmisc.loess import loess

# Annual period: a calendar year of ~365.25 days. Using the tropical-year length
# avoids a slow phase drift across the day-of-year axis.
DEFAULT_PERIOD = 365.25


# --------------------------------------------------------------------------- #
# Harmonic regression                                                         #
# --------------------------------------------------------------------------- #
def harmonic_design(doy, K: int, period: float = DEFAULT_PERIOD) -> np.ndarray:
    """Build the harmonic design matrix for ``K`` harmonics over day-of-year.

    Columns are ``[1, sin(2*pi*1*t/P), cos(2*pi*1*t/P), ..., sin(2*pi*K*t/P),
    cos(2*pi*K*t/P)]`` so the matrix has ``1 + 2K`` columns (just the intercept
    when ``K == 0``).
    """
    t = np.asarray(doy, dtype=float)
    cols = [np.ones_like(t)]
    for k in range(1, K + 1):
        ang = 2.0 * np.pi * k * t / period
        cols.append(np.sin(ang))
        cols.append(np.cos(ang))
    return np.column_stack(cols)


def fit_harmonic(doy, y, K: int, period: float = DEFAULT_PERIOD) -> np.ndarray:
    """Fit harmonic regression by OLS and return the coefficient vector."""
    X = harmonic_design(doy, K, period)
    model = sm.OLS(np.asarray(y, dtype=float), X).fit()
    return model.params


def predict_harmonic(coeffs, doy, K: int, period: float = DEFAULT_PERIOD) -> np.ndarray:
    """Predict from harmonic coefficients on new day-of-year points."""
    X = harmonic_design(doy, K, period)
    return X @ np.asarray(coeffs, dtype=float)


# --------------------------------------------------------------------------- #
# Loess                                                                       #
# --------------------------------------------------------------------------- #
def fit_loess(x, y, span: float, degree: int = 2) -> loess:
    """Fit a loess model. ``span`` is the fraction of points in each local window
    (R's ``loess`` ``span``); ``degree`` 2 matches R's default local quadratic."""
    model = loess(np.asarray(x, dtype=float), np.asarray(y, dtype=float),
                  span=span, degree=degree)
    model.fit()
    return model


def predict_loess(model: loess, xnew) -> np.ndarray:
    """Predict a fitted loess model on new points (the capability statsmodels'
    LOWESS lacks)."""
    pred = model.predict(np.asarray(xnew, dtype=float), stderror=False)
    return np.asarray(pred.values, dtype=float)


# --------------------------------------------------------------------------- #
# Cross-validation split BY YEAR                                              #
# --------------------------------------------------------------------------- #
def year_folds(years, n_splits: int = 5, seed: int = 0):
    """Partition rows into ``n_splits`` folds where *whole years* move together.

    Returns a list of ``(train_idx, test_idx)`` arrays. Each distinct year is
    held out in exactly one fold, so no year ever appears in both the training
    and test set of a fold. If there are fewer distinct years than ``n_splits``,
    only as many non-empty folds as there are years are produced.
    """
    years = np.asarray(years)
    uniq = np.unique(years)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(uniq)
    groups = np.array_split(perm, min(n_splits, len(uniq)))

    folds = []
    for group in groups:
        test_mask = np.isin(years, group)
        test_idx = np.where(test_mask)[0]
        train_idx = np.where(~test_mask)[0]
        folds.append((train_idx, test_idx))
    return folds


def _rmse(pred, actual) -> float:
    pred = np.asarray(pred, dtype=float)
    actual = np.asarray(actual, dtype=float)
    return float(np.sqrt(np.nanmean((pred - actual) ** 2)))


def cross_validate_span(
    doy, y, years, spans, n_splits: int = 5, seed: int = 0,
    period: float = DEFAULT_PERIOD,
) -> pd.DataFrame:
    """Leave-years-out CV RMSE for each loess ``span`` in the grid.

    For each fold the loess is fit on the training years' (doy, temperature)
    pairs and used to predict the held-out years' daily temperatures; the RMSE is
    averaged across folds. Returns columns ``span`` and ``cv_rmse``.
    """
    doy = np.asarray(doy, dtype=float)
    y = np.asarray(y, dtype=float)
    folds = year_folds(years, n_splits=n_splits, seed=seed)

    rows = []
    for span in spans:
        fold_rmse = []
        for train_idx, test_idx in folds:
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            model = fit_loess(doy[train_idx], y[train_idx], span=span)
            pred = predict_loess(model, doy[test_idx])
            fold_rmse.append(_rmse(pred, y[test_idx]))
        rows.append({"span": span, "cv_rmse": float(np.mean(fold_rmse))})
    return pd.DataFrame(rows)


def cross_validate_K(
    doy, y, years, Ks, n_splits: int = 5, seed: int = 0,
    period: float = DEFAULT_PERIOD,
) -> pd.DataFrame:
    """Leave-years-out CV RMSE for each harmonic order ``K`` in the grid.

    Returns columns ``K`` and ``cv_rmse``.
    """
    doy = np.asarray(doy, dtype=float)
    y = np.asarray(y, dtype=float)
    folds = year_folds(years, n_splits=n_splits, seed=seed)

    rows = []
    for K in Ks:
        fold_rmse = []
        for train_idx, test_idx in folds:
            if len(test_idx) == 0 or len(train_idx) == 0:
                continue
            coeffs = fit_harmonic(doy[train_idx], y[train_idx], K=K, period=period)
            pred = predict_harmonic(coeffs, doy[test_idx], K=K, period=period)
            fold_rmse.append(_rmse(pred, y[test_idx]))
        rows.append({"K": K, "cv_rmse": float(np.mean(fold_rmse))})
    return pd.DataFrame(rows)


def select_best_span(cv: pd.DataFrame) -> float:
    """Return the ``span`` with the lowest CV RMSE."""
    return float(cv.loc[cv["cv_rmse"].idxmin(), "span"])


def select_best_K(cv: pd.DataFrame) -> int:
    """Return the harmonic order ``K`` with the lowest CV RMSE."""
    return int(cv.loc[cv["cv_rmse"].idxmin(), "K"])
