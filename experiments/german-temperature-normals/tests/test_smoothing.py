"""Red-green tests for the pure functions in ``smoothing``.

Covers the harmonic design matrix, the loess/harmonic fit-and-predict contract,
and the 5-fold cross-validation *split by year* that re-derives Germany's own
optimal loess span and harmonic K.
"""

from __future__ import annotations

import numpy as np
import pytest

from german_normals import smoothing as S


# --------------------------------------------------------------------------- #
# harmonic_design                                                             #
# --------------------------------------------------------------------------- #
def test_harmonic_design_shape_includes_intercept():
    doy = np.arange(1, 11)
    X = S.harmonic_design(doy, K=3)
    # intercept + K sine + K cosine = 1 + 2K columns
    assert X.shape == (10, 1 + 2 * 3)


def test_harmonic_design_first_column_is_intercept():
    X = S.harmonic_design(np.arange(1, 6), K=2)
    assert np.allclose(X[:, 0], 1.0)


def test_harmonic_design_k0_is_intercept_only():
    X = S.harmonic_design(np.arange(1, 6), K=0)
    assert X.shape == (5, 1)
    assert np.allclose(X, 1.0)


def test_harmonic_design_period_values():
    # At doy == period the first harmonic returns to (sin=0, cos=1).
    period = 365.25
    X = S.harmonic_design(np.array([period]), K=1, period=period)
    # columns: [1, sin(2pi), cos(2pi)]
    assert X[0, 1] == pytest.approx(0.0, abs=1e-9)
    assert X[0, 2] == pytest.approx(1.0, abs=1e-9)


def test_harmonic_design_recovers_known_signal():
    # A pure 1-harmonic signal must be fit essentially perfectly with K>=1.
    doy = np.arange(1, 367)
    period = 366.0
    y = 3.0 + 2.0 * np.sin(2 * np.pi * doy / period) + 1.5 * np.cos(2 * np.pi * doy / period)
    coeffs = S.fit_harmonic(doy, y, K=1, period=period)
    pred = S.predict_harmonic(coeffs, doy, K=1, period=period)
    assert np.allclose(pred, y, atol=1e-6)


# --------------------------------------------------------------------------- #
# loess fit/predict                                                           #
# --------------------------------------------------------------------------- #
def test_loess_predicts_on_new_points():
    x = np.linspace(0, 10, 200)
    y = np.sin(x) + 0.01 * np.arange(200) % 1
    model = S.fit_loess(x, y, span=0.3)
    xnew = np.array([1.0, 5.0, 9.0])
    pred = S.predict_loess(model, xnew)
    assert pred.shape == (3,)
    assert np.all(np.isfinite(pred))


def test_loess_smaller_span_tracks_data_more_closely():
    rng = np.random.default_rng(1)
    x = np.linspace(0, 10, 300)
    y = np.sin(x) + rng.normal(0, 0.3, 300)
    tight = S.predict_loess(S.fit_loess(x, y, span=0.1), x)
    loose = S.predict_loess(S.fit_loess(x, y, span=0.9), x)
    # Tighter span -> smaller residuals to the noisy data.
    assert np.mean((tight - y) ** 2) < np.mean((loose - y) ** 2)


# --------------------------------------------------------------------------- #
# year_folds: 5-fold CV split BY YEAR                                         #
# --------------------------------------------------------------------------- #
def test_year_folds_partition_every_row_once():
    years = np.repeat(np.arange(1961, 1991), 3)  # 30 years, 3 rows each
    folds = S.year_folds(years, n_splits=5, seed=0)
    assert len(folds) == 5
    test_counts = np.zeros(len(years), dtype=int)
    for _, test_idx in folds:
        test_counts[test_idx] += 1
    # Each row is in exactly one test fold.
    assert np.all(test_counts == 1)


def test_year_folds_no_year_spans_train_and_test():
    years = np.repeat(np.arange(2000, 2010), 4)
    folds = S.year_folds(years, n_splits=5, seed=3)
    for train_idx, test_idx in folds:
        train_years = set(years[train_idx])
        test_years = set(years[test_idx])
        assert train_years.isdisjoint(test_years)


def test_year_folds_each_year_held_out_exactly_once():
    years = np.repeat(np.arange(1961, 1991), 2)
    folds = S.year_folds(years, n_splits=5, seed=7)
    held_out = []
    for _, test_idx in folds:
        held_out.extend(sorted(set(years[test_idx])))
    assert sorted(held_out) == sorted(set(years))


def test_year_folds_more_splits_than_years_is_safe():
    years = np.array([2001, 2001, 2002, 2002])  # only 2 distinct years
    folds = S.year_folds(years, n_splits=5, seed=0)
    # Cannot make more non-empty folds than there are years.
    nonempty = [f for f in folds if len(f[1]) > 0]
    assert len(nonempty) == 2


def test_year_folds_train_and_test_indices_disjoint():
    years = np.repeat(np.arange(1961, 1991), 5)
    for train_idx, test_idx in S.year_folds(years, n_splits=5, seed=1):
        assert set(train_idx).isdisjoint(set(test_idx))
        assert len(train_idx) + len(test_idx) == len(years)


# --------------------------------------------------------------------------- #
# cross-validation drivers return a best value                               #
# --------------------------------------------------------------------------- #
def test_cv_select_span_returns_value_in_grid():
    rng = np.random.default_rng(0)
    years = np.repeat(np.arange(1961, 1991), 366)
    doy = np.tile(np.arange(1, 367), 30)
    signal = 10 + 12 * np.sin(2 * np.pi * doy / 365.25 - 1.3)
    y = signal + rng.normal(0, 2, len(doy))
    grid = [0.1, 0.3, 0.6]
    res = S.cross_validate_span(doy, y, years, spans=grid, n_splits=5, seed=0)
    assert set(res["span"]) == set(grid)
    best = S.select_best_span(res)
    assert best in grid


def test_cv_select_K_returns_value_in_grid():
    rng = np.random.default_rng(0)
    years = np.repeat(np.arange(1961, 1991), 366)
    doy = np.tile(np.arange(1, 367), 30)
    signal = 10 + 12 * np.sin(2 * np.pi * doy / 365.25 - 1.3)
    y = signal + rng.normal(0, 2, len(doy))
    grid = [1, 2, 3, 4]
    res = S.cross_validate_K(doy, y, years, Ks=grid, n_splits=5, seed=0)
    assert set(res["K"]) == set(grid)
    best = S.select_best_K(res)
    assert best in grid
