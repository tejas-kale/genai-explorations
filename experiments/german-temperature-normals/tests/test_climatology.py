"""Red-green tests for the pure functions in ``climatology``.

These were written before the implementation. They pin down the leap-year
day-of-year convention, NaN-aware weighting, single-city behaviour and the
day-of-year aggregation that the rest of the pipeline relies on.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from german_normals import climatology as C


# --------------------------------------------------------------------------- #
# doy_index: a leap-aligned 1..366 day-of-year index                          #
# --------------------------------------------------------------------------- #
def test_doy_index_basic_new_year():
    assert C.doy_index(pd.to_datetime(["2021-01-01"]))[0] == 1


def test_doy_index_leap_day_is_60():
    # 29 Feb only exists in leap years and must land on index 60.
    assert C.doy_index(pd.to_datetime(["2020-02-29"]))[0] == 60


def test_doy_index_is_calendar_aligned_across_leap_and_nonleap():
    # 1 March must map to the SAME index (61) in both a leap and a non-leap
    # year, so that climatological days line up by calendar date. In a
    # non-leap year index 60 (29 Feb) is simply never populated.
    leap = C.doy_index(pd.to_datetime(["2020-03-01"]))[0]
    nonleap = C.doy_index(pd.to_datetime(["2021-03-01"]))[0]
    assert leap == nonleap == 61


def test_doy_index_feb28_nonleap_is_59():
    assert C.doy_index(pd.to_datetime(["2021-02-28"]))[0] == 59


def test_doy_index_dec31_is_366_in_both_year_types():
    # Because we map onto a leap reference calendar, 31 Dec is 366 always.
    assert C.doy_index(pd.to_datetime(["2020-12-31"]))[0] == 366
    assert C.doy_index(pd.to_datetime(["2021-12-31"]))[0] == 366


def test_doy_index_vectorised_length_and_range():
    dates = pd.date_range("2019-01-01", "2021-12-31", freq="D")
    doy = C.doy_index(dates)
    assert len(doy) == len(dates)
    assert doy.min() >= 1 and doy.max() <= 366


# --------------------------------------------------------------------------- #
# weighted_mean: NaN-aware, renormalising, single-city                        #
# --------------------------------------------------------------------------- #
def test_weighted_mean_equal_weights_is_plain_mean():
    v = np.array([10.0, 20.0, 30.0])
    w = np.array([1.0, 1.0, 1.0])
    assert C.weighted_mean(v, w) == pytest.approx(20.0)


def test_weighted_mean_respects_weights():
    v = np.array([0.0, 10.0])
    w = np.array([3.0, 1.0])
    assert C.weighted_mean(v, w) == pytest.approx(2.5)


def test_weighted_mean_drops_nan_and_renormalises():
    v = np.array([10.0, np.nan, 30.0])
    w = np.array([1.0, 5.0, 1.0])
    # The NaN city (with the big weight) is dropped; remaining two are equal.
    assert C.weighted_mean(v, w) == pytest.approx(20.0)


def test_weighted_mean_single_city_returns_that_value():
    assert C.weighted_mean(np.array([12.3]), np.array([0.7])) == pytest.approx(12.3)


def test_weighted_mean_all_nan_is_nan():
    out = C.weighted_mean(np.array([np.nan, np.nan]), np.array([1.0, 2.0]))
    assert np.isnan(out)


# --------------------------------------------------------------------------- #
# weighted_quantile                                                            #
# --------------------------------------------------------------------------- #
def test_weighted_quantile_median_equal_weights():
    v = np.array([1.0, 2.0, 3.0, 4.0])
    w = np.ones(4)
    assert C.weighted_quantile(v, w, 0.5) == pytest.approx(2.5)


def test_weighted_quantile_extremes():
    v = np.array([5.0, 1.0, 3.0])
    w = np.ones(3)
    assert C.weighted_quantile(v, w, 0.0) == pytest.approx(1.0)
    assert C.weighted_quantile(v, w, 1.0) == pytest.approx(5.0)


def test_weighted_quantile_weight_shifts_result():
    v = np.array([0.0, 10.0])
    # Heavily weight the low value -> median pulled toward 0.
    low = C.weighted_quantile(v, np.array([9.0, 1.0]), 0.5)
    assert low < 5.0


def test_weighted_quantile_single_value():
    assert C.weighted_quantile(np.array([7.0]), np.array([1.0]), 0.5) == pytest.approx(7.0)


def test_weighted_quantile_drops_nan():
    v = np.array([np.nan, 2.0, 4.0])
    w = np.ones(3)
    assert C.weighted_quantile(v, w, 0.5) == pytest.approx(3.0)


# --------------------------------------------------------------------------- #
# national_daily_series: collapse cities -> one value per date                #
# --------------------------------------------------------------------------- #
def _toy_panel():
    dates = pd.to_datetime(["2025-01-01", "2025-01-01", "2025-01-02", "2025-01-02"])
    return pd.DataFrame(
        {
            "date": dates,
            "city": ["A", "B", "A", "B"],
            "temperature_2m_mean": [0.0, 10.0, 5.0, 15.0],
        }
    )


def test_national_daily_series_population_weighted_mean():
    panel = _toy_panel()
    weights = pd.Series({"A": 3.0, "B": 1.0})
    out = C.national_daily_series(panel, weights, agg="mean")
    out = out.sort_values("date").reset_index(drop=True)
    # day1: (3*0 + 1*10)/4 = 2.5 ; day2: (3*5 + 1*15)/4 = 7.5
    assert out.loc[0, "temp"] == pytest.approx(2.5)
    assert out.loc[1, "temp"] == pytest.approx(7.5)
    assert list(out["date"].dt.day) == [1, 2]


def test_national_daily_series_single_city():
    dates = pd.to_datetime(["2025-01-01", "2025-01-02"])
    panel = pd.DataFrame(
        {"date": dates, "city": ["A", "A"], "temperature_2m_mean": [3.0, 4.0]}
    )
    out = C.national_daily_series(panel, pd.Series({"A": 1.0}), agg="mean")
    assert out.sort_values("date")["temp"].tolist() == pytest.approx([3.0, 4.0])


def test_national_daily_series_median_path_uses_weighted_quantile():
    # Three cities so the weighted median is clearly distinct from the mean:
    # values [0, 1, 100] -> mean ~33.7 but median ~1, robust to the outlier.
    dates = pd.to_datetime(["2025-01-01"] * 3)
    panel = pd.DataFrame(
        {
            "date": dates,
            "city": ["A", "B", "C"],
            "temperature_2m_mean": [0.0, 1.0, 100.0],
        }
    )
    weights = pd.Series({"A": 1.0, "B": 1.0, "C": 1.0})
    med = C.national_daily_series(panel, weights, agg="median")["temp"].iloc[0]
    mean = C.national_daily_series(panel, weights, agg="mean")["temp"].iloc[0]
    assert med < 5.0  # median ignores the 100 outlier
    assert mean > 30.0  # mean is dragged up by it


# --------------------------------------------------------------------------- #
# daily_climatology: per-doy median / P5 / P95 over baseline years            #
# --------------------------------------------------------------------------- #
def test_daily_climatology_shapes_and_columns():
    dates = pd.date_range("1961-01-01", "1990-12-31", freq="D")
    rng = np.random.default_rng(0)
    temp = 10 + 10 * np.sin(2 * np.pi * dates.dayofyear / 365.25) + rng.normal(0, 1, len(dates))
    series = pd.DataFrame({"date": dates, "temp": temp})
    clim = C.daily_climatology(series)
    # Every calendar day 1..366 represented (366 only from leap years).
    assert set(clim["doy"]) == set(range(1, 367))
    for col in ("median", "p5", "p95", "n"):
        assert col in clim.columns
    assert (clim["p5"] <= clim["median"]).all()
    assert (clim["median"] <= clim["p95"]).all()


def test_leap_day_slot_has_fewer_observations():
    # Because dates are projected onto a leap reference calendar, 31 Dec is
    # always slot 366 (so it gets all 30 years). The only under-sampled slot is
    # 60 = 29 Feb, populated solely by the 7 leap years in 1961-1990.
    dates = pd.date_range("1961-01-01", "1990-12-31", freq="D")
    series = pd.DataFrame({"date": dates, "temp": np.ones(len(dates))})
    clim = C.daily_climatology(series).set_index("doy")
    assert clim.loc[366, "n"] == 30
    assert clim.loc[60, "n"] < clim.loc[200, "n"]
    assert clim.loc[60, "n"] == 7  # leap years 1964,68,72,76,80,84,88
