"""Population-weighted national daily series and day-of-year baseline statistics.

The pieces here are deliberately pure (NumPy / pandas in, pandas / scalars out)
so they can be unit-tested in isolation:

* :func:`doy_index`            - a leap-aligned 1..366 day-of-year index.
* :func:`weighted_mean`        - NaN-aware population-weighted mean.
* :func:`weighted_quantile`    - NaN-aware population-weighted quantile.
* :func:`national_daily_series`- collapse the per-city panel into one value/day.
* :func:`daily_climatology`    - per-doy median / P5 / P95 over the baseline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# A leap year used as the reference calendar so that every (month, day) maps to a
# stable 1..366 slot. 29 Feb -> 60, 1 Mar -> 61 in *every* year; non-leap years
# simply never populate slot 60.
_LEAP_REFERENCE_YEAR = 2000
_LEAP_REF_START = pd.Timestamp(f"{_LEAP_REFERENCE_YEAR}-01-01")


def doy_index(dates) -> np.ndarray:
    """Map dates to a calendar-aligned day-of-year index in ``1..366``.

    Unlike a raw ``dayofyear`` (which shifts every post-February day by one
    between leap and non-leap years), this projects each date onto a fixed leap
    reference calendar by ``(month, day)``. The result lines climatological days
    up by calendar date: 1 March is always 61, 31 December is always 366, and
    29 February is the only slot (60) that non-leap years leave empty.
    """
    dts = pd.DatetimeIndex(pd.to_datetime(dates))
    ref = pd.to_datetime(
        {
            "year": np.full(len(dts), _LEAP_REFERENCE_YEAR),
            "month": dts.month,
            "day": dts.day,
        }
    )
    return ((pd.DatetimeIndex(ref) - _LEAP_REF_START).days + 1).to_numpy()


def _clean(values, weights):
    """Drop NaN values and return aligned float arrays (values, weights)."""
    v = np.asarray(values, dtype=float)
    w = np.asarray(weights, dtype=float)
    mask = ~np.isnan(v)
    return v[mask], w[mask]


def weighted_mean(values, weights) -> float:
    """Population-weighted mean, ignoring NaN values and renormalising weights.

    Returns ``nan`` if every value is NaN or all surviving weights are zero.
    """
    v, w = _clean(values, weights)
    total = w.sum()
    if v.size == 0 or total == 0:
        return float("nan")
    return float(np.dot(v, w) / total)


def weighted_quantile(values, weights, q: float) -> float:
    """Weighted quantile (``q`` in ``[0, 1]``) using the standard
    cumulative-weight interpolation, ignoring NaN values.

    With equal weights this matches ``numpy.quantile(..., method="linear")``.
    """
    v, w = _clean(values, weights)
    if v.size == 0 or w.sum() == 0:
        return float("nan")
    order = np.argsort(v)
    v, w = v[order], w[order]
    cumw = np.cumsum(w)
    # Position of each sample on a 0..1 axis (Hazen-style plotting positions).
    pos = (cumw - 0.5 * w) / w.sum()
    if q <= pos[0]:
        return float(v[0])
    if q >= pos[-1]:
        return float(v[-1])
    return float(np.interp(q, pos, v))


def national_daily_series(
    panel: pd.DataFrame,
    weights: pd.Series,
    agg: str = "mean",
    value_col: str = "temperature_2m_mean",
) -> pd.DataFrame:
    """Collapse a long per-city panel into one population-weighted value per day.

    Parameters
    ----------
    panel
        Long frame with at least ``date``, ``city`` and ``value_col`` columns.
    weights
        Series indexed by city name giving the (un-normalised) population weight.
    agg
        ``"mean"`` for a population-weighted mean (the default, interpreted as the
        temperature experienced by the average resident) or ``"median"`` for a
        population-weighted median (routes through :func:`weighted_quantile`, more
        robust to an outlier city).

    Returns
    -------
    DataFrame with columns ``date`` and ``temp``.
    """
    if agg not in {"mean", "median"}:
        raise ValueError(f"agg must be 'mean' or 'median', got {agg!r}")

    w_lookup = weights.to_dict()

    def _collapse(group: pd.DataFrame) -> float:
        vals = group[value_col].to_numpy(dtype=float)
        wts = group["city"].map(w_lookup).to_numpy(dtype=float)
        if agg == "mean":
            return weighted_mean(vals, wts)
        return weighted_quantile(vals, wts, 0.5)

    out = (
        panel.groupby("date", sort=True)[[value_col, "city"]]
        .apply(_collapse)
        .rename("temp")
        .reset_index()
    )
    return out


def daily_climatology(
    series: pd.DataFrame,
    value_col: str = "temp",
    low: float = 5.0,
    high: float = 95.0,
) -> pd.DataFrame:
    """Per day-of-year median and P5/P95 band over whatever years ``series`` spans.

    The caller is responsible for restricting ``series`` to the baseline period
    (e.g. 1961-1990) first. Returns one row per day-of-year (1..366) with columns
    ``doy``, ``median``, ``p5``, ``p95`` and ``n`` (the number of baseline-year
    observations behind each day; day 366 has far fewer, coming only from leap
    years).
    """
    df = series.copy()
    df["doy"] = doy_index(df["date"])
    grouped = df.groupby("doy")[value_col]
    clim = pd.DataFrame(
        {
            "median": grouped.median(),
            "p5": grouped.quantile(low / 100.0),
            "p95": grouped.quantile(high / 100.0),
            "n": grouped.count(),
        }
    ).reset_index()
    return clim.sort_values("doy").reset_index(drop=True)
