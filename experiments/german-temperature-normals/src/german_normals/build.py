"""Assemble the analysis dataset: try the real Open-Meteo archive, and only if
the network is blocked fall back to the clearly-labelled synthetic generator.

This is the one place that decides *real vs. synthetic*. Everything downstream
(climatology, smoothing, figure, the article) consumes the panel it returns,
together with the ``source`` string so the article can state plainly which it is.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from . import fetch, synthetic
from .locations import load_cities

# Fixed 1961-1990 reference period (NOT the WMO operational 1991-2020), so the
# climate-change signal is never baked into the reference line. Plus full 2025.
BASELINE_PERIOD = ("1961-01-01", "1990-12-31")
COMPARISON_PERIOD = ("2025-01-01", "2025-12-31")
PERIODS = [BASELINE_PERIOD, COMPARISON_PERIOD]
BASELINE_YEARS = (1961, 1990)
COMPARISON_YEAR = 2025


def _can_reach_openmeteo(timeout: float = 8.0) -> bool:
    """Quick probe so we do not hang on 56 doomed requests when egress is blocked."""
    try:
        r = requests.get(
            fetch.ARCHIVE_URL,
            params={
                "latitude": 52.52,
                "longitude": 13.41,
                "start_date": "2020-01-01",
                "end_date": "2020-01-02",
                "daily": fetch.DAILY_VARIABLE,
                "timezone": "auto",
            },
            timeout=timeout,
        )
        return r.status_code == 200
    except Exception:
        return False


def load_panel(
    cache_dir: str | Path = "data",
    prefer_real: bool = True,
    force_synthetic: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Return ``(panel, source)`` where ``source`` is ``"open-meteo"`` or
    ``"synthetic"``.

    A previously built panel of either kind is reused from ``cache_dir``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    real_cache = cache_dir / "panel_open-meteo.parquet"
    synth_cache = cache_dir / "panel_synthetic.parquet"

    cities = load_cities()

    if not force_synthetic and real_cache.exists():
        return pd.read_parquet(real_cache), "open-meteo"

    if not force_synthetic and prefer_real and _can_reach_openmeteo():
        panel = fetch.fetch_panel(cities, PERIODS, cache_dir=cache_dir)
        panel.to_parquet(real_cache, index=False)
        return panel, "open-meteo"

    if synth_cache.exists():
        return pd.read_parquet(synth_cache), "synthetic"
    panel = synthetic.generate_panel(cities, PERIODS)
    panel.to_parquet(synth_cache, index=False)
    return panel, "synthetic"


def add_calendar(panel: pd.DataFrame) -> pd.DataFrame:
    """Attach ``year`` and (leap-aligned) ``doy`` columns."""
    from .climatology import doy_index

    out = panel.copy()
    out["year"] = pd.DatetimeIndex(out["date"]).year
    out["doy"] = doy_index(out["date"])
    return out


def split_baseline_comparison(series: pd.DataFrame):
    """Split a national daily series into (baseline 1961-1990, comparison 2025)."""
    years = pd.DatetimeIndex(series["date"]).year
    lo, hi = BASELINE_YEARS
    baseline = series[(years >= lo) & (years <= hi)].copy()
    comparison = series[years == COMPARISON_YEAR].copy()
    return baseline, comparison
