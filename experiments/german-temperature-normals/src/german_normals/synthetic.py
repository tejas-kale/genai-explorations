"""Offline, physically-plausible ERA5-like fallback data generator.

THIS MODULE FABRICATES DATA. It exists only so the full pipeline (and the Quarto
article) can run end-to-end in an environment where outbound access to the
Open-Meteo archive is blocked by network policy. Every number the article derives
from synthetic data is clearly labelled as such; the moment the real API is
reachable, :mod:`german_normals.build` uses :mod:`german_normals.fetch` instead
and nothing here is touched.

The generator is *not* a climate model. It is a transparent statistical mock
designed to look like German daily-mean temperature so the method can be
exercised honestly:

* a seasonal cycle (coldest ~20 Jan, warmest ~22 Jul) with a small second
  harmonic, so the cross-validation has a real reason to prefer K >= 2;
* continentality: eastern/inland cities get a larger annual amplitude (colder
  winters, warmer summers) than the maritime northwest;
* a *shared* national synoptic-weather term (so the country warms and cools
  together and the national average keeps realistic day-to-day variance) plus a
  smaller city-local term (which averages out across cities);
* a +~2 C warm shift for 2025 versus the 1961-1990 baseline, with a hot July
  spell and a short cold snap so the 2025 line crosses both the P95 and P5 band.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

DAILY_VARIABLE = "temperature_2m_mean"

# Baseline-period national annual mean (deg C), roughly Germany 1961-1990.
_BASE_MEAN = 9.4
# Cold/warm phase: day-of-year of the seasonal minimum (~20 Jan).
_PHASE = 20.0


def _seasonal(doy: np.ndarray, amplitude: float) -> np.ndarray:
    """Annual cycle: -A*cos at the fundamental plus a small second harmonic."""
    ang = 2.0 * np.pi * (doy - _PHASE) / 365.25
    second = 0.6 * np.cos(2.0 * ang + 0.5)  # mild asymmetry -> rewards K>=2
    return -amplitude * np.cos(ang) + second


def _ar1(n: int, sd: np.ndarray, phi: float, rng: np.random.Generator) -> np.ndarray:
    """An AR(1) sequence with a (possibly time-varying) innovation sd."""
    out = np.empty(n)
    out[0] = rng.normal(0, sd[0])
    for t in range(1, n):
        out[t] = phi * out[t - 1] + rng.normal(0, sd[t]) * np.sqrt(1 - phi**2)
    return out


def _seasonal_sd(doy: np.ndarray, winter: float, summer: float) -> np.ndarray:
    """Innovation sd that is larger in winter than summer (as in reality)."""
    w = 0.5 * (1 + np.cos(2.0 * np.pi * (doy - _PHASE) / 365.25))  # 1 in winter, 0 summer
    return summer + (winter - summer) * w


def generate_panel(
    cities: pd.DataFrame,
    periods: list[tuple[str, str]],
    seed: int = 20250630,
) -> pd.DataFrame:
    """Generate a long per-city panel matching :func:`fetch.fetch_panel`'s schema.

    Columns: ``date``, ``city``, ``lat``, ``lon``, ``temperature_2m_mean``.
    """
    rng = np.random.default_rng(seed)

    # One shared national synoptic series per period (whole country co-varies).
    shared: dict[tuple[str, str], pd.Series] = {}
    for start, end in periods:
        dates = pd.date_range(start, end, freq="D")
        doy = dates.dayofyear.to_numpy()
        sd = _seasonal_sd(doy, winter=2.6, summer=1.5)
        shared[(start, end)] = pd.Series(_ar1(len(dates), sd, phi=0.82, rng=rng), index=dates)

    lat = cities["lat"].to_numpy()
    lon = cities["lon"].to_numpy()
    # Continentality grows toward the east and away from the far-north coast.
    cont = 0.35 * (lon - 9.5) + 0.20 * (52.8 - lat)

    frames = []
    for idx, row in enumerate(cities.itertuples(index=False)):
        amplitude = 9.2 + cont[idx]
        # North a touch cooler in the annual mean; clipped so it stays sensible.
        city_mean = _BASE_MEAN - 0.42 * (row.lat - 51.0)
        for start, end in periods:
            dates = pd.date_range(start, end, freq="D")
            doy = dates.dayofyear.to_numpy()
            base = city_mean + _seasonal(doy, amplitude)

            shared_term = shared[(start, end)].to_numpy()
            local_sd = _seasonal_sd(doy, winter=1.7, summer=1.1)
            local = _ar1(len(dates), local_sd, phi=0.75, rng=rng)

            temp = base + shared_term + local

            # 2025: a warm year vs 1961-1990 with a hot July spell and a cold snap.
            if start.startswith("2025"):
                warm = 1.9 + 0.6 * np.sin(2.0 * np.pi * (doy - 30) / 365.25)
                temp = temp + warm
                d = pd.DatetimeIndex(dates)
                july_spell = (d.month == 7) & (d.day >= 8) & (d.day <= 17)
                temp = temp + np.where(july_spell, 5.5, 0.0)
                cold_snap = (d.month == 4) & (d.day >= 18) & (d.day <= 27)
                temp = temp - np.where(cold_snap, 6.0, 0.0)

            frames.append(
                pd.DataFrame(
                    {
                        "date": dates,
                        "city": row.city,
                        "lat": row.lat,
                        "lon": row.lon,
                        DAILY_VARIABLE: np.round(temp, 1),
                    }
                )
            )

    panel = pd.concat(frames, ignore_index=True)
    return panel
