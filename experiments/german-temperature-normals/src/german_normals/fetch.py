"""Download daily mean temperature from the Open-Meteo Historical Archive API.

One archive call per city per period, cached to ``data/*.parquet`` so a re-run is
instant and offline. Includes retry with exponential backoff for rate limits
(HTTP 429) and transient server/network errors.

The archive endpoint serves ERA5 reanalysis (free, no API key). Unlike station
records, reanalysis is spatially complete and gap-free, which is exactly what we
want for a consistent national baseline.

    https://archive-api.open-meteo.com/v1/archive
        ?latitude=..&longitude=..&start_date=..&end_date=..
        &daily=temperature_2m_mean&timezone=auto

Note: in some sandboxed/offline environments outbound access to the Open-Meteo
hosts is blocked by network policy. In that case the higher-level dataset builder
(:mod:`german_normals.build`) transparently falls back to
:mod:`german_normals.synthetic`. This module never fabricates data: it only ever
returns what the API actually served (or a previously cached real response).
"""

from __future__ import annotations

import hashlib
import time
from pathlib import Path

import pandas as pd
import requests

ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
DAILY_VARIABLE = "temperature_2m_mean"


def _cache_path(cache_dir: Path, lat: float, lon: float, start: str, end: str) -> Path:
    key = f"{lat:.4f}_{lon:.4f}_{start}_{end}_{DAILY_VARIABLE}"
    digest = hashlib.md5(key.encode()).hexdigest()[:10]
    return Path(cache_dir) / f"omc_{lat:.2f}_{lon:.2f}_{start[:4]}_{end[:4]}_{digest}.parquet"


def fetch_city(
    lat: float,
    lon: float,
    start_date: str,
    end_date: str,
    cache_dir: str | Path = "data",
    session: requests.Session | None = None,
    max_retries: int = 5,
    backoff_base: float = 2.0,
    timeout: int = 60,
) -> pd.DataFrame:
    """Fetch one city's daily mean temperature for ``[start_date, end_date]``.

    Returns a DataFrame with columns ``date`` (datetime64) and
    ``temperature_2m_mean`` (float, degrees C). Responses are cached to parquet;
    a cached file short-circuits the network call.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    path = _cache_path(cache_dir, lat, lon, start_date, end_date)
    if path.exists():
        return pd.read_parquet(path)

    sess = session or requests.Session()
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "daily": DAILY_VARIABLE,
        "timezone": "auto",
    }

    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = sess.get(ARCHIVE_URL, params=params, timeout=timeout)
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"retryable status {resp.status_code}")
            resp.raise_for_status()
            payload = resp.json()
            daily = payload["daily"]
            df = pd.DataFrame(
                {
                    "date": pd.to_datetime(daily["time"]),
                    DAILY_VARIABLE: pd.to_numeric(daily[DAILY_VARIABLE], errors="coerce"),
                }
            )
            df.to_parquet(path, index=False)
            return df
        except Exception as err:  # noqa: BLE001 - retry on any transient failure
            last_err = err
            if attempt < max_retries - 1:
                time.sleep(backoff_base ** attempt)  # 1, 2, 4, 8, 16 s
    raise RuntimeError(
        f"Open-Meteo fetch failed for ({lat}, {lon}) {start_date}..{end_date}: {last_err}"
    )


def fetch_panel(
    cities: pd.DataFrame,
    periods: list[tuple[str, str]],
    cache_dir: str | Path = "data",
    session: requests.Session | None = None,
    progress: bool = True,
) -> pd.DataFrame:
    """Fetch every city over every period and return one long tidy panel.

    Columns: ``date``, ``city``, ``lat``, ``lon``, ``temperature_2m_mean``.
    ``cities`` must have ``city``, ``lat`` and ``lon`` columns.
    """
    sess = session or requests.Session()
    frames = []
    rows = list(cities.itertuples(index=False))
    for i, row in enumerate(rows):
        for start, end in periods:
            df = fetch_city(row.lat, row.lon, start, end, cache_dir=cache_dir, session=sess)
            df = df.assign(city=row.city, lat=row.lat, lon=row.lon)
            frames.append(df)
        if progress:
            print(f"  fetched {i + 1}/{len(rows)}: {row.city}")
    panel = pd.concat(frames, ignore_index=True)
    return panel[["date", "city", "lat", "lon", DAILY_VARIABLE]]
