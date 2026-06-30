"""One reusable entry point that runs (or loads from cache) the whole analysis.

Both ``run_analysis.py`` and the Quarto article call :func:`compute_all`, so the
numbers in the prose can never drift from the numbers in the pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from . import build, climatology as C, smoothing as S
from .locations import load_cities

SPAN_GRID = [0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50]
K_GRID = [1, 2, 3, 4, 5, 6]
DOY_AXIS = np.arange(1, 367)


def _products_exist(data_dir: Path) -> bool:
    needed = ["national_series", "climatology", "normals", "cv_span", "cv_K",
              "comparison_2025"]
    return all((data_dir / f"{n}.parquet").exists() for n in needed) and (
        data_dir / "results.json"
    ).exists()


def compute_all(data_dir: str | Path = "data", use_cache: bool = True,
                seed: int = 42) -> dict:
    """Return every product the article needs, computing and caching on first run.

    Keys: ``cities``, ``national``, ``climatology``, ``normals``, ``cv_span``,
    ``cv_K``, ``comparison``, ``results``.
    """
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if use_cache and _products_exist(data_dir):
        return {
            "cities": load_cities(),
            "national": pd.read_parquet(data_dir / "national_series.parquet"),
            "climatology": pd.read_parquet(data_dir / "climatology.parquet"),
            "normals": pd.read_parquet(data_dir / "normals.parquet"),
            "cv_span": pd.read_parquet(data_dir / "cv_span.parquet"),
            "cv_K": pd.read_parquet(data_dir / "cv_K.parquet"),
            "comparison": pd.read_parquet(data_dir / "comparison_2025.parquet"),
            "results": json.loads((data_dir / "results.json").read_text()),
        }

    cities = load_cities()
    panel, source = build.load_panel(cache_dir=data_dir)
    weights = cities.set_index("city")["weight"]

    national = C.national_daily_series(panel, weights, agg="mean")
    national["year"] = pd.DatetimeIndex(national["date"]).year
    national["doy"] = C.doy_index(national["date"])
    national.to_parquet(data_dir / "national_series.parquet", index=False)

    baseline, comp2025 = build.split_baseline_comparison(national)
    clim = C.daily_climatology(baseline)
    clim.to_parquet(data_dir / "climatology.parquet", index=False)

    doy_b = baseline["doy"].to_numpy()
    y_b = baseline["temp"].to_numpy()
    yr_b = baseline["year"].to_numpy()

    cv_span = S.cross_validate_span(doy_b, y_b, yr_b, spans=SPAN_GRID, n_splits=5, seed=seed)
    cv_K = S.cross_validate_K(doy_b, y_b, yr_b, Ks=K_GRID, n_splits=5, seed=seed)
    best_span = S.select_best_span(cv_span)
    best_K = S.select_best_K(cv_K)
    cv_span.to_parquet(data_dir / "cv_span.parquet", index=False)
    cv_K.to_parquet(data_dir / "cv_K.parquet", index=False)

    normal_loess = S.predict_loess(S.fit_loess(doy_b, y_b, span=best_span), DOY_AXIS)
    coeffs = S.fit_harmonic(doy_b, y_b, K=best_K)
    normal_harm = S.predict_harmonic(coeffs, DOY_AXIS, K=best_K)
    normals = pd.DataFrame({"doy": DOY_AXIS, "loess": normal_loess, "harmonic": normal_harm})
    normals.to_parquet(data_dir / "normals.parquet", index=False)

    comp = comp2025.merge(normals[["doy", "loess"]], on="doy", how="left")
    comp = comp.merge(clim[["doy", "median", "p5", "p95"]], on="doy", how="left")
    comp["anomaly"] = comp["temp"] - comp["loess"]
    comp = comp.sort_values("date").reset_index(drop=True)
    comp.to_parquet(data_dir / "comparison_2025.parquet", index=False)

    n_days = len(comp)
    warm_days = int((comp["anomaly"] > 0).sum())
    results = {
        "source": source,
        "n_cities": int(len(cities)),
        "best_span": float(best_span),
        "best_K": int(best_K),
        "span_grid": SPAN_GRID,
        "K_grid": K_GRID,
        "cv_span_best_rmse": float(cv_span["cv_rmse"].min()),
        "cv_K_best_rmse": float(cv_K["cv_rmse"].min()),
        "n_days_2025": n_days,
        "mean_anomaly_2025": round(float(comp["anomaly"].mean()), 3),
        "mean_temp_2025": round(float(comp["temp"].mean()), 3),
        "mean_normal": round(float(comp["loess"].mean()), 3),
        "days_above_p95": int((comp["temp"] > comp["p95"]).sum()),
        "days_below_p5": int((comp["temp"] < comp["p5"]).sum()),
        "warm_days": warm_days,
        "warm_day_share": round(100 * warm_days / n_days, 1),
        "max_loess_harmonic_gap": round(float(np.max(np.abs(normal_loess - normal_harm))), 3),
        "baseline_annual_mean": round(float(baseline["temp"].mean()), 3),
        "cv_span_worst_rmse": float(cv_span["cv_rmse"].max()),
        "cv_K_worst_rmse": float(cv_K["cv_rmse"].max()),
    }
    (data_dir / "results.json").write_text(json.dumps(results, indent=2))

    return {
        "cities": cities, "national": national, "climatology": clim,
        "normals": normals, "cv_span": cv_span, "cv_K": cv_K,
        "comparison": comp, "results": results,
    }
