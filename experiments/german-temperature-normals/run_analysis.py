"""End-to-end driver: build the dataset, choose the loess span and harmonic K by
leave-years-out cross-validation, compute the 2025-vs-normal results, and cache
every product the Quarto article and figure consume.

Run from the project root:  python run_analysis.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent / "src"))

from german_normals import build, climatology as C, smoothing as S  # noqa: E402
from german_normals.locations import load_cities  # noqa: E402

DATA = Path(__file__).parent / "data"
DATA.mkdir(exist_ok=True)

SPAN_GRID = [0.08, 0.12, 0.16, 0.20, 0.25, 0.30, 0.40, 0.50]
K_GRID = [1, 2, 3, 4, 5, 6]


def main() -> None:
    cities = load_cities()
    print(f"[1/6] {len(cities)} cities; loading panel ...")
    panel, source = build.load_panel(cache_dir=DATA)
    print(f"      data source: {source}  ({len(panel):,} city-day rows)")

    print("[2/6] population-weighted national daily series ...")
    weights = cities.set_index("city")["weight"]
    national = C.national_daily_series(panel, weights, agg="mean")
    national["year"] = pd.DatetimeIndex(national["date"]).year
    national["doy"] = C.doy_index(national["date"])
    national.to_parquet(DATA / "national_series.parquet", index=False)

    baseline, comp2025 = build.split_baseline_comparison(national)
    print(f"      baseline rows {len(baseline):,} (1961-1990); 2025 rows {len(comp2025)}")

    print("[3/6] day-of-year climatology (median / P5 / P95) over 1961-1990 ...")
    clim = C.daily_climatology(baseline)
    clim.to_parquet(DATA / "climatology.parquet", index=False)

    print("[4/6] 5-fold CV by year -> loess span ...")
    cv_span = S.cross_validate_span(
        baseline["doy"].to_numpy(), baseline["temp"].to_numpy(),
        baseline["year"].to_numpy(), spans=SPAN_GRID, n_splits=5, seed=42,
    )
    best_span = S.select_best_span(cv_span)
    cv_span.to_parquet(DATA / "cv_span.parquet", index=False)
    print(cv_span.to_string(index=False))
    print(f"      best span = {best_span}")

    print("[5/6] 5-fold CV by year -> harmonic K ...")
    cv_K = S.cross_validate_K(
        baseline["doy"].to_numpy(), baseline["temp"].to_numpy(),
        baseline["year"].to_numpy(), Ks=K_GRID, n_splits=5, seed=42,
    )
    best_K = S.select_best_K(cv_K)
    cv_K.to_parquet(DATA / "cv_K.parquet", index=False)
    print(cv_K.to_string(index=False))
    print(f"      best K = {best_K}")

    print("[6/6] fit final normals + 2025 comparison ...")
    doy_axis = np.arange(1, 367)
    loess_model = S.fit_loess(baseline["doy"].to_numpy(), baseline["temp"].to_numpy(),
                              span=best_span)
    normal_loess = S.predict_loess(loess_model, doy_axis)
    coeffs = S.fit_harmonic(baseline["doy"].to_numpy(), baseline["temp"].to_numpy(), K=best_K)
    normal_harm = S.predict_harmonic(coeffs, doy_axis, K=best_K)

    normals = pd.DataFrame({"doy": doy_axis, "loess": normal_loess, "harmonic": normal_harm})
    normals.to_parquet(DATA / "normals.parquet", index=False)

    # ---- 2025 vs the smoothed (loess) normal -----------------------------
    comp = comp2025.merge(normals[["doy", "loess"]], on="doy", how="left")
    comp = comp.merge(clim[["doy", "median", "p5", "p95"]], on="doy", how="left")
    comp["anomaly"] = comp["temp"] - comp["loess"]
    comp = comp.sort_values("date").reset_index(drop=True)
    comp.to_parquet(DATA / "comparison_2025.parquet", index=False)

    n_days = len(comp)
    mean_anom = float(comp["anomaly"].mean())
    days_above_p95 = int((comp["temp"] > comp["p95"]).sum())
    days_below_p5 = int((comp["temp"] < comp["p5"]).sum())
    warm_days = int((comp["anomaly"] > 0).sum())
    mean_2025 = float(comp["temp"].mean())
    mean_normal = float(comp["loess"].mean())
    # difference of the loess and harmonic normals (the smoothing question)
    max_normal_gap = float(np.max(np.abs(normal_loess - normal_harm)))
    rmse_improvement = float(
        (cv_span["cv_rmse"].max() - cv_span["cv_rmse"].min()) / cv_span["cv_rmse"].max() * 100
    )

    results = {
        "source": source,
        "n_cities": int(len(cities)),
        "baseline_period": "1961-01-01..1990-12-31",
        "comparison_period": "2025-01-01..2025-12-31",
        "best_span": float(best_span),
        "best_K": int(best_K),
        "cv_span_best_rmse": float(cv_span["cv_rmse"].min()),
        "cv_K_best_rmse": float(cv_K["cv_rmse"].min()),
        "span_grid": SPAN_GRID,
        "K_grid": K_GRID,
        "n_days_2025": n_days,
        "mean_anomaly_2025": round(mean_anom, 3),
        "mean_temp_2025": round(mean_2025, 3),
        "mean_normal": round(mean_normal, 3),
        "days_above_p95": days_above_p95,
        "days_below_p5": days_below_p5,
        "warm_days": warm_days,
        "warm_day_share": round(100 * warm_days / n_days, 1),
        "max_loess_harmonic_gap": round(max_normal_gap, 3),
        "cv_rmse_reduction_pct": round(rmse_improvement, 1),
        "baseline_annual_mean": round(float(baseline["temp"].mean()), 3),
    }
    (DATA / "results.json").write_text(json.dumps(results, indent=2))
    print("\n=== RESULTS ===")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
