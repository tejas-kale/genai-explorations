"""The final fan chart: 2025 daily temperature against the 1961-1990 normal.

Layers, from back to front:

1. a light grey P5-P95 baseline band with dashed edges;
2. the smoothed loess normal (the reference line);
3. salmon / blue anomaly ribbons filling between the 2025 line and the normal
   (above-normal warm in salmon, below-normal cool in blue);
4. *darker* salmon / blue where 2025 pushes outside the P5-P95 band entirely;
5. the 2025 daily line itself;
6. circled monthly-extrema callouts (the single most extreme anomaly per month);
7. a small Germany map inset showing the population-weighted city sample.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Month-start day-of-year on the leap reference calendar (so 1 Mar = 61).
_MONTH_STARTS = [1, 32, 61, 92, 122, 153, 183, 214, 245, 275, 306, 336]
_MONTH_LABELS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                 "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
_MONTH_MIDS = [(s + e) / 2 for s, e in zip(_MONTH_STARTS, _MONTH_STARTS[1:] + [367])]

SALMON = "#e8896b"
SALMON_DARK = "#c5462a"
BLUE = "#6b9fe8"
BLUE_DARK = "#2a5bc5"
NORMAL_COLOR = "#2b2b2b"
BAND_COLOR = "#9aa0a6"


def _monthly_extrema(comp: pd.DataFrame) -> pd.DataFrame:
    """For each month, the single day of largest absolute anomaly."""
    c = comp.copy()
    c["month"] = pd.DatetimeIndex(c["date"]).month
    idx = c.groupby("month")["anomaly"].apply(lambda s: s.abs().idxmax())
    return c.loc[idx.to_numpy()].sort_values("doy")


def _add_map_inset(fig, cities: pd.DataFrame, geojson_path: str | Path) -> None:
    try:
        import geopandas as gpd

        germany = gpd.read_file(geojson_path)
    except Exception:  # pragma: no cover - inset is optional
        return
    # Upper-right corner (autumn high-temperature region) is empty -> tuck the
    # map there so it never overlaps the winter data or the legend.
    ax = fig.add_axes([0.80, 0.50, 0.165, 0.36])
    germany.boundary.plot(ax=ax, color="#555555", linewidth=0.8)
    germany.plot(ax=ax, color="#f2f2f2", zorder=0)
    sizes = 6 + 60 * (cities["population"] / cities["population"].max())
    ax.scatter(cities["lon"], cities["lat"], s=sizes, c=SALMON_DARK,
               edgecolor="white", linewidth=0.3, alpha=0.85, zorder=3)
    ax.set_axis_off()
    ax.set_title(f"{len(cities)} cities,\npopulation-weighted", fontsize=7.5, color="#444")


def make_fan_chart(
    comp: pd.DataFrame,
    normals: pd.DataFrame,
    clim: pd.DataFrame,
    cities: pd.DataFrame,
    results: dict,
    geojson_path: str | Path,
    out_path: str | Path,
) -> Path:
    """Render the fan chart to ``out_path`` (PNG) and return the path.

    ``comp`` is the 2025 daily series merged with ``loess`` (normal), ``p5``,
    ``p95`` and ``anomaly``; ``normals`` holds the loess/harmonic normals over
    doy 1..366; ``clim`` holds the day-of-year band.
    """
    comp = comp.sort_values("doy").reset_index(drop=True)
    doy = comp["doy"].to_numpy()
    temp = comp["temp"].to_numpy()
    normal = comp["loess"].to_numpy()
    p5 = comp["p5"].to_numpy()
    p95 = comp["p95"].to_numpy()

    fig, ax = plt.subplots(figsize=(14, 7.6))

    # 1. baseline band (full doy axis) ------------------------------------
    nd = normals["doy"].to_numpy()
    band = clim.set_index("doy").reindex(np.arange(1, 367)).interpolate()
    ax.fill_between(np.arange(1, 367), band["p5"], band["p95"],
                    color=BAND_COLOR, alpha=0.18, zorder=1)
    ax.plot(np.arange(1, 367), band["p5"], color=BAND_COLOR, lw=0.9, ls=(0, (5, 3)), zorder=2)
    ax.plot(np.arange(1, 367), band["p95"], color=BAND_COLOR, lw=0.9, ls=(0, (5, 3)), zorder=2)

    # 2. smoothed normal --------------------------------------------------
    ax.plot(nd, normals["loess"], color=NORMAL_COLOR, lw=2.0, zorder=5,
            label=f"1961-1990 normal (loess, span {results['best_span']:g})")

    # 3. anomaly ribbons (within band) ------------------------------------
    ax.fill_between(doy, normal, temp, where=temp >= normal, interpolate=True,
                    color=SALMON, alpha=0.55, zorder=3, label="2025 above normal")
    ax.fill_between(doy, normal, temp, where=temp < normal, interpolate=True,
                    color=BLUE, alpha=0.55, zorder=3, label="2025 below normal")

    # 4. darker fills outside the P5-P95 band -----------------------------
    ax.fill_between(doy, p95, temp, where=temp > p95, interpolate=True,
                    color=SALMON_DARK, alpha=0.85, zorder=4, label="above P95")
    ax.fill_between(doy, p5, temp, where=temp < p5, interpolate=True,
                    color=BLUE_DARK, alpha=0.85, zorder=4, label="below P5")

    # 5. the 2025 line ----------------------------------------------------
    ax.plot(doy, temp, color="#1a1a1a", lw=0.7, alpha=0.6, zorder=6)

    # 6. circled monthly extrema -----------------------------------------
    extrema = _monthly_extrema(comp)
    y0, y1 = ax.get_ylim() if ax.get_ylim()[0] < ax.get_ylim()[1] else (-10, 30)
    for _, r in extrema.iterrows():
        warm = r["anomaly"] >= 0
        edge = SALMON_DARK if warm else BLUE_DARK
        ax.scatter([r["doy"]], [r["temp"]], s=130, facecolors="none",
                   edgecolors=edge, linewidths=1.6, zorder=7)
        off = 1.8 if warm else -1.8
        ax.annotate(
            f"{'+' if warm else ''}{r['anomaly']:.1f}°C",
            (r["doy"], r["temp"]), xytext=(r["doy"], r["temp"] + off),
            ha="center", va="bottom" if warm else "top", fontsize=7.5,
            color=edge, fontweight="bold", zorder=8,
        )

    # axes cosmetics ------------------------------------------------------
    ax.set_xlim(1, 366)
    ax.set_xticks(_MONTH_MIDS)
    ax.set_xticklabels(_MONTH_LABELS)
    for s in _MONTH_STARTS[1:]:
        ax.axvline(s, color="#e6e6e6", lw=0.6, zorder=0)
    ax.set_ylabel("Daily mean temperature (°C)")
    ax.grid(axis="y", color="#eeeeee", lw=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    src = "Open-Meteo ERA5" if results["source"] == "open-meteo" else "SYNTHETIC placeholder data"
    ax.set_title(
        "Germany 2025 daily temperature vs. the 1961-1990 normal",
        fontsize=16, fontweight="bold", loc="left", pad=22,
    )
    ax.text(
        0, 1.02,
        f"Population-weighted across {results['n_cities']} cities · "
        f"mean 2025 anomaly {results['mean_anomaly_2025']:+.1f} °C · "
        f"{results['days_above_p95']} days above P95, {results['days_below_p5']} below P5",
        transform=ax.transAxes, fontsize=10.5, color="#444", va="bottom",
    )
    # Legend in the empty winter top-left corner.
    ax.legend(loc="upper left", bbox_to_anchor=(0.005, 0.995), fontsize=8.5,
              ncol=2, frameon=False)
    fig.text(0.5, 0.015,
             f"Source: {src}. Baseline 1961-1990. Day-of-year band = 5th-95th percentile of "
             f"the population-weighted national daily series.",
             ha="center", fontsize=8, color="#777")

    _add_map_inset(fig, cities, geojson_path)

    fig.subplots_adjust(left=0.06, right=0.985, top=0.86, bottom=0.10)
    out_path = Path(out_path)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
