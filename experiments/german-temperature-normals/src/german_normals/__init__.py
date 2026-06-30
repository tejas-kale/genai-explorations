"""German daily temperature normals (1961-1990) vs. 2025.

A Python reproduction — for Germany — of Dominic Roye's R analysis of Spanish
temperature normals built from Open-Meteo ERA5 reanalysis. The package is split
into small, importable, testable modules:

- ``locations``   : the population-weighted table of German cities.
- ``fetch``       : Open-Meteo Historical Archive download + parquet cache.
- ``synthetic``   : an offline, physically-plausible ERA5-like fallback.
- ``climatology`` : population-weighted national daily series and the
                    day-of-year median / P5 / P95 baseline statistics.
- ``smoothing``   : loess and harmonic fits plus 5-fold cross-validation
                    (split by year) to choose the loess span and harmonic K.
- ``figure``      : the final fan chart with the Germany map inset.
"""

__all__ = [
    "locations",
    "fetch",
    "synthetic",
    "climatology",
    "smoothing",
    "figure",
]
