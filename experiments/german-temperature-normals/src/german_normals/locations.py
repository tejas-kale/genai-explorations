"""Population-weighted table of German cities.

The goal — exactly as in the Spanish original — is a *population*-representative
sample, not an area-representative grid: we weight toward where people actually
live so the resulting national temperature curve reflects the climate
experienced by the average German resident, not the average German hectare.

The table holds the 16 Bundesland capitals plus the largest German cities by
population (~55 places in total). Germany is a single contiguous climate domain,
so unlike Spain (with its Canary Islands) no regional exclusions are needed; the
list still deliberately spans the maritime northwest (Kiel, Hamburg, Bremen,
Rostock, Oldenburg), the continental east (Berlin, Dresden, Leipzig, Cottbus),
the Rhine/Ruhr conurbation (Cologne, Essen, Dortmund, Duesseldorf) and the
southern uplands (Munich, Stuttgart, Freiburg, Augsburg).

Population figures are approximate Destatis ~2022/2023 city totals. They are used
*only* as relative weights, so their precision does not matter — what matters is
the relative pull of, say, Berlin versus Schwerin.
"""

from __future__ import annotations

import pandas as pd

# (city, bundesland, lat, lon, population)
# "capital" flag is derived below from the set of Bundesland capitals.
_CITIES: list[tuple[str, str, float, float, int]] = [
    # --- 16 Bundesland capitals -------------------------------------------
    ("Berlin", "Berlin", 52.5200, 13.4050, 3_677_000),
    ("Hamburg", "Hamburg", 53.5511, 9.9937, 1_906_000),
    ("Munich", "Bayern", 48.1351, 11.5820, 1_512_000),
    ("Stuttgart", "Baden-Wuerttemberg", 48.7758, 9.1829, 634_000),
    ("Duesseldorf", "Nordrhein-Westfalen", 51.2277, 6.7735, 629_000),
    ("Wiesbaden", "Hessen", 50.0782, 8.2398, 278_000),
    ("Mainz", "Rheinland-Pfalz", 49.9929, 8.2473, 217_000),
    ("Saarbruecken", "Saarland", 49.2401, 6.9969, 180_000),
    ("Erfurt", "Thueringen", 50.9848, 11.0299, 214_000),
    ("Kiel", "Schleswig-Holstein", 54.3233, 10.1228, 247_000),
    ("Magdeburg", "Sachsen-Anhalt", 52.1205, 11.6276, 238_000),
    ("Hannover", "Niedersachsen", 52.3759, 9.7320, 535_000),
    ("Dresden", "Sachsen", 51.0504, 13.7373, 556_000),
    ("Potsdam", "Brandenburg", 52.3906, 13.0645, 183_000),
    ("Schwerin", "Mecklenburg-Vorpommern", 53.6355, 11.4012, 96_000),
    ("Bremen", "Bremen", 53.0793, 8.8017, 567_000),
    # --- other large cities ------------------------------------------------
    ("Cologne", "Nordrhein-Westfalen", 50.9375, 6.9603, 1_073_000),
    ("Frankfurt", "Hessen", 50.1109, 8.6821, 759_000),
    ("Dortmund", "Nordrhein-Westfalen", 51.5136, 7.4653, 588_000),
    ("Essen", "Nordrhein-Westfalen", 51.4556, 7.0116, 579_000),
    ("Leipzig", "Sachsen", 51.3397, 12.3731, 597_000),
    ("Nuremberg", "Bayern", 49.4521, 11.0767, 518_000),
    ("Duisburg", "Nordrhein-Westfalen", 51.4344, 6.7623, 498_000),
    ("Bochum", "Nordrhein-Westfalen", 51.4818, 7.2162, 365_000),
    ("Wuppertal", "Nordrhein-Westfalen", 51.2562, 7.1508, 355_000),
    ("Bielefeld", "Nordrhein-Westfalen", 52.0302, 8.5325, 334_000),
    ("Bonn", "Nordrhein-Westfalen", 50.7374, 7.0982, 331_000),
    ("Muenster", "Nordrhein-Westfalen", 51.9607, 7.6261, 316_000),
    ("Mannheim", "Baden-Wuerttemberg", 49.4875, 8.4660, 311_000),
    ("Karlsruhe", "Baden-Wuerttemberg", 49.0069, 8.4037, 308_000),
    ("Augsburg", "Bayern", 48.3705, 10.8978, 296_000),
    ("Moenchengladbach", "Nordrhein-Westfalen", 51.1805, 6.4428, 261_000),
    ("Gelsenkirchen", "Nordrhein-Westfalen", 51.5177, 7.0857, 260_000),
    ("Braunschweig", "Niedersachsen", 52.2689, 10.5268, 249_000),
    ("Aachen", "Nordrhein-Westfalen", 50.7753, 6.0839, 249_000),
    ("Chemnitz", "Sachsen", 50.8278, 12.9214, 247_000),
    ("Halle", "Sachsen-Anhalt", 51.4969, 11.9688, 238_000),
    ("Freiburg", "Baden-Wuerttemberg", 47.9990, 7.8421, 231_000),
    ("Krefeld", "Nordrhein-Westfalen", 51.3388, 6.5853, 227_000),
    ("Luebeck", "Schleswig-Holstein", 53.8655, 10.6866, 217_000),
    ("Oberhausen", "Nordrhein-Westfalen", 51.4963, 6.8638, 210_000),
    ("Rostock", "Mecklenburg-Vorpommern", 54.0924, 12.0991, 209_000),
    ("Kassel", "Hessen", 51.3127, 9.4797, 201_000),
    ("Hagen", "Nordrhein-Westfalen", 51.3671, 7.4633, 189_000),
    ("Oldenburg", "Niedersachsen", 53.1435, 8.2146, 169_000),
    ("Osnabrueck", "Niedersachsen", 52.2799, 8.0472, 165_000),
    ("Regensburg", "Bayern", 49.0134, 12.1016, 153_000),
    ("Ingolstadt", "Bayern", 48.7665, 11.4258, 138_000),
    ("Wuerzburg", "Bayern", 49.7913, 9.9534, 127_000),
    ("Cottbus", "Brandenburg", 51.7563, 14.3329, 98_000),
    ("Goettingen", "Niedersachsen", 51.5413, 9.9158, 119_000),
    ("Trier", "Rheinland-Pfalz", 49.7499, 6.6371, 111_000),
    ("Kaiserslautern", "Rheinland-Pfalz", 49.4401, 7.7491, 100_000),
    ("Flensburg", "Schleswig-Holstein", 54.7937, 9.4460, 92_000),
    ("Konstanz", "Baden-Wuerttemberg", 47.6603, 9.1758, 85_000),
]

# The set of Bundesland capitals (first 16 entries), used to flag the table.
_CAPITAL_CITIES = {row[0] for row in _CITIES[:16]}


def load_cities() -> pd.DataFrame:
    """Return the city table with a normalised population ``weight`` column.

    Columns: ``city``, ``bundesland``, ``lat``, ``lon``, ``population``,
    ``is_capital`` and ``weight`` (population / total population, summing to 1).
    """
    df = pd.DataFrame(
        _CITIES,
        columns=["city", "bundesland", "lat", "lon", "population"],
    )
    df = df.drop_duplicates(subset="city").reset_index(drop=True)
    df["is_capital"] = df["city"].isin(_CAPITAL_CITIES)
    df["weight"] = df["population"] / df["population"].sum()
    return df


if __name__ == "__main__":  # pragma: no cover - manual inspection helper
    cities = load_cities()
    print(f"{len(cities)} cities, weights sum to {cities['weight'].sum():.6f}")
    print(cities.sort_values("population", ascending=False).head(10).to_string())
