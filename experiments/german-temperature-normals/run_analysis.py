"""CLI driver: build the dataset, choose span/K by cross-validation, compute the
2025 results and cache every product. Thin wrapper around
``german_normals.analysis.compute_all``.

    python run_analysis.py            # use cache if present
    python run_analysis.py --fresh    # recompute from scratch
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "src"))

from german_normals.analysis import compute_all  # noqa: E402

DATA = Path(__file__).parent / "data"


def main() -> None:
    fresh = "--fresh" in sys.argv
    products = compute_all(data_dir=DATA, use_cache=not fresh)
    print("cv (span):")
    print(products["cv_span"].to_string(index=False))
    print("cv (K):")
    print(products["cv_K"].to_string(index=False))
    print("\n=== RESULTS ===")
    print(json.dumps(products["results"], indent=2))


if __name__ == "__main__":
    main()
