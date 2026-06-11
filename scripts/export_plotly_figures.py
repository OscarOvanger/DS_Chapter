"""Export standalone Plotly HTML figures for the Quarto chapter.

Run after ``prepare_data.py``:

    .venv\\Scripts\\python.exe scripts\\export_plotly_figures.py

Writes:

  figures/dc_temperature.html
  figures/dc_precipitation.html

Embed these in ``Downscaling_BS.qmd`` with raw HTML iframes (no Python execution at render time).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import data_io as io
from src.plotly_bcsd import build_distribution_cache, export_distribution_html
from src.plotly_panels import build_monthly_cache, export_delta_monthly_html
from src.plotly_sdsm_regression import build_sdsm_cache, export_sdsm_regression_html


def main() -> None:
    data = io.load_delta_change_data(progress=True)
    dc_cache = build_monthly_cache(data)
    bcsd_cache = build_distribution_cache(data)
    paths: list = []
    for var in ("T", "P"):
        s, f = export_delta_monthly_html(var, data=data, cache=dc_cache)
        paths.extend([s, f])
        paths.append(export_distribution_html(var, data=data, cache=bcsd_cache))
    sdsm_cache = build_sdsm_cache(data)
    for var in ("T", "P"):
        paths.append(export_sdsm_regression_html(var, data=data, cache=sdsm_cache))
    for p in paths:
        print(f"  wrote {p.name} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
