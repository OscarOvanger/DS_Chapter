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

from src.plotly_panels import export_all


def main() -> None:
    paths = export_all()
    for p in paths:
        print(f"  wrote {p.name} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
