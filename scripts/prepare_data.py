"""One-time data preparation for the downscaling testbed.

Run this once (it caches everything to ``data/``); afterwards the notebook just
reads the cached NetCDF files and stays fast and offline.

    .venv\\Scripts\\python.exe scripts\\prepare_data.py

Downloads, for the historical (1991-2000) and future (2016-2025) periods:
  * PRISM (target, ~4 km)  from the ACIS GridData service   -- no API key needed
  * CMIP6 (coarse input)  monthly Amon from Pangeo via intake-esm
  * ERA5  (optional, ~25 km) from ARCO-ERA5 for later BCSD cells

Already-cached periods are skipped, so re-running is cheap and resumable.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

# Allow "python scripts/prepare_data.py" to import the src package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import config as C
from src import data_io as io


def _step(name: str, path: Path, fn) -> None:
    if path.exists():
        print(f"  [skip] {name:18s} already cached -> {path.name}")
        return
    t0 = time.time()
    print(f"  [get ] {name:18s} downloading ...", flush=True)
    ds = fn()
    dims = dict(ds.sizes)
    print(f"  [done] {name:18s} {dims} in {time.time() - t0:5.1f}s -> {path.name}")


def main() -> None:
    print(f"Preparing data for Travis County into {C.DATA}")

    print("PRISM (ACIS GridData):")
    _step("PRISM historical", C.PRISM_HIST_NC,
          lambda: io.download_prism_daily(C.HIST, C.PRISM_HIST_NC))
    _step("PRISM future", C.PRISM_FUT_NC,
          lambda: io.download_prism_daily(C.FUT, C.PRISM_FUT_NC))

    print(f"CMIP6 (Pangeo Amon, model={C.CMIP6_SOURCE_ID}):")
    _step(
        "CMIP6 historical",
        C.CMIP6_HIST_NC,
        lambda: io.download_cmip6_monthly(C.HIST, C.CMIP6_HIST_EXPERIMENT, C.CMIP6_HIST_NC),
    )
    _step(
        "CMIP6 future",
        C.CMIP6_FUT_NC,
        lambda: io.download_cmip6_monthly(C.FUT, C.CMIP6_FUT_EXPERIMENT, C.CMIP6_FUT_NC),
    )

    print("ERA5 (ARCO-ERA5, optional for BCSD):")
    _step("ERA5 historical", C.ERA5_HIST_NC,
          lambda: io.download_era5_daily(C.HIST, C.ERA5_HIST_NC))
    _step("ERA5 future", C.ERA5_FUT_NC,
          lambda: io.download_era5_daily(C.FUT, C.ERA5_FUT_NC))

    county = C.TRAVIS_COUNTY_GEOJSON
    if county.exists():
        print(f"  [skip] county outline     already cached -> {county.name}")
    else:
        import geopandas as gpd

        print("  [get ] county outline     downloading …", flush=True)
        url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_20m.zip"
        gdf = (
            gpd.read_file(url)
            .query(f"STATEFP == '{C.STATE_FIPS}' and COUNTYFP == '{C.COUNTY_FIPS}'")
            .to_crs(4326)
        )
        gdf.to_file(county, driver="GeoJSON")
        print(f"  [done] county outline     -> {county.name}")

    print("Plotly chapter figures:")
    try:
        from src.plotly_panels import export_all

        for p in export_all():
            print(f"  [done] {p.name}")
    except Exception as exc:
        print(f"  [skip] export_plotly_figures ({exc})")
        print("         Run: .venv\\Scripts\\python.exe scripts\\export_plotly_figures.py")

    print("All data ready. Open Downscaling_BS.qmd or DS_notebook.ipynb.")


if __name__ == "__main__":
    main()
