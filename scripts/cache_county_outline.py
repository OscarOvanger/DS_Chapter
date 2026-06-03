"""Cache Travis County outline to data/travis_county.geojson (one-time)."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import geopandas as gpd

from src import config as C

url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_20m.zip"
gdf = (
    gpd.read_file(url)
    .query(f"STATEFP == '{C.STATE_FIPS}' and COUNTYFP == '{C.COUNTY_FIPS}'")
    .to_crs(4326)
)
out = C.DATA / "travis_county.geojson"
gdf.to_file(out, driver="GeoJSON")
print(f"Wrote {out}")
