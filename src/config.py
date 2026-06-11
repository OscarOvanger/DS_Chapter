"""Shared configuration for the Austin/Travis County downscaling testbed.

Everything that the notebook and helper scripts need to agree on lives here:
where data is cached, which region and time periods we study, and the handful
of physical constants used when converting raw ERA5 / PRISM fields to the
variables we downscale (daily mean temperature in degC, daily precipitation in mm).
"""
from __future__ import annotations

from pathlib import Path

# --- Paths -----------------------------------------------------------------
# Project root is one level up from this file (src/ -> project root).
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
DATA.mkdir(exist_ok=True)
FIGURES = ROOT / "figures"
FIGURES.mkdir(exist_ok=True)

# Standalone Plotly HTML (month slider) for the Quarto chapter.
DC_TEMPERATURE_HTML = FIGURES / "dc_temperature.html"
DC_PRECIPITATION_HTML = FIGURES / "dc_precipitation.html"
BCSD_DIST_TEMPERATURE_HTML = FIGURES / "bcsd_distribution_temperature.html"
BCSD_DIST_PRECIPITATION_HTML = FIGURES / "bcsd_distribution_precipitation.html"
SDSM_REGRESSION_TEMPERATURE_HTML = FIGURES / "sdsm_regression_temperature.html"
SDSM_REGRESSION_PRECIPITATION_HTML = FIGURES / "sdsm_regression_precipitation.html"

# Cached, analysis-ready files produced by scripts/prepare_data.py.
# The notebook only ever reads these, so it stays fast and offline.
ERA5_HIST_NC = DATA / "era5_hist_daily.nc"
ERA5_FUT_NC = DATA / "era5_fut_daily.nc"
PRISM_HIST_NC = DATA / "prism_hist_daily.nc"
PRISM_FUT_NC = DATA / "prism_fut_daily.nc"
CMIP6_HIST_NC = DATA / "cmip6_hist_monthly.nc"
CMIP6_FUT_NC = DATA / "cmip6_fut_monthly.nc"
TRAVIS_COUNTY_GEOJSON = DATA / "travis_county.geojson"

# --- Study region: Travis County, Texas (Austin) ---------------------------
# County FIPS = state 48 (TX) + county 453 (Travis).
STATE_FIPS = "48"
COUNTY_FIPS = "453"

# Bounding box of Travis County (minlon, minlat, maxlon, maxlat), degrees.
# Used directly for the PRISM (target) request.
TRAVIS_BBOX = (-98.17, 30.02, -97.37, 30.63)

# ERA5 is coarse (0.25 deg). We pad the request so the county sits inside a
# margin of input cells -- interpolating PRISM-resolution deltas from ERA5 is
# far better behaved when the county is surrounded by data rather than clipped
# at the edge of the grid.
ERA5_PAD = 0.4  # degrees of padding on every side


def era5_bbox() -> tuple[float, float, float, float]:
    """Padded bounding box (minlon, minlat, maxlon, maxlat) for the ERA5 pull."""
    w, s, e, n = TRAVIS_BBOX
    return (w - ERA5_PAD, s - ERA5_PAD, e + ERA5_PAD, n + ERA5_PAD)


def coarse_bbox() -> tuple[float, float, float, float]:
    """Padded bbox for coarse models (CMIP6, ERA5) over Travis County."""
    return era5_bbox()


# --- Time periods ----------------------------------------------------------
# A 10-year "historical / training" period and a 5-year-plus "future / test"
# period. The future PRISM field is the ground truth every downscaler is
# scored against -- it is never shown to the models, only used for evaluation.
HIST = ("1991-01-01", "2000-12-31")  # 10 years: historical baseline
FUT = ("2016-01-01", "2025-12-31")   # future projection target / ground truth

# --- Data sources ----------------------------------------------------------
# ARCO-ERA5: Google's analysis-ready, cloud-optimised ERA5 (0.25 deg, hourly).
# No CDS API key and no request queue -- we lazily slice the bytes we need.
ARCO_ERA5_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"

# ACIS GridData: free PRISM access (grid id 21), ~4 km daily.
ACIS_GRIDDATA_URL = "https://data.rcc-acis.org/GridData"
ACIS_PRISM_GRID = "21"

# Pangeo CMIP6 (monthly Amon) for delta-change demos.
PANGEO_CMIP6_CATALOG = "https://storage.googleapis.com/cmip6/pangeo-cmip6.json"
CMIP6_SOURCE_ID = "ACCESS-CM2"
CMIP6_MEMBER_ID = "r1i1p1f1"
CMIP6_HIST_EXPERIMENT = "historical"
CMIP6_FUT_EXPERIMENT = "ssp245"
CMIP6_TABLE_ID = "Amon"
CMIP6_GRID_LABEL = "gn"
CMIP6_FALLBACK_SOURCES = ("ACCESS-CM2", "MPI-ESM1-2-LR")

# CMIP6 experiment id per cached period (hist vs fut).
CMIP6_EXPERIMENT = {"hist": CMIP6_HIST_EXPERIMENT, "fut": CMIP6_FUT_EXPERIMENT}

# --- Physical constants ----------------------------------------------------
KELVIN = 273.15      # ERA5 2m_temperature is in Kelvin
M_TO_MM = 1000.0     # ERA5 total_precipitation is in metres (per hour)
# CMIP6 pr (kg m-2 s-1 == mm s-1) -> mm/day (86400 s per day).
KG_M2_S_TO_MM_DAY = 86400.0
