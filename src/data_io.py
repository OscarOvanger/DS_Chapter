"""Data loading and caching for the downscaling testbed.

Two sources, both free and key-less:

* **ERA5 (input / coarse)** comes from ARCO-ERA5, an analysis-ready ERA5 Zarr
  store on Google Cloud. We lazily slice the Travis bounding box and time range,
  so only the bytes we need are downloaded -- no CDS API key and no request queue.

* **PRISM (target / ~4 km)** comes from the ACIS GridData web service. A single
  request returns the whole bounding-box grid for a month, which is far faster
  than the per-grid-cell approach.

Both are aggregated to the two variables we downscale -- daily mean temperature
(``T``, degC) and daily total precipitation (``P``, mm) -- and cached to NetCDF.
The notebook calls the ``load_*`` functions, which read the cache and raise a
clear error if ``scripts/prepare_data.py`` has not been run yet.
"""
from __future__ import annotations

import calendar
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import xarray as xr

from . import config as C

# ---------------------------------------------------------------------------
# ERA5 (ARCO-ERA5 on Google Cloud)
# ---------------------------------------------------------------------------


def _era5_to_daily(ds: xr.Dataset) -> xr.Dataset:
    """Aggregate hourly ERA5 to the daily T (degC) / P (mm) we downscale."""
    ds = ds.rename(latitude="lat", longitude="lon")
    # ARCO-ERA5 longitudes are 0..360; convert to -180..180 for plotting/PRISM.
    ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
    ds = ds.sortby("lat", ascending=False)
    daily = xr.Dataset(
        {
            "T": ds["2m_temperature"].resample(time="1D").mean() - C.KELVIN,
            "P": (ds["total_precipitation"].resample(time="1D").sum() * C.M_TO_MM).clip(min=0),
        }
    )
    daily["T"].attrs.update(units="degC", long_name="daily mean 2 m temperature")
    daily["P"].attrs.update(units="mm", long_name="daily total precipitation")
    return daily


def download_era5_daily(period: tuple[str, str], out_path: Path) -> xr.Dataset:
    """Slice ARCO-ERA5 over the padded Travis bbox for ``period`` and cache to NetCDF."""
    w, s, e, n = C.era5_bbox()
    ds = xr.open_zarr(C.ARCO_ERA5_STORE, chunks={"time": 24 * 31}, storage_options={"token": "anon"})
    ds = ds[["2m_temperature", "total_precipitation"]].sel(
        time=slice(period[0], period[1]),
        latitude=slice(n, s),               # ERA5 latitude is descending
        longitude=slice(w % 360, e % 360),  # ERA5 longitude is 0..360
    )
    daily = _era5_to_daily(ds).compute()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    daily.to_netcdf(out_path)
    return daily


def load_era5(period: str) -> xr.Dataset:
    """Load cached daily ERA5. ``period`` is ``"hist"`` or ``"fut"``."""
    path = {"hist": C.ERA5_HIST_NC, "fut": C.ERA5_FUT_NC}[period]
    return _open_cached(path, "ERA5")


# ---------------------------------------------------------------------------
# PRISM (ACIS GridData)
# ---------------------------------------------------------------------------


def _acis_month(year: int, month: int, bbox: tuple[float, float, float, float]) -> dict:
    """Fetch one month of PRISM (avgt in degC, pcpn in mm) for the bbox."""
    w, s, e, n = bbox
    last = calendar.monthrange(year, month)[1]
    payload = {
        "bbox": f"{w},{s},{e},{n}",
        "sdate": f"{year}-{month:02d}-01",
        "edate": f"{year}-{month:02d}-{last:02d}",
        "grid": C.ACIS_PRISM_GRID,
        "elems": [
            {"name": "avgt", "units": "degreeC"},  # PRISM daily mean temperature
            {"name": "pcpn", "units": "mm"},       # PRISM daily precipitation
        ],
        "meta": ["ll"],
    }
    r = requests.post(C.ACIS_GRIDDATA_URL, json=payload, timeout=180)
    r.raise_for_status()
    return r.json()


def _clean(grid) -> np.ndarray:
    """ACIS returns -999 (and occasionally 'M') for missing cells -> NaN."""
    arr = np.array(grid, dtype="float64")
    arr[arr < -900] = np.nan
    return arr


def download_prism_daily(period: tuple[str, str], out_path: Path) -> xr.Dataset:
    """Fetch PRISM month-by-month over the Travis bbox for ``period`` and cache to NetCDF."""
    months = pd.date_range(period[0], period[1], freq="MS")
    lat = lon = None
    times: list[pd.Timestamp] = []
    T_days: list[np.ndarray] = []
    P_days: list[np.ndarray] = []

    for ts in months:
        j = _acis_month(ts.year, ts.month, C.TRAVIS_BBOX)
        if lat is None:
            latg = np.array(j["meta"]["lat"], dtype="float64")
            long_ = np.array(j["meta"]["lon"], dtype="float64")
            lat, lon = latg[:, 0], long_[0, :]
        for day in j["data"]:
            times.append(pd.Timestamp(day[0]))
            T_days.append(_clean(day[1]))
            P_days.append(_clean(day[2]))

    ds = xr.Dataset(
        {
            "T": (("time", "lat", "lon"), np.stack(T_days)),
            "P": (("time", "lat", "lon"), np.stack(P_days)),
        },
        coords={"time": pd.DatetimeIndex(times), "lat": lat, "lon": lon},
    ).sortby("lat", ascending=False)
    ds["T"].attrs.update(units="degC", long_name="PRISM daily mean temperature")
    ds["P"].attrs.update(units="mm", long_name="PRISM daily precipitation")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    ds.to_netcdf(out_path)
    return ds


def load_prism(period: str) -> xr.Dataset:
    """Load cached daily PRISM. ``period`` is ``"hist"`` or ``"fut"``."""
    path = {"hist": C.PRISM_HIST_NC, "fut": C.PRISM_FUT_NC}[period]
    return _open_cached(path, "PRISM")


# ---------------------------------------------------------------------------
# CMIP6 (Pangeo / intake-esm, monthly Amon)
# ---------------------------------------------------------------------------


def _subset_bbox(ds: xr.Dataset, bbox: tuple[float, float, float, float]) -> xr.Dataset:
    """Spatial subset for Travis padded bbox (handles lon 0–360 and lat order)."""
    w, s, e, n = bbox
    ds = _normalize_lon_lat(ds)
    if float(ds.lon.max()) > 180:
        ds = ds.sel(lon=slice(w % 360, e % 360))
    else:
        ds = ds.sel(lon=slice(w, e))
    lat = ds.lat.values
    if lat[0] < lat[-1]:
        ds = ds.sel(lat=slice(s, n))
    else:
        ds = ds.sel(lat=slice(n, s))
    return ds


def _normalize_lon_lat(ds: xr.Dataset) -> xr.Dataset:
    """Rename CMIP6 coords to lat/lon and use -180..180 longitudes."""
    ren = {k: v for k, v in {"latitude": "lat", "longitude": "lon"}.items() if k in ds.coords}
    if ren:
        ds = ds.rename(ren)
    if "lon" in ds.coords:
        ds = ds.assign_coords(lon=(((ds.lon + 180) % 360) - 180)).sortby("lon")
    if "lat" in ds.coords:
        ds = ds.sortby("lat", ascending=False)
    return ds


def _cmip6_to_monthly(ds: xr.Dataset) -> xr.Dataset:
    """Map CMIP6 Amon variables to project T (degC) and P (mm/day)."""
    out = {}
    if "tas" in ds:
        out["T"] = ds["tas"] - C.KELVIN
    if "pr" in ds:
        out["P"] = ds["pr"] * C.KG_M2_S_TO_MM_DAY
    monthly = xr.Dataset(out)
    monthly["T"].attrs.update(units="degC", long_name="CMIP6 monthly mean near-surface temperature")
    monthly["P"].attrs.update(units="mm/day", long_name="CMIP6 monthly mean precipitation rate")
    return monthly


def _open_cmip6_zarr(cat, source_id: str, experiment_id: str, variable_id: str) -> xr.Dataset:
    """Open one CMIP6 Zarr asset from the Pangeo catalog (anonymous GCS)."""
    sub = cat.search(
        source_id=source_id,
        experiment_id=experiment_id,
        variable_id=variable_id,
        table_id=C.CMIP6_TABLE_ID,
        member_id=C.CMIP6_MEMBER_ID,
        grid_label=C.CMIP6_GRID_LABEL,
    )
    if sub.df.empty:
        raise ValueError(
            f"No CMIP6 asset for {source_id} / {experiment_id} / {variable_id} "
            f"({C.CMIP6_TABLE_ID}, {C.CMIP6_MEMBER_ID})"
        )
    zstore = sub.df.iloc[0]["zstore"]
    return xr.open_zarr(
        zstore,
        chunks={"time": 12},
        storage_options={"token": "anon"},
    )


def download_cmip6_monthly(
    period: tuple[str, str],
    experiment_id: str,
    out_path: Path,
    source_id: str | None = None,
) -> xr.Dataset:
    """Fetch monthly CMIP6 tas/pr over the padded Travis bbox and cache to NetCDF."""
    import intake

    w, s, e, n = C.coarse_bbox()
    sources = [source_id] if source_id else list(C.CMIP6_FALLBACK_SOURCES)
    last_err: Exception | None = None
    ds: xr.Dataset | None = None

    cat = intake.open_esm_datastore(C.PANGEO_CMIP6_CATALOG)
    for sid in sources:
        try:
            parts = []
            for var in ("tas", "pr"):
                raw = _open_cmip6_zarr(cat, sid, experiment_id, var)
                raw = _subset_bbox(raw, (w, s, e, n)).sel(time=slice(period[0], period[1]))
                parts.append(raw)
            ds = xr.merge(parts, compat="override")
            ds.attrs["source_id"] = sid
            ds.attrs["experiment_id"] = experiment_id
            break
        except Exception as exc:
            last_err = exc
            continue

    if ds is None:
        raise RuntimeError(f"CMIP6 download failed for {experiment_id}: {last_err}") from last_err

    monthly = _cmip6_to_monthly(ds).compute()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    monthly.to_netcdf(out_path)
    return monthly


def load_cmip6(period: str) -> xr.Dataset:
    """Load cached monthly CMIP6. ``period`` is ``"hist"`` or ``"fut"``."""
    path = {"hist": C.CMIP6_HIST_NC, "fut": C.CMIP6_FUT_NC}[period]
    return _open_cached(path, "CMIP6")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _open_cached(path: Path, name: str) -> xr.Dataset:
    if not path.exists():
        raise FileNotFoundError(
            f"{name} cache not found at {path}.\n"
            "Run the one-time data preparation first:\n"
            "    .venv\\Scripts\\python.exe scripts\\prepare_data.py"
        )
    return xr.open_dataset(path)


def regrid_to(src: xr.Dataset, target: xr.Dataset, method: str = "linear") -> xr.Dataset:
    """Interpolate ``src`` onto the lat/lon grid of ``target`` (e.g. PRISM<->ERA5)."""
    return src.interp(lat=target.lat, lon=target.lon, method=method)


def load_all() -> dict[str, xr.Dataset]:
    """Convenience loader returning every cached field the notebook uses."""
    return {
        "era5_hist": load_era5("hist"),
        "era5_fut": load_era5("fut"),
        "prism_hist": load_prism("hist"),
        "prism_fut": load_prism("fut"),
    }


def load_delta_change_data(progress: bool = True) -> dict[str, xr.Dataset]:
    """PRISM + CMIP6 caches for delta-change interactive cells."""
    steps = [
        ("PRISM 1991–2000", "prism_hist", lambda: load_prism("hist")),
        ("PRISM 2016–2025", "prism_fut", lambda: load_prism("fut")),
        ("CMIP6 historical", "cmip6_hist", lambda: load_cmip6("hist")),
        ("CMIP6 future", "cmip6_fut", lambda: load_cmip6("fut")),
    ]
    out: dict[str, xr.Dataset] = {}
    if progress:
        from tqdm.auto import tqdm

        bar = tqdm(steps, desc="Loading cached data", unit="file")
        for label, key, loader in bar:
            bar.set_postfix_str(label, refresh=False)
            print(f"  -> {label}", flush=True)
            out[key] = loader().load()
        return out

    for _label, key, loader in steps:
        out[key] = loader().load()
    return out
