"""Delta change (DC) downscaling -- the simplest climatological downscaler.

Idea: we have high-resolution *historical* observations (PRISM) and a coarse
climate model for both the historical and future periods (CMIP6 for the chapter
demos; ERA5 for later BCSD cells). The model's *change* from past to future (the
"delta") is assumed to be reliable even if its absolute values are biased, so we
add that delta onto the observed history.

Two flavours, matched to the variable:

* **Additive delta (temperature):**  T_future(x) = T_obs(x) + (T̄_fut - T̄_hist)
* **Multiplicative delta (precipitation):**  P_future(x) = P_obs(x) * (P̄_fut / P̄_hist)
"""
from __future__ import annotations

import xarray as xr

KIND = {"T": "add", "P": "mult"}


def monthly_climatology(ds: xr.Dataset) -> xr.Dataset:
    """Calendar-month mean fields (month=1..12) averaged over all years in ``ds``."""
    return ds.groupby("time.month").mean("time")


def monthly_delta(coarse_hist: xr.Dataset, coarse_fut: xr.Dataset, var: str) -> xr.DataArray:
    """Monthly delta on the coarse grid: additive for T, multiplicative ratio for P."""
    hist_m = monthly_climatology(coarse_hist)[var]
    fut_m = monthly_climatology(coarse_fut)[var]
    if KIND[var] == "add":
        return fut_m - hist_m
    return fut_m / hist_m.clip(min=1e-6)


def apply_monthly_delta(
    prism_monthly: xr.DataArray,
    delta_coarse: xr.DataArray,
    var: str,
    method: str = "linear",
) -> xr.DataArray:
    """Apply a coarse monthly delta (or ratio) to a PRISM monthly climatology field."""
    nlat = delta_coarse.sizes.get("lat", 1)
    nlon = delta_coarse.sizes.get("lon", 1)
    if nlat * nlon == 1:
        # Single coarse cell over the county: uniform delta (no bogus NaN extrapolation).
        delta_val = float(delta_coarse.mean(skipna=True))
        if KIND[var] == "add":
            return prism_monthly + delta_val
        return (prism_monthly * delta_val).clip(min=0)
    delta_hi = delta_coarse.interp(lat=prism_monthly.lat, lon=prism_monthly.lon, method=method)
    if KIND[var] == "add":
        return prism_monthly + delta_hi
    return (prism_monthly * delta_hi).clip(min=0)


def precompute_monthly_cache(data: dict[str, xr.Dataset]) -> dict[str, xr.Dataset | xr.DataArray]:
    """Monthly climatologies and coarse deltas (compute once before interactive widgets)."""
    prism_hist_m = monthly_climatology(data["prism_hist"])
    prism_fut_m = monthly_climatology(data["prism_fut"])
    coarse_hist_m = monthly_climatology(data["cmip6_hist"])
    coarse_fut_m = monthly_climatology(data["cmip6_fut"])
    return {
        "prism_hist_m": prism_hist_m,
        "prism_fut_m": prism_fut_m,
        "coarse_delta_T": coarse_fut_m["T"] - coarse_hist_m["T"],
        "coarse_delta_P": coarse_fut_m["P"] / coarse_hist_m["P"].clip(min=1e-6),
    }


def project_month_from_cache(
    cache: dict[str, xr.Dataset | xr.DataArray],
    var: str,
    month: int,
    method: str = "linear",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Baseline, projection, and truth for one month using a :func:`precompute_monthly_cache` result."""
    delta_key = "coarse_delta_T" if var == "T" else "coarse_delta_P"
    baseline = cache["prism_hist_m"][var].sel(month=month)
    truth = cache["prism_fut_m"][var].sel(month=month)
    dlt = cache[delta_key].sel(month=month)
    proj = apply_monthly_delta(baseline, dlt, var, method=method)
    return baseline, proj, truth


def project_monthly_maps(
    prism_hist: xr.Dataset,
    prism_fut: xr.Dataset,
    coarse_hist: xr.Dataset,
    coarse_fut: xr.Dataset,
    var: str,
    month: int,
    method: str = "linear",
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray]:
    """Baseline, delta-change projection, and truth monthly maps for one calendar month."""
    cache = precompute_monthly_cache(
        {
            "prism_hist": prism_hist,
            "prism_fut": prism_fut,
            "cmip6_hist": coarse_hist,
            "cmip6_fut": coarse_fut,
        }
    )
    return project_month_from_cache(cache, var, month, method=method)


def monthly_deltas(coarse_hist: xr.Dataset, coarse_fut: xr.Dataset) -> xr.Dataset:
    """12 monthly coarse deltas: additive for T (degC), multiplicative for P (ratio)."""
    return xr.Dataset(
        {
            "delta_T": monthly_delta(coarse_hist, coarse_fut, "T"),
            "delta_P": monthly_delta(coarse_hist, coarse_fut, "P"),
        }
    )


def _to_prism_daily(monthly: xr.DataArray, prism: xr.Dataset, method: str) -> xr.DataArray:
    """Interpolate a monthly coarse field to the PRISM grid, broadcast to daily."""
    try:
        hi = monthly.interp(lat=prism.lat, lon=prism.lon, method=method)
    except Exception:
        hi = monthly.interp(lat=prism.lat, lon=prism.lon, method="linear")
    months = prism.time.dt.month
    return hi.sel(month=months).drop_vars("month").assign_coords(time=prism.time)


def apply_delta(prism_hist: xr.Dataset, deltas: xr.Dataset, method: str = "cubic") -> xr.Dataset:
    """Apply monthly deltas to historical PRISM daily fields -> future projection."""
    dT = _to_prism_daily(deltas["delta_T"], prism_hist, method)
    dP = _to_prism_daily(deltas["delta_P"], prism_hist, method)
    proj = xr.Dataset(
        {
            "T": prism_hist["T"] + dT,
            "P": (prism_hist["P"] * dP).clip(min=0),
        }
    )
    proj["T"].attrs.update(units="degC", long_name="delta-change projected temperature")
    proj["P"].attrs.update(units="mm", long_name="delta-change projected precipitation")
    return proj


def delta_change(
    prism_hist: xr.Dataset,
    coarse_hist: xr.Dataset,
    coarse_fut: xr.Dataset,
    method: str = "cubic",
) -> xr.Dataset:
    """End-to-end delta change: compute coarse deltas and apply to historical PRISM."""
    return apply_delta(prism_hist, monthly_deltas(coarse_hist, coarse_fut), method=method)
