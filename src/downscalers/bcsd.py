"""BCSD -- Bias-Corrected Spatial Disaggregation (Wood et al., 2004).

Delta change has two weaknesses BCSD targets: (1) it only corrects the *mean*
shift, ignoring how the model misrepresents the rest of the distribution, and
(2) it just reuses the historical day sequence. BCSD instead works on the whole
distribution and lets the model's actual future variability through, in two steps.

Step 1 -- Bias correction (coarse scale), via Quantile Delta Mapping (QDM):
    We first upscale the high-res observations (PRISM) to the ERA5 grid. For each
    coarse cell and calendar month we then quantile-map the model: a future ERA5
    value is ranked within the future model distribution, the model's change at
    that quantile is measured, and that change is transplanted onto the observed
    distribution. This corrects ERA5's distributional biases while preserving its
    projected trend -- additively for temperature, multiplicatively for precip.

Step 2 -- Spatial disaggregation:
    The bias-corrected coarse field is expressed as a factor relative to the coarse
    observed climatology, interpolated to the fine PRISM grid, and applied to the
    fine-scale PRISM climatology -- which carries the ~4 km spatial detail (terrain,
    urban heat) that the coarse model can never see.

Note: here ERA5 reanalysis stands in for a GCM. Because the "future" ERA5 is real
reanalysis, BCSD can recover actual day-to-day weather -- with a true GCM you would
only expect the statistics to match, not individual days.
"""
from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.stats import rankdata

# Per-variable correction style: temperature additive, precipitation multiplicative.
KIND = {"T": "add", "P": "mult"}


def _qdm_1d(obs_h: np.ndarray, mod_h: np.ndarray, mod_f: np.ndarray,
            kind: str, eps: float = 1e-6) -> np.ndarray:
    """Quantile Delta Mapping for one cell / month (1-D time series)."""
    obs_h = obs_h[~np.isnan(obs_h)]
    mod_h = mod_h[~np.isnan(mod_h)]
    out = np.full(mod_f.shape, np.nan)
    valid = ~np.isnan(mod_f)
    if obs_h.size == 0 or mod_h.size == 0 or not valid.any():
        return out

    mf = mod_f[valid]
    tau = rankdata(mf) / (mf.size + 1)          # quantile of each future value
    mod_h_q = np.quantile(mod_h, tau)            # model value at that quantile (hist)
    obs_h_q = np.quantile(obs_h, tau)            # obs value at that quantile (hist)

    if kind == "mult":
        delta = mf / np.clip(mod_h_q, eps, None)         # model's relative change
        out[valid] = np.clip(np.clip(obs_h_q, 0, None) * delta, 0, None)
    else:
        delta = mf - mod_h_q                              # model's absolute change
        out[valid] = obs_h_q + delta
    return out


def qdm_bias_correct(obs_coarse_hist: xr.DataArray, mod_hist: xr.DataArray,
                     mod_fut: xr.DataArray, kind: str) -> xr.DataArray:
    """Quantile-map the future coarse model onto observed statistics, per cell & month."""
    out = mod_fut.copy()
    fmonth = mod_fut.time.dt.month.values
    for mo in range(1, 13):
        fsel = fmonth == mo
        if not fsel.any():
            continue
        o = obs_coarse_hist.sel(time=obs_coarse_hist.time.dt.month == mo).values
        m = mod_hist.sel(time=mod_hist.time.dt.month == mo).values
        f = mod_fut.sel(time=fsel).values
        corr = np.empty_like(f)
        for i in range(f.shape[1]):
            for j in range(f.shape[2]):
                corr[:, i, j] = _qdm_1d(o[:, i, j], m[:, i, j], f[:, i, j], kind)
        out.values[fsel] = corr
    return out


def _clim_on(times, monthly_clim: xr.DataArray) -> xr.DataArray:
    """Broadcast a (month, lat, lon) climatology onto a daily time axis."""
    return (
        monthly_clim.sel(month=times.dt.month)
        .drop_vars("month")
        .assign_coords(time=times)
    )


# Precipitation safeguards for the multiplicative ratio (Wood et al. discuss the
# divide-by-near-zero problem): floor the climatological denominator at a physical
# rate and cap the resulting multiplier so a dry climatology can't produce runaway
# values. The cap is generous, so it only trims pathological ratios.
_P_DENOM_FLOOR = 0.5   # mm/day
_P_FACTOR_CAP = 15.0


def spatial_disaggregate(bc_coarse_fut: xr.DataArray, obs_coarse_hist: xr.DataArray,
                         prism_var_hist: xr.DataArray, kind: str) -> xr.DataArray:
    """Distribute the bias-corrected coarse field onto the fine PRISM grid."""
    obs_clim = obs_coarse_hist.groupby("time.month").mean("time")
    obs_clim_daily = _clim_on(bc_coarse_fut.time, obs_clim)

    # Coarse "factor": how the bias-corrected future departs from the obs climatology.
    if kind == "mult":
        factor = (bc_coarse_fut / obs_clim_daily.clip(min=_P_DENOM_FLOOR)).clip(0, _P_FACTOR_CAP)
    else:
        factor = bc_coarse_fut - obs_clim_daily

    factor_fine = factor.interp(lat=prism_var_hist.lat, lon=prism_var_hist.lon, method="linear")

    # Fine climatology supplies the high-resolution spatial pattern.
    prism_clim = prism_var_hist.groupby("time.month").mean("time")
    prism_clim_daily = _clim_on(bc_coarse_fut.time, prism_clim)

    if kind == "mult":
        return (prism_clim_daily * factor_fine.clip(0, _P_FACTOR_CAP)).clip(min=0)
    return prism_clim_daily + factor_fine


def bcsd(prism_hist: xr.Dataset, era5_hist: xr.Dataset, era5_fut: xr.Dataset) -> xr.Dataset:
    """End-to-end BCSD: QDM bias correction + spatial disaggregation for T and P.

    Returns a fine-scale future projection on the future calendar, ready to compare
    against the true future PRISM field.
    """
    # Upscale observations (PRISM) to the coarse ERA5 grid for the bias-correction step.
    obs_coarse = prism_hist.interp(lat=era5_hist.lat, lon=era5_hist.lon, method="linear")

    out = {}
    for var, kind in KIND.items():
        bc = qdm_bias_correct(obs_coarse[var], era5_hist[var], era5_fut[var], kind)
        out[var] = spatial_disaggregate(bc, obs_coarse[var], prism_hist[var], kind)

    ds = xr.Dataset(out)
    ds["T"].attrs.update(units="degC", long_name="BCSD projected temperature")
    ds["P"].attrs.update(units="mm", long_name="BCSD projected precipitation")
    return ds
