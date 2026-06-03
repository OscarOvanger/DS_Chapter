"""Evaluation metrics for the downscaling testbed.

Because these are *climatological* downscalers (delta change, BCSD, ...), the
projected and the true future weather will never match on a given calendar day --
so day-by-day RMSE is the wrong yardstick. What we care about is whether the
*distribution* and *seasonal climatology* are reproduced. The metrics here are
chosen accordingly and returned as a tidy table so several methods can be
compared side by side.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from .plotting import county_mean, flat_valid


def _percentile_bias(proj: np.ndarray, truth: np.ndarray, q: float) -> float:
    return float(np.nanpercentile(proj, q) - np.nanpercentile(truth, q))


def compare(proj: xr.Dataset, truth: xr.Dataset, var: str, label: str) -> dict:
    """Climatological skill metrics for one downscaler against the truth field."""
    p = flat_valid(proj[var])
    t = flat_valid(truth[var])

    # Seasonal climatology error: county-mean annual cycle, projected vs truth.
    cyc_p = county_mean(proj[var]).groupby("time.month").mean().values
    cyc_t = county_mean(truth[var]).groupby("time.month").mean().values
    clim_rmse = float(np.sqrt(np.nanmean((cyc_p - cyc_t) ** 2)))

    row = {
        "method": label,
        "mean_bias": float(np.nanmean(p) - np.nanmean(t)),
        "std_ratio": float(np.nanstd(p) / np.nanstd(t)),
        "clim_cycle_RMSE": clim_rmse,
        "p95_bias": _percentile_bias(p, t, 95),
        # Wasserstein (earth-mover) distance summarises whole-distribution mismatch.
        "wasserstein": float(stats.wasserstein_distance(p, t)),
    }
    if var == "P":
        # Wet-day frequency error (threshold 1 mm/day), a key precipitation metric.
        row["wetday_freq_bias"] = float((p >= 1).mean() - (t >= 1).mean())
    return row


def compare_table(methods: dict[str, xr.Dataset], truth: xr.Dataset, var: str) -> pd.DataFrame:
    """Build a comparison table: one row per method, metrics vs the truth field.

    ``methods`` maps a display label to that method's projected dataset.
    """
    rows = [compare(ds, truth, var, label) for label, ds in methods.items()]
    return pd.DataFrame(rows).set_index("method").round(3)
