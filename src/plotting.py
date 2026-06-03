"""Reusable plotting helpers so the notebook cells stay short and visual.

Every function takes already-computed xarray fields and returns a Matplotlib
figure. The goal is consistent, readable visuals (shared colour scales, a county
outline, sensible colormaps) without repeating boilerplate in each notebook cell.
"""
from __future__ import annotations

import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import xarray as xr

from . import config as C

# Diverging maps for temperature (RdBu_r) and a colour-blind-safe brown/green
# for precipitation (BrBG); sequential maps for absolute fields.
CMAP = {"T": "RdYlBu_r", "P": "YlGnBu"}
CMAP_DIFF = {"T": "RdBu_r", "P": "BrBG"}
LABEL = {"T": "Temperature (degC)", "P": "Precipitation (mm/day)"}

_COUNTY = None


def set_style() -> None:
    """Apply a clean, consistent look for all figures in the notebook."""
    sns.set_theme(context="notebook", style="whitegrid")
    plt.rcParams.update({"figure.dpi": 110, "axes.titlesize": 11, "figure.titlesize": 13})


def ensure_county_outline(progress: bool = False) -> None:
    """Load the Travis County outline once (local cache preferred)."""
    _status("Loading Travis County outline …", progress)
    _county()


def _county():
    """Travis County outline (cached) for overlaying on maps."""
    global _COUNTY
    if _COUNTY is None:
        if C.TRAVIS_COUNTY_GEOJSON.exists():
            _COUNTY = gpd.read_file(C.TRAVIS_COUNTY_GEOJSON)
        else:
            url = "https://www2.census.gov/geo/tiger/GENZ2023/shp/cb_2023_us_county_20m.zip"
            _COUNTY = (
                gpd.read_file(url)
                .query(f"STATEFP == '{C.STATE_FIPS}' and COUNTYFP == '{C.COUNTY_FIPS}'")
                .to_crs(4326)
            )
    return _COUNTY


def _status(msg: str, enabled: bool = True) -> None:
    """Print setup status (flushed; visible in VS Code Jupyter while the cell runs)."""
    if not enabled:
        return
    print(msg, flush=True)


def _outline(ax) -> None:
    try:
        _county().boundary.plot(ax=ax, color="k", linewidth=0.8)
    except Exception:
        pass  # never let a network hiccup break a plot


def county_mean(ds: xr.Dataset | xr.DataArray):
    """Spatial average over the county footprint (ignoring NaN cells)."""
    return ds.mean(("lat", "lon"), skipna=True)


# ---------------------------------------------------------------------------
# Spatial map panels
# ---------------------------------------------------------------------------


def map_panels(fields: list[tuple[xr.DataArray, str]], var: str, suptitle: str = "",
               diff: bool = False, symmetric: bool = False):
    """Plot a row of spatial maps that share one colour scale.

    ``fields`` is a list of ``(DataArray, title)``. Set ``diff=True`` for
    difference maps (diverging colormap) and ``symmetric=True`` to centre the
    colour scale on zero.
    """
    cmap = (CMAP_DIFF if diff else CMAP)[var]
    arrs = [a for a, _ in fields]
    vmin = float(min(np.nanmin(a) for a in arrs))
    vmax = float(max(np.nanmax(a) for a in arrs))
    if symmetric:
        m = max(abs(vmin), abs(vmax))
        vmin, vmax = -m, m

    n = len(fields)
    fig, axes = plt.subplots(1, n, figsize=(4.1 * n, 3.6), constrained_layout=True)
    axes = np.atleast_1d(axes)
    mesh = None
    for ax, (arr, title) in zip(axes, fields):
        mesh = ax.pcolormesh(arr.lon, arr.lat, arr, cmap=cmap, vmin=vmin, vmax=vmax, shading="auto")
        _outline(ax)
        ax.set_title(title)
        ax.set_xlabel("lon")
        ax.set_aspect("equal")
    axes[0].set_ylabel("lat")
    label = LABEL[var] if not diff else f"Δ {LABEL[var]}"
    fig.colorbar(mesh, ax=axes, shrink=0.85, label=label)
    if suptitle:
        fig.suptitle(suptitle)
    return fig


def show_figure(fig) -> None:
    """Display a figure in Jupyter without blocking on ``plt.show()``."""
    try:
        from IPython.display import display

        display(fig)
    except ImportError:
        plt.show()
    else:
        plt.close(fig)


# ---------------------------------------------------------------------------
# Distribution / seasonal-cycle diagnostics
# ---------------------------------------------------------------------------


def monthly_cycle(series: dict[str, xr.Dataset], var: str, suptitle: str = ""):
    """Mean annual cycle (county mean by calendar month) for each labelled field."""
    fig, ax = plt.subplots(figsize=(8, 4), constrained_layout=True)
    markers = ["o", "s", "^", "D", "v"]
    for (label, ds), mk in zip(series.items(), markers):
        m = county_mean(ds[var]).groupby("time.month").mean()
        ax.plot(m.month, m.values, marker=mk, label=label)
    ax.set(xlabel="Month", ylabel=LABEL[var], xticks=range(1, 13))
    ax.legend()
    if suptitle:
        ax.set_title(suptitle)
    return fig


def ecdf(arrays: dict[str, np.ndarray], var: str, suptitle: str = ""):
    """Empirical CDF for each labelled field -- shows how the distribution shifts."""
    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for label, arr in arrays.items():
        a = np.asarray(arr).ravel()
        a = a[~np.isnan(a)]
        sns.ecdfplot(x=a, label=label, ax=ax)
    ax.set(xlabel=LABEL[var], ylabel="Cumulative probability")
    ax.legend()
    ax.grid(True, alpha=0.3)
    if suptitle:
        ax.set_title(suptitle)
    return fig


def flat_valid(da: xr.DataArray) -> np.ndarray:
    """Flatten a DataArray to its finite values (for histograms / ECDFs)."""
    a = np.asarray(da.values).ravel()
    return a[~np.isnan(a)]
