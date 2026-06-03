"""Interactive figures for the notebook (ipywidgets).

Each public function returns a small interactive UI so a notebook cell can be a
single line. The three builders mirror the chapter's first six cells:

* ``delta_monthly_panels(var)``    -- Cells 1 & 2: three monthly climatology maps
  (PRISM baseline, CMIP6 delta-change projection, PRISM truth) with a month slider.
* ``delta_panels(data, var)``      -- legacy four-panel daily ERA5 delta story.
* ``distribution_hist(data, var)`` -- Cells 3 & 4: overlaid histograms of the
  high-res obs, coarse historical, and coarse future distributions.
* ``qm_qdm_explorer(data, var)``   -- Cells 5 & 6: a quantile-mapping explainer with
  a crosshair map, the three CDFs with quantile-matching lines, and QM-vs-QDM
  transfer functions; the grid cell is selectable.

Sliders render live in Jupyter / VS Code. A static viewer (GitHub, nbviewer) shows
the default state, so every cell is built to look right before any slider is touched.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
from ipywidgets import (IntSlider, Layout, SelectionSlider, VBox,
                        interactive_output)

from . import plotting as viz
from .downscalers.delta_change import project_month_from_cache


def _ui(draw, controls: dict):
    """Wire control widgets to a draw function and stack them in a VBox."""
    return VBox([*controls.values(), interactive_output(draw, controls)])

# Per-variable styling / correction kind.
KIND = {"T": "add", "P": "mult"}
UNIT = {"T": "°C", "P": "mm/day"}


# ---------------------------------------------------------------------------
# small numeric helpers (empirical CDF + quantile mapping)
# ---------------------------------------------------------------------------


def _clean(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, float).ravel()
    return a[~np.isnan(a)]


def _cdf_eval(samples: np.ndarray, x: float) -> float:
    """Empirical CDF F(x): fraction of samples <= x."""
    s = np.sort(_clean(samples))
    return float(np.searchsorted(s, x, side="right") / max(s.size, 1))


def _quantile(samples: np.ndarray, tau) -> np.ndarray:
    return np.nanquantile(_clean(samples), np.clip(tau, 0.0, 1.0))


def qm(x, mod_hist, obs_hist, kind):
    """Classic quantile mapping: rank x in the historical model, read off obs."""
    tau = np.array([_cdf_eval(mod_hist, xi) for xi in np.atleast_1d(x)])
    return _quantile(obs_hist, tau)


def qdm(x, mod_hist, mod_fut, obs_hist, kind, eps=1e-6):
    """Quantile Delta Mapping: rank x in the *future* model, transplant its change."""
    x = np.atleast_1d(x)
    tau = np.array([_cdf_eval(mod_fut, xi) for xi in x])
    mh = _quantile(mod_hist, tau)
    oh = _quantile(obs_hist, tau)
    if kind == "mult":
        return np.clip(oh, 0, None) * (x / np.clip(mh, eps, None))
    return oh + (x - mh)


# ---------------------------------------------------------------------------
# date / cell selection helpers
# ---------------------------------------------------------------------------


def _sel_day(ds: xr.Dataset, year: int, month: int, day: int) -> xr.Dataset:
    """Select the calendar day (year, month, day), snapping to the nearest valid date."""
    day = min(day, 28) if month == 2 else day
    target = pd.Timestamp(year=year, month=month, day=min(day, 28) if month == 2 else day)
    return ds.sel(time=target, method="nearest")


def _years(ds: xr.Dataset) -> list[int]:
    return sorted(np.unique(ds.time.dt.year.values).tolist())


# ---------------------------------------------------------------------------
# Cells 1 & 2 -- monthly delta-change (CMIP6 + PRISM)
# ---------------------------------------------------------------------------


def _status(msg: str, progress: bool = True) -> None:
    """Notebook-visible status line (VS Code often buffers tqdm until the cell ends)."""
    if progress:
        print(msg, flush=True)


def delta_monthly_panels(var: str, data: dict | None = None, progress: bool = True):
    """Three map panels with an ipywidgets month slider (for Jupyter).

    For the Quarto chapter, use standalone HTML via ``scripts/export_plotly_figures.py``.
    """
    from tqdm.auto import tqdm

    from . import data_io as io
    from .plotly_panels import build_monthly_cache

    _status(f"[1/4] Starting delta-change panels for {viz.LABEL[var]} …", progress)

    viz.set_style()
    if data is None:
        _status("[2/4] Loading cached PRISM + CMIP6 from data/ …", progress)
        data = io.load_delta_change_data(progress=progress)
    elif progress:
        _status("[2/4] Using pre-loaded data.", progress)

    _status("[3/4] Computing monthly climatologies …", progress)
    if progress:
        with tqdm(total=1, desc="Monthly climatology") as bar:
            bar.set_postfix_str("all datasets", refresh=False)
            cache = build_monthly_cache(data)
            bar.update(1)
    else:
        cache = build_monthly_cache(data)

    viz.ensure_county_outline(progress=progress)

    kind = KIND[var]
    op = "+" if kind == "add" else "×"

    def _draw(month: int):
        baseline, proj, truth = project_month_from_cache(cache, var, month)
        month_name = pd.Timestamp(2000, month, 1).strftime("%B")
        fields = [
            (baseline, f"1. PRISM 1991–2000\n{month_name} mean"),
            (proj, f"2. Delta-change projection\nbaseline {op} CMIP6 Δ"),
            (truth, f"3. PRISM truth 2016–2025\n{month_name} mean"),
        ]
        fig = viz.map_panels(
            fields,
            var,
            suptitle=(
                f"Delta change — {viz.LABEL[var]} "
                f"({'additive' if kind == 'add' else 'multiplicative'})"
            ),
        )
        viz.show_figure(fig)

    _status("[4/4] Building interactive widget (first map: January) …", progress)
    controls = dict(month=IntSlider(1, 1, 12, 1, description="month"))
    ui = _ui(_draw, controls)
    _status("Done — use the month slider below.", progress)
    try:
        from IPython.display import display

        display(ui)
    except ImportError:
        pass
    return ui


# ---------------------------------------------------------------------------
# Legacy daily ERA5 delta-change panels
# ---------------------------------------------------------------------------


def _delta(data: dict, var: str):
    prism_h, prism_f = data["prism_hist"], data["prism_fut"]
    era5_h, era5_f = data["era5_hist"], data["era5_fut"]
    kind = KIND[var]

    # Monthly ERA5 deltas, interpolated to the PRISM grid (computed once).
    hist_m = era5_h[var].groupby("time.month").mean("time")
    fut_m = era5_f[var].groupby("time.month").mean("time")
    delta_m = (fut_m - hist_m) if kind == "add" else (fut_m / hist_m.clip(min=1e-6))
    delta_hi = delta_m.interp(lat=prism_h.lat, lon=prism_h.lon, method="linear")

    cmap = viz.CMAP[var]
    dcmap = viz.CMAP_DIFF[var]
    dlabel = f"Δ{var} ({UNIT[var]})" if kind == "add" else f"{var} ratio (×)"

    def _draw(month, day, base_year, truth_year):
        truth = _sel_day(prism_f, truth_year, month, day)[var]
        baseline = _sel_day(prism_h, base_year, month, day)[var]
        dlt = delta_hi.sel(month=month)
        proj = (baseline + dlt) if kind == "add" else (baseline * dlt).clip(min=0)

        # Shared colour scale for the three same-unit panels (truth/baseline/proj).
        same = [truth, baseline, proj]
        vmin = float(min(np.nanmin(a) for a in same))
        vmax = float(max(np.nanmax(a) for a in same))

        fig, ax = plt.subplots(1, 4, figsize=(16, 3.8), constrained_layout=True)
        date = f"{month:02d}-{day:02d}"
        panels = [
            (truth, f"1. PRISM truth\n{truth_year}-{date}", cmap, vmin, vmax, viz.LABEL[var]),
            (baseline, f"2. PRISM historical\n{base_year}-{date}", cmap, vmin, vmax, viz.LABEL[var]),
            (dlt, f"3. ERA5 Δ (month {month})\nfuture − historical" if kind == "add"
                  else f"3. ERA5 ratio (month {month})\nfuture ÷ historical", dcmap, None, None, dlabel),
            (proj, f"4. Delta-change result\npanel 2 {'+' if kind=='add' else '×'} panel 3",
             cmap, vmin, vmax, viz.LABEL[var]),
        ]
        for a, (arr, title, cm, lo, hi, lab) in zip(ax, panels):
            if lo is None:  # delta panel: symmetric (add) or centred-on-1 (mult)
                if kind == "add":
                    m = float(np.nanmax(np.abs(arr))); lo, hi = -m, m
                else:
                    d = float(np.nanmax(np.abs(arr - 1))); lo, hi = 1 - d, 1 + d
            mesh = a.pcolormesh(arr.lon, arr.lat, arr, cmap=cm, vmin=lo, vmax=hi, shading="auto")
            viz._outline(a)
            a.set_title(title, fontsize=9)
            a.set_xlabel("lon"); a.set_aspect("equal")
            fig.colorbar(mesh, ax=a, shrink=0.8, label=lab)
        ax[0].set_ylabel("lat")
        fig.suptitle(f"Delta change — {viz.LABEL[var]} ({'additive' if kind=='add' else 'multiplicative'})",
                     fontsize=12)
        plt.show()

    hist_years, fut_years = _years(prism_h), _years(prism_f)
    controls = dict(
        month=IntSlider(1, 1, 12, 1, description="month"),
        day=IntSlider(15, 1, 28, 1, description="day"),
        base_year=SelectionSlider(options=hist_years, value=hist_years[0], description="baseline yr"),
        truth_year=SelectionSlider(options=fut_years,
                                   value=fut_years[len(fut_years) // 2], description="truth yr"),
    )
    return _draw, controls


def delta_panels(data: dict, var: str):
    """Four map panels telling the delta-change story, with day/year sliders."""
    return _ui(*_delta(data, var))


# ---------------------------------------------------------------------------
# Cells 3 & 4 -- distribution histograms
# ---------------------------------------------------------------------------


def distribution_hist(data: dict, var: str):
    """Overlaid histograms: high-res obs vs coarse historical vs coarse future."""
    sets = {
        "PRISM 1991–2000 (high-res obs)": viz.flat_valid(data["prism_hist"][var]),
        "ERA5 1991–2000 (coarse historical)": viz.flat_valid(data["era5_hist"][var]),
        "ERA5 2016–2025 (coarse forecast)": viz.flat_valid(data["era5_fut"][var]),
    }
    colors = ["#1b9e77", "#7570b3", "#d95f02"]

    lo = min(a.min() for a in sets.values())
    hi = max(np.nanpercentile(a, 99.5) for a in sets.values())
    bins = np.linspace(lo, hi, 60)

    fig, ax = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    for (label, arr), c in zip(sets.items(), colors):
        ax.hist(arr, bins=bins, density=True, alpha=0.5, color=c, label=label,
                edgecolor="white", linewidth=0.2)
    ax.set(xlabel=viz.LABEL[var], ylabel="density",
           title=f"Daily {viz.LABEL[var].lower()} distributions")
    if var == "P":
        ax.set_yscale("log")  # precipitation is heavily zero-inflated
        ax.set_ylabel("density (log)")
    ax.legend()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Cells 5 & 6 -- QM / QDM explainer
# ---------------------------------------------------------------------------


def _qmqdm(data: dict, var: str, date: str | None = None):
    era5_f, era5_h = data["era5_fut"], data["era5_hist"]
    prism_h = data["prism_hist"]
    kind = KIND[var]
    unit = UNIT[var]

    # Upscale PRISM (the bias-correction target) to the ERA5 grid once.
    obs_coarse = prism_h[var].interp(lat=era5_h.lat, lon=era5_h.lon, method="linear")

    if date is None:
        date = "2020-05-17" if var == "T" else "2018-10-01"
    fut_day = era5_f[var].sel(time=date, method="nearest")
    actual_date = pd.Timestamp(fut_day.time.values).strftime("%Y-%m-%d")
    month = pd.Timestamp(fut_day.time.values).month

    nlat, nlon = era5_f.sizes["lat"], era5_f.sizes["lon"]

    def _draw(i, j):
        la = float(era5_f.lat.values[i]); lo = float(era5_f.lon.values[j])
        x0 = float(fut_day.isel(lat=i, lon=j))

        # Same-month samples at this cell for the three distributions.
        msel = lambda d: d.time.dt.month == month
        obs_h = _clean(obs_coarse.sel(time=msel(obs_coarse)).isel(lat=i, lon=j).values)
        mod_h = _clean(era5_h[var].sel(time=msel(era5_h)).isel(lat=i, lon=j).values)
        mod_f = _clean(era5_f[var].sel(time=msel(era5_f)).isel(lat=i, lon=j).values)

        x_qm = float(qm(x0, mod_h, obs_h, kind)[0])
        x_qdm = float(qdm(x0, mod_h, mod_f, obs_h, kind)[0])
        tau_f = _cdf_eval(mod_f, x0)

        fig = plt.figure(figsize=(11, 10), constrained_layout=True)
        gs = fig.add_gridspec(3, 1, height_ratios=[1.1, 1.0, 1.0])

        # --- (a) ERA5 map with crosshair on the selected cell ------------------
        axm = fig.add_subplot(gs[0])
        mesh = axm.pcolormesh(fut_day.lon, fut_day.lat, fut_day, cmap=viz.CMAP[var], shading="auto")
        viz._outline(axm)
        axm.axhline(la, ls=":", color="k", lw=1); axm.axvline(lo, ls=":", color="k", lw=1)
        axm.plot(lo, la, "x", color="k", ms=12, mew=2.5)
        axm.set_xticks(list(axm.get_xticks()) + [lo]); axm.set_yticks(list(axm.get_yticks()) + [la])
        axm.set(xlabel="lon", ylabel="lat")
        axm.set_xlim(float(fut_day.lon.min()), float(fut_day.lon.max()))
        axm.set_ylim(float(fut_day.lat.min()), float(fut_day.lat.max()))
        fig.colorbar(mesh, ax=axm, shrink=0.85, label=viz.LABEL[var])
        axm.set_title(f"ERA5 forecast {actual_date}  •  cell ({la:.2f}, {lo:.2f})  •  "
                      f"{var} = {x0:.2f} {unit}")

        # --- (b) three CDFs with the quantile-matching construction ------------
        axc = fig.add_subplot(gs[1])
        for arr, c, lab in [(obs_h, "#1b9e77", "PRISM obs (hist)"),
                            (mod_h, "#7570b3", "ERA5 (hist)"),
                            (mod_f, "#d95f02", "ERA5 (future)")]:
            s = np.sort(arr)
            axc.plot(s, np.linspace(0, 1, s.size), color=c, label=lab, lw=2)
        axc.axvline(x0, ls=":", color="#d95f02")
        axc.axhline(tau_f, ls=":", color="grey")
        for xv, c in [(x0, "#d95f02"), (x_qdm, "#1b9e77")]:
            axc.plot([xv, xv], [0, tau_f], ls=":", color=c)
            axc.annotate(f"{xv:.1f}", (xv, 0), textcoords="offset points", xytext=(0, -14),
                         ha="center", color=c, fontsize=9)
        axc.annotate(f"τ = {tau_f:.2f}", (axc.get_xlim()[0], tau_f), textcoords="offset points",
                     xytext=(4, 4), color="grey", fontsize=9)
        axc.set(xlabel=f"{var} ({unit})", ylabel="cumulative probability",
                title="Quantile matching: forecast value → its quantile → observed value")
        axc.legend(loc="lower right")

        # --- (c) QM vs QDM transfer functions ---------------------------------
        axt = fig.add_subplot(gs[2])
        grid = np.linspace(float(np.nanmin(mod_f)), float(np.nanmax(mod_f)), 80)
        axt.plot(grid, grid, ls="--", color="grey", lw=1, label="1:1 (no correction)")
        axt.plot(grid, qm(grid, mod_h, obs_h, kind), color="#377eb8", lw=2,
                 label="QM (→ historical obs quantiles)")
        axt.plot(grid, qdm(grid, mod_h, mod_f, obs_h, kind), color="#e41a1c", lw=2,
                 label="QDM (→ future quantiles + Δ)")
        axt.plot(x0, x_qm, "o", color="#377eb8"); axt.plot(x0, x_qdm, "o", color="#e41a1c")
        axt.axvline(x0, ls=":", color="grey")
        axt.set(xlabel=f"ERA5 forecast value ({unit})", ylabel=f"bias-corrected value ({unit})",
                title=f"Transfer functions at this cell   •   QM: {x_qm:.2f}   QDM: {x_qdm:.2f} {unit}")
        axt.legend(loc="upper left")
        plt.show()

    controls = dict(
        i=IntSlider(nlat // 2, 0, nlat - 1, 1, description="lat idx", layout=Layout(width="350px")),
        j=IntSlider(nlon // 2, 0, nlon - 1, 1, description="lon idx", layout=Layout(width="350px")),
    )
    return _draw, controls


def qm_qdm_explorer(data: dict, var: str, date: str | None = None):
    """Quantile-mapping explainer for one selectable ERA5 cell on a fixed date."""
    return _ui(*_qmqdm(data, var, date))
