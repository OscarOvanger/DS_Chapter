"""Interactive BCSD / Quantile-Delta-Mapping (QDM) explorer for fixed year 2025.

The figure teaches one chain, per PRISM grid cell and calendar month:

    selected future CMIP6 monthly value (year 2025)
        -> percentile tau in the CMIP6 *future* CDF        tau = F_cf_m(x_fut)
        -> same-percentile value in the CMIP6 *historical* CDF   q_ch = Q_ch_m(tau)
        -> same-percentile value in the PRISM *historical* CDF   q_ph = Q_ph_m,cell(tau)
        -> BCSD projected value on the PRISM grid           x_bcsd

QDM equations (tau is *computed*, never user-chosen):

    Temperature (additive delta):
        delta  = x_fut - q_ch
        x_bcsd = q_ph + delta
    Precipitation (multiplicative ratio):
        ratio  = x_fut / max(q_ch, EPS)        # EPS guards divide-by-zero
        ratio  = clip(ratio, 0, RATIO_MAX)     # guard against q_ch ~ 0 blow-ups
        x_bcsd = q_ph * ratio

Three vertically stacked panels:
    1. BCSD-downscaled 2025 monthly field (satellite underlay + heatmap).
    2. Observed PRISM 2025 monthly field (same colour scale for comparison).
    3. Empirical CDFs (PRISM hist, CMIP6 hist, CMIP6 future) for the clicked
       cell/month, with tau, the four quantile markers, dashed guides, and an
       annotation box spelling out the computation.

Interactions: click a PRISM cell, and a native month slider. The future year
is fixed to FIXED_YEAR.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xarray as xr

from . import basemap as bm
from . import config as C
from . import data_io as io
from . import plotting as viz

KIND = {"T": "add", "P": "mult"}
EPS = 1e-6
RATIO_MAX = 5.0  # clip QDM precip ratio so a near-zero q_ch can't blow the field up
TAU_GRID = np.linspace(0.0, 1.0, 41)
FIXED_YEAR = 2025
MAP_OPACITY = 0.78

SERIES = (
    ("PRISM historical (1991–2000)", "#0d9488"),
    ("CMIP6 historical (1991–2000)", "#6366f1"),
    ("CMIP6 future (2016–2025)", "#ea580c"),
)
# Quantile-mapping markers (all at y = tau).
MARKER_LABELS = (
    "CMIP6 future 2025 value",
    "CMIP6 historical at tau",
    "PRISM historical at tau",
    "BCSD projected value",
)
MARKER_COLORS = ("#ea580c", "#6366f1", "#0d9488", "#111827")
MARKER_SYMBOLS = ("circle", "circle", "circle", "diamond")

# Subplot layout (kept in sync with make_subplots below and the JS frame builder).
ROW_HEIGHTS = [0.32, 0.26, 0.42]
VSPACE = 0.09

OUT_PATHS = {
    "T": C.FIGURES / "bcsd_distribution_temperature.html",
    "P": C.FIGURES / "bcsd_distribution_precipitation.html",
}


# ---------------------------------------------------------------------------
# Small numeric helpers
# ---------------------------------------------------------------------------


def _squeeze(da: xr.DataArray) -> xr.DataArray:
    return da.squeeze(drop=True)


def _domains() -> list[tuple[float, float]]:
    """Paper-coordinate [bottom, top] for each subplot row (matches make_subplots)."""
    total = 1.0 - 2.0 * VSPACE
    heights = [r * total for r in ROW_HEIGHTS]
    doms: list[tuple[float, float]] = []
    top = 1.0
    for h in heights:
        bottom = top - h
        doms.append((round(bottom, 4), round(top, 4)))
        top = bottom - VSPACE
    return doms


def _heatmap_ranges(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    dx = (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else 0.02
    dy = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 0.02
    return (
        [float(x[0] - dx / 2), float(x[-1] + dx / 2)],
        [float(y[0] - dy / 2), float(y[-1] + dy / 2)],
    )


def _quantile_curve(samples: np.ndarray) -> list[float]:
    """Empirical inverse-CDF Q(tau) on TAU_GRID (rounded to keep the HTML small)."""
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if s.size == 0:
        return [float("nan")] * TAU_GRID.size
    return [round(float(v), 3) for v in np.quantile(s, TAU_GRID)]


def _tau_of(samples: np.ndarray, x: float | None) -> float:
    """tau = F(x): empirical CDF rank of x within the sample."""
    s = np.asarray(samples, float)
    s = s[np.isfinite(s)]
    if s.size == 0 or x is None or not np.isfinite(x):
        return 0.5
    return float(np.sum(s <= x) / (s.size + 1))


def _value_at(da: xr.DataArray, month: int, year: int) -> float | None:
    sel = da.sel(time=(da.time.dt.year == year) & (da.time.dt.month == month))
    if sel.size == 0:
        return None
    val = float(np.asarray(sel.values, float).ravel()[0])
    return val if np.isfinite(val) else None


# ---------------------------------------------------------------------------
# Cache: everything precomputed so the browser only *reads* values
# ---------------------------------------------------------------------------


def build_distribution_cache(data: dict[str, xr.Dataset]) -> dict:
    prism_h = data["prism_hist"]
    prism_f = data.get("prism_fut")
    cmip6_h = data["cmip6_hist"]
    cmip6_f = data["cmip6_fut"]

    lats = [float(v) for v in prism_h.lat.values]
    lons = [float(v) for v in prism_h.lon.values]
    nlat, nlon = len(lats), len(lons)

    ph_da = {v: prism_h[v] for v in ("T", "P")}
    pf_da = {v: (prism_f[v] if prism_f is not None else None) for v in ("T", "P")}
    # CMIP6 is a single coarse cell covering the whole domain, so its CDFs are
    # spatially constant -- the same curve is reused for every PRISM cell.
    ch_da = {v: _squeeze(cmip6_h[v]) for v in ("T", "P")}
    cf_da = {v: _squeeze(cmip6_f[v]) for v in ("T", "P")}

    prism_cdf: dict[str, list] = {v: [] for v in ("T", "P")}        # [m][lat][lon][tau]
    cmip_hist_cdf: dict[str, list] = {v: [] for v in ("T", "P")}    # [m][tau]
    cmip_fut_cdf: dict[str, list] = {v: [] for v in ("T", "P")}     # [m][tau]
    x_fut: dict[str, list] = {v: [] for v in ("T", "P")}           # [m] scalar
    tau: dict[str, list] = {v: [] for v in ("T", "P")}            # [m] scalar
    q_cmip_hist: dict[str, list] = {v: [] for v in ("T", "P")}     # [m] scalar
    q_prism_hist: dict[str, list] = {v: [] for v in ("T", "P")}    # [m][lat][lon]
    proj_maps: dict[str, list] = {v: [] for v in ("T", "P")}       # [m][lat][lon] x_bcsd
    prism_true: dict[str, list | None] = {v: [] for v in ("T", "P")}  # [m][lat][lon]

    for var in ("T", "P"):
        ph, ch, cf = ph_da[var], ch_da[var], cf_da[var]
        pf = pf_da[var]
        for month in range(1, 13):
            # --- PRISM historical CDF per cell ---
            # Use PRISM *monthly means* (one value per year) so all three
            # distributions live on the same monthly support as CMIP6. This keeps
            # the quantile mapping (q at a shared tau) conceptually consistent.
            p_sel = ph.sel(time=ph.time.dt.month == month)
            p_month = p_sel.groupby("time.year").mean("time").values  # (nyears, nlat, nlon)
            p_q = np.round(np.nanquantile(p_month, TAU_GRID, axis=0), 3)  # (41, nlat, nlon)
            prism_cdf[var].append(np.moveaxis(p_q, 0, -1).tolist())

            # --- CMIP6 historical / future CDFs (single coarse cell) ---
            ch_month = ch.sel(time=ch.time.dt.month == month).values.ravel()
            cf_month = cf.sel(time=cf.time.dt.month == month).values.ravel()
            cmip_hist_cdf[var].append(_quantile_curve(ch_month))
            cmip_fut_cdf[var].append(_quantile_curve(cf_month))

            # --- tau computed from the selected 2025 CMIP6 future realization ---
            xf = _value_at(cf, month, FIXED_YEAR)
            t = _tau_of(cf_month, xf)
            x_fut[var].append(None if xf is None else round(xf, 3))
            tau[var].append(round(t, 4))

            # --- inverse-CDF values at that tau ---
            ch_clean = ch_month[np.isfinite(ch_month)]
            q_ch = float(np.quantile(ch_clean, t)) if ch_clean.size else float("nan")
            q_cmip_hist[var].append(round(q_ch, 3))
            q_ph = np.nanquantile(p_month, t, axis=0)  # (nlat, nlon) PRISM hist at tau
            q_prism_hist[var].append(np.round(q_ph, 3).tolist())

            # --- BCSD projected field for this month (one value per PRISM cell) ---
            if xf is None or not np.isfinite(q_ch):
                field = np.full((nlat, nlon), np.nan)
            elif KIND[var] == "mult":
                ratio = float(np.clip(xf / max(q_ch, EPS), 0.0, RATIO_MAX))
                field = np.clip(q_ph * ratio, 0.0, None)
            else:
                field = q_ph + (xf - q_ch)
            proj_maps[var].append(np.round(field, 3).tolist())

            # --- observed PRISM 2025 monthly mean (truth for comparison) ---
            if pf is not None:
                sel = pf.sel(time=(pf.time.dt.year == FIXED_YEAR) & (pf.time.dt.month == month))
                true_field = sel.mean("time").values if sel.sizes.get("time", 0) else np.full((nlat, nlon), np.nan)
                prism_true[var].append(np.round(np.asarray(true_field, float), 3).tolist())
            else:
                prism_true[var].append(None)

    x = np.asarray(lons, float)
    y = np.asarray(lats, float)
    x_range, y_range = _heatmap_ranges(x, y)
    sat_uri = bm.satellite_basemap_uri(x_range[0], y_range[0], x_range[1], y_range[1])

    # Shared colour scale across BCSD field and PRISM truth so the panels compare.
    z_lim = {}
    for var in ("T", "P"):
        stack = [np.asarray(proj_maps[var], float)]
        if prism_true[var][0] is not None:
            stack.append(np.asarray(prism_true[var], float))
        allz = np.concatenate([s.ravel() for s in stack])
        z_lim[var] = [
            float(np.nanpercentile(allz, 2)),
            float(np.nanpercentile(allz, 98)),
        ]

    return {
        "nlat": nlat,
        "nlon": nlon,
        "lats": lats,
        "lons": lons,
        "fixed_year": FIXED_YEAR,
        "tau_grid": TAU_GRID.tolist(),
        "prism_cdf": prism_cdf,
        "cmip_hist_cdf": cmip_hist_cdf,
        "cmip_fut_cdf": cmip_fut_cdf,
        "x_fut": x_fut,
        "tau": tau,
        "q_cmip_hist": q_cmip_hist,
        "q_prism_hist": q_prism_hist,
        "proj_maps": proj_maps,
        "prism_true": prism_true,
        "has_truth": prism_true["T"][0] is not None,
        "z_lim": z_lim,
        "sat_uri": sat_uri,
        "x_range": x_range,
        "y_range": y_range,
        "domains": [list(d) for d in _domains()],
        "marker_colors": list(MARKER_COLORS),
        "marker_labels": list(MARKER_LABELS),
    }


# ---------------------------------------------------------------------------
# Frame assembly (Python side; mirrored in JS for clicks)
# ---------------------------------------------------------------------------


def _fmt(v: float | None, nd: int = 2) -> str:
    if v is None or not np.isfinite(v):
        return "n/a"
    return f"{v:.{nd}f}"


def _cdf_line(curve: list[float], x0: float, x1: float) -> tuple[list[float], list[float]]:
    """Empirical CDF as a monotone line that runs y=0..1 across the axis range.

    Flat at y=0 from x0 to the smallest value, rises through Q(tau), then flat at
    y=1 out to x1.
    """
    xs = [x0] + list(curve) + [x1]
    ys = [0.0] + TAU_GRID.tolist() + [1.0]
    return xs, ys


def _frame(cache: dict, var: str, month: int, li: int, lj: int) -> dict:
    m = month - 1
    unit = "degC" if var == "T" else "mm/day"
    mon = pd.Timestamp(2000, month, 1).strftime("%b")

    prism = cache["prism_cdf"][var][m][li][lj]
    ch = cache["cmip_hist_cdf"][var][m]
    cf = cache["cmip_fut_cdf"][var][m]
    x_fut = cache["x_fut"][var][m]
    tau = cache["tau"][var][m]
    q_ch = cache["q_cmip_hist"][var][m]
    q_ph = cache["q_prism_hist"][var][m][li][lj]
    x_bcsd = cache["proj_maps"][var][m][li][lj]
    proj_z = cache["proj_maps"][var][m]
    true_z = cache["prism_true"][var][m] if cache["has_truth"] else None

    marker_x = [x_fut, q_ch, q_ph, x_bcsd]

    # Shared x range covering all curves and all markers for this month/cell.
    vals = [v for c in (prism, ch, cf) for v in c if v is not None and np.isfinite(v)]
    vals += [v for v in marker_x if v is not None and np.isfinite(v)]
    if vals:
        xmin, xmax = min(vals), max(vals)
        pad = max((xmax - xmin) * 0.08, 0.3 if var == "T" else 0.1)
        x_range = [xmin - pad, xmax + pad]
    else:
        x_range = [0.0, 1.0]

    px, py = _cdf_line(prism, *x_range)
    cx, cy = _cdf_line(ch, *x_range)
    fx, fy = _cdf_line(cf, *x_range)
    marker_y = [tau] * 4

    shapes = _shapes(cache, li, lj, tau, x_range, marker_x)
    annotations = _annotations(cache, var, mon, tau, x_fut, q_ch, q_ph, x_bcsd, unit)

    return {
        "proj_z": proj_z,
        "true_z": true_z,
        "px": px, "py": py,
        "cx": cx, "cy": cy,
        "fx": fx, "fy": fy,
        "marker_x": marker_x,
        "marker_y": marker_y,
        "x_range": x_range,
        "shapes": shapes,
        "annotations": annotations,
    }


def _shapes(cache, li, lj, tau, x_range, marker_x) -> list[dict]:
    shapes: list[dict] = []
    # Crosshair on both map panels (rows 1 and 2).
    for xref, yref in (("x", "y"), ("x2", "y2")):
        shapes.append({
            "type": "line", "xref": xref, "yref": yref,
            "x0": cache["lons"][lj], "x1": cache["lons"][lj],
            "y0": cache["y_range"][0], "y1": cache["y_range"][1],
            "line": {"color": "white", "width": 1.5, "dash": "dot"},
        })
        shapes.append({
            "type": "line", "xref": xref, "yref": yref,
            "x0": cache["x_range"][0], "x1": cache["x_range"][1],
            "y0": cache["lats"][li], "y1": cache["lats"][li],
            "line": {"color": "white", "width": 1.5, "dash": "dot"},
        })
    # Horizontal dashed guide at y = tau (panel 3).
    shapes.append({
        "type": "line", "xref": "x3", "yref": "y3",
        "x0": x_range[0], "x1": x_range[1], "y0": tau, "y1": tau,
        "line": {"color": "#9ca3af", "width": 1, "dash": "dash"},
    })
    # Vertical dashed guides at each mapped quantile value (panel 3).
    for xv, col in zip(marker_x, MARKER_COLORS):
        if xv is None or not np.isfinite(xv):
            continue
        shapes.append({
            "type": "line", "xref": "x3", "yref": "y3",
            "x0": xv, "x1": xv, "y0": 0, "y1": 1,
            "line": {"color": col, "width": 1, "dash": "dot"},
        })
    return shapes


def _annotations(cache, var, mon, tau, x_fut, q_ch, q_ph, x_bcsd, unit) -> list[dict]:
    d1, d2, d3 = cache["domains"]
    yr = cache["fixed_year"]
    lab = "temperature" if var == "T" else "precipitation"
    titles = [
        dict(text=f"<b>BCSD-downscaled {lab} — {mon} {yr}, tau = {_fmt(tau)}</b>",
             x=0.5, y=d1[1] + 0.022, xref="paper", yref="paper",
             xanchor="center", yanchor="bottom", showarrow=False, font=dict(size=13)),
        dict(text=(f"<b>Observed PRISM monthly field — {mon} {yr}</b>" if cache["has_truth"]
                   else "<b>Observed PRISM 2025 field not available in this dataset</b>"),
             x=0.5, y=d2[1] + 0.014, xref="paper", yref="paper",
             xanchor="center", yanchor="bottom", showarrow=False, font=dict(size=13)),
        dict(text="<b>Empirical CDFs & quantile-delta mapping (click a grid cell)</b>",
             x=0.5, y=d3[1] + 0.014, xref="paper", yref="paper",
             xanchor="center", yanchor="bottom", showarrow=False, font=dict(size=13)),
    ]
    if var == "T":
        delta = None if (x_fut is None or q_ch is None or not np.isfinite(x_fut) or not np.isfinite(q_ch)) else x_fut - q_ch
        box = ("Month: %s<br>Year: %d<br>"
               "tau = F_CMIP_future(x) = %s<br>"
               "CMIP6 future 2025 = %s %s<br>"
               "CMIP6 historical at tau = %s %s<br>"
               "PRISM historical at tau = %s %s<br>"
               "delta = %s %s<br>"
               "<b>BCSD projected = %s %s</b>") % (
            mon, yr, _fmt(tau), _fmt(x_fut), unit, _fmt(q_ch), unit,
            _fmt(q_ph), unit, _fmt(delta), unit, _fmt(x_bcsd), unit)
    else:
        ratio = None
        if x_fut is not None and q_ch is not None and np.isfinite(x_fut) and np.isfinite(q_ch):
            ratio = float(np.clip(x_fut / max(q_ch, EPS), 0.0, RATIO_MAX))
        box = ("Month: %s<br>Year: %d<br>"
               "tau = F_CMIP_future(x) = %s<br>"
               "CMIP6 future 2025 = %s %s<br>"
               "CMIP6 historical at tau = %s %s<br>"
               "PRISM historical at tau = %s %s<br>"
               "ratio = %s<br>"
               "<b>BCSD projected = %s %s</b>") % (
            mon, yr, _fmt(tau), _fmt(x_fut), unit, _fmt(q_ch), unit,
            _fmt(q_ph), unit, _fmt(ratio), _fmt(x_bcsd), unit)
    titles.append(dict(
        text=box, x=0.985, y=d3[0] + 0.015, xref="paper", yref="paper",
        xanchor="right", yanchor="bottom", align="left", showarrow=False,
        font=dict(size=10, color="#111827"),
        bordercolor="#9ca3af", borderwidth=1, borderpad=6,
        bgcolor="rgba(255,255,255,0.82)",
    ))
    return titles


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------


def _slider_steps(cache: dict, var: str, li: int, lj: int) -> list[dict]:
    lons, lats = cache["lons"], cache["lats"]
    steps: list[dict] = []
    for month in range(1, 13):
        fr = _frame(cache, var, month, li, lj)
        steps.append(dict(
            label=pd.Timestamp(2000, month, 1).strftime("%b"),
            method="update",
            args=[
                {
                    "z": [fr["proj_z"], fr["true_z"]],
                    "x": [lons, lons, fr["px"], fr["cx"], fr["fx"], fr["marker_x"]],
                    "y": [lats, lats, fr["py"], fr["cy"], fr["fy"], fr["marker_y"]],
                },
                {
                    "shapes": fr["shapes"],
                    "annotations": fr["annotations"],
                    "xaxis3.range": fr["x_range"],
                },
            ],
        ))
    return steps


def distribution_cdf_figure(
    var: str,
    cache: dict,
    month0: int = 1,
    lat_i0: int | None = None,
    lon_i0: int | None = None,
) -> go.Figure:
    nlat, nlon = cache["nlat"], cache["nlon"]
    li0 = nlat // 2 if lat_i0 is None else lat_i0
    lj0 = nlon // 2 if lon_i0 is None else lon_i0
    label = viz.LABEL[var]
    unit = "degC" if var == "T" else "mm/day"
    x_axis3 = "Monthly temperature (degC)" if var == "T" else "Monthly precipitation (mm/day)"

    fr = _frame(cache, var, month0, li0, lj0)

    fig = make_subplots(
        rows=3,
        cols=1,
        row_heights=ROW_HEIGHTS,
        vertical_spacing=VSPACE,
    )

    # --- Panels 1 & 2: satellite underlay + heatmaps -----------------------
    for row, z, show in ((1, fr["proj_z"], True), (2, fr["true_z"], False)):
        if cache.get("sat_uri"):
            fig.add_layout_image(dict(
                source=cache["sat_uri"], xref="x" if row == 1 else "x2",
                yref="y" if row == 1 else "y2",
                x=cache["x_range"][0], y=cache["y_range"][0],
                sizex=cache["x_range"][1] - cache["x_range"][0],
                sizey=cache["y_range"][1] - cache["y_range"][0],
                xanchor="left", yanchor="bottom", sizing="stretch",
                layer="below", opacity=1.0,
            ), row=row, col=1)
        z_arr = np.asarray(z, float) if z is not None else np.full((nlat, nlon), np.nan)
        fig.add_trace(go.Heatmap(
            x=cache["lons"], y=cache["lats"], z=z_arr,
            zmin=cache["z_lim"][var][0], zmax=cache["z_lim"][var][1],
            colorscale=viz.CMAP[var], opacity=MAP_OPACITY,
            colorbar=dict(title=label, len=0.5, y=0.78, thickness=12) if show else None,
            showscale=show, showlegend=False,
            hovertemplate="lon %{x:.3f}<br>lat %{y:.3f}<br>%{z:.2f}<extra></extra>",
            name="BCSD" if row == 1 else "PRISM truth",
        ), row=row, col=1)

    # --- Panel 3: three empirical CDFs -------------------------------------
    for (name, color), (xs, ys) in zip(
        SERIES, ((fr["px"], fr["py"]), (fr["cx"], fr["cy"]), (fr["fx"], fr["fy"])), strict=True
    ):
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines", name=name,
            line=dict(color=color, width=2.5, shape="linear"),
            hovertemplate=f"%{{x:.2f}} {unit}<br>P=%{{y:.2f}}<extra></extra>",
            legend="legend2",
        ), row=3, col=1)

    # --- Panel 3: quantile-mapping markers (all at y = tau) ----------------
    fig.add_trace(go.Scatter(
        x=fr["marker_x"], y=fr["marker_y"], mode="markers",
        marker=dict(size=11, color=list(MARKER_COLORS), symbol=list(MARKER_SYMBOLS),
                    line=dict(width=1, color="white")),
        customdata=list(MARKER_LABELS),
        hovertemplate="%{customdata}: %{x:.2f} " + unit + "<extra></extra>",
        showlegend=False, legend="legend2",
    ), row=3, col=1)

    fig.update_layout(
        width=760,
        height=1280,
        margin=dict(t=70, b=190, l=64, r=92),
        autosize=True,
        shapes=fr["shapes"],
        annotations=fr["annotations"],
        legend2=dict(
            orientation="h", yanchor="top", y=-0.045, x=0.5, xanchor="center",
            font=dict(size=10),
        ),
        sliders=[dict(
            active=month0 - 1,
            currentvalue={"prefix": "Month: ", "font": {"size": 12}},
            y=-0.11, x=0.5, len=0.9, pad=dict(t=8),
            steps=_slider_steps(cache, var, li0, lj0),
        )],
    )
    for ax, row in (("", 1), ("2", 2)):
        fig.update_xaxes(range=cache["x_range"], showgrid=False, zeroline=False,
                         title_text="Longitude", row=row, col=1)
        fig.update_yaxes(range=cache["y_range"], showgrid=False, zeroline=False,
                         title_text="Latitude", row=row, col=1)
    fig.update_xaxes(title_text=x_axis3, range=fr["x_range"], row=3, col=1,
                     showgrid=True, gridcolor="#eef2f7")
    fig.update_yaxes(title_text="Cumulative probability", range=[0, 1], row=3, col=1,
                     showgrid=True, gridcolor="#eef2f7")

    fig._bcsd_post_script = _post_script(var, li0, lj0)  # type: ignore[attr-defined]
    return fig


# ---------------------------------------------------------------------------
# Client-side click handler (mirrors _frame so clicks recompute everything)
# ---------------------------------------------------------------------------


def _post_script(var: str, li0: int, lj0: int) -> str:
    colors = json.dumps(list(MARKER_COLORS))
    return f"""
window.bcsdState = {{ var: "{var}", li: {li0}, lj: {lj0} }};
const BCSD_COLORS = {colors};
const BCSD_EPS = {EPS}, BCSD_RATIO_MAX = {RATIO_MAX};
const BCSD_MONTHS = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"];

function bcsdFmt(v, nd) {{
  if (v === null || v === undefined || isNaN(v)) return "n/a";
  return v.toFixed(nd === undefined ? 2 : nd);
}}

function bcsdCdfLine(curve, x0, x1, tauGrid) {{
  const xs = [x0].concat(curve, [x1]);
  const ys = [0].concat(tauGrid, [1]);
  return [xs, ys];
}}

function bcsdFrame(c, v, month, li, lj) {{
  const m = month - 1;
  const unit = v === "T" ? "degC" : "mm/day";
  const mon = BCSD_MONTHS[m];
  const prism = c.prism_cdf[v][m][li][lj];
  const ch = c.cmip_hist_cdf[v][m];
  const cf = c.cmip_fut_cdf[v][m];
  const xFut = c.x_fut[v][m];
  const tau = c.tau[v][m];
  const qCh = c.q_cmip_hist[v][m];
  const qPh = c.q_prism_hist[v][m][li][lj];
  const xBcsd = c.proj_maps[v][m][li][lj];
  const projZ = c.proj_maps[v][m];
  const trueZ = c.has_truth ? c.prism_true[v][m] : null;
  const markerX = [xFut, qCh, qPh, xBcsd];

  const vals = [];
  [prism, ch, cf].forEach((cu) => cu.forEach((x) => {{ if (x !== null && !isNaN(x)) vals.push(x); }}));
  markerX.forEach((x) => {{ if (x !== null && !isNaN(x)) vals.push(x); }});
  let xRange;
  if (vals.length) {{
    const xmin = Math.min(...vals), xmax = Math.max(...vals);
    const pad = Math.max((xmax - xmin) * 0.08, v === "P" ? 0.1 : 0.3);
    xRange = [xmin - pad, xmax + pad];
  }} else xRange = [0, 1];

  const [px, py] = bcsdCdfLine(prism, xRange[0], xRange[1], c.tau_grid);
  const [cx, cy] = bcsdCdfLine(ch, xRange[0], xRange[1], c.tau_grid);
  const [fx, fy] = bcsdCdfLine(cf, xRange[0], xRange[1], c.tau_grid);
  const markerY = [tau, tau, tau, tau];

  const shapes = bcsdShapes(c, li, lj, tau, xRange, markerX);
  const annotations = bcsdAnnotations(c, v, mon, tau, xFut, qCh, qPh, xBcsd, unit);

  return {{ projZ, trueZ, px, py, cx, cy, fx, fy, markerX, markerY, xRange, shapes, annotations }};
}}

function bcsdShapes(c, li, lj, tau, xRange, markerX) {{
  const shapes = [];
  [["x","y"],["x2","y2"]].forEach(([xr, yr]) => {{
    shapes.push({{ type:"line", xref:xr, yref:yr, x0:c.lons[lj], x1:c.lons[lj],
      y0:c.y_range[0], y1:c.y_range[1], line:{{color:"white", width:1.5, dash:"dot"}} }});
    shapes.push({{ type:"line", xref:xr, yref:yr, x0:c.x_range[0], x1:c.x_range[1],
      y0:c.lats[li], y1:c.lats[li], line:{{color:"white", width:1.5, dash:"dot"}} }});
  }});
  shapes.push({{ type:"line", xref:"x3", yref:"y3", x0:xRange[0], x1:xRange[1],
    y0:tau, y1:tau, line:{{color:"#9ca3af", width:1, dash:"dash"}} }});
  markerX.forEach((xv, i) => {{
    if (xv === null || isNaN(xv)) return;
    shapes.push({{ type:"line", xref:"x3", yref:"y3", x0:xv, x1:xv, y0:0, y1:1,
      line:{{color:BCSD_COLORS[i], width:1, dash:"dot"}} }});
  }});
  return shapes;
}}

function bcsdAnnotations(c, v, mon, tau, xFut, qCh, qPh, xBcsd, unit) {{
  const d = c.domains, yr = c.fixed_year;
  const lab = v === "T" ? "temperature" : "precipitation";
  const titles = [
    {{ text:"<b>BCSD-downscaled "+lab+" — "+mon+" "+yr+", tau = "+bcsdFmt(tau)+"</b>",
       x:0.5, y:d[0][1]+0.022, xref:"paper", yref:"paper", xanchor:"center",
       yanchor:"bottom", showarrow:false, font:{{size:13}} }},
    {{ text:(c.has_truth ? "<b>Observed PRISM monthly field — "+mon+" "+yr+"</b>"
                          : "<b>Observed PRISM 2025 field not available in this dataset</b>"),
       x:0.5, y:d[1][1]+0.014, xref:"paper", yref:"paper", xanchor:"center",
       yanchor:"bottom", showarrow:false, font:{{size:13}} }},
    {{ text:"<b>Empirical CDFs & quantile-delta mapping (click a grid cell)</b>",
       x:0.5, y:d[2][1]+0.014, xref:"paper", yref:"paper", xanchor:"center",
       yanchor:"bottom", showarrow:false, font:{{size:13}} }},
  ];
  let box;
  if (v === "T") {{
    const delta = (xFut === null || qCh === null || isNaN(xFut) || isNaN(qCh)) ? null : xFut - qCh;
    box = "Month: "+mon+"<br>Year: "+yr+"<br>tau = F_CMIP_future(x) = "+bcsdFmt(tau)
        +"<br>CMIP6 future 2025 = "+bcsdFmt(xFut)+" "+unit
        +"<br>CMIP6 historical at tau = "+bcsdFmt(qCh)+" "+unit
        +"<br>PRISM historical at tau = "+bcsdFmt(qPh)+" "+unit
        +"<br>delta = "+bcsdFmt(delta)+" "+unit
        +"<br><b>BCSD projected = "+bcsdFmt(xBcsd)+" "+unit+"</b>";
  }} else {{
    let ratio = null;
    if (xFut !== null && qCh !== null && !isNaN(xFut) && !isNaN(qCh))
      ratio = Math.max(0, Math.min(BCSD_RATIO_MAX, xFut / Math.max(qCh, BCSD_EPS)));
    box = "Month: "+mon+"<br>Year: "+yr+"<br>tau = F_CMIP_future(x) = "+bcsdFmt(tau)
        +"<br>CMIP6 future 2025 = "+bcsdFmt(xFut)+" "+unit
        +"<br>CMIP6 historical at tau = "+bcsdFmt(qCh)+" "+unit
        +"<br>PRISM historical at tau = "+bcsdFmt(qPh)+" "+unit
        +"<br>ratio = "+bcsdFmt(ratio)
        +"<br><b>BCSD projected = "+bcsdFmt(xBcsd)+" "+unit+"</b>";
  }}
  titles.push({{ text:box, x:0.985, y:d[2][0]+0.015, xref:"paper", yref:"paper",
    xanchor:"right", yanchor:"bottom", align:"left", showarrow:false,
    font:{{size:10, color:"#111827"}}, bordercolor:"#9ca3af", borderwidth:1,
    borderpad:6, bgcolor:"rgba(255,255,255,0.82)" }});
  return titles;
}}

function bcsdBuildSteps(c, v, li, lj) {{
  const steps = [];
  for (let month = 1; month <= 12; month++) {{
    const fr = bcsdFrame(c, v, month, li, lj);
    steps.push({{
      label: BCSD_MONTHS[month - 1], method: "update",
      args: [
        {{ z:[fr.projZ, fr.trueZ], x:[c.lons, c.lons, fr.px, fr.cx, fr.fx, fr.markerX],
           y:[c.lats, c.lats, fr.py, fr.cy, fr.fy, fr.markerY] }},
        {{ shapes: fr.shapes, annotations: fr.annotations, "xaxis3.range": fr.xRange }},
      ],
    }});
  }}
  return steps;
}}

(function() {{
  const gd = document.querySelector(".plotly-graph-div");
  if (!gd || !window.bcsdCache) return;
  gd.on("plotly_click", function(evt) {{
    if (!evt.points.length || gd.data[evt.points[0].curveNumber].type !== "heatmap") return;
    const c = window.bcsdCache, s = window.bcsdState, pt = evt.points[0];
    let bestI = 0, bestJ = 0, bestD = Infinity;
    for (let i = 0; i < c.nlat; i++) {{
      for (let j = 0; j < c.nlon; j++) {{
        const dd = (c.lats[i] - pt.y) ** 2 + (c.lons[j] - pt.x) ** 2;
        if (dd < bestD) {{ bestD = dd; bestI = i; bestJ = j; }}
      }}
    }}
    s.li = bestI; s.lj = bestJ;
    const steps = bcsdBuildSteps(c, s.var, s.li, s.lj);
    const m = (gd.layout.sliders && gd.layout.sliders[0]) ? gd.layout.sliders[0].active : 0;
    Plotly.update(gd, steps[m].args[0], Object.assign({{}}, steps[m].args[1], {{
      "sliders[0].steps": steps, "sliders[0].active": m,
    }}));
  }});
}})();
"""


def export_distribution_html(
    var: str,
    data: dict[str, xr.Dataset] | None = None,
    cache: dict | None = None,
) -> Path:
    out = OUT_PATHS[var]
    if data is None:
        data = io.load_delta_change_data(progress=True)
    if cache is None:
        cache = build_distribution_cache(data)
    fig = distribution_cdf_figure(var, cache)
    post_script = getattr(fig, "_bcsd_post_script", "")
    cache_json = json.dumps(cache)

    out.parent.mkdir(parents=True, exist_ok=True)
    html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        config={"scrollZoom": True, "displayModeBar": True, "responsive": True},
        post_script=f"window.bcsdCache = {cache_json};\n{post_script}",
    )
    out.write_text(html, encoding="utf-8")
    return out


def export_all(data: dict[str, xr.Dataset] | None = None) -> list[Path]:
    if data is None:
        data = io.load_delta_change_data(progress=True)
    cache = build_distribution_cache(data)
    return [export_distribution_html(var, data=data, cache=cache) for var in ("T", "P")]
