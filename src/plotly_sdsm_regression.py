"""Interactive SDSM-style linear regression demo for Travis County.

Fits monthly OLS with seasonal harmonics at selected PRISM cells:

    y_t = beta_0 + beta_1 * x_t + beta_2 * sin(2*pi*m/12) + beta_3 * cos(2*pi*m/12) + epsilon_t

Temperature uses PRISM and CMIP6 directly; precipitation uses log1p transforms.
Exports standalone Plotly HTML with a dropdown over four representative grid cells.
"""
from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import statsmodels.api as sm
from scipy import stats as sp_stats
import xarray as xr

from . import basemap as bm
from . import config as C
from . import data_io as io
from . import plotting as viz

OUT_PATHS = {
    "T": C.SDSM_REGRESSION_TEMPERATURE_HTML,
    "P": C.SDSM_REGRESSION_PRECIPITATION_HTML,
}

COLOR_PRISM = "#0d9488"
COLOR_CMIP6 = "#6366f1"
COLOR_PROJ = "#ea580c"
COLOR_MARKER = "#dc2626"

COEF_NAMES = (
    "beta_0 (intercept)",
    "beta_1 (predictor)",
    "beta_2 (sin month)",
    "beta_3 (cos month)",
)
COEF_NAMES_P = (
    "beta_0 (intercept)",
    "beta_1 (log1p CMIP6 monthly)",
    "beta_2 (sin month)",
    "beta_3 (cos month)",
)


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------


def _squeeze(da: xr.DataArray) -> xr.DataArray:
    return da.squeeze(drop=True)


def _monthly_prism(var: str, da: xr.DataArray) -> xr.DataArray:
    """Aggregate daily PRISM to monthly means (T) or monthly totals (P)."""
    if var == "P":
        return da.resample(time="MS").sum()
    return da.resample(time="MS").mean()


def _monthly_cmip6(var: str, da: xr.DataArray) -> xr.DataArray:
    """CMIP6 monthly T (degC) or P converted to monthly total (mm)."""
    da = _squeeze(da)
    if var == "P":
        return da * da.time.dt.days_in_month
    return da


def _ylabel(var: str) -> str:
    if var == "T":
        return "Temperature (degC)"
    return "Monthly precipitation (mm)"


def _heatmap_ranges(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    dx = (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else 0.02
    dy = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 0.02
    return (
        [float(x[0] - dx / 2), float(x[-1] + dx / 2)],
        [float(y[0] - dy / 2), float(y[-1] + dy / 2)],
    )


def _transform(var: str, x: np.ndarray | pd.Series) -> np.ndarray:
    """Identity for temperature; log1p for monthly precipitation totals."""
    arr = np.asarray(x, dtype=float)
    if var == "P":
        return np.log1p(np.clip(arr, 0, None))
    return arr


def _inverse_transform(var: str, x: np.ndarray) -> np.ndarray:
    if var == "P":
        return np.expm1(x)
    return x


def _design_matrix(df: pd.DataFrame, x_col: str = "x_fit") -> np.ndarray:
    months = df["month"].values
    return np.column_stack(
        [
            np.ones(len(df)),
            df[x_col].values,
            np.sin(2 * np.pi * months / 12),
            np.cos(2 * np.pi * months / 12),
        ]
    )


def _select_center_cell(prism_hist: xr.Dataset) -> tuple[int, int]:
    center_lon = (C.TRAVIS_BBOX[0] + C.TRAVIS_BBOX[2]) / 2
    center_lat = (C.TRAVIS_BBOX[1] + C.TRAVIS_BBOX[3]) / 2
    li = int(np.argmin(np.abs(prism_hist.lat.values - center_lat)))
    lj = int(np.argmin(np.abs(prism_hist.lon.values - center_lon)))
    return li, lj


def _monthly_cell_series(
    prism_monthly: xr.DataArray,
    cmip6_monthly: xr.DataArray,
    period: tuple[str, str],
) -> pd.DataFrame:
    start, end = period
    y = prism_monthly.sel(time=slice(start, end))
    x = cmip6_monthly.sel(time=slice(start, end))
    y_s = y.to_series()
    x_s = x.to_series()
    y_s.index = y_s.index.to_period("M")
    x_s.index = x_s.index.to_period("M")
    df = pd.DataFrame({"y": y_s, "x": x_s}).dropna()
    df["month"] = df.index.month
    return df


def _county_boundary_traces() -> list[go.Scatter]:
    if not C.TRAVIS_COUNTY_GEOJSON.exists():
        return []
    traces: list[go.Scatter] = []
    gdf = gpd.read_file(C.TRAVIS_COUNTY_GEOJSON)
    for geom in gdf.geometry:
        if geom is None:
            continue
        polys = [geom] if geom.geom_type == "Polygon" else list(geom.geoms)
        for poly in polys:
            xs, ys = poly.exterior.xy
            traces.append(
                go.Scatter(
                    x=list(xs),
                    y=list(ys),
                    mode="lines",
                    line=dict(color="white", width=1.5),
                    hoverinfo="skip",
                    showlegend=False,
                )
            )
    return traces


def _qq_data(residuals: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    resid = np.asarray(residuals, float)
    n = resid.size
    theoretical = sp_stats.norm.ppf((np.arange(1, n + 1) - 0.5) / n)
    sorted_resid = np.sort(resid)
    q1_t, q3_t = np.quantile(theoretical, [0.25, 0.75])
    q1_r, q3_r = np.quantile(sorted_resid, [0.25, 0.75])
    slope = (q3_r - q1_r) / (q3_t - q1_t) if q3_t != q1_t else 1.0
    intercept = q1_r - slope * q1_t
    x_qq = np.array([theoretical.min(), theoretical.max()])
    y_qq = intercept + slope * x_qq
    return theoretical, sorted_resid, y_qq


def _obs_fit_range(y_obs: np.ndarray, y_hat: np.ndarray) -> tuple[float, float]:
    lo = float(min(y_obs.min(), y_hat.min()) - 0.5)
    hi = float(max(y_obs.max(), y_hat.max()) + 0.5)
    if hi - lo < 1e-6:
        lo, hi = lo - 0.5, hi + 0.5
    return lo, hi


# ---------------------------------------------------------------------------
# Cell fitting
# ---------------------------------------------------------------------------


def _fit_cell(
    var: str,
    li: int,
    lj: int,
    prism_hist_m: xr.DataArray,
    prism_fut_m: xr.DataArray,
    cmip6_hist_v: xr.DataArray,
    cmip6_fut_v: xr.DataArray,
    prism_hist: xr.Dataset,
) -> dict:
    cell_hist = prism_hist_m.isel(lat=li, lon=lj)
    cell_fut = prism_fut_m.isel(lat=li, lon=lj)
    df_hist = _monthly_cell_series(cell_hist, cmip6_hist_v, C.HIST)
    df_fut = _monthly_cell_series(cell_fut, cmip6_fut_v, C.FUT)

    df_hist = df_hist.copy()
    df_fut = df_fut.copy()
    df_hist["y_fit"] = _transform(var, df_hist["y"])
    df_hist["x_fit"] = _transform(var, df_hist["x"])
    df_fut["y_fit"] = _transform(var, df_fut["y"])
    df_fut["x_fit"] = _transform(var, df_fut["x"])

    X_hist = _design_matrix(df_hist)
    y_fit_hist = df_hist["y_fit"].values
    model = sm.OLS(y_fit_hist, X_hist).fit()

    y_hat_fit_hist = model.predict(X_hist)
    residuals_fit = y_fit_hist - y_hat_fit_hist

    y_hist_disp = df_hist["y"].values
    y_hat_hist_disp = _inverse_transform(var, y_hat_fit_hist)

    X_fut = _design_matrix(df_fut)
    y_hat_fit_fut = model.predict(X_fut)
    y_hat_fut_disp = _inverse_transform(var, y_hat_fit_fut)
    y_fut_disp = df_fut["y"].values

    hist_times = [t.isoformat() for t in df_hist.index.to_timestamp()]
    fut_times = [t.isoformat() for t in df_fut.index.to_timestamp()]

    return {
        "li": li,
        "lj": lj,
        "cell_lat": round(float(prism_hist.lat.values[li]), 4),
        "cell_lon": round(float(prism_hist.lon.values[lj]), 4),
        "n_hist": len(df_hist),
        "y_hist": [round(v, 4) for v in y_hist_disp],
        "y_hat_hist": [round(v, 4) for v in y_hat_hist_disp],
        "residuals": [round(v, 4) for v in residuals_fit],
        "y_fut": [round(v, 4) if np.isfinite(v) else None for v in y_fut_disp],
        "y_hat_fut": [round(v, 4) for v in y_hat_fut_disp],
        "hist_times": hist_times,
        "fut_times": fut_times,
        "params": [round(v, 4) for v in model.params],
        "rsquared": round(float(model.rsquared), 4),
        "resid_sd": round(float(np.sqrt(model.mse_resid)), 4),
        "hist_mean_y": round(float(np.mean(y_hist_disp)), 3),
        "fut_mean_y_hat": round(float(np.mean(y_hat_fut_disp)), 3),
    }


def _pick_representative_cells(
    var: str,
    prism_hist: xr.Dataset,
    prism_hist_m: xr.DataArray,
    prism_fut_m: xr.DataArray,
    cmip6_hist_v: xr.DataArray,
    cmip6_fut_v: xr.DataArray,
) -> list[tuple[str, str, int, int]]:
    lats = prism_hist.lat.values
    lons = prism_hist.lon.values
    li_c, lj_c = _select_center_cell(prism_hist)
    center_lat = float(lats[li_c])
    lj_w = int(np.argmin(np.abs(lons - lons.min())))
    lj_e = int(np.argmin(np.abs(lons - lons.max())))
    li_w = int(np.argmin(np.abs(lats - center_lat)))
    li_e = int(np.argmin(np.abs(lats - center_lat)))

    worst_r2 = float("inf")
    worst_li, worst_lj = li_c, lj_c
    for li in range(len(lats)):
        for lj in range(len(lons)):
            try:
                r = _fit_cell(
                    var, li, lj, prism_hist_m, prism_fut_m,
                    cmip6_hist_v, cmip6_fut_v, prism_hist,
                )
                if r["rsquared"] < worst_r2:
                    worst_r2 = r["rsquared"]
                    worst_li, worst_lj = li, lj
            except Exception:
                continue

    return [
        ("center", "Central Austin", li_c, lj_c),
        ("west", "Western Travis (cooler edge)", li_w, lj_w),
        ("east", "Eastern Travis (warmer edge)", li_e, lj_e),
        ("weak", "Lower R² example", worst_li, worst_lj),
    ]


def build_sdsm_cache(data: dict[str, xr.Dataset] | None = None) -> dict:
    if data is None:
        data = io.load_delta_change_data(progress=True)

    prism_hist = data["prism_hist"]
    prism_fut = data["prism_fut"]
    cmip6_hist = data["cmip6_hist"]
    cmip6_fut = data["cmip6_fut"]

    lats = [float(v) for v in prism_hist.lat.values]
    lons = [float(v) for v in prism_hist.lon.values]
    x_range, y_range = _heatmap_ranges(np.asarray(lons), np.asarray(lats))
    sat_uri = bm.satellite_basemap_uri(x_range[0], y_range[0], x_range[1], y_range[1])

    cache: dict = {
        "lats": lats,
        "lons": lons,
        "x_range": x_range,
        "y_range": y_range,
        "sat_uri": sat_uri,
        "vars": {},
    }

    for var in ("T", "P"):
        prism_hist_m = _monthly_prism(var, prism_hist[var])
        prism_fut_m = _monthly_prism(var, prism_fut[var])
        cmip6_hist_v = _monthly_cmip6(var, cmip6_hist[var])
        cmip6_fut_v = _monthly_cmip6(var, cmip6_fut[var])

        cell_specs = _pick_representative_cells(
            var, prism_hist, prism_hist_m, prism_fut_m, cmip6_hist_v, cmip6_fut_v,
        )
        cells = []
        for key, label, li, lj in cell_specs:
            fit = _fit_cell(
                var, li, lj, prism_hist_m, prism_fut_m,
                cmip6_hist_v, cmip6_fut_v, prism_hist,
            )
            fit["key"] = key
            fit["label"] = label
            fit["var"] = var
            cells.append(fit)

        z_map = np.asarray(
            prism_hist_m.sel(time=slice(*C.HIST)).mean("time").values, dtype=float,
        )
        cache["vars"][var] = {
            "cells": cells,
            "z_map": z_map.tolist(),
            "zmin": float(np.nanmin(z_map)),
            "zmax": float(np.nanmax(z_map)),
            "unit": "degC" if var == "T" else "mm/month",
            "diag_label": "degC" if var == "T" else "log1p units",
            "ylabel": _ylabel(var),
            "colorscale": viz.CMAP[var],
        }

    return cache


def print_regression_summary(var: str, cell: dict) -> None:
    print(f"\n--- SDSM {var} — {cell['label']} ---")
    print(f"  Cell lat/lon: {cell['cell_lat']}, {cell['cell_lon']}")
    print(f"  Historical samples: {cell['n_hist']}")
    coef_names = COEF_NAMES_P if var == "P" else COEF_NAMES
    for name, val in zip(coef_names, cell["params"]):
        print(f"  {name}: {val}")
    if var == "P":
        print("  (log1p regression on monthly totals, mm)")
    print(f"  R²: {cell['rsquared']}, residual SD: {cell['resid_sd']}")
    print(f"  Hist mean PRISM: {cell['hist_mean_y']}, fut mean projection: {cell['fut_mean_y_hat']}")


# ---------------------------------------------------------------------------
# Figure + interactivity
# ---------------------------------------------------------------------------


def _cell_frame(cell: dict) -> dict:
    y_obs = np.asarray(cell["y_hist"], float)
    y_hat = np.asarray(cell["y_hat_hist"], float)
    lo, hi = _obs_fit_range(y_obs, y_hat)
    theo, sorted_r, y_qq = _qq_data(np.asarray(cell["residuals"], float))
    return {
        "marker_x": [cell["cell_lon"]],
        "marker_y": [cell["cell_lat"]],
        "obs_x": cell["y_hat_hist"],
        "obs_y": cell["y_hist"],
        "line_x": [lo, hi],
        "line_y": [lo, hi],
        "r2_text": f"R² = {cell['rsquared']:.3f}",
        "r2_x": lo + 0.05 * (hi - lo),
        "r2_y": hi - 0.05 * (hi - lo),
        "hist_t": cell["hist_times"],
        "y_hist": cell["y_hist"],
        "y_hat_hist": cell["y_hat_hist"],
        "fut_t": cell["fut_times"],
        "y_hat_fut": cell["y_hat_fut"],
        "y_fut": cell["y_fut"],
        "qq_x": [round(v, 4) for v in theo],
        "qq_y": [round(v, 4) for v in sorted_r],
        "qq_line_x": [round(v, 4) for v in theo[[0, -1]]],
        "qq_line_y": [round(v, 4) for v in y_qq],
        "resid_x": cell["y_hat_hist"],
        "resid_y": cell["residuals"],
        "annotation": _annotation_text(cell, var=cell.get("var", "T")),
    }


def _annotation_text(cell: dict, var: str = "T") -> str:
    coef_names = COEF_NAMES_P if var == "P" else COEF_NAMES
    lines = [
        f"<b>{cell['label']}</b>",
        f"lat {cell['cell_lat']:.4f}, lon {cell['cell_lon']:.4f}",
        f"R² = {cell['rsquared']:.3f}, residual SD = {cell['resid_sd']:.3f}",
    ]
    if var == "P":
        lines.insert(1, "log1p regression on monthly totals (mm)")
    for name, val in zip(coef_names, cell["params"]):
        short = name.split("(")[0].strip()
        lines.append(f"{short} = {val:.4f}")
    return "<br>".join(lines)


def sdsm_regression_figure(var: str, cache: dict, cell_idx: int = 0) -> go.Figure:
    vcache = cache["vars"][var]
    cell = vcache["cells"][cell_idx]
    fr = _cell_frame(cell)

    var_title = "temperature" if var == "T" else "precipitation"
    map_title = (
        f"PRISM grid and selected cell (1991-2000 mean monthly {var_title})"
    )
    fig_title = (
        f"SDSM linear regression: PRISM {var_title} from CMIP6"
        if var == "T"
        else "SDSM log1p regression: PRISM monthly precipitation from CMIP6"
    )

    fig = make_subplots(
        rows=3,
        cols=2,
        subplot_titles=(
            map_title,
            None,
            "Observed vs fitted (historical)",
            "Monthly time series",
            "Residual QQ-plot (Gaussian noise check)",
            "Residuals vs fitted values",
        ),
        specs=[
            [{"colspan": 2}, None],
            [{"type": "xy"}, {"type": "xy"}],
            [{"type": "xy"}, {"type": "xy"}],
        ],
        row_heights=[0.38, 0.31, 0.31],
        vertical_spacing=0.10,
        horizontal_spacing=0.10,
    )

    trace_idx: dict[str, int] = {}
    idx = 0

    if cache.get("sat_uri"):
        fig.add_layout_image(
            dict(
                source=cache["sat_uri"],
                xref="x",
                yref="y",
                x=cache["x_range"][0],
                y=cache["y_range"][0],
                sizex=cache["x_range"][1] - cache["x_range"][0],
                sizey=cache["y_range"][1] - cache["y_range"][0],
                xanchor="left",
                yanchor="bottom",
                sizing="stretch",
                layer="below",
                opacity=1.0,
            ),
            row=1,
            col=1,
        )

    heat_opacity = bm.heatmap_opacity() if cache.get("sat_uri") else 1.0
    fig.add_trace(
        go.Heatmap(
            x=cache["lons"],
            y=cache["lats"],
            z=vcache["z_map"],
            zmin=vcache["zmin"],
            zmax=vcache["zmax"],
            colorscale=vcache["colorscale"],
            opacity=heat_opacity,
            showscale=True,
            colorbar=dict(
                title=vcache["ylabel"],
                len=0.35,
                y=0.82,
                yanchor="middle",
                thickness=12,
                x=1.02,
            ),
            hovertemplate="lon %{x:.3f}<br>lat %{y:.3f}<br>%{z:.2f}<extra></extra>",
            name="Mean monthly field",
        ),
        row=1,
        col=1,
    )
    idx += 1

    for ct in _county_boundary_traces():
        fig.add_trace(ct, row=1, col=1)
        idx += 1

    fig.add_trace(
        go.Scatter(
            x=fr["marker_x"],
            y=fr["marker_y"],
            mode="markers",
            marker=dict(size=14, color=COLOR_MARKER, symbol="x", line=dict(width=2, color="white")),
            name="Selected cell",
        ),
        row=1,
        col=1,
    )
    trace_idx["marker"] = idx
    idx += 1

    fig.add_trace(
        go.Scatter(
            x=fr["obs_x"],
            y=fr["obs_y"],
            mode="markers",
            marker=dict(size=7, color=COLOR_PRISM, opacity=0.75),
            name="Historical months",
        ),
        row=2,
        col=1,
    )
    trace_idx["obs"] = idx
    idx += 1

    fig.add_trace(
        go.Scatter(
            x=fr["line_x"],
            y=fr["line_y"],
            mode="lines",
            line=dict(color="#374151", dash="dash", width=1.5),
            name="1:1 line",
        ),
        row=2,
        col=1,
    )
    trace_idx["line11"] = idx
    idx += 1

    fig.add_trace(
        go.Scatter(
            x=fr["hist_t"],
            y=fr["y_hist"],
            mode="lines+markers",
            line=dict(color=COLOR_PRISM, width=1.5),
            marker=dict(size=4),
            name="PRISM observed (hist)",
        ),
        row=2,
        col=2,
    )
    trace_idx["ts_obs"] = idx
    idx += 1

    fig.add_trace(
        go.Scatter(
            x=fr["hist_t"],
            y=fr["y_hat_hist"],
            mode="lines",
            line=dict(color=COLOR_CMIP6, width=1.5, dash="dot"),
            name="SDSM fitted (hist)",
        ),
        row=2,
        col=2,
    )
    trace_idx["ts_fit"] = idx
    idx += 1

    fig.add_trace(
        go.Scatter(
            x=fr["fut_t"],
            y=fr["y_hat_fut"],
            mode="lines+markers",
            line=dict(color=COLOR_PROJ, width=1.5),
            marker=dict(size=4),
            name="SDSM projection (fut)",
        ),
        row=2,
        col=2,
    )
    trace_idx["ts_proj"] = idx
    idx += 1

    fig.add_trace(
        go.Scatter(
            x=fr["fut_t"],
            y=fr["y_fut"],
            mode="lines+markers",
            line=dict(color=COLOR_PRISM, width=1.5, dash="dash"),
            marker=dict(size=4, symbol="diamond-open"),
            name="PRISM evaluation (fut)",
        ),
        row=2,
        col=2,
    )
    trace_idx["ts_eval"] = idx
    idx += 1

    fig.add_trace(
        go.Scatter(
            x=fr["qq_x"],
            y=fr["qq_y"],
            mode="markers",
            marker=dict(size=6, color=COLOR_CMIP6, opacity=0.8),
            name="Residual quantiles",
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    trace_idx["qq"] = idx
    idx += 1

    fig.add_trace(
        go.Scatter(
            x=fr["qq_line_x"],
            y=fr["qq_line_y"],
            mode="lines",
            line=dict(color="#374151", dash="dash", width=1.5),
            showlegend=False,
        ),
        row=3,
        col=1,
    )
    trace_idx["qq_line"] = idx
    idx += 1

    fig.add_trace(
        go.Scatter(
            x=fr["resid_x"],
            y=fr["resid_y"],
            mode="markers",
            marker=dict(size=7, color=COLOR_PROJ, opacity=0.75),
            showlegend=False,
        ),
        row=3,
        col=2,
    )
    trace_idx["resid"] = idx
    idx += 1

    fig.add_trace(
        go.Scatter(
            x=[fr["r2_x"]],
            y=[fr["r2_y"]],
            mode="text",
            text=[fr["r2_text"]],
            textfont=dict(size=11, color="#111827"),
            showlegend=False,
            hoverinfo="skip",
        ),
        row=2,
        col=1,
    )
    trace_idx["r2"] = idx
    idx += 1

    diag_label = vcache["diag_label"]
    fig.update_annotations(font_size=11)
    fig.add_annotation(_stats_annotation(fr["annotation"]))
    stats_ann_index = len(fig.layout.annotations) - 1
    n_traces = idx + 1

    dropdown_buttons = [
        dict(
            label=c["label"],
            method="update",
            args=_update_args(c, trace_idx, n_traces, stats_ann_index),
        )
        for c in vcache["cells"]
    ]

    fig.update_layout(
        autosize=True,
        width=920,
        height=1080,
        margin=dict(t=80, b=56, l=52, r=88),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, font=dict(size=10)),
        title=dict(
            text=fig_title,
            x=0.5,
            xanchor="center",
            font=dict(size=14),
        ),
        updatemenus=[
            dict(
                type="dropdown",
                direction="down",
                active=cell_idx,
                x=0.0,
                xanchor="left",
                y=1.14,
                yanchor="top",
                buttons=dropdown_buttons,
            )
        ],
    )

    fig.update_xaxes(range=cache["x_range"], showgrid=False, zeroline=False, row=1, col=1)
    fig.update_yaxes(range=cache["y_range"], showgrid=False, zeroline=False, row=1, col=1)

    x_fit_label = (
        "Fitted PRISM temperature (degC)"
        if var == "T"
        else "Fitted PRISM monthly precipitation (mm)"
    )
    y_obs_label = (
        "Observed PRISM temperature (degC)"
        if var == "T"
        else "Observed PRISM monthly precipitation (mm)"
    )
    fig.update_xaxes(title_text=x_fit_label, row=2, col=1)
    fig.update_yaxes(title_text=y_obs_label, row=2, col=1)
    fig.update_yaxes(title_text=vcache["ylabel"], row=2, col=2)
    fig.update_xaxes(title_text="Theoretical normal quantile", row=3, col=1)
    fig.update_yaxes(title_text=f"Residual quantile ({diag_label})", row=3, col=1)
    fig.update_xaxes(title_text="Fitted values (display space)", row=3, col=2)
    fig.update_yaxes(title_text=f"Residuals ({diag_label})", row=3, col=2)

    for row, col in [(2, 1), (2, 2), (3, 1), (3, 2)]:
        fig.update_xaxes(showgrid=True, zeroline=False, row=row, col=col)
        fig.update_yaxes(showgrid=True, zeroline=False, row=row, col=col)

    fig.add_hline(y=0, line_width=1, line_dash="dash", line_color="#374151", row=3, col=2)

    fig._sdsm_trace_idx = trace_idx  # type: ignore[attr-defined]
    fig._sdsm_stats_ann_index = stats_ann_index  # type: ignore[attr-defined]
    return fig


def _stats_annotation(text: str) -> dict:
    return dict(
        text=text,
        xref="paper",
        yref="paper",
        x=0.99,
        y=0.99,
        xanchor="right",
        yanchor="top",
        align="left",
        showarrow=False,
        visible=True,
        font=dict(size=10),
        bordercolor="#9ca3af",
        borderwidth=1,
        borderpad=6,
        bgcolor="rgba(255,255,255,0.92)",
    )


def _update_args(
    cell: dict,
    trace_idx: dict[str, int],
    n_traces: int,
    stats_ann_index: int,
) -> list:
    fr = _cell_frame(cell)
    x_up: list = [None] * n_traces
    y_up: list = [None] * n_traces
    x_up[trace_idx["marker"]] = fr["marker_x"]
    y_up[trace_idx["marker"]] = fr["marker_y"]
    x_up[trace_idx["obs"]] = fr["obs_x"]
    y_up[trace_idx["obs"]] = fr["obs_y"]
    x_up[trace_idx["line11"]] = fr["line_x"]
    y_up[trace_idx["line11"]] = fr["line_y"]
    x_up[trace_idx["ts_obs"]] = fr["hist_t"]
    y_up[trace_idx["ts_obs"]] = fr["y_hist"]
    x_up[trace_idx["ts_fit"]] = fr["hist_t"]
    y_up[trace_idx["ts_fit"]] = fr["y_hat_hist"]
    x_up[trace_idx["ts_proj"]] = fr["fut_t"]
    y_up[trace_idx["ts_proj"]] = fr["y_hat_fut"]
    x_up[trace_idx["ts_eval"]] = fr["fut_t"]
    y_up[trace_idx["ts_eval"]] = fr["y_fut"]
    x_up[trace_idx["qq"]] = fr["qq_x"]
    y_up[trace_idx["qq"]] = fr["qq_y"]
    x_up[trace_idx["qq_line"]] = fr["qq_line_x"]
    y_up[trace_idx["qq_line"]] = fr["qq_line_y"]
    x_up[trace_idx["resid"]] = fr["resid_x"]
    y_up[trace_idx["resid"]] = fr["resid_y"]
    x_up[trace_idx["r2"]] = [fr["r2_x"]]
    y_up[trace_idx["r2"]] = [fr["r2_y"]]
    text_up: list = [None] * n_traces
    text_up[trace_idx["r2"]] = [fr["r2_text"]]
    ann_text = fr["annotation"]
    layout = {
        f"annotations[{stats_ann_index}].text": ann_text,
        f"annotations[{stats_ann_index}].visible": True,
    }
    return [{"x": x_up, "y": y_up, "text": text_up}, layout]


_STATS_SYNC_JS = """
(function() {
  function hook(gd) {
    if (!gd || !window.sdsmStatsSync) return;
    const cfg = window.sdsmStatsSync;
    let syncing = false;
    function sync() {
      if (syncing) return;
      const menu = gd.layout.updatemenus && gd.layout.updatemenus[0];
      const active = menu ? menu.active : 0;
      const text = cfg.texts[active];
      if (!text) return;
      const cur = gd.layout.annotations && gd.layout.annotations[cfg.index];
      if (cur && cur.text === text && cur.visible) return;
      syncing = true;
      const patch = {};
      patch["annotations[" + cfg.index + "].text"] = text;
      patch["annotations[" + cfg.index + "].visible"] = true;
      Plotly.relayout(gd, patch).finally(function() { syncing = false; });
    }
    gd.on("plotly_update", function() { setTimeout(sync, 50); });
    gd.on("plotly_restyle", function() { setTimeout(sync, 50); });
  }
  const gd = document.querySelector(".plotly-graph-div");
  if (gd && gd.data) hook(gd);
  else {
    const obs = new MutationObserver(function() {
      const el = document.querySelector(".plotly-graph-div");
      if (el && el.data) { hook(el); obs.disconnect(); }
    });
    obs.observe(document.body, { childList: true, subtree: true });
  }
})();
"""


def export_sdsm_regression_html(
    var: str,
    data: dict[str, xr.Dataset] | None = None,
    cache: dict | None = None,
) -> Path:
    out = OUT_PATHS[var]
    if cache is None:
        cache = build_sdsm_cache(data=data)
    for cell in cache["vars"][var]["cells"]:
        print_regression_summary(var, cell)
    fig = sdsm_regression_figure(var, cache, cell_idx=0)
    stats_idx = fig._sdsm_stats_ann_index  # type: ignore[attr-defined]
    stats_texts = [
        _annotation_text(c, var=var) for c in cache["vars"][var]["cells"]
    ]
    sync_json = json.dumps({"index": stats_idx, "texts": stats_texts})
    out.parent.mkdir(parents=True, exist_ok=True)
    cfg = {"scrollZoom": True, "displayModeBar": True, "responsive": True}
    html = fig.to_html(
        include_plotlyjs="cdn",
        full_html=True,
        config=cfg,
        post_script=f"window.sdsmStatsSync = {sync_json};\n{_STATS_SYNC_JS}",
    )
    out.write_text(html, encoding="utf-8")
    return out


def export_all(data: dict[str, xr.Dataset] | None = None) -> list[Path]:
    if data is None:
        data = io.load_delta_change_data(progress=True)
    cache = build_sdsm_cache(data=data)
    return [export_sdsm_regression_html(var, data=data, cache=cache) for var in ("T", "P")]


def main() -> None:
    paths = export_all()
    for p in paths:
        print(f"Wrote {p} ({p.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
