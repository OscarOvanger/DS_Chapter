"""Standalone Plotly HTML figures for the delta-change chapter section."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import xarray as xr

from . import basemap as bm
from . import config as C
from . import data_io as io
from .downscalers.delta_change import monthly_climatology, project_month_from_cache
from . import plotting as viz

KIND = {"T": "add", "P": "mult"}
CMAP = {"T": "RdYlBu_r", "P": "YlGnBu"}
PANEL_TITLES = [
    "1. PRISM 1991–2000",
    "2. Delta-change projection",
    "3. PRISM truth 2016–2025",
]
OUT_PATHS = {"T": C.DC_TEMPERATURE_HTML, "P": C.DC_PRECIPITATION_HTML}


def _paths(var: str) -> tuple[Path, Path]:
    stem = "dc_temperature" if var == "T" else "dc_precipitation"
    return C.FIGURES / f"{stem}.html", C.FIGURES / f"{stem}_fragment.html"


def build_monthly_cache(data: dict[str, xr.Dataset]) -> dict:
    """Monthly climatologies + coarse deltas (shared by export and notebooks)."""
    cache = {
        "prism_hist_m": monthly_climatology(data["prism_hist"]),
        "prism_fut_m": monthly_climatology(data["prism_fut"]),
        "_coarse_hist_m": monthly_climatology(data["cmip6_hist"]),
        "_coarse_fut_m": monthly_climatology(data["cmip6_fut"]),
    }
    ch, cf = cache["_coarse_hist_m"], cache["_coarse_fut_m"]
    cache["coarse_delta_T"] = cf["T"] - ch["T"]
    cache["coarse_delta_P"] = cf["P"] / ch["P"].clip(min=1e-6)
    del cache["_coarse_hist_m"], cache["_coarse_fut_m"]
    return cache


def _mesh(da) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray(da.values, dtype=float)
    lon = np.asarray(da.lon.values, dtype=float)
    lat = np.asarray(da.lat.values, dtype=float)
    return z, lon, lat


def _heatmap_ranges(x: np.ndarray, y: np.ndarray) -> tuple[list[float], list[float]]:
    """Axis limits that match heatmap cells (no extra grey margin)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = (x[-1] - x[0]) / (len(x) - 1) if len(x) > 1 else 0.02
    dy = (y[-1] - y[0]) / (len(y) - 1) if len(y) > 1 else 0.02
    return [float(x[0] - dx / 2), float(x[-1] + dx / 2)], [float(y[0] - dy / 2), float(y[-1] + dy / 2)]


def _add_satellite_underlay(
    fig: go.Figure,
    uri: str,
    x_range: list[float],
    y_range: list[float],
    row: int,
) -> None:
    fig.add_layout_image(
        dict(
            source=uri,
            xref="x",
            yref="y",
            x=x_range[0],
            y=y_range[0],
            sizex=x_range[1] - x_range[0],
            sizey=y_range[1] - y_range[0],
            xanchor="left",
            yanchor="bottom",
            sizing="stretch",
            layer="below",
            opacity=1.0,
        ),
        row=row,
        col=1,
    )


def delta_monthly_figure(var: str, cache: dict, month0: int = 1) -> go.Figure:
    """Plotly figure: three stacked maps, optional satellite underlay, month slider."""
    months = list(range(1, 13))
    fields = [project_month_from_cache(cache, var, m) for m in months]
    vmin = float(min(np.nanmin(_mesh(da)[0]) for triple in fields for da in triple))
    vmax = float(max(np.nanmax(_mesh(da)[0]) for triple in fields for da in triple))

    b0, p0, t0 = fields[month0 - 1]
    fig = make_subplots(
        rows=3,
        cols=1,
        subplot_titles=PANEL_TITLES,
        vertical_spacing=0.06,
    )

    _, x0, y0 = _mesh(b0)
    x_range, y_range = _heatmap_ranges(x0, y0)
    sat_uri = bm.satellite_basemap_uri(x_range[0], y_range[0], x_range[1], y_range[1])
    heat_opacity = bm.heatmap_opacity() if sat_uri else 1.0

    for row, da in enumerate((b0, p0, t0), start=1):
        if sat_uri:
            _add_satellite_underlay(fig, sat_uri, x_range, y_range, row)
        z, x, y = _mesh(da)
        fig.add_trace(
            go.Heatmap(
                x=x,
                y=y,
                z=z,
                zmin=vmin,
                zmax=vmax,
                colorscale=CMAP[var],
                opacity=heat_opacity,
                colorbar=dict(
                    title=viz.LABEL[var],
                    len=0.58,
                    y=0.50,
                    yanchor="middle",
                    yref="paper",
                    x=1.01,
                    xanchor="left",
                    thickness=14,
                )
                if row == 3
                else None,
                showscale=row == 3,
                hovertemplate="lon %{x:.3f}<br>lat %{y:.3f}<br>%{z:.2f}<extra></extra>",
            ),
            row=row,
            col=1,
        )

    steps = []
    for m, (b, p, t) in zip(months, fields, strict=True):
        name = pd.Timestamp(2000, m, 1).strftime("%B")
        z_b, _, _ = _mesh(b)
        z_p, _, _ = _mesh(p)
        z_t, _, _ = _mesh(t)
        steps.append(
            dict(
                label=name,
                method="update",
                args=[
                    {"z": [z_b, z_p, z_t], "zmin": [vmin] * 3, "zmax": [vmax] * 3},
                    {},
                ],
            )
        )

    layout: dict = {
        "autosize": True,
        "width": 680,
        "height": 980,
        "margin": dict(t=44, b=56, l=40, r=72),
        "sliders": [
            dict(
                active=month0 - 1,
                currentvalue={"prefix": "Month: "},
                pad=dict(t=40),
                steps=steps,
            )
        ],
    }
    fig.update_layout(**layout)
    fig.update_annotations(font_size=11)
    for row in range(1, 4):
        fig.update_xaxes(
            range=x_range,
            showgrid=False,
            zeroline=False,
            constrain="domain",
            row=row,
            col=1,
        )
        fig.update_yaxes(
            range=y_range,
            showgrid=False,
            zeroline=False,
            row=row,
            col=1,
        )
    return fig


def export_delta_monthly_html(
    var: str,
    out_path: Path | None = None,
    data: dict[str, xr.Dataset] | None = None,
    cache: dict | None = None,
) -> tuple[Path, Path]:
    """Write standalone HTML + an embed fragment for Quarto (``{{< include >}}``)."""
    standalone, fragment = _paths(var)
    if out_path is not None:
        standalone = Path(out_path)
    if data is None:
        data = io.load_delta_change_data(progress=True)
    if cache is None:
        cache = build_monthly_cache(data)
    fig = delta_monthly_figure(var, cache)
    standalone.parent.mkdir(parents=True, exist_ok=True)
    cfg = {"scrollZoom": True, "displayModeBar": True, "responsive": True}
    fig.write_html(standalone, include_plotlyjs="cdn", full_html=True, config=cfg)
    fig.write_html(fragment, include_plotlyjs="cdn", full_html=False, config=cfg)
    return standalone, fragment


def export_all(out_dir: Path | None = None) -> list[Path]:
    """Export temperature and precipitation HTML + Quarto embed fragments."""
    if out_dir is not None:
        C.FIGURES.mkdir(parents=True, exist_ok=True)
    data = io.load_delta_change_data(progress=True)
    cache = build_monthly_cache(data)
    paths: list[Path] = []
    for var in ("T", "P"):
        s, f = export_delta_monthly_html(var, data=data, cache=cache)
        paths.extend([s, f])
    return paths
