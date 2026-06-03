"""Cached satellite basemap image for Plotly map overlays (no API key)."""
from __future__ import annotations

import base64
import math
from io import BytesIO
from pathlib import Path

from PIL import Image
import requests

from . import config as C

_ESRI_TILE = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
_CACHE = C.FIGURES / "travis_satellite_basemap.jpg"
_HEATMAP_OPACITY = 0.72


def heatmap_opacity() -> float:
    return _HEATMAP_OPACITY


def _lon_lat_to_tile(lon: float, lat: float, zoom: int) -> tuple[int, int]:
    n = 2**zoom
    x = int((lon + 180.0) / 360.0 * n)
    lat_rad = math.radians(lat)
    y = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    return x, y


def _fetch_tile(z: int, x: int, y: int, session: requests.Session) -> Image.Image:
    url = _ESRI_TILE.format(z=z, y=y, x=x)
    r = session.get(url, timeout=15)
    r.raise_for_status()
    return Image.open(BytesIO(r.content)).convert("RGB")


def _stitch_tiles(
    west: float, south: float, east: float, north: float, zoom: int = 11
) -> Image.Image:
    x0, y1 = _lon_lat_to_tile(west, north, zoom)
    x1, y0 = _lon_lat_to_tile(east, south, zoom)
    session = requests.Session()
    session.headers["User-Agent"] = "DS_Chapter-downscaling/1.0"
    rows: list[list[Image.Image | None]] = []
    for y in range(y0, y1 + 1):
        row: list[Image.Image | None] = []
        for x in range(x0, x1 + 1):
            try:
                row.append(_fetch_tile(zoom, x, y, session))
            except requests.RequestException:
                row.append(None)
        rows.append(row)
    tile_w, tile_h = 256, 256
    canvas = Image.new("RGB", ((x1 - x0 + 1) * tile_w, (y1 - y0 + 1) * tile_h), (40, 40, 40))
    for j, row in enumerate(rows):
        for i, tile in enumerate(row):
            if tile is not None:
                canvas.paste(tile, (i * tile_w, j * tile_h))
    return canvas


def _to_data_uri(path: Path) -> str:
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def satellite_basemap_uri(
    west: float,
    south: float,
    east: float,
    north: float,
    *,
    zoom: int = 11,
    refresh: bool = False,
) -> str | None:
    """Return a JPEG data URI for the domain, fetching tiles once if needed."""
    C.FIGURES.mkdir(parents=True, exist_ok=True)
    if _CACHE.exists() and not refresh:
        return _to_data_uri(_CACHE)

    pad = 0.02
    try:
        img = _stitch_tiles(west - pad, south - pad, east + pad, north + pad, zoom=zoom)
        img.save(_CACHE, format="JPEG", quality=85)
        return _to_data_uri(_CACHE)
    except (requests.RequestException, OSError, ValueError):
        return None
