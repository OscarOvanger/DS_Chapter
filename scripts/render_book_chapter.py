"""Render Downscaling_BS.qmd to a self-contained interactive HTML chapter.

Requires Plotly figures first:

    .venv\\Scripts\\python.exe scripts\\export_plotly_figures.py
    .venv\\Scripts\\python.exe scripts\\render_book_chapter.py

Uses portable Quarto in ``.tools/quarto`` if ``quarto`` is not on PATH.
Output: ``Downscaling_BS.html`` — keep ``figures/dc_*.html`` alongside it so plot iframes load.
"""
from __future__ import annotations

import os
import subprocess
import sys
import webbrowser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUARTO = ROOT / ".tools" / "quarto" / "bin" / "quarto.exe"
QMD = ROOT / "Downscaling_BS.qmd"
OUT = ROOT / "Downscaling_BS.html"
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"


def main() -> None:
    if not QMD.exists():
        raise FileNotFoundError(QMD)
    sys.path.insert(0, str(ROOT))
    from src import config as C

    missing = [
        p
        for p in (
            C.DC_TEMPERATURE_HTML,
            C.DC_PRECIPITATION_HTML,
            C.FIGURES / "dc_temperature_fragment.html",
            C.FIGURES / "dc_precipitation_fragment.html",
        )
        if not p.exists()
    ]
    if missing:
        raise FileNotFoundError(
            "Missing figure files. Run: .venv\\Scripts\\python.exe scripts\\export_plotly_figures.py\n"
            + "\n".join(f"  - {p}" for p in missing)
        )
    if not QUARTO.exists():
        raise FileNotFoundError(
            f"Portable Quarto not found at {QUARTO}.\n"
            "Download Quarto for Windows and extract to .tools/quarto, or install from https://quarto.org"
        )
    if not VENV_PY.exists():
        raise FileNotFoundError(f"Virtualenv Python not found: {VENV_PY}")

    env = os.environ.copy()
    env["QUARTO_PYTHON"] = str(VENV_PY)
    # Ensure Jupyter uses the project venv when Quarto executes chunks.
    env["PATH"] = str(VENV_PY.parent) + os.pathsep + env.get("PATH", "")

    print("Rendering interactive chapter with Quarto …", flush=True)
    print(f"  Quarto: {QUARTO}", flush=True)
    print(f"  Python: {VENV_PY}", flush=True)

    subprocess.run(
        [str(QUARTO), "render", str(QMD), "--to", "html"],
        cwd=ROOT,
        env=env,
        check=True,
    )

    if not OUT.exists():
        raise RuntimeError(f"Expected output not found: {OUT}")

    html = OUT.read_text(encoding="utf-8", errors="replace")
    if html.count("<iframe") < 2 or "&lt;script&gt;window.PlotlyConfig" in html:
        raise RuntimeError(
            f"{OUT.name} is missing plot iframes (Plotly fragment was embedded as escaped text). "
            "Use the iframe blocks in Downscaling_BS.qmd, not {{< include >}} on *_fragment.html."
        )

    print(f"\nDone: {OUT}", flush=True)
    print("Open this file in a browser for the interactive book chapter.", flush=True)
    webbrowser.open(OUT.as_uri())


if __name__ == "__main__":
    main()
