"""Build the static site under ``docs/`` for GitHub Pages.

Run from the project root::

    .venv\\Scripts\\python.exe scripts\\export_plotly_figures.py
    .venv\\Scripts\\python.exe scripts\\prepare_github_pages.py

Publishes ``docs/index.html`` plus ``docs/figures/dc_*.html``.
Site URL (after GitHub Pages is enabled): https://oscarovanger.github.io/<repo-name>/
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
FIGURES = DOCS / "figures"
QUARTO = ROOT / ".tools" / "quarto" / "bin" / "quarto.exe"
QMD = ROOT / "Downscaling_BS.qmd"
CHAPTER_HTML = ROOT / "Downscaling_BS.html"
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
PLOT_FILES = (
    "dc_temperature.html",
    "dc_precipitation.html",
    "bcsd_distribution_temperature.html",
    "bcsd_distribution_precipitation.html",
    "sdsm_regression_temperature.html",
    "sdsm_regression_precipitation.html",
)


def _run_export() -> None:
    subprocess.run(
        [str(VENV_PY), str(ROOT / "scripts" / "export_plotly_figures.py")],
        cwd=ROOT,
        check=True,
    )


def _run_quarto() -> None:
    if not QUARTO.exists():
        raise FileNotFoundError(f"Quarto not found: {QUARTO}")
    env = os.environ.copy()
    env["QUARTO_PYTHON"] = str(VENV_PY)
    subprocess.run(
        [str(QUARTO), "render", str(QMD), "--to", "html"],
        cwd=ROOT,
        env=env,
        check=True,
    )


def _validate_chapter(html: str) -> None:
    if html.count("<iframe") < 2 or "&lt;script&gt;window.PlotlyConfig" in html:
        raise RuntimeError(
            "Chapter HTML is missing plot iframes. Re-render Downscaling_BS.qmd."
        )


def main() -> None:
    sys.path.insert(0, str(ROOT))
    from src import config as C

    missing = [C.FIGURES / name for name in PLOT_FILES if not (C.FIGURES / name).exists()]
    if missing:
        print("Exporting Plotly figures …", flush=True)
        _run_export()

    if not CHAPTER_HTML.exists():
        print("Rendering Quarto chapter …", flush=True)
        _run_quarto()

    html = CHAPTER_HTML.read_text(encoding="utf-8", errors="replace")
    _validate_chapter(html)

    publish_md = DOCS / "PUBLISH.md"
    publish_text = publish_md.read_text(encoding="utf-8") if publish_md.exists() else None
    if FIGURES.exists():
        shutil.rmtree(FIGURES)
    FIGURES.mkdir(parents=True)

    shutil.copy2(CHAPTER_HTML, DOCS / "index.html")
    for name in PLOT_FILES:
        shutil.copy2(C.FIGURES / name, FIGURES / name)
    (DOCS / ".nojekyll").touch()
    if publish_text is not None:
        publish_md.write_text(publish_text, encoding="utf-8")

    print(f"\nGitHub Pages site ready in: {DOCS}", flush=True)
    print("  docs/index.html", flush=True)
    for name in PLOT_FILES:
        print(f"  docs/figures/{name}", flush=True)
    print(
        "\nNext: push to GitHub and enable Pages (see docs/PUBLISH.md).",
        flush=True,
    )


if __name__ == "__main__":
    main()
