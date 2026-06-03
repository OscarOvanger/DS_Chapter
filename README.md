# Statistical downscaling — Austin / Travis County

Interactive book chapter (Quarto) with delta-change maps (PRISM + CMIP6).

## Local chapter

```bash
.venv\Scripts\python.exe scripts\export_plotly_figures.py
.venv\Scripts\python.exe scripts\render_book_chapter.py
```

Open `Downscaling_BS.html` in a browser (keep `figures/` beside it).

## Publish on GitHub Pages

```bash
.venv\Scripts\python.exe scripts\prepare_github_pages.py
```

Then push `docs/` to GitHub and enable Pages — full steps in [docs/PUBLISH.md](docs/PUBLISH.md).

Live URL after publishing: **https://oscarovanger.github.io/DS_Chapter/** (if the repo is named `DS_Chapter`).
