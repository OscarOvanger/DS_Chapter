"""Generate DS_notebook.ipynb (embed standalone Plotly HTML figures)."""
from pathlib import Path

import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s.strip("\n")))
code = lambda s: cells.append(nbf.v4.new_code_cell(s.strip("\n")))

md(r"""
# Delta change — Austin / Travis County

Companion to the *Downscaling* chapter. **PRISM** + **CMIP6** monthly delta change.

Run once before these cells:

    .venv\Scripts\python.exe scripts\prepare_data.py
    .venv\Scripts\python.exe scripts\export_plotly_figures.py
""")

code(r"""import sys
from pathlib import Path

sys.path.insert(0, str(Path.cwd()))

from IPython.display import IFrame
from src.plotly_panels import export_delta_monthly_html

path, _ = export_delta_monthly_html("T")
IFrame(src=str(path), width="100%", height=500)""")

code(r"""from IPython.display import IFrame
from src.plotly_panels import export_delta_monthly_html

path, _ = export_delta_monthly_html("P")
IFrame(src=str(path), width="100%", height=500)""")

nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {"display_name": "Python 3 (.venv)", "language": "python", "name": "python3"},
    "language_info": {"name": "python"},
}
out = Path(__file__).resolve().parent.parent / "DS_notebook.ipynb"
nbf.write(nb, out)
print("wrote", out, "with", len(cells), "cells")
