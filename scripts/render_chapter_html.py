"""Alias for ``render_book_chapter.py`` (full interactive HTML chapter)."""
from __future__ import annotations

import runpy
from pathlib import Path

if __name__ == "__main__":
    runpy.run_path(str(Path(__file__).resolve().parent / "render_book_chapter.py"), run_name="__main__")
