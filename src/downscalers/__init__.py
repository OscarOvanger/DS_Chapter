"""Downscaling methods covered in the chapter.

Each module implements one method with small, documented functions so the notebook
can call a single high-level entry point and focus on the *outputs*:

* ``delta_change`` -- DC: additive (T) / multiplicative (P) delta change.
* ``bcsd``         -- BCSD: QDM bias correction + spatial disaggregation.
"""
from .bcsd import bcsd
from .delta_change import (
    apply_monthly_delta,
    delta_change,
    monthly_climatology,
    monthly_delta,
    monthly_deltas,
    precompute_monthly_cache,
    project_month_from_cache,
    project_monthly_maps,
)

__all__ = [
    "apply_monthly_delta",
    "delta_change",
    "monthly_climatology",
    "monthly_delta",
    "monthly_deltas",
    "precompute_monthly_cache",
    "project_month_from_cache",
    "project_monthly_maps",
    "bcsd",
]
