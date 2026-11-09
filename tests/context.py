"""Shared test helpers for the xdfem2D test suite.

Makes ``src/`` importable without installing the package and provides small
utilities (relative-tolerance assertions and beam-property helpers) used across
the analytical-validation tests.
"""
from __future__ import annotations
import os
import sys

# Make the package importable from a source checkout (src/ layout).
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC = os.path.join(_ROOT, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)

from xdfem2d import Structure2D  # noqa: E402  (import after sys.path tweak)


def rect_inertia(b: float, h: float) -> float:
    """Second moment of area of a rectangular section about its strong axis."""
    return b * h ** 3 / 12.0


def rect_area(b: float, h: float) -> float:
    return b * h


def assert_close(test_case, actual, expected, rel=1e-6, abs_tol=1e-9, msg=""):
    """Assert ``actual`` is close to ``expected`` (relative + absolute tol).

    The FEM results for these textbook cases are essentially exact, so the
    default tolerances are tight; loosen ``rel`` per-call when comparing against
    series/approximate analytical formulas.
    """
    tol = max(abs_tol, rel * abs(expected))
    diff = abs(actual - expected)
    label = f"{msg}: " if msg else ""
    test_case.assertLessEqual(
        diff, tol,
        f"{label}expected {expected!r}, got {actual!r} (|diff|={diff:.3e} > tol={tol:.3e})",
    )


def midspan_moment(results, case_id: str, elem_id: str) -> float:
    """Peak |M| along an element's moment diagram for a given load case."""
    import numpy as np
    M = results["element_distribution"][case_id][elem_id]["M"]
    return float(np.max(np.abs(M)))
