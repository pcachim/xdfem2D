"""
Save xdfem2D results to JSON or Excel (.xlsx).

Usage
-----
from xdfem2d.results_io import save_json, save_excel

results = struc.calculate()
save_json(results,  'output.json')
save_excel(results, 'output.xlsx')
"""
from __future__ import annotations
import json
from pathlib import Path

from xdfem2d.file_io import NpEncoder


# ---------------------------------------------------------------------------
# JSON
# ---------------------------------------------------------------------------

def save_json(results: dict, path: str | Path):
    """Write the full results dict to a JSON file.

    The results contain NumPy arrays (force/moment diagrams) and NumPy scalars,
    so the shared :class:`xdfem2d.file_io.NpEncoder` converts them to native
    Python types.
    """
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, cls=NpEncoder)


# ---------------------------------------------------------------------------
# Excel
# ---------------------------------------------------------------------------

def save_excel(results: dict, path: str | Path, domain: str = "plane",
               struc=None):
    """Write results to an .xlsx workbook, one sheet per result table.

    Delegates to the shared report builders in :mod:`xdfem2d.report_io` (the same
    code the GUI uses), so there is a single Excel implementation. Tables report
    analysis cases and combinations; raw load cases are load definitions only and
    are not tabulated. ``domain`` ('plane' | 'plate') sets the result vocabulary
    (membrane stresses vs slab forces, ux/uy/θz vs w/θx/θy).

    The results dict carries no domain of its own, so a plate model saved with
    the default would come out with plane headers. Pass the model as *struc* and
    the domain is read from it — ``save_excel(results, path, struc=model)`` — so
    a script never has to spell it out. An explicit *struc* takes precedence over
    *domain*.
    """
    if struc is not None:
        domain = getattr(struc, "domain", "plane")
    from xdfem2d.report_io import _results_tables, _xlsx_from_tables
    _xlsx_from_tables(str(path), _results_tables(results, domain))


def save_structure_excel(struc, path: str | Path):
    """Write the model (input) data to an .xlsx, one table per sheet."""
    from xdfem2d.report_io import _model_tables, _xlsx_from_tables
    _xlsx_from_tables(str(path), _model_tables(struc))

