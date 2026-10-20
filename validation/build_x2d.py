"""Generate the validation .x2d models into ``validation/models/``.

Run:  python validation/build_x2d.py
Each model is written structure-only (no results); the test and the report
runner re-solve it, so the .x2d always exercises the current engine.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xdfem2d import save_x2d                       # noqa: E402
import validation_cases as vc                      # noqa: E402
import modal_cases as mdc                          # noqa: E402

MODELS = ROOT / "validation" / "models"


def main():
    MODELS.mkdir(parents=True, exist_ok=True)
    n = 0
    for model_id, build in vc.ALL_MODELS + mdc.MODAL_MODELS:
        struc = build()
        path = MODELS / f"{model_id}.x2d"
        save_x2d(struc, None, path)
        print(f"  wrote {path.relative_to(ROOT)}")
        n += 1
    print(f"{n} models written to {MODELS.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
