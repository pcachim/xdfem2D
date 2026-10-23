"""Validation of the constraint engine against closed-form benchmarks.

Kept apart from ``test_validation_x2d.py`` because constraints are imposed by
penalty: the answers match the closed form to ~1e-6 relative (not to machine
precision), and the penalty leaves a tiny spurious reaction, so the strict
``places=6`` global-equilibrium check used for the other cases does not apply.
Here equilibrium is checked with a relative tolerance instead.

Cases (validation/validation_cases.py, list CONSTRAINT_CASES):
  * cn-1 — rigid offset: an eccentric load carried through a rigid link becomes
    an axial force plus a moment at a column top; the slave node follows the
    top's rigid-body motion.
  * cn-2 — equal-DOF: two cantilever tips tied to the same vertical
    displacement share the total load equally.
"""
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import context  # noqa: F401  (puts src/ on the path)

ROOT = Path(__file__).resolve().parent.parent.parent
_VAL = ROOT / "validation"
if str(_VAL) not in sys.path:
    sys.path.insert(0, str(_VAL))

import validation_cases as vc          # noqa: E402
from xdfem2d import load_x2d, save_x2d  # noqa: E402

MODELS = _VAL / "models"


def _model_path(model_id: str) -> Path:
    p = MODELS / f"{model_id}.x2d"
    if not p.exists():
        MODELS.mkdir(parents=True, exist_ok=True)
        save_x2d(dict(vc.ALL_MODELS)[model_id](), None, p)
    return p


def _solve(model_id: str):
    struc, _, _ = load_x2d(_model_path(model_id))
    return struc, struc.calculate()


class TestConstraintClosedForm(unittest.TestCase):
    def test_quantities(self):
        for c in vc.CONSTRAINT_CASES:
            struc, res = _solve(c.id)
            for q in c.quantities:
                with self.subTest(case=c.id, quantity=q.label):
                    a, m = q.analytical, q.measured(res, struc)
                    if not q.signed:
                        a, m = abs(a), abs(m)
                    tol = max(q.abs_tol, q.rel_tol * abs(a))
                    self.assertLessEqual(
                        abs(m - a), tol,
                        f"{c.id} — {q.label}: analytical {a:.6g} {q.unit}, "
                        f"FEM {m:.6g} {q.unit} (|Δ|={abs(m - a):.3e} > {tol:.3e})")


class TestConstraintEquilibrium(unittest.TestCase):
    """ΣReactions + ΣAppliedLoads ≈ 0, to a relative tolerance (constraints add
    no external force, but the penalty leaves a small residual)."""

    REL = 1.0e-4

    def test_equilibrium(self):
        for c in vc.CONSTRAINT_CASES:
            struc, res = _solve(c.id)
            with self.subTest(case=c.id):
                fx, fy = vc.applied_resultant(struc, vc.CASE)
                rx, ry = vc.reaction_resultant(res, vc.CASE)
                scale = max(1.0, abs(fx), abs(fy))
                self.assertLessEqual(abs(rx + fx), self.REL * scale, f"{c.id} ΣFx")
                self.assertLessEqual(abs(ry + fy), self.REL * scale, f"{c.id} ΣFy")


class TestRigidLinkKinematics(unittest.TestCase):
    """cn-1: the slave node is the exact rigid-body image of the master."""

    def test_slave_follows_master(self):
        struc, res = _solve("cn-1")
        d = res["displacements"][vc.CASE]
        m, s = d["N1"], d["N2"]
        dx = struc.nodes["N2"].x - struc.nodes["N1"].x   # = CN1_E
        dy = struc.nodes["N2"].y - struc.nodes["N1"].y   # = 0
        self.assertAlmostEqual(s[2], m[2], places=6)                 # tz_s = tz_m
        self.assertAlmostEqual(s[0], m[0] - dy * m[2], places=6)     # ux
        self.assertAlmostEqual(s[1], m[1] + dx * m[2], places=6)     # uy


class TestEqualDofSharing(unittest.TestCase):
    """cn-2: the tie makes the two tips equal and moves load off the heavier
    side (without the tie the tips would differ by 3x)."""

    def test_tips_equal_and_shared(self):
        struc, res = _solve("cn-2")
        d = res["displacements"][vc.CASE]
        self.assertAlmostEqual(d["B1"][1], d["B2"][1], places=6)
        # each base carries the average, not its own applied load
        R = res["reactions"][vc.CASE]
        self.assertAlmostEqual(R["A1"][1], vc.CN2_R, delta=1e-2)
        self.assertAlmostEqual(R["A2"][1], vc.CN2_R, delta=1e-2)


if __name__ == "__main__":
    unittest.main(verbosity=2)
