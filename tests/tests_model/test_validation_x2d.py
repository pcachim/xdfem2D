"""Validation suite driven by the ``validation/`` .x2d models.

Loads each committed ``.x2d``, re-solves it with the current engine, and checks
the results against the closed-form values in ``validation/validation_cases.py``
(which mirror *Validacao_Programa_MEF.docx*). Also checks global equilibrium on
every case and the mesh convergence of the CST cantilever (Case 3.2).

If the model files are missing they are generated on the fly, so the suite is
self-contained. Run:  python -m unittest tests.test_validation_x2d
"""
from __future__ import annotations

import os
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
    """Path to a model's .x2d, generating it if absent."""
    p = MODELS / f"{model_id}.x2d"
    if not p.exists():
        MODELS.mkdir(parents=True, exist_ok=True)
        save_x2d(dict(vc.ALL_MODELS)[model_id](), None, p)
    return p


def _solve(model_id: str):
    struc, _, _ = load_x2d(_model_path(model_id))
    return struc, struc.calculate()


class TestClosedFormCases(unittest.TestCase):
    """Every scalar quantity of cases 2.1–3.4 (except 3.2) vs its formula."""

    def test_quantities(self):
        for c in vc.CASES:
            struc, res = _solve(c.id)
            for q in c.quantities:
                with self.subTest(case=c.id, quantity=q.label):
                    a = q.analytical
                    m = q.measured(res, struc)
                    if not q.signed:
                        a, m = abs(a), abs(m)
                    tol = max(q.abs_tol, q.rel_tol * abs(a))
                    self.assertLessEqual(
                        abs(m - a), tol,
                        f"{c.id} — {q.label}: analytical {a:.6g} {q.unit}, "
                        f"FEM {m:.6g} {q.unit} (|Δ|={abs(m - a):.3e} > {tol:.3e})")


class TestPlateBendingCases(unittest.TestCase):
    """Plate cases P.1 / P.2 vs their analytical series.

    Kept apart from the in-plane cases: their loads and reactions are
    out-of-plane (w, Rz), so the in-plane equilibrium check does not apply. The
    thin slab (DKT) is checked against the Kirchhoff/Navier series and the thick
    slab (MITC3) against the exact Mindlin (first-order-shear) series."""

    def test_quantities(self):
        for c in vc.PLATE_CASES:
            struc, res = _solve(c.id)
            for q in c.quantities:
                with self.subTest(case=c.id, quantity=q.label):
                    a, m = abs(q.analytical), abs(q.measured(res, struc))
                    tol = max(q.abs_tol, q.rel_tol * abs(a))
                    self.assertLessEqual(
                        abs(m - a), tol,
                        f"{c.id} — {q.label}: analytical {a:.6g} {q.unit}, "
                        f"FEM {m:.6g} {q.unit} (|Δ|={abs(m - a):.3e} > {tol:.3e})")

    def test_vertical_equilibrium(self):
        """ΣRz balances the applied transverse load on each plate case."""
        for c in vc.PLATE_CASES:
            struc, res = _solve(c.id)
            with self.subTest(case=c.id):
                reac = res["reactions"][vc.CASE]
                sumRz = sum(v[0] for v in reac.values())
                applied = abs(vc.PLATE_PZ) * vc.PLATE_A ** 2   # pressure · area
                self.assertAlmostEqual(abs(sumRz), applied, places=4)

    def test_mitc3_captures_shear_the_dkt_misses(self):
        """The thick-plate Mindlin deflection is a few percent above the
        Kirchhoff (thin-theory) value — the transverse shear the MITC3 adds."""
        self.assertGreater(vc.PLATE_THICK_W, vc.PLATE_THICK_KIRCHHOFF_W)
        frac = vc.PLATE_THICK_W / vc.PLATE_THICK_KIRCHHOFF_W - 1.0
        self.assertGreater(frac, 0.02)         # a real, several-percent effect


class TestGrillageCases(unittest.TestCase):
    """Grillage (plate-domain bar) cases G.1–G.3 vs their closed forms.

    A grillage bar carries out-of-plane bending (FL³/3EI), St-Venant torsion
    (TL/GJ) and, on a crossed grid, the load splits between the beams by
    stiffness — all exact, so a tight tolerance applies."""

    def test_quantities(self):
        for c in vc.GRILLAGE_CASES:
            struc, res = _solve(c.id)
            for q in c.quantities:
                with self.subTest(case=c.id, quantity=q.label):
                    a, m = abs(q.analytical), abs(q.measured(res, struc))
                    tol = max(q.abs_tol, q.rel_tol * abs(a))
                    self.assertLessEqual(
                        abs(m - a), tol,
                        f"{c.id} — {q.label}: analytical {a:.6g} {q.unit}, "
                        f"FEM {m:.6g} {q.unit} (|Δ|={abs(m - a):.3e} > {tol:.3e})")

    def test_torsion_uses_the_engine_J(self):
        """The analytical twist is TL/GJ with the same St-Venant J the element
        derives, so the torsion case is a genuine check of GJ, not a tautology
        with a hand-picked J."""
        struc, res = _solve("G.2")
        m = abs(vc.uy(res, vc.CASE, f"N{vc.GRID_NSEG}"))
        self.assertAlmostEqual(m, abs(vc.GRID_TWIST), places=9)


class TestCutCases(unittest.TestCase):
    """Section-cut resultants (``xdfem2d.cuts.cut_result``) vs the same
    closed-form V(x), M(x) validated for case 2.2 — one cut on a shared node,
    one inside an element."""

    def test_quantities(self):
        for c in vc.CUT_CASES:
            struc, res = _solve(c.id)
            for q in c.quantities:
                with self.subTest(case=c.id, quantity=q.label):
                    a = abs(q.analytical)
                    m = abs(q.measured(res, struc))
                    tol = max(q.abs_tol, q.rel_tol * a)
                    self.assertLessEqual(
                        abs(m - a), tol,
                        f"{c.id} — {q.label}: analytical {a:.6g} {q.unit}, "
                        f"FEM {m:.6g} {q.unit} (|Δ|={abs(m - a):.3e} > {tol:.3e})")


class TestGlobalEquilibrium(unittest.TestCase):
    """ΣReactions + ΣAppliedLoads − ΣSpringForces ≈ 0 on every case.

    Springs are not supports, so the force they take never appears in
    ``reactions``; on the sprung cases it has to be added explicitly (it is
    zero on every case without springs, so the check is uniform)."""

    def test_equilibrium(self):
        for c in vc.CASES:
            struc, res = _solve(c.id)
            with self.subTest(case=c.id):
                fx, fy = vc.applied_resultant(struc, vc.CASE)
                rx, ry = vc.reaction_resultant(res, vc.CASE)
                sx, sy = vc.spring_resultant(res, struc, vc.CASE)
                self.assertAlmostEqual(rx + fx - sx, 0.0, places=6,
                                       msg=f"{c.id} ΣFx")
                self.assertAlmostEqual(ry + fy - sy, 0.0, places=6,
                                       msg=f"{c.id} ΣFy")

    def test_equilibrium_3_2(self):
        for mid, _b, _lbl in vc.CASE_3_2_MESHES:
            struc, res = _solve(mid)
            with self.subTest(mesh=mid):
                fx, fy = vc.applied_resultant(struc, vc.CASE)
                rx, ry = vc.reaction_resultant(res, vc.CASE)
                self.assertAlmostEqual(rx + fx, 0.0, places=5)
                self.assertAlmostEqual(ry + fy, 0.0, places=5)


class TestCase32Convergence(unittest.TestCase):
    """Case 3.2 — CST cantilever: monotone convergence toward beam theory.

    A CST mesh overestimates the flexural stiffness, so the tip deflection
    approaches the (approximate) beam-theory target 17.26 mm from below as the
    mesh is refined. We check that it grows monotonically and reaches within
    5 % on the finest mesh — not an exact value, which CST cannot give here."""

    def setUp(self):
        self.defl = []
        for mid, _b, _lbl in vc.CASE_3_2_MESHES:
            struc, res = _solve(mid)
            self.defl.append(vc.case_3_2_tip_deflection(res, struc))

    def test_monotone_convergence(self):
        for a, b in zip(self.defl, self.defl[1:]):
            self.assertGreater(b, a, "refinement must increase the deflection")

    def test_finest_within_5pct_from_below(self):
        finest = self.defl[-1]
        target = vc.CASE_3_2_TARGET
        self.assertLess(finest, target, "CST must not overshoot beam theory")
        self.assertGreater(finest, 0.95 * target,
                           f"finest {finest:.3f} mm too far below {target} mm")


class TestArcConvergence(unittest.TestCase):
    """Quarter-circle cantilever (arc object): the tip deflection converges to
    the closed-form δ_v = π P R³/(4EI) as the arc is refined into more chords.
    The straight-chord mesh plus axial/shear make it approach from just above
    the bending-only value; we check monotone convergence and a tight band on
    the finest mesh."""

    def setUp(self):
        self.defl = []
        for mid, _b, _lbl in vc.ARC_MESHES:
            struc, res = _solve(mid)
            self.defl.append(vc.arc_tip_deflection(res, struc))

    def test_monotone_convergence(self):
        for a, b in zip(self.defl, self.defl[1:]):
            self.assertGreaterEqual(b, a - 1e-6)

    def test_finest_matches_analytical(self):
        self.assertLess(abs(self.defl[-1] - vc.ARC_TARGET),
                        0.015 * vc.ARC_TARGET,
                        f"{self.defl[-1]:.4f} vs {vc.ARC_TARGET:.4f} mm")


class TestCase32AllmanConvergence(unittest.TestCase):
    """Case 3.2 with the Allman element: same cantilever wall-beam, but the
    drilling DOF (left free) makes it converge to beam theory faster than the
    CST. We check monotone convergence and that the finest Allman mesh is
    closer to the target than the same-size CST mesh (its whole point)."""

    def _defl(self, meshes):
        out = []
        for mid, _b, _lbl in meshes:
            struc, res = _solve(mid)
            nid = vc.node_at(struc, 4.0, 0.25)
            out.append(abs(vc.uy(res, vc.CASE, nid)) * vc.MM)
        return out

    def test_monotone_convergence(self):
        d = self._defl(vc.CASE_3_2_ALLMAN_MESHES)
        for a, b in zip(d, d[1:]):
            self.assertGreater(b, a)

    def test_allman_beats_cst_at_equal_mesh(self):
        allman = self._defl(vc.CASE_3_2_ALLMAN_MESHES)[-1]      # 32×8
        cst = self._defl(vc.CASE_3_2_MESHES[1:2])[0]            # 32×8 CST
        t = vc.CASE_3_2_TARGET
        self.assertLess(abs(allman - t), abs(cst - t),
                        f"Allman {allman:.2f} should beat CST {cst:.2f} "
                        f"toward {t}")

    def test_finest_within_3pct(self):
        finest = self._defl(vc.CASE_3_2_ALLMAN_MESHES)[-1]
        self.assertGreater(finest, 0.97 * vc.CASE_3_2_TARGET)
        self.assertLess(finest, vc.CASE_3_2_TARGET)


if __name__ == "__main__":
    unittest.main(verbosity=2)
