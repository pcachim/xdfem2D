"""Validation of node springs and element (Winkler) springs against closed-form
solutions — the spring counterpart of ``test_validation_x2d.py``.

The exact linear cases (s-a1…s-a4, s-b1, s-b5) live in ``validation_cases.CASES``
and are already checked quantity-by-quantity by ``test_validation_x2d``; this
module adds what does not fit that mould:

  * the convergence families (beam on elastic foundation), because the element
    spring is LUMPED (k·L/2 at each end node) and so is only O(h²)-accurate on a
    non-uniform state;
  * the lumping artefact of the uniform state (a spurious M of order q·h²/8 that
    must be quartered at every refinement);
  * the unilateral (tension-/compression-only) cases, solved by NonLinear
    analysis cases, including the classic rigid footing with partial contact;
  * the limiting behaviours k→0 (no spring) and k→∞ (rigid support), and the
    equivalence of a 'local' element spring with its rotated 'global' twin.

Run:  python -m pytest tests/tests_model/test_springs_validation.py
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

import validation_cases as vc              # noqa: E402
from xdfem2d import Structure2D            # noqa: E402


def _solve(build):
    s = build()
    return s, s.calculate()


def _check(testcase, case, res, struc):
    for q in case.quantities:
        with testcase.subTest(case=case.id, quantity=q.label):
            a, m = q.analytical, q.measured(res, struc)
            if not q.signed:
                a, m = abs(a), abs(m)
            tol = max(q.abs_tol, q.rel_tol * abs(a))
            testcase.assertLessEqual(
                abs(m - a), tol,
                f"{case.id} — {q.label}: teórico {a:.6g} {q.unit}, "
                f"MEF {m:.6g} {q.unit} (|Δ|={abs(m - a):.3e} > {tol:.3e})")


# ── Exact linear cases (re-checked here so the module is self-contained) ─────
class TestLinearSpringCases(unittest.TestCase):
    """s-a1…s-a4, s-b1, s-b5 — every quantity vs its closed form."""

    def test_quantities(self):
        for c in vc.CASES:
            if not c.id.startswith("s-"):
                continue
            struc, res = _solve(c.build)
            _check(self, c, res, struc)

    def test_equilibrium_includes_springs(self):
        for c in vc.CASES:
            if not c.id.startswith("s-"):
                continue
            struc, res = _solve(c.build)
            with self.subTest(case=c.id):
                fx, fy = vc.applied_resultant(struc, vc.CASE)
                rx, ry = vc.reaction_resultant(res, vc.CASE)
                sx, sy = vc.spring_resultant(res, struc, vc.CASE)
                self.assertAlmostEqual(rx + fx - sx, 0.0, places=6)
                self.assertAlmostEqual(ry + fy - sy, 0.0, places=6)


# ── Limiting behaviour of a node spring ─────────────────────────────────────
def _cantilever_with_ky(k):
    """vc's s-a2 cantilever with an arbitrary tip spring stiffness."""
    s = Structure2D()
    s.add_material("C", vc.E, 0.0, alpha=vc.ALPHA, poisson=vc.NU)
    s.add_section("S", "C", b=vc.B, h=vc.H)
    s.add_node("A", 0, 0); s.add_node("B", vc.SA2_L, 0)
    s.add_bar_element("E", "A", "B", "S")
    s.add_support("FIX", ux=True, uy=True, tz=True); s.assign_support("A", "FIX")
    if k:
        s.add_node_spring("B", ky=k)
    s.add_load_case("LC"); s.add_point_load("B", "LC", fy=-vc.SA2_P)
    return s


class TestNodeSpringLimits(unittest.TestCase):

    def test_zero_stiffness_equals_no_spring(self):
        a = _cantilever_with_ky(0.0).calculate()["displacements"]["LC"]["B"][1]
        b = _cantilever_with_ky(None).calculate()["displacements"]["LC"]["B"][1]
        free = -vc.SA2_P * vc.SA2_L ** 3 / (3 * vc.E * vc.I_BAR)
        self.assertAlmostEqual(a, b, places=12)
        self.assertAlmostEqual(a, free, delta=abs(free) * 1e-9)

    def test_huge_stiffness_approaches_rigid_support(self):
        """k → ∞ must reproduce the propped cantilever: the tip does not move
        and the fixed-end moment is the textbook 3PL/16 (load at midspan)."""
        L, P = vc.SA2_L, vc.SA2_P
        s = Structure2D()
        s.add_material("C", vc.E, 0.0)
        s.add_section("S", "C", b=vc.B, h=vc.H)
        s.add_node("A", 0, 0); s.add_node("C", L / 2, 0); s.add_node("B", L, 0)
        s.add_bar_element("E1", "A", "C", "S")
        s.add_bar_element("E2", "C", "B", "S")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("A", "FIX")
        s.add_node_spring("B", ky=1.0e8 * vc.SA2_KB)
        s.add_load_case("LC"); s.add_point_load("C", "LC", fy=-P)
        res = s.calculate()
        d = res["displacements"]["LC"]["B"][1]
        free = P * L ** 3 / (3 * vc.E * vc.I_BAR)
        self.assertLess(abs(d), 1e-6 * free)
        m = abs(res["reactions"]["LC"]["A"][2])
        self.assertAlmostEqual(m, 3 * P * L / 16.0, delta=1e-5 * P * L)

    def test_rotational_spring_limits(self):
        """kt→0 → simply supported (M_apoio=0); kt→∞ → clamped (qL²/12)."""
        def m_end(kt):
            s = Structure2D()
            s.add_material("C", vc.E, 0.0)
            s.add_section("S", "C", b=vc.B, h=vc.H)
            s.add_node("A", 0, 0); s.add_node("C", vc.SA3_L / 2, 0)
            s.add_node("B", vc.SA3_L, 0)
            s.add_bar_element("E1", "A", "C", "S")
            s.add_bar_element("E2", "C", "B", "S")
            s.add_support("PIN", ux=True, uy=True)
            s.add_support("ROL", ux=False, uy=True)
            s.assign_support("A", "PIN"); s.assign_support("B", "ROL")
            if kt:
                s.add_node_spring("A", kt=kt); s.add_node_spring("B", kt=kt)
            s.add_load_case("LC")
            s.add_distributed_load("E1", "LC", fye=-vc.SA3_Q, fyd=-vc.SA3_Q)
            s.add_distributed_load("E2", "LC", fye=-vc.SA3_Q, fyd=-vc.SA3_Q)
            r = s.calculate()
            return abs(r["element_forces"]["LC"]["E1"]["i"][2])

        clamped = vc.SA3_Q * vc.SA3_L ** 2 / 12.0
        self.assertAlmostEqual(m_end(0.0), 0.0, places=6)
        self.assertAlmostEqual(m_end(1.0e6 * vc.SA3_KT), clamped,
                               delta=1e-4 * clamped)


# ── Element (Winkler) springs ───────────────────────────────────────────────
class TestWinklerUniformState(unittest.TestCase):
    """Rigid block on a uniform foundation: w = q/k for any mesh, and the
    spurious bending moment left by the lumping decays as O(h²)."""

    def test_settlement_is_mesh_independent(self):
        for mid, build, lbl in vc.SB1_MESHES:
            struc, res = _solve(build)
            with self.subTest(mesh=lbl):
                for nid in ("N0", f"N{len(struc.nodes) - 1}"):
                    w = vc.uy(res, vc.CASE, nid)
                    self.assertAlmostEqual(w, vc.SB1_W,
                                           delta=1e-5 * abs(vc.SB1_W))

    def test_lumping_moment_is_second_order(self):
        m = []
        for mid, build, lbl in vc.SB1_MESHES:
            struc, res = _solve(build)
            h = vc.SB1_L / (len(struc.nodes) - 1)
            mmax = vc.max_abs_M_all(res, vc.CASE)
            # the artefact is the free-span parabola of the un-balanced q
            self.assertAlmostEqual(mmax, vc.SB1_Q * h ** 2 / 8.0,
                                   delta=1e-3 * vc.SB1_Q * h ** 2 / 8.0)
            m.append(mmax)
        for a, b in zip(m, m[1:]):
            self.assertAlmostEqual(b / a, 0.25, delta=1e-3)   # halving h → M/4


class TestBeamOnElasticFoundation(unittest.TestCase):
    """Hetényi's closed forms, approached as O(h²) by the lumped foundation."""

    def _sweep(self, meshes, extract):
        out = []
        for mid, build, lbl in meshes:
            struc, res = _solve(build)
            out.append((extract(res, struc), vc.max_abs_M_all(res, vc.CASE)))
        return out

    def test_infinite_beam_central_load(self):
        vals = self._sweep(vc.SB2_MESHES, vc.winkler_w0)
        errs_w = [abs(w - vc.SB2_W0) for w, _m in vals]
        errs_m = [abs(m - vc.SB2_M0) for _w, m in vals]
        self.assertTrue(all(b < a for a, b in zip(errs_m, errs_m[1:])),
                        f"M must converge monotonically: {errs_m}")
        self.assertLess(errs_w[-1], 0.01 * vc.SB2_W0,
                        f"w0 = {vals[-1][0]:.6g} vs {vc.SB2_W0:.6g}")
        self.assertLess(errs_m[-1], 0.01 * vc.SB2_M0,
                        f"M0 = {vals[-1][1]:.6g} vs {vc.SB2_M0:.6g}")

    def test_semi_infinite_beam_end_load(self):
        vals = self._sweep(vc.SB3_MESHES, vc.winkler_w_end)
        errs_w = [abs(w - vc.SB3_W0) for w, _m in vals]
        errs_m = [abs(m - vc.SB3_MMAX) for _w, m in vals]
        for errs, name in ((errs_w, "w"), (errs_m, "M")):
            self.assertTrue(all(b < a for a, b in zip(errs, errs[1:])),
                            f"{name} must converge monotonically: {errs}")
        self.assertLess(errs_w[-1], 0.02 * vc.SB3_W0)
        self.assertLess(errs_m[-1], 0.02 * vc.SB3_MMAX)


class TestLocalElementSpring(unittest.TestCase):
    """A 'local' (axial/transverse) foundation on an inclined bar must equal the
    rotated 'global' block — and for ka = kt the two are literally the same."""

    def test_isotropic_local_equals_global(self):
        _s1, r1 = _solve(vc.build_s_b5_iso_local)
        _s2, r2 = _solve(vc.build_s_b5_iso_global)
        for comp in (0, 1):
            self.assertAlmostEqual(r1["displacements"][vc.CASE]["A"][comp],
                                   r2["displacements"][vc.CASE]["A"][comp],
                                   places=12)

    def test_isotropic_local_is_orientation_independent(self):
        """ka = kt → the block is k·I, so an inclined bar settles exactly like a
        horizontal one: w = q/k with no horizontal drift."""
        _s, r = _solve(vc.build_s_b5_iso_local)
        u, v = r["displacements"][vc.CASE]["A"][:2]
        w = vc.SB5_Q / vc.SB5_K_ISO
        self.assertAlmostEqual(v, -w, delta=1e-5 * w)
        self.assertAlmostEqual(u, 0.0, delta=1e-5 * w)

    def test_anisotropic_block_matches_the_rotation(self):
        """ka ≠ kt: the rigid-body translation must solve G·(u,v) = (0, −q)
        with G = R·diag(ka,kt)·Rᵀ — i.e. the load in Y also produces a drift
        in X, which a 'global' spring could never reproduce."""
        _s, r = _solve(vc.build_s_b5)
        u, v = r["displacements"][vc.CASE]["A"][:2]
        self.assertAlmostEqual(u, vc.SB5_U, delta=1e-6 * abs(vc.SB5_U))
        self.assertAlmostEqual(v, vc.SB5_V, delta=1e-6 * abs(vc.SB5_V))
        self.assertGreater(abs(u), 0.1 * abs(v), "deve haver acoplamento x–y")


# ── Unilateral springs (NonLinear analysis cases) ───────────────────────────
class TestUnilateralNodeSprings(unittest.TestCase):
    """s-c1a/b, s-c2a/b: tension-/compression-only tip spring on a cantilever.
    Active → the bilateral closed form; inactive → the bare cantilever."""

    def test_quantities(self):
        for c in vc.SPRING_NL_CASES:
            struc, res = _solve(c.build)
            self.assertTrue(res["analysis_cases"]["NL"]["converged"], c.id)
            _check(self, c, res, struc)

    def test_linear_case_ignores_the_mode(self):
        """A Linear case must stay bilateral even when the spring lifts off."""
        _s, res = _solve(vc.build_s_c1b)
        lin = res["analysis_cases"]["LIN"]["displacements"]["B"][1]
        self.assertAlmostEqual(abs(lin), vc.SC1_D_ACTIVE,
                               delta=1e-9 * vc.SC1_D_ACTIVE)


class TestRigidFootingPartialContact(unittest.TestCase):
    """Rigid strip footing, load at e = 1.5 m > L/6: the compression-only
    Winkler foundation separates over part of the base. Closed form:
    contact length a = 3(L/2 − e), triangular pressure p_max = 2N/a, so the
    settlement at the loaded edge is p_max/k. The lumped discrete model
    approaches it from below as the mesh is refined."""

    def setUp(self):
        self.runs = []
        for mid, build, lbl in vc.SC3_MESHES:
            struc, res = _solve(build)
            self.assertTrue(res["analysis_cases"]["NL"]["converged"], mid)
            self.runs.append((lbl, struc, res))

    def test_contact_length(self):
        for lbl, struc, res in self.runs:
            h = vc.SC3_L / int(lbl)
            with self.subTest(mesh=lbl):
                self.assertAlmostEqual(vc.c3_contact_length(res, struc),
                                       vc.SC3_A, delta=h)

    def test_edge_settlement_converges(self):
        errs = []
        for lbl, struc, res in self.runs:
            w = vc.c3_w_edge(res, struc)
            self.assertLess(abs(w), abs(vc.SC3_WMAX),
                            "o modelo discreto aproxima por baixo")
            errs.append(abs(w - vc.SC3_WMAX))
        for a, b in zip(errs, errs[1:]):
            self.assertLess(b, a, f"convergência não monótona: {errs}")
        self.assertLess(errs[-1], 0.01 * abs(vc.SC3_WMAX))

    def test_uplift_differs_from_the_bilateral_solution(self):
        """The whole point: ignoring the uplift (Linear) gives a different,
        fully-linear pressure diagram — the two must not coincide."""
        _lbl, struc, res = self.runs[-1]
        nl = vc.c3_w_edge(res, struc, "NL")
        lin = vc.c3_w_edge(res, struc, "LIN")
        self.assertGreater(abs(nl - lin), 0.05 * abs(nl))
        # and the Linear model keeps the whole base in contact
        d = res["analysis_cases"]["LIN"]["displacements"]
        self.assertGreater(max(d[n.id][1] for n in struc.nodes.values()), 0.0,
                           "o modelo linear traciona a fundação no bordo oposto")


if __name__ == "__main__":
    unittest.main(verbosity=2)
