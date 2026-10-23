"""Allman triangle — the classic patch test (Allman 1984).

The reference validation for a membrane element with drilling (rotational) DOFs.
On the recommended 4-triangle square with a FREE interior node, prescribe the
boundary DOFs — translations AND the drilling rotation θ — to an exact field and
check every element reproduces the constant stress it implies, to machine
precision. Prescribing θ on the boundary is part of the test: a drilling element
is only asked to reproduce a state once its rotational DOFs are pinned to it (a
free boundary θ is an ill-posed patch, not a failure of the element).

Covers the three families from the reference: rigid body (translation and
rotation), and constant strain (uniaxial, biaxial, pure shear).
"""
from __future__ import annotations

import unittest

import numpy as np

import context  # noqa: F401  (puts src/ on the path)
from xdfem2d import Structure2D

E = 30.0e6
NU = 0.20
D = E / (1 - NU ** 2) * np.array([[1, NU, 0], [NU, 1, 0], [0, 0, (1 - NU) / 2]])


def _patch_stresses(field):
    """Solve the 4-triangle patch with the boundary prescribed to *field*
    (x, y) -> (u, v, θ) and the interior node free. Return the per-element
    (sx, sy, txy) list and the interior node's (u, v, θ)."""
    s = Structure2D()
    s.add_material("C", E, 0.0, poisson=NU)
    s.add_tri_section("T", "C", thickness=0.1, formulation="Allman")
    s.add_node("1", 0, 0); s.add_node("2", 1, 0)
    s.add_node("3", 1, 1); s.add_node("4", 0, 1)
    s.add_node("5", 0.5, 0.5)                       # interior — left free
    for e, (i, j) in enumerate([("1", "2"), ("2", "3"), ("3", "4"), ("4", "1")]):
        s.add_tri_element(f"e{e}", i, j, "5", "T")
    s.add_support("B", ux=True, uy=True, tz=True)
    s.add_load_case("LC")
    for nid, (x, y) in {"1": (0, 0), "2": (1, 0),
                        "3": (1, 1), "4": (0, 1)}.items():
        u, v, th = field(x, y)
        s.assign_support(nid, "B")
        s.create_support_settlement(nid, "LC", ux=u, uy=v, tz=th)
    r = s.calculate()
    ts = r["tri_stress"]["LC"]
    sig = [(d["sx"], d["sy"], d["txy"]) for d in ts.values()]
    return sig, r["displacements"]["LC"]["5"]


class TestAllmanPatch(unittest.TestCase):

    def _assert_uniform(self, field, exp, msg):
        sig, _ = _patch_stresses(field)
        for comp, ex in zip(zip(*sig), exp):
            scale = max(abs(ex), 1.0)
            for v in comp:
                self.assertLessEqual(abs(v - ex) / scale, 1e-9,
                                     f"{msg}: got {v:.6g}, expected {ex:.6g}")

    def test_rigid_translation(self):
        self._assert_uniform(lambda x, y: (0.005, -0.003, 0.0),
                             (0.0, 0.0, 0.0), "rigid translation")

    def test_rigid_rotation(self):
        """The drilling-DOF test: a rigid rotation θ=ω with u=-ωy, v=ωx must
        produce no strain. This is what a wrong drilling formulation breaks."""
        w = 1e-3
        self._assert_uniform(lambda x, y: (-w * y, w * x, w),
                             (0.0, 0.0, 0.0), "rigid rotation")

    def test_constant_strain_uniaxial(self):
        a = 1e-3
        exp = tuple(D @ np.array([a, 0.0, 0.0]))     # εx=a, εy=0
        self._assert_uniform(lambda x, y: (a * x, 0.0, 0.0), exp, "uniaxial")

    def test_constant_strain_biaxial(self):
        a = 1e-3
        exp = tuple(D @ np.array([a, a, 0.0]))
        self._assert_uniform(lambda x, y: (a * x, a * y, 0.0), exp, "biaxial")

    def test_constant_strain_pure_shear(self):
        g = 1e-3
        exp = tuple(D @ np.array([0.0, 0.0, g]))     # γxy=g
        self._assert_uniform(lambda x, y: (g / 2 * y, g / 2 * x, 0.0),
                             exp, "pure shear")

    def test_interior_node_solves_to_the_field(self):
        """The free interior node must land on the prescribed linear field —
        u(0.5,0.5)=0.0005, v=0, θ=0 for the uniaxial case."""
        a = 1e-3
        _, u5 = _patch_stresses(lambda x, y: (a * x, 0.0, 0.0))
        self.assertAlmostEqual(u5[0], a * 0.5, places=9)
        self.assertAlmostEqual(u5[1], 0.0, places=9)
        self.assertAlmostEqual(u5[2], 0.0, places=9)


class TestAllmanThermal(unittest.TestCase):
    """Temperature on the Allman element: the load vector and stress recovery
    now cover it, checked against closed form."""

    ALPHA = 1e-5

    def _material(self, s):
        s.add_material("C", E, 0.0, poisson=NU, alpha=self.ALPHA)
        s.add_tri_section("T", "C", thickness=0.1, formulation="Allman")

    def test_fully_restrained_gives_the_thermal_stress(self):
        """All DOFs fixed, uniform ΔT: total strain 0, so σ = −D·ε₀."""
        dT = 50.0
        s = Structure2D(); self._material(s)
        s.add_node("1", 0, 0); s.add_node("2", 1, 0); s.add_node("3", 0, 1)
        s.add_tri_element("e", "1", "2", "3", "T")
        s.add_support("F", ux=True, uy=True, tz=True)
        for n in ("1", "2", "3"):
            s.assign_support(n, "F")
        s.add_load_case("LC")
        s.add_tri_temperature_load("e", "LC", dt_i=dT, dt_j=dT, dt_k=dT)
        d = s.calculate()["tri_stress"]["LC"]["e"]
        e0 = self.ALPHA * dT
        exp = -(D @ np.array([e0, e0, 0.0]))
        self.assertAlmostEqual(d["sx"], exp[0], places=3)
        self.assertAlmostEqual(d["sy"], exp[1], places=3)
        self.assertAlmostEqual(d["txy"], 0.0, places=6)

    def test_free_expansion_gives_zero_stress(self):
        """Minimally restrained plate, uniform ΔT: free to expand, so no stress
        and the far corner moves by α·ΔT·L. Drilling DOF left free."""
        dT = 30.0
        s = Structure2D(); self._material(s)
        L, h = 2.0, 1.0
        ids = {}
        for j in range(2):
            for i in range(3):
                nid = f"n{i}_{j}"; s.add_node(nid, L * i / 2, h * j)
                ids[(i, j)] = nid
        e = 0
        for i in range(2):
            a, b = ids[(i, 0)], ids[(i + 1, 0)]
            c, d = ids[(i + 1, 1)], ids[(i, 1)]
            s.add_tri_element(f"e{e}", a, b, c, "T"); e += 1
            s.add_tri_element(f"e{e}", a, c, d, "T"); e += 1
        s.add_support("PIN", ux=True, uy=True); s.add_support("RY", ux=False, uy=True)
        s.assign_support(ids[(0, 0)], "PIN"); s.assign_support(ids[(2, 0)], "RY")
        s.add_load_case("LC")
        for t in s.tri_elements:
            s.add_tri_temperature_load(t.id, "LC", dt_i=dT, dt_j=dT, dt_k=dT)
        r = s.calculate()
        vm = max(v["vm"] for v in r["tri_stress"]["LC"].values())
        self.assertLess(vm, 1.0)                         # ≈0 kN/m²
        self.assertAlmostEqual(r["displacements"]["LC"][ids[(2, 0)]][0],
                               self.ALPHA * dT * L, places=9)


if __name__ == "__main__":
    unittest.main(verbosity=2)
