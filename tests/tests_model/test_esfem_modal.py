"""Modal reference: ES-FEM vs CST on the *same* triangular mesh.

A slender cantilever wall (in-plane bending dominated) meshed into T3 triangles.
The CST triangle is over-stiff, so on a coarse mesh it *overestimates* the
natural frequencies; the edge-based smoothing of ES-FEM relaxes that excess
stiffness, so on the same mesh ES-FEM gives lower frequencies that are closer to
the mesh-converged value.

Two properties are checked:
  * on one coarse mesh, the ES-FEM fundamental frequency is lower than the CST
    one (softer), and both are real and positive;
  * against a fine-mesh reference, the coarse ES-FEM fundamental frequency is
    closer than the coarse CST one.

Requires SciPy (the modal eigensolver). Run with, e.g.::

    python -m pytest tests/tests_model/test_esfem_modal.py -v
"""
from __future__ import annotations

import math
import unittest

import context  # noqa: F401  (puts src/ on the path)
from xdfem2d import Structure2D


# Cantilever wall geometry / material (concrete-like, SI: kN, m, t).
_L, _H, _T = 4.0, 1.0, 0.2          # length, height, thickness [m]
_E = 30.0e6                          # kN/m²  (~30 GPa)
_GAMMA = 25.0                        # kN/m³
_NU = 0.2


def _wall(formulation: str, nx: int, ny: int) -> Structure2D:
    """A cantilever wall (fixed left edge) meshed nx×ny into 2 triangles/cell,
    with self-weight-derived mass and a modal case (3 modes)."""
    s = Structure2D()
    s.add_material("C", elastic_modulus=_E, unit_weight=_GAMMA,
                   material_type="Concrete", poisson=_NU)
    s.add_tri_section("W", "C", thickness=_T, formulation=formulation)

    def nid(i, j):
        return f"n{i}_{j}"

    for i in range(nx + 1):
        for j in range(ny + 1):
            s.add_node(nid(i, j), i * _L / nx, j * _H / ny)

    t = 0
    for i in range(nx):
        for j in range(ny):
            a, b = nid(i, j), nid(i + 1, j)
            c, d = nid(i + 1, j + 1), nid(i, j + 1)
            t += 1; s.add_tri_element(f"t{t}", a, b, c, "W")
            t += 1; s.add_tri_element(f"t{t}", a, c, d, "W")

    # Fix the left edge (x = 0).
    s.add_support("FIX", ux=True, uy=True, tz=True)
    for j in range(ny + 1):
        s.assign_support(nid(0, j), "FIX")

    # Mass from self-weight: a load case with self-weight, and a Mass case that
    # converts its vertical nodal forces to mass (γ·t·A/g), so the total mass is
    # the same on any mesh density.
    s.add_load_case("G", self_weight_factor=1.0)
    s.add_analysis_case("MASS", "Mass", {"G": 1.0})
    s.add_analysis_case("MODAL", "Modal", {}, modal_case_id="MASS", num_modes=3)
    return s


def _f1(struc: Structure2D) -> float:
    """Fundamental frequency [Hz] of a solved model."""
    r = struc.calculate()
    mi = r["analysis_cases"]["MODAL"]["modal_info"]
    assert mi, "no modes computed"
    return mi[0]["frequency"]


class TestESFEMModal(unittest.TestCase):

    def test_both_give_a_real_positive_fundamental(self):
        for form in ("CST", "ES-FEM"):
            f = _f1(_wall(form, 6, 3))
            self.assertTrue(math.isfinite(f) and f > 0.0,
                            f"{form}: f1 = {f}")

    def test_esfem_is_softer_than_cst_on_the_same_mesh(self):
        f_cst = _f1(_wall("CST", 6, 3))
        f_es = _f1(_wall("ES-FEM", 6, 3))
        print(f"\n[ES-FEM modal] same 6x3 mesh — f1:  "
              f"CST = {f_cst:.4f} Hz   ES-FEM = {f_es:.4f} Hz   "
              f"(ES-FEM is {100 * (f_cst - f_es) / f_cst:.1f}% softer)")
        self.assertLess(f_es, f_cst,
                        f"ES-FEM ({f_es:.4f} Hz) should be < CST "
                        f"({f_cst:.4f} Hz) on the same mesh")

    def test_esfem_is_closer_to_the_converged_reference(self):
        # Fine ES-FEM as the best estimate of the exact fundamental frequency.
        f_ref = _f1(_wall("ES-FEM", 24, 12))
        f_cst = _f1(_wall("CST", 6, 3))
        f_es = _f1(_wall("ES-FEM", 6, 3))
        err_cst = abs(f_cst - f_ref)
        err_es = abs(f_es - f_ref)
        print(f"\n[ES-FEM modal] fundamental frequency f1 [Hz]:\n"
              f"    reference (ES-FEM 24x12) = {f_ref:.4f}\n"
              f"    CST     6x3              = {f_cst:.4f}   "
              f"(error {err_cst:.4f} Hz, {100 * err_cst / f_ref:.1f}%)\n"
              f"    ES-FEM  6x3              = {f_es:.4f}   "
              f"(error {err_es:.4f} Hz, {100 * err_es / f_ref:.1f}%)")
        self.assertLess(err_es, err_cst,
                        f"coarse ES-FEM error ({err_es:.4f}) should be < coarse "
                        f"CST error ({err_cst:.4f}); ref = {f_ref:.4f} Hz")


if __name__ == "__main__":
    unittest.main()
