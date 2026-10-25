"""CST plane triangle: element math, the patch test, and frame↔membrane coupling."""
from __future__ import annotations

import unittest

import numpy as np

from context import Structure2D
from xdfem2d.tri_elements import (plane_D, cst_B_area, cst_stiffness,
                                   tri_stresses)
from xdfem2d.assembly import assemble_stiffness
from xdfem2d.loads import assemble_loads
from xdfem2d.solver import solve


class TestCSTElement(unittest.TestCase):
    def test_area_and_B_shape(self):
        B, A = cst_B_area([(0, 0), (2, 0), (0, 1)])
        self.assertAlmostEqual(A, 1.0)            # ½·2·1
        self.assertEqual(B.shape, (3, 6))

    def test_single_element_reproduces_constant_strain(self):
        # Impose a linear displacement field on the 3 nodes; the recovered
        # strain must equal the exact constant strain of that field.
        coords = [(0.0, 0.0), (2.0, 0.0), (0.5, 1.5)]
        E, nu, t = 30e6, 0.2, 0.1
        # u = a·x + b·y ; v = c·x + d·y  → εx=a, εy=d, γxy=b+c
        a, b, c, d = 1e-3, 5e-4, 3e-4, -2e-4
        ue = np.array([a*coords[0][0]+b*coords[0][1], c*coords[0][0]+d*coords[0][1],
                       a*coords[1][0]+b*coords[1][1], c*coords[1][0]+d*coords[1][1],
                       a*coords[2][0]+b*coords[2][1], c*coords[2][0]+d*coords[2][1]])
        _k, B, D, _A = cst_stiffness(coords, E, nu, t)
        eps = B @ ue
        self.assertAlmostEqual(eps[0], a, places=9)          # εx
        self.assertAlmostEqual(eps[1], d, places=9)          # εy
        self.assertAlmostEqual(eps[2], b + c, places=9)      # γxy
        sig = D @ eps
        exact = D @ np.array([a, d, b + c])
        np.testing.assert_allclose(sig, exact, rtol=1e-9)


class TestPatchTest(unittest.TestCase):
    def test_patch_constant_stress(self):
        """A mesh of CSTs with an interior node, driven by a linear boundary
        displacement field, must give the interior node the exact field value
        and the same constant stress in every element."""
        s = Structure2D()
        s.add_material("M", 30e6, 0.0, poisson=0.2)   # no self-weight
        s.add_tri_section("S", "M", thickness=0.1)
        s.add_load_case("LC")
        s.add_node("1", 0.0, 0.0); s.add_node("2", 2.0, 0.0)
        s.add_node("3", 2.0, 1.0); s.add_node("4", 0.0, 1.0)
        s.add_node("5", 0.7, 0.4)                     # interior
        for tid, (i, j, k) in enumerate([("1", "2", "5"), ("2", "3", "5"),
                                         ("3", "4", "5"), ("4", "1", "5")], 1):
            s.add_tri_element(f"T{tid}", i, j, k, "S")
        # Linear field u=a·x+b·y, v=c·x+d·y
        a, b, c, d = 1e-3, 5e-4, 3e-4, -2e-4
        field = lambda x, y: (a*x + b*y, c*x + d*y)
        s.add_support("B", ux=True, uy=True)
        for nid in ("1", "2", "3", "4"):
            s.assign_support(nid, "B")
            n = s.nodes[nid]; ux, uy = field(n.x, n.y)
            s.create_support_settlement(nid, "LC", ux=ux, uy=uy)

        K = assemble_stiffness(s)
        F = assemble_loads(s)
        U = solve(s, K, F)
        u0 = U[:, 0]
        # Interior node 5 must take the exact linear field value.
        b5 = s.node_dof_index["5"]
        exp_u, exp_v = field(0.7, 0.4)
        self.assertAlmostEqual(u0[b5],   exp_u, places=9)
        self.assertAlmostEqual(u0[b5+1], exp_v, places=9)
        # Every element: the same constant stress = D·[a, d, b+c].
        D = plane_D(30e6, 0.2)
        exact = D @ np.array([a, d, b + c])
        st = tri_stresses(s, u0)
        for tid, sig in st.items():
            np.testing.assert_allclose([sig['sx'], sig['sy'], sig['txy']],
                                       exact, rtol=1e-7, atol=1e-6)


class TestFrameMembraneCoupling(unittest.TestCase):
    def test_membrane_only_node_tz_autofixed_and_solves(self):
        # A panel of two triangles, pinned on one edge, pulled on the other.
        # Membrane-only nodes have no rotational stiffness → tz auto-restrained;
        # the model must solve (no singular matrix).
        s = Structure2D()
        s.add_material("M", 30e6, 0.0, poisson=0.2)
        s.add_tri_section("S", "M", thickness=0.2)
        s.add_load_case("LC")
        s.add_node("1", 0, 0); s.add_node("2", 1, 0)
        s.add_node("3", 1, 1); s.add_node("4", 0, 1)
        s.add_tri_element("A", "1", "2", "3", "S")
        s.add_tri_element("B", "1", "3", "4", "S")
        s.add_support("Fix", ux=True, uy=True)
        s.assign_support("1", "Fix"); s.assign_support("4", "Fix")
        s.add_point_load("2", "LC", fx=100.0)
        s.add_point_load("3", "LC", fx=100.0)
        res = s.calculate()                          # must not raise
        # Loaded nodes move in +x; reactions balance the applied load.
        d2 = res["analysis_cases"]["LC"]["displacements"]["2"]
        self.assertGreater(d2[0], 0.0)


class TestTriLoads(unittest.TestCase):
    def test_self_weight_reaction_equals_total_weight(self):
        # Unit square panel, γ·t·A downward; vertical reactions must sum to the
        # total weight of the two triangles.
        s = Structure2D()
        gamma, t = 25.0, 0.2
        s.add_material("C", 30e6, gamma, poisson=0.2)
        s.add_tri_section("S", "C", thickness=t)
        lc = s.add_load_case("G"); lc.self_weight_factor = 1.0
        s.add_node("1", 0, 0); s.add_node("2", 1, 0)
        s.add_node("3", 1, 1); s.add_node("4", 0, 1)
        s.add_tri_element("A", "1", "2", "3", "S")
        s.add_tri_element("B", "1", "3", "4", "S")
        s.add_support("Pin", ux=True, uy=True)
        for nid in ("1", "2"):
            s.assign_support(nid, "Pin")
        res = s.calculate()
        reac = res["analysis_cases"]["G"]["reactions"]
        ry = sum(r[1] for r in reac.values())
        W = gamma * t * 1.0                       # area of square = 1
        self.assertAlmostEqual(ry, W, places=6)   # upward reactions = weight

    def test_edge_pressure_reaction(self):
        # Push the free (right) edge of a pinned panel with a normal pressure;
        # the horizontal reactions must balance the applied resultant.
        s = Structure2D()
        s.add_material("C", 30e6, 0.0, poisson=0.2)
        s.add_tri_section("S", "C", thickness=0.5)
        s.add_load_case("P")
        s.add_node("1", 0, 0); s.add_node("2", 1, 0)
        s.add_node("3", 1, 1); s.add_node("4", 0, 1)
        s.add_tri_element("A", "1", "2", "3", "S")
        s.add_tri_element("B", "1", "3", "4", "S")
        s.add_support("Pin", ux=True, uy=True)
        for nid in ("1", "4"):
            s.assign_support(nid, "Pin")
        # Edge 2→3 (right side), normal pressure pn pushing in −x (right-hand
        # normal of 2→3 points +x, so pn<0 pushes left/into the panel).
        s.add_tri_edge_load("EL", "A", "2", "3", "P", coord_sys="local", pn=-10.0)
        res = s.calculate()
        reac = res["analysis_cases"]["P"]["reactions"]
        rx = sum(r[0] for r in reac.values())
        # Applied resultant Fx = pn·nx·thk·L = -10·1·0.5·1 = -5 kN → ΣRx = +5.
        self.assertAlmostEqual(rx, 5.0, places=6)

    def test_edge_load_global_fxy(self):
        # Global Fx/Fy per unit length on the right edge of a pinned panel.
        s = Structure2D()
        s.add_material("C", 30e6, 0.0, poisson=0.2)
        s.add_tri_section("S", "C", thickness=0.5)
        s.add_load_case("P")
        s.add_node("1", 0, 0); s.add_node("2", 1, 0)
        s.add_node("3", 1, 1); s.add_node("4", 0, 1)
        s.add_tri_element("A", "1", "2", "3", "S")
        s.add_tri_element("B", "1", "3", "4", "S")
        s.add_support("Pin", ux=True, uy=True)
        for nid in ("1", "4"):
            s.assign_support(nid, "Pin")
        # Edge 2→3 (length 1), fx = 8 kN/m → resultant 8 kN in +x.
        s.add_tri_edge_load("EL", "A", "2", "3", "P", fx=8.0)
        res = s.calculate()
        reac = res["analysis_cases"]["P"]["reactions"]
        rx = sum(r[0] for r in reac.values())
        self.assertAlmostEqual(rx, -8.0, places=6)   # reactions balance +8 kN


class TestSurfaceEdgeLoad(unittest.TestCase):
    def test_surface_edge_load_equilibrium(self):
        from xdfem2d.geo_expand import expand_geometry
        s = Structure2D()
        s.add_material("C", 30e6, 0.0, poisson=0.2)
        s.add_tri_section("W", "C", thickness=0.3)
        s.add_load_case("P")
        s.add_node("BL", 0, 0); s.add_node("BR", 4, 0)
        s.add_geo_rectangle("R", (0, 0), (4, 2),
                            section_name="W", target_size=0.5)
        s.add_support("Pin", ux=True, uy=True)
        s.assign_support("BL", "Pin"); s.assign_support("BR", "Pin")
        # Top side of the rectangle: corners are R.p0..R.p3 (CCW from BL).
        m = s.geometry_objects["R"]
        top_a, top_b = m.node_ids[3], m.node_ids[2]   # top-left → top-right
        s.add_surface_edge_load("SEL", "R", top_a, top_b, "P", fy=-10.0)
        res = s.calculate()
        case = next(iter(res["analysis_cases"]))
        reac = res["analysis_cases"][case]["reactions"]
        ry = sum(r[1] for r in reac.values())
        # Applied resultant Fy = -10 kN/m × 4 m = -40 kN → ΣRy = +40.
        self.assertAlmostEqual(ry, 40.0, places=6)


class TestTriStressResults(unittest.TestCase):
    def test_stress_present_in_results(self):
        s = Structure2D()
        s.add_material("C", 30e6, 0.0, poisson=0.2)
        s.add_tri_section("S", "C", thickness=0.1)
        s.add_load_case("P")
        s.add_node("1", 0, 0); s.add_node("2", 2, 0)
        s.add_node("3", 2, 1); s.add_node("4", 0, 1)
        s.add_tri_element("A", "1", "2", "3", "S")
        s.add_tri_element("B", "1", "3", "4", "S")
        s.add_support("Pin", ux=True, uy=True)
        for nid in ("1", "4"):
            s.assign_support(nid, "Pin")
        s.add_point_load("2", "P", fx=50.0); s.add_point_load("3", "P", fx=50.0)
        res = s.calculate()
        ts = res["analysis_cases"]["P"]["tri_stress"]
        self.assertEqual(set(ts), {"A", "B"})
        for d in ts.values():
            self.assertTrue(all(k in d for k in
                                ("sx", "sy", "txy", "s1", "s2", "vm")))
        self.assertGreater(max(d["vm"] for d in ts.values()), 0.0)


class TestTriSectionMigration(unittest.TestCase):
    def test_legacy_material_thickness_migrates_to_section(self):
        # A pre-section file stored material_name + thickness on the element.
        from xdfem2d.structure_io import _from_dict, _to_dict
        legacy = {
            'materials': [{'name': 'M', 'elastic_modulus': 30e6,
                           'unit_weight': 0.0, 'poisson': 0.2}],
            'nodes': [{'id': '1', 'x': 0, 'y': 0}, {'id': '2', 'x': 1, 'y': 0},
                      {'id': '3', 'x': 0, 'y': 1}],
            'tri_elements': [{'id': 'A', 'node_i': '1', 'node_j': '2',
                              'node_k': '3', 'material_name': 'M',
                              'thickness': 0.25, 'plane_strain': True}],
        }
        s = _from_dict(legacy)
        self.assertEqual(len(s.tri_sections), 1)
        sec = s.tri_sections[s.tri_elements[0].section_name]
        self.assertEqual(sec.material_name, 'M')
        self.assertAlmostEqual(sec.thickness, 0.25)
        self.assertTrue(sec.plane_strain)
        # And it round-trips through the new (section-based) format.
        s2 = _from_dict(_to_dict(s))
        self.assertEqual(s2.tri_elements[0].section_name,
                         s.tri_elements[0].section_name)
        self.assertAlmostEqual(
            s2.tri_sections[s2.tri_elements[0].section_name].thickness, 0.25)


if __name__ == "__main__":
    unittest.main()
