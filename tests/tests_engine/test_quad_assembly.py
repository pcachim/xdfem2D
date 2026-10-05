"""Quad assembly + solver wiring (dev/IMPLEMENT_QUAD.md Phase 4).

Unlike the Phase 1/2 kernel tests (pure functions, hand-assembled), these go
through the full ``Structure2D.calculate()`` pipeline — the point of Phase 4
is that ``assembly.py``/``loads.py`` wiring reproduces the kernel-level
results exactly, and that a model mixing triangles and quads solves
consistently (shared-node compatibility, global equilibrium).
"""
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)
from context import assert_close


class TestQM6CantileverFullPipeline(unittest.TestCase):
    """Same pure-bending cantilever as test_quad_elements.py's
    TestQ4QM6PureBending, but built and solved through Structure2D — proves
    _quad_element_matrix/_quad_elements_triplets/assemble_stiffness wiring,
    not just the qm6_stiffness kernel."""

    E, NU, L, H, T = 30.0e9, 0.2, 2.0, 0.2, 0.1
    M_TIP = 1000.0

    def _tip_deflection(self, n):
        s = Structure2D(domain='plane')
        s.add_material('M', elastic_modulus=self.E, unit_weight=0.0,
                       poisson=self.NU)
        s.add_quad_section('QS', 'M', thickness=self.T, formulation='QM6')
        for i in range(n + 1):
            s.add_node(f'B{i}', i * self.L / n, 0.0)
            s.add_node(f'T{i}', i * self.L / n, self.H)
        for i in range(n):
            s.add_quad_element(f'Q{i}', f'B{i}', f'B{i+1}', f'T{i+1}', f'T{i}',
                               'QS')
        s.add_support('FIX', ux=True, uy=True, tz=True)
        s.assign_support('B0', 'FIX')
        s.assign_support('T0', 'FIX')
        s.add_load_case('LC')
        F = self.M_TIP / self.H
        s.add_point_load(f'T{n}', 'LC', fx=F)
        s.add_point_load(f'B{n}', 'LC', fx=-F)
        s.add_analysis_case('AC', 'Linear', {'LC': 1.0})
        r = s.calculate()
        disp = r['analysis_cases']['AC']['displacements']
        return 0.5 * (disp[f'T{n}'][1] + disp[f'B{n}'][1])

    def test_matches_beam_theory(self):
        I = self.T * self.H ** 3 / 12.0
        exact = self.M_TIP * self.L ** 2 / (2.0 * self.E * I)
        for n in (2, 4, 8):
            d = self._tip_deflection(n)
            assert_close(self, abs(d), exact, rel=1e-6,
                         msg=f"full-pipeline QM6 tip deflection, n={n}")


class TestMITC4PlateFullPipeline(unittest.TestCase):
    """Simply-supported square plate under UDL, same benchmark as
    test_quad_elements_plate.py's TestSimplySupportedPlateBenchmark, but
    through Structure2D — proves the plate-domain (w,tx,ty) DOF wiring and
    the quad area-load path in loads.py."""

    A, T, NU, Q, N = 4.0, 0.1, 0.3, 10_000.0, 8
    E = 30.0e9
    ALPHA = 0.00406

    def test_within_1pct(self):
        s = Structure2D(domain='plate')
        s.add_material('M', elastic_modulus=self.E, unit_weight=0.0,
                       poisson=self.NU)
        s.add_quad_section('QS', 'M', thickness=self.T, formulation='MITC4')
        h = self.A / self.N
        nn = self.N + 1

        def nid(i, j):
            return f'N{i}_{j}'

        for i in range(nn):
            for j in range(nn):
                s.add_node(nid(i, j), i * h, j * h)
        for i in range(self.N):
            for j in range(self.N):
                s.add_quad_element(f'Q{i}_{j}', nid(i, j), nid(i + 1, j),
                                   nid(i + 1, j + 1), nid(i, j + 1), 'QS')
        s.add_support('SS', ux=True, uy=False, tz=False)
        for i in range(nn):
            for j in range(nn):
                if i in (0, self.N) or j in (0, self.N):
                    s.assign_support(nid(i, j), 'SS')
        s.add_load_case('LC')
        for i in range(self.N):
            for j in range(self.N):
                s.add_area_load(f'Q{i}_{j}', 'LC', pz=-self.Q)
        s.add_analysis_case('AC', 'Linear', {'LC': 1.0})
        r = s.calculate()
        disp = r['analysis_cases']['AC']['displacements']
        ic = self.N // 2
        w_center = disp[nid(ic, ic)][0]

        D = self.E * self.T ** 3 / (12.0 * (1.0 - self.NU ** 2))
        exact = -self.ALPHA * self.Q * self.A ** 4 / D
        assert_close(self, w_center, exact, rel=0.01,
                     msg="full-pipeline MITC4 SS-plate centre w")


class TestMixedTriQuadPlate(unittest.TestCase):
    """A plate meshed with quads (MITC4) on one side and triangles (DKT) on
    the other, sharing an interface of nodes — dev/IMPLEMENT_QUAD.md Phase 4
    validation: global equilibrium and displacement continuity across the
    tri/quad boundary. Both element kinds carry the same 3 DOFs/node
    (w, tx, ty) in the plate domain (``Structure2D.node_dof_index`` always
    allocates 3 DOFs per node, unused ones simply left at zero stiffness —
    see ``TriSection``'s docstring), so there is no DOF-count mismatch to
    resolve at the interface, unlike the plane-domain CST-vs-Allman case
    this document's Phase 4 section anticipated."""

    def _mixed_model(self):
        s = Structure2D(domain='plate')
        s.add_material('M', elastic_modulus=30.0e9, unit_weight=0.0,
                       poisson=0.2)
        s.add_quad_section('QS', 'M', thickness=0.15, formulation='MITC4')
        s.add_tri_section('TS', 'M', thickness=0.15, formulation='DKT')

        def nid(i, j):
            return f'N{i}_{j}'
        for i in range(4):
            for j in range(3):
                s.add_node(nid(i, j), float(i), float(j))

        qid = 0
        for i in range(2):
            for j in range(2):
                s.add_quad_element(f'Q{qid}', nid(i, j), nid(i + 1, j),
                                   nid(i + 1, j + 1), nid(i, j + 1), 'QS')
                qid += 1
        tid = 0
        for j in range(2):
            s.add_tri_element(f'T{tid}', nid(2, j), nid(3, j), nid(3, j + 1),
                              'TS'); tid += 1
            s.add_tri_element(f'T{tid}', nid(2, j), nid(3, j + 1), nid(2, j + 1),
                              'TS'); tid += 1

        s.add_support('SS', ux=True, uy=False, tz=False)
        for i in range(4):
            for j in range(3):
                if i in (0, 3) or j in (0, 2):
                    s.assign_support(nid(i, j), 'SS')

        s.add_load_case('LC')
        q = -5000.0
        for eid in list(s.quad_elements_by_id) + list(s.tri_elements_by_id):
            s.add_area_load(eid, 'LC', pz=q)
        s.add_analysis_case('AC', 'Linear', {'LC': 1.0})
        return s, q

    def test_domain_consistent(self):
        s, _ = self._mixed_model()
        self.assertEqual(s.domain_problems(), [])
        self.assertEqual(s.reference_problems(), [])

    def test_solves_and_is_in_global_equilibrium(self):
        s, q = self._mixed_model()
        r = s.calculate()
        reac = r['analysis_cases']['AC']['reactions']
        total_reac_w = sum(v[0] for v in reac.values())
        total_area = 2.0 * 2.0 + 1.0 * 2.0   # 2x2 quad block + 1x2 tri block
        total_load = q * total_area
        assert_close(self, total_reac_w, -total_load, rel=1e-9,
                     msg="mixed tri/quad global equilibrium")

    def test_interface_displacement_is_smooth(self):
        """The shared-edge interior node (touched by both a quad and a
        triangle) must not show a discontinuity relative to its all-quad and
        all-tri interior neighbours — a crude but real compatibility check:
        adjacent interior nodes' deflections stay within the same order of
        magnitude and sign, rather than jumping (which a DOF mismatch or a
        mis-wired dispatch would produce)."""
        s, _ = self._mixed_model()
        r = s.calculate()
        disp = r['analysis_cases']['AC']['displacements']
        w_quad_interior = disp['N1_1'][0]      # touched only by quads
        w_interface = disp['N2_1'][0]          # touched by a quad AND a tri
        self.assertLess(w_quad_interior, 0.0)   # deflects downward under -pz
        self.assertLess(w_interface, 0.0)
        ratio = w_interface / w_quad_interior
        self.assertGreater(ratio, 0.5)
        self.assertLess(ratio, 2.0)


class TestQuadSelfWeight(unittest.TestCase):
    def test_reaction_balances_weight(self):
        s = Structure2D(domain='plane')
        s.add_material('M', elastic_modulus=30.0e9, unit_weight=25.0,
                       poisson=0.2)
        s.add_quad_section('QS', 'M', thickness=0.2, formulation='QM6')
        s.add_node('N1', 0.0, 0.0); s.add_node('N2', 1.0, 0.0)
        s.add_node('N3', 1.0, 1.0); s.add_node('N4', 0.0, 1.0)
        s.add_quad_element('Q1', 'N1', 'N2', 'N3', 'N4', 'QS')
        s.add_support('FIX', ux=True, uy=True, tz=True)
        s.assign_support('N1', 'FIX')
        s.assign_support('N2', 'FIX')
        s.add_load_case('SW', self_weight_factor=1.0)
        s.add_analysis_case('AC', 'Linear', {'SW': 1.0})
        r = s.calculate()
        reac = r['analysis_cases']['AC']['reactions']
        total_uy = sum(v[1] for v in reac.values())
        expected = 25.0 * 0.2 * 1.0   # gamma * t * area
        assert_close(self, total_uy, expected, rel=1e-9,
                     msg="quad self-weight reaction")


class TestDomainMismatchRejected(unittest.TestCase):
    def test_plate_formulation_in_plane_model_flagged(self):
        s = Structure2D(domain='plane')
        s.add_material('M', elastic_modulus=30.0e9, unit_weight=0.0,
                       poisson=0.2)
        s.add_quad_section('QSp', 'M', thickness=0.2, formulation='MITC4')
        s.add_node('a', 0, 0); s.add_node('b', 1, 0)
        s.add_node('c', 1, 1); s.add_node('d', 0, 1)
        s.add_quad_element('Qp', 'a', 'b', 'c', 'd', 'QSp')
        probs = s.domain_problems()
        self.assertTrue(any('MITC4' in p for p in probs), probs)
        with self.assertRaises(ValueError):
            s.check_references()


if __name__ == '__main__':
    unittest.main()
