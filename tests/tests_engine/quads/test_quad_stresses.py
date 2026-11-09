"""Quad stress/moment recovery + dispatch (dev/IMPLEMENT_QUAD.md Phase 5).

Phase 5's actual contract, per the plan document: extend the existing merge
point (``tri_elements.tri_stresses_dispatch``) so it also returns quad
stresses/moments, still under ``results['tri_stress']`` — and prove the
dispatch returns one dict keyed by element id across tri *and* quad elements,
with no id collisions.

These tests do not re-derive new closed-form benchmarks (Phases 1/2 already
did that at the kernel level, with `q4_stress_at_centroid`/
`qm6_stress_at_centroid`/`dkt4_moment_entry`/`mitc4_moment_entry` tested
directly against constant-strain/curvature patch tests). What Phase 5 adds is
*wiring*: correct (E, nu, t, coords, local displacement) extraction from a
solved ``Structure2D`` model and correct formulation dispatch. So the checks
here are the same shape as Phase 4's own validation style: run the full
``Structure2D.calculate()`` pipeline, then confirm the dispatched result
matches the already-tested kernel function called directly with the same
inputs.
"""
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)
from context import assert_close

from xdfem2d.quad_elements import qm6_stress_at_centroid
from xdfem2d.quad_elements_dkt4 import dkt4_moment_entry
from xdfem2d.quad_elements_mitc4 import mitc4_moment_entry

E = 30.0e9
NU = 0.2


class TestQM6StressDispatchWiring(unittest.TestCase):
    """Same QM6 cantilever as test_quad_assembly.py's
    TestQM6CantileverFullPipeline, now checking the recovered stress (not
    just the deflection) matches the Phase 1 kernel called directly with the
    solved displacements."""

    L, H, T = 2.0, 0.2, 0.1
    M_TIP = 1000.0
    N = 4

    def _solve(self):
        s = Structure2D(domain='plane')
        s.add_material('M', elastic_modulus=E, unit_weight=0.0, poisson=NU)
        s.add_quad_section('QS', 'M', thickness=self.T, formulation='QM6')
        for i in range(self.N + 1):
            s.add_node(f'B{i}', i * self.L / self.N, 0.0)
            s.add_node(f'T{i}', i * self.L / self.N, self.H)
        for i in range(self.N):
            s.add_quad_element(f'Q{i}', f'B{i}', f'B{i+1}', f'T{i+1}', f'T{i}',
                               'QS')
        s.add_support('FIX', ux=True, uy=True, tz=True)
        s.assign_support('B0', 'FIX')
        s.assign_support('T0', 'FIX')
        s.add_load_case('LC')
        F = self.M_TIP / self.H
        s.add_point_load(f'T{self.N}', 'LC', fx=F)
        s.add_point_load(f'B{self.N}', 'LC', fx=-F)
        s.add_analysis_case('AC', 'Linear', {'LC': 1.0})
        return s, s.calculate()

    def test_every_quad_has_a_dispatch_entry(self):
        s, r = self._solve()
        ts = r['analysis_cases']['AC']['tri_stress']
        for qid in s.quad_elements_by_id:
            self.assertIn(qid, ts, f"quad '{qid}' missing from tri_stress")
        self.assertEqual(len(ts), self.N)

    def test_matches_kernel_called_directly(self):
        """Cross-check one interior quad: extract its local displacements
        from the solved analysis case and call qm6_stress_at_centroid
        directly — must match the dispatched entry exactly."""
        s, r = self._solve()
        disp = r['analysis_cases']['AC']['displacements']
        ts = r['analysis_cases']['AC']['tri_stress']

        e = 2   # an interior element, away from the fixed-end boundary layer
        quad = s.quad_elements_by_id[f'Q{e}']
        coords = [(s.nodes[n].x, s.nodes[n].y)
                  for n in (quad.node_i, quad.node_j, quad.node_k, quad.node_l)]
        ue = []
        for n in (quad.node_i, quad.node_j, quad.node_k, quad.node_l):
            ue += list(disp[n][:2])
        expected = qm6_stress_at_centroid(coords, E, NU, self.T, ue)

        got = ts[f'Q{e}']
        for key in ('sx', 'sy', 'txy', 's1', 's2', 'vm'):
            assert_close(self, got[key], expected[key], rel=1e-9, abs_tol=1e-6,
                         msg=f"dispatch vs kernel {key}")
        self.assertEqual(got['formulation'], 'QM6')

    def test_load_case_level_result_also_populated(self):
        """The other dispatch path — results['tri_stress'][load_case_id],
        built from the raw solved displacement vector rather than a
        displacement dict — must also carry every quad, consistent with the
        analysis-case-level values (a single Linear case at factor 1.0)."""
        s, r = self._solve()
        ts_lc = r['tri_stress']['LC']
        ts_ac = r['analysis_cases']['AC']['tri_stress']
        self.assertEqual(set(ts_lc), set(ts_ac))
        for qid in ts_lc:
            assert_close(self, ts_lc[qid]['sx'], ts_ac[qid]['sx'], rel=1e-9,
                         abs_tol=1e-6, msg=f"LC vs AC sx for {qid}")

    def test_pure_bending_sanity(self):
        """One element through the depth: each quad's own centroid sits
        exactly on the neutral axis, so the reported centroid sx and txy
        (pure bending, no transverse shear) must both be numerically zero."""
        _s, r = self._solve()
        ts = r['analysis_cases']['AC']['tri_stress']
        scale = self.M_TIP / (self.T * self.H ** 2)   # a representative stress scale
        for qid, rec in ts.items():
            self.assertLess(abs(rec['sx']), 1e-6 * scale, f"{qid} sx")
            self.assertLess(abs(rec['txy']), 1e-6 * scale, f"{qid} txy")


class TestQuadPlateMomentDispatchWiring(unittest.TestCase):
    """Same simply-supported square plate as test_quad_assembly.py's
    TestMITC4PlateFullPipeline, run for both MITC4 and DKT4 sections,
    checking the recovered moment matches the Phase 2 kernel called directly
    with the solved displacements."""

    A, T, NU, Q, N = 4.0, 0.1, 0.3, 10_000.0, 4

    def _solve(self, formulation):
        s = Structure2D(domain='plate')
        s.add_material('M', elastic_modulus=E, unit_weight=0.0, poisson=self.NU)
        s.add_quad_section('QS', 'M', thickness=self.T, formulation=formulation)
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
        return s, s.calculate()

    def test_mitc4_every_quad_has_entry_and_matches_kernel(self):
        self._check('MITC4', mitc4_moment_entry)

    def test_dkt4_every_quad_has_entry_and_matches_kernel(self):
        self._check('DKT4', dkt4_moment_entry)

    def _check(self, formulation, entry_fn):
        s, r = self._solve(formulation)
        disp = r['analysis_cases']['AC']['displacements']
        ts = r['analysis_cases']['AC']['tri_stress']
        for qid in s.quad_elements_by_id:
            self.assertIn(qid, ts, f"quad '{qid}' missing from tri_stress")
        self.assertEqual(len(ts), self.N * self.N)

        qid = f'Q{self.N // 2}_{self.N // 2}'   # the centre element
        quad = s.quad_elements_by_id[qid]
        coords = [(s.nodes[n].x, s.nodes[n].y)
                  for n in (quad.node_i, quad.node_j, quad.node_k, quad.node_l)]
        u12 = []
        for n in (quad.node_i, quad.node_j, quad.node_k, quad.node_l):
            u12 += list(disp[n])
        expected = entry_fn(coords, E, self.NU, self.T, u12)

        got = ts[qid]
        for key in ('mx', 'my', 'mxy'):
            assert_close(self, got[key], expected[key], rel=1e-9, abs_tol=1e-3,
                         msg=f"{formulation} dispatch vs kernel {key}")
        self.assertEqual(got['formulation'], formulation)
        # Sagging under a downward UDL on a simply-supported slab: positive
        # centre moment, the same sign convention DKT/MITC3 already use.
        self.assertGreater(got['mx'], 0.0)
        self.assertGreater(got['my'], 0.0)


class TestMixedTriQuadDispatchHasNoCollisions(unittest.TestCase):
    """A plate meshed with MITC4 quads on one side and DKT triangles on the
    other (same layout as test_quad_assembly.py's TestMixedTriQuadPlate) —
    the dispatched tri_stress dict must carry every tri *and* every quad
    element id, with the right count (proves the two .update() merges in
    tri_stresses_dispatch don't drop or overwrite entries)."""

    def _mixed_model(self):
        s = Structure2D(domain='plate')
        s.add_material('M', elastic_modulus=30.0e9, unit_weight=0.0, poisson=0.2)
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
        for eid in list(s.quad_elements_by_id) + list(s.tri_elements_by_id):
            s.add_area_load(eid, 'LC', pz=-5000.0)
        s.add_analysis_case('AC', 'Linear', {'LC': 1.0})
        return s

    def test_dispatch_covers_every_element_exactly_once(self):
        s = self._mixed_model()
        r = s.calculate()
        ts = r['analysis_cases']['AC']['tri_stress']
        all_ids = set(s.quad_elements_by_id) | set(s.tri_elements_by_id)
        self.assertEqual(set(ts), all_ids)
        self.assertEqual(len(ts), len(s.quad_elements) + len(s.tri_elements))
        for tid in s.tri_elements_by_id:
            self.assertEqual(ts[tid]['formulation'], 'DKT')
        for qid in s.quad_elements_by_id:
            self.assertEqual(ts[qid]['formulation'], 'MITC4')


class TestCrossKindIdCollisionRejected(unittest.TestCase):
    """dev/IMPLEMENT_QUAD.md Phase 5 finding: a triangle and a quad sharing
    an id would silently overwrite each other in the merged tri_stress dict
    (both `.update()` into the same dict, keyed by element id) — this is a
    pre-existing gap that already applied to bar-vs-triangle ids before quads
    existed (still unfixed, out of Phase 5's scope) but Phase 5's own merge
    makes a tri/quad collision consequential, so it is rejected at
    creation time, in both directions."""

    def _base(self):
        s = Structure2D(domain='plane')
        s.add_material('M', elastic_modulus=30.0e9, unit_weight=0.0, poisson=0.2)
        s.add_tri_section('TS', 'M', thickness=0.2, formulation='CST')
        s.add_quad_section('QS', 'M', thickness=0.2, formulation='QM6')
        for nid, (x, y) in {'n1': (0, 0), 'n2': (1, 0), 'n3': (1, 1),
                            'n4': (0, 1)}.items():
            s.add_node(nid, x, y)
        return s

    def test_quad_id_colliding_with_existing_tri_rejected(self):
        s = self._base()
        s.add_tri_element('E1', 'n1', 'n2', 'n3', 'TS')
        with self.assertRaises(ValueError):
            s.add_quad_element('E1', 'n1', 'n2', 'n3', 'n4', 'QS')

    def test_tri_id_colliding_with_existing_quad_rejected(self):
        s = self._base()
        s.add_quad_element('E1', 'n1', 'n2', 'n3', 'n4', 'QS')
        with self.assertRaises(ValueError):
            s.add_tri_element('E1', 'n1', 'n2', 'n3', 'TS')

    def test_distinct_ids_still_fine(self):
        s = self._base()
        s.add_tri_element('T1', 'n1', 'n2', 'n3', 'TS')
        s.add_quad_element('Q1', 'n1', 'n2', 'n3', 'n4', 'QS')
        self.assertIn('T1', s.tri_elements_by_id)
        self.assertIn('Q1', s.quad_elements_by_id)


if __name__ == '__main__':
    unittest.main()
