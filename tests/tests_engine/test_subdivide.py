"""Dividing bars and triangles: the geometry, and everything attached to it.

The point of these is not that a bar splits into N bars — that part is
arithmetic. It is that what hangs off the element id comes across intact: a
trapezoidal load re-interpolated rather than duplicated, a point load moved to
the piece that holds it, a moment hinge left only at the true ends, and — for
triangles — a conforming mesh with no node quietly created twice along a shared
edge.

The strongest checks here are invariants a wrong implementation cannot fake:
the reactions before and after a division are identical, because refining a
mesh changes the answer's precision, not its equilibrium.
"""
from __future__ import annotations

import collections
import unittest

from context import Structure2D  # noqa: F401  (puts src/ on the path)

from xdfem2d import subdivide as S


def _beam():
    """A 6 m simply supported beam under 10 kN/m, one element."""
    s = Structure2D()
    s.add_material('C', elastic_modulus=30e6, unit_weight=0.0)
    s.add_section('SEC', 'C', b=0.3, h=0.5)
    s.add_node('N1', 0.0, 0.0)
    s.add_node('N2', 6.0, 0.0)
    s.add_bar_element('E1', 'N1', 'N2', 'SEC')
    s.add_support('PIN', ux=True, uy=True)
    s.add_support('ROL', ux=False, uy=True)
    s.assign_support('N1', 'PIN')
    s.assign_support('N2', 'ROL')
    s.add_load_case('G')
    s.add_distributed_load('E1', 'G', fye=-10.0, fyd=-10.0)
    s.add_analysis_case('ULS', 'Linear', {'G': 1.0})
    return s


def _wall():
    """A 2x2 m square as two triangles sharing the diagonal N1-N3."""
    s = Structure2D()
    s.add_material('C', elastic_modulus=33e6, unit_weight=0.0)
    s.add_tri_section('W', 'C', thickness=0.2)
    for nid, (x, y) in {'N1': (0, 0), 'N2': (2, 0),
                        'N3': (2, 2), 'N4': (0, 2)}.items():
        s.add_node(nid, x, y)
    s.add_tri_element('T1', 'N1', 'N2', 'N3', 'W')
    s.add_tri_element('T2', 'N1', 'N3', 'N4', 'W')
    s.add_support('FIX', ux=True, uy=True)
    s.assign_support('N1', 'FIX')
    s.assign_support('N2', 'FIX')
    s.add_load_case('Q')
    s.add_point_load('N3', 'Q', fx=50.0)
    s.add_point_load('N4', 'Q', fx=50.0)
    s.add_analysis_case('ULS', 'Linear', {'Q': 1.5})
    return s


def _sum_reactions(struc, case, comp):
    r = struc.calculate()['reactions'][case]
    return sum(v[comp] for v in r.values())


class TestBarGeometry(unittest.TestCase):

    def test_by_count(self):
        s = _beam()
        out = S.divide_bars(s, ['E1'], count=4)
        self.assertEqual(len(s.bar_elements), 4)
        self.assertEqual(out['elements'], 4)
        self.assertEqual(out['new_nodes'], 3)

    def test_by_size(self):
        s = _beam()
        S.divide_bars(s, ['E1'], size=1.5)          # 6 / 1.5 = 4
        self.assertEqual(len(s.bar_elements), 4)

    def test_size_rounds_to_the_nearest_whole_number(self):
        s = _beam()
        S.divide_bars(s, ['E1'], size=2.5)          # 6 / 2.5 = 2.4 -> 2
        self.assertEqual(len(s.bar_elements), 2)

    def test_a_division_into_one_is_a_no_op(self):
        s = _beam()
        S.divide_bars(s, ['E1'], count=1)
        self.assertEqual([e.id for e in s.bar_elements], ['E1'])

    def test_the_new_nodes_lie_on_the_bar(self):
        s = _beam()
        S.divide_bars(s, ['E1'], count=3)
        xs = sorted(n.x for n in s.nodes.values())
        self.assertEqual(xs, [0.0, 2.0, 4.0, 6.0])
        self.assertTrue(all(abs(n.y) < 1e-12 for n in s.nodes.values()))

    def test_the_preview_matches_what_is_built(self):
        for kw in (dict(count=5), dict(size=0.7)):
            with self.subTest(**kw):
                a, b = _beam(), _beam()
                pv = S.preview_bars(a, ['E1'], **kw)
                S.divide_bars(b, ['E1'], **kw)
                self.assertEqual(pv['elements'], len(b.bar_elements))
                self.assertEqual(pv['new_nodes'],
                                 len(b.nodes) - len(a.nodes))


class TestBarEquilibriumIsUnchanged(unittest.TestCase):
    """A finer discretisation of the same beam carries the same load to the
    same supports."""

    def test_uniform_load(self):
        before = _sum_reactions(_beam(), 'G', 1)
        s = _beam()
        S.divide_bars(s, ['E1'], count=7)
        self.assertAlmostEqual(_sum_reactions(s, 'G', 1), before, places=6)
        self.assertAlmostEqual(before, 60.0, places=6)   # 10 kN/m x 6 m

    def test_triangular_load_is_split_not_duplicated(self):
        """0 -> -12 over 6 m: total 36 kN. Duplicating the trapezoid onto each
        sub-bar would multiply the load; interpolating keeps the total."""
        def s_with_triangle():
            s = _beam()
            s.distributed_loads.clear()
            s.add_distributed_load('E1', 'G', fye=0.0, fyd=-12.0)
            return s
        before = _sum_reactions(s_with_triangle(), 'G', 1)
        s = s_with_triangle()
        S.divide_bars(s, ['E1'], count=3)
        self.assertAlmostEqual(_sum_reactions(s, 'G', 1), before, places=6)
        self.assertAlmostEqual(before, 36.0, places=6)


class TestBarAttachmentsCrossOver(unittest.TestCase):

    def test_hinges_stay_at_the_true_ends_only(self):
        """The trap: copying the hinges to every piece turns a continuous
        member into a mechanism."""
        s = _beam()
        s.bar_elements_by_id['E1'].hinge_i = True
        s.bar_elements_by_id['E1'].hinge_j = True
        S.divide_bars(s, ['E1'], count=3)
        bars = sorted(s.bar_elements, key=lambda e: e.id)
        self.assertEqual([(e.hinge_i, e.hinge_j) for e in bars],
                         [(True, False), (False, False), (False, True)])

    def test_buckling_overrides_and_column_flag_copy_to_every_piece(self):
        """dev/BUCKLING_COLUMN_PERSISTENCE.md fase 2: dividing a bar marked
        as a column, with K overrides set, must not silently drop them on
        the resulting sub-bars."""
        s = _beam()
        e1 = s.bar_elements_by_id['E1']
        e1.sd_ky = 2.0; e1.sd_kz = 1.5; e1.sd_klt = 1.1; e1.sd_ltb = False
        e1.is_column = True
        S.divide_bars(s, ['E1'], count=3)
        for e in s.bar_elements:
            self.assertEqual(e.sd_ky, 2.0)
            self.assertEqual(e.sd_kz, 1.5)
            self.assertEqual(e.sd_klt, 1.1)
            self.assertFalse(e.sd_ltb)
            self.assertTrue(e.is_column)

    def test_a_point_load_moves_to_the_piece_that_holds_it(self):
        s = _beam()
        s.add_element_point_load('E1', 'G', a=4.5, fy=-20.0)
        S.divide_bars(s, ['E1'], count=3)            # pieces 0-2, 2-4, 4-6
        self.assertEqual(len(s.element_point_loads), 1)
        p = s.element_point_loads[0]
        self.assertEqual(p.element_id, 'E1/3')
        self.assertAlmostEqual(p.a, 0.5, places=6)

    def test_a_point_load_keeps_the_structure_in_equilibrium(self):
        def s_with_point():
            s = _beam()
            s.distributed_loads.clear()
            s.add_element_point_load('E1', 'G', a=2.0, fy=-15.0)
            return s
        before = _sum_reactions(s_with_point(), 'G', 1)
        s = s_with_point()
        S.divide_bars(s, ['E1'], count=4)
        self.assertAlmostEqual(_sum_reactions(s, 'G', 1), before, places=6)
        self.assertAlmostEqual(before, 15.0, places=6)

    def test_a_spring_copies_to_every_piece(self):
        s = _beam()
        s.add_element_spring('E1', ky=1000.0)
        S.divide_bars(s, ['E1'], count=3)
        self.assertEqual(len(s.element_springs), 3)
        self.assertTrue(all(v.ky == 1000.0 for v in s.element_springs.values()))

    def test_a_thermal_load_copies_to_every_piece(self):
        s = _beam()
        s.add_temperature_load('E1', 'G', delta_t_uniform=20.0)
        S.divide_bars(s, ['E1'], count=3)
        self.assertEqual(len(s.temperature_loads), 3)


class TestTriangleGeometry(unittest.TestCase):

    def test_each_triangle_becomes_n_squared(self):
        for n in (2, 3, 4, 8, 12):
            with self.subTest(n=n):
                s = _wall()
                out = S.divide_triangles(s, ['T1', 'T2'], n)
                self.assertEqual(out['elements'], 2 * n * n)
                self.assertEqual(len(s.tri_elements), 2 * n * n)

    def test_below_two_is_a_no_op(self):
        s = _wall()
        S.divide_triangles(s, ['T1', 'T2'], 1)
        self.assertEqual(len(s.tri_elements), 2)

    def test_the_preview_matches_what_is_built(self):
        for n in (2, 3, 4, 8, 12):
            with self.subTest(n=n):
                a, b = _wall(), _wall()
                pv = S.preview_triangles(a, ['T1', 'T2'], n)
                S.divide_triangles(b, ['T1', 'T2'], n)
                self.assertEqual(pv['elements'], len(b.tri_elements))
                self.assertEqual(pv['new_nodes'],
                                 len(b.nodes) - len(a.nodes))


class TestTriangleMeshIsConforming(unittest.TestCase):
    """The one thing that can go silently wrong: a node made twice along the
    shared diagonal, giving a mesh that looks continuous and has a crack."""

    def test_no_two_nodes_share_a_coordinate(self):
        for n in (2, 3, 4, 8, 12):
            with self.subTest(n=n):
                s = _wall()
                S.divide_triangles(s, ['T1', 'T2'], n)
                coords = collections.Counter(
                    (round(nd.x, 9), round(nd.y, 9))
                    for nd in s.nodes.values())
                self.assertEqual([c for c, k in coords.items() if k > 1], [])

    def test_the_shared_diagonal_is_split_once_for_both(self):
        """Both triangles share edge N1-N3. Divided, its interior points must
        be the same nodes for both — so the node count is the deduped one, not
        twice the per-triangle figure."""
        s = _wall()
        S.divide_triangles(s, ['T1', 'T2'], 4)
        # 25 for a 4x4 lattice over the square; 2 * 15 - 5(shared diagonal) = 25
        self.assertEqual(len(s.nodes), 25)

    def test_equilibrium_survives_the_refinement(self):
        before = _sum_reactions(_wall(), 'Q', 0)
        s = _wall()
        S.divide_triangles(s, ['T1', 'T2'], 4)
        self.assertAlmostEqual(_sum_reactions(s, 'Q', 0), before, places=6)
        self.assertAlmostEqual(before, -100.0, places=6)


class TestTriangleEdgeLoads(unittest.TestCase):
    """A uniform edge load on a triangle is spread over the sub-segments of
    that edge, and its total is unchanged."""

    def _loaded(self):
        s = Structure2D()
        s.add_material('C', elastic_modulus=33e6, unit_weight=0.0)
        s.add_tri_section('W', 'C', thickness=0.2)
        s.add_node('N1', 0, 0)
        s.add_node('N2', 4, 0)
        s.add_node('N3', 0, 3)
        s.add_tri_element('T1', 'N1', 'N2', 'N3', 'W')
        s.add_support('FIX', ux=True, uy=True)
        for nid in ('N1', 'N2', 'N3'):
            s.assign_support(nid, 'FIX')
        s.add_load_case('Q')
        s.add_tri_edge_load('EL1', 'T1', 'N1', 'N2', 'Q', fy=-10.0)
        s.add_analysis_case('ULS', 'Linear', {'Q': 1.5})
        return s

    def test_it_becomes_one_load_per_sub_segment(self):
        s = self._loaded()
        S.divide_triangles(s, ['T1'], 4)
        self.assertEqual(len(s.tri_edge_loads), 4)

    def test_every_sub_load_points_at_real_ids(self):
        s = self._loaded()
        S.divide_triangles(s, ['T1'], 4)
        for e in s.tri_edge_loads:
            self.assertIn(e.tri_id, s.tri_elements_by_id)
            self.assertIn(e.node_a, s.nodes)
            self.assertIn(e.node_b, s.nodes)

    def test_the_total_edge_load_is_preserved(self):
        before = _sum_reactions(self._loaded(), 'Q', 1)   # 10 kN/m x 4 m
        s = self._loaded()
        S.divide_triangles(s, ['T1'], 4)
        self.assertAlmostEqual(_sum_reactions(s, 'Q', 1), before, places=6)
        self.assertAlmostEqual(before, 40.0, places=6)


class TestSelectionFiltering(unittest.TestCase):

    def test_only_bars_among_the_ids_are_divided(self):
        s = _wall()
        s.add_material('C2', elastic_modulus=30e6, unit_weight=0.0) \
            if 'C2' not in s.materials else None
        s.add_section('SEC', 'C', b=0.3, h=0.5)
        s.add_bar_element('B1', 'N1', 'N2', 'SEC')
        out = S.divide_bars(s, ['B1', 'T1'], count=2)   # T1 is a triangle
        self.assertEqual(out['bars'], 1)

    def test_dividing_an_empty_selection_does_nothing(self):
        s = _beam()
        out = S.divide_bars(s, [], count=4)
        self.assertEqual(out['elements'], 0)
        self.assertEqual(len(s.bar_elements), 1)


def _quad_slab():
    """A 2x2 m plate meshed as a single MITC4 quad, under a pressure and a
    thermal gradient — enough to check the loads come across a division."""
    s = Structure2D(domain='plate')
    s.add_material('C', elastic_modulus=33e6, unit_weight=0.0, poisson=0.2)
    s.add_quad_section('W', 'C', thickness=0.2, formulation='MITC4')
    for nid, (x, y) in {'N1': (0, 0), 'N2': (2, 0),
                        'N3': (2, 2), 'N4': (0, 2)}.items():
        s.add_node(nid, x, y)
    s.add_quad_element('Q1', 'N1', 'N2', 'N3', 'N4', 'W')
    s.add_load_case('Q')
    s.add_area_load('Q1', 'Q', pz=-5.0)
    s.create_temperature('Q1', uniform=10.0, gradient=3.0, load_case='Q')
    return s


class TestQuadDivision(unittest.TestCase):

    def test_one_quad_becomes_n_squared_sub_quads(self):
        for n in (2, 3, 4):
            with self.subTest(n=n):
                s = _quad_slab()
                out = S.divide_quads(s, ['Q1'], n)
                self.assertEqual(out['quads'], 1)
                self.assertEqual(out['elements'], n * n)
                self.assertEqual(len(s.quad_elements), n * n)

    def test_area_is_conserved(self):
        s = _quad_slab()
        S.divide_quads(s, ['Q1'], 3)

        def qa(q):
            ns = [s.nodes[n] for n in (q.node_i, q.node_j, q.node_k, q.node_l)]
            a = 0.0
            for i in range(4):
                b = (i + 1) % 4
                a += ns[i].x * ns[b].y - ns[b].x * ns[i].y
            return abs(a) / 2.0
        self.assertAlmostEqual(sum(qa(q) for q in s.quad_elements), 4.0,
                               places=6)

    def test_mesh_is_conforming_no_duplicate_edge_nodes(self):
        s = _quad_slab()
        S.divide_quads(s, ['Q1'], 4)
        # (n+1)^2 lattice over the one square: 25 nodes, none duplicated.
        self.assertEqual(len(s.nodes), 25)

    def test_pressure_and_temperature_carry_to_every_sub_quad(self):
        s = _quad_slab()
        S.divide_quads(s, ['Q1'], 2)
        self.assertEqual(len(s.quad_area_loads), 4)
        self.assertTrue(all(a.pz == -5.0 for a in s.quad_area_loads))
        self.assertEqual(len(s.quad_temperature_loads), 4)
        self.assertTrue(all(t.dt_gradient == 3.0
                            for t in s.quad_temperature_loads))
        # A uniform parent field samples back to the same mean everywhere.
        self.assertTrue(all(abs(t.dt_mean - 10.0) < 1e-9
                            for t in s.quad_temperature_loads))

    def test_only_quads_among_the_ids_are_divided(self):
        s = _quad_slab()
        out = S.divide_quads(s, ['Q1', 'nope'], 2)
        self.assertEqual(out['quads'], 1)

    def test_round_trips_through_save(self):
        from xdfem2d.structure_io import _from_dict, _to_dict
        s = _quad_slab()
        S.divide_quads(s, ['Q1'], 3)
        back = _from_dict(_to_dict(s))
        self.assertEqual(len(back.quad_elements), 9)


class TestItSurvivesASaveAndReload(unittest.TestCase):
    """The division edits the input model, so it has to round-trip — a refined
    model that comes back coarse would be a division that never happened."""

    def test_bars(self):
        from xdfem2d.structure_io import _from_dict, _to_dict
        s = _beam()
        S.divide_bars(s, ['E1'], count=4)
        back = _from_dict(_to_dict(s))
        self.assertEqual(len(back.bar_elements), 4)

    def test_triangles(self):
        from xdfem2d.structure_io import _from_dict, _to_dict
        s = _wall()
        S.divide_triangles(s, ['T1', 'T2'], 3)
        back = _from_dict(_to_dict(s))
        self.assertEqual(len(back.tri_elements), 18)


if __name__ == '__main__':
    unittest.main()
