"""Springs on line objects, and node-spring propagation along surface edges.

Two engine features. A foundation spring on a line object expands to its bars,
like a line distributed load — kx/ky per unit length, value or field. And a
node spring on the corners of a surface propagates to the mesh nodes along each
edge, the elastic twin of the edge-support propagation.

Edge-spring propagation follows the surface's ``edge_spring_mode``: ``"linear"``
(default) interpolates kx/ky/kt linearly between the two corners (a missing
corner spring counts as 0); ``"none"`` generates nothing. The global
``propagate_edge_springs`` flag remains a master on/off.
"""
from __future__ import annotations

import unittest

from context import Structure2D  # noqa: F401

from xdfem2d.geo_expand import expand_geometry


def _line(kx=0.0, ky=0.0, kx_field="", ky_field="", fields=None,
          mode_x="both", div=3):
    s = Structure2D()
    s.add_material('C', elastic_modulus=30e6, unit_weight=0.0)
    s.add_section('S', 'C', b=0.3, h=0.5)
    s.add_node('L.p0', 0, 0)
    s.add_node('L.p1', 6, 0)
    s.add_geo_line('L', 0, 0, 6, 0, section_name='S', divisions=div)
    for name, expr in (fields or {}).items():
        s.add_field(name, expr)
    s.add_line_element_spring('L', kx=kx, ky=ky, kx_field=kx_field,
                              ky_field=ky_field, mode_x=mode_x)
    return s


def _surface(spring_a=None, spring_b=None):
    s = Structure2D()
    s.add_material('C', elastic_modulus=33e6, unit_weight=0.0)
    s.add_tri_section('W', 'C', thickness=0.2)
    for nid, (x, y) in {'R.p0': (0, 0), 'R.p1': (4, 0),
                        'R.p2': (4, 3), 'R.p3': (0, 3)}.items():
        s.add_node(nid, x, y)
    s.add_geo_rectangle('R', (0, 0), (4, 3), section_name='W',
                        target_size=1.0)
    if spring_a:
        s.add_node_spring('R.p0', **spring_a)
    if spring_b:
        s.add_node_spring('R.p1', **spring_b)
    return s


class TestLineElementSpring(unittest.TestCase):

    def test_uniform_reaches_every_bar(self):
        mesh, _ = expand_geometry(_line(ky=1000.0))
        self.assertEqual(len(mesh.element_springs), 3)
        self.assertEqual({v.ky for v in mesh.element_springs.values()},
                         {1000.0})

    def test_a_field_follows_the_object(self):
        mesh, _ = expand_geometry(
            _line(ky_field='k', fields={'k': 'x*100'}))
        self.assertEqual(
            sorted(round(v.ky) for v in mesh.element_springs.values()),
            [100, 300, 500])   # midpoints x = 1, 3, 5

    def test_the_modes_carry_through(self):
        mesh, _ = expand_geometry(_line(kx=5.0, mode_x='compression'))
        self.assertEqual({v.mode_x for v in mesh.element_springs.values()},
                         {'compression'})

    def test_it_replaces_rather_than_stacks(self):
        s = _line(ky=1000.0)
        s.add_line_element_spring('L', ky=9.0)
        self.assertEqual(len(s.line_element_springs), 1)
        self.assertEqual(s.line_element_springs[0].ky, 9.0)


class TestSurfaceEdgeSpringPropagation(unittest.TestCase):

    def _base(self, mesh):
        return {nid for nid, n in mesh.nodes.items() if abs(n.y) < 1e-9}

    def test_two_matching_corners_spring_the_whole_edge(self):
        mesh, tr = expand_geometry(
            _surface(spring_a={'ky': 5000.0}, spring_b={'ky': 5000.0}))
        base = self._base(mesh)
        self.assertGreater(len(base), 2)
        self.assertTrue(base <= set(mesh.node_springs))
        self.assertTrue(all(mesh.node_springs[n].ky == 5000.0 for n in base))

    def test_different_corner_springs_interpolate_linearly(self):
        # 5000 at x=0 (R.p0), 9999 at x=4 (R.p1): the base edge nodes take
        # values strictly between the two corner values.
        mesh, _ = expand_geometry(
            _surface(spring_a={'ky': 5000.0}, spring_b={'ky': 9999.0}))
        base = self._base(mesh)
        inner = [mesh.node_springs[n].ky for n in base
                 if n in mesh.node_springs and n not in ('R.p0', 'R.p1')]
        self.assertTrue(inner)
        self.assertTrue(all(5000.0 < k < 9999.0 for k in inner), inner)

    def test_one_sprung_corner_interpolates_to_zero(self):
        # Only R.p0 sprung (5000): along its edges the value decays linearly to
        # 0 at the far corner, so the generated springs are in (0, 5000).
        mesh, _ = expand_geometry(_surface(spring_a={'ky': 5000.0}))
        self.assertGreater(len(mesh.node_springs), 1)
        inner = [sp.ky for nid, sp in mesh.node_springs.items()
                 if nid != 'R.p0']
        self.assertTrue(inner)
        self.assertTrue(all(0.0 < k < 5000.0 for k in inner), inner)

    def test_none_mode_leaves_the_corners_alone(self):
        s = _surface(spring_a={'ky': 5000.0}, spring_b={'ky': 5000.0})
        s.geometry_objects['R'].edge_spring_mode = "none"
        mesh, _ = expand_geometry(s)
        self.assertEqual(len(mesh.node_springs), 2)   # corners only

    def test_the_flag_turns_it_off(self):
        s = _surface(spring_a={'ky': 5000.0}, spring_b={'ky': 5000.0})
        s.propagate_edge_springs = False
        mesh, _ = expand_geometry(s)
        self.assertEqual(len(mesh.node_springs), 2)   # corners only


class TestRoundTrip(unittest.TestCase):

    def test_line_spring_and_flag_survive(self):
        from xdfem2d.structure_io import _from_dict, _to_dict
        s = _line(ky_field='k', fields={'k': 'x*100'}, mode_x='tension')
        s.propagate_edge_springs = False
        back = _from_dict(_to_dict(s))
        ls = back.line_element_springs[0]
        self.assertEqual((ls.ky_field, ls.mode_x), ('k', 'tension'))
        self.assertFalse(back.propagate_edge_springs)


if __name__ == '__main__':
    unittest.main()
