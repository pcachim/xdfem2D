"""Supports along the edge of a meshed surface.

A surface is defined by its corner nodes, and until it is meshed those are the
only nodes that exist. So "fixed along the base" could not be written at all —
the nodes along the base are created by the mesher, long after supports have
been assigned. Asked for exactly that, a language model supported the two base
corners, and I read it as an engineering mistake. It was not: it was the whole
of what the format allowed.

Propagation follows the surface's ``edge_support_mode``, comparing the corners
by their restrained DOFs (not by support name):
  * ``"common"`` (default) — the DOFs restrained at BOTH corners (their
    intersection), named by the canonical supports table (FIXED/PIN/…), reusing
    an existing equal support. An empty intersection restrains nothing.
  * ``"equal"`` — only when both corners restrain the same DOF set.
  * ``"none"`` — nothing.
The global ``propagate_edge_supports`` flag remains a master on/off.
"""
from __future__ import annotations

import unittest

from context import Structure2D  # noqa: F401  (puts src/ on the path)

from xdfem2d.geo_expand import expand_geometry


def _wall(support_names=('FIX', 'FIX'), target=0.5):
    """A 4x3 m wall as a region, with the two base corners supported."""
    s = Structure2D()
    s.add_material('C30/37', elastic_modulus=33e6, unit_weight=25.0)
    s.add_tri_section('W1', 'C30/37', thickness=0.2)
    s.add_geo_rectangle('R1', (0.0, 0.0), (4.0, 3.0),
                        section_name='W1', target_size=target)
    for name in set(support_names):
        s.add_support(name, ux=True, uy=True, tz=False)
    s.assign_support('R1.p0', support_names[0])
    s.assign_support('R1.p1', support_names[1])
    s.add_load_case('Q')
    s.add_point_load('R1.p3', 'Q', fx=100.0)
    s.add_analysis_case('ULS', 'Linear', {'Q': 1.5})
    return s


def _base_nodes(mesh):
    return {nid for nid, n in mesh.nodes.items() if abs(n.y) < 1e-9}


def _supported(mesh):
    return {a.node_id for a in mesh.support_assignments}


class TestTheEdgeIsRestrained(unittest.TestCase):

    def setUp(self):
        self.mesh, self.trace = expand_geometry(_wall())

    def test_the_mesh_has_nodes_between_the_corners(self):
        """Otherwise the test proves nothing."""
        self.assertGreater(len(_base_nodes(self.mesh)), 2)

    def test_every_node_on_that_edge_is_supported(self):
        self.assertTrue(_base_nodes(self.mesh) <= _supported(self.mesh))

    def test_the_other_edges_are_not(self):
        top = {nid for nid, n in self.mesh.nodes.items()
               if abs(n.y - 3.0) < 1e-9}
        self.assertFalse(top & _supported(self.mesh))

    def test_the_trace_says_what_it_added(self):
        added = self.trace['R1'].get('edge_supports')
        self.assertTrue(added)
        self.assertTrue(all(name == 'FIX' for _, name in added))

    def test_the_corners_are_not_supported_twice(self):
        counts = {}
        for a in self.mesh.support_assignments:
            counts[a.node_id] = counts.get(a.node_id, 0) + 1
        self.assertEqual(max(counts.values()), 1)


class TestItChangesTheAnswer(unittest.TestCase):
    """The point of the whole thing: a wall held along its base is stiffer
    than one held at two points, and until now only the second could be
    written."""

    def test_the_wall_is_stiffer_than_on_two_corners(self):
        held = expand_geometry(_wall())[0]
        loose_struc = _wall()
        loose_struc.propagate_edge_supports = False
        loose = expand_geometry(loose_struc)[0]

        def drift(mesh):
            r = mesh.calculate()
            return max(abs(r['displacements']['Q'][nid][0])
                       for nid, n in mesh.nodes.items()
                       if abs(n.y - 3.0) < 1e-9)

        self.assertLess(drift(held), drift(loose))

    def test_equilibrium_still_holds(self):
        mesh = expand_geometry(_wall())[0]
        r = mesh.calculate()
        rx = sum(v[0] for v in r['reactions']['Q'].values())
        self.assertAlmostEqual(rx, -100.0, places=6)


class TestPropagationModes(unittest.TestCase):

    def test_different_supports_propagate_their_common_dofs(self):
        """FIX (ux,uy) at one end, ROLLER (uy) at the other: the base edge gets
        the DOF they share (uy), reusing the existing ROLLER support."""
        s = _wall()
        s.add_support('ROLLER', ux=False, uy=True, tz=False)
        s.support_assignments[-1].support_name = 'ROLLER'
        mesh = expand_geometry(s)[0]
        self.assertTrue(_base_nodes(mesh) <= _supported(mesh))
        base_names = {a.support_name for a in mesh.support_assignments
                      if a.node_id in _base_nodes(mesh)
                      and a.node_id not in ('R1.p0', 'R1.p1')}
        self.assertEqual(base_names, {'ROLLER'})   # the common DOF is uy

    def test_restraints_are_compared_not_names(self):
        """Two supports with identical restraints under different names DO
        propagate now — the comparison is by DOF, not by name."""
        s = _wall(support_names=('A', 'B'))
        mesh = expand_geometry(s)[0]
        self.assertTrue(_base_nodes(mesh) <= _supported(mesh))

    def test_exclusive_dofs_propagate_nothing(self):
        """ux-only at one corner, uy-only at the other: empty intersection, so
        the common mode restrains nothing."""
        s = _wall()
        s.add_support('RX', ux=True, uy=False, tz=False)
        s.add_support('RY', ux=False, uy=True, tz=False)
        s.support_assignments[-2].support_name = 'RX'
        s.support_assignments[-1].support_name = 'RY'
        self.assertEqual(len(_supported(expand_geometry(s)[0])), 2)

    def test_equal_mode_requires_identical_dofs(self):
        # FIX (ux,uy) + ROLLER (uy): not identical → equal mode does nothing.
        s = _wall()
        s.add_support('ROLLER', ux=False, uy=True, tz=False)
        s.support_assignments[-1].support_name = 'ROLLER'
        s.geometry_objects['R1'].edge_support_mode = 'equal'
        self.assertEqual(len(_supported(expand_geometry(s)[0])), 2)

    def test_equal_mode_propagates_when_identical(self):
        s = _wall(support_names=('A', 'B'))   # both (ux,uy)
        s.geometry_objects['R1'].edge_support_mode = 'equal'
        mesh = expand_geometry(s)[0]
        self.assertTrue(_base_nodes(mesh) <= _supported(mesh))

    def test_none_mode_generates_nothing(self):
        s = _wall()
        s.geometry_objects['R1'].edge_support_mode = 'none'
        self.assertEqual(len(_supported(expand_geometry(s)[0])), 2)

    def test_one_supported_corner_propagates_nothing(self):
        s = _wall()
        s.support_assignments.pop()
        self.assertEqual(len(_supported(expand_geometry(s)[0])), 1)

    def test_it_can_be_turned_off(self):
        s = _wall()
        s.propagate_edge_supports = False
        self.assertEqual(len(_supported(expand_geometry(s)[0])), 2)


class TestTheFlagTravelsWithTheModel(unittest.TestCase):
    """In the file, not in the application's preferences: the same model must
    give the same answer wherever it is opened."""

    def test_it_defaults_to_on(self):
        self.assertTrue(Structure2D().propagate_edge_supports)

    def test_it_survives_a_round_trip(self):
        from xdfem2d.structure_io import _from_dict, _to_dict
        s = _wall()
        s.propagate_edge_supports = False
        back = _from_dict(_to_dict(s))
        self.assertFalse(back.propagate_edge_supports)

    def test_a_file_written_before_this_existed_defaults_to_on(self):
        from xdfem2d.structure_io import _from_dict, _to_dict
        d = _to_dict(_wall())
        del d['propagate_edge_supports']
        self.assertTrue(_from_dict(d).propagate_edge_supports)


class TestRemovalFacade(unittest.TestCase):
    """delete_*/free_edge — the intent-level inverses of create_*/pin/support.
    Each undoes the corresponding authoring call without the caller reaching
    for the low-level remove_*."""

    def _model(self):
        s = Structure2D()
        s.add_material('C30/37', elastic_modulus=33e6, unit_weight=25.0)
        s.add_tri_section('W1', 'C30/37', thickness=0.2)
        s.add_geo_rectangle('R1', (0.0, 0.0), (4.0, 3.0), section_name='W1')
        s.add_section('BS', 'C30/37', b=0.3, h=0.5)
        s.add_node('N1', 0.0, 0.0)
        s.add_node('N2', 4.0, 0.0)
        s.add_bar_element('B1', 'N1', 'N2', 'BS')
        s.add_load_case('Q')
        return s

    def test_free_edge_undoes_support_edge(self):
        s = self._model()
        s.support_edge('R1', 'right', 'pin')       # writes edge_supports[1]
        self.assertNotEqual(s.geometry_objects['R1'].edge_supports[1], 'free')
        s.free_edge('R1', 'right')
        self.assertEqual(s.geometry_objects['R1'].edge_supports[1], 'free')

    def test_support_edge_is_the_general_case_of_pin_edge(self):
        a, b = self._model(), self._model()
        a.support_edge('R1', 'right', 'pin')
        b.pin_edge('R1', 'right')
        self.assertEqual(a.geometry_objects['R1'].edge_supports,
                         b.geometry_objects['R1'].edge_supports)

    def test_free_edge_clears_the_per_edge_list(self):
        s = self._model()
        s.add_support('SIMPLE', ux=True, uy=True)
        s.support_object_edge('R1', 'right', 'SIMPLE')
        self.assertEqual(s.geometry_objects['R1'].edge_supports[1], 'SIMPLE')
        s.free_edge('R1', 'right')
        self.assertEqual(s.geometry_objects['R1'].edge_supports[1], 'free')

    def test_delete_support_unassigns_a_node(self):
        s = self._model()
        s.pin('N1')
        self.assertTrue(any(a.node_id == 'N1' for a in s.support_assignments))
        s.delete_support('N1')
        self.assertFalse(any(a.node_id == 'N1' for a in s.support_assignments))

    def test_delete_load(self):
        s = self._model()
        s.add_point_load('N1', 'Q', fy=-10.0)
        s.delete_load('N1')
        self.assertFalse([p for p in s.point_loads if p.node_id == 'N1'])

    def test_delete_temperature(self):
        s = self._model()
        s.add_temperature_load('B1', 'Q', delta_t_uniform=20.0)
        self.assertTrue(s.temperature_loads)
        s.delete_temperature('B1')
        self.assertFalse(s.temperature_loads)

    def test_delete_spring(self):
        s = self._model()
        s.add_node_spring('N1', kx=1000.0)
        self.assertIn('N1', s.node_springs)
        s.delete_spring('N1')
        self.assertNotIn('N1', s.node_springs)

    def test_removes_are_noops_when_nothing_matches(self):
        s = self._model()
        s.delete_support('N2')
        s.free_edge('R1', 'left')
        s.delete_load('N2')
        s.delete_temperature('B1')
        s.delete_spring('N2')                       # none raise, nothing changes


if __name__ == '__main__':
    unittest.main()
