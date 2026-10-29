"""One name for one boundary condition, wherever it was made.

Two ways of writing the same structure had drifted apart. Written by hand or by
an assistant, a model has a few shared definitions — PIN, ROLLER-X — put on many
nodes. Drawn in the application, it had one definition per supported node, named
after that node: a meshed wall base produced forty near-identical entries and
the idea of a support *type* meant nothing.

They describe the same structure and one consequence was concrete. Propagating a
support along the edge of a meshed region matches by name, so it fired on models
an assistant wrote and never on models anyone drew — and when that question came
up I presented the mismatch as an edge case. It was not: it was what the
application produced every single time.

The suffix is the direction the node can still move in: ROLLER-X rolls along x,
so ux is free and uy is held.
"""
from __future__ import annotations

import unittest

from context import Structure2D  # noqa: F401  (puts src/ on the path)

from xdfem2d.models import (CANONICAL_SUPPORTS, CANONICAL_SUPPORTS_PLATE,
                            canonical_support_name)


class TestTheNames(unittest.TestCase):

    def test_every_restrained_combination_has_a_name(self):
        """Seven of the eight. The eighth is no support at all."""
        seen = set()
        for ux in (False, True):
            for uy in (False, True):
                for tz in (False, True):
                    with self.subTest(ux=ux, uy=uy, tz=tz):
                        name = canonical_support_name(ux, uy, tz)
                        if (ux, uy, tz) == (False, False, False):
                            self.assertIsNone(name)
                        else:
                            self.assertTrue(name)
                            seen.add(name)
        self.assertEqual(len(seen), 7)

    def test_no_two_combinations_share_a_name(self):
        self.assertEqual(len(set(CANONICAL_SUPPORTS.values())),
                         len(CANONICAL_SUPPORTS))

    def test_the_familiar_two(self):
        self.assertEqual(canonical_support_name(True, True, False), 'PIN')
        self.assertEqual(canonical_support_name(True, True, True), 'FIXED')

    def test_the_suffix_is_the_direction_it_moves_in(self):
        """ROLLER-X rolls along x, so x is the free one. Named for the
        movement because that is how the support is pictured — and a name that
        has to be guessed is the failure this project keeps having."""
        self.assertEqual(canonical_support_name(False, True, False), 'ROLLER-X')
        self.assertEqual(canonical_support_name(True, False, False), 'ROLLER-Y')

    def test_guided_follows_the_same_rule(self):
        self.assertEqual(canonical_support_name(False, True, True), 'GUIDED-X')
        self.assertEqual(canonical_support_name(True, False, True), 'GUIDED-Y')

    def test_rotation_only(self):
        self.assertEqual(canonical_support_name(False, False, True), 'BLOCK')

    def test_nothing_restrained_is_not_a_support(self):
        self.assertIsNone(canonical_support_name(False, False, False))


class TestPlateNames(unittest.TestCase):
    """In the plate domain the three slots mean (w, θx, θy), so the same triple
    gets a slab-edge name, not the plane machine name."""

    def test_every_restrained_combination_has_a_plate_name(self):
        seen = set()
        for w in (False, True):
            for tx in (False, True):
                for ty in (False, True):
                    name = canonical_support_name(w, tx, ty, domain='plate')
                    if (w, tx, ty) == (False, False, False):
                        self.assertIsNone(name)
                    else:
                        self.assertTrue(name)
                        seen.add(name)
        self.assertEqual(len(seen), 7)

    def test_no_two_plate_combinations_share_a_name(self):
        self.assertEqual(len(set(CANONICAL_SUPPORTS_PLATE.values())),
                         len(CANONICAL_SUPPORTS_PLATE))

    def test_the_slab_edge_vocabulary(self):
        c = lambda *a: canonical_support_name(*a, domain='plate')  # noqa: E731
        self.assertEqual(c(True, False, False), 'SIMPLE')
        self.assertEqual(c(True, True, True), 'CLAMPED')
        self.assertEqual(c(True, True, False), 'CLAMP-X')
        self.assertEqual(c(True, False, True), 'CLAMP-Y')
        self.assertEqual(c(False, True, True), 'BLOCK')
        self.assertEqual(c(False, True, False), 'SYM-X')
        self.assertEqual(c(False, False, True), 'SYM-Y')

    def test_plane_is_the_default_and_differs_from_plate(self):
        self.assertEqual(canonical_support_name(True, False, False), 'ROLLER-Y')
        self.assertEqual(
            canonical_support_name(True, False, False, domain='plate'), 'SIMPLE')

    def test_a_plate_structure_names_its_supports_the_slab_way(self):
        s = Structure2D(domain='plate')
        for i in range(3):
            s.add_node(f'N{i}', float(i), 0.0)
            s.assign_support(f'N{i}', s.support_for(True, False, False))
        self.assertEqual(list(s.supports), ['SIMPLE'])


class TestReusingADefinition(unittest.TestCase):

    def setUp(self):
        self.s = Structure2D()
        for i in range(6):
            self.s.add_node(f'N{i}', float(i), 0.0)

    def test_many_nodes_share_one_definition(self):
        """The whole point: forty entries become one."""
        for i in range(6):
            self.s.assign_support(f'N{i}', self.s.support_for(True, True))
        self.assertEqual(list(self.s.supports), ['PIN'])
        self.assertEqual(len(self.s.support_assignments), 6)

    def test_different_restraints_get_different_definitions(self):
        self.s.support_for(True, True)
        self.s.support_for(False, True)
        self.assertEqual(sorted(self.s.supports), ['PIN', 'ROLLER-X'])

    def test_nothing_restrained_creates_nothing(self):
        self.assertIsNone(self.s.support_for())
        self.assertEqual(self.s.supports, {})

    def test_a_name_the_user_chose_is_reused_rather_than_duplicated(self):
        """A file holding a 'BASE' with these restraints keeps using BASE. The
        alternative — a canonical twin beside it — is the clutter this is
        meant to remove, arriving by another door."""
        self.s.add_support('BASE', ux=True, uy=True, tz=True)
        self.assertEqual(self.s.support_for(True, True, True), 'BASE')
        self.assertEqual(list(self.s.supports), ['BASE'])

    def test_a_taken_name_meaning_something_else_is_not_stolen(self):
        """Renaming what someone else defined would be the worse trespass."""
        self.s.add_support('PIN', ux=False, uy=True, tz=True)
        self.assertEqual(self.s.support_for(True, True), 'PIN-2')
        self.assertEqual(self.s.supports['PIN'].ux, False)

    def test_the_second_collision_keeps_counting(self):
        self.s.add_support('PIN', ux=False, uy=True, tz=True)
        self.s.add_support('PIN-2', ux=False, uy=False, tz=True)
        self.assertEqual(self.s.support_for(True, True), 'PIN-3')


class TestPruning(unittest.TestCase):
    """Changing the last node that used a ROLLER-X leaves the definition
    behind, and a table filling with entries nobody asked for is the mess this
    set out to remove."""

    def setUp(self):
        self.s = Structure2D()
        self.s.add_node('N1', 0.0, 0.0)
        self.s.add_node('N2', 1.0, 0.0)

    def test_an_unused_definition_is_dropped(self):
        self.s.assign_support('N1', self.s.support_for(False, True))
        self.s.support_assignments.clear()
        self.assertEqual(self.s.prune_unused_supports(), ['ROLLER-X'])
        self.assertEqual(self.s.supports, {})

    def test_one_still_in_use_is_kept(self):
        self.s.assign_support('N1', self.s.support_for(True, True))
        self.assertEqual(self.s.prune_unused_supports(), [])
        self.assertIn('PIN', self.s.supports)

    def test_only_the_named_ones_are_considered(self):
        """What the user wrote is theirs to keep, used or not."""
        self.s.add_support('BASE', ux=True, uy=True)
        self.s.support_for(False, False, True)          # BLOCK, unused
        gone = self.s.prune_unused_supports(only={'BLOCK'})
        self.assertEqual(gone, ['BLOCK'])
        self.assertIn('BASE', self.s.supports)

    def test_pruning_nothing_is_not_an_error(self):
        self.assertEqual(self.s.prune_unused_supports(), [])


class TestItMakesTheTwoWaysMeet(unittest.TestCase):
    """The defect that started this: a support propagated along the edge of a
    meshed region matches by name, so a drawn model never propagated."""

    def test_two_corners_drawn_the_new_way_share_a_name(self):
        s = Structure2D()
        s.add_material('C30/37', elastic_modulus=33e6, unit_weight=25.0)
        s.add_tri_section('W1', 'C30/37', thickness=0.2)
        s.add_geo_rectangle('R1', (0.0, 0.0), (4.0, 3.0),
                            section_name='W1', target_size=0.5)
        for corner in ('R1.p0', 'R1.p1'):
            s.assign_support(corner, s.support_for(True, True))
        s.add_load_case('Q')
        s.add_point_load('R1.p3', 'Q', fx=100.0)
        s.add_analysis_case('ULS', 'Linear', {'Q': 1.5})

        from xdfem2d.geo_expand import expand_geometry
        mesh, _ = expand_geometry(s)
        base = {nid for nid, n in mesh.nodes.items() if abs(n.y) < 1e-9}
        supported = {a.node_id for a in mesh.support_assignments}
        self.assertGreater(len(base), 2)
        self.assertTrue(base <= supported, 'the edge did not propagate')


if __name__ == '__main__':
    unittest.main()
