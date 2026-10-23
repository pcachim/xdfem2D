"""Neutral "area element" accessors and quad-aware staging.

dev/refactor_area_path.md Phase 1: a triangle and a quad are the same
modelling role, so Structure2D exposes area_elements()/area_element_by_id()/
area_kind_of()/remove_area_element() that treat both as one "area element",
and the staging helpers (workflows) use them so a quad can finally be assigned
to a construction stage — before, quad ids were silently ignored.
"""
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)
from xdfem2d import workflows as W


def _mixed():
    """A plate model with one triangle and one quad sharing an edge."""
    s = Structure2D(domain='plate')
    s.add_material('M', elastic_modulus=30e6, unit_weight=25.0, poisson=0.2)
    s.add_plate_section('T', 'M', thickness=0.2)
    s.add_quad_section('Q', 'M', thickness=0.2, formulation='MITC4')
    for nid, (x, y) in {'n1': (0, 0), 'n2': (1, 0), 'n3': (1, 1),
                        'n4': (0, 1), 'n5': (2, 0), 'n6': (2, 1)}.items():
        s.add_node(nid, x, y)
    s.add_tri_element('T1', 'n1', 'n2', 'n3', 'T')
    s.add_quad_element('Q1', 'n2', 'n5', 'n6', 'n3', 'Q')
    return s


class TestAccessors(unittest.TestCase):

    def test_area_elements_lists_both_kinds(self):
        s = _mixed()
        self.assertEqual({e.id for e in s.area_elements()}, {'T1', 'Q1'})

    def test_by_id_resolves_both_and_misses_gracefully(self):
        s = _mixed()
        self.assertEqual(s.area_element_by_id('T1').id, 'T1')
        self.assertEqual(s.area_element_by_id('Q1').id, 'Q1')
        self.assertIsNone(s.area_element_by_id('nope'))

    def test_kind_of(self):
        s = _mixed()
        self.assertEqual(s.area_kind_of('T1'), 'tri')
        self.assertEqual(s.area_kind_of('Q1'), 'quad')
        self.assertIsNone(s.area_kind_of('nope'))

    def test_remove_area_element_dispatches_by_kind(self):
        s = _mixed()
        self.assertTrue(s.remove_area_element('Q1'))
        self.assertNotIn('Q1', s.quad_elements_by_id)
        self.assertTrue(s.remove_area_element('T1'))
        self.assertNotIn('T1', s.tri_elements_by_id)
        self.assertFalse(s.remove_area_element('Q1'))   # already gone
        self.assertFalse(s.remove_area_element('missing'))

    def test_remove_area_element_purges_attached_loads(self):
        s = _mixed()
        s.add_load_case('Q')
        s.add_area_load('Q1', 'Q', pz=-3.0)
        self.assertEqual(len(s.quad_area_loads), 1)
        s.remove_area_element('Q1')
        self.assertEqual(len(s.quad_area_loads), 0)


class TestQuadAwareStaging(unittest.TestCase):

    def test_assign_stage_updates_a_quad(self):
        s = _mixed()
        n = W.assign_stage(s, ['T1', 'Q1'], 2)
        self.assertEqual(n, 2)
        self.assertEqual(s.quad_elements_by_id['Q1'].stage, 2)
        self.assertEqual(s.tri_elements_by_id['T1'].stage, 2)

    def test_assign_stage_ignores_unknown_ids(self):
        s = _mixed()
        self.assertEqual(W.assign_stage(s, ['nope'], 3), 0)

    def test_stage_numbers_include_quad_stages(self):
        s = _mixed()
        W.assign_stage(s, ['Q1'], 3)   # tri stays at default 1
        self.assertEqual(W.stage_numbers(s), [1, 3])

    def test_phases_activate_quads_by_stage(self):
        s = _mixed()
        W.assign_stage(s, ['T1'], 1)
        W.assign_stage(s, ['Q1'], 2)
        phases = W.phases_from_stages(s)
        by_id = {p.id: p.active_elements for p in phases}
        self.assertNotIn('Q1', by_id['Stage 1'])   # quad not yet built
        self.assertIn('T1', by_id['Stage 1'])
        self.assertIn('Q1', by_id['Stage 2'])       # quad appears at its stage
        self.assertIn('T1', by_id['Stage 2'])


class TestPhaseActivationWithQuads(unittest.TestCase):
    """dev/refactor_area_path.md Phase 2: a quad is filtered in/out of a phase
    by its stage, its nodes survive the orphan prune only while it is active,
    and its per-phase self-weight is applied when it first appears."""

    def _two_quads(self):
        from xdfem2d.structure import Structure2D
        s = Structure2D(domain='plate')
        s.add_material('m', elastic_modulus=30e6, unit_weight=25.0, poisson=0.2)
        s.add_quad_section('Q', 'm', thickness=0.2, formulation='MITC4')
        for nid, (x, y) in {'n1': (0, 0), 'n2': (1, 0), 'n3': (1, 1),
                            'n4': (0, 1), 'n5': (2, 0), 'n6': (2, 1)}.items():
            s.add_node(nid, x, y)
        s.add_quad_element('Q1', 'n1', 'n2', 'n3', 'n4', 'Q')
        s.add_quad_element('Q2', 'n2', 'n5', 'n6', 'n3', 'Q')
        return s

    def test_inactive_quad_and_its_orphan_nodes_are_removed(self):
        from xdfem2d.models import Variant
        s = self._two_quads()
        sub = s.derive_variant(Variant(id='p', active_elements={'Q1'}),
                               {}, filter_tri=True)
        self.assertEqual([q.id for q in sub.quad_elements], ['Q1'])
        # n5/n6 are used only by the inactive Q2 → pruned; Q1's stay.
        self.assertEqual(set(sub.nodes), {'n1', 'n2', 'n3', 'n4'})

    def test_active_quad_nodes_survive_the_prune(self):
        from xdfem2d.models import Variant
        s = self._two_quads()
        sub = s.derive_variant(Variant(id='p', active_elements={'Q1', 'Q2'}),
                               {}, filter_tri=True)
        self.assertEqual(set(sub.nodes),
                         {'n1', 'n2', 'n3', 'n4', 'n5', 'n6'})

    def test_per_phase_self_weight_hits_only_the_new_quad(self):
        from xdfem2d.models import Variant
        from xdfem2d.phasing import _mask_self_weight_to_new
        s = self._two_quads()
        s.add_load_case('SW', self_weight_factor=1.0)
        sub = s.derive_variant(Variant(id='p', active_elements={'Q1', 'Q2'}),
                               {}, filter_tri=True)
        _mask_self_weight_to_new(sub, {'Q2'})
        # Plate domain stores fz in the fx slot; γ·t·A/4 = 25·0.2·1/4 = 1.25.
        by_node = {pl.node_id: round(pl.fx, 4) for pl in sub.point_loads}
        for nid in ('n2', 'n3', 'n5', 'n6'):   # Q2's vertices
            self.assertEqual(by_node.get(nid), -1.25)
        for nid in ('n1', 'n4'):               # Q1-only vertices, no new SW
            self.assertNotIn(nid, by_node)


if __name__ == '__main__':
    unittest.main()
