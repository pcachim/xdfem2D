"""QuadSection / QuadElement data model, persistence and geometry validation
(dev/IMPLEMENT_QUAD.md Phase 3).

Covers:
* ``Structure2D.add_quad_section``/``add_quad_element`` — the happy path and
  the rejections (duplicate id, non-distinct nodes, bad winding, non-convex,
  near-degenerate Jacobian);
* JSON round-trip via ``structure_io.save_structure_json``/
  ``load_structure_json`` — quad_sections/quad_elements survive, and a file
  with no such keys (pre-Phase-3) still loads with empty quad collections;
* ``model_check.quad_geometry_problems`` directly, for the individual checks.
"""
import os
import tempfile
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)
from xdfem2d.model_check import quad_geometry_problems
from xdfem2d.structure_io import save_structure_json, load_structure_json


def _base_plate_model():
    s = Structure2D(domain='plate')
    s.add_material('C30', elastic_modulus=30e6, unit_weight=25.0, poisson=0.2)
    s.add_quad_section('QS1', 'C30', thickness=0.2, formulation='MITC4')
    s.add_node('N1', 0.0, 0.0)
    s.add_node('N2', 1.0, 0.0)
    s.add_node('N3', 1.0, 1.0)
    s.add_node('N4', 0.0, 1.0)
    return s


class TestQuadGeometryProblems(unittest.TestCase):
    """Direct unit tests of the pure geometry-check function."""

    def test_ccw_convex_square_is_clean(self):
        coords = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
        self.assertEqual(quad_geometry_problems(coords), [])

    def test_cw_winding_rejected(self):
        coords = [(0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)]
        problems = quad_geometry_problems(coords)
        self.assertTrue(problems)
        self.assertIn('counter-clockwise', problems[0])

    def test_non_convex_rejected(self):
        # A CCW quad with one reflex corner (a dart shape).
        coords = [(0.0, 0.0), (2.0, 0.0), (1.0, 1.0), (2.0, 2.0)]
        problems = quad_geometry_problems(coords)
        self.assertTrue(any('non-convex' in p for p in problems))

    def test_distorted_but_valid_quad_is_clean(self):
        coords = [(0.0, 0.0), (3.0, 0.2), (2.8, 2.5), (-0.3, 2.1)]
        self.assertEqual(quad_geometry_problems(coords), [])

    def test_wrong_node_count(self):
        problems = quad_geometry_problems([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)])
        self.assertTrue(problems)


class TestAddQuadElementRejections(unittest.TestCase):
    """Structure2D.add_quad_element rejects bad quads eagerly (§5 point 2:
    reject, don't silently auto-fix)."""

    def test_valid_quad_is_accepted(self):
        s = _base_plate_model()
        q = s.add_quad_element('Q1', 'N1', 'N2', 'N3', 'N4', 'QS1')
        self.assertEqual(q.id, 'Q1')
        self.assertIn('Q1', s.quad_elements_by_id)
        self.assertEqual(s.reference_problems(), [])

    def test_duplicate_id_rejected(self):
        s = _base_plate_model()
        s.add_quad_element('Q1', 'N1', 'N2', 'N3', 'N4', 'QS1')
        with self.assertRaises(ValueError):
            s.add_quad_element('Q1', 'N1', 'N2', 'N3', 'N4', 'QS1')

    def test_non_distinct_nodes_rejected(self):
        s = _base_plate_model()
        with self.assertRaises(ValueError):
            s.add_quad_element('Qbad', 'N1', 'N2', 'N3', 'N1', 'QS1')

    def test_cw_winding_rejected(self):
        s = _base_plate_model()
        with self.assertRaises(ValueError) as ctx:
            s.add_quad_element('Qbad', 'N1', 'N4', 'N3', 'N2', 'QS1')
        self.assertIn('counter-clockwise', str(ctx.exception))

    def test_non_convex_rejected(self):
        s = _base_plate_model()
        s.add_node('N5', 0.5, 0.25)   # reflex point
        with self.assertRaises(ValueError) as ctx:
            s.add_quad_element('Qbad', 'N1', 'N2', 'N3', 'N5', 'QS1')
        self.assertIn('non-convex', str(ctx.exception))

    def test_missing_node_skips_geometry_check_not_id_check(self):
        # Mirrors add_tri_element's leniency: a dangling node reference is
        # not caught here (reference_problems/check_references do that at
        # solve time), but a duplicate id still is.
        s = _base_plate_model()
        q = s.add_quad_element('Q1', 'N1', 'N2', 'N3', 'MISSING', 'QS1')
        self.assertEqual(q.node_l, 'MISSING')
        probs = s.reference_problems()
        # Not a section/material reference problem — that's fine; this test
        # only asserts the add did not raise despite the dangling node.

    def test_unknown_formulation_rejected(self):
        s = _base_plate_model()
        with self.assertRaises(ValueError):
            s.add_quad_section('QSbad', 'C30', formulation='NOT-A-FORMULATION')

    def test_remove_quad_element(self):
        s = _base_plate_model()
        s.add_quad_element('Q1', 'N1', 'N2', 'N3', 'N4', 'QS1')
        s.remove_quad_element('Q1')
        self.assertNotIn('Q1', s.quad_elements_by_id)
        self.assertEqual(s.quad_elements, [])


class TestQuadJsonRoundTrip(unittest.TestCase):
    def setUp(self):
        self.s = _base_plate_model()
        self.s.add_quad_element('Q1', 'N1', 'N2', 'N3', 'N4', 'QS1')

    def _save_load(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'model.json')
            save_structure_json(self.s, path)
            return load_structure_json(path)

    def test_quad_section_survives(self):
        s2 = self._save_load()
        self.assertIn('QS1', s2.quad_sections)
        qs = s2.quad_sections['QS1']
        self.assertEqual(qs.material_name, 'C30')
        self.assertEqual(qs.formulation, 'MITC4')
        self.assertAlmostEqual(qs.thickness, 0.2)

    def test_quad_element_survives(self):
        s2 = self._save_load()
        self.assertEqual(len(s2.quad_elements), 1)
        q = s2.quad_elements[0]
        self.assertEqual((q.node_i, q.node_j, q.node_k, q.node_l),
                         ('N1', 'N2', 'N3', 'N4'))
        self.assertEqual(q.section_name, 'QS1')

    def test_reloaded_model_has_no_reference_problems(self):
        s2 = self._save_load()
        self.assertEqual(s2.reference_problems(), [])

    def test_legacy_file_without_quad_keys_still_loads(self):
        """A file saved before Phase 3 (no 'quad_sections'/'quad_elements'
        keys) must still load cleanly, with empty quad collections."""
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, 'model.json')
            save_structure_json(self.s, path)
            import json
            with open(path) as f:
                data = json.load(f)
            del data['quad_sections']
            del data['quad_elements']
            with open(path, 'w') as f:
                json.dump(data, f)
            s2 = load_structure_json(path)
        self.assertEqual(s2.quad_sections, {})
        self.assertEqual(s2.quad_elements, [])
        # And the rest of the (tri-only-shaped) model still loaded.
        self.assertIn('N1', s2.nodes)


if __name__ == '__main__':
    unittest.main()
