"""Two guards from dev/refactor_area_path.md items F and D-Part 1.

F  — cross-kind element id collision: a bar, a triangle and a quad may not
     share an id (implement_quad.md Phase 13). Phase 5 already closed tri↔quad;
     this closes bar↔tri and bar↔quad in both directions.

D1 — P-Delta safety guard: a GeometricNonlinear case on a plane model with no
     bar elements would silently return an all-zero field (the geometric
     stiffness is assembled from bars only). It is now refused up front.
"""
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)


def _plane_with_all_kinds():
    s = Structure2D(domain='plane')
    s.add_material('m', elastic_modulus=30e6, unit_weight=25.0, poisson=0.2)
    s.add_section('S', 'm', b=0.3, h=0.5)
    s.add_tri_section('T', 'm', thickness=0.2)
    s.add_quad_section('Q', 'm', thickness=0.2, formulation='QM6')
    for nid, (x, y) in {'n1': (0, 0), 'n2': (1, 0), 'n3': (1, 1),
                        'n4': (0, 1)}.items():
        s.add_node(nid, x, y)
    return s


class TestCrossKindIdCollision(unittest.TestCase):

    def test_bar_id_rejected_for_triangle_and_quad(self):
        s = _plane_with_all_kinds()
        s.add_bar_element('E1', 'n1', 'n2', 'S')
        with self.assertRaises(ValueError):
            s.add_tri_element('E1', 'n1', 'n2', 'n3', 'T')
        with self.assertRaises(ValueError):
            s.add_quad_element('E1', 'n1', 'n2', 'n3', 'n4', 'Q')

    def test_triangle_and_quad_ids_rejected_for_a_bar(self):
        s = _plane_with_all_kinds()
        s.add_tri_element('T1', 'n1', 'n2', 'n3', 'T')
        s.add_quad_element('Q1', 'n1', 'n2', 'n3', 'n4', 'Q')
        with self.assertRaises(ValueError):
            s.add_bar_element('T1', 'n1', 'n2', 'S')
        with self.assertRaises(ValueError):
            s.add_bar_element('Q1', 'n1', 'n2', 'S')

    def test_distinct_ids_across_all_three_kinds_are_fine(self):
        s = _plane_with_all_kinds()
        s.add_bar_element('B1', 'n1', 'n2', 'S')
        s.add_tri_element('T1', 'n1', 'n2', 'n3', 'T')
        s.add_quad_element('Q1', 'n1', 'n2', 'n3', 'n4', 'Q')
        self.assertEqual(set(s.bar_elements_by_id), {'B1'})
        self.assertEqual(set(s.tri_elements_by_id), {'T1'})
        self.assertEqual(set(s.quad_elements_by_id), {'Q1'})


class TestPDeltaGuard(unittest.TestCase):

    def _quad_wall_pdelta(self):
        s = Structure2D(domain='plane')
        s.add_material('m', elastic_modulus=30e6, unit_weight=25.0, poisson=0.2)
        s.add_quad_section('Q', 'm', thickness=0.2, formulation='QM6')
        for nid, (x, y) in {'n1': (0, 0), 'n2': (1, 0), 'n3': (1, 1),
                            'n4': (0, 1)}.items():
            s.add_node(nid, x, y)
        s.add_quad_element('Q1', 'n1', 'n2', 'n3', 'n4', 'Q')
        s.add_load_case('D')
        s.add_analysis_case('PD', 'GeometricNonlinear', {'D': 1.0})
        return s

    def test_pdelta_on_a_barless_plane_model_is_refused(self):
        s = self._quad_wall_pdelta()
        probs = s.domain_problems()
        self.assertTrue(any('P-Delta' in p for p in probs))
        with self.assertRaises(ValueError):
            s.check_references()

    def test_pdelta_is_allowed_once_a_bar_exists(self):
        s = self._quad_wall_pdelta()
        s.add_section('S', 'm', b=0.3, h=0.5)
        s.add_bar_element('B1', 'n1', 'n2', 'S')
        self.assertFalse(any('P-Delta' in p for p in s.domain_problems()))

    def test_linear_case_on_a_barless_model_is_fine(self):
        s = self._quad_wall_pdelta()
        # Swap the P-Delta case for a Linear one — no guard should fire.
        s.analysis_cases.clear(); s.analysis_cases_by_id.clear()
        s.add_analysis_case('L', 'Linear', {'D': 1.0})
        self.assertFalse(any('P-Delta' in p for p in s.domain_problems()))


if __name__ == '__main__':
    unittest.main()
