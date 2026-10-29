"""Structure2D quad-aware area/element accessors — the single source of truth
the GUI (analysis guard, exporters, load pickers/managers) uses so a quad-only
model is never mistaken for "no elements" and quad loads are never hidden."""
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)
from xdfem2d import templates as T


def _quad_only_slab():
    # Flat slab meshed as quadrilaterals → only quad_elements, no tris/bars.
    return T.flat_slab(6.0, 6.0, 2, 2, 0.22, 3.0, prefer_quad=True)


class TestElementAccessors(unittest.TestCase):
    def test_has_elements_and_count_for_quad_only(self):
        s = _quad_only_slab()
        self.assertEqual(len(s.tri_elements), 0)
        self.assertGreater(len(s.quad_elements), 0)
        self.assertTrue(s.has_elements())
        self.assertEqual(s.element_count(), len(s.quad_elements))

    def test_area_element_ids_and_iter(self):
        s = _quad_only_slab()
        ids = s.area_element_ids()
        self.assertEqual(set(ids), set(s.quad_elements_by_id))
        kinds = {k for _id, k, _e in s.iter_area_elements()}
        self.assertEqual(kinds, {"quad"})

    def test_empty_model_has_no_elements(self):
        s = Structure2D(domain="plate")
        self.assertFalse(s.has_elements())
        self.assertEqual(s.element_count(), 0)


class TestAreaLoadIterators(unittest.TestCase):
    def setUp(self):
        self.s = _quad_only_slab()
        self.qid = self.s.quad_elements[0].id
        q = self.s.quad_elements[0]
        self.s.add_area_load(self.qid, "SW", pz=-4.0)
        self.s.add_quad_temperature_load(self.qid, "SW", dt_gradient=8.0)
        self.s.add_quad_edge_load("EL1", self.qid, q.node_i, q.node_j, "SW",
                                  fx=1.0)

    def test_iter_area_loads_includes_quad(self):
        kinds = {k for k, _ in self.s.iter_area_loads()}
        self.assertIn("quad", kinds)

    def test_iter_area_temperature_loads_includes_quad(self):
        kinds = {k for k, _ in self.s.iter_area_temperature_loads()}
        self.assertIn("quad", kinds)

    def test_iter_area_edge_loads_includes_quad(self):
        kinds = {k for k, _ in self.s.iter_area_edge_loads()}
        self.assertIn("quad", kinds)

    def test_quad_only_model_solves(self):
        self.assertIn("displacements", self.s.calculate())


if __name__ == "__main__":
    unittest.main()
