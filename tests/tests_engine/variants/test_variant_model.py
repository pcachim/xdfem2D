"""Per-variant own action model (inherit / clean / custom loads & combos)."""
import os
import tempfile
import unittest

from context import Structure2D, assert_close
from xdfem2d.models import Variant
from xdfem2d.workflows import make_variant_model


def _beam():
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    s.add_node("N1", 0.0, 0.0); s.add_node("N2", 5.0, 0.0); s.add_node("N3", 10.0, 0.0)
    s.add_bar_element("E1", "N1", "N2", "S"); s.add_bar_element("E2", "N2", "N3", "S")
    s.add_support("PIN", ux=True, uy=True); s.add_support("ROLLER", uy=True)
    s.assign_support("N1", "PIN"); s.assign_support("N3", "ROLLER")
    s.add_load_case("LC"); s.add_distributed_load("E1", "LC", fye=-10.0, fyd=-10.0)
    return s


class TestInheritVsClean(unittest.TestCase):
    def test_inherit_copies_actions(self):
        base = _beam()
        m = make_variant_model(base, inherit=True)
        self.assertEqual([lc.id for lc in m.load_cases], ["LC"])
        self.assertEqual(len(m.distributed_loads), 1)
        self.assertEqual(m.variants, {})         # nested overlays stripped

    def test_clean_strips_actions_keeps_geometry(self):
        base = _beam()
        m = make_variant_model(base, inherit=False)
        self.assertEqual(m.load_cases, [])
        self.assertEqual(m.distributed_loads, [])
        self.assertEqual(len(m.nodes), 3)        # geometry kept
        self.assertEqual(len(m.bar_elements), 2)
        self.assertIn("PIN", m.supports)


class TestDeriveUsesModel(unittest.TestCase):
    def test_variant_with_own_loads(self):
        base = _beam()
        # Variant model: same geometry, different load (heavier on E2 instead).
        m = make_variant_model(base, inherit=False)
        m.add_load_case("LC")
        m.add_distributed_load("E2", "LC", fye=-20.0, fyd=-20.0)
        v = Variant(id="V1", description="own loads", model=m)
        base.add_variant(v)
        res_own = base.derive_variant(v).calculate()
        # Inherited variant (base loads: on E1) for comparison.
        res_base = base.derive_variant(Variant(id="V0")).calculate()
        m_own = res_own["element_forces"]["LC"]["E2"]["i"][2]
        m_base = res_base["element_forces"]["LC"]["E2"]["i"][2]
        # The variant's own load (on E2) changes E2's moment vs the base (load on E1).
        self.assertGreater(abs(m_own - m_base), 1.0)

    def test_variant_without_model_inherits(self):
        base = _beam()
        v = Variant(id="V0")          # no model → inherits base actions
        sub = base.derive_variant(v)
        res = sub.calculate()
        self.assertIn("LC", res["element_forces"])
        # base load on E1 present
        self.assertGreater(abs(res["element_forces"]["LC"]["E1"]["i"][1]), 1.0)


class TestRoundTrip(unittest.TestCase):
    def test_variant_model_persists(self):
        base = _beam()
        m = make_variant_model(base, inherit=False)
        m.add_load_case("WIND")
        m.add_distributed_load("E2", "WIND", fye=-5.0, fyd=-5.0)
        base.add_variant(Variant(id="V1", description="wind only", model=m))
        fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
        try:
            base.save(path); r = Structure2D.load(path)
        finally:
            os.remove(path)
        v = r.variants["V1"]
        self.assertIsNotNone(v.model)
        self.assertEqual([lc.id for lc in v.model.load_cases], ["WIND"])
        self.assertEqual(len(v.model.distributed_loads), 1)


if __name__ == "__main__":
    unittest.main()
