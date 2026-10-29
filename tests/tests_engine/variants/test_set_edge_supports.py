"""A support set can set the edges of an area object, whose mesh nodes cannot be listed one by one."""
import unittest

from context import Structure2D
from xdfem2d.models import SupportSet, ConstructionPhase, ConstructionSequence
from xdfem2d.phasing import solve_sequence, _effective_edge_restraints
from xdfem2d.structure_io import _to_dict, _from_dict


def _slab():
    s = Structure2D(domain="plate")
    s.create_polygon([(0, 0), (6, 0), (6, 4), (0, 4)], thickness=0.2, id="Slab")
    return s


PIN = (True, False, False)


class SetEdgeSupportsTests(unittest.TestCase):
    def test_a_set_gives_the_object_its_edges(self):
        s = _slab()
        s.add_support_set(SupportSet(id="E", edge_supports={"Slab": [PIN, None, PIN, None]}))
        sub = s.derive_variant(__import__("xdfem2d").models.Variant(id="v"), [s.support_sets["E"]])
        names = sub.geometry_objects["Slab"].edge_supports
        self.assertEqual([n == "free" for n in names], [False, True, False, True])
        self.assertEqual(sub.supports[names[0]].ux, True)
        self.assertIsNone(getattr(s.geometry_objects["Slab"], "edge_supports", None),
                          "the base is not changed")

    def test_an_unlisted_object_keeps_its_own_edges(self):
        s = _slab()
        s.pin_edge("Slab", "bottom")
        before = list(s.geometry_objects["Slab"].edge_supports)
        s.add_support_set(SupportSet(id="E"))
        s2 = s.derive_variant(__import__("xdfem2d").models.Variant(id="v"), [s.support_sets["E"]])
        self.assertEqual(s2.geometry_objects["Slab"].edge_supports, before)

    def test_a_wrong_object_or_edge_count_is_refused(self):
        s = _slab()
        with self.assertRaises(ValueError):
            s.apply_support_set(SupportSet(id="E", edge_supports={"Nope": [PIN] * 4}))
        with self.assertRaises(ValueError):
            s.apply_support_set(SupportSet(id="E", edge_supports={"Slab": [PIN] * 3}))

    def test_it_survives_a_save_and_the_references(self):
        s = _slab()
        s.add_support_set(SupportSet(id="E", edge_supports={"Slab": [PIN, None, PIN, None]}))
        back = _from_dict(_to_dict(s))
        self.assertEqual(back.support_sets["E"].edge_supports["Slab"], [PIN, None, PIN, None])
        from xdfem2d import references
        references.rename(s, "object", "Slab", "Slab2")
        self.assertIn("Slab2", s.support_sets["E"].edge_supports)
        references.remove(s, "object", "Slab2")
        self.assertEqual(s.support_sets["E"].edge_supports, {})

    def test_the_phase_state_and_the_release_guard(self):
        s = _slab()
        s.add_support_set(SupportSet(id="E", edge_supports={"Slab": [PIN, None, None, None]}))
        eff = _effective_edge_restraints(s, ["E"], None)
        self.assertEqual(eff["Slab"][0], PIN)
        self.assertEqual(_effective_edge_restraints(s, [], None), {})
        seq = ConstructionSequence(id="Q", phases=[
            ConstructionPhase(id="P1", support_set_ids=["E"]),
            ConstructionPhase(id="P2", support_set_ids=[])])
        with self.assertRaises(ValueError) as cm:
            solve_sequence(s, seq)
        self.assertIn("release_supports", str(cm.exception))
        seq.phases[1].release_supports = False
        solve_sequence(s, seq)


if __name__ == "__main__":
    unittest.main()
