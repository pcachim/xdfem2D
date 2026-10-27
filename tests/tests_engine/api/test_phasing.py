"""Tests for the Phase-3 construction-phasing core (xdfem2d.phasing).

The central check is the *monolithic equivalence*: a phase with no inherited
state, applying all loads on the full geometry, must equal the ordinary
single-shot analysis. Plus additivity of the initial state and freezing.
"""
import unittest

from context import Structure2D, assert_close
from xdfem2d.models import ConstructionPhase, ElementInitialState
from xdfem2d.phasing import solve_phase, initial_state_from_results


def _beam():
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    s.add_node("N1", 0.0, 0.0)
    s.add_node("N2", 5.0, 0.0)
    s.add_node("N3", 10.0, 0.0)
    s.add_bar_element("E1", "N1", "N2", "S")
    s.add_bar_element("E2", "N2", "N3", "S")
    s.add_support("PIN", ux=True, uy=True)
    s.add_support("ROLLER", ux=False, uy=True)
    s.assign_support("N1", "PIN")
    s.assign_support("N3", "ROLLER")
    s.add_load_case("LC")
    s.add_distributed_load("E1", "LC", fye=-10.0, fyd=-10.0)
    s.add_distributed_load("E2", "LC", fye=-10.0, fyd=-10.0)
    return s


class TestMonolithicEquivalence(unittest.TestCase):
    """Zero initial state + all loads on full geometry == single-shot analysis."""

    def setUp(self):
        self.s = _beam()
        self.ref = self.s.calculate()
        self.phase = ConstructionPhase(
            id="P", active_elements={"E1", "E2"}, applied_cases=["LC"])
        self.out = solve_phase(self.s, self.phase)

    def test_element_forces_match(self):
        for eid in ("E1", "E2"):
            for end in ("i", "j"):
                for k in range(3):
                    assert_close(self, self.out["element_forces"][eid][end][k],
                                 self.ref["element_forces"]["LC"][eid][end][k],
                                 rel=1e-9, abs_tol=1e-6, msg=f"{eid}.{end}[{k}]")

    def test_displacements_match(self):
        for nid in ("N1", "N2", "N3"):
            for k in range(3):
                assert_close(self, self.out["displacements"][nid][k],
                             self.ref["displacements"]["LC"][nid][k],
                             rel=1e-9, abs_tol=1e-9)


class TestInitialStateAdditive(unittest.TestCase):
    def test_pure_initial_state_no_loads(self):
        s = _beam()
        st = {"E1": ElementInitialState(i=(1.0, 2.0, 3.0), j=(-1.0, -2.0, -3.0))}
        phase = ConstructionPhase(id="P", active_elements={"E1", "E2"},
                                  applied_cases=[], initial_state=st)
        out = solve_phase(s, phase)
        # No loads → increment is zero → final == initial state exactly.
        self.assertEqual(out["element_forces"]["E1"]["i"], [1.0, 2.0, 3.0])
        self.assertEqual(out["element_forces"]["E1"]["j"], [-1.0, -2.0, -3.0])

    def test_final_is_increment_plus_initial(self):
        s = _beam()
        # First solve to obtain a realistic state, seed it as initial.
        ref = s.calculate()
        st = initial_state_from_results(ref, "LC", element_ids={"E1", "E2"})
        phase = ConstructionPhase(id="P", active_elements={"E1", "E2"},
                                  applied_cases=["LC"], initial_state=st)
        out = solve_phase(s, phase)
        # Final = increment(LC) + initial(LC) = 2× LC.
        for end in ("i", "j"):
            for k in range(3):
                assert_close(self, out["element_forces"]["E1"][end][k],
                             2.0 * ref["element_forces"]["LC"]["E1"][end][k],
                             rel=1e-9, abs_tol=1e-6)


class TestStrengthening(unittest.TestCase):
    """Old bars carry initial state; a new bar enters with zero initial state."""

    def test_new_element_has_no_initial(self):
        s = _beam()
        ref = s.calculate()
        st = initial_state_from_results(ref, "LC", element_ids={"E1"})  # only E1
        phase = ConstructionPhase(id="P", active_elements={"E1", "E2"},
                                  applied_cases=["LC"], initial_state=st)
        out = solve_phase(s, phase)
        # E2 (the "new" bar) carries only the increment, no inherited state.
        for end in ("i", "j"):
            for k in range(3):
                assert_close(self, out["element_forces"]["E2"][end][k],
                             ref["element_forces"]["LC"]["E2"][end][k],
                             rel=1e-9, abs_tol=1e-6)


class TestFreezing(unittest.TestCase):
    def test_initial_state_object_unchanged(self):
        s = _beam()
        st_obj = ElementInitialState(i=(5.0, 0.0, 0.0), j=(-5.0, 0.0, 0.0))
        phase = ConstructionPhase(id="P", active_elements={"E1", "E2"},
                                  applied_cases=["LC"],
                                  initial_state={"E1": st_obj})
        _ = solve_phase(s, phase)
        self.assertEqual(st_obj.i, (5.0, 0.0, 0.0))
        self.assertEqual(st_obj.j, (-5.0, 0.0, 0.0))


class TestSinglePhaseDegenerate(unittest.TestCase):
    def test_partial_geometry_phase(self):
        # A phase that activates only E1 behaves like the analysis of that
        # sub-structure (N3 kept as it is supported).
        s = _beam()
        phase = ConstructionPhase(id="P", active_elements={"E1"},
                                  applied_cases=["LC"])
        out = solve_phase(s, phase)
        self.assertIn("E1", out["element_forces"])
        self.assertNotIn("E2", out["element_forces"])


if __name__ == "__main__":
    unittest.main()
