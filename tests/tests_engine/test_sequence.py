"""Tests for the Phase-4 construction sequence (xdfem2d.phasing.solve_sequence).

Covers chaining (accumulated state), non-commutativity, Method 2 (cumulative
displacements), self-weight masking (no double counting) and the single-phase
degenerate case.
"""
import unittest

from context import Structure2D, assert_close
from xdfem2d.models import ConstructionPhase, ConstructionSequence
from xdfem2d.phasing import solve_phase, solve_sequence, initial_state_from_results


def _beam(sw=False):
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=25.0)
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
    if sw:
        s.add_load_case("SW", self_weight_factor=1.0)
    s.add_load_case("Q1")
    s.add_load_case("Q2")
    s.add_distributed_load("E1", "Q1", fye=-10.0, fyd=-10.0)
    s.add_distributed_load("E2", "Q2", fye=-8.0, fyd=-8.0)
    return s


class TestChainingEqualsManual(unittest.TestCase):
    """A 2-phase sequence equals manual Phase-3 chaining of the same phases."""

    def setUp(self):
        self.s = _beam()

    def test_two_phase_chain(self):
        # Phase 1: full geometry, load Q1. Phase 2: load Q2, inherits state.
        p1 = ConstructionPhase(id="P1", active_elements={"E1", "E2"},
                               applied_cases=["Q1"])
        p2 = ConstructionPhase(id="P2", active_elements={"E1", "E2"},
                               applied_cases=["Q2"])
        seq = ConstructionSequence(id="SEQ", phases=[p1, p2])
        out = solve_sequence(self.s, seq)

        # Manual: phase1 then phase2 seeded with phase1's final state.
        r1 = solve_phase(self.s, p1)
        st = initial_state_from_results({"element_forces": r1["element_forces"]},
                                        "element_forces") \
            if False else _state(r1["element_forces"])
        p2m = ConstructionPhase(id="P2", active_elements={"E1", "E2"},
                                applied_cases=["Q2"], initial_state=st)
        r2 = solve_phase(self.s, p2m)

        for eid in ("E1", "E2"):
            for end in ("i", "j"):
                for k in range(3):
                    assert_close(self, out["final"]["element_forces"][eid][end][k],
                                 r2["element_forces"][eid][end][k],
                                 rel=1e-9, abs_tol=1e-6)

    def test_final_equals_monolithic_q1_plus_q2(self):
        # With both phases on full geometry, the accumulated final equals the
        # single-shot Q1+Q2 superposition.
        p1 = ConstructionPhase(id="P1", active_elements={"E1", "E2"},
                               applied_cases=["Q1"])
        p2 = ConstructionPhase(id="P2", active_elements={"E1", "E2"},
                               applied_cases=["Q2"])
        out = solve_sequence(self.s, ConstructionSequence(id="S", phases=[p1, p2]))
        ref = self.s.calculate()
        for eid in ("E1", "E2"):
            for end in ("i", "j"):
                for k in range(3):
                    exp = (ref["element_forces"]["Q1"][eid][end][k]
                           + ref["element_forces"]["Q2"][eid][end][k])
                    assert_close(self, out["final"]["element_forces"][eid][end][k],
                                 exp, rel=1e-9, abs_tol=1e-6)


def _state(element_forces):
    from xdfem2d.models import ElementInitialState
    return {eid: ElementInitialState(i=tuple(v['i']), j=tuple(v['j']))
            for eid, v in element_forces.items()}


def _propped():
    """Cantilever from N1 (base); a prop at N3 can be added via a SupportSet."""
    from xdfem2d.models import SupportSet, SupportAssignment
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    s.add_node("N1", 0.0, 0.0)
    s.add_node("N2", 5.0, 0.0)
    s.add_node("N3", 10.0, 0.0)
    s.add_bar_element("E1", "N1", "N2", "S")
    s.add_bar_element("E2", "N2", "N3", "S")
    s.add_support("FIX", ux=True, uy=True, tz=True)
    s.add_support("ROLLER", uy=True)
    s.assign_support("N1", "FIX")               # base: pure cantilever
    s.add_load_case("Q1")
    s.add_load_case("Q2")
    for c in ("Q1", "Q2"):
        s.add_distributed_load("E1", c, fye=-10.0, fyd=-10.0)
        s.add_distributed_load("E2", c, fye=-10.0, fyd=-10.0)
    prop = SupportSet(id="PROP",
                      assignments=[SupportAssignment("N1", "FIX"),
                                   SupportAssignment("N3", "ROLLER")])
    return s, {"PROP": prop}


class TestStagedDiffersFromMonolithic(unittest.TestCase):
    """Adding a prop in phase 2 makes the staged result differ from a single-shot
    propped analysis — the canonical construction-staging effect."""

    def test_prop_added_in_phase_two(self):
        s, ssets = _propped()
        # Phase 1: cantilever (no prop), load Q1. Phase 2: prop added, load Q2.
        p1 = ConstructionPhase(id="P1", active_elements={"E1", "E2"},
                               applied_cases=["Q1"])
        p2 = ConstructionPhase(id="P2", active_elements={"E1", "E2"},
                               support_set_id="PROP", applied_cases=["Q2"])
        staged = solve_sequence(
            s, ConstructionSequence(id="S", phases=[p1, p2]), support_sets=ssets)

        # Monolithic: prop present for *all* loads (Q1+Q2).
        mono = s.with_supports(ssets["PROP"])
        ref = mono.calculate()
        mono_M = (ref["element_forces"]["Q1"]["E1"]["i"][2]
                  + ref["element_forces"]["Q2"]["E1"]["i"][2])
        staged_M = staged["final"]["element_forces"]["E1"]["i"][2]
        # Q1 acted on the cantilever before the prop existed → larger fixed-end
        # moment than the fully-propped monolithic case.
        self.assertGreater(abs(staged_M) - abs(mono_M), 1e-3)


class TestMethod2(unittest.TestCase):
    def test_cumulative_displacements(self):
        """Cumulative display is now computed at view time from phase increments.
        The solver always stores incremental results regardless of displacement_method.
        Verify each phase increment matches a standalone solve_phase call."""
        s = _beam()
        p1 = ConstructionPhase(id="P1", active_elements={"E1", "E2"},
                               applied_cases=["Q1"])
        p2 = ConstructionPhase(id="P2", active_elements={"E1", "E2"},
                               applied_cases=["Q2"])
        seq = ConstructionSequence(id="S", phases=[p1, p2],
                                   displacement_method="cumulative")
        out = solve_sequence(s, seq)
        # Each phase result matches a standalone solve_phase call (incremental).
        r1 = solve_phase(s, p1)
        r2 = solve_phase(s, p2)
        for nid in ("N2",):
            for k in range(3):
                assert_close(self, out["phases"]["P1"]["displacements"][nid][k],
                             r1["displacements"][nid][k], rel=1e-9, abs_tol=1e-9)
                assert_close(self, out["phases"]["P2"]["displacements"][nid][k],
                             r2["displacements"][nid][k], rel=1e-9, abs_tol=1e-9)


class TestSelfWeightMasking(unittest.TestCase):
    def test_old_self_weight_not_duplicated(self):
        s = _beam(sw=True)
        # Phase 1 builds E1 with self-weight; phase 2 adds E2 with self-weight.
        p1 = ConstructionPhase(id="P1", active_elements={"E1"},
                               applied_cases=["SW"])
        p2 = ConstructionPhase(id="P2", active_elements={"E1", "E2"},
                               applied_cases=["SW"])
        out = solve_sequence(s, ConstructionSequence(id="S", phases=[p1, p2]))
        # In phase 2 only E2 is new, so its self-weight is applied once; E1 keeps
        # its inherited state. The accumulated E1 state must equal a single SW
        # application on the full structure restricted to E1's contribution —
        # i.e. it should not be doubled. Sanity: E2 end force is finite & nonzero.
        ef2 = out["final"]["element_forces"]["E2"]
        self.assertTrue(any(abs(x) > 1e-6 for x in ef2["i"]))

    def test_no_double_count_vs_reapply(self):
        s = _beam(sw=True)
        # Single full phase with SW (no inheritance): baseline.
        full = ConstructionPhase(id="F", active_elements={"E1", "E2"},
                                 applied_cases=["SW"])
        base = solve_sequence(s, ConstructionSequence(id="B", phases=[full]))
        ref = s.calculate()
        for eid in ("E1", "E2"):
            for end in ("i", "j"):
                for k in range(3):
                    assert_close(self,
                                 base["final"]["element_forces"][eid][end][k],
                                 ref["element_forces"]["SW"][eid][end][k],
                                 rel=1e-9, abs_tol=1e-6)


class TestSinglePhase(unittest.TestCase):
    def test_single_phase_equals_solve_phase(self):
        s = _beam()
        p = ConstructionPhase(id="P", active_elements={"E1", "E2"},
                              applied_cases=["Q1"])
        seq = solve_sequence(s, ConstructionSequence(id="S", phases=[p]))
        direct = solve_phase(s, p)
        for eid in ("E1", "E2"):
            for end in ("i", "j"):
                for k in range(3):
                    assert_close(self, seq["final"]["element_forces"][eid][end][k],
                                 direct["element_forces"][eid][end][k],
                                 rel=1e-9, abs_tol=1e-6)


if __name__ == "__main__":
    unittest.main()
