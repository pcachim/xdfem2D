"""A support set changes the supports of a construction phase; it does not replace them.

A set of provisional supports (a prop) is just the prop. The base's own supports stay in every phase that uses it, a node
the set lists takes the set's restraint (none ticked: no support there), and what the base does elsewhere still reaches
the phase. It used to be the whole list of supports of the scenario: a prop set had to repeat every support of the
structure, and editing the base's supports left it behind."""
import unittest

from context import Structure2D
from xdfem2d.models import (SupportSet, ConstructionPhase, ConstructionSequence)
from xdfem2d.phasing import solve_sequence, _effective_restraints
from xdfem2d.workflows import current_restraints


def _beam():
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    for nid, x in (("N1", 0.0), ("N2", 5.0), ("N3", 10.0)):
        s.add_node(nid, x, 0.0)
    s.add_bar_element("E1", "N1", "N2", "S"); s.add_bar_element("E2", "N2", "N3", "S")
    s.add_support("PIN", ux=True, uy=True); s.add_support("ROLLER", uy=True)
    s.assign_support("N1", "PIN"); s.assign_support("N3", "ROLLER")
    s.add_load_case("Q"); s.add_distributed_load("E1", "Q", fye=-10.0, fyd=-10.0)
    return s


PROP = lambda: SupportSet(id="PROP", restraints={"N2": (False, True, False)})          # noqa: E731


class TestSetOverlaysTheBase(unittest.TestCase):
    def test_a_prop_set_keeps_the_bases_supports(self):
        s = _beam()
        m = s.derive_variant(__import__("xdfem2d.models", fromlist=["Variant"]).Variant(id="p"), PROP())
        self.assertEqual(current_restraints(m), {"N1": (True, True, False), "N2": (False, True, False),
                                                 "N3": (False, True, False)})

    def test_nothing_ticked_takes_a_base_support_away(self):
        s = _beam()
        free = SupportSet(id="F", restraints={"N3": (False, False, False)})
        self.assertEqual(sorted(current_restraints(s.with_supports(free))), ["N1"])

    def test_a_listed_node_takes_the_sets_restraint(self):
        s = _beam()
        fix = SupportSet(id="X", restraints={"N1": (True, True, True)})
        self.assertEqual(current_restraints(s.with_supports(fix))["N1"], (True, True, True))
        self.assertEqual(current_restraints(s.with_supports(fix))["N3"], (False, True, False))

    def test_a_listed_spring_replaces_that_one_and_the_others_stay(self):
        from xdfem2d.models import NodeSpring
        s = _beam()
        s.add_node_spring("N1", ky=100.0); s.add_node_spring("N2", ky=200.0)
        m = s.with_supports(SupportSet(id="K", node_springs={"N2": NodeSpring(node_id="N2", ky=999.0)}))
        self.assertEqual(m.node_springs["N1"].ky, 100.0)
        self.assertEqual(m.node_springs["N2"].ky, 999.0)

    def test_the_effective_restraints_are_the_same_as_what_is_derived(self):
        s = _beam()
        s.add_support_set(PROP())
        self.assertEqual(_effective_restraints(s, "PROP", s.support_sets),
                         current_restraints(s.with_supports(PROP())))
        self.assertEqual(_effective_restraints(s, None, s.support_sets), current_restraints(s))
        free = SupportSet(id="F", restraints={"N3": (False, False, False)})
        s.add_support_set(free)
        self.assertEqual(sorted(_effective_restraints(s, "F", s.support_sets)), ["N1"])


class TestPhasesFollowTheBase(unittest.TestCase):
    def _seq(self, s):
        p1 = ConstructionPhase(id="P1", active_elements={"E1", "E2"}, applied_cases=["Q"])
        p2 = ConstructionPhase(id="P2", active_elements={"E1", "E2"}, applied_cases=["Q"],
                               support_set_ids=["PROP"])
        s.add_construction_sequence(ConstructionSequence(id="S", phases=[p1, p2]))
        return solve_sequence(s, s.construction_sequences["S"], s.support_sets)

    def test_a_support_the_base_gains_reaches_a_phase_that_uses_a_set(self):
        from xdfem2d.models import Variant
        s = _beam()
        s.add_support_set(PROP())
        s.add_support("RX", ux=True); s.assign_support("N3", "RX")          # the base changes N3 afterwards
        phase_model = s.derive_variant(Variant(id="p"), s.support_sets["PROP"])
        self.assertEqual(current_restraints(phase_model)["N3"], (True, False, False))
        self.assertEqual(current_restraints(phase_model)["N2"], (False, True, False))     # and the prop is there
        self._seq(s)                                                        # the sequence solves

    def test_a_set_that_lists_a_missing_node_is_refused_with_its_name(self):
        s = _beam()
        s.add_support_set(SupportSet(id="PROP", restraints={"N9": (False, True, False)}))
        with self.assertRaises(ValueError) as cm:
            self._seq(s)
        self.assertIn("N9", str(cm.exception))


if __name__ == "__main__":
    unittest.main()


# ── several sets in force, Remove by id, and what deleting a set leaves behind ──

from xdfem2d.models import Operation, OpAction, OpTarget                     # noqa: E402
from xdfem2d.phasing import resolve_phase_state                                # noqa: E402

A, R = OpAction.ADD, OpAction.REMOVE
SUP = OpTarget.SUPPORTS


def _op(i, action, ids, target=SUP):
    return Operation(id=i, action=action, target=target, group_kind="", ids=tuple(ids))


class TestSeveralSetsPerPhase(unittest.TestCase):
    def test_adding_a_second_set_keeps_the_first(self):
        ids = resolve_phase_state({"E1"}, [], [_op("a", A, ["P1"])], {"E1"})[1]
        ids = resolve_phase_state({"E1"}, ids, [_op("b", A, ["P2"])], {"E1"})[1]
        self.assertEqual(ids, ["P1", "P2"])

    def test_removing_takes_out_only_the_named_set(self):
        ids = resolve_phase_state({"E1"}, ["P1", "P2"], [_op("c", R, ["P1"])], {"E1"})[1]
        self.assertEqual(ids, ["P2"])

    def test_removing_a_set_that_is_not_in_force_changes_nothing(self):
        ids = resolve_phase_state({"E1"}, ["P1"], [_op("c", R, ["P9"])], {"E1"})[1]
        self.assertEqual(ids, ["P1"])

    def test_adding_one_already_there_does_not_repeat_it(self):
        ids = resolve_phase_state({"E1"}, ["P1"], [_op("a", A, ["P1"])], {"E1"})[1]
        self.assertEqual(ids, ["P1"])

    def test_two_props_are_both_in_the_phase_and_the_later_set_wins_a_shared_node(self):
        from xdfem2d.models import Variant
        s = _beam()
        s.add_support_set(SupportSet(id="P1", restraints={"N2": (False, True, False)}))
        s.add_support_set(SupportSet(id="P2", restraints={"N2": (True, True, False),
                                                           "N3": (False, False, False)}))
        m = s.derive_variant(Variant(id="p"), [s.support_sets["P1"], s.support_sets["P2"]])
        got = current_restraints(m)
        self.assertEqual(got["N2"], (True, True, False))       # the second set, over the first
        self.assertNotIn("N3", got)                            # and it freed the base's support at N3
        self.assertEqual(_effective_restraints(s, ["P1", "P2"], s.support_sets), got)

    def test_taking_one_of_two_props_away_releases_only_that_ones_reaction(self):
        s = _beam()
        s.add_load_case("Q2")
        s.add_distributed_load("E1", "Q2", fye=-10.0, fyd=-10.0)
        s.add_distributed_load("E2", "Q2", fye=-10.0, fyd=-10.0)
        s.add_support_set(SupportSet(id="P1", restraints={"N2": (False, True, False)}))
        s.add_support_set(SupportSet(id="P2", restraints={"N3": (False, True, False)}))
        E, L = OpTarget.ELEMENTS, OpTarget.LOADS
        built = _op("a", A, ["E1", "E2"], E)
        built.group_kind = "scene"
        ph = [ConstructionPhase(id="A", operations=[built, _op("b", A, ["Q"], L)]),
              ConstructionPhase(id="B", operations=[_op("c", A, ["P1", "P2"]), _op("d", A, ["Q2"], L)]),
              ConstructionPhase(id="C", operations=[_op("e", R, ["P1"])])]
        out = solve_sequence(s, ConstructionSequence(id="S", phases=ph), s.support_sets)
        rb, rc = out["phases"]["B"]["reactions"], out["phases"]["C"]["reactions"]
        self.assertGreater(abs(rb["N2"][1]), 1.0)                 # both props carried load in B
        self.assertGreater(abs(rb["N3"][1]), 1.0)
        self.assertAlmostEqual(rc.get("N2", [0, 0, 0])[1], 0.0, places=6)   # P1 gone: its reaction is released
        self.assertGreater(abs(rc["N3"][1]), 1.0)                          # P2 is still there and still carries


class TestDeletingOrRenamingASet(unittest.TestCase):
    def _with_a_sequence(self):
        s = _beam()
        s.add_support_set(SupportSet(id="PROP", restraints={"N2": (False, True, False)}))
        E, L = OpTarget.ELEMENTS, OpTarget.LOADS
        e = _op("a", A, ["E1", "E2"], E)
        e.group_kind = "scene"
        s.add_construction_sequence(ConstructionSequence(id="S", phases=[
            ConstructionPhase(id="P1", operations=[e, _op("b", A, ["Q"], L)]),
            ConstructionPhase(id="P2", operations=[_op("c", A, ["PROP"])]),
            ConstructionPhase(id="P3", operations=[_op("d", R, ["PROP"])])]))
        for p in s.construction_sequences["S"].phases:                    # the resolved cache, as the panel saves it
            p.support_set_ids = ["PROP"] if p.id == "P2" else []
        return s

    def test_deleting_the_set_takes_it_out_of_the_phases_and_their_operations(self):
        s = self._with_a_sequence()
        s.remove_support_set("PROP")
        ph = {p.id: p for p in s.construction_sequences["S"].phases}
        self.assertEqual(ph["P2"].support_set_ids, [])
        self.assertEqual([o.id for o in ph["P2"].operations], [])          # an operation that only named it has nothing left
        self.assertEqual([o.id for o in ph["P3"].operations], [])
        self.assertEqual(s.reference_problems(), [])
        solve_sequence(s, s.construction_sequences["S"], s.support_sets)    # and the sequence still solves

    def test_an_operation_that_names_a_missing_set_is_reported_before_the_run(self):
        s = self._with_a_sequence()
        del s.support_sets["PROP"]                                         # gone without the references cleaned
        problems = s.reference_problems()
        self.assertTrue(any("PROP" in p for p in problems), problems)

    def test_an_element_with_the_sets_name_is_not_mistaken_for_it(self):
        s = _beam()
        s.add_support_set(SupportSet(id="E1", restraints={"N2": (False, True, False)}))   # same id as a bar
        s.add_construction_sequence(ConstructionSequence(id="S", phases=[
            ConstructionPhase(id="P1", operations=[_op("a", A, ["E1", "E2"], OpTarget.ELEMENTS)])]))
        s.remove_support_set("E1")
        self.assertEqual(s.construction_sequences["S"].phases[0].operations[0].ids, ("E1", "E2"))

    def test_renaming_the_set_carries_the_phases_and_operations_along(self):
        s = self._with_a_sequence()
        s.construction_sequences["S"].phases[1].support_set_ids = ["PROP"]
        s.rename_support_set("PROP", "PUNTAL")
        self.assertIn("PUNTAL", s.support_sets)
        self.assertNotIn("PROP", s.support_sets)
        self.assertEqual(s.support_sets["PUNTAL"].id, "PUNTAL")
        p2 = s.construction_sequences["S"].phases[1]
        self.assertEqual(p2.support_set_ids, ["PUNTAL"])
        self.assertEqual(list(p2.operations[0].ids), ["PUNTAL"])
        solve_sequence(s, s.construction_sequences["S"], s.support_sets)
        s.add_support_set(SupportSet(id="X"))
        with self.assertRaises(ValueError):
            s.rename_support_set("X", "PUNTAL")                           # that name is taken
        with self.assertRaises(KeyError):
            s.rename_support_set("NOPE", "Y")

    def test_the_list_survives_a_save_and_open(self):
        import os, tempfile
        s = self._with_a_sequence()
        s.add_support_set(SupportSet(id="PROP2", restraints={"N3": (False, True, False)}))
        s.construction_sequences["S"].phases[1].support_set_ids = ["PROP", "PROP2"]
        p = os.path.join(tempfile.mkdtemp(), "m.json")
        s.save(p)
        r = Structure2D.load(p)
        self.assertEqual(r.construction_sequences["S"].phases[1].support_set_ids, ["PROP", "PROP2"])
