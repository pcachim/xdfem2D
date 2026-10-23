"""API / code-robustness tests.

These exercise the public ``Structure2D`` interface rather than the numerics:
input validation, DOF bookkeeping, save/load round-trips, rename/remove
cascades, and load-combination superposition. They guard against regressions in
the *code* (as opposed to the *calculation*).
"""
import json
import os
import tempfile
import unittest

from context import Structure2D, assert_close


def _simple_beam():
    """A reusable simply-supported beam with one load case."""
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=25.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    s.add_node("N1", 0.0, 0.0)
    s.add_node("N2", 5.0, 0.0)
    s.add_bar_element("E1", "N1", "N2", "S")
    s.add_support("PIN", ux=True, uy=True)
    s.add_support("ROLLER", ux=False, uy=True)
    s.assign_support("N1", "PIN")
    s.assign_support("N2", "ROLLER")
    s.add_load_case("LC")
    s.add_distributed_load("E1", "LC", fye=-10.0, fyd=-10.0)
    return s


class TestInputValidation(unittest.TestCase):
    def test_duplicate_node_raises(self):
        s = Structure2D()
        s.add_node("N1", 0.0, 0.0)
        with self.assertRaises(ValueError):
            s.add_node("N1", 1.0, 0.0)

    def test_duplicate_element_raises(self):
        s = _simple_beam()
        with self.assertRaises(ValueError):
            s.add_bar_element("E1", "N1", "N2", "S")

    def test_section_unknown_material_raises(self):
        s = Structure2D()
        with self.assertRaises(ValueError):
            s.add_section("S", "DOES_NOT_EXIST", b=0.3, h=0.6)

    def test_calculate_without_load_case_raises(self):
        s = Structure2D()
        s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
        s.add_section("S", "M", b=0.3, h=0.6)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", 5.0, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        with self.assertRaises(ValueError):
            s.calculate()

    def test_calculate_without_elements_raises(self):
        s = Structure2D()
        s.add_load_case("LC")
        with self.assertRaises(ValueError):
            s.calculate()

    def test_bad_combo_type_raises(self):
        s = _simple_beam()
        with self.assertRaises(ValueError):
            s.add_load_combination("CX", {"LC": 1.0}, combo_type="NopeType")

    def test_old_portuguese_combo_type_still_accepted(self):
        # combo_type was renamed from Portuguese to English (26/09/2026); the
        # old values are still accepted (and stored as their English
        # equivalent) so a script written before the rename keeps working.
        s = _simple_beam()
        c = s.add_load_combination("CX", {"LC": 1.0}, combo_type="SomaLinear")
        self.assertEqual(c.combo_type, "LinearSum")
        c2 = s.add_load_combination("CY", {"LC": 1.0}, combo_type="Envolvente")
        self.assertEqual(c2.combo_type, "Envelope")

    def test_bad_coord_sys_raises(self):
        s = _simple_beam()
        with self.assertRaises(ValueError):
            s.add_distributed_load("E1", "LC", fye=-1.0, coord_sys="diagonal")


class TestLoadCaseTwins(unittest.TestCase):
    """Every load case gets a twin Linear analysis case <id> (factor 1), so
    combinations always combine analysis cases."""

    def test_add_load_case_creates_twin(self):
        s = Structure2D()
        s.add_load_case("G")
        self.assertIn("G", s.analysis_cases_by_id)
        twin = s.analysis_cases_by_id["G"]
        self.assertEqual(twin.analysis_type, "Linear")
        self.assertEqual(twin.coefficients, {"G": 1.0})

    def test_remove_load_case_removes_twin(self):
        s = Structure2D()
        s.add_load_case("G")
        s.remove_load_case("G")
        self.assertNotIn("G", s.analysis_cases_by_id)

    def test_rename_load_case_renames_twin(self):
        s = Structure2D()
        s.add_load_case("G")
        s.rename_load_case("G", "DEAD")
        self.assertNotIn("G", s.analysis_cases_by_id)
        self.assertIn("DEAD", s.analysis_cases_by_id)
        self.assertEqual(s.analysis_cases_by_id["DEAD"].coefficients, {"DEAD": 1.0})

    def test_no_duplicate_twin(self):
        s = Structure2D()
        s.add_load_case("G")
        n = len(s.analysis_cases)
        # twin_analysis_case_id is idempotent
        self.assertEqual(s.twin_analysis_case_id("G"), "G")
        self.assertEqual(len(s.analysis_cases), n)

    def test_save_load_preserves_twins_and_combo(self):
        s = Structure2D()
        s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
        s.add_section("S", "M", b=0.3, h=0.6)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", 5.0, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_load_case("G")
        s.add_load_case("Q")
        s.add_load_combination("ULS", {"G": 1.35, "Q": 1.5})
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "m.json")
            s.save(path)
            s2 = Structure2D.load(path)
        # twins restored, not duplicated
        self.assertEqual(sorted(a.id for a in s2.analysis_cases), ["G", "Q"])
        self.assertEqual(s2.load_combinations[0].coefficients,
                         {"G": 1.35, "Q": 1.5})

    def test_legacy_combo_normalised_to_twin(self):
        # A legacy combination referencing a load case is mapped to its twin.
        from xdfem2d.models import LoadCombination
        s = Structure2D()
        s.add_load_case("G")
        s.load_combinations.append(
            LoadCombination(id="C", coefficients={"G": 1.35}, combo_type="LinearSum"))
        s.normalize_combinations_to_analysis_cases()
        self.assertEqual(s.load_combinations[0].coefficients, {"G": 1.35})


class TestCombinationOverAnalysisCases(unittest.TestCase):
    """Combinations reference analysis cases (the twins); results and diagrams
    must match the equivalent direct load-case combination."""

    def _beam(self):
        s = Structure2D()
        s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
        s.add_section("S", "M", b=0.3, h=0.6)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", 5.0, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_support("PIN", ux=True, uy=True)
        s.add_support("ROLLER", ux=False, uy=True)
        s.assign_support("N1", "PIN")
        s.assign_support("N2", "ROLLER")
        s.add_load_case("G")
        s.add_load_case("Q")
        s.add_distributed_load("E1", "G", fye=-10.0, fyd=-10.0)
        s.add_point_load("N2", "Q", fy=-8.0)
        return s

    def test_combo_over_twins_reaction(self):
        s = self._beam()
        s.add_load_combination("ULS", {"G": 1.35, "Q": 1.5},
                               combo_type="LinearSum")
        r = s.calculate()
        rg = r["reactions"]["G"]["N2"][1]
        rq = r["reactions"]["Q"]["N2"][1]
        got = r["combinations"]["ULS"]["reactions"]["N2"][1]
        assert_close(self, got, 1.35 * rg + 1.5 * rq, msg="combo over twins")

    def test_combo_over_twins_has_diagram(self):
        s = self._beam()
        s.add_load_combination("ULS", {"G": 1.35, "Q": 1.5},
                               combo_type="LinearSum")
        r = s.calculate()
        dist = r["combo_distribution"]["ULS"]["E1"]
        self.assertTrue(any(abs(v) > 1e-9 for v in dist["M"]))


class TestSectionOverrides(unittest.TestCase):
    """area_override / inertia_override replace the computed section properties."""

    def test_overrides_take_effect(self):
        s = Structure2D()
        s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
        sec = s.add_section("S", "M", b=0.3, h=0.6,
                            area_override=1.0, inertia_override=2.0)
        self.assertAlmostEqual(sec.area, 1.0)
        self.assertAlmostEqual(sec.inertia, 2.0)

    def test_inertia_override_changes_deflection(self):
        """Doubling I should halve the cantilever tip deflection (δ ∝ 1/I)."""
        def tip(inertia_override):
            s = Structure2D()
            s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
            s.add_section("S", "M", b=0.3, h=0.6, inertia_override=inertia_override)
            s.add_node("N1", 0.0, 0.0)
            s.add_node("N2", 4.0, 0.0)
            s.add_bar_element("E1", "N1", "N2", "S")
            s.add_support("FIX", ux=True, uy=True, tz=True)
            s.assign_support("N1", "FIX")
            s.add_load_case("LC")
            s.add_point_load("N2", "LC", fy=-10.0)
            return abs(s.calculate()["displacements"]["LC"]["N2"][1])
        assert_close(self, tip(0.01), 2.0 * tip(0.02), rel=1e-9,
                     msg="tip deflection inversely proportional to I")


class TestDofBookkeeping(unittest.TestCase):
    def test_num_dofs(self):
        s = _simple_beam()
        self.assertEqual(s.num_dofs, 3 * len(s.nodes))

    def test_dof_index_invalidated_on_add_node(self):
        s = _simple_beam()
        n0 = s.num_dofs
        s.add_node("N3", 10.0, 0.0)
        self.assertEqual(s.num_dofs, n0 + 3)


class TestRemoveCascade(unittest.TestCase):
    def test_remove_node_clears_references(self):
        s = _simple_beam()
        s.add_point_load("N2", "LC", fy=-5.0)
        s.remove_node("N2")
        self.assertNotIn("N2", s.nodes)
        self.assertFalse(any(pl.node_id == "N2" for pl in s.point_loads))
        self.assertFalse(any(a.node_id == "N2" for a in s.support_assignments))

    def test_remove_element_clears_loads(self):
        s = _simple_beam()
        s.remove_element("E1")
        self.assertNotIn("E1", s.bar_elements_by_id)
        self.assertFalse(any(dl.element_id == "E1" for dl in s.distributed_loads))

    def test_remove_missing_node_raises(self):
        s = _simple_beam()
        with self.assertRaises(KeyError):
            s.remove_node("NOPE")


class TestRename(unittest.TestCase):
    def test_rename_node_updates_elements_and_loads(self):
        s = _simple_beam()
        s.add_point_load("N2", "LC", fy=-5.0)
        s.rename_node("N2", "END")
        self.assertIn("END", s.nodes)
        self.assertNotIn("N2", s.nodes)
        self.assertTrue(any(e.node_j == "END" for e in s.bar_elements))
        self.assertTrue(any(pl.node_id == "END" for pl in s.point_loads))

    def test_rename_to_existing_raises(self):
        s = _simple_beam()
        with self.assertRaises(ValueError):
            s.rename_node("N1", "N2")

    def test_results_invariant_under_rename(self):
        """Renaming a node must not change the computed answer."""
        s1 = _simple_beam()
        r1 = s1.calculate()
        ry1 = r1["reactions"]["LC"]["N1"][1]

        s2 = _simple_beam()
        s2.rename_node("N1", "LEFT")
        r2 = s2.calculate()
        ry2 = r2["reactions"]["LC"]["LEFT"][1]
        assert_close(self, ry2, ry1, msg="reaction unchanged after rename")


class TestSaveLoad(unittest.TestCase):
    def test_round_trip_preserves_results(self):
        s = _simple_beam()
        r_before = s.calculate()

        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "beam.json")
            s.save(path)
            self.assertTrue(os.path.isfile(path))
            s2 = Structure2D.load(path)

        # Topology preserved
        self.assertEqual(set(s2.nodes), set(s.nodes))
        self.assertEqual(len(s2.bar_elements), len(s.bar_elements))

        # Numerics preserved
        r_after = s2.calculate()
        for nid in s.nodes:
            for k in range(3):
                assert_close(
                    self,
                    r_after["reactions"]["LC"][nid][k],
                    r_before["reactions"]["LC"][nid][k],
                    abs_tol=1e-9,
                    msg=f"reaction[{nid}][{k}] after reload",
                )

    def test_loading_a_pre_rename_file_translates_combo_type(self):
        # combo_type was renamed from Portuguese to English (26/09/2026). A
        # .x2d/.json written before that still has the old value on disk;
        # loading it must translate it, and saving it again must write the
        # new English value -- not silently keep re-saving Portuguese.
        s = _simple_beam()
        s.add_load_combination("CX", {"LC": 1.0}, combo_type="LinearSum")
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "beam.json")
            s.save(path)
            with open(path) as fh:
                data = json.load(fh)
            # Simulate a file saved by an older version of the engine.
            for lc in data["load_combinations"]:
                if lc["id"] == "CX":
                    lc["combo_type"] = "SomaLinear"
            with open(path, "w") as fh:
                json.dump(data, fh)

            s2 = Structure2D.load(path)
            combo = next(c for c in s2.load_combinations if c.id == "CX")
            self.assertEqual(combo.combo_type, "LinearSum")

            # Re-saving must persist the translated (English) value.
            path2 = os.path.join(d, "beam2.json")
            s2.save(path2)
            with open(path2) as fh:
                data2 = json.load(fh)
            saved_ctype = next(lc["combo_type"] for lc in data2["load_combinations"]
                               if lc["id"] == "CX")
            self.assertEqual(saved_ctype, "LinearSum")


class TestCombinationSuperposition(unittest.TestCase):
    """A LinearSum (1.35·G + 1.5·Q) combo must equal the weighted case sum."""

    def setUp(self):
        s = Structure2D()
        s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
        s.add_section("S", "M", b=0.3, h=0.6)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", 5.0, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_support("PIN", ux=True, uy=True)
        s.add_support("ROLLER", ux=False, uy=True)
        s.assign_support("N1", "PIN")
        s.assign_support("N2", "ROLLER")
        s.add_load_case("G")
        s.add_load_case("Q")
        s.add_distributed_load("E1", "G", fye=-10.0, fyd=-10.0)
        s.add_distributed_load("E1", "Q", fye=-4.0, fyd=-4.0)
        s.add_load_combination("ULS", {"G": 1.35, "Q": 1.5}, combo_type="LinearSum")
        self.r = s.calculate()

    def test_reaction_is_weighted_sum(self):
        rg = self.r["reactions"]["G"]["N1"][1]
        rq = self.r["reactions"]["Q"]["N1"][1]
        expected = 1.35 * rg + 1.5 * rq
        got = self.r["combinations"]["ULS"]["reactions"]["N1"][1]
        assert_close(self, got, expected, msg="ULS reaction superposition")


if __name__ == "__main__":
    unittest.main(verbosity=2)
