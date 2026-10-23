"""validate_for_analysis: the Run preconditions, shared by the GUI and any
headless caller (xdfem2d.workflows)."""
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)
from xdfem2d.workflows import validate_for_analysis


def _codes(struc):
    return [p['code'] for p in validate_for_analysis(struc)]


class TestValidateForAnalysis(unittest.TestCase):
    def test_empty_model_reports_everything_in_gui_order(self):
        self.assertEqual(
            _codes(Structure2D()),
            ['no_nodes', 'no_elements', 'no_supports',
             'no_load_cases', 'no_analysis_cases'])

    def test_every_problem_has_a_message(self):
        for p in validate_for_analysis(Structure2D()):
            self.assertTrue(p['msg'])

    def test_nodes_clear_only_their_own_problem(self):
        s = Structure2D()
        s.add_node('A', 0.0, 0.0)
        s.add_node('B', 5.0, 0.0)
        self.assertEqual(
            _codes(s),
            ['no_elements', 'no_supports', 'no_load_cases',
             'no_analysis_cases'])

    def test_load_case_clears_its_problem(self):
        s = Structure2D()
        s.add_load_case('G')
        self.assertNotIn('no_load_cases', _codes(s))


if __name__ == '__main__':
    unittest.main()


def _beam():
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
    s.add_distributed_load("E1", "G", fye=-10.0, fyd=-10.0)
    return s


class TestRunAnalysis(unittest.TestCase):
    def test_complete_model_has_no_problems_once_it_has_a_case(self):
        s = _beam()
        s.create_analysis_case()
        self.assertEqual(validate_for_analysis(s), [])

    def test_matches_calculate(self):
        from xdfem2d.workflows import run_analysis
        s = _beam()
        out = run_analysis(s)
        ref = _beam().calculate()
        self.assertEqual(out.results["reactions"]["G"]["N2"][1],
                         ref["reactions"]["G"]["N2"][1])
        self.assertIs(out.rcache["base"], out.results)
        self.assertFalse(out.incremental)
        self.assertEqual(out.sequence_errors, {})

    def test_scope_all_without_variants_is_the_base_only(self):
        from xdfem2d.workflows import run_analysis
        out = run_analysis(_beam(), scope="all")
        self.assertEqual(list(out.rcache), ["base"])
        self.assertEqual(out.variant_errors, {})
        self.assertIn("combinations", out.results)

    def test_solve_failure_raises(self):
        from xdfem2d.workflows import run_analysis
        with self.assertRaises(ValueError):
            run_analysis(Structure2D())
