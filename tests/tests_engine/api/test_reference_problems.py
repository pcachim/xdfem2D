"""Dangling-reference detection (Structure2D.reference_problems / check_references).

The failure these guard against is the quiet one: an analysis case whose
coefficient names a load case that does not exist is not an error to the solver,
it is a sum with a missing term, and every result comes out zero. A four-span
beam whose ULS case combined a load case 'G' that had been named 'ULS' ran
without complaint and drew nothing. So the reference check has to see the case
references, not only the material/section ones it began with.
"""
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)


def _beam():
    s = Structure2D()
    s.add_material('steel', 210e6, 78.5, poisson=0.3)
    s.add_section('sec', 'steel', 0.1, 0.2,
                  area_override=0.01, inertia_override=1e-4)
    s.add_node('N1', 0, 0)
    s.add_node('N2', 5, 0)
    s.add_bar_element('E1', 'N1', 'N2', 'sec')
    s.add_support('fix', True, True, True)
    s.assign_support('N1', 'fix')
    s.add_support('roll', False, True, False)
    s.assign_support('N2', 'roll')
    s.add_load_case('G')
    s.add_distributed_load('E1', 'G', fye=-10, fyd=-10)
    return s


class TestReferenceProblems(unittest.TestCase):
    def test_clean_model_has_no_problems(self):
        s = _beam()
        s.add_analysis_case('ULS', 'Linear', {'G': 1.35})
        self.assertEqual(s.reference_problems(), [])
        s.calculate()   # must not raise

    def test_analysis_case_missing_load_case(self):
        """The exact fault in the reported four-span-beam file."""
        s = _beam()
        s.add_analysis_case('ULS', 'Linear', {'MISSING': 1.35})
        probs = s.reference_problems()
        self.assertTrue(any("analysis case 'ULS'" in p and "'MISSING'" in p
                            for p in probs), probs)
        with self.assertRaises(ValueError):
            s.calculate()

    def test_analysis_case_may_reference_another_case(self):
        """Nesting is legal: a coefficient key that is an analysis case, not a
        load case, is resolved through — so it is not a dangling reference."""
        s = _beam()
        s.add_analysis_case('G_case', 'Linear', {'G': 1.0})
        s.add_analysis_case('ULS', 'Linear', {'G_case': 1.35})
        self.assertEqual(s.reference_problems(), [])

    def test_combination_missing_analysis_case(self):
        """add_load_combination() now rejects a dangling reference immediately
        (see the "massive bug correction at load combinations" fix: silently
        accepting one let a combination quietly resolve through an unfactored
        load case instead of its analysis-case twin)."""
        s = _beam()
        s.add_analysis_case('ULS', 'Linear', {'G': 1.35})
        with self.assertRaises(ValueError):
            s.add_load_combination('C1', {'NOPE': 1.0})

    def test_combination_missing_analysis_case_after_deletion(self):
        """reference_problems() still has a job: it must catch a dangling
        combination reference that add_load_combination()'s own up-front
        validation can no longer produce — e.g. a legacy/hand-edited file
        loaded with a stale coefficient key. (remove_analysis_case() itself
        pops the reference from every combination's coefficients, so it
        cannot be used here to reach the dangling state — the mutation has
        to be simulated directly, the way a corrupted load would.)"""
        s = _beam()
        s.add_analysis_case('ULS', 'Linear', {'G': 1.35})
        combo = s.add_load_combination('C1', {'ULS': 1.0})
        combo.coefficients['GHOST'] = 1.0
        self.assertTrue(any("combination 'C1'" in p for p in
                            s.reference_problems()))

    def test_load_on_missing_load_case(self):
        s = _beam()
        s.add_distributed_load('E1', 'NOSUCH', fye=-5, fyd=-5)
        s.add_analysis_case('ULS', 'Linear', {'G': 1.35})
        self.assertTrue(any("'NOSUCH'" in p for p in s.reference_problems()))

    def test_load_on_missing_element(self):
        # add_distributed_load itself now rejects an unknown element id
        # eagerly (Structure2D._require_exists), so the dangling reference
        # this guards against -- a load left behind after its element was
        # removed by some other path than remove_element (which prunes
        # distributed_loads itself) -- has to be reproduced by hand here.
        s = _beam()
        s.add_distributed_load('E1', 'G', fye=-5, fyd=-5)
        s.distributed_loads[-1].element_id = 'NOELEM'
        s.add_analysis_case('ULS', 'Linear', {'G': 1.35})
        self.assertTrue(any("element 'NOELEM'" in p
                            for p in s.reference_problems()))

    def test_support_on_missing_node(self):
        s = _beam()
        s.assign_support('NONODE', 'fix')
        s.add_analysis_case('ULS', 'Linear', {'G': 1.35})
        self.assertTrue(any("node 'NONODE'" in p
                            for p in s.reference_problems()))

    def test_spectrum_missing_function(self):
        s = _beam()
        s.add_analysis_case('MASS', 'Mass', {'G': 1.0})
        s.add_analysis_case('MOD', 'Modal', modal_case_id='MASS')
        s.add_analysis_case('SP', 'Spectrum', modal_case_id='MOD',
                            spectrum_id='NOSPEC')
        self.assertTrue(any("spectral function 'NOSPEC'" in p
                            for p in s.reference_problems()))

    def test_material_reference_still_caught(self):
        s = _beam()
        s.add_analysis_case('ULS', 'Linear', {'G': 1.35})
        s.sections['sec'].material_name = 'ghost'
        self.assertTrue(any("material 'ghost'" in p
                            for p in s.reference_problems()))
        with self.assertRaises(ValueError):
            s.calculate()


if __name__ == '__main__':
    unittest.main(verbosity=2)
