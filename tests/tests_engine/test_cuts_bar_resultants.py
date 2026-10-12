"""Phase-3 Cut tests: bar resultants (interpolation, rotation to cut axes,
moment transfer to the reference point) for plane and plate cuts, and for
LinearSum combinations. No Qt is involved, and none of this calls
``calculate()`` — the diagrams are fabricated directly in the shape the
solver itself produces (``results['analysis_cases'][case]
['element_distribution']``), so the tests exercise the new geometry/algebra
in ``cuts.py`` in isolation from the solver. See dev/CUT_PLAN.md."""
import unittest

from context import Structure2D
from xdfem2d.cuts import cut_bar_resultant


def _plane_bar():
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    s.add_node("N1", 0.0, 0.0)
    s.add_node("N2", 4.0, 0.0)
    s.add_bar_element("E1", "N1", "N2", "S")
    return s


def _plate_bar():
    s = Structure2D(domain="plate")
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    s.add_node("N1", 0.0, 0.0)
    s.add_node("N2", 4.0, 0.0)
    s.add_bar_element("E1", "N1", "N2", "S")
    return s


class TestPlaneBarResultant(unittest.TestCase):
    """Hand-computed against the formulas in cuts.py's own module notes:
    N=5, V=10 constant, M linear 0->40 over the 4 m bar; vertical cut at
    midspan; bar direction (1,0) points to the cut's *negative* side
    (n=(-1,0)), so the stored diagram is negated before use."""

    def setUp(self):
        self.struc = _plane_bar()
        self.cut = self.struc.add_cut("C1", 2.0, -1.0, 2.0, 1.0)
        self.results = {
            'analysis_cases': {
                'LC1': {'element_distribution': {
                    'E1': {'x': [0.0, 4.0], 'N': [5.0, 5.0],
                           'V': [10.0, 10.0], 'M': [0.0, 40.0]},
                }},
            },
        }

    def test_resultant_at_midpoint_reference(self):
        r = cut_bar_resultant(self.struc, self.results, self.cut, 'LC1')
        self.assertEqual(r['reason'], '')
        self.assertEqual(len(r['bars']), 1)
        res = r['resultant']
        self.assertAlmostEqual(res['N'], 5.0)
        self.assertAlmostEqual(res['V'], -10.0)
        self.assertAlmostEqual(res['M'], -20.0)
        # The single bar's own contribution equals the total.
        self.assertEqual(r['bars'][0]['forces_cut'], res)
        self.assertEqual(r['bars'][0]['elem_id'], 'E1')
        self.assertAlmostEqual(r['bars'][0]['a'], 2.0)
        self.assertAlmostEqual(r['bars'][0]['s'], 0.5)

    def test_resultant_with_offset_reference_point(self):
        r = cut_bar_resultant(self.struc, self.results, self.cut, 'LC1',
                               ref_point=(2.0, 5.0))
        res = r['resultant']
        # N, V unaffected by moving the reference point (pure rotation).
        self.assertAlmostEqual(res['N'], 5.0)
        self.assertAlmostEqual(res['V'], -10.0)
        # M picks up the moment of the transferred force: -25.
        self.assertAlmostEqual(res['M'], -45.0)

    def test_reversing_the_cut(self):
        # Same physical line, opposite drawing direction: t and n both flip,
        # and so does which side counts as "positive" (n=(-1,0) for C1 means
        # positive=i-side; n=(1,0) for C2 means positive=j-side). N and V are
        # *projections* onto (t, n): the underlying force vector and the
        # basis it is measured against both flip, so the numbers come out
        # the same — this is the correct, physically consistent outcome
        # (verified by hand against the diagram's own i/j convention), not
        # an accident. M is different: it is an absolute (global-frame)
        # moment about the shared reference point (both cuts share the same
        # midpoint here), so swapping which side is "positive" (i onto j vs
        # j onto i, Newton's third law about the same point) does flip it.
        rev_cut = self.struc.add_cut("C2", 2.0, 1.0, 2.0, -1.0)
        r1 = cut_bar_resultant(self.struc, self.results, self.cut, 'LC1')
        r2 = cut_bar_resultant(self.struc, self.results, rev_cut, 'LC1')
        self.assertAlmostEqual(r2['resultant']['N'], r1['resultant']['N'])
        self.assertAlmostEqual(r2['resultant']['V'], r1['resultant']['V'])
        self.assertAlmostEqual(r2['resultant']['M'], -r1['resultant']['M'])

    def test_missing_case_reports_a_reason(self):
        r = cut_bar_resultant(self.struc, self.results, self.cut, 'NOPE')
        self.assertNotEqual(r['reason'], '')
        self.assertEqual(r['bars'], [])
        self.assertEqual(r['resultant'], {'N': 0.0, 'V': 0.0, 'M': 0.0})

    def test_cut_missing_all_bars_is_empty_but_not_an_error(self):
        far_cut = self.struc.add_cut("C3", 20.0, -1.0, 20.0, 1.0)
        r = cut_bar_resultant(self.struc, self.results, far_cut, 'LC1')
        self.assertEqual(r['reason'], '')
        self.assertEqual(r['bars'], [])
        self.assertEqual(r['resultant'], {'N': 0.0, 'V': 0.0, 'M': 0.0})


class TestPlaneCombination(unittest.TestCase):
    def test_soma_linear_is_the_weighted_sum(self):
        struc = _plane_bar()
        cut = struc.add_cut("C1", 2.0, -1.0, 2.0, 1.0)
        results = {
            'analysis_cases': {
                'A': {'element_distribution': {
                    'E1': {'x': [0.0, 4.0], 'N': [5.0, 5.0],
                           'V': [10.0, 10.0], 'M': [0.0, 40.0]},
                }},
                'B': {'element_distribution': {
                    'E1': {'x': [0.0, 4.0], 'N': [2.0, 2.0],
                           'V': [-4.0, -4.0], 'M': [0.0, -8.0]},
                }},
            },
        }
        # A combination may only reference analysis cases (or other
        # combinations) — 'A'/'B' here are fabricated result-only ids, so
        # register them as (dummy) analysis cases before combining them.
        struc.add_analysis_case('A')
        struc.add_analysis_case('B')
        struc.add_load_combination('COMBO', {'A': 1.5, 'B': -0.5})

        ra = cut_bar_resultant(struc, results, cut, 'A')['resultant']
        rb = cut_bar_resultant(struc, results, cut, 'B')['resultant']
        rc = cut_bar_resultant(struc, results, cut, 'COMBO')['resultant']

        for k in ('N', 'V', 'M'):
            expected = 1.5 * ra[k] - 0.5 * rb[k]
            self.assertAlmostEqual(rc[k], expected)

    def test_envelope_combination_reports_a_reason(self):
        struc = _plane_bar()
        cut = struc.add_cut("C1", 2.0, -1.0, 2.0, 1.0)
        results = {'analysis_cases': {'A': {'element_distribution': {
            'E1': {'x': [0.0, 4.0], 'N': [5.0, 5.0],
                   'V': [10.0, 10.0], 'M': [0.0, 40.0]},
        }}}}
        struc.add_analysis_case('A')
        struc.add_load_combination('ENV', {'A': 1.0}, combo_type='Envelope')
        r = cut_bar_resultant(struc, results, cut, 'ENV')
        self.assertNotEqual(r['reason'], '')


class TestPlateBarResultant(unittest.TestCase):
    """Same hand-computed cut as the plane case, but the bar carries a
    grillage triple (T in the N slot, transverse V, bending M)."""

    def setUp(self):
        self.struc = _plate_bar()
        self.cut = self.struc.add_cut("C1", 2.0, -1.0, 2.0, 1.0)
        self.results = {
            'analysis_cases': {
                'LC1': {'element_distribution': {
                    'E1': {'x': [0.0, 4.0], 'N': [6.0, 6.0],
                           'V': [3.0, 3.0], 'M': [0.0, 20.0]},
                }},
            },
        }

    def test_resultant_at_midpoint_reference(self):
        r = cut_bar_resultant(self.struc, self.results, self.cut, 'LC1')
        res = r['resultant']
        self.assertAlmostEqual(res['V'], -3.0)
        self.assertAlmostEqual(res['Mb'], -10.0)
        self.assertAlmostEqual(res['Mt'], 6.0)

    def test_shear_transfer_changes_only_mt_for_a_y_offset(self):
        r = cut_bar_resultant(self.struc, self.results, self.cut, 'LC1',
                               ref_point=(2.0, 5.0))
        res = r['resultant']
        self.assertAlmostEqual(res['V'], -3.0)
        self.assertAlmostEqual(res['Mb'], -10.0)   # unaffected by this offset
        self.assertAlmostEqual(res['Mt'], -9.0)    # picks up the V transfer

    def test_missing_case_zero_resultant_has_plate_keys(self):
        r = cut_bar_resultant(self.struc, self.results, self.cut, 'NOPE')
        self.assertEqual(r['resultant'], {'V': 0.0, 'Mb': 0.0, 'Mt': 0.0})


if __name__ == "__main__":
    unittest.main()
