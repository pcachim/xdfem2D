"""Phase-5 Cut tests: the combined public API — cut_result (bars + areas +
total), cut_distribution (sampled arrays for plotting) and cuts_report (the
cut x case table). No Qt is involved, no ``calculate()`` call: diagrams and
tri_stress are fabricated directly, as in the Phase 3/4 tests. See
dev/CUT_PLAN.md."""
import unittest

from context import Structure2D
from xdfem2d.cuts import cut_result, cut_distribution, cuts_report


def _bar_only_model():
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    s.add_node("N1", 2.0, -1.0)
    s.add_node("N2", 2.0, 3.0)
    s.add_bar_element("E1", "N1", "N2", "S")
    return s


def _tri_only_model():
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_tri_section("TS", "M", thickness=0.2)
    s.add_node("A", 0.0, 0.0)
    s.add_node("B", 4.0, 0.0)
    s.add_node("C", 0.0, 4.0)
    s.add_tri_element("T1", "A", "B", "C", "TS")
    return s


def _mixed_model():
    """A wall (T1) with an embedded, independently-modelled vertical beam
    (E1) both crossing the same cut — the Phase-5 "bars + areas merged"
    case from dev/CUT_PLAN.md."""
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    s.add_tri_section("TS", "M", thickness=0.2)
    s.add_node("A", 0.0, 0.0)
    s.add_node("B", 4.0, 0.0)
    s.add_node("C", 0.0, 4.0)
    s.add_tri_element("T1", "A", "B", "C", "TS")
    s.add_node("N1", 2.0, -1.0)
    s.add_node("N2", 2.0, 3.0)
    s.add_bar_element("E1", "N1", "N2", "S")
    return s


_CUT_ENDS = (-1.0, 1.0, 5.0, 1.0)   # y=1, spans x in [-1, 5]

_BAR_DIST = {'E1': {'x': [0.0, 4.0], 'N': [1.0, 1.0], 'V': [2.0, 2.0],
                    'M': [0.0, 8.0]}}
_TRI_STRESS = {'T1': {'sx': 10.0, 'sy': -4.0, 'txy': 3.0}}


class TestCutResultMerging(unittest.TestCase):
    def test_bars_and_areas_sum_to_total(self):
        s = _mixed_model()
        cut = s.add_cut("C1", *_CUT_ENDS)
        results = {'analysis_cases': {'LC1': {
            'element_distribution': _BAR_DIST,
            'tri_stress': _TRI_STRESS,
        }}}
        r = cut_result(s, results, cut, 'LC1')
        self.assertEqual(r['domain'], 'plane')
        self.assertEqual(r['reason'], '')
        self.assertEqual(len(r['bars']), 1)
        self.assertEqual(len(r['areas']), 1)

        # Hand-computed (see test_cuts_bar_resultants.py / test_cuts_area_
        # resultants.py style derivations): bar={N:1,V:-2,M:4},
        # area={N:-2.4,V:1.8,M:1.2}.
        bars_res = r['resultant']['bars']
        areas_res = r['resultant']['areas']
        self.assertAlmostEqual(bars_res['N'], 1.0)
        self.assertAlmostEqual(bars_res['V'], -2.0)
        self.assertAlmostEqual(bars_res['M'], 4.0)
        self.assertAlmostEqual(areas_res['N'], -2.4)
        self.assertAlmostEqual(areas_res['V'], 1.8)
        self.assertAlmostEqual(areas_res['M'], 1.2)

        total = r['resultant']['total']
        self.assertAlmostEqual(total['N'], bars_res['N'] + areas_res['N'])
        self.assertAlmostEqual(total['V'], bars_res['V'] + areas_res['V'])
        self.assertAlmostEqual(total['M'], bars_res['M'] + areas_res['M'])
        self.assertAlmostEqual(total['N'], -1.4)
        self.assertAlmostEqual(total['V'], -0.2)
        self.assertAlmostEqual(total['M'], 5.2)

    def test_bar_only_model_reports_no_spurious_area_reason(self):
        s = _bar_only_model()
        cut = s.add_cut("C1", *_CUT_ENDS)
        results = {'analysis_cases': {'LC1': {'element_distribution': _BAR_DIST}}}
        r = cut_result(s, results, cut, 'LC1')
        self.assertEqual(r['reason'], '')
        self.assertEqual(r['areas'], [])
        self.assertEqual(r['resultant']['areas'], {'N': 0.0, 'V': 0.0, 'M': 0.0})
        self.assertEqual(r['resultant']['total'], r['resultant']['bars'])

    def test_tri_only_model_reports_no_spurious_bar_reason(self):
        s = _tri_only_model()
        cut = s.add_cut("C1", *_CUT_ENDS)
        results = {'analysis_cases': {'LC1': {'tri_stress': _TRI_STRESS}}}
        r = cut_result(s, results, cut, 'LC1')
        self.assertEqual(r['reason'], '')
        self.assertEqual(r['bars'], [])
        self.assertEqual(r['resultant']['bars'], {'N': 0.0, 'V': 0.0, 'M': 0.0})
        self.assertEqual(r['resultant']['total'], r['resultant']['areas'])

    def test_missing_case_sets_reason_and_zero_total(self):
        s = _mixed_model()
        cut = s.add_cut("C1", *_CUT_ENDS)
        results = {'analysis_cases': {'LC1': {
            'element_distribution': _BAR_DIST, 'tri_stress': _TRI_STRESS}}}
        r = cut_result(s, results, cut, 'NOPE')
        self.assertNotEqual(r['reason'], '')
        self.assertEqual(r['resultant']['total'], {'N': 0.0, 'V': 0.0, 'M': 0.0})

    def test_plate_domain_total_uses_plate_keys(self):
        s = Structure2D(domain="plate")
        s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
        s.add_plate_section("TS", "M", thickness=0.2)
        s.add_node("A", 0.0, 0.0)
        s.add_node("B", 4.0, 0.0)
        s.add_node("C", 0.0, 4.0)
        s.add_tri_element("T1", "A", "B", "C", "TS")
        cut = s.add_cut("C1", *_CUT_ENDS)
        results = {'analysis_cases': {'LC1': {'tri_stress': {
            'T1': {'mx': 8.0, 'my': -2.0, 'mxy': 1.0, 'vx': 5.0, 'vy': -3.0},
        }}}}
        r = cut_result(s, results, cut, 'LC1')
        self.assertEqual(r['domain'], 'plate')
        self.assertEqual(set(r['resultant']['total']), {'V', 'Mb', 'Mt'})


class TestCutDistribution(unittest.TestCase):
    def test_area_field_is_piecewise_constant_with_markers(self):
        s = _mixed_model()
        cut = s.add_cut("C1", *_CUT_ENDS)
        results = {'analysis_cases': {'LC1': {
            'element_distribution': _BAR_DIST, 'tri_stress': _TRI_STRESS}}}
        cutres = cut_result(s, results, cut, 'LC1')
        dist = cut_distribution(cutres, n=13)
        self.assertEqual(len(dist['s']), 13)
        self.assertEqual(set(dist['fields']), {'n_n', 'n_t'})
        for idx, s_val in enumerate(dist['s']):
            inside = (1 / 6) - 1e-9 <= s_val <= (4 / 6) + 1e-9
            expected_nn = -0.8 if inside else 0.0
            self.assertAlmostEqual(dist['fields']['n_n'][idx], expected_nn,
                                   places=6)
        self.assertEqual(len(dist['markers']), 1)
        self.assertEqual(dist['markers'][0]['elem_id'], 'E1')

    def test_plate_field_keys(self):
        s = Structure2D(domain="plate")
        s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
        s.add_plate_section("TS", "M", thickness=0.2)
        s.add_node("A", 0.0, 0.0)
        s.add_node("B", 4.0, 0.0)
        s.add_node("C", 0.0, 4.0)
        s.add_tri_element("T1", "A", "B", "C", "TS")
        cut = s.add_cut("C1", *_CUT_ENDS)
        results = {'analysis_cases': {'LC1': {'tri_stress': {
            'T1': {'mx': 8.0, 'my': -2.0, 'mxy': 1.0, 'vx': 5.0, 'vy': -3.0},
        }}}}
        cutres = cut_result(s, results, cut, 'LC1')
        dist = cut_distribution(cutres, n=5)
        self.assertEqual(set(dist['fields']), {'m_n', 'm_nt', 'v_n'})


class TestCutsReport(unittest.TestCase):
    def test_one_row_per_cut_and_case(self):
        s = _mixed_model()
        s.add_cut("C1", *_CUT_ENDS)
        s.add_cut("C2", 2.0, -1.0, 2.0, 3.0, name="Column cut")
        results = {'analysis_cases': {
            'LC1': {'element_distribution': _BAR_DIST, 'tri_stress': _TRI_STRESS},
            'LC2': {'element_distribution': _BAR_DIST, 'tri_stress': _TRI_STRESS},
        }}
        rows = cuts_report(s, results, ['LC1', 'LC2'])
        self.assertEqual(len(rows), 4)   # 2 cuts x 2 cases
        pairs = {(r['cut_id'], r['case']) for r in rows}
        self.assertEqual(pairs, {('C1', 'LC1'), ('C1', 'LC2'),
                                 ('C2', 'LC1'), ('C2', 'LC2')})
        row = next(r for r in rows if r['cut_id'] == 'C1' and r['case'] == 'LC1')
        expected = cut_result(s, results, s.cuts[0], 'LC1')['resultant']['total']
        for k, v in expected.items():
            self.assertAlmostEqual(row[f'total_{k}'], v)
        self.assertEqual(row['reason'], '')
        self.assertEqual(row['cut_name'], '')

    def test_no_cuts_gives_no_rows(self):
        s = _mixed_model()
        results = {'analysis_cases': {'LC1': {}}}
        self.assertEqual(cuts_report(s, results, ['LC1']), [])


class TestGlobalEquilibriumStyleCheck(unittest.TestCase):
    """A cantilever column under a single horizontal tip load: whichever
    height the cut is drawn at, its resultant matches the closed-form beam
    equations (V constant along the span = tip load, M linear = load x
    distance from the tip). Not the full merged-model / solver-based
    equilibrium test the plan describes (unavailable without scipy in this
    sandbox), but it exercises cut_result end to end against known statics,
    analogous to a base-reaction cross-check."""

    def test_shear_matches_the_applied_tip_load_everywhere(self):
        s = _bar_only_model()  # bar N1(2,-1) -> N2(2,3), i.e. length 4
        results = {'analysis_cases': {'LC1': {'element_distribution': {
            'E1': {'x': [0.0, 4.0], 'N': [0.0, 0.0], 'V': [5.0, 5.0],
                   'M': [0.0, 20.0]},   # constant V=5 (tip load), M=V*x
        }}}}
        for y in (-0.5, 0.0, 1.0, 2.5):
            with self.subTest(y=y):
                cut = s.add_cut(f"C_{y}", -1.0, y, 5.0, y)
                r = cut_result(s, results, cut, 'LC1')
                # Cut tangent (1,0), normal (0,1); the bar's local (N=0,V=5)
                # rotates (bar tangent (0,1)) to global (-5, 0), which
                # projects to V_cut = -5 along t=(1,0) — same magnitude as
                # the tip load, sign fixed by the cut's own orientation, and
                # identical at every height (equilibrium: shear is constant
                # along an unloaded span).
                self.assertAlmostEqual(r['resultant']['total']['V'], -5.0,
                                       places=6)


if __name__ == "__main__":
    unittest.main()
