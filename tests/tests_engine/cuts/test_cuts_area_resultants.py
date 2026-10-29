"""Phase-4 Cut tests: area (triangle) resultants — tensor rotation to cut
axes, line integration over the clipped sub-segment, and moment/shear
transfer to the reference point. No Qt is involved, and none of this calls
``calculate()``: tri_stress records are fabricated directly in the shape
``tri_elements.tri_stresses_dispatch`` produces, so the tests exercise the
new geometry/algebra in ``cuts.py`` in isolation from the solver. See
dev/CUT_PLAN.md."""
import unittest

from context import Structure2D
from xdfem2d.cuts import cut_area_resultant


def _plane_tri():
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_tri_section("TS", "M", thickness=0.2)
    s.add_node("A", 0.0, 0.0)
    s.add_node("B", 4.0, 0.0)
    s.add_node("C", 0.0, 4.0)
    s.add_tri_element("T1", "A", "B", "C", "TS")
    return s


def _plate_tri():
    s = Structure2D(domain="plate")
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_plate_section("TS", "M", thickness=0.2)
    s.add_node("A", 0.0, 0.0)
    s.add_node("B", 4.0, 0.0)
    s.add_node("C", 0.0, 4.0)
    s.add_tri_element("T1", "A", "B", "C", "TS")
    return s


class TestPlaneAreaResultant(unittest.TestCase):
    """Hand-computed (see cuts.py module notes for the formulas): cut y=1
    from (-1,1) to (5,1) through the T1 triangle (clip at s in
    [1/6, 4/6], length 3, established in test_cuts_geometry.py), constant
    stress sx=10, sy=-4, txy=3, thickness 0.2."""

    def setUp(self):
        self.struc = _plane_tri()
        self.cut = self.struc.add_cut("C1", -1.0, 1.0, 5.0, 1.0)
        self.results = {
            'analysis_cases': {
                'LC1': {'tri_stress': {
                    'T1': {'sx': 10.0, 'sy': -4.0, 'txy': 3.0},
                }},
            },
        }

    def test_resultant_at_midpoint_reference(self):
        r = cut_area_resultant(self.struc, self.results, self.cut, 'LC1')
        self.assertEqual(r['reason'], '')
        self.assertEqual(len(r['areas']), 1)
        a = r['areas'][0]
        self.assertEqual(a['tri_id'], 'T1')
        self.assertAlmostEqual(a['s0'], 1 / 6)
        self.assertAlmostEqual(a['s1'], 4 / 6)
        self.assertAlmostEqual(a['length'], 3.0)
        self.assertAlmostEqual(a['local']['n_n'], -0.8)   # sy * thickness
        self.assertAlmostEqual(a['local']['n_t'], 0.6)    # txy * thickness
        res = r['resultant']
        self.assertAlmostEqual(res['N'], -2.4)
        self.assertAlmostEqual(res['V'], 1.8)
        self.assertAlmostEqual(res['M'], 1.2)
        self.assertEqual(a['contribution'], res)

    def test_missing_case_reports_a_reason(self):
        r = cut_area_resultant(self.struc, self.results, self.cut, 'NOPE')
        self.assertNotEqual(r['reason'], '')
        self.assertEqual(r['areas'], [])
        self.assertEqual(r['resultant'], {'N': 0.0, 'V': 0.0, 'M': 0.0})

    def test_cut_missing_the_triangle_is_empty_but_not_an_error(self):
        far_cut = self.struc.add_cut("C2", 20.0, 20.0, 21.0, 21.0)
        r = cut_area_resultant(self.struc, self.results, far_cut, 'LC1')
        self.assertEqual(r['reason'], '')
        self.assertEqual(r['areas'], [])
        self.assertEqual(r['resultant'], {'N': 0.0, 'V': 0.0, 'M': 0.0})


class TestPlaneUniformStressPatchTest(unittest.TestCase):
    """Phase-4 validation case from dev/CUT_PLAN.md: a uniform stress field
    gives an exact resultant. A cut normal to X through a uniaxial sx field
    should report N = sx * thickness * cut_length exactly, V = M = 0,
    regardless of where along the strip the cut is drawn."""

    def test_uniaxial_field_gives_exact_normal_force(self):
        s = Structure2D()
        s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
        s.add_tri_section("TS", "M", thickness=0.25)
        # Two triangles forming a 6x4 rectangular strip, split by a diagonal
        # so the cut crosses both.
        s.add_node("A", 0.0, 0.0)
        s.add_node("B", 6.0, 0.0)
        s.add_node("C", 6.0, 4.0)
        s.add_node("D", 0.0, 4.0)
        s.add_tri_element("T1", "A", "B", "C", "TS")
        s.add_tri_element("T2", "A", "C", "D", "TS")
        sx = 12.0
        results = {'analysis_cases': {'LC1': {'tri_stress': {
            'T1': {'sx': sx, 'sy': 0.0, 'txy': 0.0},
            'T2': {'sx': sx, 'sy': 0.0, 'txy': 0.0},
        }}}}
        # Vertical cut, normal along +X, spanning the full height.
        for x in (1.0, 3.0, 5.0):
            with self.subTest(x=x):
                cut = s.add_cut(f"C{x}", x, -1.0, x, 5.0)
                r = cut_area_resultant(s, results, cut, 'LC1')
                self.assertAlmostEqual(r['resultant']['N'], sx * 0.25 * 4.0,
                                       places=6)
                self.assertAlmostEqual(r['resultant']['V'], 0.0, places=6)
                self.assertAlmostEqual(r['resultant']['M'], 0.0, places=6)


class TestPlaneAreaCombination(unittest.TestCase):
    def test_soma_linear_is_the_weighted_sum(self):
        struc = _plane_tri()
        cut = struc.add_cut("C1", -1.0, 1.0, 5.0, 1.0)
        results = {'analysis_cases': {
            'A': {'tri_stress': {'T1': {'sx': 10.0, 'sy': -4.0, 'txy': 3.0}}},
            'B': {'tri_stress': {'T1': {'sx': 2.0, 'sy': 1.0, 'txy': -1.0}}},
        }}
        # A combination may only reference analysis cases (or other
        # combinations) — 'A'/'B' here are fabricated result-only ids, so
        # register them as (dummy) analysis cases before combining them.
        struc.add_analysis_case('A')
        struc.add_analysis_case('B')
        struc.add_load_combination('COMBO', {'A': 1.5, 'B': -0.5})

        ra = cut_area_resultant(struc, results, cut, 'A')['resultant']
        rb = cut_area_resultant(struc, results, cut, 'B')['resultant']
        rc = cut_area_resultant(struc, results, cut, 'COMBO')['resultant']
        for k in ('N', 'V', 'M'):
            self.assertAlmostEqual(rc[k], 1.5 * ra[k] - 0.5 * rb[k])

    def test_envelope_combination_reports_a_reason(self):
        struc = _plane_tri()
        cut = struc.add_cut("C1", -1.0, 1.0, 5.0, 1.0)
        results = {'analysis_cases': {'A': {'tri_stress': {
            'T1': {'sx': 10.0, 'sy': -4.0, 'txy': 3.0}}}}}
        struc.add_analysis_case('A')
        struc.add_load_combination('ENV', {'A': 1.0}, combo_type='Envelope')
        r = cut_area_resultant(struc, results, cut, 'ENV')
        self.assertNotEqual(r['reason'], '')


class TestPlateAreaResultant(unittest.TestCase):
    """Same cut/triangle geometry as the plane case, with a plate moment
    record: mx=8, my=-2, mxy=1, vx=5, vy=-3."""

    def setUp(self):
        self.struc = _plate_tri()
        self.cut = self.struc.add_cut("C1", -1.0, 1.0, 5.0, 1.0)
        self.results = {
            'analysis_cases': {
                'LC1': {'tri_stress': {
                    'T1': {'mx': 8.0, 'my': -2.0, 'mxy': 1.0,
                           'vx': 5.0, 'vy': -3.0},
                }},
            },
        }

    def test_resultant_at_midpoint_reference(self):
        r = cut_area_resultant(self.struc, self.results, self.cut, 'LC1')
        self.assertEqual(r['reason'], '')
        a = r['areas'][0]
        self.assertAlmostEqual(a['local']['m_n'], -2.0)   # my (n=(0,1))
        self.assertAlmostEqual(a['local']['m_nt'], 1.0)   # mxy
        self.assertAlmostEqual(a['local']['v_n'], -3.0)   # vy
        res = r['resultant']
        self.assertAlmostEqual(res['V'], -9.0)
        self.assertAlmostEqual(res['Mb'], -6.0)
        self.assertAlmostEqual(res['Mt'], -1.5)

    def test_missing_case_zero_resultant_has_plate_keys(self):
        r = cut_area_resultant(self.struc, self.results, self.cut, 'NOPE')
        self.assertEqual(r['resultant'], {'V': 0.0, 'Mb': 0.0, 'Mt': 0.0})


class TestSharedResultantShapeWithBars(unittest.TestCase):
    """Phase 5 will sum bar + area resultants directly — verify the two
    return exactly the same key sets for a given domain."""

    def test_plane_keys_match(self):
        from xdfem2d.cuts import cut_bar_resultant
        bar_struc = Structure2D()
        bar_struc.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
        bar_struc.add_section("S", "M", b=0.3, h=0.6)
        bar_struc.add_node("N1", 0.0, 0.0)
        bar_struc.add_node("N2", 4.0, 0.0)
        bar_struc.add_bar_element("E1", "N1", "N2", "S")
        cut = bar_struc.add_cut("C1", 2.0, -1.0, 2.0, 1.0)
        bar_r = cut_bar_resultant(bar_struc, {'analysis_cases': {'LC1': {
            'element_distribution': {'E1': {'x': [0.0, 4.0], 'N': [1.0, 1.0],
                                            'V': [1.0, 1.0], 'M': [0.0, 1.0]}},
        }}}, cut, 'LC1')

        area_struc = _plane_tri()
        area_cut = area_struc.add_cut("C1", -1.0, 1.0, 5.0, 1.0)
        area_r = cut_area_resultant(area_struc, {'analysis_cases': {'LC1': {
            'tri_stress': {'T1': {'sx': 1.0, 'sy': 1.0, 'txy': 1.0}},
        }}}, area_cut, 'LC1')

        self.assertEqual(set(bar_r['resultant']), set(area_r['resultant']))


if __name__ == "__main__":
    unittest.main()
