"""dev/IMPLEMENT_QUAD.md Phase 7: quad area resultants for section cuts —
the quad analogue of test_cuts_area_resultants.py's triangle tests.

``find_quad_crossings`` splits each quad into its two ``i-k`` diagonal
triangles for the purely geometric clip; because the quad's reported
stress/moment is one element-constant record (like a triangle's), a cut
that only touches one half reports a single hit, and a cut that crosses the
internal diagonal reports two hits for the same quad id whose lengths sum to
the correct total. No ``calculate()`` involved — tri_stress records are
fabricated directly, same style as the triangle tests.
"""
import unittest

from context import Structure2D
from xdfem2d.cuts import cut_area_resultant, find_quad_crossings


def _plane_quad():
    """A(0,0) B(6,0) C(6,4) D(0,4) — the i-k diagonal is A-C, i.e.
    y = (2/3)x."""
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_quad_section("QS", "M", thickness=0.2, formulation="Q4")
    s.add_node("A", 0.0, 0.0)
    s.add_node("B", 6.0, 0.0)
    s.add_node("C", 6.0, 4.0)
    s.add_node("D", 0.0, 4.0)
    s.add_quad_element("Q1", "A", "B", "C", "D", "QS")
    return s


def _plate_quad():
    s = Structure2D(domain="plate")
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_quad_section("QS", "M", thickness=0.2, formulation="MITC4")
    s.add_node("A", 0.0, 0.0)
    s.add_node("B", 6.0, 0.0)
    s.add_node("C", 6.0, 4.0)
    s.add_node("D", 0.0, 4.0)
    s.add_quad_element("Q1", "A", "B", "C", "D", "QS")
    return s


class TestQuadCrossingSingleHalf(unittest.TestCase):
    """A cut confined to one of the two i-k diagonal triangles produces a
    single hit for the quad."""

    def test_one_hit_when_the_cut_stays_on_one_side_of_the_diagonal(self):
        s = _plane_quad()
        # y=1: the A-C diagonal (x = 1.5*y) sits at x=1.5, so x in [5,6] is
        # entirely in the B-C-A half (x >= 1.5).
        cut = s.add_cut("C1", 5.0, 1.0, 7.0, 1.0)
        hits = find_quad_crossings(s, cut)
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["tri_id"], "Q1")
        self.assertAlmostEqual(hits[0]["p1"][0] - hits[0]["p0"][0], 1.0)


class TestQuadCrossingBothHalves(unittest.TestCase):
    """A cut that crosses the internal i-k diagonal produces two hits for
    the same quad id, whose lengths sum to the full width."""

    def test_two_hits_summing_to_the_quad_width(self):
        s = _plane_quad()
        # y=2: the diagonal sits at x=3, splitting the [-1, 7] cut across
        # both halves; only the x in [0, 6] portion is inside the quad.
        cut = s.add_cut("C1", -1.0, 2.0, 7.0, 2.0)
        hits = find_quad_crossings(s, cut)
        self.assertEqual(len(hits), 2)
        self.assertTrue(all(h["tri_id"] == "Q1" for h in hits))
        total_len = sum(abs(h["p1"][0] - h["p0"][0]) for h in hits)
        self.assertAlmostEqual(total_len, 6.0)


class TestPlaneQuadAreaResultant(unittest.TestCase):
    """Uniaxial field patch test (mirrors
    TestPlaneUniformStressPatchTest for triangles): a vertical cut (normal
    along +x) reports N = sx * t * length exactly, whether it crosses the
    internal i-k diagonal or not. The diagonal (A-C: y = (2/3)x) splits the
    quad into a lower-right half (triangle1, y < (2/3)x) and an upper-left
    half (triangle2, y > (2/3)x); a cut confined to y > (2/3)x at a given x
    stays in triangle2 only (single hit), while a full-height cut crosses
    the diagonal (two hits)."""

    def test_single_half_cut_matches_hand_calc(self):
        s = _plane_quad()
        sx = 12.0
        results = {'analysis_cases': {'LC1': {'tri_stress': {
            'Q1': {'sx': sx, 'sy': 0.0, 'txy': 0.0},
        }}}}
        # x=1: diagonal is at y=2/3; y in [2, 6] clips to [2, 4] inside the
        # quad, entirely above the diagonal → one half only.
        cut = s.add_cut("C1", 1.0, 2.0, 1.0, 6.0)
        r = cut_area_resultant(s, results, cut, 'LC1')
        self.assertEqual(r['reason'], '')
        self.assertEqual(len(r['areas']), 1)
        self.assertAlmostEqual(r['resultant']['N'], sx * 0.2 * 2.0, places=6)
        self.assertAlmostEqual(r['resultant']['V'], 0.0, places=6)

    def test_diagonal_crossing_cut_still_matches_hand_calc(self):
        """The whole point of splitting into two triangles: a cut that
        crosses the internal diagonal must still integrate to the exact
        uniaxial resultant across the full quad height, not half of it."""
        s = _plane_quad()
        sx = 12.0
        results = {'analysis_cases': {'LC1': {'tri_stress': {
            'Q1': {'sx': sx, 'sy': 0.0, 'txy': 0.0},
        }}}}
        # x=3: diagonal is at y=2, inside the full-height [-1, 5] cut,
        # clipped to [0, 4] → crosses both halves.
        cut = s.add_cut("C1", 3.0, -1.0, 3.0, 5.0)
        r = cut_area_resultant(s, results, cut, 'LC1')
        self.assertEqual(r['reason'], '')
        self.assertEqual(len(r['areas']), 2)
        self.assertAlmostEqual(r['resultant']['N'], sx * 0.2 * 4.0, places=6)
        self.assertAlmostEqual(r['resultant']['V'], 0.0, places=6)

    def test_cut_missing_the_quad_is_empty_but_not_an_error(self):
        s = _plane_quad()
        results = {'analysis_cases': {'LC1': {'tri_stress': {
            'Q1': {'sx': 1.0, 'sy': 0.0, 'txy': 0.0},
        }}}}
        far_cut = s.add_cut("C2", 20.0, 20.0, 21.0, 21.0)
        r = cut_area_resultant(s, results, far_cut, 'LC1')
        self.assertEqual(r['reason'], '')
        self.assertEqual(r['areas'], [])


class TestPlateQuadAreaResultant(unittest.TestCase):
    """Same shape check as TestPlateAreaResultant, for a plate quad."""

    def test_resultant_matches_the_rotated_moment_field(self):
        s = _plate_quad()
        results = {'analysis_cases': {'LC1': {'tri_stress': {
            'Q1': {'mx': 8.0, 'my': -2.0, 'mxy': 1.0, 'vx': 5.0, 'vy': -3.0},
        }}}}
        cut = s.add_cut("C1", 5.0, 1.0, 7.0, 1.0)   # horizontal → n = (0, 1)
        r = cut_area_resultant(s, results, cut, 'LC1')
        a = r['areas'][0]
        self.assertAlmostEqual(a['local']['m_n'], -2.0)   # my
        self.assertAlmostEqual(a['local']['m_nt'], 1.0)   # mxy
        self.assertAlmostEqual(a['local']['v_n'], -3.0)   # vy
        self.assertAlmostEqual(r['resultant']['V'], -3.0 * 1.0)
        self.assertAlmostEqual(r['resultant']['Mb'], -2.0 * 1.0)


class TestQuadAndTriangleShareTheSameElementSpace(unittest.TestCase):
    """A mixed tri/quad model: cut_area_resultant sums both kinds under one
    resultant, and a triangle and a quad id never collide (Phase 5)."""

    def test_mixed_model_sums_both_kinds(self):
        s = Structure2D()
        s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
        s.add_tri_section("TS", "M", thickness=0.2)
        s.add_quad_section("QS", "M", thickness=0.2, formulation="Q4")
        s.add_node("A", 0.0, 0.0); s.add_node("B", 4.0, 0.0)
        s.add_node("C", 4.0, 4.0); s.add_node("D", 0.0, 4.0)
        s.add_node("E", 8.0, 0.0); s.add_node("F", 8.0, 4.0)
        s.add_tri_element("T1", "B", "E", "F", "TS")
        s.add_tri_element("T2", "B", "F", "C", "TS")
        s.add_quad_element("Q1", "A", "B", "C", "D", "QS")
        sx = 10.0
        results = {'analysis_cases': {'LC1': {'tri_stress': {
            'T1': {'sx': sx, 'sy': 0.0, 'txy': 0.0},
            'T2': {'sx': sx, 'sy': 0.0, 'txy': 0.0},
            'Q1': {'sx': sx, 'sy': 0.0, 'txy': 0.0},
        }}}}
        cut = s.add_cut("C1", 2.0, -1.0, 2.0, 5.0)   # through the quad only
        r1 = cut_area_resultant(s, results, cut, 'LC1')
        self.assertAlmostEqual(r1['resultant']['N'], sx * 0.2 * 4.0, places=6)
        cut2 = s.add_cut("C2", 6.0, -1.0, 6.0, 5.0)   # through the triangles only
        r2 = cut_area_resultant(s, results, cut2, 'LC1')
        self.assertAlmostEqual(r2['resultant']['N'], sx * 0.2 * 4.0, places=6)


if __name__ == "__main__":
    unittest.main()
