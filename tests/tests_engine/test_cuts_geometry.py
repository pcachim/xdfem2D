"""Phase-2 Cut tests: pure geometry kernel (segment/segment intersection,
segment/triangle clipping) and the Structure2D-level crossing finders.
No Qt is involved. See dev/CUT_PLAN.md."""
import math
import unittest

from context import Structure2D
from xdfem2d.cuts import (
    segment_intersection, clip_segment_to_triangle,
    find_bar_crossings, find_triangle_crossings,
)


class TestSegmentIntersection(unittest.TestCase):
    def test_simple_crossing(self):
        hit = segment_intersection((0, 0), (2, 2), (0, 2), (2, 0))
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.x, 1.0)
        self.assertAlmostEqual(hit.y, 1.0)
        self.assertAlmostEqual(hit.t, 0.5)
        self.assertAlmostEqual(hit.u, 0.5)

    def test_parallel_no_intersection(self):
        hit = segment_intersection((0, 0), (2, 0), (0, 1), (2, 1))
        self.assertIsNone(hit)

    def test_collinear_overlap_is_not_a_crossing(self):
        # Documented policy: a cut running along a bar has no unique
        # crossing point and is reported as no crossing.
        hit = segment_intersection((0, 0), (4, 0), (1, 0), (3, 0))
        self.assertIsNone(hit)

    def test_touch_at_endpoint_registers(self):
        hit = segment_intersection((0, 0), (2, 0), (2, 0), (2, 2))
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.t, 1.0)
        self.assertAlmostEqual(hit.u, 0.0)
        self.assertAlmostEqual(hit.x, 2.0)
        self.assertAlmostEqual(hit.y, 0.0)

    def test_beyond_segment_end_is_not_a_crossing(self):
        # The infinite lines cross at x=1.1, past the end of segment A.
        hit = segment_intersection((0, 0), (1, 0), (1.1, -1), (1.1, 1))
        self.assertIsNone(hit)

    def test_within_tolerance_of_segment_end_registers(self):
        # Lines cross a hair past the end of segment A; within tol it still
        # counts, clamped to t=1.
        hit = segment_intersection((0, 0), (1, 0), (1 + 1e-12, -1),
                                    (1 + 1e-12, 1), tol=1e-6)
        self.assertIsNotNone(hit)
        self.assertAlmostEqual(hit.t, 1.0)

    def test_non_crossing_segments(self):
        # Segments on non-intersecting lines within their own ranges.
        hit = segment_intersection((0, 0), (1, 0), (5, 5), (6, 6))
        self.assertIsNone(hit)

    def test_degenerate_segment_is_parallel(self):
        # A zero-length "segment" has no direction; denom is 0, no crossing.
        hit = segment_intersection((0, 0), (0, 0), (0, -1), (0, 1))
        self.assertIsNone(hit)


class TestClipSegmentToTriangle(unittest.TestCase):
    TRI = [(0.0, 0.0), (4.0, 0.0), (0.0, 4.0)]

    def test_full_crossing(self):
        # y=1 line through the triangle, entering at x=0 (s=1/6) and
        # leaving at the hypotenuse x+y=4 => x=3 (s=4/6).
        clip = clip_segment_to_triangle((-1, 1), (5, 1), self.TRI)
        self.assertIsNotNone(clip)
        s0, s1 = clip
        self.assertAlmostEqual(s0, 1 / 6)
        self.assertAlmostEqual(s1, 4 / 6)

    def test_fully_outside(self):
        clip = clip_segment_to_triangle((10, 10), (20, 20), self.TRI)
        self.assertIsNone(clip)

    def test_fully_inside(self):
        clip = clip_segment_to_triangle((1, 1), (2, 1), self.TRI)
        self.assertIsNotNone(clip)
        s0, s1 = clip
        self.assertAlmostEqual(s0, 0.0)
        self.assertAlmostEqual(s1, 1.0)

    def test_grazes_a_single_vertex(self):
        # Vertical line x=4 only meets the closed triangle at vertex (4,0).
        clip = clip_segment_to_triangle((4, -1), (4, 1), self.TRI)
        if clip is not None:
            s0, s1 = clip
            self.assertAlmostEqual(s0, s1, places=6)

    def test_along_one_edge(self):
        # Collinear with the y=0 edge, overlapping it for x in [0, 4].
        clip = clip_segment_to_triangle((-1, 0), (5, 0), self.TRI)
        self.assertIsNotNone(clip)
        s0, s1 = clip
        self.assertAlmostEqual(s0, 1 / 6)
        self.assertAlmostEqual(s1, 5 / 6)

    def test_fully_outside_collinear_with_an_edge_line(self):
        clip = clip_segment_to_triangle((-5, -5), (-1, -5), self.TRI)
        self.assertIsNone(clip)

    def test_clockwise_triangle_is_handled(self):
        # Same triangle, vertices listed clockwise: must give the same answer.
        cw_tri = [self.TRI[0], self.TRI[2], self.TRI[1]]
        clip = clip_segment_to_triangle((-1, 1), (5, 1), cw_tri)
        self.assertIsNotNone(clip)
        s0, s1 = clip
        self.assertAlmostEqual(s0, 1 / 6)
        self.assertAlmostEqual(s1, 4 / 6)

    def test_degenerate_cut_segment(self):
        clip = clip_segment_to_triangle((1, 1), (1, 1), self.TRI)
        self.assertIsNone(clip)


def _frame():
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    s.add_node("N1", 0.0, 0.0)
    s.add_node("N2", 4.0, 0.0)
    s.add_node("N3", 4.0, 3.0)
    s.add_bar_element("E1", "N1", "N2", "S")
    s.add_bar_element("E2", "N2", "N3", "S")
    return s


def _tri_model():
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_tri_section("TS", "M", thickness=0.2)
    s.add_node("A", 0.0, 0.0)
    s.add_node("B", 4.0, 0.0)
    s.add_node("C", 0.0, 4.0)
    s.add_tri_element("T1", "A", "B", "C", "TS")
    return s


class TestFindBarCrossings(unittest.TestCase):
    def test_crosses_one_bar(self):
        s = _frame()
        cut = s.add_cut("C1", 2.0, -1.0, 2.0, 1.0)
        hits = find_bar_crossings(s, cut)
        self.assertEqual(len(hits), 1)
        h = hits[0]
        self.assertEqual(h["elem_id"], "E1")
        self.assertAlmostEqual(h["a"], 2.0)   # 2 m from N1 along E1
        self.assertAlmostEqual(h["s"], 0.5)   # midpoint of the cut
        self.assertAlmostEqual(h["x"], 2.0)
        self.assertAlmostEqual(h["y"], 0.0)

    def test_misses_all_bars(self):
        s = _frame()
        cut = s.add_cut("C1", 10.0, -1.0, 10.0, 1.0)
        self.assertEqual(find_bar_crossings(s, cut), [])

    def test_crosses_both_bars_of_an_l_frame(self):
        s = _frame()
        # Diagonal cut through both members of the L: crosses E1 (N1-N2,
        # y=0, x in [0,4]) at (1.8, 0) and E2 (N2-N3, x=4, y in [0,3]) at
        # (4, 2.75), both away from the shared corner node.
        cut = s.add_cut("C1", 1.0, -1.0, 5.0, 4.0)
        hits = find_bar_crossings(s, cut)
        elem_ids = sorted(h["elem_id"] for h in hits)
        self.assertEqual(elem_ids, ["E1", "E2"])

    def test_zero_length_cut_raises(self):
        s = _frame()
        cut = s.add_cut("C1", 1.0, 1.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            find_bar_crossings(s, cut)


class TestFindTriangleCrossings(unittest.TestCase):
    def test_crosses_one_triangle(self):
        s = _tri_model()
        cut = s.add_cut("C1", -1.0, 1.0, 5.0, 1.0)
        hits = find_triangle_crossings(s, cut)
        self.assertEqual(len(hits), 1)
        h = hits[0]
        self.assertEqual(h["tri_id"], "T1")
        self.assertAlmostEqual(h["s0"], 1 / 6)
        self.assertAlmostEqual(h["s1"], 4 / 6)

    def test_misses_the_triangle(self):
        s = _tri_model()
        cut = s.add_cut("C1", 10.0, 10.0, 20.0, 20.0)
        self.assertEqual(find_triangle_crossings(s, cut), [])

    def test_tangential_contact_is_dropped(self):
        s = _tri_model()
        cut = s.add_cut("C1", 4.0, -1.0, 4.0, 1.0)  # grazes vertex B only
        self.assertEqual(find_triangle_crossings(s, cut), [])

    def test_zero_length_cut_raises(self):
        s = _tri_model()
        cut = s.add_cut("C1", 1.0, 1.0, 1.0, 1.0)
        with self.assertRaises(ValueError):
            find_triangle_crossings(s, cut)


if __name__ == "__main__":
    unittest.main()
