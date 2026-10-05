"""Tests for deformed-shape interpolation and scale helper (xdfem2d.postprocess)."""
import math
import unittest

import numpy as np

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)
from xdfem2d.postprocess import hermite_deformed, nice_scale, local_deflection


class _Node:
    def __init__(self, x, y):
        self.x = x
        self.y = y


class TestHermiteDeformed(unittest.TestCase):
    def test_endpoints_match_nodal_displacements(self):
        ni, nj = _Node(0.0, 0.0), _Node(4.0, 0.0)
        di, dj = [0.01, -0.02, 0.003], [-0.005, 0.04, -0.001]
        X, Y = hermite_deformed(ni, nj, di, dj, sf=1.0, n=21)
        # First point = node i + its displacement; last = node j + its displacement.
        self.assertAlmostEqual(X[0], ni.x + di[0], places=9)
        self.assertAlmostEqual(Y[0], ni.y + di[1], places=9)
        self.assertAlmostEqual(X[-1], nj.x + dj[0], places=9)
        self.assertAlmostEqual(Y[-1], nj.y + dj[1], places=9)

    def test_zero_displacement_is_straight_line(self):
        ni, nj = _Node(0.0, 0.0), _Node(3.0, 4.0)   # inclined element
        X, Y = hermite_deformed(ni, nj, [0, 0, 0], [0, 0, 0], sf=5.0, n=11)
        # Undeformed: points lie on the straight i–j axis (cross-product ≈ 0).
        cross = (X - ni.x) * (nj.y - ni.y) - (Y - ni.y) * (nj.x - ni.x)
        np.testing.assert_allclose(cross, 0.0, atol=1e-9)

    def test_scale_factor_scales_transverse_deflection(self):
        ni, nj = _Node(0.0, 0.0), _Node(4.0, 0.0)
        dj = [0.0, 0.0, 0.01]            # tip rotation only
        _, Y1 = hermite_deformed(ni, nj, [0, 0, 0], dj, sf=1.0, n=21)
        _, Y2 = hermite_deformed(ni, nj, [0, 0, 0], dj, sf=2.0, n=21)
        np.testing.assert_allclose(Y2, 2.0 * Y1, rtol=1e-9, atol=1e-12)

    def test_zero_length_element_is_safe(self):
        ni = nj = _Node(1.0, 1.0)
        X, Y = hermite_deformed(ni, nj, [0, 0, 0], [0, 0, 0], sf=1.0)
        self.assertEqual(len(X), 2)


class TestNiceScale(unittest.TestCase):
    def test_rounds_up_to_1_2_5_decade(self):
        self.assertEqual(nice_scale(0.3), 0.5)
        self.assertEqual(nice_scale(1.0), 1.0)
        self.assertEqual(nice_scale(1.5), 2.0)
        self.assertEqual(nice_scale(7.0), 10.0)
        self.assertEqual(nice_scale(45.0), 50.0)

    def test_nonpositive_returns_one(self):
        self.assertEqual(nice_scale(0.0), 1.0)
        self.assertEqual(nice_scale(-3.0), 1.0)


class TestLocalDeflection(unittest.TestCase):
    """The transverse deflection along a bar, and its robustness to an M/xs
    length mismatch (which used to raise a broadcasting ValueError)."""

    def test_simply_supported_udl_midspan(self):
        ni, nj = _Node(0.0, 0.0), _Node(6.0, 0.0)
        EI = 30e6 * 3.125e-3
        xs = np.linspace(0.0, 6.0, 51)
        w, R = 20.0, 20.0 * 6.0 / 2.0
        M = R * xs - w * xs ** 2 / 2.0
        s, v = local_deflection(ni, nj, [0, 0, 0], [0, 0, 0], M, xs, EI)
        self.assertEqual(s.size, v.size)
        # 5·w·L⁴/(384·EI) = 3.6 mm
        self.assertAlmostEqual(abs(v).max() * 1000.0, 3.6, places=2)

    def test_mismatched_M_and_xs_do_not_raise(self):
        ni, nj = _Node(0.0, 0.0), _Node(6.0, 0.0)
        EI = 30e6 * 3.125e-3
        xs = np.linspace(0.0, 6.0, 51)          # 51 stations
        M = np.zeros(31)                         # 31 moments — different length
        s, v = local_deflection(ni, nj, [0, 0, 0], [0, 0, 0], M, xs, EI)
        self.assertEqual(s.size, v.size)         # consistent arrays, no crash

    def test_mismatched_grid_still_integrates_not_hermite(self):
        """When M and xs are on different grids the diagram is resampled and
        integrated, not dropped to the Hermite end-only shape — so a loaded
        member's true deflection survives (≈ the aligned-grid result, not 0)."""
        ni, nj = _Node(0.0, 0.0), _Node(6.0, 0.0)
        EI = 30e6 * 3.125e-3
        w, R = 20.0, 20.0 * 6.0 / 2.0
        xs51 = np.linspace(0.0, 6.0, 51)
        M51 = R * xs51 - w * xs51 ** 2 / 2.0
        _, v_aligned = local_deflection(ni, nj, [0, 0, 0], [0, 0, 0], M51, xs51, EI)
        xs31 = np.linspace(0.0, 6.0, 31)         # M on a coarser grid
        M31 = R * xs31 - w * xs31 ** 2 / 2.0
        s, v = local_deflection(ni, nj, [0, 0, 0], [0, 0, 0], M31, xs51, EI)
        self.assertEqual(s.size, 51)
        # Close to the aligned 3.6 mm (small resampling error), not the Hermite 0.
        self.assertAlmostEqual(abs(v).max() * 1000.0, abs(v_aligned).max() * 1000.0,
                               places=1)
        self.assertGreater(abs(v).max() * 1000.0, 3.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
