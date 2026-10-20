"""Tests for the EC8:2004 response spectra (xdfem2d.seismic).

The ordinates are delegated to eurocodepy (a required dependency).
"""
import math
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)
from xdfem2d.seismic import ec8_spectrum, ec8_design_spectrum


AG, S, TB, TC, TD, XI = 2.0, 1.2, 0.15, 0.5, 2.0, 5.0


def _Se_at(T):
    """Reference EN 1998-1 ordinate at period T for the parameters above."""
    eta = max(math.sqrt(10.0 / (5.0 + XI)), 0.55)
    if T <= TB:
        return AG * S * (1.0 + (T / TB) * (eta * 2.5 - 1.0))
    if T <= TC:
        return AG * S * eta * 2.5
    if T <= TD:
        return AG * S * eta * 2.5 * (TC / T)
    return AG * S * eta * 2.5 * (TC * TD / T**2)


class TestEc8Spectrum(unittest.TestCase):
    def setUp(self):
        self.pts = ec8_spectrum(AG, S, TB, TC, TD, XI)
        self.byT = {round(T, 6): Se for T, Se in self.pts}

    def test_eta_at_five_percent_is_one(self):
        # ξ = 5% → η = 1, so the plateau equals ag·S·2.5.
        plateau = AG * S * 1.0 * 2.5
        # TC is a corner period and is always sampled.
        self.assertAlmostEqual(self.byT[round(TC, 6)], round(plateau, 6), places=5)

    def test_value_at_T0_is_agS(self):
        self.assertAlmostEqual(self.byT[0.0], round(AG * S, 6), places=6)

    def test_corner_periods_match_closed_form(self):
        for T in (0.0, TB, TC, TD):
            self.assertIn(round(T, 6), self.byT)
            self.assertAlmostEqual(self.byT[round(T, 6)], round(_Se_at(T), 6),
                                   places=5, msg=f"Se at T={T}")

    def test_plateau_between_TB_and_TC(self):
        plateau = round(AG * S * 2.5, 6)
        for T, Se in self.pts:
            if TB < T <= TC:
                self.assertAlmostEqual(Se, plateau, places=5)

    def test_descending_branch_is_monotonic(self):
        tail = [Se for T, Se in self.pts if T > TC]
        self.assertTrue(all(a >= b - 1e-9 for a, b in zip(tail, tail[1:])),
                        "spectrum must not increase beyond TC")

    def test_eta_floor_for_high_damping(self):
        # Very high damping is clamped at η = 0.55.
        pts = ec8_spectrum(AG, S, TB, TC, TD, 100.0)
        byT = {round(T, 6): Se for T, Se in pts}
        self.assertAlmostEqual(byT[round(TC, 6)], round(AG * S * 0.55 * 2.5, 6),
                               places=5)


class TestEc8DesignSpectrum(unittest.TestCase):
    Q, BETA = 3.0, 0.2

    def setUp(self):
        self.pts = ec8_design_spectrum(AG, S, TB, TC, TD, self.Q, self.BETA)
        self.byT = {round(T, 6): Sd for T, Sd in self.pts}

    def test_plateau_uses_behaviour_factor(self):
        # TB <= T <= TC plateau = ag·S·2.5/q (no damping correction).
        self.assertAlmostEqual(self.byT[round(TC, 6)],
                               round(AG * S * 2.5 / self.Q, 6), places=5)

    def test_value_at_T0(self):
        # Sd(0) = ag·S·(2/3).
        self.assertAlmostEqual(self.byT[0.0], round(AG * S * 2.0 / 3.0, 6), places=6)

    def test_lower_bound_floor(self):
        floor = self.BETA * AG
        for _T, Sd in self.pts:
            self.assertGreaterEqual(Sd, floor - 1e-9)
        # The long-period tail hits the floor.
        self.assertAlmostEqual(self.byT[round(4.0, 6)], round(floor, 6), places=5)

    def test_design_below_elastic_for_q_gt_1(self):
        el = {round(T, 6): v for T, v in ec8_spectrum(AG, S, TB, TC, TD, 5.0)}
        # On the plateau the design value is the elastic one divided by q.
        self.assertLess(self.byT[round(TC, 6)], el[round(TC, 6)])


if __name__ == "__main__":
    unittest.main(verbosity=2)
