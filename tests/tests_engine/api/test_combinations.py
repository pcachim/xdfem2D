"""Tests for EN 1990 load-combination generation (xdfem2d.combinations).

`_generate_ec_combos` delegates to eurocodepy (a required dependency). They check the generated factors against the
EN 1990 6.10 / 6.16 expressions by hand.
"""
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)
from xdfem2d.combinations import _generate_ec_combos
from xdfem2d.models import LoadCase, ActionType


# factors_map: lc_id -> (gamma_fav, gamma_unf, psi0, psi1, psi2)
FACTORS = {
    "G": (1.0, 1.35, 1.0, 1.0, 1.0),
    "Q": (0.0, 1.5, 0.7, 0.5, 0.3),
    "W": (0.0, 1.5, 0.6, 0.2, 0.0),
}


def _cases(*ids):
    at = {"G": ActionType.G, "Q": ActionType.Q, "W": ActionType.W}
    return [LoadCase(id=i, action_type=at[i]) for i in ids]


def _coeffs(combos):
    """Coefficient dicts only (order-independent comparison)."""
    return [c for _name, c, _label in combos]


class TestUls(unittest.TestCase):
    def setUp(self):
        self.combos = _generate_ec_combos(_cases("G", "Q", "W"), FACTORS, {"ULS"})

    def test_two_uls_combinations(self):
        self.assertEqual(len(self.combos), 2)
        for _n, _c, label in self.combos:
            self.assertEqual(label, "ULS")

    def test_q_leading_combo(self):
        # 1.35 G + 1.5 Q + 1.5·ψ0,W W  (ψ0,W = 0.6 → 0.90)
        self.assertIn({"G": 1.35, "Q": 1.5, "W": 0.9}, _coeffs(self.combos))

    def test_w_leading_combo(self):
        # 1.35 G + 1.5·ψ0,Q Q + 1.5 W  (ψ0,Q = 0.7 → 1.05)
        self.assertIn({"G": 1.35, "Q": 1.05, "W": 1.5}, _coeffs(self.combos))


class TestGravityOnly(unittest.TestCase):
    def test_dead_plus_live(self):
        # G + Q (no wind), ULS: 1.35 G + 1.5 Q must be generated.
        # (Regression guard for the eurocodepy wind/temperature empty-group bug.)
        combos = _generate_ec_combos(_cases("G", "Q"), FACTORS, {"ULS"})
        self.assertIn({"G": 1.35, "Q": 1.5}, _coeffs(combos))


class TestSls(unittest.TestCase):
    def test_quasi_permanent(self):
        # SLS-QP: G + ψ2,Q Q  (ψ2,Q = 0.3); W drops out (ψ2,W = 0).
        combos = _generate_ec_combos(_cases("G", "Q", "W"), FACTORS, {"SLS-QP"})
        self.assertIn({"G": 1.0, "Q": 0.3}, _coeffs(combos))

    def test_characteristic_has_leading_variable(self):
        # SLS-K: the leading variable enters with factor 1.0.
        combos = _generate_ec_combos(_cases("G", "Q", "W"), FACTORS, {"SLS-K"})
        self.assertTrue(combos)
        self.assertTrue(any(c.get("Q") == 1.0 for c in _coeffs(combos)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
