"""Tests for the Eurocode material database (xdfem2d.databases).

`databases` sources everything from eurocodepy (a required dependency) with
no embedded fallback.
"""
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)

from xdfem2d import databases


class TestMaterialDb(unittest.TestCase):
    def setUp(self):
        self.db = databases._build_eco_db()

    def test_has_all_categories(self):
        for cat in ("Concrete (EC2)", "Structural Steel (EC3)",
                    "Reinforcing Steel", "Timber (EC5)"):
            self.assertIn(cat, self.db)

    def test_concrete_grades_present(self):
        conc = self.db["Concrete (EC2)"]
        for grade in ("C20/25", "C30/37", "C50/60"):
            self.assertIn(grade, conc)

    def test_concrete_values_are_sane(self):
        # Each grade is (E [kN/m²], unit weight [kN/m³], alpha [1/°C]).
        E, gamma, alpha = self.db["Concrete (EC2)"]["C30/37"]
        self.assertGreater(E, 20e6)        # ~30+ GPa in kN/m²
        self.assertLess(E, 50e6)
        self.assertAlmostEqual(gamma, 25.0, delta=2.0)
        self.assertAlmostEqual(alpha, 1e-5, delta=5e-6)

    def test_ecm_increases_with_grade(self):
        conc = self.db["Concrete (EC2)"]
        self.assertLess(conc["C20/25"][0], conc["C50/60"][0])

    def test_steel_modulus(self):
        # Structural steel E ≈ 210 GPa = 210e6 kN/m².
        any_grade = next(iter(self.db["Structural Steel (EC3)"].values()))
        self.assertAlmostEqual(any_grade[0], 210e6, delta=5e6)


class TestMaterialsLoader(unittest.TestCase):
    def test_returns_category_dict(self):
        eco = databases._load_eurocodepy_json()
        self.assertIsInstance(eco, dict)
        self.assertIn("Concrete", eco)
        self.assertIn("Grade", eco["Concrete"])


class TestSteelProfiles(unittest.TestCase):
    def test_returns_mapping(self):
        profiles = databases._load_steel_profiles()
        self.assertIsInstance(profiles, dict)

    def test_profile_entries_well_formed_when_present(self):
        profiles = databases._load_steel_profiles()
        for _family, items in profiles.items():
            for p in items:
                self.assertIn("name", p)
                self.assertIn("A_m2", p)
                self.assertGreater(p["A_m2"], 0.0)
                break


if __name__ == "__main__":
    unittest.main(verbosity=2)
