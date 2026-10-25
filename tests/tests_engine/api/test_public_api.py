"""What the package promises, and what it merely offers.

This matters now in a way it did not before. The engine is to be published on
PyPI under the LGPLv3; from the first release, taking a name out of __all__ breaks
somebody else's code and costs a major version. Sixty-odd names had accumulated
there without anyone deciding that each was worth keeping still for years.

So the surface is split in two, in writing: STABLE is a promise, UNSTABLE is an
offer. These tests hold the split honest — every name resolves, nothing is in
both halves, and nothing reaches the namespace by accident.
"""
from __future__ import annotations

import unittest

import xdfem2d


class TestTheSurfaceIsWhatItSaysItIs(unittest.TestCase):

    def test_every_promised_name_exists(self):
        for name in xdfem2d.__all__:
            with self.subTest(name=name):
                self.assertTrue(hasattr(xdfem2d, name))

    def test_the_two_halves_do_not_overlap(self):
        self.assertEqual(set(xdfem2d.STABLE) & set(xdfem2d.UNSTABLE), set())

    def test_all_is_exactly_the_two_halves(self):
        self.assertEqual(xdfem2d.__all__,
                         xdfem2d.STABLE + xdfem2d.UNSTABLE)


class TestNothingArrivesByAccident(unittest.TestCase):
    """An unaliased `from pathlib import Path` in this module makes
    `from xdfem2d import Path` work, and someone eventually depends on it."""

    def test_the_standard_library_does_not_leak(self):
        for name in ('Path', 'version', 'PackageNotFoundError', 'json', 'os',
                     'sys', 'annotations'):
            with self.subTest(name=name):
                self.assertFalse(hasattr(xdfem2d, name))

    def test_the_undeclared_three_are_gone(self):
        """Reachable through the package but absent from __all__: usable,
        undocumented, promised to nobody. Every caller imports them from their
        own modules, which is where they stay."""
        for name in ('import_extend', 'make_variant_model',
                     'current_restraints'):
            with self.subTest(name=name):
                self.assertFalse(hasattr(xdfem2d, name))

    def test_they_are_still_reachable_where_they_live(self):
        """Removed from the package surface, not from the package."""
        from xdfem2d.import_io import import_extend        # noqa: F401
        from xdfem2d.workflows import current_restraints   # noqa: F401
        from xdfem2d.workflows import make_variant_model   # noqa: F401


class TestTheCoreIsWhatAScriptNeeds(unittest.TestCase):
    """STABLE has to carry a whole working script on its own, or the promise
    is not worth making."""

    def test_a_model_can_be_built_solved_and_saved_from_stable_names_alone(self):
        import tempfile
        from pathlib import Path

        stable = set(xdfem2d.STABLE)
        for name in ('Structure2D', 'save_x2d', 'load_x2d',
                     'design_reinforcement'):
            self.assertIn(name, stable)

        s = xdfem2d.Structure2D()
        s.add_material('C30', elastic_modulus=30e6, unit_weight=25.0)
        s.add_section('S1', 'C30', b=0.3, h=0.6)
        s.add_node('N1', 0.0, 0.0)
        s.add_node('N2', 5.0, 0.0)
        s.add_bar_element('E1', 'N1', 'N2', 'S1')
        s.add_support('PIN', ux=True, uy=True)
        s.add_support('ROLL', uy=True)
        s.assign_support('N1', 'PIN')
        s.assign_support('N2', 'ROLL')
        s.add_load_case('LC')
        s.add_distributed_load('E1', 'LC', fye=-10.0, fyd=-10.0)
        s.add_analysis_case('ULS', 'Linear', {'LC': 1.35})
        results = s.calculate()
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / 'm.x2d'
            xdfem2d.save_x2d(s, results, out)
            back, _, _ = xdfem2d.load_x2d(str(out))
        self.assertEqual(sorted(back.nodes), ['N1', 'N2'])


class TestTheScriptToolsAreOnTheSurface(unittest.TestCase):
    """They are the reason the package is pleasant to use with a language
    model — any language model, not only the one in the application."""

    def test_they_are_exported(self):
        for name in ('model_check', 'model_to_python', 'check_script',
                     'check_summary'):
            with self.subTest(name=name):
                self.assertIn(name, xdfem2d.UNSTABLE)
                self.assertTrue(callable(getattr(xdfem2d, name)))

    def test_they_work_through_the_package(self):
        s = xdfem2d.Structure2D()
        s.add_material('M', elastic_modulus=30e6, unit_weight=25.0)
        s.add_section('S', 'M', b=0.3, h=0.6)
        s.add_node('N1', 0.0, 0.0)
        s.add_node('N2', 5.0, 0.0)
        s.add_bar_element('E1', 'N1', 'N2', 'S')
        s.add_support('FIX', ux=True, uy=True, tz=True)
        s.assign_support('N1', 'FIX')
        source = xdfem2d.model_to_python(s)
        self.assertEqual(xdfem2d.check_script(source), [])
        self.assertEqual(xdfem2d.model_check(s), [])

    def test_model_check_finds_what_it_should(self):
        """The first version of the test above used a lone node and asserted a
        clean check — the node was unattached, and the check was right."""
        s = xdfem2d.Structure2D()
        s.add_node('LOOSE', 0.0, 0.0)
        found = xdfem2d.model_check(s)
        self.assertEqual([i['type'] for i in found], ['free_node'])
        self.assertIn('LOOSE', found[0]['msg'])


if __name__ == '__main__':
    unittest.main()
