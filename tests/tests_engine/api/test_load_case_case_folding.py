"""create_load_case (and the create_* load methods that lean on it) fold a
name that only differs by case into an existing load case instead of making
a near-duplicate.

Why this exists: a small model asked to build a model then add a load to it
in a separate turn reliably renames its own load case by accident -- 'SW'
becomes 'sw' or 'Sw' a turn later (qwen3.5:4b, 25/09/2026 calibration). Every
other id in the model (nodes, bars, sections, materials) stays exact-match on
purpose -- see as_node_id/as_section_name/as_material_name in _compat.py --
load cases are the one deliberate exception, and only through the create_*
facade: add_load_case and the raw add_*_load methods are untouched, so a
script written directly against them still gets exact matching with no
surprises.

The rule is "whoever creates the name first wins": the first spelling a name
is created with is what gets stored, and every later call, whatever case it
uses, resolves back to that same one.
"""
from __future__ import annotations

import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)


def _beam():
    s = Structure2D()
    s.add_material('C30', 33.0e6, 25.0)
    s.add_section('SEC', 'C30', 0.3, 0.5)
    s.add_node('N1', 0.0, 0.0)
    s.add_node('N2', 5.0, 0.0)
    s.add_bar_element('E1', 'N1', 'N2', 'SEC')
    return s


class TestCreateLoadCaseFolds(unittest.TestCase):
    def test_a_case_variant_returns_the_existing_case(self):
        s = _beam()
        sw = s.create_load_case('SW', self_weight_factor=1.0)
        again = s.create_load_case('sw')
        self.assertIs(again, sw)
        self.assertEqual(list(s.load_cases_by_id), ['SW'])

    def test_the_first_spelling_is_the_one_kept(self):
        """Whichever case created the name first is what is stored -- later
        calls fold into it, not the other way round."""
        s = _beam()
        s.create_load_case('Sw')
        s.create_load_case('SW')
        s.create_load_case('sw')
        self.assertEqual(list(s.load_cases_by_id), ['Sw'])

    def test_folding_does_not_overwrite_the_existing_case(self):
        """self_weight_factor/action_type on a folding call are ignored, same
        as any other get-or-create -- the case already has its own values."""
        s = _beam()
        s.create_load_case('SW', self_weight_factor=1.0)
        again = s.create_load_case('sw', self_weight_factor=0.5)
        self.assertEqual(again.self_weight_factor, 1.0)

    def test_no_existing_match_creates_a_new_case_as_spelled(self):
        s = _beam()
        s.create_load_case('Q')
        self.assertEqual(list(s.load_cases_by_id), ['Q'])


class TestCreateLoadsAutoCreateAndFold(unittest.TestCase):
    def test_a_named_case_that_does_not_exist_yet_is_created(self):
        s = _beam()
        s.create_node_load('N2', load_case='HOR', p=80.0, direction='left')
        self.assertIn('HOR', s.load_cases_by_id)

    def test_a_later_call_in_a_different_case_reuses_it(self):
        s = _beam()
        s.create_node_load('N2', load_case='HOR', p=80.0, direction='left')
        s.create_uniform_load('E1', load_case='hor', value=-5.0)
        self.assertEqual(list(s.load_cases_by_id), ['HOR'])
        # both loads landed on the one real case, not a phantom 'hor'
        self.assertTrue(all(pl.load_case_id == 'HOR' for pl in s.point_loads))
        self.assertTrue(all(dl.load_case_id == 'HOR'
                            for dl in s.distributed_loads))

    def test_omitting_load_case_still_uses_the_default(self):
        """None must keep going through the untouched default-case path, not
        the new named-fold path."""
        s = _beam()
        s.create_node_load('N2', p=10.0, direction='down')
        self.assertEqual(list(s.load_cases_by_id), ['LC1'])


class TestAddStaysExactMatch(unittest.TestCase):
    def test_add_load_case_refuses_an_exact_duplicate(self):
        """The original bug this whole feature grew out of: add_load_case had
        no existence check at all, not even exact-match -- a second call with
        the same id silently orphaned the first LoadCase in self.load_cases
        while load_cases_by_id moved on to the newer one. It must now raise,
        same as add_node does for a duplicate node id."""
        s = _beam()
        s.add_load_case('SW')
        with self.assertRaises(ValueError):
            s.add_load_case('SW')
        self.assertEqual(len(s.load_cases), 1)

    def test_add_load_case_refuses_a_case_variant_too(self):
        """The raw layer does not fold -- it refuses just as loudly as an
        exact match, so a case variant can never sneak past it and become a
        second, near-duplicate case."""
        s = _beam()
        s.add_load_case('SW')
        with self.assertRaises(ValueError):
            s.add_load_case('sw')
        self.assertEqual(list(s.load_cases_by_id), ['SW'])

    def test_add_load_case_still_creates_a_genuinely_new_case(self):
        s = _beam()
        s.add_load_case('SW')
        s.add_load_case('Q')
        self.assertEqual(set(s.load_cases_by_id), {'SW', 'Q'})

    def test_add_point_load_does_not_validate_or_fold_its_load_case(self):
        """Unchanged, pre-existing behaviour: add_point_load stores whatever
        load_case_id string it is given, without checking it exists."""
        s = _beam()
        s.add_load_case('SW')
        pl = s.add_point_load('N1', 'sw', fy=-5.0)
        self.assertEqual(pl.load_case_id, 'sw')
        self.assertNotIn('sw', s.load_cases_by_id)


if __name__ == '__main__':
    unittest.main()


class TestCreateAnalysisCaseFolds(unittest.TestCase):
    """The analysis-case twin of TestCreateLoadCaseFolds -- same bug, same
    fix, one namespace over (see create_analysis_case/add_analysis_case)."""

    def test_a_case_variant_returns_the_existing_case(self):
        s = _beam()
        ac = s.create_analysis_case('AC1', analysis_type='Linear')
        again = s.create_analysis_case('ac1')
        self.assertIs(again, ac)
        self.assertEqual(list(s.analysis_cases_by_id), ['AC1'])

    def test_add_analysis_case_refuses_an_exact_duplicate(self):
        s = _beam()
        s.add_analysis_case('AC1')
        with self.assertRaises(ValueError):
            s.add_analysis_case('AC1')
        self.assertEqual(len(s.analysis_cases), 1)

    def test_add_analysis_case_refuses_a_case_variant_too(self):
        s = _beam()
        s.add_analysis_case('AC1')
        with self.assertRaises(ValueError):
            s.add_analysis_case('ac1')
        self.assertEqual(list(s.analysis_cases_by_id), ['AC1'])

    def test_add_analysis_case_still_creates_a_genuinely_new_case(self):
        s = _beam()
        s.add_analysis_case('AC1')
        s.add_analysis_case('AC2')
        self.assertEqual(set(s.analysis_cases_by_id), {'AC1', 'AC2'})

    def test_a_load_cases_own_twin_analysis_case_is_not_broken_by_the_guard(self):
        """add_load_case(create_analysis_case=True) creates a twin analysis
        case through the same add_analysis_case that now refuses duplicates
        -- adding a second, differently-named load case must still work."""
        s = _beam()
        s.add_load_case('SW', self_weight_factor=1.0)
        s.add_load_case('Q')
        self.assertEqual(list(s.analysis_cases_by_id), ['SW', 'Q'])


class TestCreateLoadCombinationFolds(unittest.TestCase):
    """The load-combination twin of the load-case/analysis-case tests above.
    Combinations are a plain list (no *_by_id dict), so the duplicate shows
    up the other way round -- the SECOND entry becomes unreachable by id,
    since every lookup elsewhere is next(c for c in ... if c.id == ...) --
    but create_load_combination/add_load_combination are fixed the same way
    (26/09/2026): create_load_combination used to be a bare alias for
    add_load_combination; it is now its own function with the same fold."""

    def test_a_case_variant_returns_the_existing_combination(self):
        s = _beam()
        s.create_load_case('SW', self_weight_factor=1.0)
        combo = s.create_load_combination('ULS', {'SW': 1.35})
        again = s.create_load_combination('uls', {'SW': 99.0})
        self.assertIs(again, combo)
        self.assertEqual(len(s.load_combinations), 1)
        # the fold ignores the new coefficients, same as the other two
        self.assertEqual(again.coefficients, {'SW': 1.35})

    def test_add_load_combination_refuses_an_exact_duplicate(self):
        s = _beam()
        s.create_load_case('SW', self_weight_factor=1.0)
        s.add_load_combination('ULS', {'SW': 1.35})
        with self.assertRaises(ValueError):
            s.add_load_combination('ULS', {'SW': 1.0})
        self.assertEqual(len(s.load_combinations), 1)

    def test_add_load_combination_refuses_a_case_variant_too(self):
        s = _beam()
        s.create_load_case('SW', self_weight_factor=1.0)
        s.add_load_combination('ULS', {'SW': 1.35})
        with self.assertRaises(ValueError):
            s.add_load_combination('uls', {'SW': 1.0})
        self.assertEqual(len(s.load_combinations), 1)

    def test_add_load_combination_still_creates_a_genuinely_new_one(self):
        s = _beam()
        s.create_load_case('SW', self_weight_factor=1.0)
        s.add_load_combination('ULS', {'SW': 1.35})
        s.add_load_combination('SLS', {'SW': 1.0})
        self.assertEqual({c.id for c in s.load_combinations}, {'ULS', 'SLS'})
