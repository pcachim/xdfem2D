"""Running a single added analysis case on a solved model (feature G).

An analysis case carries a ``solved`` flag. A full run marks every case solved;
adding one leaves it pending. ``solve_pending_cases`` computes just the pending
ones — reusing the per-load-case results for a Linear case (no stiffness), and
re-assembling K only when a Mass/Modal/Spectrum case is actually pending. The
result must equal a full re-run, which is the invariant these tests hold.
"""
from __future__ import annotations

import unittest

from context import Structure2D  # noqa: F401

from xdfem2d.solver import solve_pending_cases


def _beam(*extra_cases):
    s = Structure2D()
    s.add_material('C', elastic_modulus=30e6, unit_weight=25.0)
    s.add_section('S', 'C', b=0.3, h=0.5)
    s.add_node('N1', 0, 0)
    s.add_node('N2', 6, 0)
    s.add_bar_element('E1', 'N1', 'N2', 'S')
    s.add_support('PIN', ux=True, uy=True, tz=True)
    s.add_support('ROL', ux=False, uy=True)
    s.assign_support('N1', 'PIN')
    s.assign_support('N2', 'ROL')
    s.add_load_case('G', self_weight_factor=1.0)
    s.add_load_case('Q')
    s.add_distributed_load('E1', 'Q', fye=-10, fyd=-10)
    s.add_analysis_case('SLS', 'Linear', {'G': 1.0, 'Q': 1.0})
    for cid, coeffs in extra_cases:
        s.add_analysis_case(cid, 'Linear', coeffs)
    return s


def _ac_disp_diff(a, b, cid):
    da = a['analysis_cases'][cid]['displacements']
    db = b['analysis_cases'][cid]['displacements']
    return max(abs(da[n][k] - db[n][k]) for n in da for k in range(3))


class TestTheFlag(unittest.TestCase):

    def test_a_new_case_starts_unsolved(self):
        s = _beam()
        s.calculate()
        s.add_analysis_case('ULS', 'Linear', {'G': 1.35, 'Q': 1.5})
        self.assertFalse(s.analysis_cases_by_id['ULS'].solved)
        self.assertTrue(s.analysis_cases_by_id['SLS'].solved)

    def test_a_full_run_marks_all_solved(self):
        s = _beam(('ULS', {'G': 1.35, 'Q': 1.5}))
        s.calculate()
        self.assertTrue(all(ac.solved for ac in s.analysis_cases))

    def test_round_trip(self):
        from xdfem2d.structure_io import _from_dict, _to_dict
        s = _beam()
        s.calculate()
        back = _from_dict(_to_dict(s))
        self.assertTrue(back.analysis_cases_by_id['SLS'].solved)


class TestLinearIncrementMatchesFull(unittest.TestCase):

    def test_the_added_case_equals_a_full_rerun(self):
        full = _beam(('ULS', {'G': 1.35, 'Q': 1.5})).calculate()

        s = _beam()
        r = s.calculate()                    # SLS only
        s.add_analysis_case('ULS', 'Linear', {'G': 1.35, 'Q': 1.5})
        r2 = solve_pending_cases(s, r)

        self.assertAlmostEqual(_ac_disp_diff(r2, full, 'ULS'), 0.0, places=9)
        self.assertTrue(s.analysis_cases_by_id['ULS'].solved)

    def test_the_old_case_is_untouched(self):
        s = _beam()
        r = s.calculate()
        before = {n: list(v) for n, v in
                  r['analysis_cases']['SLS']['displacements'].items()}
        s.add_analysis_case('ULS', 'Linear', {'G': 1.35, 'Q': 1.5})
        r2 = solve_pending_cases(s, r)
        after = r2['analysis_cases']['SLS']['displacements']
        for n in before:
            self.assertEqual(before[n], after[n])


class TestModalIncrementNeedsStiffness(unittest.TestCase):

    def _frame(self, *extra):
        s = Structure2D()
        s.add_material('C', elastic_modulus=30e6, unit_weight=25.0)
        s.add_section('S', 'C', b=0.3, h=0.5)
        s.add_node('N1', 0, 0)
        s.add_node('N2', 6, 0)
        s.add_bar_element('E1', 'N1', 'N2', 'S')
        s.add_support('PIN', ux=True, uy=True, tz=True)
        s.add_support('ROL', ux=False, uy=True)
        s.assign_support('N1', 'PIN')
        s.assign_support('N2', 'ROL')
        s.add_load_case('G', self_weight_factor=1.0)
        s.add_analysis_case('MASS', 'Mass', {'G': 1.0})
        s.add_analysis_case('MOD', 'Modal', modal_case_id='MASS', num_modes=3)
        for cid in extra:
            s.add_analysis_case(cid, 'Modal', modal_case_id='MASS', num_modes=3)
        return s

    def _freqs(self, res, cid):
        mi = res['analysis_cases'][cid]['modal_info']
        return mi.get('frequencies') if isinstance(mi, dict) else mi

    def test_a_new_modal_case_matches_a_full_rerun(self):
        full = self._frame('MOD2').calculate()
        s = self._frame()
        r = s.calculate()
        s.add_analysis_case('MOD2', 'Modal', modal_case_id='MASS', num_modes=3)
        r2 = solve_pending_cases(s, r)
        self.assertEqual(self._freqs(r2, 'MOD2'), self._freqs(full, 'MOD2'))
        self.assertTrue(s.analysis_cases_by_id['MOD2'].solved)


class TestCombinationsAndFallback(unittest.TestCase):

    def test_a_new_combination_appears_without_a_pending_case(self):
        s = _beam()
        r = s.calculate()                    # everything solved
        s.add_load_combination('C1', {'SLS': 1.4}, combo_type='LinearSum')
        r2 = solve_pending_cases(s, r)
        self.assertIn('C1', r2['combinations'])

    def test_a_combination_equals_a_full_rerun(self):
        ref = _beam()
        ref.add_load_combination('C1', {'SLS': 1.4}, combo_type='LinearSum')
        full = ref.calculate()

        s = _beam()
        r = s.calculate()
        s.add_load_combination('C1', {'SLS': 1.4}, combo_type='LinearSum')
        r2 = solve_pending_cases(s, r)
        da = r2['combinations']['C1']['displacements']
        db = full['combinations']['C1']['displacements']
        self.assertAlmostEqual(
            max(abs(da[n][k] - db[n][k]) for n in da for k in range(3)),
            0.0, places=9)

    def test_it_falls_back_when_per_load_case_results_are_absent(self):
        """A model reloaded without the per-load-case results cannot superpose;
        solve_pending signals that with None so the caller does a full run."""
        s = _beam()
        self.assertIsNone(solve_pending_cases(s, {'analysis_cases': {}}))


if __name__ == '__main__':
    unittest.main()
