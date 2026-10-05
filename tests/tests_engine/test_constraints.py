"""Multi-point constraints — expansion (Phase 0) and penalty solve (Phase 1).

Two layers are checked:

* :func:`xdfem2d.constraints.expand_constraints` turns the high-level kinds
  ('equal_dof', 'rigid_link') and the raw 'equation' into the canonical
  ``sum(coef * dof) = value`` form. These are pure, NumPy-free assertions on the
  emitted DOF indices and coefficients.

* the penalty solver actually enforces the constraints, checked against
  closed-form answers: two nodes on parallel vertical springs tied by an
  equal-DOF share the load; a rigid link makes a slave node follow the master's
  rigid-body rotation.
"""
import unittest

from context import Structure2D
from xdfem2d.constraints import expand_constraints
from xdfem2d.models import COMPONENT_OFFSET


def _dof(s, nid, comp):
    return s.node_dof_index[nid] + COMPONENT_OFFSET[comp]


class TestExpansion(unittest.TestCase):
    def _two_nodes(self):
        s = Structure2D()
        s.add_node('A', 0.0, 0.0)
        s.add_node('B', 2.0, 1.0)
        return s

    def test_equal_dof_ties_each_extra_node_to_the_first(self):
        s = self._two_nodes()
        s.add_node('C', 4.0, 0.0)
        s.add_equal_dof(['A', 'B', 'C'], ['uy'])
        eqs = expand_constraints(s)
        # two equations: B-A and C-A, both on uy, rhs 0
        self.assertEqual(len(eqs), 2)
        for terms, value in eqs:
            self.assertEqual(value, 0.0)
            coefs = dict((d, c) for d, c in terms)
            self.assertEqual(coefs[_dof(s, 'A', 'uy')], -1.0)

    def test_equal_dof_multi_component(self):
        s = self._two_nodes()
        s.add_equal_dof(['A', 'B'], ['ux', 'tz'])
        eqs = expand_constraints(s)
        self.assertEqual(len(eqs), 2)   # one per component

    def test_rigid_link_kinematics(self):
        s = self._two_nodes()          # master A(0,0), slave B(2,1)
        s.add_rigid_link('A', ['B'])
        eqs = expand_constraints(s)
        self.assertEqual(len(eqs), 3)   # ux, uy, tz per slave
        by_slave_comp = {}
        for terms, value in eqs:
            self.assertEqual(value, 0.0)
            d = dict((dof, c) for dof, c in terms)
            # identify which slave component this equation is about (coef +1)
            for comp in ('ux', 'uy', 'tz'):
                if d.get(_dof(s, 'B', comp)) == 1.0:
                    by_slave_comp[comp] = d
        # ux_s - ux_m + dy*tz_m = 0  (dy = 1)
        self.assertAlmostEqual(by_slave_comp['ux'][_dof(s, 'A', 'ux')], -1.0)
        self.assertAlmostEqual(by_slave_comp['ux'][_dof(s, 'A', 'tz')], 1.0)
        # uy_s - uy_m - dx*tz_m = 0  (dx = 2)
        self.assertAlmostEqual(by_slave_comp['uy'][_dof(s, 'A', 'uy')], -1.0)
        self.assertAlmostEqual(by_slave_comp['uy'][_dof(s, 'A', 'tz')], -2.0)
        # tz_s - tz_m = 0
        self.assertAlmostEqual(by_slave_comp['tz'][_dof(s, 'A', 'tz')], -1.0)

    def test_disabled_constraint_is_skipped(self):
        s = self._two_nodes()
        c = s.add_equal_dof(['A', 'B'], ['uy'])
        c.enabled = False
        self.assertEqual(expand_constraints(s), [])

    def test_invalid_component_rejected(self):
        s = self._two_nodes()
        with self.assertRaises(ValueError):
            s.add_equal_dof(['A', 'B'], ['bogus'])


class TestPenaltySolve(unittest.TestCase):
    L, E, I = 2.0, 30e6, 1e-4      # EI = 3000 -> tip stiffness k = 3EI/L^3

    def _two_cantilevers(self, tie=True):
        """Two identical horizontal cantilevers, bases fixed, tips at N2 and N4.
        Each tip's vertical stiffness (rotation free) is k = 3EI/L^3. Optionally
        tie the two tips' vertical displacement."""
        s = Structure2D()
        s.add_material('M', elastic_modulus=self.E, unit_weight=0.0)
        s.add_section('S', 'M', b=0.1, h=0.2,
                      area_override=0.01, inertia_override=self.I)
        s.add_node('N1', 0.0, 0.0);  s.add_node('N2', self.L, 0.0)
        s.add_node('N3', 0.0, -1.0); s.add_node('N4', self.L, -1.0)
        s.add_bar_element('E1', 'N1', 'N2', 'S')
        s.add_bar_element('E2', 'N3', 'N4', 'S')
        s.add_support('FIX', ux=True, uy=True, tz=True)
        s.assign_support('N1', 'FIX'); s.assign_support('N3', 'FIX')
        s.add_load_case('LC')
        s.add_point_load('N2', 'LC', fy=-10.0)
        s.add_point_load('N4', 'LC', fy=-30.0)
        s.add_analysis_case('CASE', 'Linear', {'LC': 1.0})
        if tie:
            s.add_equal_dof(['N2', 'N4'], ['uy'])
        return s

    def test_equal_dof_shares_load_between_tips(self):
        s = self._two_cantilevers(tie=True)
        r = s.calculate()
        uy2 = r['displacements']['LC']['N2'][1]
        uy4 = r['displacements']['LC']['N4'][1]
        k = 3.0 * self.E * self.I / self.L ** 3
        expected = (-10.0 - 30.0) / (2.0 * k)   # parallel springs share the load
        self.assertAlmostEqual(uy2, uy4, places=6)
        self.assertAlmostEqual(uy2, expected, places=5)

    def test_untied_tips_deflect_independently(self):
        s = self._two_cantilevers(tie=False)
        r = s.calculate()
        k = 3.0 * self.E * self.I / self.L ** 3
        self.assertAlmostEqual(r['displacements']['LC']['N2'][1], -10.0 / k, places=5)
        self.assertAlmostEqual(r['displacements']['LC']['N4'][1], -30.0 / k, places=5)
        # untied, they are NOT equal
        self.assertNotAlmostEqual(r['displacements']['LC']['N2'][1],
                                  r['displacements']['LC']['N4'][1], places=5)

    def test_rigid_link_slave_follows_master(self):
        """A cantilever tip (master) carries a free slave node offset by (1, 0).
        Whatever the master does, the slave must be its rigid-body image:
        ux_s = ux_m - dy*tz_m, uy_s = uy_m + dx*tz_m, tz_s = tz_m."""
        s = Structure2D()
        s.add_material('M', elastic_modulus=self.E, unit_weight=0.0)
        s.add_section('S', 'M', b=0.1, h=0.2,
                      area_override=0.01, inertia_override=self.I)
        s.add_node('N1', 0.0, 0.0)
        s.add_node('N2', self.L, 0.0)      # master (beam tip)
        s.add_node('N3', self.L + 1.0, 0.0)  # slave, dx = 1, dy = 0
        s.add_bar_element('E1', 'N1', 'N2', 'S')
        s.add_support('FIX', ux=True, uy=True, tz=True)
        s.assign_support('N1', 'FIX')
        s.add_load_case('LC')
        s.add_point_load('N2', 'LC', fy=-10.0, mz=4.0)
        s.add_analysis_case('CASE', 'Linear', {'LC': 1.0})
        s.add_rigid_link('N2', ['N3'])

        r = s.calculate()
        ux_m, uy_m, tz_m = r['displacements']['LC']['N2']
        ux_s, uy_s, tz_s = r['displacements']['LC']['N3']
        dx, dy = 1.0, 0.0
        self.assertAlmostEqual(tz_s, tz_m, places=5)
        self.assertAlmostEqual(ux_s, ux_m - dy * tz_m, places=5)
        self.assertAlmostEqual(uy_s, uy_m + dx * tz_m, places=5)


class TestPersistence(unittest.TestCase):
    def _model_with_constraints(self):
        s = Structure2D()
        s.add_material('M', elastic_modulus=30e6, unit_weight=0.0)
        s.add_section('S', 'M', b=0.1, h=0.2,
                      area_override=0.01, inertia_override=1e-4)
        s.add_node('N1', 0.0, 0.0); s.add_node('N2', 2.0, 0.0)
        s.add_node('N3', 3.0, 0.0); s.add_node('N4', 2.0, -1.0)
        s.add_bar_element('E1', 'N1', 'N2', 'S')
        s.add_support('FIX', ux=True, uy=True, tz=True)
        s.assign_support('N1', 'FIX')
        s.add_rigid_link('N2', ['N3'], id='RIGID1')
        s.add_equal_dof(['N2', 'N4'], ['uy'], id='EQUAL1')
        c = s.add_constraint_equation([('N4', 'ux', 1.0)], value=0.0, id='EQ1')
        c.enabled = False
        s.add_load_case('LC')
        s.add_point_load('N2', 'LC', fy=-10.0)
        s.add_analysis_case('CASE', 'Linear', {'LC': 1.0})
        return s

    def test_round_trip_preserves_constraints(self):
        import tempfile, os
        from xdfem2d.structure_io import (save_structure_json,
                                          load_structure_json)
        s = self._model_with_constraints()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'm.x2d')
            save_structure_json(s, p)
            s2 = load_structure_json(p)

        self.assertEqual(set(s2.constraints), {'RIGID1', 'EQUAL1', 'EQ1'})
        r = s2.constraints['RIGID1']
        self.assertEqual(r.kind, 'rigid_link')
        self.assertEqual(r.master, 'N2')
        self.assertEqual(r.slaves, ['N3'])
        eq = s2.constraints['EQUAL1']
        self.assertEqual(eq.kind, 'equal_dof')
        self.assertEqual(eq.components, ['uy'])
        self.assertFalse(s2.constraints['EQ1'].enabled)
        # and the reloaded model still solves with identical results
        self.assertAlmostEqual(
            s.calculate()['displacements']['LC']['N3'][2],
            s2.calculate()['displacements']['LC']['N3'][2], places=8)

    def test_old_file_without_constraints_loads_clean(self):
        import tempfile, os, json
        from xdfem2d.structure_io import (save_structure_json,
                                          load_structure_json)
        s = self._model_with_constraints()
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'm.x2d')
            save_structure_json(s, p)
            with open(p) as fh:
                data = json.load(fh)
            data.pop('constraints', None)      # simulate a pre-feature file
            with open(p, 'w') as fh:
                json.dump(data, fh)
            s2 = load_structure_json(p)
        self.assertEqual(s2.constraints, {})


class TestValidation(unittest.TestCase):
    def _beam(self):
        s = Structure2D()
        s.add_material('M', elastic_modulus=30e6, unit_weight=0.0)
        s.add_section('S', 'M', b=0.1, h=0.2,
                      area_override=0.01, inertia_override=1e-4)
        s.add_node('N1', 0.0, 0.0); s.add_node('N2', 2.0, 0.0)
        s.add_bar_element('E1', 'N1', 'N2', 'S')
        s.add_support('FIX', ux=True, uy=True, tz=True)
        s.assign_support('N1', 'FIX')
        return s

    def _types(self, s):
        from xdfem2d.model_check import model_check
        return {i['type'] for i in model_check(s)}

    def test_clean_constraint_has_no_issue(self):
        s = self._beam()
        s.add_node('N3', 3.0, 0.0)
        s.add_bar_element('E2', 'N2', 'N3', 'S')   # give N3 a rotational DOF
        s.add_rigid_link('N2', ['N3'])
        types = self._types(s)
        self.assertFalse({'constraint_bad_ref', 'constraint_self_link',
                          'constraint_cycle', 'constraint_vs_support',
                          'constraint_no_rotation'} & types)

    def test_missing_node_reference(self):
        s = self._beam()
        s.add_equal_dof(['N2', 'GHOST'], ['uy'])
        self.assertIn('constraint_bad_ref', self._types(s))

    def test_self_link(self):
        s = self._beam()
        s.add_rigid_link('N2', ['N2'])
        self.assertIn('constraint_self_link', self._types(s))

    def test_mutual_rigid_links_are_a_cycle(self):
        s = self._beam()
        s.add_rigid_link('N1', ['N2'])
        s.add_rigid_link('N2', ['N1'])
        self.assertIn('constraint_cycle', self._types(s))

    def test_conflict_with_support(self):
        s = self._beam()               # N1 is fully fixed
        s.add_equal_dof(['N1', 'N2'], ['uy'])
        self.assertIn('constraint_vs_support', self._types(s))

    def test_tz_on_cst_only_node(self):
        s = Structure2D()
        s.add_material('M', elastic_modulus=30e6, unit_weight=0.0, poisson=0.2)
        s.add_tri_section('TS', 'M', thickness=0.2)   # CST by default
        for nid, (x, y) in {'A': (0, 0), 'B': (1, 0), 'C': (0, 1),
                            'D': (1, 1)}.items():
            s.add_node(nid, x, y)
        s.add_tri_element('T1', 'A', 'B', 'C', 'TS')
        s.add_tri_element('T2', 'B', 'D', 'C', 'TS')
        s.add_equal_dof(['B', 'C'], ['tz'])
        self.assertIn('constraint_no_rotation', self._types(s))

    def test_disabled_constraint_skips_solve_checks(self):
        s = self._beam()
        c = s.add_equal_dof(['N1', 'N2'], ['uy'])   # would clash with support
        c.enabled = False
        self.assertNotIn('constraint_vs_support', self._types(s))


class TestConstraintForces(unittest.TestCase):
    """Method 2: constraint force = residual of the physical stiffness at the
    coupled DOFs (stored in results['constraint_forces'])."""

    def _rigid_offset(self):
        s = Structure2D()
        s.add_material('M', elastic_modulus=30e6, unit_weight=0.0)
        s.add_section('S', 'M', b=0.1, h=0.2,
                      area_override=0.01, inertia_override=1e-4)
        s.add_node('N0', 0.0, 0.0)
        s.add_node('N1', 0.0, 3.0)      # master
        s.add_node('N2', 1.0, 3.0)      # slave, loaded
        s.add_bar_element('COL', 'N0', 'N1', 'S')
        s.add_support('FIX', ux=True, uy=True, tz=True); s.assign_support('N0', 'FIX')
        s.add_rigid_link('N1', ['N2'], id='RL')
        s.add_load_case('LC'); s.add_point_load('N2', 'LC', fy=-40.0)
        s.add_analysis_case('C', 'Linear', {'LC': 1.0})
        return s

    def test_slave_force_balances_the_applied_load(self):
        s = self._rigid_offset()
        r = s.calculate()
        cf = r['constraint_forces']['LC']
        # the link pulls the loaded slave up with ~40 kN (balances the -40 load)
        self.assertAlmostEqual(cf['N2'][1], 40.0, places=3)
        # master receives the transferred vertical force and moment
        self.assertAlmostEqual(cf['N1'][1], -40.0, places=3)
        self.assertAlmostEqual(cf['N1'][2], -40.0, places=2)   # M = P*e = 40

    def test_forces_present_on_analysis_case(self):
        s = self._rigid_offset()
        r = s.calculate()
        acf = r['analysis_cases']['C']['constraint_forces']
        self.assertAlmostEqual(acf['N2'][1], 40.0, places=3)

    def test_equal_dof_tie_force_is_equal_and_opposite(self):
        s = Structure2D()
        s.add_material('M', elastic_modulus=30e6, unit_weight=0.0)
        s.add_section('S', 'M', b=0.1, h=0.2,
                      area_override=0.01, inertia_override=1e-4)
        s.add_node('A1', 0, 0); s.add_node('B1', 3, 0)
        s.add_node('A2', 0, -1); s.add_node('B2', 3, -1)
        s.add_bar_element('E1', 'A1', 'B1', 'S')
        s.add_bar_element('E2', 'A2', 'B2', 'S')
        s.add_support('FIX', ux=True, uy=True, tz=True)
        s.assign_support('A1', 'FIX'); s.assign_support('A2', 'FIX')
        s.add_equal_dof(['B1', 'B2'], ['uy'], id='EQ')
        s.add_load_case('LC')
        s.add_point_load('B1', 'LC', fy=-20.0); s.add_point_load('B2', 'LC', fy=-60.0)
        s.add_analysis_case('C', 'Linear', {'LC': 1.0})
        r = s.calculate()
        cf = r['constraint_forces']['LC']
        self.assertAlmostEqual(cf['B1'][1], -cf['B2'][1], places=3)  # tie force

    def test_appears_in_results_tables(self):
        from xdfem2d.report_io import _results_tables
        s = self._rigid_offset()
        titles = [t[0] for t in _results_tables(s.calculate())]
        self.assertIn("Constraint forces", titles)

    def test_forces_on_linear_and_envelope_combinations(self):
        s = self._rigid_offset()
        s.add_load_combination('ULS', {'C': 1.35}, combo_type='LinearSum')
        s.add_load_combination('ENV', {'C': 1.0}, combo_type='Envelope')
        r = s.calculate()
        # LinearSum scales the force by the coefficient
        uls = r['combinations']['ULS']['constraint_forces']
        self.assertAlmostEqual(uls['N2'][1], 40.0 * 1.35, places=2)
        # Envelope carries a max/min band
        env = r['combinations']['ENV']['constraint_forces']
        self.assertIn('max', env)
        self.assertAlmostEqual(env['max']['N2'][1], 40.0, places=2)

    def test_combination_rows_in_results_table(self):
        from xdfem2d.report_io import _results_tables
        s = self._rigid_offset()
        s.add_load_combination('ULS', {'C': 1.35}, combo_type='LinearSum')
        s.add_load_combination('ENV', {'C': 1.0}, combo_type='Envelope')
        cf = next(t for t in _results_tables(s.calculate())
                  if t[0] == "Constraint forces")
        cases = {row[1] for row in cf[2]}
        self.assertIn('ULS', cases)
        self.assertIn('ENV (MAX)', cases)

    def test_no_table_without_constraints(self):
        from xdfem2d.report_io import _results_tables
        s = Structure2D()
        s.add_material('M', elastic_modulus=30e6, unit_weight=0.0)
        s.add_section('S', 'M', b=0.1, h=0.2,
                      area_override=0.01, inertia_override=1e-4)
        s.add_node('N0', 0, 0); s.add_node('N1', 3, 0)
        s.add_bar_element('E', 'N0', 'N1', 'S')
        s.add_support('FIX', ux=True, uy=True, tz=True); s.assign_support('N0', 'FIX')
        s.add_load_case('LC'); s.add_point_load('N1', 'LC', fy=-10.0)
        s.add_analysis_case('C', 'Linear', {'LC': 1.0})
        titles = [t[0] for t in _results_tables(s.calculate())]
        self.assertNotIn("Constraint forces", titles)


class TestModelJsonConstraints(unittest.TestCase):
    """missing_pieces() validation and round-trip build from a model dict."""

    def _base(self):
        # A minimal solvable model, as a dict, to attach constraints to.
        return {
            'nodes': [{'id': 'N1', 'x': 0, 'y': 0},
                      {'id': 'N2', 'x': 2, 'y': 0},
                      {'id': 'N3', 'x': 3, 'y': 0}],
            'materials': [{'name': 'M', 'elastic_modulus': 30e6,
                           'unit_weight': 0.0}],
            'sections': [{'name': 'S', 'material_name': 'M', 'b': 0.1,
                          'h': 0.2}],
            'bar_elements': [{'id': 'E1', 'node_i': 'N1', 'node_j': 'N2',
                              'section_name': 'S'}],
            'supports': [{'name': 'FIX', 'ux': True, 'uy': True, 'tz': True}],
            'support_assignments': [{'node_id': 'N1', 'support_name': 'FIX'}],
            'load_cases': [{'id': 'LC'}],
            'analysis_cases': [{'id': 'C', 'analysis_type': 'Linear',
                                'coefficients': {'LC': 1.0}}],
        }

    def _missing(self, data):
        from xdfem2d.model_json import missing_pieces
        return missing_pieces(data)

    def test_valid_constraints_add_no_complaint(self):
        d = self._base()
        d['constraints'] = [
            {'kind': 'rigid_link', 'master': 'N2', 'slaves': ['N3']},
            {'kind': 'equal_dof', 'nodes': ['N2', 'N3'], 'components': ['uy']},
        ]
        self.assertEqual(self._missing(d), [])

    def test_slaves_without_master_flagged(self):
        d = self._base()
        d['constraints'] = [{'kind': 'rigid_link', 'slaves': ['N3']}]
        self.assertTrue(any('has slaves but no' in m for m in self._missing(d)))

    def test_dangling_node_reference_flagged(self):
        d = self._base()
        d['constraints'] = [{'kind': 'equal_dof', 'nodes': ['N2', 'GHOST'],
                             'components': ['uy']}]
        self.assertTrue(any('GHOST' in m for m in self._missing(d)))

    def test_bad_component_flagged(self):
        d = self._base()
        d['constraints'] = [{'kind': 'equal_dof', 'nodes': ['N2', 'N3'],
                             'components': ['bogus']}]
        self.assertTrue(any('bogus' in m for m in self._missing(d)))

    def test_model_without_constraints_is_complete(self):
        self.assertEqual(self._missing(self._base()), [])

    def test_load_builds_the_constraints(self):
        from xdfem2d.model_json import load
        d = self._base()
        d['constraints'] = [{'kind': 'rigid_link', 'master': 'N2',
                             'slaves': ['N3'], 'id': 'R1'}]
        s = load(d)
        self.assertIn('R1', s.constraints)
        self.assertEqual(s.constraints['R1'].master, 'N2')


if __name__ == '__main__':
    unittest.main()
