"""Tests for unilateral (tension-/compression-only) node springs solved by
NonLinear analysis cases (xdfem2d.solver active-set solve).

A horizontal cantilever (N1 fixed, N2 free, 5 m) carries a vertical point load
at the free end and has a vertical spring at N2. The spring's unilateral mode
is exercised in both its active and inactive regimes and checked against the
equivalent linear models (bilateral spring / no spring).
"""
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)


def _cantilever(spring_mode, fy, kt=0.0):
    s = Structure2D()
    s.add_material('M', elastic_modulus=2.1e8, unit_weight=0.0)
    s.add_section('S', 'M', b=0.1, h=0.1,
                  area_override=0.01, inertia_override=1e-4)
    s.add_node('N1', 0.0, 0.0)
    s.add_node('N2', 5.0, 0.0)
    s.add_support('fix', ux=True, uy=True, tz=True)
    s.assign_support('N1', 'fix')
    s.add_bar_element('E1', 'N1', 'N2', 'S')
    if spring_mode is not None:
        s.add_node_spring('N2', kx=0.0, ky=1e5, kt=kt, mode_y=spring_mode)
    s.add_load_case('LC1')
    s.add_point_load('N2', 'LC1', fy=fy)
    s.add_analysis_case('NL', 'NonLinear', {'LC1': 1.0})
    s.add_analysis_case('LIN', 'Linear', {'LC1': 1.0})
    return s


class TestUnilateralSprings(unittest.TestCase):

    def test_compression_spring_active(self):
        # Downward load → N2 moves down (uy<0) → compression-only spring ACTIVE.
        acs = _cantilever('compression', -50.0).calculate()['analysis_cases']
        self.assertTrue(acs['NL']['converged'])
        self.assertAlmostEqual(acs['NL']['displacements']['N2'][1],
                               acs['LIN']['displacements']['N2'][1], places=6)

    def test_compression_spring_inactive(self):
        # Upward load → N2 moves up (uy>0) → compression-only spring INACTIVE,
        # so the response matches the cantilever with no spring at all.
        nl = _cantilever('compression', +50.0).calculate()['analysis_cases']['NL']
        ref = Structure2D()
        ref.add_material('M', elastic_modulus=2.1e8, unit_weight=0.0)
        ref.add_section('S', 'M', b=0.1, h=0.1,
                        area_override=0.01, inertia_override=1e-4)
        ref.add_node('N1', 0.0, 0.0); ref.add_node('N2', 5.0, 0.0)
        ref.add_support('fix', ux=True, uy=True, tz=True)
        ref.assign_support('N1', 'fix')
        ref.add_bar_element('E1', 'N1', 'N2', 'S')
        ref.add_load_case('LC1'); ref.add_point_load('N2', 'LC1', fy=+50.0)
        ref.add_analysis_case('LIN', 'Linear', {'LC1': 1.0})
        ref_uy = ref.calculate()['analysis_cases']['LIN']['displacements']['N2'][1]

        self.assertTrue(nl['converged'])
        self.assertAlmostEqual(nl['displacements']['N2'][1], ref_uy, places=6)
        self.assertAlmostEqual(nl['spring_forces']['N2'][1], 0.0, places=6)

    def test_tension_spring_active(self):
        # Upward load + tension-only spring → spring ACTIVE (resists), matching
        # the bilateral linear spring result.
        acs = _cantilever('tension', +50.0).calculate()['analysis_cases']
        self.assertTrue(acs['NL']['converged'])
        self.assertAlmostEqual(acs['NL']['displacements']['N2'][1],
                               acs['LIN']['displacements']['N2'][1], places=6)

    def test_tension_spring_inactive_matches_no_spring(self):
        # Downward load + tension-only spring → INACTIVE.
        nl = _cantilever('tension', -50.0).calculate()['analysis_cases']['NL']
        self.assertTrue(nl['converged'])
        self.assertAlmostEqual(nl['spring_forces']['N2'][1], 0.0, places=6)

    def test_nonlinear_case_has_element_distribution(self):
        # The NL case must carry N/V/M diagrams (like a Linear case) so the GUI
        # can render them. When the spring is active they match the linear case.
        acs = _cantilever('compression', -50.0).calculate()['analysis_cases']
        self.assertIn('element_distribution', acs['NL'])
        import numpy as np
        m_nl = np.asarray(acs['NL']['element_distribution']['E1']['M'])
        m_lin = np.asarray(acs['LIN']['element_distribution']['E1']['M'])
        self.assertTrue(np.allclose(m_nl, m_lin))

    def test_nonlinear_without_modes_equals_linear(self):
        # A NonLinear case with only bilateral springs must match the Linear case.
        acs = _cantilever('both', -50.0).calculate()['analysis_cases']
        self.assertAlmostEqual(acs['NL']['displacements']['N2'][1],
                               acs['LIN']['displacements']['N2'][1], places=9)

    def test_spring_mode_roundtrip(self):
        # mode_* survives a to_dict / from_dict round-trip.
        from xdfem2d.structure_io import _to_dict, _from_dict
        s = _cantilever('compression', -50.0)
        s2 = _from_dict(_to_dict(s))
        self.assertEqual(s2.node_springs['N2'].mode_y, 'compression')
        self.assertEqual(s2.node_springs['N2'].mode_x, 'both')

    def test_invalid_mode_rejected(self):
        s = Structure2D()
        s.add_node('N1', 0.0, 0.0)
        with self.assertRaises(ValueError):
            s.add_node_spring('N1', ky=1.0, mode_y='bogus')


def _beam_on_winkler(spring_mode, fy):
    """Cantilever E1 (N1 fixed → N2 free) with a global-Y element foundation
    spring and a vertical end load."""
    s = Structure2D()
    s.add_material('M', elastic_modulus=2.1e8, unit_weight=0.0)
    s.add_section('S', 'M', b=0.1, h=0.1,
                  area_override=0.01, inertia_override=1e-4)
    s.add_node('N1', 0.0, 0.0)
    s.add_node('N2', 5.0, 0.0)
    s.add_support('fix', ux=True, uy=True, tz=True)
    s.assign_support('N1', 'fix')
    s.add_bar_element('E1', 'N1', 'N2', 'S')
    if spring_mode is not None:
        s.add_element_spring('E1', kx=0.0, ky=2e4, coord_sys='global',
                             mode_y=spring_mode)
    s.add_load_case('LC1')
    s.add_point_load('N2', 'LC1', fy=fy)
    s.add_analysis_case('NL', 'NonLinear', {'LC1': 1.0})
    s.add_analysis_case('LIN', 'Linear', {'LC1': 1.0})
    return s


class TestUnilateralElementSprings(unittest.TestCase):

    def test_compression_element_spring_active(self):
        # Downward load → free end moves down → compression-only Winkler ACTIVE.
        acs = _beam_on_winkler('compression', -40.0).calculate()['analysis_cases']
        self.assertTrue(acs['NL']['converged'])
        self.assertAlmostEqual(acs['NL']['displacements']['N2'][1],
                               acs['LIN']['displacements']['N2'][1], places=6)

    def test_compression_element_spring_inactive(self):
        # Upward load → free end moves up → compression-only Winkler INACTIVE,
        # so the response matches the bare cantilever (no element spring).
        nl = _beam_on_winkler('compression', +40.0).calculate()['analysis_cases']['NL']
        ref = _beam_on_winkler(None, +40.0).calculate()['analysis_cases']['LIN']
        self.assertTrue(nl['converged'])
        self.assertAlmostEqual(nl['displacements']['N2'][1],
                               ref['displacements']['N2'][1], places=6)

    def test_nonlinear_bilateral_element_spring_equals_linear(self):
        acs = _beam_on_winkler('both', -40.0).calculate()['analysis_cases']
        self.assertAlmostEqual(acs['NL']['displacements']['N2'][1],
                               acs['LIN']['displacements']['N2'][1], places=9)

    def test_element_spring_mode_roundtrip(self):
        from xdfem2d.structure_io import _to_dict, _from_dict
        s = _beam_on_winkler('compression', -40.0)
        s2 = _from_dict(_to_dict(s))
        self.assertEqual(s2.element_springs['E1'].mode_y, 'compression')
        self.assertEqual(s2.element_springs['E1'].mode_x, 'both')

    def test_invalid_element_mode_rejected(self):
        s = Structure2D()
        s.add_material('M', elastic_modulus=2.1e8, unit_weight=0.0)
        s.add_section('S', 'M', b=0.1, h=0.1,
                      area_override=0.01, inertia_override=1e-4)
        s.add_node('N1', 0.0, 0.0); s.add_node('N2', 5.0, 0.0)
        s.add_bar_element('E1', 'N1', 'N2', 'S')
        with self.assertRaises(ValueError):
            s.add_element_spring('E1', ky=1.0, mode_y='bogus')


if __name__ == '__main__':
    unittest.main()
