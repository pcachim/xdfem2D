"""Global displacement field along a bar (postprocess.member_displacement_field).

Checked against the textbook simply-supported beam under a uniform load, where
every quantity has a closed form: the midspan deflection is 5wL⁴/384EI, the end
rotations are ∓wL³/24EI, and the rotation is zero at midspan by symmetry. The
field's end rotations must also land back on the nodal values the solver stored,
which is what makes a rotation diagram read continuously into the next member.
"""
import unittest

import numpy as np

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)
from xdfem2d.postprocess import member_displacement_field


def _ss_beam(L=6.0, w=20.0, E=30e6, I=3.125e-3, A=0.09):
    s = Structure2D()
    s.add_material('m', E, 25.0, poisson=0.2)
    s.add_section('sec', 'm', 0.3, 0.3, area_override=A, inertia_override=I)
    s.add_node('N1', 0.0, 0.0)
    s.add_node('N2', L, 0.0)
    s.add_bar_element('E1', 'N1', 'N2', 'sec')
    s.add_support('pin', True, True, False)
    s.assign_support('N1', 'pin')
    s.add_support('roll', False, True, False)
    s.assign_support('N2', 'roll')
    s.add_load_case('G')
    s.add_distributed_load('E1', 'G', fye=-w, fyd=-w)
    s.add_analysis_case('AC', 'Linear', {'G': 1.0})
    r = s.calculate()
    ac = r['analysis_cases']['AC']
    d = ac['element_distribution']['E1']
    disp = ac['displacements']
    return (s, np.asarray(d['M']), np.asarray(d['x']),
            disp['N1'], disp['N2'], E * I, w, L)


class TestMemberDisplacementField(unittest.TestCase):
    def test_midspan_deflection_matches_closed_form(self):
        s, M, xs, di, dj, EI, w, L = _ss_beam()
        st, ux, uy, rz = member_displacement_field(
            s.nodes['N1'], s.nodes['N2'], di, dj, M, xs, EI)
        mid = int(np.argmin(np.abs(st - L / 2)))
        self.assertAlmostEqual(uy[mid], -5 * w * L ** 4 / (384 * EI), places=6)

    def test_rotation_ends_match_nodes_and_theory(self):
        s, M, xs, di, dj, EI, w, L = _ss_beam()
        st, ux, uy, rz = member_displacement_field(
            s.nodes['N1'], s.nodes['N2'], di, dj, M, xs, EI)
        theory = w * L ** 3 / (24 * EI)
        # Ends: equal to the nodal rotations, which equal ∓wL³/24EI.
        self.assertAlmostEqual(rz[0], di[2], places=9)
        self.assertAlmostEqual(rz[-1], dj[2], places=9)
        self.assertAlmostEqual(rz[0], -theory, places=6)
        self.assertAlmostEqual(rz[-1], +theory, places=6)
        # Zero at midspan by symmetry.
        mid = int(np.argmin(np.abs(st - L / 2)))
        self.assertAlmostEqual(rz[mid], 0.0, places=7)

    def test_no_axial_load_gives_no_axial_displacement(self):
        s, M, xs, di, dj, EI, w, L = _ss_beam()
        st, ux, uy, rz = member_displacement_field(
            s.nodes['N1'], s.nodes['N2'], di, dj, M, xs, EI)
        self.assertLess(np.max(np.abs(ux)), 1e-9)
        # Supported ends do not deflect.
        self.assertAlmostEqual(uy[0], 0.0, places=9)
        self.assertAlmostEqual(uy[-1], 0.0, places=9)

    def test_stations_span_the_member(self):
        s, M, xs, di, dj, EI, w, L = _ss_beam()
        st, ux, uy, rz = member_displacement_field(
            s.nodes['N1'], s.nodes['N2'], di, dj, M, xs, EI)
        self.assertEqual(st.size, ux.size)
        self.assertEqual(ux.size, uy.size)
        self.assertEqual(uy.size, rz.size)
        self.assertAlmostEqual(st[0], 0.0, places=9)
        self.assertAlmostEqual(st[-1], L, places=6)

    def test_no_diagram_falls_back_to_linear_rotation(self):
        """With no moment diagram the rotation cannot be integrated, so it is a
        straight line between the end rotations — the right answer for a member
        with nothing loading it along its length."""
        class _N:
            def __init__(self, x, y):
                self.x, self.y = x, y
        di = [0.0, 0.0, -1e-3]
        dj = [0.0, 0.0, 3e-3]
        st, ux, uy, rz = member_displacement_field(
            _N(0, 0), _N(4, 0), di, dj, None, None, 1.0)
        self.assertAlmostEqual(rz[0], di[2], places=9)
        self.assertAlmostEqual(rz[-1], dj[2], places=9)
        self.assertAlmostEqual(rz[len(rz) // 2], 0.5 * (di[2] + dj[2]),
                               delta=abs(dj[2] - di[2]) * 0.2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
