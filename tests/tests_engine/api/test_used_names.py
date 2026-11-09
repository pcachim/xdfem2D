"""Which fields and spectral functions a solved model must not let change.

An object load stores a field by *name* and re-evaluates it at solve time, so
editing that field would alter the analysed input behind the results. The
application locks exactly those — no more. A field sampled to a number when the
load was added (a point load's field) leaves no name and is not protected.
"""
from __future__ import annotations

import unittest

from context import Structure2D  # noqa: F401


def _model():
    s = Structure2D()
    s.add_material('C', elastic_modulus=30e6, unit_weight=0.0)
    s.add_section('S', 'C', b=0.3, h=0.5)
    s.add_node('L.p0', 0, 0)
    s.add_node('L.p1', 6, 0)
    s.add_geo_line('L', 0, 0, 6, 0, section_name='S', divisions=2)
    for n in ('ft', 'fg', 'fd', 'fk', 'fa', 'unused'):
        s.add_field(n, 'x')
    s.add_load_case('Q')
    return s


class TestUsedFieldNames(unittest.TestCase):

    def test_line_temperature_fields_count(self):
        s = _model()
        s.add_line_temperature_load('L', 'Q', field_name='ft',
                                    grad_field_name='fg')
        self.assertEqual(s.used_field_names(), {'ft', 'fg'})

    def test_line_distributed_and_spring_and_area(self):
        s = _model()
        s.add_line_distributed_load('L', 'Q', fx_field='fd')
        s.add_line_element_spring('L', kx_field='fk')
        s.add_node('A.p0', 0, 0)
        s.add_node('A.p1', 4, 0)
        s.add_node('A.p2', 4, 3)
        s.add_node('A.p3', 0, 3)
        s.add_tri_section('W', 'C', thickness=0.2)
        s.add_geo_rectangle('A', (0, 0), (4, 3), section_name='W')
        s.add_area_temperature_load('A', 'Q', field_name='fa')
        self.assertEqual(s.used_field_names(), {'fd', 'fk', 'fa'})

    def test_an_unused_field_is_not_listed(self):
        s = _model()
        s.add_line_distributed_load('L', 'Q', fx_field='fd')
        self.assertNotIn('unused', s.used_field_names())

    def test_a_field_sampled_to_a_value_leaves_no_name(self):
        """A point load samples the field and stores the number, so nothing is
        protected — the load no longer depends on the field."""
        s = _model()
        s.add_point_load('L.p0', 'Q', fx=s.field_node_values('fd').get('L.p0'))
        self.assertEqual(s.used_field_names(), set())


class TestUsedSpectralNames(unittest.TestCase):

    def test_a_spectrum_case_marks_its_function_used(self):
        s = _model()
        s.add_spectral_function('SPEC', points=[(0, 1), (1, 0.5)])
        s.add_spectral_function('OTHER', points=[(0, 1)])
        s.add_analysis_case('EQ', 'Spectrum')
        s.analysis_cases_by_id['EQ'].spectrum_id = 'SPEC'
        self.assertEqual(s.used_spectral_function_names(), {'SPEC'})

    def test_none_used_when_no_spectrum_case(self):
        s = _model()
        s.add_spectral_function('SPEC', points=[(0, 1)])
        self.assertEqual(s.used_spectral_function_names(), set())


if __name__ == '__main__':
    unittest.main()
