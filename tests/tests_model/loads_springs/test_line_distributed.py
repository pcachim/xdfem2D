"""Distributed load on a line object, expanded to its bars at solve time.

The Manage/Add pattern, applied to distributed loads. The engine part is the
expansion: each direction is a constant (uniform on every generated bar) or a
field (the trapezoid follows the field, sampled at each bar's two nodes), and
the two are independent. The strongest check is equilibrium — a uniform load
over a simply supported span carries its total to the two ends.
"""
from __future__ import annotations

import unittest

from context import Structure2D  # noqa: F401


def _obj(fx=0.0, fy=0.0, fx_field="", fy_field="", fields=None, div=2):
    s = Structure2D()
    s.add_material('C', elastic_modulus=30e6, unit_weight=0.0)
    s.add_section('S', 'C', b=0.3, h=0.5)
    s.add_node('L.p0', 0, 0)
    s.add_node('L.p1', 6, 0)
    s.add_geo_line('L', 0, 0, 6, 0, section_name='S', divisions=div)
    s.add_load_case('Q')
    for name, expr in (fields or {}).items():
        s.add_field(name, expr)
    s.add_line_distributed_load('L', 'Q', fx=fx, fy=fy,
                                fx_field=fx_field, fy_field=fy_field)
    return s


def _rows(mesh):
    return sorted(
        (round((mesh.nodes[d.element_id and
                mesh.bar_elements_by_id[d.element_id].node_i].x), 1),
         round(d.fxe, 2), round(d.fxd, 2), round(d.fye, 2), round(d.fyd, 2))
        for d in mesh.distributed_loads)


class TestExpansion(unittest.TestCase):

    def test_uniform_reaches_every_bar(self):
        from xdfem2d.geo_expand import expand_geometry
        mesh, _ = expand_geometry(_obj(fy=-10.0))
        self.assertEqual(len(mesh.distributed_loads), 2)
        self.assertEqual({(d.fye, d.fyd) for d in mesh.distributed_loads},
                         {(-10.0, -10.0)})

    def test_a_field_makes_the_trapezoid_follow_it(self):
        from xdfem2d.geo_expand import expand_geometry
        mesh, _ = expand_geometry(
            _obj(fy_field='q', fields={'q': 'x*-2'}))
        # bars 0-3 and 3-6: field -2x at ends → (0,-6) and (-6,-12)
        self.assertEqual(
            [(r[3], r[4]) for r in _rows(mesh)],
            [(-0.0, -6.0), (-6.0, -12.0)])

    def test_two_different_fields_for_x_and_y(self):
        from xdfem2d.geo_expand import expand_geometry
        mesh, _ = expand_geometry(
            _obj(fx_field='a', fy_field='b',
                 fields={'a': 'x', 'b': 'x*-1'}, div=1))
        d = mesh.distributed_loads[0]
        self.assertEqual((d.fxe, d.fxd), (0.0, 6.0))
        self.assertEqual((d.fye, d.fyd), (0.0, -6.0))

    def test_it_replaces_rather_than_stacks(self):
        s = _obj(fy=-10.0)
        s.add_line_distributed_load('L', 'Q', fy=-99.0)
        self.assertEqual(len(s.line_distributed_loads), 1)
        self.assertEqual(s.line_distributed_loads[0].fy, -99.0)


class TestEquilibrium(unittest.TestCase):

    def test_a_uniform_load_carries_its_total_to_the_supports(self):
        from xdfem2d.geo_expand import expand_geometry
        mesh, _ = expand_geometry(_obj(fy=-10.0))
        mesh.add_support('PIN', ux=True, uy=True)
        mesh.add_support('ROL', ux=False, uy=True)
        ends = sorted(mesh.nodes, key=lambda n: mesh.nodes[n].x)
        mesh.assign_support(ends[0], 'PIN')
        mesh.assign_support(ends[-1], 'ROL')
        mesh.add_analysis_case('LC', 'Linear', {'Q': 1.0})
        r = mesh.calculate()
        Ry = sum(v[1] for v in r['reactions']['Q'].values())
        self.assertAlmostEqual(Ry, 60.0, places=6)   # 10 kN/m x 6 m


class TestRoundTrip(unittest.TestCase):

    def test_it_survives_save_and_reload(self):
        from xdfem2d.structure_io import _from_dict, _to_dict
        s = _obj(fx_field='a', fy_field='b', fields={'a': 'x', 'b': 'x*-1'})
        back = _from_dict(_to_dict(s))
        l = back.line_distributed_loads[0]
        self.assertEqual((l.fx_field, l.fy_field, l.coord_sys),
                         ('a', 'b', 'global'))


if __name__ == '__main__':
    unittest.main()
