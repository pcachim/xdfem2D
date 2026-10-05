"""Thermal load on CST triangles, checked against closed-form solutions.

This is the analogue of the bar temperature load, and it touches three places
that all have to agree: the load vector, the stress recovery, and the way both
carry through a combination. The strongest check is the free-expansion one — a
statically determinate triangle heated uniformly must expand with *zero*
stress. If the stress recovery forgot to subtract the thermal strain, that
triangle would report the stress of a fully restrained one, and the number
would look plausible.

The subtlety the design rests on is also pinned here: a constant-strain
triangle only feels the *mean* of its three nodal temperatures, so per-node
values on a single element equal the uniform mean. The gradient only does
something once the mesh is fine enough for neighbouring triangles to take
different means — which is why this pairs with the triangle-division feature.
"""
from __future__ import annotations

import unittest

from context import Structure2D  # noqa: F401  (puts src/ on the path)


E, NU, ALPHA, DT, T = 30e6, 0.2, 1e-5, 50.0, 0.2


def _one_triangle(plane_strain=False, restrained=False, dt=(DT, DT, DT),
                  factor=1.0):
    s = Structure2D()
    s.add_material('C', elastic_modulus=E, unit_weight=0.0,
                   poisson=NU, alpha=ALPHA)
    s.add_tri_section('W', 'C', thickness=T, plane_strain=plane_strain)
    s.add_node('N1', 0.0, 0.0)
    s.add_node('N2', 1.0, 0.0)
    s.add_node('N3', 0.0, 1.0)
    s.add_tri_element('T1', 'N1', 'N2', 'N3', 'W')
    if restrained:
        s.add_support('FIX', ux=True, uy=True)
        for n in ('N1', 'N2', 'N3'):
            s.assign_support(n, 'FIX')
    else:
        # Statically determinate: N1 pinned, N2 on a y-roller — free to expand.
        s.add_support('PIN', ux=True, uy=True)
        s.add_support('ROLy', ux=False, uy=True)
        s.assign_support('N1', 'PIN')
        s.assign_support('N2', 'ROLy')
    s.add_load_case('T')
    s.add_tri_temperature_load('T1', 'T', *dt)
    s.add_analysis_case('LC', 'Linear', {'T': factor})
    return s


class TestTheGuardianCase(unittest.TestCase):
    """A free triangle heated uniformly: zero stress, pure expansion. This is
    the one that fails loudly if the stress recovery does not subtract the
    thermal strain."""

    def setUp(self):
        self.r = _one_triangle(restrained=False).calculate()['tri_stress']['T']

    def test_stress_is_zero(self):
        s = self.r['T1']
        for comp in ('sx', 'sy', 'txy'):
            self.assertAlmostEqual(s[comp], 0.0, places=6)

    def test_it_actually_expanded(self):
        d = _one_triangle(restrained=False).calculate()['displacements']['T']
        self.assertAlmostEqual(d['N2'][0], ALPHA * DT * 1.0, places=9)


class TestFullyRestrained(unittest.TestCase):
    """Closed form for a triangle that cannot move at all."""

    def test_plane_stress(self):
        r = _one_triangle(restrained=True).calculate()
        sx = r['tri_stress']['T']['T1']['sx']
        self.assertAlmostEqual(sx, -E * ALPHA * DT / (1 - NU), places=3)

    def test_the_state_is_equibiaxial(self):
        s = _one_triangle(restrained=True).calculate()['tri_stress']['T']['T1']
        self.assertAlmostEqual(s['sx'], s['sy'], places=6)
        self.assertAlmostEqual(s['txy'], 0.0, places=6)

    def test_plane_strain_uses_the_1_plus_nu_factor(self):
        """The factor that must match the D it multiplies: fully restrained
        plane strain gives -EαΔT/(1-2ν), not -EαΔT/(1-ν)."""
        r = _one_triangle(restrained=True, plane_strain=True).calculate()
        sx = r['tri_stress']['T']['T1']['sx']
        self.assertAlmostEqual(sx, -E * ALPHA * DT / (1 - 2 * NU), places=3)


class TestOnlyTheMeanMatters(unittest.TestCase):
    """A CST cannot feel more than the mean of its three nodal temperatures."""

    def test_per_node_equals_the_uniform_mean(self):
        varied = _one_triangle(restrained=True, dt=(10.0, 50.0, 90.0))
        uniform = _one_triangle(restrained=True, dt=(50.0, 50.0, 50.0))
        a = varied.calculate()['tri_stress']['T']['T1']['sx']
        b = uniform.calculate()['tri_stress']['T']['T1']['sx']
        self.assertAlmostEqual(a, b, places=6)

    def test_the_mean_property_is_exposed(self):
        from xdfem2d.models import TriTemperatureLoad
        tl = TriTemperatureLoad('T1', 'T', 10.0, 20.0, 60.0)
        self.assertAlmostEqual(tl.dt_mean, 30.0)


class TestCarriesThroughCombinations(unittest.TestCase):
    """Stress is linear, so the thermal term of an analysis case or a
    combination is the same combination of the per-case thermal strains — the
    path that recovers combo stress from combined displacement must remove it
    too, or a ULS thermal stress comes out as if unrestrained."""

    def _model(self):
        s = _one_triangle(restrained=True, factor=1.5)
        s.add_load_combination('C1', {'LC': 1.0}, combo_type='LinearSum')
        return s.calculate()

    def test_analysis_case_scales_the_thermal_stress(self):
        r = self._model()
        sx = r['analysis_cases']['LC']['tri_stress']['T1']['sx']
        self.assertAlmostEqual(sx, 1.5 * (-E * ALPHA * DT / (1 - NU)), places=2)

    def test_combination_of_the_case_matches_it(self):
        r = self._model()
        sx = r['combinations']['C1']['tri_stress']['T1']['sx']
        self.assertAlmostEqual(sx, 1.5 * (-E * ALPHA * DT / (1 - NU)), places=2)


class TestBookkeeping(unittest.TestCase):

    def test_re_applying_accumulates_rather_than_replacing(self):
        """Two calls for the same triangle/case must stack, not overwrite --
        the solver sums unconditionally over every stored entry, so the
        restrained stress must reflect both temperature loads combined."""
        s = _one_triangle(restrained=True)
        s.add_tri_temperature_load('T1', 'T', 99.0, 99.0, 99.0)
        self.assertEqual(len(s.tri_temperature_loads), 2)
        self.assertEqual(s.tri_temperature_loads[0].dt_i, DT)
        self.assertEqual(s.tri_temperature_loads[1].dt_i, 99.0)

        r = s.calculate()
        sx = r['tri_stress']['T']['T1']['sx']
        self.assertAlmostEqual(sx, -E * ALPHA * (DT + 99.0) / (1 - NU),
                               places=2)

    def test_round_trip(self):
        from xdfem2d.structure_io import _from_dict, _to_dict
        s = _one_triangle(restrained=True, dt=(10.0, 20.0, 30.0))
        back = _from_dict(_to_dict(s))
        tl = back.tri_temperature_loads[0]
        self.assertEqual((tl.tri_id, tl.dt_i, tl.dt_j, tl.dt_k),
                         ('T1', 10.0, 20.0, 30.0))

    def test_no_thermal_load_leaves_stress_untouched(self):
        """A model with no temperature must recover exactly as before —
        the correction is skipped, not applied as zero with rounding."""
        s = _one_triangle(restrained=True)
        s.tri_temperature_loads.clear()
        s.point_loads.clear()
        r = s.calculate()
        self.assertAlmostEqual(r['tri_stress']['T']['T1']['sx'], 0.0, places=6)


class TestDivisionPreservesTemperature(unittest.TestCase):
    """The division feature must carry the field across — a refined model that
    lost its temperature would be a division that silently changed the load."""

    def test_the_field_survives_a_split(self):
        from xdfem2d import subdivide

        def area(struc, t):
            a = struc.nodes[t.node_i]; b = struc.nodes[t.node_j]
            c = struc.nodes[t.node_k]
            return abs((b.x - a.x) * (c.y - a.y)
                       - (c.x - a.x) * (b.y - a.y)) / 2.0

        s = Structure2D()
        s.add_material('C', elastic_modulus=E, unit_weight=0.0,
                       poisson=NU, alpha=ALPHA)
        s.add_tri_section('W', 'C', thickness=T)
        s.add_node('N1', 0, 0)
        s.add_node('N2', 4, 0)
        s.add_node('N3', 0, 3)
        s.add_tri_element('T1', 'N1', 'N2', 'N3', 'W')
        s.add_load_case('T')
        s.add_tri_temperature_load('T1', 'T', 0.0, 60.0, 0.0)  # mean 20
        subdivide.divide_triangles(s, ['T1'], 3)
        loads = s.tri_temperature_loads
        self.assertEqual(len(loads), 9)
        tot = sum(area(s, s.tri_elements_by_id[t.tri_id]) for t in loads)
        wavg = sum(t.dt_mean * area(s, s.tri_elements_by_id[t.tri_id])
                   for t in loads) / tot
        self.assertAlmostEqual(wavg, 20.0, places=6)


if __name__ == '__main__':
    unittest.main()


class TestAreaObjectTemperature(unittest.TestCase):
    """A temperature on a rectangle or surface reaches the triangles it meshes
    into — the object's cells do not exist until expand_geometry, so the load
    is stored against the object and expanded there, like a surface edge load.
    """

    def _wall(self, field=False):
        from xdfem2d import Structure2D
        s = Structure2D()
        s.add_material('C', elastic_modulus=E, unit_weight=0.0,
                       poisson=NU, alpha=ALPHA)
        s.add_tri_section('W', 'C', thickness=0.2)
        for nid, (x, y) in {'R.p0': (0, 0), 'R.p1': (4, 0),
                            'R.p2': (4, 3), 'R.p3': (0, 3)}.items():
            s.add_node(nid, x, y)
        s.add_geo_rectangle('R', (0, 0), (4, 3),
                            section_name='W', target_size=1.0)
        s.add_load_case('T')
        if field:
            s.add_field('g', 'y*10')
            s.add_area_temperature_load('R', 'T', field_name='g')
        else:
            s.add_area_temperature_load('R', 'T', dt_uniform=DT)
        return s

    def test_uniform_reaches_every_generated_triangle(self):
        from xdfem2d.geo_expand import expand_geometry
        mesh, _ = expand_geometry(self._wall())
        self.assertGreater(len(mesh.tri_elements), 0)
        self.assertEqual(len(mesh.tri_temperature_loads),
                         len(mesh.tri_elements))
        self.assertEqual({t.dt_mean for t in mesh.tri_temperature_loads}, {DT})

    def test_a_field_varies_across_the_mesh(self):
        from xdfem2d.geo_expand import expand_geometry
        mesh, _ = expand_geometry(self._wall(field=True))
        means = {round(t.dt_mean, 3) for t in mesh.tri_temperature_loads}
        self.assertGreater(len(means), 1)   # a staircase of means over y

    def test_it_replaces_rather_than_stacks(self):
        s = self._wall()
        s.add_area_temperature_load('R', 'T', dt_uniform=99.0)
        self.assertEqual(len(s.area_temperature_loads), 1)
        self.assertEqual(s.area_temperature_loads[0].dt_uniform, 99.0)

    def test_round_trip(self):
        from xdfem2d.structure_io import _from_dict, _to_dict
        back = _from_dict(_to_dict(self._wall(field=True)))
        a = back.area_temperature_loads[0]
        self.assertEqual((a.object_id, a.load_case_id, a.field_name),
                         ('R', 'T', 'g'))

    def test_a_restrained_meshed_wall_has_the_closed_form_stress(self):
        """Uniform ΔT, every mesh node fixed: each cell is the fully restrained
        equibiaxial state, so the load actually reached them."""
        from xdfem2d.geo_expand import expand_geometry
        s = self._wall()
        mesh, _ = expand_geometry(s)
        for nid in list(mesh.nodes):
            mesh.add_support('FIX', ux=True, uy=True)
            mesh.assign_support(nid, 'FIX')
        mesh.add_analysis_case('LC', 'Linear', {'T': 1.0})
        r = mesh.calculate()
        expected = -E * ALPHA * DT / (1 - NU)
        for sig in r['tri_stress']['T'].values():
            self.assertAlmostEqual(sig['sx'], expected, places=2)


class TestLineObjectTemperature(unittest.TestCase):
    """A temperature on a line / arc / polyline reaches the bars it subdivides
    into, the line-object counterpart of the area temperature. A bar keeps both
    thermal effects — the uniform ΔT and the through-depth gradient."""

    def _beam_object(self, field=False, grad=0.0):
        from xdfem2d import Structure2D
        s = Structure2D()
        s.add_material('C', elastic_modulus=E, unit_weight=0.0, alpha=ALPHA)
        s.add_section('S', 'C', b=0.3, h=0.5)
        s.add_node('L.p0', 0, 0)
        s.add_node('L.p1', 6, 0)
        s.add_geo_line('L', 0, 0, 6, 0, section_name='S', divisions=3)
        s.add_load_case('T')
        if field:
            s.add_field('g', 'x*10')
            s.add_line_temperature_load('L', 'T', field_name='g',
                                        dt_gradient=grad)
        else:
            s.add_line_temperature_load('L', 'T', dt_uniform=DT,
                                        dt_gradient=grad)
        return s

    def test_uniform_reaches_every_generated_bar(self):
        from xdfem2d.geo_expand import expand_geometry
        mesh, _ = expand_geometry(self._beam_object())
        self.assertEqual(len(mesh.bar_elements), 3)
        self.assertEqual(len(mesh.temperature_loads), 3)
        self.assertEqual({t.delta_t_uniform for t in mesh.temperature_loads},
                         {DT})

    def test_the_gradient_is_carried_to_every_bar(self):
        from xdfem2d.geo_expand import expand_geometry
        mesh, _ = expand_geometry(self._beam_object(grad=15.0))
        self.assertEqual({t.delta_t_gradient for t in mesh.temperature_loads},
                         {15.0})

    def test_a_field_is_sampled_at_each_bar_midpoint(self):
        from xdfem2d.geo_expand import expand_geometry
        mesh, _ = expand_geometry(self._beam_object(field=True))
        # x*10 at midpoints x = 1, 3, 5 → 10, 30, 50
        self.assertEqual(
            sorted(round(t.delta_t_uniform, 1)
                   for t in mesh.temperature_loads),
            [10.0, 30.0, 50.0])

    def test_a_restrained_bar_carries_the_axial_thermal_force(self):
        """-EA·α·ΔT in every generated bar — proof the load reached them."""
        from xdfem2d.geo_expand import expand_geometry
        mesh, _ = expand_geometry(self._beam_object())
        mesh.add_support('FIX', ux=True, uy=True, tz=True)
        for nid, n in mesh.nodes.items():
            if n.x in (0.0, 6.0):
                mesh.assign_support(nid, 'FIX')
        mesh.add_analysis_case('LC', 'Linear', {'T': 1.0})
        r = mesh.calculate()
        A = 0.3 * 0.5
        for ef in r['element_forces']['T'].values():
            self.assertAlmostEqual(ef['i'][0], -E * A * ALPHA * DT, places=1)

    def test_it_replaces_rather_than_stacks(self):
        s = self._beam_object()
        s.add_line_temperature_load('L', 'T', dt_uniform=99.0)
        self.assertEqual(len(s.line_temperature_loads), 1)
        self.assertEqual(s.line_temperature_loads[0].dt_uniform, 99.0)

    def test_round_trip(self):
        from xdfem2d.structure_io import _from_dict, _to_dict
        back = _from_dict(_to_dict(self._beam_object(field=True, grad=15.0)))
        l = back.line_temperature_loads[0]
        self.assertEqual((l.object_id, l.dt_gradient, l.field_name),
                         ('L', 15.0, 'g'))


class TestLineGradientFromField(unittest.TestCase):
    """Either component of a line-object temperature may come from a field, the
    same one or different ones. A field over the gradient sets how its
    magnitude varies in space; the through-depth profile stays linear."""

    def _obj(self, uni_field="", grad_field="", dt=0.0, grad=0.0):
        from xdfem2d import Structure2D
        s = Structure2D()
        s.add_material('C', elastic_modulus=E, unit_weight=0.0, alpha=ALPHA)
        s.add_section('S', 'C', b=0.3, h=0.5)
        s.add_node('L.p0', 0, 0)
        s.add_node('L.p1', 6, 0)
        s.add_geo_line('L', 0, 0, 6, 0, section_name='S', divisions=3)
        s.add_load_case('T')
        s.add_field('u', 'x*10')
        s.add_field('g', 'x*2')
        s.add_line_temperature_load('L', 'T', dt_uniform=dt, dt_gradient=grad,
                                    field_name=uni_field,
                                    grad_field_name=grad_field)
        return s

    def _rows(self, s):
        from xdfem2d.geo_expand import expand_geometry
        mesh, _ = expand_geometry(s)
        return sorted(
            (round((mesh.nodes[b.node_i].x + mesh.nodes[b.node_j].x) / 2, 1),
             round(t.delta_t_uniform, 1), round(t.delta_t_gradient, 1))
            for b, t in zip(mesh.bar_elements, mesh.temperature_loads))

    def test_two_different_fields(self):
        rows = self._rows(self._obj(uni_field='u', grad_field='g'))
        self.assertEqual(rows, [(1.0, 10.0, 2.0), (3.0, 30.0, 6.0),
                                (5.0, 50.0, 10.0)])

    def test_the_same_field_for_both(self):
        rows = self._rows(self._obj(uni_field='u', grad_field='u'))
        self.assertEqual(rows, [(1.0, 10.0, 10.0), (3.0, 30.0, 30.0),
                                (5.0, 50.0, 50.0)])

    def test_gradient_from_field_with_numeric_uniform(self):
        rows = self._rows(self._obj(dt=99.0, grad_field='g'))
        self.assertTrue(all(u == 99.0 for _x, u, _g in rows))
        self.assertEqual([g for _x, _u, g in rows], [2.0, 6.0, 10.0])

    def test_round_trip_keeps_both_field_names(self):
        from xdfem2d.structure_io import _from_dict, _to_dict
        back = _from_dict(_to_dict(self._obj(uni_field='u', grad_field='g')))
        l = back.line_temperature_loads[0]
        self.assertEqual((l.field_name, l.grad_field_name), ('u', 'g'))
