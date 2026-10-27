"""QuadAreaSpring + QuadTemperatureLoad — the two features Phase 6's own
implementation notes flagged as missing, forcing ``geo_expand._expand_surface``
to fall back to an all-triangle mesh whenever a surface carried an area
spring or a temperature load. Both are now real models with real element
kernels (``quad_elements.q4_thermal_load``/``qm6_thermal_load``,
``quad_elements_dkt4.dkt4_thermal_load``,
``quad_elements_mitc4.mitc4_thermal_load``), so this mirrors the existing
triangle validation style — closed-form / free-expansion checks, run through
the full ``Structure2D.calculate()`` pipeline — one level up from
``tests_engine``'s kernel-only checks.

Units: m, kN, kNm.
"""
from __future__ import annotations

import unittest

from context import Structure2D  # noqa: F401  (puts src/ on the path)


E, NU, ALPHA, DT, T = 30e6, 0.2, 1e-5, 50.0, 0.2


def _quad_grid(s, L, n, sec):
    """n×n structured mesh of the square [0,L]^2, quads only. Returns
    {(i, j): node_id}."""
    ids = {}
    for i in range(n + 1):
        for j in range(n + 1):
            nid = f'N{i}_{j}'
            ids[(i, j)] = nid
            s.add_node(nid, L * i / n, L * j / n)
    k = 0
    for i in range(n):
        for j in range(n):
            a, b, c, d = (ids[(i, j)], ids[(i + 1, j)],
                          ids[(i + 1, j + 1)], ids[(i, j + 1)])
            s.add_quad_element(f'Q{k}', a, b, c, d, sec)
            k += 1
    return ids


# ---------------------------------------------------------------------------
# QuadAreaSpring (Winkler bed) — MITC4 and DKT4
# ---------------------------------------------------------------------------

class TestQuadWinklerAreaSpring(unittest.TestCase):
    """The quad analogue of test_dkt.test_winkler_area_spring_uniform_bed:
    kz·A/4 lumped to each vertex's w DOF must exactly balance a uniform
    pz·A/4, so a slab resting entirely on a Winkler bed settles as a rigid
    body with zero moment everywhere."""

    def _run(self, formulation):
        s = Structure2D(domain='plate')
        s.add_material('C', E, 25.0, poisson=NU)
        s.add_quad_section('P', 'C', thickness=T, formulation=formulation)
        s.add_load_case('LC')
        L, n, pz, kz = 4.0, 6, -8.0, 5000.0
        ids = _quad_grid(s, L, n, 'P')
        for qid in list(s.quad_elements_by_id):
            s.add_area_load(qid, 'LC', pz=pz)
            s.add_area_spring(qid, kz=kz)
        s.add_support('R', tx=True, ty=True)   # bed has no rotational stiffness
        for nid in ids.values():
            s.assign_support(nid, 'R')
        r = s.calculate()
        ws = [d[0] for d in r['displacements']['LC'].values()]
        self.assertAlmostEqual(min(ws), pz / kz, places=6)
        self.assertAlmostEqual(max(ws) - min(ws), 0.0, places=9)
        for v in r['tri_stress']['LC'].values():
            self.assertAlmostEqual(abs(v['mx']) + abs(v['my']) + abs(v['mxy']),
                                   0.0, places=6)

    def test_mitc4(self):
        self._run('MITC4')

    def test_dkt4(self):
        self._run('DKT4')


class TestQuadAreaSpringStiffensSlab(unittest.TestCase):
    """Adding a Winkler bed under a simply supported, loaded MITC4 slab
    reduces the centre deflection, exactly as it does for the CST/DKT case."""

    def test_bed_reduces_centre_deflection(self):
        L, q, n, kz = 6.0, -10.0, 8, 2000.0

        def build(bed):
            s = Structure2D(domain='plate')
            s.add_material('C', E, 25.0, poisson=NU)
            s.add_quad_section('P', 'C', thickness=T, formulation='MITC4')
            s.add_load_case('LC')
            ids = _quad_grid(s, L, n, 'P')
            s.add_support('SS', w=True)
            for (i, j), nid in ids.items():
                if i in (0, n) or j in (0, n):
                    s.assign_support(nid, 'SS')
            for qid in list(s.quad_elements_by_id):
                s.add_area_load(qid, 'LC', pz=q)
                if bed:
                    s.add_area_spring(qid, kz=kz)
            return s, ids[(n // 2, n // 2)]

        s0, c = build(False)
        s1, c = build(True)
        w0 = s0.calculate()['displacements']['LC'][c][0]
        w1 = s1.calculate()['displacements']['LC'][c][0]
        self.assertLess(w1, 0)
        self.assertLess(abs(w1), abs(w0))


class TestQuadAreaSpringBookkeeping(unittest.TestCase):

    def test_add_area_spring_dispatches_to_quad(self):
        s = Structure2D(domain='plate')
        s.add_material('C', E, 25.0, poisson=NU)
        s.add_quad_section('P', 'C', thickness=T, formulation='MITC4')
        s.add_node('N1', 0, 0); s.add_node('N2', 1, 0)
        s.add_node('N3', 1, 1); s.add_node('N4', 0, 1)
        s.add_quad_element('Q1', 'N1', 'N2', 'N3', 'N4', 'P')
        s.add_area_spring('Q1', kz=4200.0)
        self.assertEqual(len(s.quad_area_springs), 1)
        self.assertEqual(s.quad_area_springs[0].kz, 4200.0)
        self.assertEqual(s.quad_area_springs[0].quad_id, 'Q1')

    def test_remove_area_spring_purges_it(self):
        s = Structure2D(domain='plate')
        s.add_material('C', E, 25.0, poisson=NU)
        s.add_quad_section('P', 'C', thickness=T, formulation='MITC4')
        s.add_node('N1', 0, 0); s.add_node('N2', 1, 0)
        s.add_node('N3', 1, 1); s.add_node('N4', 0, 1)
        s.add_quad_element('Q1', 'N1', 'N2', 'N3', 'N4', 'P')
        s.add_area_spring('Q1', kz=100.0)
        s.remove_area_spring('Q1')
        self.assertEqual(s.quad_area_springs, [])

    def test_remove_quad_element_purges_its_spring(self):
        s = Structure2D(domain='plate')
        s.add_material('C', E, 25.0, poisson=NU)
        s.add_quad_section('P', 'C', thickness=T, formulation='MITC4')
        s.add_node('N1', 0, 0); s.add_node('N2', 1, 0)
        s.add_node('N3', 1, 1); s.add_node('N4', 0, 1)
        s.add_quad_element('Q1', 'N1', 'N2', 'N3', 'N4', 'P')
        s.add_area_spring('Q1', kz=100.0)
        s.remove_quad_element('Q1')
        self.assertEqual(s.quad_area_springs, [])

    def test_a_winkler_area_spring_in_a_plane_model_is_refused(self):
        s = Structure2D()
        s.add_material('C', E, 25.0, poisson=NU)
        s.add_quad_section('P', 'C', thickness=T, formulation='Q4')
        s.add_node('N1', 0, 0); s.add_node('N2', 1, 0)
        s.add_node('N3', 1, 1); s.add_node('N4', 0, 1)
        s.add_quad_element('Q1', 'N1', 'N2', 'N3', 'N4', 'P')
        s.add_area_spring('Q1', kz=100.0)
        self.assertTrue(any('Winkler' in x for x in s.domain_problems()))

    def test_round_trip(self):
        from xdfem2d.structure_io import _to_dict, _from_dict
        s = Structure2D(domain='plate')
        s.add_material('C', E, 25.0, poisson=NU)
        s.add_quad_section('P', 'C', thickness=T, formulation='MITC4')
        s.add_node('N1', 0, 0); s.add_node('N2', 1, 0)
        s.add_node('N3', 1, 1); s.add_node('N4', 0, 1)
        s.add_quad_element('Q1', 'N1', 'N2', 'N3', 'N4', 'P')
        s.add_area_spring('Q1', kz=4200.0)
        q = _from_dict(_to_dict(s))
        self.assertEqual(q.quad_area_springs[0].kz, 4200.0)
        self.assertEqual(q.quad_area_springs[0].quad_id, 'Q1')


# ---------------------------------------------------------------------------
# QuadTemperatureLoad — Q4/QM6 (in-plane) free expansion
# ---------------------------------------------------------------------------

def _one_quad(formulation='Q4', restrained=False, dt=(DT, DT, DT, DT)):
    s = Structure2D()
    s.add_material('C', elastic_modulus=E, unit_weight=0.0,
                   poisson=NU, alpha=ALPHA)
    s.add_quad_section('P', 'C', thickness=T, formulation=formulation)
    s.add_node('N1', 0.0, 0.0)
    s.add_node('N2', 1.0, 0.0)
    s.add_node('N3', 1.0, 1.0)
    s.add_node('N4', 0.0, 1.0)
    s.add_quad_element('Q1', 'N1', 'N2', 'N3', 'N4', 'P')
    if restrained:
        s.add_support('FIX', ux=True, uy=True)
        for n in ('N1', 'N2', 'N3', 'N4'):
            s.assign_support(n, 'FIX')
    else:
        # Statically determinate: N1 pinned, N2 on a y-roller.
        s.add_support('PIN', ux=True, uy=True)
        s.add_support('ROLy', ux=False, uy=True)
        s.assign_support('N1', 'PIN')
        s.assign_support('N2', 'ROLy')
    s.add_load_case('T')
    s.add_quad_temperature_load('Q1', 'T', *dt)
    return s


class TestQuadThermalFreeExpansion(unittest.TestCase):
    """A free Q4/QM6 quad heated uniformly must expand with zero stress —
    the membrane analogue of the CST's guardian case, and the check that
    proves qm6_thermal_load's condensation is consistent with qm6_stiffness's
    (a naive/inconsistent condensation would report the stress of a
    partially-restrained element instead of zero)."""

    def _check(self, formulation):
        r = _one_quad(formulation, restrained=False).calculate()['tri_stress']['T']
        s = r['Q1']
        for comp in ('sx', 'sy', 'txy'):
            self.assertAlmostEqual(s[comp], 0.0, places=5)

    def test_q4(self):
        self._check('Q4')

    def test_qm6(self):
        self._check('QM6')


class TestQuadThermalFullyRestrained(unittest.TestCase):
    """Closed form for a fully restrained quad: the equibiaxial thermal
    stress, same as the fully restrained CST case."""

    def _check(self, formulation):
        r = _one_quad(formulation, restrained=True).calculate()
        s = r['tri_stress']['T']['Q1']
        expected = -E * ALPHA * DT / (1 - NU)
        self.assertAlmostEqual(s['sx'], expected, places=1)
        self.assertAlmostEqual(s['sy'], expected, places=1)
        self.assertAlmostEqual(s['txy'], 0.0, places=5)

    def test_q4(self):
        self._check('Q4')

    def test_qm6(self):
        self._check('QM6')


class TestQuadThermalBookkeeping(unittest.TestCase):

    def test_re_applying_accumulates_rather_than_replacing(self):
        """Two calls for the same quad/case must stack, not overwrite -- the
        solver sums unconditionally over every stored entry, so the
        restrained stress must reflect both temperature loads combined."""
        s = _one_quad(restrained=True)
        s.add_quad_temperature_load('Q1', 'T', 99.0, 99.0, 99.0, 99.0)
        self.assertEqual(len(s.quad_temperature_loads), 2)
        self.assertEqual(s.quad_temperature_loads[0].dt_i, DT)
        self.assertEqual(s.quad_temperature_loads[1].dt_i, 99.0)

        r = s.calculate()
        sx = r['tri_stress']['T']['Q1']['sx']
        self.assertAlmostEqual(sx, -E * ALPHA * (DT + 99.0) / (1 - NU),
                               places=1)

    def test_dt_mean_is_the_mean_of_four_nodes(self):
        from xdfem2d.models import QuadTemperatureLoad
        tl = QuadTemperatureLoad('Q1', 'T', 10.0, 20.0, 30.0, 40.0)
        self.assertAlmostEqual(tl.dt_mean, 25.0)

    def test_round_trip(self):
        from xdfem2d.structure_io import _from_dict, _to_dict
        s = _one_quad(restrained=True, dt=(10.0, 20.0, 30.0, 40.0))
        back = _from_dict(_to_dict(s))
        tl = back.quad_temperature_loads[0]
        self.assertEqual((tl.quad_id, tl.dt_i, tl.dt_j, tl.dt_k, tl.dt_l),
                         ('Q1', 10.0, 20.0, 30.0, 40.0))

    def test_create_temperature_dispatches_to_quad(self):
        s = _one_quad(restrained=True)
        s.quad_temperature_loads.clear()
        tl = s.create_temperature('Q1', DT, 0.0, 'T')
        self.assertEqual(len(s.quad_temperature_loads), 1)
        self.assertEqual(tl.quad_id, 'Q1')
        self.assertEqual(tl.dt_i, DT)


# ---------------------------------------------------------------------------
# QuadTemperatureLoad — DKT4/MITC4 (plate, through-thickness gradient)
# ---------------------------------------------------------------------------

def _base_thermal_quad(formulation='DKT4'):
    s = Structure2D(domain='plate')
    s.add_material('C', E, 25.0, poisson=NU, alpha=ALPHA)
    s.add_quad_section('P', 'C', thickness=T, formulation=formulation)
    s.add_load_case('LC')
    return s


class TestQuadThermalGradientFreePlate(unittest.TestCase):
    """A DKT4/MITC4 plate free to curve under a uniform gradient reports
    zero moment everywhere — the quad analogue of
    test_dkt.test_thermal_gradient_free_plate_reports_zero_moment, and the
    check that proves dkt4_thermal_load's fan-and-condense construction is
    consistent with dkt4_stiffness's (mitc4_thermal_load needs no
    condensation, so this also exercises the simpler path)."""

    def _check(self, formulation):
        s = _base_thermal_quad(formulation)
        L, n = 2.0, 4
        ids = _quad_grid(s, L, n, 'P')
        s.add_support('W', w=True)
        for corner in ((0, 0), (n, 0), (0, n)):
            s.assign_support(ids[corner], 'W')
        for qid in list(s.quad_elements_by_id):
            s.add_quad_temperature_load(qid, 'LC', dt_gradient=20.0)
        r = s.calculate()
        for v in r['tri_stress']['LC'].values():
            self.assertAlmostEqual(v['mx'], 0.0, places=5)
            self.assertAlmostEqual(v['my'], 0.0, places=5)
            self.assertAlmostEqual(v['mxy'], 0.0, places=5)

    def test_dkt4(self):
        self._check('DKT4')

    def test_mitc4(self):
        self._check('MITC4')


class TestQuadThermalGradientClampedPlate(unittest.TestCase):
    """A fully clamped DKT4 plate under a uniform gradient stays flat and
    carries the Timoshenko thermal moment m = -(alpha*dT/t)*D*(1+nu) — same
    closed form the triangle DKT case checks."""

    def test_dkt4(self):
        E_, NU_, T_ = 33e6, 0.2, 0.2
        D = E_ * T_ ** 3 / (12.0 * (1.0 - NU_ * NU_))
        s = Structure2D(domain='plate')
        s.add_material('C', E_, 25.0, poisson=NU_, alpha=ALPHA)
        s.add_quad_section('P', 'C', thickness=T_, formulation='DKT4')
        s.add_load_case('LC')
        L, n, dt = 2.0, 8, 20.0
        ids = _quad_grid(s, L, n, 'P')
        s.add_support('ENC', w=True, tx=True, ty=True)
        for (i, j), nid in ids.items():
            if i in (0, n) or j in (0, n):
                s.assign_support(nid, 'ENC')
        for qid in list(s.quad_elements_by_id):
            s.add_quad_temperature_load(qid, 'LC', dt_gradient=dt)
        r = s.calculate()
        m_exact = -(ALPHA * dt / T_) * D * (1 + NU_)
        for v in r['tri_stress']['LC'].values():
            self.assertAlmostEqual(v['mx'], m_exact, delta=abs(m_exact) * 0.05)
            self.assertAlmostEqual(v['my'], m_exact, delta=abs(m_exact) * 0.05)
            self.assertAlmostEqual(v['mxy'], 0.0, places=4)
        wmax = max(abs(d[0]) for d in r['displacements']['LC'].values())
        self.assertAlmostEqual(wmax, 0.0, places=8)


class TestQuadThermalGradientBookkeeping(unittest.TestCase):

    def test_round_trip_and_script(self):
        from xdfem2d.structure_io import _to_dict, _from_dict
        from xdfem2d.script_export import to_python
        s = _base_thermal_quad()
        _quad_grid(s, 2.0, 1, 'P')
        s.add_quad_temperature_load('Q0', 'LC', dt_gradient=15.0)
        q = _from_dict(_to_dict(s))
        self.assertEqual(q.quad_temperature_loads[0].dt_gradient, 15.0)
        self.assertIn('dt_gradient=15.0', to_python(s).replace(' ', ''))


if __name__ == '__main__':
    unittest.main()
