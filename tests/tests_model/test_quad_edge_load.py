"""Edge loads on quad elements — the 4-node analogue of triangle edge loads.

A uniform edge load on one edge of a quad is distributed to the two edge nodes
as consistent nodal loads (½·L each), assembled, persisted and purged with the
element, exactly like TriEdgeLoad.
"""
import pytest

import context  # noqa: F401
from xdfem2d import Structure2D
from xdfem2d.structure_io import _to_dict, _from_dict


def _wall():
    s = Structure2D(domain='plane')
    s.add_material('m', elastic_modulus=30e6, unit_weight=0.0, poisson=0.2)
    s.add_quad_section('Q', 'm', thickness=0.5, formulation='QM6')
    for nid, (x, y) in {'n1': (0, 0), 'n2': (2, 0), 'n3': (2, 1),
                        'n4': (0, 1)}.items():
        s.add_node(nid, x, y)
    s.add_quad_element('Q1', 'n1', 'n2', 'n3', 'n4', 'Q')
    s.add_support('FIX', ux=True, uy=True)
    s.assign_support('n1', 'FIX'); s.assign_support('n4', 'FIX')
    s.add_load_case('Q')
    return s


def test_edge_load_resultant_balances_reactions():
    s = _wall()
    # Edge n2-n3 has length 1; fy = -10 kN/m -> total -10 kN.
    s.add_quad_edge_load('EL1', 'Q1', 'n2', 'n3', 'Q', fy=-10.0)
    r = s.calculate()
    reac = r['reactions']['Q']
    assert sum(v[1] for v in reac.values()) == pytest.approx(10.0, rel=1e-9)


def test_it_lumps_half_to_each_edge_node():
    s = _wall()
    s.add_quad_edge_load('EL1', 'Q1', 'n2', 'n3', 'Q', fy=-10.0)
    from xdfem2d import loads as L
    import numpy as np
    F = np.zeros((s.num_dofs, 1))
    L._apply_quad_edge_loads(s, F, {'Q': 0})
    b2 = s.node_dof_index['n2']; b3 = s.node_dof_index['n3']
    assert F[b2 + 1, 0] == pytest.approx(-5.0, rel=1e-9)
    assert F[b3 + 1, 0] == pytest.approx(-5.0, rel=1e-9)


def test_round_trip_and_purge():
    s = _wall()
    s.add_quad_edge_load('EL1', 'Q1', 'n2', 'n3', 'Q', fy=-10.0)
    back = _from_dict(_to_dict(s))
    assert len(back.quad_edge_loads) == 1
    e = back.quad_edge_loads[0]
    assert e.quad_id == 'Q1' and e.fy == -10.0
    # Removing the quad drops its edge loads.
    s.remove_quad_element('Q1')
    assert s.quad_edge_loads == []


if __name__ == '__main__':
    import sys
    sys.exit(pytest.main([__file__, '-q']))
