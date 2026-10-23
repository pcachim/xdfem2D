"""create_edge_load takes p in kN/m: the edge carries p * L whatever the thickness
(a stored surface edge load is a pressure that the solver multiplies by it)."""
import context  # noqa: F401  (adds src/ to sys.path)
import pytest

from xdfem2d.geo_expand import expand_geometry
from xdfem2d.templates import default_structure


def _reaction(model, case='Q', axis=1):
    flat = expand_geometry(model)[0]
    res = flat.calculate()
    return sum(v[axis] for v in res['reactions'][case].values())


@pytest.mark.parametrize("thickness", [0.2, 0.35, 1.0])
def test_perp_edge_load_resultant_is_p_times_length(thickness):
    m = default_structure(domain='plane')
    m.create_polygon(outline=[0, 0, 6, 0, 6, 3, 0, 3], thickness=thickness,
                     material='C30/37', id='Wall')
    m.fix_edge('Wall', 0)
    m.create_load_case('Q', action_type='Q')
    m.create_edge_load('Wall', 2, 'Q', p=50)          # the top edge, 6 m
    assert abs(_reaction(m)) == pytest.approx(300.0, rel=1e-6)


def test_tangential_edge_load_resultant_is_p_times_length():
    m = default_structure(domain='plane')
    m.create_polygon(outline=[0, 0, 6, 0, 6, 3, 0, 3], thickness=0.25,
                     material='C30/37', id='Wall')
    m.fix_edge('Wall', 0)
    m.create_load_case('Q', action_type='Q')
    m.create_edge_load('Wall', 2, 'Q', p=20, direction='tang')
    assert abs(_reaction(m, axis=0)) == pytest.approx(120.0, rel=1e-6)
