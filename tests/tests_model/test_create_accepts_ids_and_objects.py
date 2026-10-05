"""Every create_* call takes an id as text or the object another create_* call
returned, for every argument that names something that already exists: a node, a
bar, a section, a load case, a region, a material.

The objects are what a script has in hand right after making them; the ids are
all it has in a later reply (the variables do not outlive the reply). The
assistant writes both, so both have to work, in every position.
"""
import context  # noqa: F401  (adds src/ to sys.path)
import pytest

from xdfem2d import Structure2D


def _fresh():
    m = Structure2D()
    sec = m.create_rc_section('S1', b=0.3, h=0.5)
    n1, n2 = m.create_node(0, 0), m.create_node(5, 0)
    n3 = m.create_node(5, 3)
    bar = m.create_bar_element(sec, n1, n2)
    lc = m.create_load_case('G')
    m.create_load_case('Q')
    m.create_area_section('A1', 'C30/37', thickness=0.2)
    return m, dict(sec=sec, n1=n1, n2=n2, n3=n3, bar=bar, lc=lc)


def _plate():
    m = Structure2D(domain='plate')
    reg = m.create_polygon([0, 0, 4, 3], thickness=0.2, id='L1')
    m.create_load_case('G')
    return m, dict(reg=reg, lc=m.load_cases[0])


IDS = dict(sec='S1', n1='N1', n2='N2', n3='N3', bar='B1', lc='G', reg='L1')

CALLS = {
    'bar element: section and both nodes':
        lambda m, o: m.create_bar_element(o['sec'], o['n2'], o['n3']),
    'area element: section and nodes':
        lambda m, o: m.create_area_element('A1', o['n1'], o['n2'], o['n3']),
    'uniform load: bar and case':
        lambda m, o: m.create_uniform_load(o['bar'], o['lc'], 10),
    'bar distributed load: bar and case':
        lambda m, o: m.create_bar_distributed_load(o['bar'], o['lc'], 1, 2),
    'bar point load: bar and case':
        lambda m, o: m.create_bar_point_load(o['bar'], o['lc'], 5),
    'node load: node and case':
        lambda m, o: m.create_node_load(o['n2'], o['lc'], 5),
    'self weight: case':
        lambda m, o: m.create_self_weight(1.0, o['lc']),
    'support: node':
        lambda m, o: m.create_support(o['n1'], ux=True, uy=True),
    'support settlement: node and case':
        lambda m, o: m.create_support_settlement(o['n1'], o['lc'], uy=-0.01),
    'temperature: bar and case':
        lambda m, o: m.create_temperature(o['bar'], o['lc'], 10),
    'pin, fix, roller: a node':
        lambda m, o: (m.pin(o['n1']), m.fix(o['n2']), m.roller(o['n3'])),
    'pin: a list of nodes':
        lambda m, o: m.pin([o['n1'], o['n2']]),
    'analysis case: case as a coefficient key':
        lambda m, o: m.create_analysis_case('A', 'Linear', {o['lc']: 1.35}),
    'load combination: case as a coefficient key':
        lambda m, o: m.create_load_combination('C', {o['lc']: 1.35}),
}


@pytest.mark.parametrize("name", list(CALLS))
@pytest.mark.parametrize("form", ["object", "id"])
def test_the_call_takes_both(name, form):
    m, objs = _fresh()
    args = objs if form == 'object' else {k: IDS[k] for k in objs}
    CALLS[name](m, args)


@pytest.mark.parametrize("form", ["object", "id"])
def test_the_region_and_its_edges_take_both(form):
    m, objs = _plate()
    reg = objs['reg'] if form == 'object' else 'L1'
    lc = objs['lc'] if form == 'object' else 'G'
    m.create_uniform_load(reg, lc, 5)
    m.create_edge_load(reg, 'top', lc, 5)
    m.support_edge(reg, 'bottom', 'pin')


@pytest.mark.parametrize("form", ["object", "id"])
def test_a_material_is_taken_either_way(form):
    m = Structure2D()
    mat = m.create_rc_material('C30/37')
    ref = mat if form == 'object' else mat.name
    m.create_bar_section('S9', ref, b=0.3, h=0.5)
    m.create_area_section('A9', ref, thickness=0.2)
    m.create_polygon([0, 0, 4, 3], thickness=0.2, material=ref)


def test_the_object_and_the_id_give_the_same_coefficients():
    m, o = _fresh()
    a = m.create_analysis_case('A', 'Linear', {o['lc']: 1.35, 'Q': 1.5})
    assert a.coefficients == {'G': 1.35, 'Q': 1.5}
    c = m.create_load_combination('C', {o['lc']: 1.35})
    assert dict(c.coefficients or c.analysis_coefficients) == {'G': 1.35}


def test_case_objects_are_hashable_by_id():
    m, o = _fresh()
    ac = m.analysis_cases[0]
    assert hash(o['lc']) == hash('G')
    assert {o['lc']: 1}[o['lc']] == 1
    assert hash(ac) == hash(ac.id)
