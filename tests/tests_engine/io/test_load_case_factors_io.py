"""Category and psi of a load case survive save, load and script export."""
from __future__ import annotations

from xdfem2d.structure import Structure2D as Structure
from xdfem2d.structure_io import _to_dict as structure_to_dict, _from_dict as structure_from_dict


def _model():
    s = Structure()
    s.add_load_case('Q_C', 0.0, action_type='Q', category='C', psi2=0.4)
    s.add_load_case('G1', 1.0, action_type='G')
    return s


def test_roundtrip_keeps_category_and_overrides():
    d = structure_to_dict(_model())
    q = next(lc for lc in d['load_cases'] if lc['id'] == 'Q_C')
    assert q['category'] == 'C' and q['psi2'] == 0.4 and 'psi0' not in q
    g = next(lc for lc in d['load_cases'] if lc['id'] == 'G1')
    assert 'category' not in g and not any(k.startswith('psi') for k in g)
    r = structure_from_dict(d).load_cases_by_id['Q_C']
    assert (r.category, r.psi0, r.psi1, r.psi2) == ('C', None, None, 0.4)


def test_a_file_without_the_fields_still_loads():
    d = structure_to_dict(_model())
    for lc in d['load_cases']:
        for k in ('category', 'psi0', 'psi1', 'psi2'):
            lc.pop(k, None)
    lc = structure_from_dict(d).load_cases_by_id['Q_C']
    assert lc.category == '' and lc.psi2 is None


def test_script_export_writes_them_only_when_set():
    from xdfem2d.script_export import to_python as export_script
    text = export_script(_model())
    line_q = next(l for l in text.splitlines() if "add_load_case('Q_C'" in l)
    line_g = next(l for l in text.splitlines() if "add_load_case('G1'" in l)
    assert "category='C'" in line_q and 'psi2=0.4' in line_q
    assert 'category' not in line_g and 'psi' not in line_g
