"""Phase 0 of the plate/grillage extension: the ``domain`` field.

Defaults, validation, (de)serialisation compatibility — a file without the
flag loads as 'plane' and behaves exactly as before — and the domain gates
on unsupported features.
"""
import pytest

import context  # noqa: F401
from xdfem2d import Structure2D
from xdfem2d.structure_io import _to_dict, _from_dict


def _plate_grid_model():
    s = Structure2D(domain='plate')
    s.add_material('C', 33e6, 25.0, poisson=0.2)
    s.add_section('S', 'C', 0.3, 0.5, shape='Rectangular')
    s.add_node('N1', 0.0, 0.0)
    s.add_node('N2', 4.0, 0.0)
    s.add_bar_element('E1', 'N1', 'N2', 'S')
    s.add_support('ENC', w=True, tx=True, ty=True)
    s.assign_support('N1', 'ENC')
    s.add_load_case('LC')
    s.add_point_load('N2', 'LC', fz=-10.0)
    return s


def test_default_domain_is_plane():
    s = Structure2D()
    assert s.domain == 'plane'
    assert s.dof_labels == ('ux', 'uy', 'tz')


def test_plate_domain_labels():
    s = Structure2D(domain='plate')
    assert s.dof_labels == ('w', 'tx', 'ty')


def test_unknown_domain_rejected():
    with pytest.raises(ValueError):
        Structure2D(domain='volume')


def test_round_trip_keeps_domain():
    s = _plate_grid_model()
    d = _to_dict(s)
    assert d['domain'] == 'plate'
    q = _from_dict(d)
    assert q.domain == 'plate'
    assert q.point_loads[0].fx == -10.0        # fz alias fills slot 0
    assert q.supports['ENC'].ux and q.supports['ENC'].uy and q.supports['ENC'].tz


def test_file_without_domain_loads_as_plane():
    s = _plate_grid_model()
    d = _to_dict(s)
    del d['domain']
    q = _from_dict(d)
    assert q.domain == 'plane'


def test_reader_accepts_plate_aliases():
    d = _to_dict(_plate_grid_model())
    d['supports'] = [{'name': 'ENC', 'w': True, 'tx': True, 'ty': True}]
    d['point_loads'] = [{'node_id': 'N2', 'load_case_id': 'LC', 'fz': -7.5}]
    q = _from_dict(d)
    assert q.supports['ENC'].ux and q.supports['ENC'].uy and q.supports['ENC'].tz
    assert q.point_loads[0].fx == -7.5


def test_torsion_override_round_trip():
    s = _plate_grid_model()
    s.sections['S'].torsion_override = 1.23e-3
    q = _from_dict(_to_dict(s))
    assert q.sections['S'].torsion == pytest.approx(1.23e-3)


def test_plate_domain_refuses_triangles():
    s = _plate_grid_model()
    s.add_tri_section('T', 'C', thickness=0.2)
    s.add_node('N3', 2.0, 2.0)
    s.add_tri_element('TR1', 'N1', 'N2', 'N3', 'T')
    with pytest.raises(ValueError, match='plate'):
        s.calculate()


def test_plate_domain_runs_modal():
    """Phase 3: dynamics is available in the plate domain — a Mass + Modal pair
    solves and returns a positive out-of-plane frequency (the "x" slot holds the
    vertical direction)."""
    s = _plate_grid_model()
    s.add_analysis_case('MASS', 'Mass', {'LC': 1.0})
    s.add_analysis_case('MOD', 'Modal', modal_case_id='MASS', num_modes=1)
    mod = s.calculate()['analysis_cases']['MOD']
    assert 'error' not in mod
    assert mod['modal_info'][0]['frequency'] > 0.0
    assert mod.get('plate') is True


def test_plate_domain_refuses_pdelta():
    """P-Delta stays blocked in the plate domain (its geometric stiffness needs
    in-plane axial force)."""
    s = _plate_grid_model()
    s.add_analysis_case('PD', 'GeometricNonlinear', {'LC': 1.0})
    assert any('P-Delta' in p for p in s.domain_problems())
    with pytest.raises(ValueError, match='plate'):
        s.calculate()


def test_plane_domain_has_no_domain_problems():
    s = Structure2D()
    assert s.domain_problems() == []


def test_script_export_includes_domain():
    from xdfem2d.script_export import to_python
    text = to_python(_plate_grid_model())
    assert "Structure2D(domain='plate')" in text


def test_checked_loader_reads_domain():
    import json, tempfile, os
    from xdfem2d.structure_io_checked import load_structure_json_checked
    d = _to_dict(_plate_grid_model())
    fd, path = tempfile.mkstemp(suffix='.json')
    try:
        with os.fdopen(fd, 'w') as f:
            json.dump(d, f)
        s, report = load_structure_json_checked(path)
        assert s.domain == 'plate'
        assert not [i for i in report.issues if i.severity == 'error'] \
            if hasattr(report, 'issues') else True
    finally:
        os.unlink(path)
