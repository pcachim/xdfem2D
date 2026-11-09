"""Geometry objects were renamed: line→segment, polyline→multisegment,
surface→polygon. The new names are the API; the old method names stay as
aliases, and files saved with the old serialized `kind` strings still load.
"""
import context  # noqa: F401
from xdfem2d import Structure2D
from xdfem2d.models import GeoSegment, GeoMultisegment, GeoPolygon
from xdfem2d.structure_io import _to_dict, _from_dict


def _model():
    s = Structure2D()
    s.add_material('C', 30e6, 25.0)
    s.add_section('B', 'C', b=0.3, h=0.5)
    s.add_tri_section('M', 'C', thickness=0.2)
    return s


def test_new_methods_build_the_renamed_classes():
    s = _model()
    s.add_geo_segment('L1', 0, 0, 4, 0, section_name='B')
    s.add_geo_multisegment('P1', [(0, 1), (2, 1), (4, 2)], section_name='B')
    s.add_geo_polygon('S1', [(0, 3), (4, 3), (4, 6), (0, 6)],
                      section_name='M')
    assert isinstance(s.geometry_objects['L1'], GeoSegment)
    assert isinstance(s.geometry_objects['P1'], GeoMultisegment)
    assert isinstance(s.geometry_objects['S1'], GeoPolygon)


def test_old_method_names_still_work_as_aliases():
    s = _model()
    assert s.add_geo_line.__func__ is s.add_geo_segment.__func__
    assert s.add_geo_polyline.__func__ is s.add_geo_multisegment.__func__
    assert s.add_geo_surface.__func__ is s.add_geo_polygon.__func__
    s.add_geo_line('L', 0, 0, 4, 0, section_name='B')
    s.add_geo_surface('S', [(0, 3), (4, 3), (4, 6), (0, 6)],
                      section_name='M')
    assert isinstance(s.geometry_objects['L'], GeoSegment)
    assert isinstance(s.geometry_objects['S'], GeoPolygon)


def test_serialisation_writes_the_new_kinds():
    s = _model()
    s.add_geo_segment('L1', 0, 0, 4, 0, section_name='B')
    s.add_geo_multisegment('P1', [(0, 1), (2, 1)], section_name='B')
    s.add_geo_polygon('S1', [(0, 3), (4, 3), (4, 6), (0, 6)],
                      section_name='M')
    kinds = {o['id']: o['kind'] for o in _to_dict(s)['geometry_objects']}
    assert kinds == {'L1': 'segment', 'P1': 'multisegment', 'S1': 'polygon'}


def test_old_serialised_kinds_still_load():
    """A file written before the rename (kind = line / polyline / surface) opens
    into the renamed classes."""
    base = _to_dict(_model())
    base['geometry_objects'] = [
        {'kind': 'line', 'id': 'X', 'section_name': 'B', 'divisions': 1,
         'max_chord': 0.0, 'node_ids': []},
        {'kind': 'polyline', 'id': 'Y', 'section_name': 'B', 'divisions': 1,
         'max_chord': 0.0, 'closed': False, 'node_ids': []},
        {'kind': 'surface', 'id': 'Z', 'tri_section_name': 'M',
         'node_ids': []},
    ]
    q = _from_dict(base)
    assert isinstance(q.geometry_objects['X'], GeoSegment)
    assert isinstance(q.geometry_objects['Y'], GeoMultisegment)
    assert isinstance(q.geometry_objects['Z'], GeoPolygon)


def test_model_check_accepts_both_old_and_new_kinds():
    from xdfem2d.model_json import missing_pieces
    for kind in ('segment', 'line', 'multisegment', 'polyline'):
        data = {'geometry_objects': [
            {'kind': kind, 'id': 'o', 'section_name': 'B',
             'node_ids': ['a', 'b']}]}
        assert not any('not one of' in p for p in missing_pieces(data)), kind
    for kind in ('polygon', 'surface', 'rectangle'):
        data = {'geometry_objects': [
            {'kind': kind, 'id': 'o', 'tri_section_name': 'M',
             'node_ids': ['a', 'b', 'c', 'd']}]}
        assert not any('not one of' in p for p in missing_pieces(data)), kind


def test_surface_objects_take_section_name_with_tri_section_name_alias():
    """add_geo_rectangle / add_geo_polygon take ``section_name`` (consistent
    with the curve objects); the old ``tri_section_name`` keyword is still
    accepted and equivalent. The stored attribute and the .x2d field stay
    ``tri_section_name`` — the persisted field is deliberately not renamed
    (see models.GeoRectangle), so old scripts and exported files keep working.
    """
    s = _model()
    r_new = s.add_geo_rectangle('RN', (0, 0), (4, 3), section_name='M')
    r_old = s.add_geo_rectangle('RO', (5, 0), (9, 3), tri_section_name='M')
    p_new = s.add_geo_polygon('PN', [(0, 4), (4, 4), (4, 7)], section_name='M')
    p_old = s.add_geo_polygon('PO', [(5, 4), (9, 4), (9, 7)],
                              tri_section_name='M')
    # Both keywords land on the same stored attribute.
    for o in (r_new, r_old, p_new, p_old):
        assert o.tri_section_name == 'M'
    # And it survives a round trip under the unchanged .x2d field name.
    reloaded = _from_dict(_to_dict(s))
    for oid in ('RN', 'RO', 'PN', 'PO'):
        assert reloaded.geometry_objects[oid].tri_section_name == 'M'
