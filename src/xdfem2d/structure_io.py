"""
Save and load the full Structure2D definition.

Supported formats:
  - JSON  (.json)          — human-readable, lossless
"""
from __future__ import annotations
import json
from pathlib import Path

from .structure import Structure2D


# ============================================================
# JSON
# ============================================================

def _beams_from_dict(d) -> dict:
    """``{tag: {name, overrides}}`` from a saved file; tolerant of absent or
    malformed entries (an old file has no 'beams')."""
    out = {}
    if isinstance(d, dict):
        for tag, b in d.items():
            b = b if isinstance(b, dict) else {}
            ov = b.get('overrides')
            out[str(tag)] = {'name': str(b.get('name') or ''),
                             'overrides': dict(ov) if isinstance(ov, dict)
                             else {}}
    return out


def _beam_detail_from_dict(s, d) -> None:
    """Attach the stored bar decisions to *s*. A document that cannot be read
    (corrupt, or written by a newer program) never blocks opening the model:
    the detail stays empty and the reason is kept in
    ``s.beam_detail_load_error`` for the application to report."""
    from .detailing import DetailDocument, DetailError
    try:
        s.beam_detail = DetailDocument.from_dict(d)
        s.beam_detail_load_error = ""
    except DetailError as e:
        s.beam_detail = DetailDocument()
        s.beam_detail_load_error = str(e)


def save_structure_json(struc: Structure2D, path: str | Path):
    """Serialise the full structure to a JSON file."""
    data = _to_dict(struc)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2)


def load_structure_json(path: str | Path) -> Structure2D:
    """Deserialise a structure from a JSON file."""
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return _from_dict(data)


# ============================================================
# Internal helpers — dict serialisation
# ============================================================

def _geo_objects_to_list(s: Structure2D) -> list:
    """Serialise objects to dicts. Node-driven: only node_ids + parameters (the
    geometry lives in the referenced nodes, saved as normal nodes)."""
    from .models import (GeoArc, GeoMultisegment, GeoSegment,
                         GeoRectangle, GeoPolygon)
    out = []
    for m in getattr(s, "geometry_objects", {}).values():
        # Surface objects (mesh into triangles) carry a TriSection + target size.
        if isinstance(m, (GeoRectangle, GeoPolygon)):
            surf = {'id': m.id,
                    'tri_section_name': getattr(m, 'tri_section_name', ''),
                    'target_size': getattr(m, 'target_size', 0.5),
                    'prefer_quad': bool(getattr(m, 'prefer_quad', False)),
                    'edge_support_mode': getattr(m, 'edge_support_mode', 'common'),
                    'edge_spring_mode': getattr(m, 'edge_spring_mode', 'linear'),
                    'node_ids': list(getattr(m, 'node_ids', []))}
            edge_supports = getattr(m, 'edge_supports', None)
            if edge_supports:
                surf['edge_supports'] = list(edge_supports)
            kind = 'rectangle' if isinstance(m, GeoRectangle) else 'polygon'
            out.append({'kind': kind, **surf})
            continue
        common = {'id': m.id, 'section_name': m.section_name,
                  'divisions': m.divisions, 'max_chord': m.max_chord,
                  'loads': [dict(l) for l in getattr(m, 'loads', []) or []],
                  'intersect_crossings': bool(getattr(m, 'intersect_crossings', False)),
                  'is_column': getattr(m, 'is_column', None),
                  'sd_ky': getattr(m, 'sd_ky', None),
                  'sd_kz': getattr(m, 'sd_kz', None),
                  'sd_klt': getattr(m, 'sd_klt', None),
                  'sd_ltb': bool(getattr(m, 'sd_ltb', True)),
                  'beam': getattr(m, 'beam', None),
                  'node_ids': list(getattr(m, 'node_ids', []))}
        if isinstance(m, GeoArc):
            out.append({'kind': 'arc', **common})
        elif isinstance(m, GeoMultisegment):
            ms = {'kind': 'multisegment', 'closed': m.closed, **common}
            # Only written when actually set — an absent key means "use
            # divisions uniformly", same as every file saved before this
            # field existed, so an unmodified multisegment's .x2d is unchanged.
            span_div = getattr(m, 'span_divisions', None)
            if span_div:
                ms['span_divisions'] = list(span_div)
            out.append(ms)
        elif isinstance(m, GeoSegment):
            out.append({'kind': 'segment', **common})
    return out


def _geo_objects_from_list(s: Structure2D, data: list):
    """Rebuild objects from serialised dicts (absent/empty = no objects). The
    defining nodes were saved as normal nodes and are referenced via node_ids."""
    from .models import (GeoArc, GeoMultisegment, GeoSegment,
                         GeoRectangle, GeoPolygon)
    for m in data or []:
        kind = m.get('kind')
        try:
            # Object kinds were renamed (line→segment, polyline→multisegment,
            # surface→polygon); the old spellings are still read so files saved
            # before the rename keep opening.
            if kind in ('rectangle', 'surface', 'polygon'):
                cls = GeoRectangle if kind == 'rectangle' else GeoPolygon
                obj = cls(id=str(m['id']),
                          tri_section_name=m.get('tri_section_name', ''),
                          target_size=float(m.get('target_size', 0.5)),
                          prefer_quad=bool(m.get('prefer_quad', False)),
                          edge_support_mode=m.get('edge_support_mode', 'common'),
                          edge_spring_mode=m.get('edge_spring_mode', 'linear'))
                if m.get('edge_supports'):
                    obj.edge_supports = list(m['edge_supports'])
                obj.node_ids = list(m.get('node_ids', []))
                s.geometry_objects[obj.id] = obj
                continue
            # str(): an object's id is meant to be a string everywhere else in
            # the model, but nothing ever enforced that at creation time, so a
            # script or an old file that gave one a bare number (e.g.
            # add_geo_rectangle(5, ...)) stored an int — which then crashed
            # expand_geometry's sorted(geometry_objects) outright the moment
            # it sat alongside the usual string ids ("'<' not supported
            # between instances of 'int' and 'str'"). Coercing on load is the
            # one place that fixes it for every object already saved with a
            # numeric id, not just future ones.
            common = dict(id=str(m['id']), section_name=m.get('section_name', ''),
                          divisions=int(m.get('divisions', 1)),
                          max_chord=float(m.get('max_chord', 0.0)))
            if kind == 'arc':
                obj = GeoArc(**common)
            elif kind in ('multisegment', 'polyline'):
                obj = GeoMultisegment(closed=bool(m.get('closed', False)), **common)
                span_div = m.get('span_divisions')
                if span_div:
                    obj.span_divisions = [int(d) for d in span_div]
            elif kind in ('segment', 'line'):
                obj = GeoSegment(**common)
            else:
                continue
            obj.loads = [dict(l) for l in m.get('loads', []) or []]
            obj.intersect_crossings = bool(m.get('intersect_crossings', False))
            obj.is_column = m.get('is_column')
            obj.sd_ky = m.get('sd_ky')
            obj.sd_kz = m.get('sd_kz')
            obj.sd_klt = m.get('sd_klt')
            obj.sd_ltb = bool(m.get('sd_ltb', True))
            obj.beam = m.get('beam')
            obj.node_ids = list(m.get('node_ids', []))
            s.geometry_objects[obj.id] = obj
        except (KeyError, ValueError):
            continue


def _to_dict(s: Structure2D) -> dict:
    return {
        # Analysis domain. Written always; absent on read means 'plane', so
        # every file saved before the field existed keeps its meaning. The
        # per-entry keys below stay the canonical (plane) spellings in BOTH
        # domains — they are positional slots — so an older reader given a
        # plate file still sees every value; readers accept the plate aliases
        # (w/tx/ty, fz/mx/my) as well.
        'domain': getattr(s, 'domain', 'plane'),
        'project_info': dict(getattr(s, 'project_info', {}) or {}),
        'propagate_edge_supports': bool(
            getattr(s, 'propagate_edge_supports', True)),
        'propagate_edge_springs': bool(
            getattr(s, 'propagate_edge_springs', True)),
        'nodes': [
            {'id': n.id, 'x': n.x, 'y': n.y}
            for n in s.nodes.values()
        ],
        'materials': [
            {'name': m.name, 'elastic_modulus': m.elastic_modulus,
             'unit_weight': m.unit_weight, 'alpha': m.alpha,
             'unit_mass': m.unit_mass,   # None → derive from unit_weight/g at load time
             'material_type': m.material_type.value,
             'poisson': getattr(m, 'poisson', 0.2),
             'design': getattr(m, 'design', {}) or {}}
            for m in s.materials.values()
        ],
        'sections': [
            {'name': sc.name, 'material_name': sc.material_name, 'b': sc.b, 'h': sc.h,
             'area_override': sc.area_override, 'inertia_override': sc.inertia_override,
             'inertia_minor_override': getattr(sc, 'inertia_minor_override', None),
             'torsion_override': getattr(sc, 'torsion_override', None),
             'wel_y_override': getattr(sc, 'wel_y_override', None),
             'wpl_y_override': getattr(sc, 'wpl_y_override', None),
             'wel_z_override': getattr(sc, 'wel_z_override', None),
             'wpl_z_override': getattr(sc, 'wpl_z_override', None),
             'av_y_override': getattr(sc, 'av_y_override', None),
             'av_z_override': getattr(sc, 'av_z_override', None),
             'warping_override': getattr(sc, 'warping_override', None),
             'wt_override': getattr(sc, 'wt_override', None),
             'curve_y_override': getattr(sc, 'curve_y_override', None),
             'curve_z_override': getattr(sc, 'curve_z_override', None),
             'profile_name': sc.profile_name, 'angle': getattr(sc, 'angle', 0.0),
             'shape': sc.shape.value, 'tw': sc.tw, 'tf': sc.tf,
             'rc_cover': sc.rc_cover, 'rc_alpha_s': sc.rc_alpha_s,
             'rc_bar_phi': sc.rc_bar_phi, 'rc_shear_min': sc.rc_shear_min,
             'rc_torsion_distribution': getattr(
                 sc, 'rc_torsion_distribution', 'top_bottom'),
             'is_column': getattr(sc, 'is_column', False),
             'rc_phi_ef': getattr(sc, 'rc_phi_ef', 2.0),
             'rc_n_bars': getattr(sc, 'rc_n_bars', 4),
             'rc_n_bars_y': getattr(sc, 'rc_n_bars_y', 2),
             'rc_n_bars_z': getattr(sc, 'rc_n_bars_z', 2),
             'rc_second_order_method': getattr(
                 sc, 'rc_second_order_method', 'nominal_curvature'),
             'timber_service_class': getattr(sc, 'timber_service_class', 'SC1'),
             'beam_overrides': getattr(sc, 'beam_overrides', None)}
            for sc in s.sections.values()
        ],
        'bar_elements': [
            {'id': e.id, 'node_i': e.node_i, 'node_j': e.node_j,
             'section_name': e.section_name,
             'hinge_i': e.hinge_i,
             'hinge_j': e.hinge_j,
             'rc_design': e.rc_design,
             'rc_cover': e.rc_cover,
             'rc_cotg_theta': e.rc_cotg_theta,
             'rc_alpha_s': e.rc_alpha_s,
             'sd_ky': getattr(e, 'sd_ky', None),
             'sd_kz': getattr(e, 'sd_kz', None),
             'sd_klt': getattr(e, 'sd_klt', None),
             'sd_ltb': getattr(e, 'sd_ltb', True),
             'is_column': getattr(e, 'is_column', None),
             'beam': getattr(e, 'beam', None),
             'stage': getattr(e, 'stage', 1)}
            for e in s.bar_elements
        ],
        'tri_sections': [
            {'name': ts.name, 'material_name': ts.material_name,
             'thickness': ts.thickness, 'plane_strain': ts.plane_strain,
             'formulation': getattr(ts, 'formulation', 'CST'),
             'rc_cover': ts.rc_cover, 'rc_alpha_s': ts.rc_alpha_s,
             'rc_bar_phi': ts.rc_bar_phi, 'rc_shear_min': ts.rc_shear_min,
             # Per-face/direction covers (plate slabs). Written always; absent on
             # read means None → falls back to rc_cover, so old files are
             # unaffected.
             'rc_cover_top_x': getattr(ts, 'rc_cover_top_x', None),
             'rc_cover_top_y': getattr(ts, 'rc_cover_top_y', None),
             'rc_cover_bot_x': getattr(ts, 'rc_cover_bot_x', None),
             'rc_cover_bot_y': getattr(ts, 'rc_cover_bot_y', None)}
            for ts in getattr(s, 'tri_sections', {}).values()
        ],
        'tri_elements': [
            {'id': t.id, 'node_i': t.node_i, 'node_j': t.node_j,
             'node_k': t.node_k, 'section_name': t.section_name,
             'stage': getattr(t, 'stage', 1)}
            for t in getattr(s, 'tri_elements', [])
        ],
        # 4-node quadrilateral elements (dev/IMPLEMENT_QUAD.md Phase 3).
        # Absent-key compatible: an old file has no 'quad_sections'/
        # 'quad_elements' keys, .get(..., []) below reads that as "none",
        # and a tri-only model round-trips byte-identical in shape.
        'quad_sections': [
            {'name': qs.name, 'material_name': qs.material_name,
             'thickness': qs.thickness, 'plane_strain': qs.plane_strain,
             'formulation': getattr(qs, 'formulation', 'MITC4'),
             'rc_cover': qs.rc_cover, 'rc_alpha_s': qs.rc_alpha_s,
             'rc_bar_phi': qs.rc_bar_phi, 'rc_shear_min': qs.rc_shear_min,
             'rc_cover_top_x': getattr(qs, 'rc_cover_top_x', None),
             'rc_cover_top_y': getattr(qs, 'rc_cover_top_y', None),
             'rc_cover_bot_x': getattr(qs, 'rc_cover_bot_x', None),
             'rc_cover_bot_y': getattr(qs, 'rc_cover_bot_y', None)}
            for qs in getattr(s, 'quad_sections', {}).values()
        ],
        'quad_elements': [
            {'id': q.id, 'node_i': q.node_i, 'node_j': q.node_j,
             'node_k': q.node_k, 'node_l': q.node_l,
             'section_name': q.section_name,
             'stage': getattr(q, 'stage', 1)}
            for q in getattr(s, 'quad_elements', [])
        ],
        'tri_edge_loads': [
            {'id': e.id, 'tri_id': e.tri_id, 'node_a': e.node_a,
             'node_b': e.node_b, 'load_case_id': e.load_case_id,
             'fx': e.fx, 'fy': e.fy, 'coord_sys': e.coord_sys,
             'pn': e.pn, 'pt': e.pt}
            for e in getattr(s, 'tri_edge_loads', [])
        ],
        'quad_edge_loads': [
            {'id': e.id, 'quad_id': e.quad_id, 'node_a': e.node_a,
             'node_b': e.node_b, 'load_case_id': e.load_case_id,
             'fx': e.fx, 'fy': e.fy, 'coord_sys': e.coord_sys,
             'pn': e.pn, 'pt': e.pt}
            for e in getattr(s, 'quad_edge_loads', [])
        ],
        'surface_edge_loads': [
            {'id': e.id, 'object_id': e.object_id, 'node_a': e.node_a,
             'node_b': e.node_b, 'load_case_id': e.load_case_id,
             'fx': e.fx, 'fy': e.fy, 'coord_sys': e.coord_sys,
             'pn': e.pn, 'pt': e.pt}
            for e in getattr(s, 'surface_edge_loads', [])
        ],
        'tri_area_loads': [
            {'tri_id': a.tri_id, 'load_case_id': a.load_case_id, 'pz': a.pz}
            for a in getattr(s, 'tri_area_loads', [])
        ],
        'quad_area_loads': [
            {'quad_id': a.quad_id, 'load_case_id': a.load_case_id, 'pz': a.pz}
            for a in getattr(s, 'quad_area_loads', [])
        ],
        'surface_area_loads': [
            {'object_id': a.object_id, 'load_case_id': a.load_case_id,
             'pz': a.pz}
            for a in getattr(s, 'surface_area_loads', [])
        ],
        'tri_area_springs': [
            {'tri_id': a.tri_id, 'kz': a.kz}
            for a in getattr(s, 'tri_area_springs', [])
        ],
        'quad_area_springs': [
            {'quad_id': a.quad_id, 'kz': a.kz}
            for a in getattr(s, 'quad_area_springs', [])
        ],
        'surface_area_springs': [
            {'object_id': a.object_id, 'kz': a.kz}
            for a in getattr(s, 'surface_area_springs', [])
        ],
        'punch_columns': [
            {'id': c.id, 'node_id': c.node_id, 'shape': c.shape,
             'bx': c.bx, 'by': c.by, 'position': c.position,
             'force': c.force, 'dx': c.dx, 'dy': c.dy,
             'beta_min': getattr(c, 'beta_min', 1.0)}
            for c in getattr(s, 'punch_columns', [])
        ],
        'fields': [
            {'name': f.name, 'expression': f.expression}
            for f in getattr(s, 'fields', {}).values()
        ],
        'geometry_objects': _geo_objects_to_list(s),
        'beams': {tag: {'name': b.get('name', ''),
                        'overrides': dict(b.get('overrides') or {})}
                  for tag, b in getattr(s, 'beams', {}).items()},
        # only when there are manual decisions: a model without any is saved
        # exactly as before
        **({'beam_detail': s.beam_detail.to_dict()}
           if getattr(getattr(s, 'beam_detail', None), 'beams', None) else {}),
        'supports': [
            {'name': sp.name, 'ux': sp.ux, 'uy': sp.uy, 'tz': sp.tz}
            for sp in s.supports.values()
        ],
        'support_assignments': [
            {'node_id': a.node_id, 'support_name': a.support_name}
            for a in s.support_assignments
        ],
        'node_springs': [
            {'node_id': sp.node_id, 'kx': sp.kx, 'ky': sp.ky, 'kt': sp.kt,
             'mode_x': sp.mode_x, 'mode_y': sp.mode_y}
            for sp in s.node_springs.values()
        ],
        'element_springs': [
            {'element_id': es.element_id, 'kx': es.kx, 'ky': es.ky,
             'coord_sys': getattr(es, 'coord_sys', 'global'),
             'mode_x': es.mode_x, 'mode_y': es.mode_y}
            for es in s.element_springs.values()
        ],
        'constraints': [
            {'id': c.id, 'kind': c.kind,
             # canonical equation is stored as a list of [node_id, comp, coef];
             # JSON has no tuples, so terms round-trip as lists.
             'terms': [[t[0], t[1], t[2]] for t in c.terms],
             'value': c.value,
             'master': c.master, 'slaves': list(c.slaves),
             'nodes': list(c.nodes), 'components': list(c.components),
             'enabled': c.enabled}
            for c in getattr(s, 'constraints', {}).values()
        ],
        'load_cases': [
            {'id': lc.id, 'self_weight_factor': lc.self_weight_factor,
             'action_type': lc.action_type.value,
             **({'load_duration': lc.load_duration}
                if getattr(lc, 'load_duration', '') else {}),
             **({'category': lc.category}
                if getattr(lc, 'category', '') else {}),
             **{k: getattr(lc, k) for k in ('psi0', 'psi1', 'psi2')
                if getattr(lc, k, None) is not None}}
            for lc in s.load_cases
        ],
        'point_loads': [
            {'node_id': pl.node_id, 'load_case_id': pl.load_case_id,
             'fx': pl.fx, 'fy': pl.fy, 'mz': pl.mz}
            for pl in s.point_loads
        ],
        'distributed_loads': [
            {'element_id': dl.element_id, 'load_case_id': dl.load_case_id,
             'fxe': dl.fxe, 'fxd': dl.fxd, 'fye': dl.fye, 'fyd': dl.fyd,
             'coord_sys': dl.coord_sys}
            for dl in s.distributed_loads
        ],
        'element_point_loads': [
            {'element_id': epl.element_id, 'load_case_id': epl.load_case_id,
             'a': epl.a, 'fx': epl.fx, 'fy': epl.fy, 'mz': epl.mz,
             'coord_sys': epl.coord_sys}
            for epl in s.element_point_loads
        ],
        'load_combinations': [
            {'id': lc.id, 'combo_type': lc.combo_type,
             'coefficients': lc.coefficients,
             'analysis_coefficients': lc.analysis_coefficients}
            for lc in s.load_combinations
        ],
        'cuts': [
            {'id': c.id, 'name': c.name,
             'x1': c.x1, 'y1': c.y1, 'x2': c.x2, 'y2': c.y2}
            for c in s.cuts
        ],
        'support_settlements': [
            {'node_id': ss.node_id, 'load_case_id': ss.load_case_id,
             'ux': ss.ux, 'uy': ss.uy, 'tz': ss.tz}
            for ss in s.support_settlements
        ],
        'temperature_loads': [
            {'element_id': tl.element_id, 'load_case_id': tl.load_case_id,
             'delta_t_uniform': tl.delta_t_uniform,
             'delta_t_gradient': tl.delta_t_gradient}
            for tl in s.temperature_loads
        ],
        'tri_temperature_loads': [
            {'tri_id': tl.tri_id, 'load_case_id': tl.load_case_id,
             'dt_i': tl.dt_i, 'dt_j': tl.dt_j, 'dt_k': tl.dt_k,
             'dt_gradient': getattr(tl, 'dt_gradient', 0.0)}
            for tl in getattr(s, 'tri_temperature_loads', [])
        ],
        'quad_temperature_loads': [
            {'quad_id': tl.quad_id, 'load_case_id': tl.load_case_id,
             'dt_i': tl.dt_i, 'dt_j': tl.dt_j, 'dt_k': tl.dt_k,
             'dt_l': tl.dt_l, 'dt_gradient': getattr(tl, 'dt_gradient', 0.0)}
            for tl in getattr(s, 'quad_temperature_loads', [])
        ],
        'area_temperature_loads': [
            {'object_id': tl.object_id, 'load_case_id': tl.load_case_id,
             'dt_uniform': tl.dt_uniform, 'field_name': tl.field_name,
             'dt_gradient': getattr(tl, 'dt_gradient', 0.0),
             'grad_field_name': getattr(tl, 'grad_field_name', '')}
            for tl in getattr(s, 'area_temperature_loads', [])
        ],
        'line_temperature_loads': [
            {'object_id': tl.object_id, 'load_case_id': tl.load_case_id,
             'dt_uniform': tl.dt_uniform, 'dt_gradient': tl.dt_gradient,
             'field_name': tl.field_name,
             'grad_field_name': getattr(tl, 'grad_field_name', '')}
            for tl in getattr(s, 'line_temperature_loads', [])
        ],
        'line_distributed_loads': [
            {'object_id': dl.object_id, 'load_case_id': dl.load_case_id,
             'fx': dl.fx, 'fy': dl.fy, 'fx_field': dl.fx_field,
             'fy_field': dl.fy_field, 'coord_sys': dl.coord_sys}
            for dl in getattr(s, 'line_distributed_loads', [])
        ],
        'line_element_springs': [
            {'object_id': ls.object_id, 'kx': ls.kx, 'ky': ls.ky,
             'kx_field': ls.kx_field, 'ky_field': ls.ky_field,
             'coord_sys': ls.coord_sys, 'mode_x': ls.mode_x,
             'mode_y': ls.mode_y}
            for ls in getattr(s, 'line_element_springs', [])
        ],
        'spectral_functions': [
            {'id': sf.id, 'description': sf.description,
             'damping': sf.damping, 'points': sf.points}
            for sf in s.spectral_functions.values()
        ],
        'nodal_masses': [
            {'node_id': nm.node_id, 'mass_case_id': nm.mass_case_id,
             'mx': nm.mx, 'my': nm.my, 'mtz': nm.mtz}
            for nm in s.nodal_masses
        ],
        'analysis_cases': [
            {'id': ac.id, 'analysis_type': ac.analysis_type,
             'coefficients': ac.coefficients,
             'num_modes': ac.num_modes,
             'modal_case_id': ac.modal_case_id,
             'spectrum_id': ac.spectrum_id,
             'combination_rule': ac.combination_rule,
             'direction': ac.direction,
             'damping': ac.damping,
             'nl_method': ac.nl_method,
             'max_iterations': ac.max_iterations,
             'tolerance': ac.tolerance,
             'beam_stiffness_factor': ac.beam_stiffness_factor,
             'column_stiffness_factor': ac.column_stiffness_factor,
             'stored_stiffness_id': ac.stored_stiffness_id,
             'sequence_id': getattr(ac, 'sequence_id', ''),
             'phase_id': getattr(ac, 'phase_id', ''),
             'solved': bool(getattr(ac, 'solved', False))}
            for ac in s.analysis_cases
        ],
        'concrete_materials': [
            {'material_name': cm.material_name, 'concrete_class': cm.concrete_class,
             'steel_class': cm.steel_class, 'gamma_c': cm.gamma_c,
             'gamma_s': cm.gamma_s, 'alpha_cc': cm.alpha_cc}
            for cm in s.concrete_materials.values()
        ],
        # Variants / phasing overlays (Phases 1–4). In the .x2d container
        # these are split out into variants.json / sequences.json (see
        # xdfem2d.file_io); merged back in here only for the plain-.json
        # format (save_structure_json/load_structure_json).
        **variant_overlays_to_dict(s),
    }


def variant_overlays_to_dict(s: Structure2D) -> dict:
    """Support sets / variants / construction sequences / variant
    combinations as a dict, shaped for either the combined structure dict
    (:func:`_to_dict`) or the split ``variants.json`` / ``sequences.json``
    payloads written by ``xdfem2d.file_io.save_x2d``."""
    return {
        'support_sets':   [_support_set_to_dict(ss)
                           for ss in getattr(s, 'support_sets', {}).values()],
        'variants':       [_variant_to_dict(v)
                           for v in getattr(s, 'variants', {}).values()],
        'construction_sequences': [_sequence_to_dict(seq)
                           for seq in getattr(s, 'construction_sequences', {}).values()],
        'variant_combinations': {
            cid: {'op': d.get('op', 'Envelope'),
                  'terms': [list(t) for t in d.get('terms', [])]}
            for cid, d in getattr(s, 'variant_combinations', {}).items()},
    }


# ── Variant / phasing (de)serialisation helpers ───────────────────────────

def _support_set_to_dict(ss) -> dict:
    return {
        'id': ss.id, 'description': getattr(ss, 'description', ''),
        'restraints': {nid: list(r)
                       for nid, r in getattr(ss, 'restraints', {}).items()},
        'assignments': [{'node_id': a.node_id, 'support_name': a.support_name}
                        for a in ss.assignments],
        'node_springs': [{'node_id': sp.node_id, 'kx': sp.kx, 'ky': sp.ky,
                          'kt': sp.kt, 'mode_x': sp.mode_x, 'mode_y': sp.mode_y}
                         for sp in ss.node_springs.values()],
        'element_springs': [{'element_id': es.element_id, 'kx': es.kx, 'ky': es.ky,
                             'coord_sys': getattr(es, 'coord_sys', 'global'),
                             'mode_x': es.mode_x, 'mode_y': es.mode_y}
                            for es in ss.element_springs.values()],
    }


def _variant_to_dict(v) -> dict:
    d = {
        'id': v.id, 'description': getattr(v, 'description', ''),
        # An empty set is normalised to None ("all"): a bar-less variant is
        # never an intent, and [] in a file used to empty the derived model.
        'active_elements': (None if not v.active_elements
                            else sorted(v.active_elements)),
        'support_set_id': v.support_set_id,
    }
    # Own action model (loads/cases/combos), serialised as a nested structure.
    if getattr(v, 'model', None) is not None:
        d['model'] = _to_dict(v.model)
    # Base-revision signature recorded at the last reconcile (see
    # xdfem2d.workflows.sync_variant_with_base): persisting it lets a reload
    # skip the redundant first sync when the base did not change.
    sig = getattr(v, '_synced_base_sig', None)
    if sig:
        d['synced_base_sig'] = sig
    return d


def _operation_to_dict(op) -> dict:
    return {
        'id': op.id,
        'action': op.action.value if hasattr(op.action, 'value') else op.action,
        'target': op.target.value if hasattr(op.target, 'value') else op.target,
        'group_kind': getattr(op, 'group_kind', ''),
        'ids': list(op.ids or ()),
        'factor': getattr(op, 'factor', 1.0),
        'note': op.note,
    }


def _phase_to_dict(p) -> dict:
    return {
        'id': p.id,
        'operations': [_operation_to_dict(op)
                       for op in getattr(p, 'operations', None) or []],
        'active_elements': (None if p.active_elements is None
                            else sorted(p.active_elements)),
        'applied_cases': list(p.applied_cases),
        'case_factors': dict(getattr(p, 'case_factors', None) or {}),
        'support_set_id': p.support_set_id,
        'inherits_from': p.inherits_from,
        'reapply_self_weight': p.reapply_self_weight,
        'release_supports': p.release_supports,
        'time': getattr(p, 'time', 0.0),
        'initial_state': {eid: {'i': list(st.i), 'j': list(st.j)}
                          for eid, st in p.initial_state.items()},
    }


def _sequence_to_dict(seq) -> dict:
    return {
        'id': seq.id,
        'displacement_method': getattr(seq, 'displacement_method', 'increment'),
        'phases': [_phase_to_dict(p) for p in seq.phases],
    }


def _from_dict(data: dict) -> Structure2D:
    # Absent 'domain' means 'plane': every file written before the field
    # existed is an in-plane model and must load exactly as it always did.
    s = Structure2D(domain=data.get('domain', 'plane'))
    pinfo = data.get('project_info')
    if isinstance(pinfo, dict):
        # Keep fixed fields first, then any custom pairs.
        merged = dict(s.project_info)
        merged.update({str(k): str(v) for k, v in pinfo.items()})
        s.project_info = merged
    # Absent in every file written before this existed, and those default to
    # on — so a model saved earlier whose surface has two consecutive corners
    # under the same support gains the nodes between them. That is a real
    # change to a saved model's results and it is deliberate: the alternative
    # was every existing wall staying held at its corners for ever. The flag
    # is here so it can be turned off per model.
    s.propagate_edge_supports = bool(
        data.get('propagate_edge_supports', True))
    s.propagate_edge_springs = bool(
        data.get('propagate_edge_springs', True))
    for n in data.get('nodes', []):
        s.add_node(n['id'], n['x'], n['y'])
    for m in data.get('materials', []):
        # _add_material_unchecked, not add_material: a saved file's design
        # dict already went through whatever created it, and the plain
        # loader stays permissive on a stale/hand-edited class label --
        # that is the checked loader's job (test_structure_io_checked.py),
        # not the open's (26/09/2026, see Structure2D._add_material_unchecked).
        s._add_material_unchecked(m['name'], m['elastic_modulus'], m['unit_weight'],
                                  alpha=m.get('alpha', 1.0e-5),
                                  unit_mass=m.get('unit_mass', None),
                                  material_type=m.get('material_type', 'Concrete'),
                                  poisson=m.get('poisson', 0.2),
                                  design=m.get('design', {}))
    for sc in data.get('sections', []):
        s.add_section(sc['name'], sc['material_name'], sc['b'], sc['h'],
                      area_override=sc.get('area_override'),
                      inertia_override=sc.get('inertia_override'),
                      inertia_minor_override=sc.get('inertia_minor_override'),
                      torsion_override=sc.get('torsion_override'),
                      wel_y_override=sc.get('wel_y_override'),
                      wpl_y_override=sc.get('wpl_y_override'),
                      wel_z_override=sc.get('wel_z_override'),
                      wpl_z_override=sc.get('wpl_z_override'),
                      av_y_override=sc.get('av_y_override'),
                      av_z_override=sc.get('av_z_override'),
                      warping_override=sc.get('warping_override'),
                      wt_override=sc.get('wt_override'),
                      curve_y_override=sc.get('curve_y_override'),
                      curve_z_override=sc.get('curve_z_override'),
                      profile_name=sc.get('profile_name'),
                      angle=sc.get('angle', 0.0),
                      shape=sc.get('shape', 'Generic'),
                      tw=sc.get('tw', 0.0), tf=sc.get('tf', 0.0),
                      rc_cover=sc.get('rc_cover', 0.045),
                      rc_alpha_s=sc.get('rc_alpha_s', 90.0),
                      rc_bar_phi=sc.get('rc_bar_phi', 16.0),
                      rc_shear_min=sc.get('rc_shear_min', True),
                      rc_torsion_distribution=sc.get(
                          'rc_torsion_distribution', 'top_bottom'),
                      is_column=sc.get('is_column', False),
                      rc_phi_ef=sc.get('rc_phi_ef', 2.0),
                      rc_n_bars=sc.get('rc_n_bars', 4),
                      rc_n_bars_y=sc.get('rc_n_bars_y', 2),
                      rc_n_bars_z=sc.get('rc_n_bars_z', 2),
                      rc_second_order_method=sc.get(
                          'rc_second_order_method', 'nominal_curvature'),
                      timber_service_class=sc.get('timber_service_class', 'SC1'),
                      beam_overrides=sc.get('beam_overrides'))
    for e in data.get('bar_elements', []):
        elem = s.add_bar_element(e['id'], e['node_i'], e['node_j'], e['section_name'],
                                 hinge_i=e.get('hinge_i', False),
                                 hinge_j=e.get('hinge_j', False))
        elem.rc_design    = e.get('rc_design', False)
        elem.rc_cover     = e.get('rc_cover', 0.0)
        elem.rc_cotg_theta= e.get('rc_cotg_theta', 1.0)
        elem.rc_alpha_s   = e.get('rc_alpha_s', 90.0)
        elem.sd_ky        = e.get('sd_ky')
        elem.sd_kz        = e.get('sd_kz')
        elem.sd_klt       = e.get('sd_klt')
        elem.sd_ltb       = e.get('sd_ltb', True)
        elem.is_column    = e.get('is_column')
        elem.beam         = e.get('beam')
        elem.stage        = int(e.get('stage', 1) or 1)
    for ts in data.get('tri_sections', []):
        s.add_tri_section(ts['name'], ts['material_name'],
                          thickness=ts.get('thickness', 0.1),
                          plane_strain=bool(ts.get('plane_strain', False)),
                          formulation=ts.get('formulation', 'CST'),
                          rc_cover=ts.get('rc_cover', 0.045),
                          rc_alpha_s=ts.get('rc_alpha_s', 90.0),
                          rc_bar_phi=ts.get('rc_bar_phi', 16.0),
                          rc_shear_min=bool(ts.get('rc_shear_min', True)),
                          rc_cover_top_x=ts.get('rc_cover_top_x'),
                          rc_cover_top_y=ts.get('rc_cover_top_y'),
                          rc_cover_bot_x=ts.get('rc_cover_bot_x'),
                          rc_cover_bot_y=ts.get('rc_cover_bot_y'))

    def _tri_section_for(mat, thk, ps):
        """Find (or create) a TriSection matching a legacy element's material +
        thickness + plane-strain (backward compat for pre-section files)."""
        for cand in s.tri_sections.values():
            if (cand.material_name == mat and abs(cand.thickness - thk) < 1e-12
                    and cand.plane_strain == ps):
                return cand.name
        name = f"CST-{mat}-{thk:g}{'-eps' if ps else ''}"
        base = name; n = 1
        while name in s.tri_sections:
            n += 1; name = f"{base}#{n}"
        s.add_tri_section(name, mat, thickness=thk, plane_strain=ps)
        return name

    for t in data.get('tri_elements', []):
        sec_name = t.get('section_name')
        if sec_name is None:   # legacy element carrying material + thickness
            sec_name = _tri_section_for(
                t['material_name'], float(t.get('thickness', 0.1)),
                bool(t.get('plane_strain', False)))
        tri = s.add_tri_element(t['id'], t['node_i'], t['node_j'], t['node_k'],
                                sec_name)
        tri.stage = int(t.get('stage', 1) or 1)
    for qs in data.get('quad_sections', []):
        s.add_quad_section(qs['name'], qs['material_name'],
                           thickness=qs.get('thickness', 0.1),
                           plane_strain=bool(qs.get('plane_strain', False)),
                           formulation=qs.get('formulation', 'MITC4'),
                           rc_cover=qs.get('rc_cover', 0.045),
                           rc_alpha_s=qs.get('rc_alpha_s', 90.0),
                           rc_bar_phi=qs.get('rc_bar_phi', 16.0),
                           rc_shear_min=bool(qs.get('rc_shear_min', True)),
                           rc_cover_top_x=qs.get('rc_cover_top_x'),
                           rc_cover_top_y=qs.get('rc_cover_top_y'),
                           rc_cover_bot_x=qs.get('rc_cover_bot_x'),
                           rc_cover_bot_y=qs.get('rc_cover_bot_y'))
    for q in data.get('quad_elements', []):
        quad = s.add_quad_element(q['id'], q['node_i'], q['node_j'],
                                  q['node_k'], q['node_l'], q['section_name'])
        quad.stage = int(q.get('stage', 1) or 1)
    for a in data.get('quad_area_loads', []):
        s.add_area_load(a['quad_id'], a['load_case_id'], pz=a.get('pz', 0.0))
    for e in data.get('tri_edge_loads', []):
        s.add_tri_edge_load(e['id'], e['tri_id'], e['node_a'], e['node_b'],
                            e['load_case_id'], fx=e.get('fx', 0.0),
                            fy=e.get('fy', 0.0),
                            coord_sys=e.get('coord_sys', 'global'),
                            pn=e.get('pn', 0.0), pt=e.get('pt', 0.0))
    for e in data.get('quad_edge_loads', []):
        s.add_quad_edge_load(e['id'], e['quad_id'], e['node_a'], e['node_b'],
                             e['load_case_id'], fx=e.get('fx', 0.0),
                             fy=e.get('fy', 0.0),
                             coord_sys=e.get('coord_sys', 'global'),
                             pn=e.get('pn', 0.0), pt=e.get('pt', 0.0))
    for fld in data.get('fields', []):
        s.add_field(fld['name'], fld.get('expression', '0.0'))
    _geo_objects_from_list(s, data.get('geometry_objects', []))
    s.beams = _beams_from_dict(data.get('beams'))
    _beam_detail_from_dict(s, data.get('beam_detail'))
    # add_surface_edge_load now validates its target is a surface object
    # (rectangle/polygon), so it has to run after _geo_objects_from_list —
    # loaded any earlier, geometry_objects is still empty and every one of
    # these would raise.
    for e in data.get('surface_edge_loads', []):
        s.add_surface_edge_load(e['id'], e['object_id'], e['node_a'],
                                e['node_b'], e['load_case_id'],
                                fx=e.get('fx', 0.0), fy=e.get('fy', 0.0),
                                coord_sys=e.get('coord_sys', 'global'),
                                pn=e.get('pn', 0.0), pt=e.get('pt', 0.0))
    # Area loads (plate domain). Rebuilt directly — add_area_load dispatches
    # on what the target currently is, and here the kind is already known.
    from .models import TriAreaLoad, SurfaceAreaLoad
    for a in data.get('tri_area_loads', []):
        s.tri_area_loads.append(TriAreaLoad(
            tri_id=a['tri_id'], load_case_id=a['load_case_id'],
            pz=a.get('pz', 0.0)))
    for a in data.get('surface_area_loads', []):
        s.surface_area_loads.append(SurfaceAreaLoad(
            object_id=a['object_id'], load_case_id=a['load_case_id'],
            pz=a.get('pz', 0.0)))
    from .models import TriAreaSpring, QuadAreaSpring, SurfaceAreaSpring
    for a in data.get('tri_area_springs', []):
        s.tri_area_springs.append(TriAreaSpring(
            tri_id=a['tri_id'], kz=a.get('kz', 0.0)))
    for a in data.get('quad_area_springs', []):
        s.quad_area_springs.append(QuadAreaSpring(
            quad_id=a['quad_id'], kz=a.get('kz', 0.0)))
    for a in data.get('surface_area_springs', []):
        s.surface_area_springs.append(SurfaceAreaSpring(
            object_id=a['object_id'], kz=a.get('kz', 0.0)))
    for c in data.get('punch_columns', []):
        s.add_punch_column(c['id'], c['node_id'],
                           shape=c.get('shape', 'rectangular'),
                           bx=c.get('bx', 0.30), by=c.get('by', 0.30),
                           position=c.get('position', 'center'),
                           force=c.get('force'),
                           dx=c.get('dx', 0.0), dy=c.get('dy', 0.0),
                           beta_min=c.get('beta_min', 1.0))
    for sp in data.get('supports', []):
        # .get, not []: a restraint left out means 'free', which is what
        # every default in add_support already says. Requiring all three
        # turned an omitted 'tz' into KeyError: 'tz' — an error naming a
        # key, from inside the loader, about a value nobody had to give.
        # Plate-domain aliases (w/tx/ty) are accepted alongside the canonical
        # keys, so a hand- or assistant-written plate model reads naturally.
        s.add_support(sp['name'],
                      sp.get('ux', sp.get('w', False)),
                      sp.get('uy', sp.get('tx', False)),
                      sp.get('tz', sp.get('ty', False)))
    for a in data.get('support_assignments', []):
        s.assign_support(a['node_id'], a['support_name'])
    for sp in data.get('node_springs', []):
        # Support both new format (node_id key) and old format (name + separate assignments)
        if 'node_id' in sp:
            s.add_node_spring(sp['node_id'],
                              sp.get('kx', sp.get('kz', 0.0)),
                              sp.get('ky', sp.get('ktx', 0.0)),
                              sp.get('kt', sp.get('kty', 0.0)),
                              mode_x=sp.get('mode_x', 'both'),
                              mode_y=sp.get('mode_y', 'both'))
    # Legacy: build node_springs from old name+assignment tables if present
    _old_springs = {sp['name']: sp for sp in data.get('node_springs', []) if 'name' in sp}
    for a in data.get('node_spring_assignments', []):
        sp = _old_springs.get(a['spring_name'], {})
        s.add_node_spring(a['node_id'], sp.get('kx', 0.0), sp.get('ky', 0.0), sp.get('kt', 0.0))
    for es in data.get('element_springs', []):
        # Support both new (no 'id') and old format (had 'id' field)
        s.add_element_spring(es['element_id'], es.get('kx', 0.0), es.get('ky', 0.0),
                             coord_sys=es.get('coord_sys', 'global'),
                             mode_x=es.get('mode_x', 'both'),
                             mode_y=es.get('mode_y', 'both'))
    # Constraints (multi-point). Rebuilt through the same add_* methods used
    # everywhere else, so a loaded model is indistinguishable from a built one.
    for c in data.get('constraints', []):
        kind = c.get('kind', 'equation')
        cid = c.get('id', '')
        if kind == 'rigid_link':
            con = s.add_rigid_link(c.get('master', ''),
                                   list(c.get('slaves', [])), id=cid)
        elif kind == 'equal_dof':
            con = s.add_equal_dof(list(c.get('nodes', [])),
                                  list(c.get('components', [])), id=cid)
        else:
            terms = [(t[0], t[1], t[2]) for t in c.get('terms', [])]
            con = s.add_constraint_equation(terms, c.get('value', 0.0), id=cid)
        con.enabled = bool(c.get('enabled', True))
    for lc in data.get('load_cases', []):
        # Twin analysis cases are restored from the saved analysis_cases table
        # below; don't auto-create them here (would duplicate).
        s.add_load_case(lc['id'], lc.get('self_weight_factor', 0.0),
                        action_type=lc.get('action_type', 'G'),
                        create_analysis_case=False,
                        load_duration=lc.get('load_duration'),
                        category=lc.get('category', ''),
                        psi0=lc.get('psi0'), psi1=lc.get('psi1'),
                        psi2=lc.get('psi2'))
    for pl in data.get('point_loads', []):
        # fz/mx/my are the plate-domain aliases of the same three slots.
        s.add_point_load(pl['node_id'], pl['load_case_id'],
                         pl.get('fx', pl.get('fz', 0.0)),
                         pl.get('fy', pl.get('mx', 0.0)),
                         pl.get('mz', pl.get('my', 0.0)))
    for dl in data.get('distributed_loads', []):
        s.add_distributed_load(
            dl['element_id'], dl['load_case_id'],
            fxe=dl.get('fxe', 0.0), fxd=dl.get('fxd', 0.0),
            fye=dl.get('fye', dl.get('fze', 0.0)),
            fyd=dl.get('fyd', dl.get('fzd', 0.0)),
            coord_sys=dl.get('coord_sys', 'global'))
    for epl in data.get('element_point_loads', []):
        s.add_element_point_load(
            epl['element_id'], epl['load_case_id'], epl.get('a', 0.0),
            fx=epl.get('fx', 0.0), fy=epl.get('fy', 0.0), mz=epl.get('mz', 0.0),
            coord_sys=epl.get('coord_sys', 'global'))
    for ss in data.get('support_settlements', []):
        s.create_support_settlement(ss['node_id'], ss['load_case_id'],
                                 ss.get('ux', ss.get('w', 0.0)),
                                 ss.get('uy', ss.get('tx', 0.0)),
                                 ss.get('tz', ss.get('ty', 0.0)))
    for tl in data.get('temperature_loads', []):
        s.add_temperature_load(tl['element_id'], tl['load_case_id'],
                               tl.get('delta_t_uniform', 0.0),
                               tl.get('delta_t_gradient', 0.0),
                               tl.get('alpha', 1e-5))
    for tl in data.get('tri_temperature_loads', []):
        s.add_tri_temperature_load(tl['tri_id'], tl['load_case_id'],
                                   tl.get('dt_i', 0.0), tl.get('dt_j', 0.0),
                                   tl.get('dt_k', 0.0),
                                   tl.get('dt_gradient', 0.0))
    for tl in data.get('quad_temperature_loads', []):
        s.add_quad_temperature_load(tl['quad_id'], tl['load_case_id'],
                                    tl.get('dt_i', 0.0), tl.get('dt_j', 0.0),
                                    tl.get('dt_k', 0.0), tl.get('dt_l', 0.0),
                                    tl.get('dt_gradient', 0.0))
    for tl in data.get('area_temperature_loads', []):
        s.add_area_temperature_load(
            tl['object_id'], tl['load_case_id'],
            dt_uniform=tl.get('dt_uniform', 0.0),
            dt_gradient=tl.get('dt_gradient', 0.0),
            field_name=tl.get('field_name', ''),
            grad_field_name=tl.get('grad_field_name', ''))
    for tl in data.get('line_temperature_loads', []):
        s.add_line_temperature_load(tl['object_id'], tl['load_case_id'],
                                    tl.get('dt_uniform', 0.0),
                                    tl.get('dt_gradient', 0.0),
                                    tl.get('field_name', ''),
                                    tl.get('grad_field_name', ''))
    for dl in data.get('line_distributed_loads', []):
        s.add_line_distributed_load(dl['object_id'], dl['load_case_id'],
                                    dl.get('fx', 0.0), dl.get('fy', 0.0),
                                    dl.get('fx_field', ''),
                                    dl.get('fy_field', ''),
                                    dl.get('coord_sys', 'global'))
    for ls in data.get('line_element_springs', []):
        s.add_line_element_spring(ls['object_id'],
                                  ls.get('kx', 0.0), ls.get('ky', 0.0),
                                  ls.get('kx_field', ''),
                                  ls.get('ky_field', ''),
                                  ls.get('coord_sys', 'global'),
                                  ls.get('mode_x', 'both'),
                                  ls.get('mode_y', 'both'))
    for sf in data.get('spectral_functions', []):
        s.add_spectral_function(sf['id'], sf.get('description', ''),
                                sf.get('damping', 0.05), sf.get('points', []))
    for nm in data.get('nodal_masses', []):
        s.add_nodal_mass(nm['node_id'], nm['mass_case_id'],
                         nm.get('mx', 0.0), nm.get('my', 0.0), nm.get('mtz', 0.0))
    for ac in data.get('analysis_cases', []):
        s.add_analysis_case(
            ac['id'], ac.get('analysis_type', 'Linear'),
            coefficients=ac.get('coefficients', {}),
            num_modes=ac.get('num_modes', 10),
            modal_case_id=ac.get('modal_case_id', ''),
            spectrum_id=ac.get('spectrum_id', ''),
            combination_rule=ac.get('combination_rule', 'SRSS'),
            direction=ac.get('direction', 'XY'),
            damping=ac.get('damping', 0.05),
            nl_method=ac.get('nl_method', 'P-Delta'),
            max_iterations=ac.get('max_iterations', 100),
            tolerance=ac.get('tolerance', 1e-6),
            beam_stiffness_factor=ac.get('beam_stiffness_factor', 1.0),
            column_stiffness_factor=ac.get('column_stiffness_factor', 1.0),
            stored_stiffness_id=ac.get('stored_stiffness_id', ''),
            sequence_id=ac.get('sequence_id', ''),
            phase_id=ac.get('phase_id', ''),
            solved=bool(ac.get('solved', False)),
        )
    # Load combinations are restored AFTER the analysis cases (and spectral
    # functions), so a NonLinearCombo can validate the non-linear analysis case
    # it wraps — otherwise the wrapped case would not yet exist.
    for lc in data.get('load_combinations', []):
        # Translate an old (pre-26/09/2026) Portuguese combo_type -- and the
        # old 'SomaSodulo' misspelling -- to its English equivalent, so a file
        # saved before the rename still loads and is written back out in
        # English the next time it is saved.
        from .models import coerce_combo_type
        ctype = coerce_combo_type(lc.get('combo_type', 'LinearSum'))
        combo = s.add_load_combination(lc['id'], lc.get('coefficients', {}),
                                       combo_type=ctype)
        # Preserve any legacy analysis_coefficients; normalised at the end.
        combo.analysis_coefficients = dict(lc.get('analysis_coefficients', {}))
    for c in data.get('cuts', []):
        s.add_cut(c['id'], c.get('x1', 0.0), c.get('y1', 0.0),
                  c.get('x2', 0.0), c.get('y2', 0.0), name=c.get('name', ''))
    for cm in data.get('concrete_materials', []):
        s.add_concrete_material(
            cm['material_name'], cm['concrete_class'], cm['steel_class'],
            cm.get('gamma_c', 1.5), cm.get('gamma_s', 1.15), cm.get('alpha_cc', 1.0))
    # Ensure load-case twins exist and combinations reference analysis cases
    # (also migrates legacy saved files where combos referenced load cases).
    s.normalize_combinations_to_analysis_cases()
    # Variants / phasing overlays. In the .x2d container these live in their
    # own variants.json / sequences.json (see xdfem2d.file_io); here they are
    # only present when *data* is the combined dict used by
    # save_structure_json/load_structure_json (the plain-.json format).
    load_variant_overlays(s, data)
    return s


def load_variant_overlays(s: Structure2D, data: dict) -> None:
    """Populate *s*'s support sets / variants / construction sequences /
    variant combinations from a dict shaped like the one
    :func:`variant_overlays_to_dict` produces.

    Shared by :func:`_from_dict` (the combined structure dict) and
    ``xdfem2d.file_io.load_x2d`` (the split ``variants.json`` /
    ``sequences.json`` payloads) so the two containers stay in sync without
    duplicating the (de)serialisation logic.
    """
    for d in data.get('support_sets', []):
        s.add_support_set(_support_set_from_dict(d))
    base_bar_ids = set(s.bar_elements_by_id)
    for d in data.get('variants', []):
        v = _variant_from_dict(d)
        # Self-heal a variant whose declared ``active_elements`` disagrees
        # with what its own model actually contains — the model is what is
        # really displayed/solved, so it must win. This happens for a
        # variant that was edited directly on the canvas (add/remove
        # element) while it was the active toolbar selection: that mutates
        # the model in place but ``active_elements`` is only ever written by
        # the Variants panel's "Save" (see
        # MainWindow._sync_active_variant_active_elements, which prevents
        # this going forward for a variant still open in the GUI — but a
        # file saved before that existed, or before this session's fix, can
        # still be loaded with the two out of sync). Left uncorrected, the
        # very next non-destructive reconcile with the base
        # (workflows.sync_variant_with_base, run on every selection change /
        # Run) trusts the stale declared set and silently re-adds bars the
        # user had removed.
        if v.model is not None:
            model_ids = set(v.model.bar_elements_by_id)
            declared = set(v.active_elements) if v.active_elements else base_bar_ids
            if model_ids != declared:
                v.active_elements = (None if model_ids == base_bar_ids
                                     else model_ids)
        s.add_variant(v)
    for d in data.get('construction_sequences', []):
        s.add_construction_sequence(_sequence_from_dict(d))
    vc = data.get('variant_combinations', {})
    if isinstance(vc, dict):
        from .models import coerce_combo_type
        s.variant_combinations = {
            cid: {'op': coerce_combo_type(cd.get('op', 'Envelope')),
                  'terms': [tuple(t) for t in cd.get('terms', [])]}
            for cid, cd in vc.items()}


def _support_set_from_dict(d):
    from .models import SupportSet, SupportAssignment, NodeSpring, ElementSpring
    return SupportSet(
        id=d['id'], description=d.get('description', ''),
        restraints={nid: tuple(r)
                    for nid, r in d.get('restraints', {}).items()},
        assignments=[SupportAssignment(node_id=a['node_id'],
                                       support_name=a['support_name'])
                     for a in d.get('assignments', [])],
        node_springs={sp['node_id']: NodeSpring(
            node_id=sp['node_id'], kx=sp.get('kx', 0.0), ky=sp.get('ky', 0.0),
            kt=sp.get('kt', 0.0), mode_x=sp.get('mode_x', 'both'),
            mode_y=sp.get('mode_y', 'both')) for sp in d.get('node_springs', [])},
        element_springs={es['element_id']: ElementSpring(
            element_id=es['element_id'], kx=es.get('kx', 0.0), ky=es.get('ky', 0.0),
            coord_sys=es.get('coord_sys', 'global'), mode_x=es.get('mode_x', 'both'),
            mode_y=es.get('mode_y', 'both')) for es in d.get('element_springs', [])},
    )


def _variant_from_dict(d):
    from .models import Variant
    ae = d.get('active_elements')
    model = _from_dict(d['model']) if d.get('model') is not None else None
    v = Variant(id=d['id'], description=d.get('description', ''),
                active_elements=(set(ae) if ae else None),
                support_set_id=d.get('support_set_id'),
                model=model)
    if d.get('synced_base_sig'):
        v._synced_base_sig = d['synced_base_sig']
    return v


def _operation_from_dict(d):
    from .models import Operation, OpAction, OpTarget
    try:
        target = OpTarget(d.get('target', 'elements'))
    except ValueError:
        # A short-lived 'springs' target existed for one session before
        # being folded into SUPPORTS (a SupportSet already carries
        # node_springs/element_springs alongside restraints) — any file
        # saved with it drops the operation's target back to a safe
        # default rather than failing to load the whole sequence.
        target = OpTarget.ELEMENTS
    return Operation(
        id=d['id'],
        action=OpAction(d.get('action', 'add')),
        target=target,
        group_kind=d.get('group_kind', ''),
        ids=tuple(d.get('ids', [])),
        factor=float(d.get('factor', 1.0)),
        note=d.get('note', ''),
    )


def _phase_from_dict(d):
    from .models import ConstructionPhase, ElementInitialState
    ae = d.get('active_elements')
    return ConstructionPhase(
        id=d['id'],
        operations=[_operation_from_dict(op)
                    for op in d.get('operations', [])],
        active_elements=(None if ae is None else set(ae)),
        applied_cases=list(d.get('applied_cases', [])),
        case_factors={k: float(v) for k, v
                     in (d.get('case_factors') or {}).items()},
        support_set_id=d.get('support_set_id'),
        inherits_from=d.get('inherits_from'),
        reapply_self_weight=d.get('reapply_self_weight', False),
        release_supports=d.get('release_supports', True),
        time=d.get('time', 0.0),
        initial_state={eid: ElementInitialState(i=tuple(v['i']), j=tuple(v['j']))
                       for eid, v in d.get('initial_state', {}).items()},
    )


def _sequence_from_dict(d):
    from .models import ConstructionSequence
    return ConstructionSequence(
        id=d['id'],
        displacement_method=d.get('displacement_method', 'increment'),
        phases=[_phase_from_dict(p) for p in d.get('phases', [])],
    )
