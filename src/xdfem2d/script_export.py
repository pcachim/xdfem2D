"""Write the open model as a commented Python script that rebuilds it.

The script calls the same public API a user would type by hand, so it doubles
as documentation of the model, as something a version control system can show a
diff of, and as a starting point for a variation ("this, but with the span at
7 m"). It is plain text: reading it changes nothing, and running it is the
reader's decision, made after reading.

Two rules shape everything here.

**Nothing is omitted silently.** A script that quietly leaves out the variants
or the construction sequences rebuilds a *different* structure while looking
complete, and the reader has no way of telling. What this module cannot express
is listed in :data:`DECLARED_UNSUPPORTED`, checked against the save format by
the tests, and — when the model actually contains one of those things — printed
in a warning block at the top of the generated file. A gap the reader can see
is a limitation; a gap they cannot is a lie.

**Arguments that match the default are left out.** Not to be clever: a call
carrying every parameter of ``add_section`` buries the two numbers that matter
among ten that never change. The defaults are read from the live signatures, so
this cannot drift away from the API it is writing.
"""
from __future__ import annotations

import inspect
from typing import Any

from xdfem2d.structure import Structure2D

# Keys of the save format that this module writes out. Compared against
# structure_io._to_dict by the tests, so adding a feature to the model without
# deciding what the exporter does with it fails there rather than here.
HANDLED = {
    'domain',   # written as Structure2D(domain=...) when it is not 'plane'
    'project_info', 'nodes', 'materials', 'concrete_materials', 'sections',
    'bar_elements', 'tri_sections', 'tri_elements', 'geometry_objects',
    'beams', 'beam_detail',
    'quad_sections', 'quad_elements', 'quad_area_loads',
    'supports', 'support_assignments', 'node_springs', 'element_springs',
    'load_cases', 'point_loads', 'distributed_loads', 'element_point_loads',
    'tri_edge_loads', 'quad_edge_loads', 'surface_edge_loads',
    'support_settlements',
    'tri_area_loads', 'surface_area_loads',
    'tri_area_springs', 'quad_area_springs', 'surface_area_springs',
    'punch_columns',
    'temperature_loads', 'tri_temperature_loads', 'quad_temperature_loads',
    'area_temperature_loads',
    'line_temperature_loads', 'line_distributed_loads', 'line_element_springs',
    'load_combinations', 'analysis_cases', 'construction_sequences',
    'variants', 'variant_combinations',
    'nodal_masses', 'spectral_functions', 'propagate_edge_supports',
    'propagate_edge_springs', 'constraints',
}

# Knowingly not written, with the sentence the reader is shown. These are the
# staging and study features: they are built through workflows rather than
# through a handful of add_* calls, and a half-faithful reconstruction of one
# would be worse than an honest gap.
DECLARED_UNSUPPORTED = {
    'fields': 'nodal fields',
    'support_sets': 'support sets',
    # Section cuts (dev/CUT_PLAN.md): entity + persistence only so far
    # (Phase 1). Script-export support belongs with the later GUI/API
    # phases; move to HANDLED once add_cut() calls are written out.
    'cuts': 'section cuts',
}


def _lit(v: Any) -> str:
    """A value as Python source. Floats keep their meaning, not their noise."""
    if isinstance(v, float):
        # repr() round-trips exactly, which matters: a coordinate written as
        # 0.1 must not come back as 0.09999999999999999, and a model rebuilt
        # from rounded numbers is a different model.
        return repr(v)
    if isinstance(v, (str, bool, int)) or v is None:
        return repr(v)
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_lit(k)}: {_lit(x)}"
                               for k, x in v.items()) + "}"
    if isinstance(v, (list, tuple)):
        inner = ", ".join(_lit(x) for x in v)
        if isinstance(v, tuple):
            # single-element tuple needs trailing comma: ('x',) not ('x')
            return f"({inner},)" if len(v) == 1 else f"({inner})"
        return f"[{inner}]"
    return repr(v)


def _defaults(method: str) -> dict:
    try:
        sig = inspect.signature(getattr(Structure2D, method))
    except (AttributeError, TypeError, ValueError):   # pragma: no cover
        return {}
    return {n: p.default for n, p in sig.parameters.items()
            if p.default is not inspect.Parameter.empty}


def _call(method: str, *args, **kwargs) -> str:
    """One ``s.method(...)`` line, without the arguments that change nothing."""
    defaults = _defaults(method)
    parts = [_lit(a) for a in args]
    for name, value in kwargs.items():
        if name in defaults and _same(value, defaults[name]):
            continue
        parts.append(f"{name}={_lit(value)}")
    return f"    model.{method}({', '.join(parts)})"


def _same(a, b) -> bool:
    """Equality that does not raise on the odd type, and treats 1.0 as 1."""
    try:
        return bool(a == b)
    except Exception:                                # noqa: BLE001
        return False


def _enum(v, fallback=""):
    """The value of an enum field, or the field itself when it is plain."""
    return getattr(v, 'value', v if v is not None else fallback)


def unsupported_in(struc) -> list[str]:
    """Which declared-unsupported features this model actually contains."""
    from xdfem2d.structure_io import _to_dict
    data = _to_dict(struc)
    return sorted(label for key, label in DECLARED_UNSUPPORTED.items()
                  if data.get(key))


def to_python(struc, source: str = "") -> str:
    """The model as a runnable, commented Python script."""
    L: list[str] = []
    add = L.append

    missing = unsupported_in(struc)
    info = dict(getattr(struc, 'project_info', {}) or {})

    # ── Header ───────────────────────────────────────────────────
    add('"""Model rebuilt in Python, generated by xdfem2D.')
    add('')
    if source:
        add(f"Source: {source}")
    # The real keys of project_info, whatever the user has put in it. Written
    # out as they are: this is a free-form dictionary with five fixed entries
    # and any others the user added, and guessing at names ('name', 'author')
    # meant no model ever had its metadata in the header.
    for key, value in info.items():
        if str(value).strip():
            add(f"{key}: {value}")
    add('')
    add('Units: metres, kN, kN*m, kN/m2. Angles in degrees where an angle is')
    add('named. Running this file builds the model, solves it, and writes the')
    add('result beside itself as an .x2d you can open in xdfem2D.')
    if missing:
        add('')
        add('!! NOT INCLUDED. The model contains features this exporter does')
        add('!! not write, so the structure built here is not the whole model:')
        for m in missing:
            add(f"!!   - {m}")
        add('!! Open the .x2d for those.')
    add('"""')
    add('from xdfem2d import Structure2D')
    add('')
    add('')
    add('def build() -> Structure2D:')
    add('    """Build and return the model."""')
    dom = getattr(struc, 'domain', 'plane')
    add('    model = Structure2D()' if dom == 'plane'
        else f'    model = Structure2D(domain={dom!r})')

    def section(title: str, n: int) -> None:
        add('')
        add(f"    # ── {title} ({n}) " + "─" * max(0, 46 - len(title)))

    # ── Project metadata ─────────────────────────────────────────
    # In the header for a reader, and here for the model: a script that
    # rebuilds everything except who designed it comes back as a different
    # file, and the round-trip test says so.
    filled = {k: v for k, v in info.items() if str(v).strip()}
    if filled:
        section('Project', len(filled))
        for key, value in filled.items():
            add(f"    model.project_info[{key!r}] = {value!r}")

    # Written only when it is off, because on is the default and a script that
    # restates every default is harder to read. Written at all because leaving
    # it out would silently re-enable it: a wall the user had deliberately kept
    # on two corner supports would come back held along its whole edge.
    if not getattr(struc, 'propagate_edge_supports', True):
        add("    model.propagate_edge_supports = False")
    if not getattr(struc, 'propagate_edge_springs', True):
        add("    model.propagate_edge_springs = False")

    # ── Materials ────────────────────────────────────────────────
    mats = list((getattr(struc, 'materials', {}) or {}).values())
    if mats:
        section('Materials', len(mats))
        for m in mats:
            add(_call('add_material', m.name, m.elastic_modulus,
                      m.unit_weight, alpha=m.alpha, unit_mass=m.unit_mass,
                      material_type=_enum(getattr(m, 'material_type', None)),
                      poisson=getattr(m, 'poisson', 0.2),
                      design=dict(getattr(m, 'design', {}) or {})))

    cmats = list((getattr(struc, 'concrete_materials', {}) or {}).values())
    if cmats:
        section('Concrete design materials', len(cmats))
        for c in cmats:
            add(_call('add_concrete_material', c.material_name,
                      c.concrete_class, c.steel_class,
                      gamma_c=c.gamma_c, gamma_s=c.gamma_s,
                      alpha_cc=c.alpha_cc))

    # ── Sections ─────────────────────────────────────────────────
    secs = list((getattr(struc, 'sections', {}) or {}).values())
    if secs:
        section('Bar sections', len(secs))
        for sc in secs:
            add(_call('add_section', sc.name, sc.material_name, sc.b, sc.h,
                      area_override=sc.area_override,
                      inertia_override=sc.inertia_override,
                      inertia_minor_override=getattr(
                          sc, 'inertia_minor_override', None),
                      torsion_override=getattr(sc, 'torsion_override', None),
                      wel_y_override=getattr(sc, 'wel_y_override', None),
                      wpl_y_override=getattr(sc, 'wpl_y_override', None),
                      wel_z_override=getattr(sc, 'wel_z_override', None),
                      wpl_z_override=getattr(sc, 'wpl_z_override', None),
                      av_y_override=getattr(sc, 'av_y_override', None),
                      av_z_override=getattr(sc, 'av_z_override', None),
                      warping_override=getattr(sc, 'warping_override', None),
                      wt_override=getattr(sc, 'wt_override', None),
                      curve_y_override=getattr(sc, 'curve_y_override', None),
                      curve_z_override=getattr(sc, 'curve_z_override', None),
                      profile_name=sc.profile_name,
                      angle=getattr(sc, 'angle', 0.0),
                      shape=_enum(getattr(sc, 'shape', None)),
                      tw=sc.tw, tf=sc.tf,
                      rc_cover=sc.rc_cover, rc_alpha_s=sc.rc_alpha_s,
                      rc_bar_phi=sc.rc_bar_phi,
                      rc_shear_min=sc.rc_shear_min,
                      rc_torsion_distribution=getattr(
                          sc, 'rc_torsion_distribution', 'top_bottom'),
                      is_column=getattr(sc, 'is_column', False),
                      rc_phi_ef=getattr(sc, 'rc_phi_ef', 2.0),
                      rc_n_bars=getattr(sc, 'rc_n_bars', 4),
                      rc_n_bars_y=getattr(sc, 'rc_n_bars_y', 2),
                      rc_n_bars_z=getattr(sc, 'rc_n_bars_z', 2),
                      rc_second_order_method=getattr(
                          sc, 'rc_second_order_method', 'nominal_curvature'),
                      timber_service_class=getattr(
                          sc, 'timber_service_class', 'SC1'),
                      beam_overrides=getattr(sc, 'beam_overrides', None)))

    tsecs = list((getattr(struc, 'tri_sections', {}) or {}).values())
    if tsecs:
        section('Triangle sections', len(tsecs))
        for ts in tsecs:
            add(_call('add_tri_section', ts.name, ts.material_name,
                      thickness=ts.thickness, plane_strain=ts.plane_strain,
                      formulation=getattr(ts, 'formulation', 'CST'),
                      rc_cover=getattr(ts, 'rc_cover', 0.045),
                      rc_alpha_s=getattr(ts, 'rc_alpha_s', 90.0),
                      rc_bar_phi=getattr(ts, 'rc_bar_phi', 16.0),
                      rc_shear_min=getattr(ts, 'rc_shear_min', True),
                      rc_cover_top_x=getattr(ts, 'rc_cover_top_x', None),
                      rc_cover_top_y=getattr(ts, 'rc_cover_top_y', None),
                      rc_cover_bot_x=getattr(ts, 'rc_cover_bot_x', None),
                      rc_cover_bot_y=getattr(ts, 'rc_cover_bot_y', None)))

    # dev/IMPLEMENT_QUAD.md Phase 3/4 — quads mirror the triangle blocks
    # above (own section type, own element list) rather than sharing them,
    # matching how the model itself keeps QuadSection/QuadElement separate
    # from TriSection/TriElement.
    qsecs = list((getattr(struc, 'quad_sections', {}) or {}).values())
    if qsecs:
        section('Quad sections', len(qsecs))
        for qs in qsecs:
            add(_call('add_quad_section', qs.name, qs.material_name,
                      thickness=qs.thickness, plane_strain=qs.plane_strain,
                      formulation=getattr(qs, 'formulation', 'MITC4'),
                      rc_cover=getattr(qs, 'rc_cover', 0.045),
                      rc_alpha_s=getattr(qs, 'rc_alpha_s', 90.0),
                      rc_bar_phi=getattr(qs, 'rc_bar_phi', 16.0),
                      rc_shear_min=getattr(qs, 'rc_shear_min', True),
                      rc_cover_top_x=getattr(qs, 'rc_cover_top_x', None),
                      rc_cover_top_y=getattr(qs, 'rc_cover_top_y', None),
                      rc_cover_bot_x=getattr(qs, 'rc_cover_bot_x', None),
                      rc_cover_bot_y=getattr(qs, 'rc_cover_bot_y', None)))

    # ── Nodes ────────────────────────────────────────────────────
    nodes = list((getattr(struc, 'nodes', {}) or {}).values())
    if nodes:
        section('Nodes', len(nodes))
        add('    # Written before the geometry objects on purpose: an object')
        add('    # reuses a node that already sits at its defining point, so')
        add('    # this order is what preserves the original node ids.')
        for n in nodes:
            add(_call('add_node', n.id, n.x, n.y))

    # ── Geometry objects ─────────────────────────────────────────
    objs = list((getattr(struc, 'geometry_objects', {}) or {}).values())
    if objs:
        section('Geometry objects', len(objs))
        kinds = sorted({type(o).__name__ for o in objs})
        add(f"    from xdfem2d.models import {', '.join(kinds)}")
        add('')
        for o in objs:
            for line in _geo_lines(o):
                add(line)

    # ── Elements ─────────────────────────────────────────────────
    bars = list(getattr(struc, 'bar_elements', []) or [])
    explicit = [b for b in bars if not _from_object(b, objs)]
    if explicit:
        section('Bar elements', len(explicit))
        for b in explicit:
            _stage = getattr(b, 'stage', 1) or 1
            _kw: dict = dict(hinge_i=b.hinge_i, hinge_j=b.hinge_j,
                             rc_design=b.rc_design, rc_cover=b.rc_cover,
                             rc_cotg_theta=b.rc_cotg_theta,
                             rc_alpha_s=b.rc_alpha_s,
                             sd_ky=getattr(b, 'sd_ky', None),
                             sd_kz=getattr(b, 'sd_kz', None),
                             sd_klt=getattr(b, 'sd_klt', None),
                             sd_ltb=getattr(b, 'sd_ltb', True),
                             is_column=getattr(b, 'is_column', None))
            if getattr(b, 'beam', None) is not None:
                _kw['beam'] = b.beam
            if _stage != 1:
                _kw['stage'] = _stage
            add(_call('add_bar_element', b.id, b.node_i, b.node_j,
                      b.section_name, **_kw))

    _beams = getattr(struc, 'beams', None) or {}
    if _beams:
        section('Beams (beam-bars design)', len(_beams))
        add(f"    model.beams = {_lit({t: {'name': b.get('name', ''), 'overrides': dict(b.get('overrides') or {})} for t, b in _beams.items()})}")

    _bd = getattr(struc, 'beam_detail', None)
    if _bd is not None and _bd.beams:
        section('Beam bars detail (manual decisions)', len(_bd.beams))
        add('    from xdfem2d.detailing import DetailDocument')
        add(f"    model.beam_detail = DetailDocument.from_dict({_lit(_bd.to_dict())})")

    tris = list(getattr(struc, 'tri_elements', []) or [])
    if tris:
        section('Triangle elements', len(tris))
        for t in tris:
            add(_call('add_tri_element', t.id, t.node_i, t.node_j, t.node_k,
                      t.section_name))

    quads = list(getattr(struc, 'quad_elements', []) or [])
    if quads:
        section('Quad elements', len(quads))
        for q in quads:
            add(_call('add_quad_element', q.id, q.node_i, q.node_j, q.node_k,
                      q.node_l, q.section_name))

    # ── Supports and springs ─────────────────────────────────────
    sups = list((getattr(struc, 'supports', {}) or {}).values())
    assigns = list(getattr(struc, 'support_assignments', []) or [])
    if sups or assigns:
        section('Supports', len(sups) + len(assigns))
        for sp in sups:
            add(_call('add_support', sp.name, ux=sp.ux, uy=sp.uy, tz=sp.tz))
        for a in assigns:
            add(_call('assign_support', a.node_id, a.support_name))

    nsp = list((getattr(struc, 'node_springs', {}) or {}).values())
    esp = list((getattr(struc, 'element_springs', {}) or {}).values())
    if nsp or esp:
        section('Springs', len(nsp) + len(esp))
        for sp in nsp:
            add(_call('add_node_spring', sp.node_id, kx=sp.kx, ky=sp.ky,
                      kt=sp.kt, mode_x=sp.mode_x, mode_y=sp.mode_y))
        for es in esp:
            add(_call('add_element_spring', es.element_id, kx=es.kx, ky=es.ky,
                      coord_sys=es.coord_sys, mode_x=es.mode_x,
                      mode_y=es.mode_y))

    cons = list((getattr(struc, 'constraints', {}) or {}).values())
    if cons:
        section('Constraints', len(cons))
        for c in cons:
            if c.kind == 'rigid_link':
                line = _call('add_rigid_link', c.master, list(c.slaves), id=c.id)
            elif c.kind == 'equal_dof':
                line = _call('add_equal_dof', list(c.nodes),
                             list(c.components), id=c.id)
            else:
                terms = [(t[0], t[1], t[2]) for t in c.terms]
                line = _call('add_constraint_equation', terms, c.value, id=c.id)
            add(line)
            if not c.enabled:
                # add_* returns the Constraint; disable it explicitly so a
                # round-tripped script reproduces a disabled constraint.
                add(f"    model.constraints[{_lit(c.id)}].enabled = False")

    # ── Load cases and loads ─────────────────────────────────────
    cases = list(getattr(struc, 'load_cases', []) or [])
    if cases:
        section('Load cases', len(cases))
        add('    # create_analysis_case=False throughout: the analysis cases')
        add('    # are written out below exactly as the model holds them, and')
        add('    # letting these create their own would duplicate them.')
        for lc in cases:
            add(_call('add_load_case', lc.id,
                      self_weight_factor=lc.self_weight_factor,
                      action_type=_enum(getattr(lc, 'action_type', None)),
                      create_analysis_case=False,
                      load_duration=getattr(lc, 'load_duration', '') or None,
                      category=getattr(lc, 'category', '') or '',
                      psi0=getattr(lc, 'psi0', None),
                      psi1=getattr(lc, 'psi1', None),
                      psi2=getattr(lc, 'psi2', None)))

    for title, attr, method, fields in (
            ('Point loads', 'point_loads', 'add_point_load',
             ('node_id', 'load_case_id', 'fx', 'fy', 'mz')),
            ('Distributed loads', 'distributed_loads', 'add_distributed_load',
             ('element_id', 'load_case_id', 'fxe', 'fxd', 'fye', 'fyd',
              'coord_sys')),
            ('Element point loads', 'element_point_loads',
             'add_element_point_load',
             ('element_id', 'load_case_id', 'a', 'fx', 'fy', 'mz',
              'coord_sys')),
            ('Support settlements', 'support_settlements',
             'add_support_settlement',
             ('node_id', 'load_case_id', 'ux', 'uy', 'tz')),
            ('Temperature loads', 'temperature_loads', 'add_temperature_load',
             # No 'alpha': add_temperature_load takes one, TemperatureLoad
             # does not store it. Writing a field the object does not have
             # raised AttributeError on the one example that uses thermal
             # loads, which is how this list stopped being copied from the
             # signature and started being checked against the dataclass.
             ('element_id', 'load_case_id', 'delta_t_uniform',
              'delta_t_gradient')),
            ('Triangle temperature loads', 'tri_temperature_loads',
             'add_tri_temperature_load',
             ('tri_id', 'load_case_id', 'dt_i', 'dt_j', 'dt_k', 'dt_gradient')),
            ('Quad temperature loads', 'quad_temperature_loads',
             'add_quad_temperature_load',
             ('quad_id', 'load_case_id', 'dt_i', 'dt_j', 'dt_k', 'dt_l',
              'dt_gradient')),
            ('Area temperature loads', 'area_temperature_loads',
             'add_area_temperature_load',
             ('object_id', 'load_case_id', 'dt_uniform', 'dt_gradient',
              'field_name', 'grad_field_name')),
            ('Line temperature loads', 'line_temperature_loads',
             'add_line_temperature_load',
             ('object_id', 'load_case_id', 'dt_uniform', 'dt_gradient',
              'field_name', 'grad_field_name')),
            ('Line distributed loads', 'line_distributed_loads',
             'add_line_distributed_load',
             ('object_id', 'load_case_id', 'fx', 'fy', 'fx_field', 'fy_field',
              'coord_sys')),
            ('Line element springs', 'line_element_springs',
             'add_line_element_spring',
             ('object_id', 'kx', 'ky', 'kx_field', 'ky_field', 'coord_sys',
              'mode_x', 'mode_y')),
            ('Triangle edge loads', 'tri_edge_loads', 'add_tri_edge_load',
             ('id', 'tri_id', 'node_a', 'node_b', 'load_case_id', 'fx', 'fy',
              'coord_sys', 'pn', 'pt')),
            ('Quad edge loads', 'quad_edge_loads', 'add_quad_edge_load',
             ('id', 'quad_id', 'node_a', 'node_b', 'load_case_id', 'fx', 'fy',
              'coord_sys', 'pn', 'pt')),
            ('Surface edge loads', 'surface_edge_loads',
             'add_surface_edge_load',
             ('id', 'object_id', 'node_a', 'node_b', 'load_case_id', 'fx',
              'fy', 'coord_sys', 'pn', 'pt')),
    ):
        rows = list(getattr(struc, attr, []) or [])
        if not rows:
            continue
        section(title, len(rows))
        positional = _positional_names(method)
        for r in rows:
            have = [f for f in fields if hasattr(r, f)]
            args = [getattr(r, f) for f in have if f in positional]
            kw = {f: getattr(r, f) for f in have if f not in positional}
            add(_call(method, *args, **kw))

    # ── Area loads (plate domain) — add_area_load dispatches on the target,
    # so the surface, triangle and quad rows all write through the same
    # call (dev/IMPLEMENT_QUAD.md Phase 4: add_area_load/remove_area_load
    # already accept a quad element id, same dispatcher). ──
    areas = (list(getattr(struc, 'surface_area_loads', []) or [])
             + list(getattr(struc, 'tri_area_loads', []) or [])
             + list(getattr(struc, 'quad_area_loads', []) or []))
    if areas:
        section('Area loads', len(areas))
        for a in areas:
            target = (getattr(a, 'object_id', None)
                      or getattr(a, 'tri_id', None)
                      or getattr(a, 'quad_id', ''))
            add(_call('add_area_load', target, a.load_case_id, pz=a.pz))

    # ── Area springs (Winkler, plate domain) — add_area_spring dispatches on
    # the target, so surface, triangle and quad rows (dev/IMPLEMENT_QUAD.md
    # Phase 6 follow-up: QuadAreaSpring) write through the same call. ──
    area_springs = (list(getattr(struc, 'surface_area_springs', []) or [])
                    + list(getattr(struc, 'tri_area_springs', []) or [])
                    + list(getattr(struc, 'quad_area_springs', []) or []))
    if area_springs:
        section('Area springs', len(area_springs))
        for a in area_springs:
            target = (getattr(a, 'object_id', None)
                      or getattr(a, 'tri_id', None)
                      or getattr(a, 'quad_id', ''))
            add(_call('add_area_spring', target, kz=a.kz))

    # ── Punching columns (plate domain) ──────────────────────────
    columns = list(getattr(struc, 'punch_columns', []) or [])
    if columns:
        section('Punching columns', len(columns))
        for c in columns:
            add(_call('add_punch_column', c.id, c.node_id, shape=c.shape,
                      bx=c.bx, by=c.by, position=c.position, force=c.force,
                      dx=c.dx, dy=c.dy, beta_min=getattr(c, 'beta_min', 1.0)))

    # ── Dynamics ─────────────────────────────────────────────────
    masses = list(getattr(struc, 'nodal_masses', []) or [])
    if masses:
        section('Nodal masses', len(masses))
        for m in masses:
            add(_call('add_nodal_mass', m.node_id, m.mass_case_id,
                      mx=m.mx, my=m.my, mtz=m.mtz))

    spectra = list((getattr(struc, 'spectral_functions', {}) or {}).values())
    if spectra:
        section('Response spectra', len(spectra))
        for sf in spectra:
            add(_call('add_spectral_function', sf.id,
                      description=getattr(sf, 'description', ''),
                      damping=getattr(sf, 'damping', 0.05),
                      points=[list(p) for p in (sf.points or [])]))

    # ── Analysis cases and combinations ──────────────────────────
    acs = list(getattr(struc, 'analysis_cases', []) or [])
    seqs_map = dict(getattr(struc, 'construction_sequences', {}) or {})
    # Build a helper that emits one construction sequence inline, preserving
    # position in the analysis_cases list (Sequence ACs are created implicitly
    # by add_construction_sequence, so interleaving keeps the order intact).
    _seq_imports_emitted = [False]

    def _emit_seq(seq_id: str, solved: bool = False) -> None:
        seq = seqs_map.get(seq_id)
        if seq is None:
            return
        if not _seq_imports_emitted[0]:
            add('    from xdfem2d.models import (ConstructionSequence,'
                ' ConstructionPhase, Operation, OpAction, OpTarget)')
            add('    from xdfem2d.phasing import resolve_phase_state as _rps')
            _seq_imports_emitted[0] = True
        add('')
        ph_lines = []
        for ph in seq.phases:
            op_lines = []
            for op in (ph.operations or []):
                op_kw: dict = {}
                if op.action.value != 'add':
                    op_kw['action'] = f'OpAction.{op.action.name}'
                if op.target.value != 'elements':
                    op_kw['target'] = f'OpTarget.{op.target.name}'
                if op.group_kind:
                    op_kw['group_kind'] = op.group_kind
                if op.ids:
                    op_kw['ids'] = tuple(op.ids)
                if op.factor != 1.0:
                    op_kw['factor'] = op.factor
                if op.note:
                    op_kw['note'] = op.note
                parts = [_lit(op.id)]
                for k, v in op_kw.items():
                    if k in ('action', 'target'):
                        parts.append(f'{k}={v}')
                    else:
                        parts.append(f'{k}={_lit(v)}')
                op_lines.append(f'Operation({", ".join(parts)})')
            ph_kw: dict = {}
            if ph.inherits_from is not None:
                ph_kw['inherits_from'] = ph.inherits_from
            if ph.support_set_id is not None:
                ph_kw['support_set_id'] = ph.support_set_id
            if ph.reapply_self_weight:
                ph_kw['reapply_self_weight'] = True
            if not ph.release_supports:
                ph_kw['release_supports'] = False
            if ph.time:
                ph_kw['time'] = ph.time
            ops_repr = f'[{", ".join(op_lines)}]' if op_lines else '[]'
            kw_str = ''.join(f', {k}={_lit(v)}' for k, v in ph_kw.items())
            ph_lines.append(
                f'ConstructionPhase({_lit(ph.id)}, operations={ops_repr}{kw_str})')
        phases_repr = ('[' + ',\n            '.join(ph_lines) + ']'
                       if ph_lines else '[]')
        dm = seq.displacement_method
        dm_kw = f', displacement_method={_lit(dm)}' if dm != 'increment' else ''
        add(f'    _seq = ConstructionSequence('
            f'{_lit(seq.id)}, phases={phases_repr}{dm_kw})')
        add(f'    model.add_construction_sequence(_seq)')
        # Resolve the cached state (active_elements, applied_cases,
        # case_factors) for each phase so the model matches a round-trip
        # through the x2d file format (which stores the resolved state).
        add(f'    _all = ({{e.id for e in model.bar_elements}}'
            f' | {{e.id for e in model.tri_elements}}'
            f' | {{e.id for e in getattr(model, "quad_elements", [])}}'
            f' | {{o.id for o in getattr(model, "geometry_objects", [])}})')
        add(f'    _prev_act, _prev_ss = None, None')
        add(f'    for _ph in _seq.phases:')
        add(f'        _act, _ss, _cases, _cf = _rps('
            f'_prev_act, _prev_ss, _ph.operations, _all, struc=model)')
        add(f'        _ph.active_elements = _act')
        add(f'        _ph.support_set_id = _ss')
        add(f'        _ph.applied_cases = _cases')
        add(f'        _ph.case_factors = _cf')
        add(f'        _prev_act, _prev_ss = _act, _ss')
        if solved:
            add(f'    model.analysis_cases_by_id[{_lit(seq.id)}].solved = True')

    # Iterate analysis_cases in their original order.  Sequence-type ACs are
    # exported as add_construction_sequence (which creates the AC implicitly);
    # all others go through add_analysis_case.
    if acs:
        n_plain = sum(1 for ac in acs
                      if getattr(ac, 'analysis_type', 'Linear') != 'Sequence')
        n_seq = len(seqs_map)
        if n_plain:
            section('Analysis cases', n_plain)
        if seqs_map and not n_plain:
            section('Construction sequences', n_seq)
        elif seqs_map:
            # mixed — label the first sequence block when we encounter it
            _seq_section_emitted = [False]
        _emitted_seqs: set = set()
        for ac in acs:
            if getattr(ac, 'analysis_type', 'Linear') == 'Sequence':
                sid = getattr(ac, 'sequence_id', ac.id) or ac.id
                if sid not in _emitted_seqs:
                    if seqs_map and n_plain and not _seq_imports_emitted[0]:
                        add('')
                        section('Construction sequences', n_seq)
                    _emit_seq(sid, solved=bool(getattr(ac, 'solved', False)))
                    _emitted_seqs.add(sid)
            else:
                # Every field that differs from its default, not a chosen few.
                extra = {k: v for k, v in _non_default_fields(ac).items()
                         if k not in ('id', 'analysis_type', 'coefficients')}
                add(_call('add_analysis_case', ac.id,
                          _enum(ac.analysis_type, 'Linear'),
                          dict(getattr(ac, 'coefficients', {}) or {}), **extra))
        # Emit any sequences that had no matching AC in the list (shouldn't
        # normally happen, but defensive).
        for sid in seqs_map:
            if sid not in _emitted_seqs:
                _emit_seq(sid)

    combos = list(getattr(struc, 'load_combinations', []) or [])
    if combos:
        section('Load combinations', len(combos))
        for c in combos:
            add(_call('add_load_combination', c.id,
                      dict(getattr(c, 'coefficients', {}) or {}),
                      combo_type=_enum(getattr(c, 'combo_type', None),
                                       'LinearSum')))

    # ── Variants and variant combinations ────────────────────────
    variants = list((getattr(struc, 'variants', {}) or {}).values())
    vc_map = dict(getattr(struc, 'variant_combinations', {}) or {})
    if variants or vc_map:
        # Variants with own model need a helper function to build that model.
        # Generate them BEFORE the build() body — collect lines separately.
        vm_helpers: list[str] = []

        def _safe_id(vid: str) -> str:
            import re
            return re.sub(r'\W', '_', vid)

        for v in variants:
            vm = getattr(v, 'model', None)
            if vm is None:
                continue
            fn = f'_vm_{_safe_id(v.id)}'
            sub = to_python(vm)
            # Extract the body of build() from the sub-script (between the
            # first 'model = Structure2D' line and '    return model').
            lines_sub = sub.splitlines()
            try:
                start = next(i for i, l in enumerate(lines_sub)
                             if 'model = Structure2D' in l)
                end = next(i for i, l in enumerate(lines_sub)
                           if l.strip() == 'return model')
            except StopIteration:  # pragma: no cover
                continue
            body_lines = lines_sub[start:end + 1]
            vm_helpers.append(f'\ndef {fn}() -> Structure2D:')
            vm_helpers.append('    """Own action model for this variant."""')
            vm_helpers.extend(body_lines)

        # Splice helper functions BEFORE the main build() definition.
        # We do this by appending a special marker that the caller post-processes,
        # OR — simpler — we insert them into L right now before the build() header.
        # Find the 'def build()' line in L and insert before it.
        if vm_helpers:
            build_idx = next((i for i, l in enumerate(L)
                              if l.startswith('def build()')), len(L))
            for j, helper_line in enumerate(vm_helpers):
                L.insert(build_idx + j, helper_line)

        if variants:
            section('Variants', len(variants))
            add('    from xdfem2d.models import Variant')
            for v in variants:
                kw: dict = {}
                if getattr(v, 'description', ''):
                    kw['description'] = v.description
                ae = v.active_elements
                if ae is not None:
                    kw['active_elements'] = set(ae)
                if v.support_set_id:
                    kw['support_set_id'] = v.support_set_id
                vm = getattr(v, 'model', None)
                kw_str = ', '.join(f'{k}={_lit(v2)}' for k, v2 in kw.items())
                sep = ', ' if kw_str else ''
                add(f'    _v = Variant({_lit(v.id)}{sep}{kw_str})')
                if vm is not None:
                    fn = f'_vm_{_safe_id(v.id)}'
                    add(f'    _v.model = {fn}()')
                add(f'    model.add_variant(_v)')

        if vc_map:
            section('Variant combinations', len(vc_map))
            for cid, d in vc_map.items():
                op = d.get('op', 'Envelope')
                terms = [list(t) for t in d.get('terms', [])]
                add(f'    model.variant_combinations[{_lit(cid)}] = '
                    f'{{"op": {_lit(op)}, "terms": {_lit(terms)}}}')

    add('')
    add('    return model')
    add('')
    add('')
    add("if __name__ == '__main__':")
    add('    from pathlib import Path')
    add('    from xdfem2d.file_io import save_x2d')
    add('')
    add('    # ── Build from script and save ────────────────────────────────')
    add('    # Builds the model defined above, writes it beside this script.')
    add('    # Open the result in xdfem2D with File > Open.')
    add('    model = build()')
    add("    out = Path(__file__).with_suffix('.x2d')")
    add('    save_x2d(model, None, out)')
    add("    print('saved:', out)")
    add('')
    add('    # ── Load an existing .x2d instead of building ─────────────────')
    add('    # Uncomment to open any saved file and work from it.')
    add("    # model, results, meta = load_x2d('path/to/model.x2d')")
    add('')
    add('    # ── Run the analysis and keep results in the saved file ───────')
    add('    #   results = model.calculate()')
    add('    #   save_x2d(model, results, out)')
    add('')
    add('    # ── Print a summary after calculating ─────────────────────────')
    add('    #   cases  = sorted(results.get("analysis_cases", {}))')
    add('    #   combos = sorted(results.get("combinations", {}))')
    add('    #   print(f"{len(model.nodes)} nodes, "')
    add('    #         f"{len(model.bar_elements)} bars, "')
    add('    #         f"{len(model.tri_elements)} triangles")')
    add('    #   print("analysis cases:", ", ".join(cases) or "none")')
    add('    #   print("combinations:  ", ", ".join(combos) or "none")')
    add('')
    return "\n".join(L)


def _non_default_fields(obj) -> dict:
    """Fields of a dataclass instance that differ from their declared default.

    Read from the class rather than guessed, so a field whose default changes
    upstream changes what the exporter writes, in the same direction.
    """
    import dataclasses
    if not dataclasses.is_dataclass(obj):             # pragma: no cover
        return {}
    out = {}
    for f in dataclasses.fields(obj):
        default = f.default
        if default is dataclasses.MISSING:
            if f.default_factory is dataclasses.MISSING:   # type: ignore[misc]
                continue
            default = f.default_factory()             # type: ignore[misc]
        value = getattr(obj, f.name, None)
        if not _same(value, default):
            out[f.name] = _enum(value)
    return out


def _positional_names(method: str) -> set:
    """Parameters with no default — those must be passed positionally."""
    try:
        sig = inspect.signature(getattr(Structure2D, method))
    except (AttributeError, TypeError, ValueError):   # pragma: no cover
        return set()
    return {n for n, p in sig.parameters.items()
            if n != 'self' and p.default is inspect.Parameter.empty}


def _from_object(bar, objs) -> bool:
    """Whether a bar was generated by expanding a geometry object.

    Those must not be written: the object writes them at solve time, and a
    script that adds both ends up with the structure twice.
    """
    for o in objs:
        if bar.id.startswith(f"{o.id}."):
            return True
    return False


def _geo_lines(o) -> list[str]:
    """One geometry object, by the nodes it is defined by.

    Not through ``add_geo_line(x1, y1, ...)``, which is the API a person would
    type. Those constructors take coordinates and then look for a node already
    sitting at each point — and in a model with two nodes at the same place
    they find whichever comes first. Exporting the frame-and-wall example that
    way rebuilt a rectangle on 'WB_13' where the original stood on 'Rect1.p1':
    the same geometry, different ids, and every result keyed by them.

    So the object is built the way the .x2d loader builds it, from the node ids
    themselves. It reaches past the public API into ``geometry_objects``, which
    is a real cost and the reason for this paragraph — but the alternative is a
    script that silently rebuilds a model onto different nodes.
    """
    cls = type(o).__name__
    args = [f"id={_lit(o.id)}"]
    if cls in ('GeoRectangle', 'GeoPolygon'):
        args.append(f"tri_section_name={_lit(getattr(o, 'tri_section_name', ''))}")
        args.append(f"target_size={_lit(getattr(o, 'target_size', 0.5))}")
    else:
        args.append(f"section_name={_lit(o.section_name)}")
        args.append(f"divisions={_lit(o.divisions)}")
        args.append(f"max_chord={_lit(o.max_chord)}")
        if cls == 'GeoMultisegment':
            args.append(f"closed={_lit(o.closed)}")
    out = [f"    obj = {cls}({', '.join(args)})",
           f"    obj.node_ids = {_lit(list(getattr(o, 'node_ids', []) or []))}"]
    if cls in ('GeoRectangle', 'GeoPolygon'):
        # Non-default edge propagation settings — omitted when they still
        # hold the dataclass default, to keep the common case's script
        # short. edge_supports (explicit per-edge restraint) always needs
        # writing when set: there is no way to reconstruct it from the mesh
        # alone, unlike edge_support_mode's own "common" default.
        if getattr(o, 'edge_support_mode', 'common') != 'common':
            out.append(f"    obj.edge_support_mode = "
                       f"{_lit(o.edge_support_mode)}")
        if getattr(o, 'edge_spring_mode', 'linear') != 'linear':
            out.append(f"    obj.edge_spring_mode = "
                       f"{_lit(o.edge_spring_mode)}")
        if getattr(o, 'edge_supports', None):
            out.append(f"    obj.edge_supports = {_lit(list(o.edge_supports))}")
    loads = [dict(x) for x in (getattr(o, 'loads', []) or [])]
    if loads:
        out.append(f"    obj.loads = {_lit(loads)}")
    if getattr(o, 'intersect_crossings', False):
        out.append("    obj.intersect_crossings = True")
    if cls in ('GeoSegment', 'GeoArc', 'GeoMultisegment'):
        if getattr(o, 'sd_ky', None) is not None:
            out.append(f"    obj.sd_ky = {_lit(o.sd_ky)}")
        if getattr(o, 'sd_kz', None) is not None:
            out.append(f"    obj.sd_kz = {_lit(o.sd_kz)}")
        if getattr(o, 'sd_klt', None) is not None:
            out.append(f"    obj.sd_klt = {_lit(o.sd_klt)}")
        if not getattr(o, 'sd_ltb', True):
            out.append(f"    obj.sd_ltb = False")
        if getattr(o, 'is_column', None) is not None:
            out.append(f"    obj.is_column = {_lit(o.is_column)}")
        if getattr(o, 'beam', None) is not None:
            out.append(f"    obj.beam = {_lit(o.beam)}")
    out.append("    model.geometry_objects[obj.id] = obj")
    return out
