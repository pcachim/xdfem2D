"""
Validating loader for the JSON structure format.

``load_structure_json`` (in ``structure_io``) is strict: the first bad key or
the first JSON syntax error stops the whole load with an exception. That is the
right behaviour for programmatic use, but a person opening a hand-edited or
third-party file wants the opposite — read the file to the end, build whatever
is valid, and be told *every* problem at once instead of fixing them one
exception at a time.

This module provides that second mode:

  struc, report = load_structure_json_checked(path)

``struc`` is a best-effort :class:`Structure2D` (valid entries loaded, bad ones
skipped). ``report`` is a :class:`LoadReport` describing what was read and every
issue found — syntax errors recovered during parsing and semantic errors met
while building. Nothing here raises for a malformed file; a report is always
returned. It never mutates the strict loader, so existing callers are unchanged.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .structure import Structure2D


# ============================================================
# Report
# ============================================================

@dataclass
class LoadIssue:
    """One problem found while reading a file.

    ``section`` is the JSON key it was found under ('nodes', 'sections', …),
    ``index`` the position in that list (None for file-level problems), ``ident``
    the entry's own id/name when it has one, and ``severity`` one of
    'syntax' | 'error' | 'warning'."""
    severity: str
    section: str
    message: str
    index: int | None = None
    ident: str | None = None
    line: int | None = None
    column: int | None = None

    def location(self) -> str:
        if self.severity == 'syntax':
            return f"line {self.line}, column {self.column}"
        parts = [self.section]
        if self.index is not None:
            parts.append(f"#{self.index}")
        if self.ident:
            parts.append(f"'{self.ident}'")
        return " ".join(parts)


@dataclass
class LoadReport:
    """What a checked load read and everything it found wrong."""
    path: str = ""
    parsed: bool = False           # did the JSON become a dict at all
    counts: dict = field(default_factory=dict)   # section -> entries loaded
    issues: list = field(default_factory=list)   # LoadIssue, in reading order

    # -- recording --------------------------------------------------------
    def add(self, severity, section, message, *, index=None, ident=None,
            line=None, column=None):
        self.issues.append(LoadIssue(severity, section, message, index=index,
                                     ident=ident, line=line, column=column))

    def loaded(self, section, n):
        if n:
            self.counts[section] = n

    # -- querying ---------------------------------------------------------
    @property
    def syntax_errors(self):
        return [i for i in self.issues if i.severity == 'syntax']

    @property
    def errors(self):
        return [i for i in self.issues if i.severity == 'error']

    @property
    def warnings(self):
        return [i for i in self.issues if i.severity == 'warning']

    @property
    def ok(self) -> bool:
        return self.parsed and not self.syntax_errors and not self.errors


# ============================================================
# Public entry point
# ============================================================

def load_structure_json_checked(path: str | Path) -> tuple[Structure2D, LoadReport]:
    """Load a JSON structure, collecting every problem instead of raising.

    Returns ``(struc, report)``. ``struc`` is always a real Structure2D — empty
    if the file could not be parsed at all, otherwise carrying every entry that
    loaded cleanly. Read ``report`` to see what was skipped and why."""
    report = LoadReport(path=str(path))
    try:
        text = Path(path).read_text(encoding='utf-8')
    except OSError as e:
        report.add('error', 'file', f"could not read the file: {e}")
        return Structure2D(), report

    data = _parse_with_recovery(text, report)
    return build_checked(data, report=report)


def build_checked(data, *, report: LoadReport | None = None
                  ) -> tuple[Structure2D, LoadReport]:
    """Best-effort Structure2D from an already-parsed model, collecting every
    problem instead of raising.

    The dict-level counterpart to :func:`load_structure_json_checked`: it takes
    the JSON already parsed (so it carries no syntax recovery of its own) and
    does the resilient build. This is the seam that lets a model lifted out of
    an assistant reply be built with the same tolerance as a file opened from
    disk, without going through a path.

    Returns ``(struc, report)``. ``struc`` is always a real Structure2D — empty
    if *data* is not a JSON object. A caller that already holds a report (the
    file loader, which recorded syntax errors while parsing) passes it in so
    those are preserved; otherwise a fresh report is made."""
    if report is None:
        report = LoadReport()
    if not isinstance(data, dict):
        if data is not None:
            report.add('error', 'file',
                       "top level of the file is not a JSON object")
        return Structure2D(), report

    report.parsed = True
    struc = _build_resilient(data, report)
    _fold_model_checks(struc, data, report)
    return struc, report


def _fold_model_checks(struc: Structure2D, data: dict, report: LoadReport):
    """Add dangling-reference and completeness findings to the report, as
    warnings — the single place both entry points get them.

    Run only on a cleanly built model: if the build already recorded errors the
    structure is half-formed, and references into rows that were skipped would be
    noise on top of the real problem. Warnings, not errors, so they never change
    ``report.ok`` — a model that opens but floats, or has a load pointing at a
    case that does not exist, still opens; the reader is simply told.

    References come from :meth:`Structure2D.reference_problems` (the canonical
    check, on what was actually loaded); completeness from
    :func:`model_json.completeness_pieces` (presence only, so the two do not
    overlap). Both are non-raising, but guarded so a check can never turn a
    successful load into a failed one."""
    if report.errors or report.syntax_errors:
        return
    try:
        for msg in struc.reference_problems():
            report.add('warning', 'reference', msg)
    except Exception:                                    # noqa: BLE001
        pass
    try:
        from .model_json import completeness_pieces
        for msg in completeness_pieces(data):
            report.add('warning', 'completeness', msg)
    except Exception:                                    # noqa: BLE001
        pass


# ============================================================
# Phase 1 — tolerant JSON parsing
# ============================================================

def _parse_with_recovery(text: str, report: LoadReport, max_fixes: int = 60):
    """Parse JSON, recording each syntax error and trying to read past it.

    Delegates to the shared tolerant parser (:func:`model_json.parse_tolerant`),
    the single implementation used by both entry points — a file opened here and
    a model lifted out of an assistant reply. A callback records every syntax
    error into *report* so more than one can be surfaced, and the reader is told
    the file needs fixing at the source. Returns the parsed object, or None when
    the text cannot be recovered within *max_fixes*."""
    from .model_json import parse_tolerant

    def _record(msg, line, column):
        report.add('syntax', 'json', msg, line=line, column=column)

    try:
        return parse_tolerant(text, on_syntax_error=_record, max_fixes=max_fixes)
    except (json.JSONDecodeError, ValueError):
        return None


# ============================================================
# Phase 2 — resilient build
# ============================================================

def _ident_of(item):
    if not isinstance(item, dict):
        return None
    for k in ('id', 'name', 'node_id', 'element_id', 'tri_id', 'object_id'):
        if item.get(k):
            return str(item[k])
    return None


def _each(data, key, fn, report):
    """Run *fn* over every entry of ``data[key]``, guarding each one.

    A bad entry is recorded and skipped; the loop always reaches the end. The
    number that loaded cleanly is recorded so the report can say what was read."""
    items = data.get(key, [])
    if items in (None, [], {}):
        return
    if not isinstance(items, list):
        report.add('error', key,
                   f"expected a list, got {type(items).__name__}")
        return
    ok = 0
    for i, item in enumerate(items):
        try:
            fn(item)
            ok += 1
        except KeyError as ex:
            report.add('error', key, f"missing required field {ex}",
                       index=i, ident=_ident_of(item))
        except Exception as ex:                              # noqa: BLE001
            report.add('error', key, str(ex),
                       index=i, ident=_ident_of(item))
    report.loaded(key, ok)


def _section(name, fn, report):
    """Guard a whole block that is not a simple per-item list."""
    try:
        fn()
    except Exception as ex:                                  # noqa: BLE001
        report.add('error', name, str(ex))


def _build_resilient(data: dict, report: LoadReport) -> Structure2D:
    """Best-effort rebuild of a Structure2D, collecting every semantic error.

    Mirrors ``structure_io._from_dict`` section by section, but each entry is
    wrapped so a single bad row cannot abort the load. Order matches the strict
    loader so cross-references (sections need materials, elements need sections,
    combinations need analysis cases) still resolve."""
    # Domain first (absent → 'plane'); an unknown value is reported and the
    # load continues as 'plane' rather than aborting on the first key.
    try:
        s = Structure2D(domain=data.get('domain', 'plane'))
    except ValueError as e:
        report.add('error', 'domain', str(e))
        s = Structure2D()

    def _project_info():
        pinfo = data.get('project_info')
        if isinstance(pinfo, dict):
            merged = dict(s.project_info)
            merged.update({str(k): str(v) for k, v in pinfo.items()})
            s.project_info = merged
    _section('project_info', _project_info, report)

    s.propagate_edge_supports = bool(data.get('propagate_edge_supports', True))
    s.propagate_edge_springs = bool(data.get('propagate_edge_springs', True))

    _each(data, 'nodes',
          lambda n: s.add_node(n['id'], n['x'], n['y']), report)
    _each(data, 'materials',
          lambda m: s.add_material(
              m['name'], m['elastic_modulus'], m['unit_weight'],
              alpha=m.get('alpha', 1.0e-5), unit_mass=m.get('unit_mass', None),
              material_type=m.get('material_type', 'Concrete'),
              poisson=m.get('poisson', 0.2), design=m.get('design', {})), report)
    _each(data, 'sections',
          lambda sc: s.add_section(
              sc['name'], sc['material_name'], sc['b'], sc['h'],
              area_override=sc.get('area_override'),
              inertia_override=sc.get('inertia_override'),
              torsion_override=sc.get('torsion_override'),
              profile_name=sc.get('profile_name'),
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
              beam_overrides=sc.get('beam_overrides')), report)

    def _bar(e):
        elem = s.add_bar_element(e['id'], e['node_i'], e['node_j'],
                                 e['section_name'],
                                 hinge_i=e.get('hinge_i', False),
                                 hinge_j=e.get('hinge_j', False))
        elem.rc_design = e.get('rc_design', False)
        elem.rc_cover = e.get('rc_cover', 0.0)
        elem.rc_cotg_theta = e.get('rc_cotg_theta', 1.0)
        elem.rc_alpha_s = e.get('rc_alpha_s', 90.0)
        # sd_ky/kz/klt/ltb were written by save_structure_json but never read
        # back here (structure_io_checked.py had fallen out of sync with
        # structure_io.py's load_structure_json) — see dev/
        # BUCKLING_COLUMN_PERSISTENCE.md §2.
        elem.sd_ky = e.get('sd_ky')
        elem.sd_kz = e.get('sd_kz')
        elem.sd_klt = e.get('sd_klt')
        elem.sd_ltb = e.get('sd_ltb', True)
        elem.is_column = e.get('is_column')
        elem.beam = e.get('beam')
        elem.stage = int(e.get('stage', 1) or 1)
    _each(data, 'bar_elements', _bar, report)

    _each(data, 'tri_sections',
          lambda ts: s.add_tri_section(
              ts['name'], ts['material_name'],
              thickness=ts.get('thickness', 0.1),
              plane_strain=bool(ts.get('plane_strain', False)),
              formulation=ts.get('formulation', 'CST'),
              rc_cover=ts.get('rc_cover', 0.045),
              rc_alpha_s=ts.get('rc_alpha_s', 90.0),
              rc_bar_phi=ts.get('rc_bar_phi', 16.0),
              rc_shear_min=bool(ts.get('rc_shear_min', True))), report)

    def _tri_section_for(mat, thk, ps):
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

    def _tri(t):
        sec_name = t.get('section_name')
        if sec_name in (None, ''):
            # A legacy element carries material + thickness instead of a named
            # section; rebuild the section from those. But if it has neither, the
            # thing it is missing is 'section_name' — reporting 'material_name'
            # (which the loop below would raise) names the wrong field and tells
            # the reader nothing about what to fix.
            if 'material_name' in t:
                sec_name = _tri_section_for(
                    t['material_name'], float(t.get('thickness', 0.1)),
                    bool(t.get('plane_strain', False)))
            else:
                raise KeyError('section_name')
        s.add_tri_element(t['id'], t['node_i'], t['node_j'], t['node_k'],
                          sec_name)
    _each(data, 'tri_elements', _tri, report)

    _each(data, 'quad_sections',
          lambda qs: s.add_quad_section(
              qs['name'], qs['material_name'],
              thickness=qs.get('thickness', 0.1),
              plane_strain=bool(qs.get('plane_strain', False)),
              formulation=qs.get('formulation', 'MITC4'),
              rc_cover=qs.get('rc_cover', 0.045),
              rc_alpha_s=qs.get('rc_alpha_s', 90.0),
              rc_bar_phi=qs.get('rc_bar_phi', 16.0),
              rc_shear_min=bool(qs.get('rc_shear_min', True))), report)

    def _quad(q):
        s.add_quad_element(q['id'], q['node_i'], q['node_j'], q['node_k'],
                           q['node_l'], q['section_name'])
    _each(data, 'quad_elements', _quad, report)

    _each(data, 'tri_edge_loads',
          lambda e: s.add_tri_edge_load(
              e['id'], e['tri_id'], e['node_a'], e['node_b'], e['load_case_id'],
              fx=e.get('fx', 0.0), fy=e.get('fy', 0.0),
              coord_sys=e.get('coord_sys', 'global'),
              pn=e.get('pn', 0.0), pt=e.get('pt', 0.0)), report)
    _each(data, 'quad_edge_loads',
          lambda e: s.add_quad_edge_load(
              e['id'], e['quad_id'], e['node_a'], e['node_b'], e['load_case_id'],
              fx=e.get('fx', 0.0), fy=e.get('fy', 0.0),
              coord_sys=e.get('coord_sys', 'global'),
              pn=e.get('pn', 0.0), pt=e.get('pt', 0.0)), report)
    _each(data, 'surface_edge_loads',
          lambda e: s.add_surface_edge_load(
              e['id'], e['object_id'], e['node_a'], e['node_b'],
              e['load_case_id'], fx=e.get('fx', 0.0), fy=e.get('fy', 0.0),
              coord_sys=e.get('coord_sys', 'global'),
              pn=e.get('pn', 0.0), pt=e.get('pt', 0.0)), report)
    from .models import TriAreaLoad, SurfaceAreaLoad
    _each(data, 'tri_area_loads',
          lambda a: s.tri_area_loads.append(TriAreaLoad(
              tri_id=a['tri_id'], load_case_id=a['load_case_id'],
              pz=a.get('pz', 0.0))), report)
    _each(data, 'surface_area_loads',
          lambda a: s.surface_area_loads.append(SurfaceAreaLoad(
              object_id=a['object_id'], load_case_id=a['load_case_id'],
              pz=a.get('pz', 0.0))), report)
    from .models import QuadAreaLoad
    _each(data, 'quad_area_loads',
          lambda a: s.quad_area_loads.append(QuadAreaLoad(
              quad_id=a['quad_id'], load_case_id=a['load_case_id'],
              pz=a.get('pz', 0.0))), report)
    from .models import TriAreaSpring, QuadAreaSpring, SurfaceAreaSpring
    _each(data, 'tri_area_springs',
          lambda a: s.tri_area_springs.append(TriAreaSpring(
              tri_id=a['tri_id'], kz=a.get('kz', 0.0))), report)
    _each(data, 'quad_area_springs',
          lambda a: s.quad_area_springs.append(QuadAreaSpring(
              quad_id=a['quad_id'], kz=a.get('kz', 0.0))), report)
    _each(data, 'surface_area_springs',
          lambda a: s.surface_area_springs.append(SurfaceAreaSpring(
              object_id=a['object_id'], kz=a.get('kz', 0.0))), report)
    _each(data, 'fields',
          lambda fld: s.add_field(fld['name'], fld.get('expression', '0.0')),
          report)

    from .structure_io import (_geo_objects_from_list, _beams_from_dict,
                               _beam_detail_from_dict)
    _section('geometry_objects',
             lambda: _geo_objects_from_list(s, data.get('geometry_objects', [])),
             report)
    s.beams = _beams_from_dict(data.get('beams'))
    _beam_detail_from_dict(s, data.get('beam_detail'))
    if s.beam_detail_load_error:
        report.add('error', 'beam_detail', s.beam_detail_load_error)

    _each(data, 'supports',
          lambda sp: s.add_support(sp['name'],
                                   sp.get('ux', sp.get('w', False)),
                                   sp.get('uy', sp.get('tx', False)),
                                   sp.get('tz', sp.get('ty', False))),
          report)
    _each(data, 'support_assignments',
          lambda a: s.assign_support(a['node_id'], a['support_name']), report)

    _each(data, 'node_springs',
          lambda sp: (s.add_node_spring(
              sp['node_id'], sp.get('kx', 0.0), sp.get('ky', 0.0),
              sp.get('kt', 0.0), mode_x=sp.get('mode_x', 'both'),
              mode_y=sp.get('mode_y', 'both')) if 'node_id' in sp else None),
          report)

    def _legacy_springs():
        old = {sp['name']: sp for sp in data.get('node_springs', [])
               if isinstance(sp, dict) and 'name' in sp}
        for a in data.get('node_spring_assignments', []):
            sp = old.get(a['spring_name'], {})
            s.add_node_spring(a['node_id'], sp.get('kx', 0.0),
                              sp.get('ky', 0.0), sp.get('kt', 0.0))
    _section('node_spring_assignments', _legacy_springs, report)

    _each(data, 'element_springs',
          lambda es: s.add_element_spring(
              es['element_id'], es.get('kx', 0.0), es.get('ky', 0.0),
              coord_sys=es.get('coord_sys', 'global'),
              mode_x=es.get('mode_x', 'both'),
              mode_y=es.get('mode_y', 'both')), report)

    def _constraint(c):
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
    _each(data, 'constraints', _constraint, report)

    _each(data, 'load_cases',
          lambda lc: s.add_load_case(
              lc['id'], lc.get('self_weight_factor', 0.0),
              action_type=lc.get('action_type', 'G'),
              create_analysis_case=False,
              load_duration=lc.get('load_duration'),
              category=lc.get('category', ''),
              psi0=lc.get('psi0'), psi1=lc.get('psi1'),
              psi2=lc.get('psi2')), report)
    _each(data, 'point_loads',
          lambda pl: s.add_point_load(
              pl['node_id'], pl['load_case_id'],
              pl.get('fx', pl.get('fz', 0.0)),
              pl.get('fy', pl.get('mx', 0.0)),
              pl.get('mz', pl.get('my', 0.0))), report)
    _each(data, 'distributed_loads',
          lambda dl: s.add_distributed_load(
              dl['element_id'], dl['load_case_id'],
              fxe=dl.get('fxe', 0.0), fxd=dl.get('fxd', 0.0),
              fye=dl.get('fye', 0.0), fyd=dl.get('fyd', 0.0),
              coord_sys=dl.get('coord_sys', 'global')), report)
    _each(data, 'element_point_loads',
          lambda epl: s.add_element_point_load(
              epl['element_id'], epl['load_case_id'], epl.get('a', 0.0),
              fx=epl.get('fx', 0.0), fy=epl.get('fy', 0.0),
              mz=epl.get('mz', 0.0),
              coord_sys=epl.get('coord_sys', 'global')), report)
    _each(data, 'support_settlements',
          lambda ss: s.create_support_settlement(
              ss['node_id'], ss['load_case_id'], ss.get('ux', 0.0),
              ss.get('uy', 0.0), ss.get('tz', 0.0)), report)
    _each(data, 'temperature_loads',
          lambda tl: s.add_temperature_load(
              tl['element_id'], tl['load_case_id'],
              tl.get('delta_t_uniform', 0.0), tl.get('delta_t_gradient', 0.0),
              tl.get('alpha', 1e-5)), report)
    _each(data, 'tri_temperature_loads',
          lambda tl: s.add_tri_temperature_load(
              tl['tri_id'], tl['load_case_id'], tl.get('dt_i', 0.0),
              tl.get('dt_j', 0.0), tl.get('dt_k', 0.0),
              tl.get('dt_gradient', 0.0)), report)
    _each(data, 'quad_temperature_loads',
          lambda tl: s.add_quad_temperature_load(
              tl['quad_id'], tl['load_case_id'], tl.get('dt_i', 0.0),
              tl.get('dt_j', 0.0), tl.get('dt_k', 0.0), tl.get('dt_l', 0.0),
              tl.get('dt_gradient', 0.0)), report)
    _each(data, 'area_temperature_loads',
          lambda tl: s.add_area_temperature_load(
              tl['object_id'], tl['load_case_id'],
              dt_uniform=tl.get('dt_uniform', 0.0),
              dt_gradient=tl.get('dt_gradient', 0.0),
              field_name=tl.get('field_name', ''),
              grad_field_name=tl.get('grad_field_name', '')), report)
    _each(data, 'line_temperature_loads',
          lambda tl: s.add_line_temperature_load(
              tl['object_id'], tl['load_case_id'], tl.get('dt_uniform', 0.0),
              tl.get('dt_gradient', 0.0), tl.get('field_name', ''),
              tl.get('grad_field_name', '')), report)
    _each(data, 'line_distributed_loads',
          lambda dl: s.add_line_distributed_load(
              dl['object_id'], dl['load_case_id'], dl.get('fx', 0.0),
              dl.get('fy', 0.0), dl.get('fx_field', ''), dl.get('fy_field', ''),
              dl.get('coord_sys', 'global')), report)
    _each(data, 'line_element_springs',
          lambda ls: s.add_line_element_spring(
              ls['object_id'], ls.get('kx', 0.0), ls.get('ky', 0.0),
              ls.get('kx_field', ''), ls.get('ky_field', ''),
              ls.get('coord_sys', 'global'), ls.get('mode_x', 'both'),
              ls.get('mode_y', 'both')), report)
    _each(data, 'spectral_functions',
          lambda sf: s.add_spectral_function(
              sf['id'], sf.get('description', ''), sf.get('damping', 0.05),
              sf.get('points', [])), report)
    _each(data, 'nodal_masses',
          lambda nm: s.add_nodal_mass(
              nm['node_id'], nm['mass_case_id'], nm.get('mx', 0.0),
              nm.get('my', 0.0), nm.get('mtz', 0.0)), report)
    _each(data, 'analysis_cases',
          lambda ac: s.add_analysis_case(
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
              solved=bool(ac.get('solved', False))), report)

    def _combo(lc):
        from .models import coerce_combo_type
        ctype = coerce_combo_type(lc.get('combo_type', 'LinearSum'))
        combo = s.add_load_combination(lc['id'], lc.get('coefficients', {}),
                                       combo_type=ctype)
        combo.analysis_coefficients = dict(lc.get('analysis_coefficients', {}))
    _each(data, 'load_combinations', _combo, report)

    _each(data, 'concrete_materials',
          lambda cm: s.add_concrete_material(
              cm['material_name'], cm['concrete_class'], cm['steel_class'],
              cm.get('gamma_c', 1.5), cm.get('gamma_s', 1.15),
              cm.get('alpha_cc', 1.0)), report)

    _section('load_combinations',
             s.normalize_combinations_to_analysis_cases, report)

    # Variants / phasing overlays reuse the strict helpers; guard per entry.
    from .structure_io import (_support_set_from_dict, _variant_from_dict,
                               _sequence_from_dict)
    _each(data, 'support_sets',
          lambda d: s.add_support_set(_support_set_from_dict(d)), report)
    _each(data, 'variants',
          lambda d: s.add_variant(_variant_from_dict(d)), report)
    _each(data, 'construction_sequences',
          lambda d: s.add_construction_sequence(_sequence_from_dict(d)), report)

    def _variant_combos():
        vc = data.get('variant_combinations', {})
        if isinstance(vc, dict):
            from .models import coerce_combo_type
            s.variant_combinations = {
                cid: {'op': coerce_combo_type(d.get('op', 'Envelope')),
                      'terms': [tuple(t) for t in d.get('terms', [])]}
                for cid, d in vc.items()}
    _section('variant_combinations', _variant_combos, report)

    return s


# ============================================================
# Human-readable report
# ============================================================

# Sections shown in this order in the "what was read" summary, with a friendly
# label. Anything not listed is appended afterwards under its raw key.
_SUMMARY_ORDER = [
    ('nodes', 'Nodes'), ('materials', 'Materials'), ('sections', 'Sections'),
    ('bar_elements', 'Bar elements'), ('tri_sections', 'Triangle sections'),
    ('tri_elements', 'Triangle elements'),
    ('quad_sections', 'Quad sections'), ('quad_elements', 'Quad elements'),
    ('supports', 'Supports'),
    ('support_assignments', 'Support assignments'),
    ('node_springs', 'Node springs'), ('element_springs', 'Element springs'),
    ('constraints', 'Constraints'), ('load_cases', 'Load cases'),
    ('point_loads', 'Point loads'), ('distributed_loads', 'Distributed loads'),
    ('element_point_loads', 'Element point loads'),
    ('support_settlements', 'Support settlements'),
    ('analysis_cases', 'Analysis cases'),
    ('load_combinations', 'Load combinations'),
    ('concrete_materials', 'Concrete materials'),
    ('variants', 'Variants'),
]


def format_report(report: LoadReport) -> str:
    """Render a LoadReport as plain text for a read-only textbox."""
    lines: list[str] = []
    name = Path(report.path).name if report.path else "(file)"
    lines.append(f"File: {name}")

    if not report.parsed:
        lines.append("")
        lines.append("The file could not be read as JSON.")
        if report.syntax_errors:
            lines.append("")
            lines.append("Syntax errors:")
            for i in report.syntax_errors:
                lines.append(f"  - {i.location()}: {i.message}")
        for i in report.errors:
            lines.append(f"  - {i.message}")
        return "\n".join(lines)

    # What was read.
    lines.append("")
    lines.append("Read from the file:")
    shown = set()
    for key, label in _SUMMARY_ORDER:
        if key in report.counts:
            lines.append(f"  {label}: {report.counts[key]}")
            shown.add(key)
    for key, n in report.counts.items():
        if key not in shown:
            lines.append(f"  {key}: {n}")
    if not report.counts:
        lines.append("  (nothing — the file has no recognised entries)")

    syn = report.syntax_errors
    err = report.errors
    warn = report.warnings

    lines.append("")
    if not syn and not err and not warn:
        lines.append("No problems found. The file is valid.")
        return "\n".join(lines)

    total = len(syn) + len(err) + len(warn)
    lines.append(f"Problems found: {total}")

    if syn:
        lines.append("")
        lines.append(f"Syntax errors ({len(syn)}) — the file was repaired to "
                     "read past these; fix them at the source:")
        for i in syn:
            lines.append(f"  - {i.location()}: {i.message}")
    if err:
        lines.append("")
        lines.append(f"Errors ({len(err)}) — these entries were skipped:")
        for i in err:
            lines.append(f"  - {i.location()}: {i.message}")
    if warn:
        lines.append("")
        lines.append(f"Warnings ({len(warn)}):")
        for i in warn:
            lines.append(f"  - {i.location()}: {i.message}")

    return "\n".join(lines)
