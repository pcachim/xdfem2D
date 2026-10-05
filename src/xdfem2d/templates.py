"""
Parametric model builders (templates) for xdfem2D.

Each function returns a ready-to-solve :class:`Structure2D`. GUI-free so they can
be used from scripts and tests; the GUI's template menu calls these.
"""
from __future__ import annotations

import math

from xdfem2d.structure import Structure2D
from xdfem2d.models import CANONICAL_SUPPORTS, CANONICAL_SUPPORTS_PLATE


def _seg_lengths(val, count, default: float = 1.0) -> list[float]:
    """Normalise a length parameter into a list of segment lengths.

    ``val`` may be a single number (uniform segments) or a sequence of
    per-segment lengths. When a sequence is given its length overrides
    ``count`` (so lists drive variable spacing); a scalar is replicated
    ``count`` times.
    """
    if isinstance(val, (list, tuple)):
        seq = []
        for x in val:
            try:
                fx = float(x)
            except (TypeError, ValueError):
                continue
            if fx > 0:
                seq.append(fx)
        if seq:
            return seq
        return [float(default)] * max(int(count), 1)
    try:
        v = float(val)
    except (TypeError, ValueError):
        v = float(default)
    return [v] * max(int(count), 1)


def _cum_positions(seg_lengths: list[float]) -> list[float]:
    """Cumulative coordinates (n+1 points) from n segment lengths."""
    pos = [0.0]
    for L in seg_lengths:
        pos.append(pos[-1] + L)
    return pos


# Area-element formulation choices, split by element type. The chosen
# formulation applies to the *active* element type (triangle or quad); an
# invalid pairing (e.g. a triangle formulation while quads are requested) is
# ignored so the paired section keeps its sensible default.
_MEMBRANE_TRI = ('ES-FEM', 'Allman', 'CST')
_MEMBRANE_QUAD = ('QM6', 'Q4')
_PLATE_TRI = ('MITC3', 'DKT')
_PLATE_QUAD = ('MITC4', 'DKT4')


def _membrane_forms(formulation, prefer_quad):
    """(tri, quad) membrane formulations, applying *formulation* to the active
    element type and defaulting the other."""
    tri, quad = 'ES-FEM', 'QM6'
    if prefer_quad and formulation in _MEMBRANE_QUAD:
        quad = formulation
    elif not prefer_quad and formulation in _MEMBRANE_TRI:
        tri = formulation
    return tri, quad


def _plate_forms(formulation, prefer_quad):
    """(tri, quad) plate-bending formulations, applying *formulation* to the
    active element type and defaulting the other."""
    tri, quad = 'MITC3', 'MITC4'
    if prefer_quad and formulation in _PLATE_QUAD:
        quad = formulation
    elif not prefer_quad and formulation in _PLATE_TRI:
        tri = formulation
    return tri, quad


def _tpl_finish_loads(s: Structure2D):
    """Give a template a self-weight load case named 'SW' (multiplier 1.0), a
    matching Linear analysis case, also 'SW', that applies it, and a ULS
    combination 'ULS-1.35SW' (1.35×SW) so every template has a ready-made
    ultimate-limit-state combination out of the box. The load case and the
    analysis case share the name deliberately — they live in separate
    namespaces and the coefficient resolves to the load case."""
    s.add_load_case('SW', self_weight_factor=1.0, create_analysis_case=False)
    s.add_analysis_case('SW', 'Linear', coefficients={'SW': 1.0})
    s.add_load_combination('ULS-1.35G', coefficients={'SW': 1.35})
    s.add_load_combination('SLS-G', coefficients={'SW': 1.0})


def _tpl_slab_pressure(s: Structure2D, target: str, pz: float):
    """Apply a slab surface pressure ``pz`` as an imposed (variable) action in
    its own load case ``'Q'`` — created on first use together with its Linear
    analysis case and folded into the ready-made ULS/SLS combinations
    (1.5·Q at ULS, 1.0·Q at SLS) — rather than lumping the pressure into the
    self-weight case ``'SW'`` (which then no longer represents self-weight
    alone). Idempotent: the ``'Q'`` case and the combination coefficients are
    only added once, so it is safe to call per element in a meshed template.

    A pressure of zero adds nothing: no 'Q' case, no empty load."""
    if not pz:
        return
    from xdfem2d.models import ActionType
    if not any(lc.id == 'Q' for lc in s.load_cases):
        s.add_load_case('Q', self_weight_factor=0.0,
                        action_type=ActionType.Q, create_analysis_case=False)
        s.add_analysis_case('Q', 'Linear', coefficients={'Q': 1.0})
        for combo_id, qf in (('ULS-1.35G', 1.5), ('SLS-G', 1.0)):
            combo = next((c for c in s.load_combinations if c.id == combo_id),
                         None)
            if combo is not None:
                combo.coefficients['Q'] = qf
    s.add_area_load(target, 'Q', pz=pz)


def _default_new_structure(domain: str = "plane") -> Structure2D:
    """An empty new model seeded with one default material, a bar section, a
    triangle section, a load case and an analysis case, so the user can start
    adding geometry immediately.

    The triangle section matters as much as the bar one: creating a surface
    object requires a triangle section to exist, so without it the very first
    Structure ▸ Objects ▸ Add surface on a new model refuses to start. The
    section's formulation follows the domain — a membrane CST in the plane
    domain, a plate-bending MITC3 in the plate domain (a slab/grillage model)."""
    s = Structure2D(domain=domain)
    # A Concrete material *with* its design strengths, so a new model is
    # designable out of the box (the section's material otherwise looks like
    # concrete but carries no fck/fyk, and the RC design silently produces
    # nothing — see rc_design._section_strengths). No explicit design= here
    # (26/09/2026): these were exactly default_design_for(CONCRETE)'s own
    # values, spelled out by hand -- add_material already fills them in on
    # its own when design is left out, and since add_material now validates
    # an EXPLICIT design's class label(s) against the live Eurocode tables
    # (see Structure2D._validate_material_class), restating them here only
    # bought this, the single most-called model constructor in the codebase,
    # a needless dependency on that lookup succeeding.
    s.add_material('Mat', elastic_modulus=30.0e6, unit_weight=25.0, alpha=1.0e-5,
                   material_type='Concrete')
    s.add_section('Sec', 'Mat', b=0.30, h=0.50, shape='Rectangular')
    if domain == "plate":
        s.add_plate_section('Slab', 'Mat', thickness=0.20)   # MITC3 (default)
        # Paired QuadSection under the same name — see TriSectionPanel
        # (Structure ▸ Sections ▸ Panel sections): every "Panel" is meant
        # to be a tri/quad pair sharing one name, and pairing it here means
        # the very first surface object on a brand new model can genuinely
        # be meshed as quads via "Mesh as", not just triangles
        # (dev/IMPLEMENT_QUAD.md Phase 8).
        s.add_quad_section('Slab', 'Mat', thickness=0.20, formulation='MITC4')
    else:
        s.add_tri_section('CST', 'Mat', thickness=0.20)
        s.add_quad_section('CST', 'Mat', thickness=0.20, formulation='QM6')
    # Every canonical support for this domain, pre-defined and ready to
    # assign — PIN/FIXED/ROLLER-.../GUIDED-.../BLOCK in the plane domain,
    # SIMPLE/CLAMPED/... in the plate domain (see CANONICAL_SUPPORTS /
    # CANONICAL_SUPPORTS_PLATE in xdfem2d.models). Without this a brand new
    # model has NO supports at all until the user visits Structure ▸
    # Supports by hand — every dropdown that lists existing supports
    # (assign_support, edge_supports, ...) starts empty, which is
    # especially confining for edge_supports: a per-edge restraint can only
    # reference a support that already exists.
    table = CANONICAL_SUPPORTS_PLATE if domain == "plate" else CANONICAL_SUPPORTS
    for (a, b, c), name in table.items():
        s.add_support(name, ux=a, uy=b, tz=c)
    _tpl_finish_loads(s)   # LC1 (self-weight 1.0) + Linear analysis case 'SW'
    return s


def _tpl_beam(support: str, span: float, b: float, h: float,
              E: float, g: float, n_seg: int = 4,
              material_type=None, shape=None, tw: float = 0.0, tf: float = 0.0,
              design: dict | None = None) -> Structure2D:
    """Single-span beam with a chosen support condition, subdivided into
    ``n_seg`` elements so the deflected shape and mid-span values are available.

    support : 'simply' (pin + roller), 'cantilever' (fixed–free),
              'fixed' (fixed–fixed), or 'propped' (fixed + roller).
    shape/tw/tf : optional section shape and wall/flange thicknesses.
    """
    s = Structure2D()
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Conc')
    s.add_material(mat_name, E, g, material_type=material_type, design=design)
    s.add_section('Sec', mat_name, b, h, shape=shape, tw=tw, tf=tf)
    n = max(1, int(n_seg))
    for i in range(n + 1):
        s.add_node(f'N{i}', i * span / n, 0.0)
    for i in range(n):
        s.add_bar_element(f'E{i}', f'N{i}', f'N{i+1}', 'Sec')
    last = f'N{n}'
    if support == 'cantilever':
        s.add_support('Fixed', ux=True, uy=True, tz=True)
        s.assign_support('N0', 'Fixed')
    elif support == 'fixed':
        s.add_support('Fixed', ux=True, uy=True, tz=True)
        s.assign_support('N0', 'Fixed'); s.assign_support(last, 'Fixed')
    elif support == 'propped':
        s.add_support('Fixed',  ux=True,  uy=True, tz=True)
        s.add_support('Roller', ux=False, uy=True, tz=False)
        s.assign_support('N0', 'Fixed'); s.assign_support(last, 'Roller')
    else:  # 'simply'
        s.add_support('Pin',    ux=True,  uy=True, tz=False)
        s.add_support('Roller', ux=False, uy=True, tz=False)
        s.assign_support('N0', 'Pin'); s.assign_support(last, 'Roller')
    _tpl_finish_loads(s)
    return s


def _tpl_wall(width: float, height: float, nx: int, ny: int,
              thickness: float, E: float, g: float, nu: float = 0.2,
              formulation: str = 'ES-FEM', prefer_quad: bool = False,
              material_type=None) -> Structure2D:
    """Rectangular wall / shear panel, fixed along its base — the simplest way
    to start a plane-stress (membrane) model. Meshed in triangles (ES-FEM by
    default) or quadrilaterals (QM6) when ``prefer_quad`` is set.

    formulation : 'ES-FEM' (default), 'Allman', or 'CST'.
    """
    s = Structure2D()
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Conc')
    s.add_material(mat_name, E, g, material_type=material_type, poisson=nu)
    _tf, _qf = _membrane_forms(formulation, prefer_quad)
    s.add_tri_section('Wall', mat_name, thickness=thickness, formulation=_tf)
    s.add_quad_section('Wall', mat_name, thickness=thickness, formulation=_qf)
    nx = max(1, int(nx)); ny = max(1, int(ny))
    ids = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            nid = f'n{i}_{j}'
            s.add_node(nid, width * i / nx, height * j / ny)
            ids[(i, j)] = nid
    e = 0
    for j in range(ny):
        for i in range(nx):
            a, bb = ids[(i, j)], ids[(i + 1, j)]
            c, d = ids[(i + 1, j + 1)], ids[(i, j + 1)]
            if prefer_quad:
                s.add_quad_element(f'e{e}', a, bb, c, d, 'Wall'); e += 1
            else:
                s.add_tri_element(f'e{e}', a, bb, c, 'Wall'); e += 1
                s.add_tri_element(f'e{e}', a, c, d, 'Wall');  e += 1
    s.add_support('Fixed', ux=True, uy=True, tz=True)
    for i in range(nx + 1):
        s.assign_support(ids[(i, 0)], 'Fixed')   # whole base fixed
    _tpl_finish_loads(s)
    return s


def _tpl_wall_beam(width: float, height: float, nx: int, ny: int,
                   thickness: float, E: float, g: float, nu: float = 0.2,
                   formulation: str = 'ES-FEM', prefer_quad: bool = False,
                   material_type=None) -> Structure2D:
    """Simply supported deep beam: a rectangular plane-stress panel supported
    at its two bottom corners (pin at the left, roller at the right) rather
    than fixed along the whole base. Meshed in triangles (ES-FEM) or
    quadrilaterals (QM6) when ``prefer_quad`` is set.

    ``width`` is the span and ``height`` the beam depth.
    """
    s = Structure2D()
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Conc')
    s.add_material(mat_name, E, g, material_type=material_type, poisson=nu)
    _tf, _qf = _membrane_forms(formulation, prefer_quad)
    s.add_tri_section('Wall', mat_name, thickness=thickness, formulation=_tf)
    s.add_quad_section('Wall', mat_name, thickness=thickness, formulation=_qf)
    nx = max(1, int(nx)); ny = max(1, int(ny))
    ids = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            nid = f'n{i}_{j}'
            s.add_node(nid, width * i / nx, height * j / ny)
            ids[(i, j)] = nid
    e = 0
    for j in range(ny):
        for i in range(nx):
            a, bb = ids[(i, j)], ids[(i + 1, j)]
            c, d = ids[(i + 1, j + 1)], ids[(i, j + 1)]
            if prefer_quad:
                s.add_quad_element(f'e{e}', a, bb, c, d, 'Wall'); e += 1
            else:
                s.add_tri_element(f'e{e}', a, bb, c, 'Wall'); e += 1
                s.add_tri_element(f'e{e}', a, c, d, 'Wall');  e += 1
    # Simple supports at the two bottom corners: pin (ux+uy) left, roller (uy) right.
    s.add_support('Pin',    ux=True,  uy=True, tz=False)
    s.add_support('Roller', ux=False, uy=True, tz=False)
    s.assign_support(ids[(0, 0)],  'Pin')
    s.assign_support(ids[(nx, 0)], 'Roller')
    _tpl_finish_loads(s)
    return s


def _tpl_portal_frame(span: float, height: float,
                      b: float, h: float, E: float, g: float,
                      material_type=None, shape=None, tw: float = 0.0, tf: float = 0.0,
                      design: dict | None = None) -> Structure2D:
    """Single-bay, single-storey portal frame with fixed bases.

    A pure ``_tpl_frame(1, 1, span, height, ...)`` call -- a portal frame IS
    a 1-bay, 1-storey frame, same sections (Beam/Col, columns rotated), same
    Fixed-base support, same material logic. Confirmed by reading both
    builders side by side, 24/09/2026, while chasing an AI-tool bug where a
    small model asked for a multi-bay/multi-storey frame picked this kind by
    name ("portico") and then had no way to pass bays/floors. Kept as its
    own function (not just deleted) because the GUI (main_window.py,
    ai_assistant.py) imports ``_tpl_portal_frame`` by name directly, and
    template_api.py's TEMPLATE_KINDS still lists 'portal_frame' as its own
    kind (span/height, its own defaults) for anyone already calling
    create_from_template('portal_frame', ...) -- only the AI-facing catalog
    in ai_context.py stops naming it, so a model only ever reaches for
    'frame'. The only externally visible change from this delegation is the
    node/element id scheme: N{col}_{fl}/C{col}_{fl}/B{col}_{fl} instead of
    N0..N3/C0,B0,C1 -- nothing in this repo's tests reads those ids.
    """
    return _tpl_frame(1, 1, span, height, b, h, E, g,
                      material_type=material_type, shape=shape, tw=tw, tf=tf,
                      design=design)


# Beam support condition → plane restraint triple (ux, uy, tz). 'free' means no
# support. 'pin' fixes both translations (canonical PIN), 'fixed' adds rotation
# (canonical FIXED). Applied per position (left / intermediate / right).
_BEAM_SUPPORT_TRIPLE = {'pin': (True, True, False), 'fixed': (True, True, True)}


def _assign_beam_supports(s, node_ids, left, intermediate, right):
    """Assign left / intermediate / right supports to an ordered list of beam
    node ids, using the canonical plane support names (PIN / FIXED). A ``free``
    position gets no support."""
    from .models import canonical_support_name
    alias = {'free': 'free', 'pin': 'pin', 'fixed': 'fixed'}

    def _assign(nid, kind):
        trip = _BEAM_SUPPORT_TRIPLE.get(alias.get(str(kind).lower(), 'free'))
        if trip is None:
            return
        name = canonical_support_name(*trip, domain='plane')
        if name not in s.supports:
            s.add_support(name, ux=trip[0], uy=trip[1], tz=trip[2])
        s.assign_support(nid, name)

    _assign(node_ids[0], left)
    _assign(node_ids[-1], right)
    for nid in node_ids[1:-1]:
        _assign(nid, intermediate)


def _tpl_continuous_beam(spans: int, span_len: float, height: float,
                          b: float, h: float, E: float, g: float,
                          material_type=None, shape=None, tw: float = 0.0, tf: float = 0.0,
                          design: dict | None = None,
                          left: str = 'pin', intermediate: str = 'pin',
                          right: str = 'pin') -> Structure2D:
    """Continuous beam with an independent support condition on the left end,
    the intermediate supports and the right end — each ``free``/``pin``/
    ``fixed`` (canonical PIN/FIXED names). A single-span beam has no
    intermediate supports.

    ``span_len`` may be a single length (uniform spans) or a list of per-span
    lengths; a list overrides ``spans`` and yields variable spacing.
    """
    s = Structure2D()
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Conc')
    s.add_material(mat_name, E, g, material_type=material_type, design=design)
    s.add_section('Sec', mat_name, b, h, shape=shape, tw=tw, tf=tf)
    seg = _seg_lengths(span_len, spans, 6.0)
    spans = len(seg)
    xs = _cum_positions(seg)
    for i in range(spans + 1):
        s.add_node(f'N{i}', xs[i], 0.0)
    for i in range(spans):
        s.add_bar_element(f'E{i}', f'N{i}', f'N{i+1}', 'Sec')
    _assign_beam_supports(s, [f'N{i}' for i in range(spans + 1)],
                          left, intermediate, right)
    _tpl_finish_loads(s)
    return s


def _tpl_frame(bays: int, floors: int, bay_w: float, floor_h: float,
               b: float, h: float, E: float, g: float,
               material_type=None, shape=None, tw: float = 0.0, tf: float = 0.0,
               design: dict | None = None) -> Structure2D:
    """Rectangular multi-bay multi-storey frame.

    ``bay_w`` and ``floor_h`` may each be a single value (uniform grid) or a
    list of per-bay widths / per-floor heights; a list overrides ``bays`` /
    ``floors`` and produces variable spacing.
    """
    s = Structure2D()
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Conc')
    s.add_material(mat_name, E, g, material_type=material_type, design=design)
    s.add_section('Beam', mat_name, b, h, shape=shape, tw=tw, tf=tf)
    s.add_section('Col',  mat_name, h, b, shape=shape, tw=tw, tf=tf)   # columns rotated
    s.add_support('Fixed', ux=True, uy=True, tz=True)
    bw = _seg_lengths(bay_w, bays, 5.0);   bays = len(bw);   xs = _cum_positions(bw)
    fh = _seg_lengths(floor_h, floors, 3.0); floors = len(fh); ys = _cum_positions(fh)
    # Nodes: (col, floor) — col 0..bays, floor 0..floors
    for fl in range(floors + 1):
        for col in range(bays + 1):
            s.add_node(f'N{col}_{fl}', xs[col], ys[fl])
    # Columns
    for fl in range(floors):
        for col in range(bays + 1):
            s.add_bar_element(f'C{col}_{fl}', f'N{col}_{fl}', f'N{col}_{fl+1}', 'Col')
    # Beams
    for fl in range(1, floors + 1):
        for col in range(bays):
            s.add_bar_element(f'B{col}_{fl}', f'N{col}_{fl}', f'N{col+1}_{fl}', 'Beam')
    # Fix base
    for col in range(bays + 1):
        s.assign_support(f'N{col}_0', 'Fixed')
    _tpl_finish_loads(s)
    return s


def _tpl_truss_warren(panels: int, span: float, depth: float,
                       b: float, h: float, E: float, g: float,
                       material_type=None, shape=None, tw: float = 0.0, tf: float = 0.0,
                       design: dict | None = None) -> Structure2D:
    """Warren truss (triangulated, no verticals)."""
    s = Structure2D()
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Conc')
    s.add_material(mat_name, E, g, material_type=material_type, design=design)
    s.add_section('Bar', mat_name, b, h, shape=shape, tw=tw, tf=tf)
    s.add_support('Pin',    ux=True, uy=True, tz=False)
    s.add_support('Roller', ux=False, uy=True, tz=False)
    dx = span / panels
    # Bottom chord nodes
    for i in range(panels + 1):
        s.add_node(f'B{i}', i * dx, 0.0)
    # Top chord nodes (offset by half panel)
    for i in range(panels):
        s.add_node(f'T{i}', (i + 0.5) * dx, depth)
    # Bottom chord members
    for i in range(panels):
        s.add_bar_element(f'BC{i}', f'B{i}', f'B{i+1}', 'Bar')
    # Diagonals and top chord
    for i in range(panels):
        s.add_bar_element(f'DL{i}', f'B{i}',   f'T{i}', 'Bar')
        s.add_bar_element(f'DR{i}', f'B{i+1}', f'T{i}', 'Bar')
        if i > 0:
            s.add_bar_element(f'TC{i}', f'T{i-1}', f'T{i}', 'Bar')
    s.assign_support('B0',       'Pin')
    s.assign_support(f'B{panels}', 'Roller')
    _tpl_finish_loads(s)
    return s


def _tpl_truss_howe(panels: int, span: float, depth: float,
                     b: float, h: float, E: float, g: float,
                     material_type=None, shape=None, tw: float = 0.0, tf: float = 0.0,
                     design: dict | None = None) -> Structure2D:
    """Howe truss (diagonals slope toward centre, verticals present).

    The panel count is kept **even** (>= 2): with one diagonal per panel an odd
    count has a central panel straddling mid-span whose single diagonal cannot
    be mirror-symmetric, so an odd request is rounded up to the next even count.
    """
    panels = max(2, panels)
    if panels % 2:
        panels += 1
    s = Structure2D()
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Conc')
    s.add_material(mat_name, E, g, material_type=material_type, design=design)
    s.add_section('Bar', mat_name, b, h, shape=shape, tw=tw, tf=tf)
    s.add_support('Pin',    ux=True, uy=True, tz=False)
    s.add_support('Roller', ux=False, uy=True, tz=False)
    dx = span / panels
    for i in range(panels + 1):
        s.add_node(f'B{i}', i * dx, 0.0)
        s.add_node(f'T{i}', i * dx, depth)
    # Chords
    for i in range(panels):
        s.add_bar_element(f'BC{i}', f'B{i}', f'B{i+1}', 'Bar')
        s.add_bar_element(f'TC{i}', f'T{i}', f'T{i+1}', 'Bar')
    # Verticals
    for i in range(panels + 1):
        s.add_bar_element(f'V{i}', f'B{i}', f'T{i}', 'Bar')
    # Diagonals: in Howe they run from bottom outer → top inner
    for i in range(panels):
        mid = panels // 2
        if i < mid:
            s.add_bar_element(f'D{i}', f'B{i}', f'T{i+1}', 'Bar')
        else:
            s.add_bar_element(f'D{i}', f'B{i+1}', f'T{i}', 'Bar')
    s.assign_support('B0',       'Pin')
    s.assign_support(f'B{panels}', 'Roller')
    _tpl_finish_loads(s)
    return s


def _tpl_truss_pratt(panels: int, span: float, depth: float,
                      b: float, h: float, E: float, g: float,
                      material_type=None, shape=None, tw: float = 0.0, tf: float = 0.0,
                      design: dict | None = None) -> Structure2D:
    """Pratt truss (diagonals in tension under gravity, verticals in compression).

    The panel count is kept **even** (>= 2), for the same symmetry reason as the
    Howe truss: an odd request is rounded up to the next even count.
    """
    panels = max(2, panels)
    if panels % 2:
        panels += 1
    s = Structure2D()
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Conc')
    s.add_material(mat_name, E, g, material_type=material_type, design=design)
    s.add_section('Bar', mat_name, b, h, shape=shape, tw=tw, tf=tf)
    s.add_support('Pin',    ux=True, uy=True, tz=False)
    s.add_support('Roller', ux=False, uy=True, tz=False)
    dx = span / panels
    for i in range(panels + 1):
        s.add_node(f'B{i}', i * dx, 0.0)
        s.add_node(f'T{i}', i * dx, depth)
    for i in range(panels):
        s.add_bar_element(f'BC{i}', f'B{i}', f'B{i+1}', 'Bar')
        s.add_bar_element(f'TC{i}', f'T{i}', f'T{i+1}', 'Bar')
    for i in range(panels + 1):
        s.add_bar_element(f'V{i}', f'B{i}', f'T{i}', 'Bar')
    # Pratt diagonals: from bottom inner → top outer (opposite to Howe)
    for i in range(panels):
        mid = panels // 2
        if i < mid:
            s.add_bar_element(f'D{i}', f'B{i+1}', f'T{i}', 'Bar')
        else:
            s.add_bar_element(f'D{i}', f'B{i}', f'T{i+1}', 'Bar')
    s.assign_support('B0',       'Pin')
    s.assign_support(f'B{panels}', 'Roller')
    _tpl_finish_loads(s)
    return s


def _tpl_truss_long(panels: int, span: float, depth: float,
                    b: float, h: float, E: float, g: float,
                    material_type=None, shape=None, tw: float = 0.0, tf: float = 0.0,
                    design: dict | None = None) -> Structure2D:
    """Long truss (X-braced): the Howe and Pratt diagonals superimposed, so
    every panel carries a full cross (X) of diagonals plus the verticals — the
    counter-braced panel Col. Stephen H. Long patented in 1830.

    With both diagonals in every panel the truss is mirror-symmetric for any
    panel count (like a Warren), so no even/odd handling is needed. The extra
    diagonal per panel makes it statically indeterminate — carried fine by the
    stiffness solver.
    """
    panels = max(2, panels)
    s = Structure2D()
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Conc')
    s.add_material(mat_name, E, g, material_type=material_type, design=design)
    s.add_section('Bar', mat_name, b, h, shape=shape, tw=tw, tf=tf)
    s.add_support('Pin',    ux=True, uy=True, tz=False)
    s.add_support('Roller', ux=False, uy=True, tz=False)
    dx = span / panels
    for i in range(panels + 1):
        s.add_node(f'B{i}', i * dx, 0.0)
        s.add_node(f'T{i}', i * dx, depth)
    for i in range(panels):
        s.add_bar_element(f'BC{i}', f'B{i}', f'B{i+1}', 'Bar')
        s.add_bar_element(f'TC{i}', f'T{i}', f'T{i+1}', 'Bar')
    for i in range(panels + 1):
        s.add_bar_element(f'V{i}', f'B{i}', f'T{i}', 'Bar')
    # Both diagonals in every panel → an X (Howe "/" + Pratt "\").
    for i in range(panels):
        s.add_bar_element(f'DH{i}', f'B{i}',   f'T{i+1}', 'Bar')   # "/"
        s.add_bar_element(f'DP{i}', f'B{i+1}', f'T{i}',   'Bar')   # "\"
    s.assign_support('B0',       'Pin')
    s.assign_support(f'B{panels}', 'Roller')
    _tpl_finish_loads(s)
    return s


def _tpl_truss_town(span: float, depth: float, pitch: float, k: int,
                    b: float, h: float, E: float, g: float,
                    material_type=None, shape=None, tw: float = 0.0, tf: float = 0.0,
                    design: dict | None = None) -> Structure2D:
    """Town lattice truss (Ithiel Town, 1820): a dense web of two crossing
    diagonal families, no interior verticals — only end posts.

    Density is set by ``pitch`` (target spacing of the chord nodes, rounded so a
    whole number of panels fits the span) and ``k`` (the lattice span: how many
    panels each diagonal jumps). Each "/" runs ``B_i -> T_{i+k}`` and each "\\"
    runs ``B_{i+k} -> T_i``; the two families cross in the interior but are
    joined only at the chords (Option A — a multiple-intersection lattice).
    End posts (verticals at the two ends) close the otherwise open corner
    panels. The web is highly statically indeterminate — fine for the stiffness
    solver.
    """
    k = max(1, int(k))
    n = max(k + 1, round(span / pitch)) if pitch and pitch > 0 else k + 1
    s = Structure2D()
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Conc')
    s.add_material(mat_name, E, g, material_type=material_type, design=design)
    s.add_section('Bar', mat_name, b, h, shape=shape, tw=tw, tf=tf)
    s.add_support('Pin',    ux=True, uy=True, tz=False)
    s.add_support('Roller', ux=False, uy=True, tz=False)
    dx = span / n
    for i in range(n + 1):
        s.add_node(f'B{i}', i * dx, 0.0)
        s.add_node(f'T{i}', i * dx, depth)
    # Chords.
    for i in range(n):
        s.add_bar_element(f'BC{i}', f'B{i}', f'B{i+1}', 'Bar')
        s.add_bar_element(f'TC{i}', f'T{i}', f'T{i+1}', 'Bar')
    # End posts only (classic Town lattice has no interior verticals).
    s.add_bar_element('VP0', 'B0', 'T0', 'Bar')
    s.add_bar_element(f'VP{n}', f'B{n}', f'T{n}', 'Bar')
    # Two crossing diagonal families, each spanning k panels.
    for i in range(n - k + 1):
        s.add_bar_element(f'DH{i}', f'B{i}',   f'T{i+k}', 'Bar')   # "/"
        s.add_bar_element(f'DP{i}', f'B{i+k}', f'T{i}',   'Bar')   # "\"
    s.assign_support('B0',  'Pin')
    s.assign_support(f'B{n}', 'Roller')
    _tpl_finish_loads(s)
    return s


def _tpl_truss_kingpost(panels: int, span: float, depth: float,
                        b: float, h: float, E: float, g: float,
                        material_type=None, shape=None, tw: float = 0.0, tf: float = 0.0,
                        design: dict | None = None) -> Structure2D:
    """King-post truss, generalised to several diagonals.

    ``panels`` counts panels along the whole tie beam, exactly like
    Warren/Howe/Pratt (rounded up to the next even number if given odd, since
    the truss is symmetric about the ridge) — not per rafter leg.

    The tie beam (bottom chord) is divided into ``panels`` equal segments;
    the top chord is two straight rafters rising from each support to a
    central ridge at ``depth`` above the tie, meeting the same panel points
    as the tie. Every *interior* tie node gets a vertical strut up to the
    rafter (the middle one, at mid-span, is the king post proper). Every
    *interior* panel (i.e. every panel except the two end ones, which sit
    right against a support and have no room for a non-duplicate diagonal)
    gets one diagonal strut, both halves radiating OUT from the king post's
    foot towards the eaves.
    """
    # An odd requested panel count (n >= 3) still rounds up to the next even
    # panel count, but its diagonals are inverted (see the diagonal loop) so
    # that n and n+1 produce visibly distinct trusses rather than the same one.
    orig_odd = (panels >= 3 and panels % 2 == 1)
    if panels != 1 and panels % 2:
        panels += 1
    if panels < 1:
        panels = 2
    s = Structure2D()
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Conc')
    s.add_material(mat_name, E, g, material_type=material_type, design=design)
    s.add_section('Bar', mat_name, b, h, shape=shape, tw=tw, tf=tf)
    s.add_support('Pin',    ux=True, uy=True, tz=False)
    s.add_support('Roller', ux=False, uy=True, tz=False)
    mid = panels // 2
    dx = span / panels
    # Tie-beam (bottom chord) nodes — the whole tie, like Warren/Howe/Pratt.
    for i in range(panels + 1):
        s.add_node(f'B{i}', i * dx, 0.0)
    # Rafter (top chord) nodes — only the interior ones are distinct from
    # the tie: the rafter starts/ends at the support height (0), so its
    # first/last points coincide with B0/B{panels}.
    if panels == 1:
        s.add_node('B2', span / 2, 0)
        s.add_node('T1', span / 2, depth)

    elif panels == 2:
        panels2 = 4
        mid2 = panels2 // 2
        dx2 = span / panels2
        for i in range(1, 4):
            y = depth * (i / mid2 if i <= mid2 else (panels2 - i) / mid2)
            s.add_node(f'T{i}', i * dx2, y)
    else:
        for i in range(1, panels):
            y = depth * (i / mid if i <= mid else (panels - i) / mid)
            s.add_node(f'T{i}', i * dx, y)

    def _top(i):
        return f'B{i}' if i in (0, panels) else f'T{i}'

    # Tie beam and the two rafters.
    if panels == 1:
        s.add_bar_element(f'BC0', 'B0', 'B2', 'Bar')
        s.add_bar_element(f'BC1', 'B2', 'B1', 'Bar')
        s.add_bar_element(f'TC0', 'B0', 'T1', 'Bar')
        s.add_bar_element(f'TC1', 'T1', 'B1', 'Bar')
        s.add_bar_element(f'V0', 'B2', 'T1', 'Bar')
    elif panels == 2:
        s.add_bar_element(f'BC0', 'B0', 'B1', 'Bar')
        s.add_bar_element(f'BC1', 'B1', 'B2', 'Bar')
        s.add_bar_element(f'TC0', 'B0', 'T1', 'Bar')
        s.add_bar_element(f'TC1', 'T1', 'T2', 'Bar')
        s.add_bar_element(f'TC2', 'T2', 'T3', 'Bar')
        s.add_bar_element(f'TC3', 'T3', 'B2', 'Bar')
        s.add_bar_element(f'V0', 'B1', 'T1', 'Bar')
        s.add_bar_element(f'V1', 'B1', 'T2', 'Bar')
        s.add_bar_element(f'V2', 'B1', 'T3', 'Bar')
    else:
        for i in range(panels):
            s.add_bar_element(f'BC{i}', f'B{i}', f'B{i+1}', 'Bar')
            s.add_bar_element(f'TC{i}', _top(i), _top(i+1), 'Bar')
        # Verticals — one per interior tie node; the one at mid-span is the
        # king post proper (full truss depth).
        for i in range(1, panels):
            s.add_bar_element(f'V{i}', f'B{i}', _top(i), 'Bar')
    # Diagonals — one per *interior* panel, radiating OUT from the king
    # post's foot (B{mid}) towards the eaves, mirroring the classic king
    # post's two struts (which spring from the post's foot up to the
    # rafters, not converge into the ridge from the outer tie nodes).
    #
    # The two end panels (i == 0 and i == panels-1) get none: a panel right
    # at a support is already a plain triangle (tie + rafter meeting at that
    # support — _top(0) == B0 and _top(panels) == B{panels}), and *both*
    # possible "diagonals" there just duplicate an existing chord member
    # (either the tie, B{i+1}-B{i} again, or the rafter, B{i}-T{i+1} again)
    # rather than adding a real cross-brace. With only 2 panels this means
    # no diagonals at all — the traditional minimal king post (post + 2
    # struts) is what panels == 4 gives you here; panels == 2 is the bare
    # tie + rafters + post.
    # For an even requested count the diagonals radiate OUT from the king-post
    # foot (left half "\", right half "/"); for an odd count (n >= 3) every
    # diagonal is inverted ("/" <-> "\"), so it points toward the ridge instead.
    for i in range(1, panels - 1):
        if (i < mid) != orig_odd:
            s.add_bar_element(f'D{i}', f'B{i+1}', _top(i), 'Bar')     # "\"
        else:
            s.add_bar_element(f'D{i}', f'B{i}', _top(i+1), 'Bar')     # "/"
    s.assign_support('B0',       'Pin')
    s.assign_support(f'B{panels}', 'Roller')
    _tpl_finish_loads(s)
    return s


def _tpl_arch(n_segments: int, span: float, rise: float,
               b: float, h: float, E: float, g: float,
               material_type=None, shape=None, tw: float = 0.0, tf: float = 0.0,
               design: dict | None = None) -> Structure2D:
    """Parabolic arch (two-hinged)."""
    import math
    s = Structure2D()
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Conc')
    s.add_material(mat_name, E, g, material_type=material_type, design=design)
    s.add_section('Sec', mat_name, b, h, shape=shape, tw=tw, tf=tf)
    s.add_support('Pin',    ux=True, uy=True, tz=False)
    s.add_support('Roller', ux=False, uy=True, tz=False)
    # Parabolic y = 4*rise*(x/span)*(1 - x/span)
    for i in range(n_segments + 1):
        x = i * span / n_segments
        y = 4.0 * rise * (x / span) * (1.0 - x / span)
        s.add_node(f'N{i}', x, y)
    for i in range(n_segments):
        s.add_bar_element(f'E{i}', f'N{i}', f'N{i+1}', 'Sec')
    s.assign_support('N0',          'Pin')
    s.assign_support(f'N{n_segments}', 'Roller')
    _tpl_finish_loads(s)
    return s


# ── Object-based variants (parametric geometry, auto-meshed) ─────────────────
# These build the same structures out of geometry OBJECTS (lines / polylines /
# rectangle) that discretise into nodes and elements at solve time, instead of
# explicit nodes and elements. The editable model stays parametric.

def _beam_end_supports(s: Structure2D, support: str, n0: str, n1: str):
    """Assign the beam's end supports (shared by the explicit and object beams)."""
    if support == 'cantilever':
        s.add_support('Fixed', ux=True, uy=True, tz=True)
        s.assign_support(n0, 'Fixed')
    elif support == 'fixed':
        s.add_support('Fixed', ux=True, uy=True, tz=True)
        s.assign_support(n0, 'Fixed'); s.assign_support(n1, 'Fixed')
    elif support == 'propped':
        s.add_support('Fixed',  ux=True,  uy=True, tz=True)
        s.add_support('Roller', ux=False, uy=True, tz=False)
        s.assign_support(n0, 'Fixed'); s.assign_support(n1, 'Roller')
    else:
        s.add_support('Pin',    ux=True,  uy=True, tz=False)
        s.add_support('Roller', ux=False, uy=True, tz=False)
        s.assign_support(n0, 'Pin'); s.assign_support(n1, 'Roller')


def _tpl_beam_obj(support: str, span: float, b: float, h: float,
                  E: float, g: float, material_type=None, shape=None,
                  tw: float = 0.0, tf: float = 0.0) -> Structure2D:
    """Single-span beam as a line object (meshed into bars at solve time)."""
    s = Structure2D()
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Conc')
    s.add_material(mat_name, E, g, material_type=material_type)
    s.add_section('Sec', mat_name, b, h, shape=shape, tw=tw, tf=tf)
    s.add_geo_line('L', 0.0, 0.0, span, 0.0, section_name='Sec', divisions=4)
    _beam_end_supports(s, support, 'L.p0', 'L.p1')
    _tpl_finish_loads(s)
    return s


def _tpl_continuous_beam_obj(spans: int, span_len: float, b: float, h: float,
                             E: float, g: float, material_type=None, shape=None,
                             tw: float = 0.0, tf: float = 0.0,
                             left: str = 'pin', intermediate: str = 'pin',
                             right: str = 'pin') -> Structure2D:
    """Continuous beam as a polyline object through the support points, with an
    independent left / intermediate / right support condition (free/pin/fixed,
    canonical names).

    ``span_len`` may be a single length or a list of per-span lengths."""
    s = Structure2D()
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Conc')
    s.add_material(mat_name, E, g, material_type=material_type)
    s.add_section('Sec', mat_name, b, h, shape=shape, tw=tw, tf=tf)
    seg = _seg_lengths(span_len, spans, 6.0)
    spans = len(seg)
    xs = _cum_positions(seg)
    verts = [(xs[i], 0.0) for i in range(spans + 1)]
    s.add_geo_polyline('L', verts, section_name='Sec', divisions=2)
    _assign_beam_supports(s, [f'L.p{i}' for i in range(spans + 1)],
                          left, intermediate, right)
    _tpl_finish_loads(s)
    return s


def _tpl_arch_obj(n_segments: int, span: float, rise: float,
                  b: float, h: float, E: float, g: float, material_type=None, shape=None,
                  tw: float = 0.0, tf: float = 0.0) -> Structure2D:
    """Parabolic arch as a polyline object (two-hinged)."""
    s = Structure2D()
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Conc')
    s.add_material(mat_name, E, g, material_type=material_type)
    s.add_section('Sec', mat_name, b, h, shape=shape, tw=tw, tf=tf)
    pts = []
    for i in range(n_segments + 1):
        x = i * span / n_segments
        pts.append((x, 4.0 * rise * (x / span) * (1.0 - x / span)))
    s.add_geo_polyline('A', pts, section_name='Sec', divisions=1)
    s.add_support('Pin',    ux=True,  uy=True, tz=False)
    s.add_support('Roller', ux=False, uy=True, tz=False)
    s.assign_support('A.p0', 'Pin')
    s.assign_support(f'A.p{n_segments}', 'Roller')
    _tpl_finish_loads(s)
    return s


def _tpl_frame_obj(bays: int, floors: int, bay_w: float, floor_h: float,
                   b: float, h: float, E: float, g: float, material_type=None, shape=None,
                   tw: float = 0.0, tf: float = 0.0) -> Structure2D:
    """Rectangular frame as line objects welded at the joints.

    One object per column (spanning the full height) and one object per floor
    beam-line (spanning the full length); the intermediate joint nodes are
    created first, so each line welds to them by coincidence while staying a
    single parametric object. ``bay_w`` / ``floor_h`` may be single values or
    lists for variable spacing."""
    s = Structure2D()
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Conc')
    s.add_material(mat_name, E, g, material_type=material_type)
    s.add_section('Beam', mat_name, b, h, shape=shape, tw=tw, tf=tf)
    s.add_section('Col',  mat_name, h, b, shape=shape, tw=tw, tf=tf)
    s.add_support('Fixed', ux=True, uy=True, tz=True)
    bw = _seg_lengths(bay_w, bays, 5.0);   bays = len(bw);   xs = _cum_positions(bw)
    fh = _seg_lengths(floor_h, floors, 3.0); floors = len(fh); ys = _cum_positions(fh)
    for fl in range(floors + 1):
        for col in range(bays + 1):
            s.add_node(f'N{col}_{fl}', xs[col], ys[fl])
    # One full-height column line per column axis (welds to every floor node).
    for col in range(bays + 1):
        x = xs[col]
        s.add_geo_line(f'C{col}', x, ys[0], x, ys[floors], section_name='Col')
    # One full-length beam line per floor (welds to every column node).
    for fl in range(1, floors + 1):
        y = ys[fl]
        s.add_geo_line(f'B{fl}', xs[0], y, xs[bays], y, section_name='Beam')
    for col in range(bays + 1):
        s.assign_support(f'N{col}_0', 'Fixed')
    _tpl_finish_loads(s)
    return s


def _tpl_wall_obj(width: float, height: float, nx: int, ny: int,
                  thickness: float, E: float, g: float, nu: float = 0.2,
                  formulation: str = 'ES-FEM',
                  prefer_quad: bool = False,
                  material_type=None, design: dict | None = None) -> Structure2D:
    """Wall panel as a rectangle object (auto-meshed into triangles, or quads
    when ``prefer_quad``). The base is fixed by assigning the same support to
    both bottom corners, which propagates along the whole base edge."""
    s = Structure2D()
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Conc')
    s.add_material(mat_name, E, g, material_type=material_type, poisson=nu, design=design)
    _tf, _qf = _membrane_forms(formulation, prefer_quad)
    s.add_tri_section('Wall', mat_name, thickness=thickness, formulation=_tf)
    s.add_quad_section('Wall', mat_name, thickness=thickness, formulation=_qf)
    target = min(width / max(int(nx), 1), height / max(int(ny), 1))
    s.add_geo_rectangle('R', (0.0, 0.0), (width, height),
                        section_name='Wall', target_size=target,
                        prefer_quad=prefer_quad)
    s.add_support('Fixed', ux=True, uy=True, tz=True)
    s.assign_support('R.p0', 'Fixed')   # bottom-left corner
    s.assign_support('R.p1', 'Fixed')   # bottom-right corner → propagates along base
    _tpl_finish_loads(s)
    return s


def _tpl_wall_beam_obj(width: float, height: float, nx: int, ny: int,
                       thickness: float, E: float, g: float, nu: float = 0.2,
                       formulation: str = 'ES-FEM',
                       prefer_quad: bool = False,
                       material_type=None) -> Structure2D:
    """Simply supported deep beam as a rectangle object (auto-meshed into
    triangles, or quads when ``prefer_quad``). Pinned at the bottom-left
    corner, roller at the bottom-right corner."""
    s = Structure2D()
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Conc')
    s.add_material(mat_name, E, g, material_type=material_type, poisson=nu)
    _tf, _qf = _membrane_forms(formulation, prefer_quad)
    s.add_tri_section('Wall', mat_name, thickness=thickness, formulation=_tf)
    s.add_quad_section('Wall', mat_name, thickness=thickness, formulation=_qf)
    target = min(width / max(int(nx), 1), height / max(int(ny), 1))
    s.add_geo_rectangle('R', (0.0, 0.0), (width, height),
                        section_name='Wall', target_size=target,
                        prefer_quad=prefer_quad)
    s.add_support('Pin',    ux=True,  uy=True, tz=False)
    s.add_support('Roller', ux=False, uy=True, tz=False)
    s.assign_support('R.p0', 'Pin')     # bottom-left corner
    s.assign_support('R.p1', 'Roller')  # bottom-right corner
    _tpl_finish_loads(s)
    return s


def _tpl_wall_frame(bays: int, floors: int, bay_w, floor_h,
                    wall_width: float, wall_nx: int,
                    b: float, h: float, E: float, g: float,
                    thickness: float, nu: float = 0.2,
                    formulation: str = 'ES-FEM', wall_side: str = 'left',
                    material_type=None, shape=None, tw: float = 0.0, tf: float = 0.0,
                    prefer_quad: bool = False) -> Structure2D:
    """Rectangular frame with a shear wall at one end (explicit mesh). The wall
    is meshed in ES-FEM triangles or QM6 quads (``prefer_quad``).

    The wall covers the full frame height and is meshed in modules aligned with
    the floor levels, so its interface nodes coincide with — and share — the
    frame's floor joints (connection at each floor). ``bay_w`` / ``floor_h`` may
    be single values or lists; ``wall_side`` is 'left' or 'right'. Frame bases
    and wall base are fixed."""
    s = Structure2D()
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Conc')
    s.add_material(mat_name, E, g, material_type=material_type, poisson=nu)
    s.add_section('Beam', mat_name, b, h, shape=shape, tw=tw, tf=tf)
    s.add_section('Col',  mat_name, h, b, shape=shape, tw=tw, tf=tf)
    _tf, _qf = _membrane_forms(formulation, prefer_quad)
    s.add_tri_section('Wall', mat_name, thickness=thickness, formulation=_tf)
    s.add_quad_section('Wall', mat_name, thickness=thickness, formulation=_qf)
    s.add_support('Fixed', ux=True, uy=True, tz=True)
    bw = _seg_lengths(bay_w, bays, 5.0);   bays = len(bw);   xs = _cum_positions(bw)
    fh = _seg_lengths(floor_h, floors, 3.0); floors = len(fh); ys = _cum_positions(fh)
    Xmax = xs[bays]
    # ── Frame (explicit nodes / bars) ──
    for fl in range(floors + 1):
        for col in range(bays + 1):
            s.add_node(f'N{col}_{fl}', xs[col], ys[fl])
    for fl in range(floors):
        for col in range(bays + 1):
            s.add_bar_element(f'C{col}_{fl}', f'N{col}_{fl}', f'N{col}_{fl+1}', 'Col')
    for fl in range(1, floors + 1):
        for col in range(bays):
            s.add_bar_element(f'B{col}_{fl}', f'N{col}_{fl}', f'N{col+1}_{fl}', 'Beam')
    for col in range(bays + 1):
        s.assign_support(f'N{col}_0', 'Fixed')
    # ── Wall (module mesh aligned to floors) ──
    nx = max(1, int(wall_nx))
    if wall_side == 'right':
        x0 = Xmax; iface_col = bays; iface_i = 0
    else:
        x0 = -float(wall_width); iface_col = 0; iface_i = nx
    wx = [x0 + wall_width * i / nx for i in range(nx + 1)]
    # Global vertical levels; each carries its floor-cote index (or None).
    cell = wall_width / nx
    levels = [(ys[0], 0)]
    for fl in range(floors):
        y_bot, y_top = ys[fl], ys[fl + 1]
        ry = max(1, round((y_top - y_bot) / cell))
        for k in range(1, ry):
            levels.append((y_bot + (y_top - y_bot) * k / ry, None))
        levels.append((y_top, fl + 1))
    node_at = {}
    for lj, (y, cote) in enumerate(levels):
        for i in range(nx + 1):
            if i == iface_i and cote is not None:
                nid = f'N{iface_col}_{cote}'          # share the frame joint
            else:
                nid = f'w{i}_{lj}'
                s.add_node(nid, wx[i], y)
            node_at[(i, lj)] = nid
    e = 0
    for lj in range(len(levels) - 1):
        for i in range(nx):
            a, bb = node_at[(i, lj)], node_at[(i + 1, lj)]
            c, d = node_at[(i + 1, lj + 1)], node_at[(i, lj + 1)]
            if prefer_quad:
                s.add_quad_element(f'we{e}', a, bb, c, d, 'Wall'); e += 1
            else:
                s.add_tri_element(f'we{e}', a, bb, c, 'Wall'); e += 1
                s.add_tri_element(f'we{e}', a, c, d, 'Wall');  e += 1
    for i in range(nx + 1):
        s.assign_support(node_at[(i, 0)], 'Fixed')    # wall base fixed
    _tpl_finish_loads(s)
    return s


def _tpl_wall_frame_obj(bays: int, floors: int, bay_w, floor_h,
                        wall_width: float, wall_nx: int,
                        b: float, h: float, E: float, g: float,
                        thickness: float, nu: float = 0.2,
                        formulation: str = 'ES-FEM', wall_side: str = 'left',
                        material_type=None, shape=None, tw: float = 0.0, tf: float = 0.0,
                        prefer_quad: bool = False) -> Structure2D:
    """Frame + shear wall as parametric objects. Frame column/beam lines weld to
    the pre-created joint nodes; the wall is one rectangle object per floor
    module, so its corners coincide with the frame joints at each floor. Wall
    meshed in triangles or quads (``prefer_quad``)."""
    s = Structure2D()
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Conc')
    s.add_material(mat_name, E, g, material_type=material_type, poisson=nu)
    s.add_section('Beam', mat_name, b, h, shape=shape, tw=tw, tf=tf)
    s.add_section('Col',  mat_name, h, b, shape=shape, tw=tw, tf=tf)
    _tf, _qf = _membrane_forms(formulation, prefer_quad)
    s.add_tri_section('Wall', mat_name, thickness=thickness, formulation=_tf)
    s.add_quad_section('Wall', mat_name, thickness=thickness, formulation=_qf)
    s.add_support('Fixed', ux=True, uy=True, tz=True)
    bw = _seg_lengths(bay_w, bays, 5.0);   bays = len(bw);   xs = _cum_positions(bw)
    fh = _seg_lengths(floor_h, floors, 3.0); floors = len(fh); ys = _cum_positions(fh)
    Xmax = xs[bays]
    for fl in range(floors + 1):
        for col in range(bays + 1):
            s.add_node(f'N{col}_{fl}', xs[col], ys[fl])
    for col in range(bays + 1):
        x = xs[col]
        s.add_geo_line(f'C{col}', x, ys[0], x, ys[floors], section_name='Col')
    for fl in range(1, floors + 1):
        y = ys[fl]
        s.add_geo_line(f'B{fl}', xs[0], y, xs[bays], y, section_name='Beam')
    nx = max(1, int(wall_nx))
    x_left = Xmax if wall_side == 'right' else -float(wall_width)
    target = wall_width / nx
    for fl in range(floors):
        y0, y1 = ys[fl], ys[fl + 1]
        s.add_geo_rectangle(f'W{fl}', (x_left, y0), (x_left + wall_width, y1),
                            section_name='Wall',
                            target_size=min(target, y1 - y0),
                            prefer_quad=prefer_quad)
    for col in range(bays + 1):
        s.assign_support(f'N{col}_0', 'Fixed')
    # Only the wall's OUTER base corner is a genuinely new node — the other
    # base corner sits on the interface edge (x=0 for wall_side='left',
    # x=Xmax for 'right') and gets welded to the frame's own base joint
    # (N0_0 or N{bays}_0), already fixed by the loop above. Referencing
    # that welded corner by its pre-weld object-relative name here (as
    # this used to do unconditionally for both p0 and p1) hits a
    # structure_io._to_dict bug: the alias is resolved to the real frame
    # node id in the exported *node list*, but not in
    # *support_assignments*, which keeps the dangling "W0.p0"/"W0.p1"
    # string — 'support_assignments: node_id=... is not in nodes' on
    # every load/verify of the exported JSON or script. Found live via the
    # xdfem2d fine-tune dataset (12 of 17 catalogue combinations for this
    # template failing this way). Fixing the *outer* corner only is both
    # correct (the interface corner is already fixed) and avoids the bug.
    outer_corner = 'W0.p1' if wall_side == 'right' else 'W0.p0'
    s.assign_support(outer_corner, 'Fixed')
    _tpl_finish_loads(s)
    return s


# ── Plate-domain templates (slabs and grillages) ────────────────────────

def _slab_grid(s, Lx, Ly, nx, ny, section, prefer_quad=False):
    """Add an nx×ny mesh over the rectangle [0,Lx]×[0,Ly] to *s*, returning
    {(i, j): node_id}. Shared by the slab templates. Each cell becomes either
    two plate-bending triangles (default) or a single quadrilateral when
    ``prefer_quad`` is set."""
    ids = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            nid = f'N{i}_{j}'
            ids[(i, j)] = nid
            s.add_node(nid, Lx * i / nx, Ly * j / ny)
    k = 0
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = (ids[(i, j)], ids[(i + 1, j)],
                          ids[(i + 1, j + 1)], ids[(i, j + 1)])
            if prefer_quad:
                s.add_quad_element(f'Q{k}', a, b, c, d, section); k += 1
            else:
                s.add_tri_element(f'T{k}', a, b, c, section); k += 1
                s.add_tri_element(f'T{k}', a, c, d, section); k += 1
    return ids


def _tpl_slab_obj(support='simply', Lx=5.0, Ly=5.0, t=0.20, nx=8, ny=8,
                  E=33e6, g=25.0, nu=0.2, pz=-5.0,
                  prefer_quad=False, formulation=None):
    """A rectangular plate slab as a rectangle **object** (auto-meshed into
    plate triangles), simply supported or clamped on all four edges under
    self-weight plus a uniform pressure pz.

    The four corner supports propagate along every edge (as in the wall object
    templates); the pressure is a surface load on the object, applied to every
    triangle it meshes into. The automatic mesher only takes a single target
    edge length, so nx/ny only set its overall fineness (the finer of the
    two directions) rather than an exact nx×ny grid — see ``_tpl_slab`` for
    that."""
    s = Structure2D(domain='plate')
    s.add_material('Mat', elastic_modulus=E, unit_weight=g, poisson=nu)
    _tf, _qf = _plate_forms(formulation, prefer_quad)
    s.add_plate_section('Slab', 'Mat', thickness=t, formulation=_tf)
    s.add_quad_section('Slab', 'Mat', thickness=t, formulation=_qf)
    target = min(Lx / max(int(nx), 1), Ly / max(int(ny), 1))
    s.add_geo_rectangle('R', (0.0, 0.0), (Lx, Ly),
                        section_name='Slab', target_size=target,
                        prefer_quad=prefer_quad)
    if support == 'clamped':
        s.add_support('CLAMPED', w=True, tx=True, ty=True); name = 'CLAMPED'
    else:
        s.add_support('SIMPLE', w=True); name = 'SIMPLE'
    for c in ('R.p0', 'R.p1', 'R.p2', 'R.p3'):   # 4 corners → all 4 edges
        s.assign_support(c, name)
    _tpl_finish_loads(s)
    _tpl_slab_pressure(s, 'R', pz)             # surface pressure on the slab
    return s


def _tpl_slab_edges_obj(edges, Lx=5.0, Ly=5.0, t=0.20, nx=6, ny=6,
                        E=33e6, g=25.0, nu=0.2, pz=-5.0,
                        prefer_quad=False, formulation=None,
                        material_type=None, design: dict | None = None):
    """A rectangular slab with a DIFFERENT support condition per edge, as a
    geometry OBJECT — via ``GeoRectangle.edge_supports``, direct per-edge
    restraint rather than the corner-intersection propagation ``_tpl_slab_obj``
    relies on (which cannot express an alternating pattern like
    clamped/simply/clamped/simply — see ``edge_supports``' own docstring in
    ``xdfem2d.models``).

    ``edges``: keys ``'left'``/``'right'``/``'bottom'``/``'top'``, each value
    ``'clamped'``, ``'simply'``, or ``'free'`` (no support on that edge at
    all — the classic three-edges-supported, one-edge-free slab). Any edge
    left out of ``edges`` defaults to ``'free'``."""
    s = Structure2D(domain='plate')
    mat_name = {'Steel': 'Steel', 'Timber': 'Timber'}.get(material_type, 'Mat')
    s.add_material(mat_name, elastic_modulus=E, unit_weight=g, poisson=nu,
                   material_type=material_type, design=design)
    _tf, _qf = _plate_forms(formulation, prefer_quad)
    s.add_plate_section('Slab', mat_name, thickness=t, formulation=_tf)
    s.add_quad_section('Slab', mat_name, thickness=t, formulation=_qf)
    edges = _normalize_edges(edges)      # accept pin/fixed aliases too
    if 'clamped' in edges.values():
        s.add_support('CLAMPED', w=True, tx=True, ty=True)
    if 'simply' in edges.values():
        s.add_support('SIMPLE', w=True)
    name_of = {'clamped': 'CLAMPED', 'simply': 'SIMPLE', 'free': None}
    target = min(Lx / max(int(nx), 1), Ly / max(int(ny), 1))
    r = s.add_geo_rectangle('R', (0.0, 0.0), (Lx, Ly),
                            section_name='Slab', target_size=target,
                            prefer_quad=prefer_quad)
    # add_geo_rectangle's own corner order is CCW from (0,0): p0 bottom-left,
    # p1 bottom-right, p2 top-right, p3 top-left — so the perimeter edges in
    # order are 0=bottom (p0-p1), 1=right (p1-p2), 2=top (p2-p3),
    # 3=left (p3-p0).
    r.edge_supports = [name_of[edges.get('bottom', 'free')],
                       name_of[edges.get('right', 'free')],
                       name_of[edges.get('top', 'free')],
                       name_of[edges.get('left', 'free')]]
    _tpl_finish_loads(s)
    _tpl_slab_pressure(s, 'R', pz)
    return s


# Slab edge condition → restraint triple (w, θx, θy) in the plate domain, and
# its canonical support name. 'free' means no support on that edge. 'pin' is
# accepted as an alias of 'simply' and 'fixed' as an alias of 'clamped'.
_EDGE_ALIAS = {'pin': 'simply', 'fixed': 'clamped',
               'simply': 'simply', 'clamped': 'clamped', 'free': 'free'}
_EDGE_TRIPLE = {'simply': (True, False, False),   # SIMPLE
                'clamped': (True, True, True)}     # CLAMPED
_EDGE_RANK = {'free': 0, 'simply': 1, 'clamped': 2}


def _normalize_edges(edges):
    """Return a {bottom,right,top,left: 'free'|'simply'|'clamped'} dict from a
    string (applied to all edges) or a partial dict."""
    keys = ('bottom', 'right', 'top', 'left')
    if isinstance(edges, str):
        edges = {k: edges for k in keys}
    return {k: _EDGE_ALIAS.get(str(edges.get(k, 'free')).lower(), 'free')
            for k in keys}


def _tpl_slab(edges='simply', Lx=5.0, Ly=5.0, t=0.20, nx=8, ny=8,
              E=33e6, g=25.0, nu=0.2, pz=-5.0, prefer_quad=False,
              formulation=None):
    """A rectangular plate slab under self-weight plus a uniform pressure pz,
    with an independent support condition on each edge.

    ``edges`` is either a string ('simply'/'pin', 'clamped'/'fixed', 'free')
    applied to all four edges, or a dict keyed 'bottom'/'right'/'top'/'left'
    with those values. Supports use the canonical plate names (SIMPLE, CLAMPED).
    Meshed as a structured nx×ny grid of plate triangles (MITC3/DKT) or quads
    (MITC4/DKT4, ``prefer_quad``); ``formulation`` picks the element."""
    from .models import canonical_support_name
    s = Structure2D(domain='plate')
    s.add_material('Mat', elastic_modulus=E, unit_weight=g, poisson=nu)
    _tf, _qf = _plate_forms(formulation, prefer_quad)
    s.add_plate_section('Slab', 'Mat', thickness=t, formulation=_tf)
    s.add_quad_section('Slab', 'Mat', thickness=t, formulation=_qf)
    ids = _slab_grid(s, Lx, Ly, nx, ny, 'Slab', prefer_quad=prefer_quad)
    ed = _normalize_edges(edges)
    # A boundary node may sit on two edges (a corner) — the stronger condition
    # (clamped > simply > free) wins.
    for (i, j), nid in ids.items():
        kinds = []
        if j == 0:  kinds.append(ed['bottom'])
        if j == ny: kinds.append(ed['top'])
        if i == 0:  kinds.append(ed['left'])
        if i == nx: kinds.append(ed['right'])
        if not kinds:
            continue
        best = max(kinds, key=lambda k: _EDGE_RANK[k])
        if best == 'free':
            continue
        w_, tx_, ty_ = _EDGE_TRIPLE[best]
        name = canonical_support_name(w_, tx_, ty_, domain='plate')
        if name not in s.supports:
            s.add_support(name, w=w_, tx=tx_, ty=ty_)
        s.assign_support(nid, name)
    _tpl_finish_loads(s)
    area_ids = (list(s.quad_elements_by_id) if prefer_quad
                else list(s.tri_elements_by_id))
    for eid in area_ids:
        _tpl_slab_pressure(s, eid, pz)
    return s


def _tpl_slab_ribbed(Lx=5.0, Ly=5.0, t=0.15, nx=8, ny=8, b=0.30, h=0.50,
                     E=33e6, g=25.0, nu=0.2, pz=-5.0, prefer_quad=False,
                     formulation=None):
    """A plate slab stiffened by edge beams (grillage) on its four sides, resting
    on the four corners — the beam/slab interaction of a rib-stiffened slab.
    Meshed as a structured nx×ny grid of plate triangles (MITC3/DKT) or quads
    (MITC4/DKT4, ``prefer_quad``)."""
    s = Structure2D(domain='plate')
    s.add_material('Mat', elastic_modulus=E, unit_weight=g, poisson=nu)
    _tf, _qf = _plate_forms(formulation, prefer_quad)
    s.add_plate_section('Slab', 'Mat', thickness=t, formulation=_tf)
    s.add_quad_section('Slab', 'Mat', thickness=t, formulation=_qf)
    s.add_section('Rib', 'Mat', b=b, h=h, shape='Rectangular')
    ids = _slab_grid(s, Lx, Ly, nx, ny, 'Slab', prefer_quad=prefer_quad)
    # Edge beams: consecutive boundary nodes along each of the four sides.
    edge = ([ids[(i, 0)] for i in range(nx + 1)]        # bottom
            + [ids[(nx, j)] for j in range(1, ny + 1)]   # right
            + [ids[(i, ny)] for i in range(nx - 1, -1, -1)]  # top
            + [ids[(0, j)] for j in range(ny - 1, 0, -1)])  # left (back to start)
    e = 0
    for u, v in zip(edge, edge[1:] + edge[:1]):
        s.add_bar_element(f'B{e}', u, v, 'Rib'); e += 1
    s.add_support('SIMPLE', w=True)
    for corner in ((0, 0), (nx, 0), (nx, ny), (0, ny)):
        s.assign_support(ids[corner], 'SIMPLE')
    _tpl_finish_loads(s)
    area_ids = (list(s.quad_elements_by_id) if prefer_quad
                else list(s.tri_elements_by_id))
    for eid in area_ids:
        _tpl_slab_pressure(s, eid, pz)
    return s


def _tpl_beam_grid(Lx=6.0, Ly=6.0, nx=3, ny=3, b=0.30, h=0.50,
                   E=33e6, g=25.0, nu=0.2):
    """An orthogonal grillage of beams (nx×ny bays), simply supported on the
    perimeter, under self-weight."""
    s = Structure2D(domain='plate')
    s.add_material('Mat', elastic_modulus=E, unit_weight=g, poisson=nu)
    s.add_section('Beam', 'Mat', b=b, h=h, shape='Rectangular')
    ids = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            nid = f'N{i}_{j}'
            ids[(i, j)] = nid
            s.add_node(nid, Lx * i / nx, Ly * j / ny)
    e = 0
    for j in range(ny + 1):                     # beams along x
        for i in range(nx):
            s.add_bar_element(f'B{e}', ids[(i, j)], ids[(i + 1, j)], 'Beam'); e += 1
    for i in range(nx + 1):                     # beams along y
        for j in range(ny):
            s.add_bar_element(f'B{e}', ids[(i, j)], ids[(i, j + 1)], 'Beam'); e += 1
    s.add_support('SIMPLE', w=True)
    for (i, j), nid in ids.items():
        if i in (0, nx) or j in (0, ny):
            s.assign_support(nid, 'SIMPLE')
    _tpl_finish_loads(s)
    return s


def _grid_coords(lines, max_size):
    """Subdivide each span between consecutive *lines* into equal parts no
    larger than *max_size*, returning (coords, line_index): the full sorted
    coordinate list (every line kept exactly) and the index of each line
    within it — so features anchored on the lines (e.g. columns) land on nodes.
    """
    coords = [float(lines[0])]
    line_index = [0]
    for a, b in zip(lines, lines[1:]):
        span = float(b) - float(a)
        m = max(1, int(math.ceil(span / max_size))) if max_size > 0 else 1
        for k in range(1, m + 1):
            coords.append(a + span * k / m)
        line_index.append(len(coords) - 1)
    return coords, line_index


def _slab_grid_xy(s, xs, ys, section, prefer_quad=False):
    """Structured plate mesh over the explicit coordinate arrays *xs*×*ys*
    (need not be uniform), returning {(i, j): node_id}. Each cell becomes two
    triangles or one quad (``prefer_quad``)."""
    nx, ny = len(xs) - 1, len(ys) - 1
    ids = {}
    for i, x in enumerate(xs):
        for j, y in enumerate(ys):
            nid = f'N{i}_{j}'
            ids[(i, j)] = nid
            s.add_node(nid, x, y)
    k = 0
    for i in range(nx):
        for j in range(ny):
            a, b, c, d = (ids[(i, j)], ids[(i + 1, j)],
                          ids[(i + 1, j + 1)], ids[(i, j + 1)])
            if prefer_quad:
                s.add_quad_element(f'Q{k}', a, b, c, d, section); k += 1
            else:
                s.add_tri_element(f'T{k}', a, b, c, section); k += 1
                s.add_tri_element(f'T{k}', a, c, d, section); k += 1
    return ids


def _tpl_flat_slab(Lx=8.0, Ly=8.0, ncx=2, ncy=2, t=0.22, max_size=0.5,
                   E=33e6, g=25.0, nu=0.2, pz=-5.0, prefer_quad=False,
                   formulation=None):
    """A flat slab carried on a grid of columns — no edge beams. ``Lx``/``Ly``
    are the **bay** span in each direction: a single value (uniform ``ncx``/
    ``ncy`` bays) or a list of per-bay spans (which then set the bay count).
    Columns sit exactly on every bay line; the mesh subdivides each bay into
    plate elements no larger than ``max_size``. Self-weight plus a pressure pz.
    Meshed as MITC3/DKT triangles or MITC4/DKT4 quads (``prefer_quad``)."""
    s = Structure2D(domain='plate')
    s.add_material('Mat', elastic_modulus=E, unit_weight=g, poisson=nu)
    _tf, _qf = _plate_forms(formulation, prefer_quad)
    s.add_plate_section('Slab', 'Mat', thickness=t, formulation=_tf)
    s.add_quad_section('Slab', 'Mat', thickness=t, formulation=_qf)
    bx = _seg_lengths(Lx, ncx); ncx = len(bx); xlines = _cum_positions(bx)
    by = _seg_lengths(Ly, ncy); ncy = len(by); ylines = _cum_positions(by)
    xs, xi = _grid_coords(xlines, max_size)
    ys, yi = _grid_coords(ylines, max_size)
    ids = _slab_grid_xy(s, xs, ys, 'Slab', prefer_quad=prefer_quad)
    s.add_support('COL', w=True)
    for ci in range(ncx + 1):
        for cj in range(ncy + 1):
            s.assign_support(ids[(xi[ci], yi[cj])], 'COL')
    _tpl_finish_loads(s)
    area_ids = (list(s.quad_elements_by_id) if prefer_quad
                else list(s.tri_elements_by_id))
    for eid in area_ids:
        _tpl_slab_pressure(s, eid, pz)
    return s


def _tpl_slab_on_grade(Lx=6.0, Ly=6.0, t=0.25, nx=12, ny=12, kz=30000.0,
                       E=33e6, g=25.0, nu=0.2, pz=-20.0, prefer_quad=False,
                       formulation=None):
    """A slab on grade: a plate slab resting on a Winkler elastic foundation
    (subgrade modulus kz [kN/m³]) with no other supports, under self-weight plus
    a pressure pz. The springs alone carry the slab. Meshed as a structured
    nx×ny grid of plate triangles (MITC3/DKT) or quads (MITC4/DKT4)."""
    s = Structure2D(domain='plate')
    s.add_material('Mat', elastic_modulus=E, unit_weight=g, poisson=nu)
    _tf, _qf = _plate_forms(formulation, prefer_quad)
    s.add_plate_section('Slab', 'Mat', thickness=t, formulation=_tf)
    s.add_quad_section('Slab', 'Mat', thickness=t, formulation=_qf)
    _slab_grid(s, Lx, Ly, nx, ny, 'Slab', prefer_quad=prefer_quad)
    _tpl_finish_loads(s)
    area_ids = (list(s.quad_elements_by_id) if prefer_quad
                else list(s.tri_elements_by_id))
    for eid in area_ids:
        s.add_area_spring(eid, kz=kz)
        _tpl_slab_pressure(s, eid, pz)
    return s


def _tpl_bridge_grillage(L=20.0, W=10.0, ng=4, ncross=8, b=0.40, h=1.20,
                         bt=0.30, ht=0.60, E=33e6, g=25.0, nu=0.2):
    """A bridge-deck grillage: ng longitudinal girders spanning L, tied by
    transverse members at ncross stations, simply supported (w) on the two
    abutment ends. Girders b×h, transverse members bt×ht. Self-weight."""
    s = Structure2D(domain='plate')
    s.add_material('Mat', elastic_modulus=E, unit_weight=g, poisson=nu)
    s.add_section('Girder', 'Mat', b=b, h=h, shape='Rectangular')
    s.add_section('Cross', 'Mat', b=bt, h=ht, shape='Rectangular')
    ids = {}
    for i in range(ncross + 1):
        for gi in range(ng):
            nid = f'N{i}_{gi}'
            ids[(i, gi)] = nid
            s.add_node(nid, L * i / ncross, W * gi / (ng - 1) if ng > 1 else 0.0)
    e = 0
    for gi in range(ng):                        # longitudinal girders
        for i in range(ncross):
            s.add_bar_element(f'G{e}', ids[(i, gi)], ids[(i + 1, gi)], 'Girder')
            e += 1
    for i in range(ncross + 1):                 # transverse diaphragms
        for gi in range(ng - 1):
            s.add_bar_element(f'C{e}', ids[(i, gi)], ids[(i, gi + 1)], 'Cross')
            e += 1
    s.add_support('SIMPLE', w=True)
    for gi in range(ng):                        # both abutment ends
        s.assign_support(ids[(0, gi)], 'SIMPLE')
        s.assign_support(ids[(ncross, gi)], 'SIMPLE')
    _tpl_finish_loads(s)
    return s


def _sector_boundary(rmin, rmax, theta_deg, hs):
    """Faceted closed boundary polygon of an annular sector, with facets about
    *hs* long, plus (has_center, full_disc) flags.

    A full disc is a **single closed outer arc** — no radial seam, so there are
    no duplicated coincident boundary nodes. A solid sector adds the two radial
    edges down to the centre; an annulus adds the inner arc."""
    theta = math.radians(theta_deg)
    has_center = rmin <= 1e-12
    full_disc = abs(theta_deg - 360.0) <= 1e-6

    def pt(r, a):
        return (r * math.cos(a), r * math.sin(a))

    def arc(r, a0, a1):
        m = max(1, round(abs(r * (a1 - a0)) / hs)) if hs > 0 else 1
        return [pt(r, a0 + (a1 - a0) * k / m) for k in range(m + 1)]

    def rad(a, r0, r1):
        m = max(1, round(abs(r1 - r0) / hs)) if hs > 0 else 1
        return [pt(r0 + (r1 - r0) * k / m, a) for k in range(m + 1)]

    if full_disc:
        verts = arc(rmax, 0.0, 2.0 * math.pi)
    elif has_center:                             # pie slice to the centre
        verts = rad(0.0, 0.0, rmax) + arc(rmax, 0.0, theta) + rad(theta, rmax, 0.0)
    else:                                        # annular sector
        verts = (arc(rmax, 0.0, theta) + rad(theta, rmax, rmin)
                 + arc(rmin, theta, 0.0) + rad(0.0, rmin, rmax))
    clean = []
    for p in verts:
        if (not clean or abs(clean[-1][0] - p[0]) > 1e-9
                or abs(clean[-1][1] - p[1]) > 1e-9):
            clean.append(p)
    if (len(clean) > 1 and abs(clean[0][0] - clean[-1][0]) < 1e-9
            and abs(clean[0][1] - clean[-1][1]) < 1e-9):
        clean.pop()
    return clean, has_center, full_disc


def _assign_sector_supports(s, bnodes, rmin, rmax, theta_deg,
                            outer, inner, radial):
    """Assign per-edge supports to the boundary nodes of a sector — the outer
    arc, the inner arc (annulus only) and the two radial (side) edges — each
    ``free``/``pin``/``fixed`` (canonical plate names SIMPLE/CLAMPED). A node on
    two edges (a corner) takes the stronger condition (fixed > pin > free).

    ``bnodes`` is an iterable of (node_id, (x, y)) — only the boundary nodes."""
    from .models import canonical_support_name
    theta = math.radians(theta_deg)
    st, ct = math.sin(theta), math.cos(theta)
    has_center = rmin <= 1e-12
    full_disc = abs(theta_deg - 360.0) <= 1e-6
    tol = 1e-6 * max(1.0, rmax)
    kind = {'outer': _EDGE_ALIAS.get(str(outer).lower(), 'free'),
            'inner': _EDGE_ALIAS.get(str(inner).lower(), 'free'),
            'radial': _EDGE_ALIAS.get(str(radial).lower(), 'free')}

    def _name(k):
        trip = _EDGE_TRIPLE.get(k)
        if trip is None:
            return None
        nm = canonical_support_name(*trip, domain='plate')
        if nm not in s.supports:
            s.add_support(nm, w=trip[0], tx=trip[1], ty=trip[2])
        return nm

    for nid, (x, y) in bnodes:
        r = math.hypot(x, y)
        on = []
        if abs(r - rmax) < tol:
            on.append('outer')
        if not has_center and abs(r - rmin) < tol:
            on.append('inner')
        if not full_disc and ((abs(y) < tol and x > -tol) or
                              (abs(x * st - y * ct) < tol and x * ct + y * st > -tol)):
            on.append('radial')
        if not on:
            continue
        best = max((kind[e] for e in on), key=lambda k: _EDGE_RANK[k])
        if best == 'free':
            continue
        nm = _name(best)
        if nm:
            s.assign_support(nid, nm)


def _tpl_slab_sector(rmin=0.0, rmax=5.0, theta_deg=90.0, max_size=0.5, t=0.20,
                     E=33e6, g=25.0, nu=0.2, pz=-5.0,
                     outer='pin', inner='free', radial='free'):
    """A circular slab sector (rmin..rmax, angle theta_deg) with an independent
    support condition on each edge — the outer arc, the inner arc (annulus only)
    and the two radial sides — each ``free``/``pin``/``fixed`` (canonical plate
    names). Self-weight plus a pressure pz. rmin = 0 gives a solid sector; a full
    disc (theta = 360) has only the outer edge.

    The curved boundary is faceted at ~``max_size`` and the interior filled by
    the general unstructured (Delaunay) mesher — the same one the object variant
    uses — avoiding the sliver triangles a naive polar grid produces near the
    centre and for narrow/annular geometries. A full disc is a single closed
    outline, so it has no duplicated radial seam."""
    from .meshing import mesh_polygon, _dist_point_seg
    s = Structure2D(domain='plate')
    s.add_material('Mat', elastic_modulus=E, unit_weight=g, poisson=nu)
    s.add_plate_section('Slab', 'Mat', thickness=t)
    s.add_quad_section('Slab', 'Mat', thickness=t, formulation='MITC4')
    hs = max(1e-9, float(max_size))
    poly, has_center, full_disc = _sector_boundary(rmin, rmax, theta_deg, hs)
    points, tris = mesh_polygon(poly, hs * 1.15)
    nid_of = {}
    for idx, (x, y) in enumerate(points):
        nid = f'N{idx}'
        s.add_node(nid, x, y); nid_of[idx] = nid
    for e, (a, b, c) in enumerate(tris):
        s.add_tri_element(f'T{e}', nid_of[a], nid_of[b], nid_of[c], 'Slab')
    # A node is on the boundary when it lies on a polygon edge (interior grid
    # nodes are kept clear of it) — those get the per-edge support.
    tol = 1e-6 * max(1.0, rmax)
    n = len(poly)
    bnodes = [(nid_of[idx], (x, y)) for idx, (x, y) in enumerate(points)
              if any(_dist_point_seg(x, y, poly[i][0], poly[i][1],
                                     poly[(i + 1) % n][0],
                                     poly[(i + 1) % n][1])[0] < tol
                     for i in range(n))]
    _assign_sector_supports(s, bnodes, rmin, rmax, theta_deg,
                            outer, inner, radial)
    _tpl_finish_loads(s)
    for tid in list(s.tri_elements_by_id):
        _tpl_slab_pressure(s, tid, pz)
    return s


def _tpl_slab_ribbed_obj(Lx=5.0, Ly=5.0, t=0.15, nx=8, ny=8, b=0.30, h=0.50,
                         E=33e6, g=25.0, nu=0.2, pz=-5.0,
                         prefer_quad=False, formulation=None):
    """A rib-stiffened slab as **objects**: a rectangle surface object (the slab,
    auto-meshed into plate triangles) framed by four line objects (the edge
    beams). The slab rests on its four corners; the pressure pz is a surface load
    on the rectangle. The rib lines are named ``B*`` so they expand before the
    rectangle, letting the slab mesh conform to the rib nodes and weld to them.
    The automatic mesher only takes a single target edge length, so nx/ny only
    set its overall fineness — see ``_tpl_slab_ribbed`` for an exact nx×ny grid."""
    s = Structure2D(domain='plate')
    s.add_material('Mat', elastic_modulus=E, unit_weight=g, poisson=nu)
    _tf, _qf = _plate_forms(formulation, prefer_quad)
    s.add_plate_section('Slab', 'Mat', thickness=t, formulation=_tf)
    s.add_quad_section('Slab', 'Mat', thickness=t, formulation=_qf)
    s.add_section('Rib', 'Mat', b=b, h=h, shape='Rectangular')
    target = min(Lx / max(int(nx), 1), Ly / max(int(ny), 1))
    s.add_geo_line('B0', 0.0, 0.0, Lx, 0.0, section_name='Rib', divisions=nx)  # bottom
    s.add_geo_line('B1', Lx, 0.0, Lx, Ly, section_name='Rib', divisions=ny)    # right
    s.add_geo_line('B2', Lx, Ly, 0.0, Ly, section_name='Rib', divisions=nx)    # top
    s.add_geo_line('B3', 0.0, Ly, 0.0, 0.0, section_name='Rib', divisions=ny)  # left
    s.add_geo_rectangle('R', (0.0, 0.0), (Lx, Ly),
                        section_name='Slab', target_size=target,
                        prefer_quad=prefer_quad)
    s.add_support('SIMPLE', w=True)
    # The rectangle reuses the rib endpoint nodes at the four corners (created
    # first), so restrain them via the object's own defining nodes.
    for c in s.geometry_objects['R'].node_ids:
        s.assign_support(c, 'SIMPLE')
    _tpl_finish_loads(s)
    _tpl_slab_pressure(s, 'R', pz)
    return s


def _tpl_flat_slab_obj(Lx=8.0, Ly=8.0, ncx=2, ncy=2, t=0.22, max_size=0.5,
                       E=33e6, g=25.0, nu=0.2, pz=-5.0,
                       prefer_quad=False, formulation=None):
    """A flat (mushroom) slab as a grid of rectangle **objects** — one panel per
    bay — carried on point columns at the panel corners. ``Lx``/``Ly`` are the
    per-bay span (single value or a list of variable spans); each panel meshes
    into plate elements no larger than ``max_size``. The column nodes are
    created first and given a w-only support; the panels reuse them by
    coincidence. Each panel's ``edge_support_mode`` is ``"none"`` so the column
    restraint stays a point support and does not run along the panel edges."""
    s = Structure2D(domain='plate')
    s.add_material('Mat', elastic_modulus=E, unit_weight=g, poisson=nu)
    _tf, _qf = _plate_forms(formulation, prefer_quad)
    s.add_plate_section('Slab', 'Mat', thickness=t, formulation=_tf)
    s.add_quad_section('Slab', 'Mat', thickness=t, formulation=_qf)
    bx = _seg_lengths(Lx, ncx); ncx = len(bx); xlines = _cum_positions(bx)
    by = _seg_lengths(Ly, ncy); ncy = len(by); ylines = _cum_positions(by)
    s.add_support('COL', w=True)
    for ci in range(ncx + 1):                    # column nodes = panel corners
        for cj in range(ncy + 1):
            nid = f'C{ci}_{cj}'
            s.add_node(nid, xlines[ci], ylines[cj])
            s.assign_support(nid, 'COL')
    for ci in range(ncx):                        # one rectangle object per bay
        for cj in range(ncy):
            pid = f'P{ci}_{cj}'
            r = s.add_geo_rectangle(
                pid, (xlines[ci], ylines[cj]), (xlines[ci + 1], ylines[cj + 1]),
                section_name='Slab', target_size=max_size,
                prefer_quad=prefer_quad)
            r.edge_support_mode = 'none'         # columns stay point supports
            _tpl_slab_pressure(s, pid, pz)
    _tpl_finish_loads(s)
    return s


def _tpl_slab_on_grade_obj(Lx=6.0, Ly=6.0, t=0.25, nx=12, ny=12, kz=30000.0,
                           E=33e6, g=25.0, nu=0.2, pz=-20.0,
                           prefer_quad=False, formulation=None):
    """A slab on grade as a rectangle **object** resting on a Winkler foundation:
    a surface area spring (subgrade modulus kz) is attached to the object, so
    every triangle it meshes into carries the elastic support; the springs alone
    equilibrate the self-weight plus pressure pz (no other supports). The
    automatic mesher only takes a single target edge length, so nx/ny only
    set its overall fineness — see ``_tpl_slab_on_grade`` for an exact nx×ny
    grid."""
    s = Structure2D(domain='plate')
    s.add_material('Mat', elastic_modulus=E, unit_weight=g, poisson=nu)
    _tf, _qf = _plate_forms(formulation, prefer_quad)
    s.add_plate_section('Slab', 'Mat', thickness=t, formulation=_tf)
    s.add_quad_section('Slab', 'Mat', thickness=t, formulation=_qf)
    target = min(Lx / max(int(nx), 1), Ly / max(int(ny), 1))
    s.add_geo_rectangle('R', (0.0, 0.0), (Lx, Ly),
                        section_name='Slab', target_size=target,
                        prefer_quad=prefer_quad)
    _tpl_finish_loads(s)
    s.add_area_spring('R', kz=kz)                # elastic foundation on the object
    _tpl_slab_pressure(s, 'R', pz)
    return s


def _tpl_slab_sector_obj(rmin=0.0, rmax=5.0, theta_deg=90.0, max_size=0.5, t=0.20,
                         E=33e6, g=25.0, nu=0.2, pz=-5.0,
                         outer='pin', inner='free', radial='free'):
    """A circular slab sector as a **polygon object**: the curved boundary is
    faceted at ~``max_size`` and auto-meshed. Each edge — outer arc, inner arc
    (annulus only) and the two radial sides — carries an independent
    ``free``/``pin``/``fixed`` support (canonical plate names), under self-weight
    plus a pressure pz. rmin = 0 gives a solid sector; theta_deg = 360 a full
    disc (a single closed outline, no radial seam).

    The mesh target is set just above the facet size so the mesher never splits
    a boundary edge — which would otherwise plant a midpoint node and leave a
    flat (zero-area) triangle along the boundary."""
    s = Structure2D(domain='plate')
    s.add_material('Mat', elastic_modulus=E, unit_weight=g, poisson=nu)
    s.add_plate_section('Slab', 'Mat', thickness=t)
    s.add_quad_section('Slab', 'Mat', thickness=t, formulation='MITC4')
    hs = max(1e-9, float(max_size))
    poly, _has_center, _full_disc = _sector_boundary(rmin, rmax, theta_deg, hs)
    s.add_geo_polygon('R', poly, section_name='Slab', target_size=hs * 1.15)
    bnodes = [(nid, (s.nodes[nid].x, s.nodes[nid].y))
              for nid in s.geometry_objects['R'].node_ids]
    _assign_sector_supports(s, bnodes, rmin, rmax, theta_deg,
                            outer, inner, radial)
    _tpl_finish_loads(s)
    _tpl_slab_pressure(s, 'R', pz)
    return s


def _tpl_beam_grid_obj(Lx=6.0, Ly=6.0, nx=3, ny=3, b=0.30, h=0.50,
                       E=33e6, g=25.0, nu=0.2):
    """An orthogonal grillage as line **objects**, modelled like the rectangular
    frame object: every grid node is created first, then one full-length line per
    grid row (along x) and per grid column (along y) welds to those nodes by
    coincidence. Simply supported (w) on the perimeter, under self-weight."""
    s = Structure2D(domain='plate')
    s.add_material('Mat', elastic_modulus=E, unit_weight=g, poisson=nu)
    s.add_section('Beam', 'Mat', b=b, h=h, shape='Rectangular')
    ids = {}
    for i in range(nx + 1):
        for j in range(ny + 1):
            nid = f'N{i}_{j}'
            ids[(i, j)] = nid
            s.add_node(nid, Lx * i / nx, Ly * j / ny)
    for j in range(ny + 1):                      # one line per row (welds along x)
        y = Ly * j / ny
        s.add_geo_line(f'X{j}', 0.0, y, Lx, y, section_name='Beam', divisions=nx)
    for i in range(nx + 1):                      # one line per column (welds along y)
        x = Lx * i / nx
        s.add_geo_line(f'Y{i}', x, 0.0, x, Ly, section_name='Beam', divisions=ny)
    s.add_support('SIMPLE', w=True)
    for (i, j), nid in ids.items():
        if i in (0, nx) or j in (0, ny):
            s.assign_support(nid, 'SIMPLE')
    _tpl_finish_loads(s)
    return s


def _tpl_bridge_grillage_obj(L=20.0, W=10.0, ng=4, ncross=8, b=0.40, h=1.20,
                             bt=0.30, ht=0.60, E=33e6, g=25.0, nu=0.2):
    """A bridge-deck grillage as line **objects**, modelled like the rectangular
    frame object: every deck node is created first, then one full-length line per
    longitudinal girder and one across the deck at each transverse station weld to
    those nodes by coincidence. Simply supported (w) at both abutment ends, under
    self-weight. Girders b×h, transverse members bt×ht."""
    s = Structure2D(domain='plate')
    s.add_material('Mat', elastic_modulus=E, unit_weight=g, poisson=nu)
    s.add_section('Girder', 'Mat', b=b, h=h, shape='Rectangular')
    s.add_section('Cross', 'Mat', b=bt, h=ht, shape='Rectangular')
    ids = {}
    for i in range(ncross + 1):
        for gi in range(ng):
            nid = f'N{i}_{gi}'
            ids[(i, gi)] = nid
            s.add_node(nid, L * i / ncross, W * gi / (ng - 1) if ng > 1 else 0.0)
    for gi in range(ng):                         # longitudinal girders
        y = W * gi / (ng - 1) if ng > 1 else 0.0
        s.add_geo_line(f'G{gi}', 0.0, y, L, y, section_name='Girder', divisions=ncross)
    for i in range(ncross + 1):                  # transverse diaphragms
        x = L * i / ncross
        s.add_geo_line(f'C{i}', x, 0.0, x, W, section_name='Cross', divisions=ng - 1)
    s.add_support('SIMPLE', w=True)
    for gi in range(ng):                         # both abutment ends
        s.assign_support(ids[(0, gi)], 'SIMPLE')
        s.assign_support(ids[(ncross, gi)], 'SIMPLE')
    _tpl_finish_loads(s)
    return s


# Public aliases ---------------------------------------------------------
default_structure = _default_new_structure
beam              = _tpl_beam
beam_obj          = _tpl_beam_obj
continuous_beam_obj = _tpl_continuous_beam_obj
frame_obj         = _tpl_frame_obj
arch_obj          = _tpl_arch_obj
wall_obj          = _tpl_wall_obj
wall              = _tpl_wall
wall_beam         = _tpl_wall_beam
wall_beam_obj     = _tpl_wall_beam_obj
wall_frame        = _tpl_wall_frame
wall_frame_obj    = _tpl_wall_frame_obj
portal_frame      = _tpl_portal_frame
continuous_beam   = _tpl_continuous_beam
frame             = _tpl_frame
truss_warren      = _tpl_truss_warren
truss_howe        = _tpl_truss_howe
truss_pratt       = _tpl_truss_pratt
truss_long        = _tpl_truss_long
truss_town        = _tpl_truss_town
truss_kingpost    = _tpl_truss_kingpost
arch              = _tpl_arch
slab              = _tpl_slab
slab_obj          = _tpl_slab_obj
slab_edges_obj    = _tpl_slab_edges_obj
slab_ribbed       = _tpl_slab_ribbed
slab_ribbed_obj   = _tpl_slab_ribbed_obj
beam_grid         = _tpl_beam_grid
beam_grid_obj     = _tpl_beam_grid_obj
flat_slab         = _tpl_flat_slab
flat_slab_obj     = _tpl_flat_slab_obj
slab_on_grade     = _tpl_slab_on_grade
slab_on_grade_obj = _tpl_slab_on_grade_obj
bridge_grillage   = _tpl_bridge_grillage
bridge_grillage_obj = _tpl_bridge_grillage_obj
slab_sector       = _tpl_slab_sector
slab_sector_obj   = _tpl_slab_sector_obj
