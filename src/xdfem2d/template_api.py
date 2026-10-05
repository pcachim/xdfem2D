"""A standard structure in one call: the template builders, with plain parameters.

``xdfem2d.templates`` holds the parametric builders behind the application's
"New from template" dialog. They take what a dialog collects (an elastic modulus
and a unit weight, a section as b and h, a number of divisions) and return a new
:class:`Structure2D`. This module is the same set as one call a script or the
assistant can write on the model that is already open:

    model.create_from_template('continuous_beam', spans=[4, 5, 6])

What changes for the caller:

- the material is a name, not numbers: ``material='C30/37'`` (concrete),
  ``material='S275'`` (steel) or ``material='C24'`` (timber) -- all through
  the one ``material`` parameter, resolved the same way as a single
  ``create_bar_section`` call; a steel catalogue cross-section is
  ``profile='IPE300'`` (its grade is still ``material``);
- the lengths are the ones the problem states: ``spans=[4, 5, 6]`` are the three
  spans, and the coordinates (0, 4, 9, 15) are worked out here;
- a linear structure comes as bars and nodes, an area (a wall, a slab) as one
  geometry object that is meshed when the model is solved, the way the
  assistant's own ``create_polygon`` makes it;
- it fills an EMPTY model only. With something already in the model the
  incremental calls are the way, so ids and definitions of the two never mix.

The kinds and their parameters are in :data:`TEMPLATE_KINDS`; everything here
is validated before anything is built, and an error says what the call takes.
"""
from __future__ import annotations

import copy
import math

# Names a caller may plausibly write for a kind, mapped to the kind.
_ALIASES = {
    'simply_supported_beam': 'beam', 'simple_beam': 'beam', 'single_span_beam': 'beam',
    'cantilever': 'beam', 'continuous': 'continuous_beam',
    'multi_span_beam': 'continuous_beam', 'continuous_girder': 'continuous_beam',
    'portal': 'portal_frame', 'frame_portal': 'portal_frame',
    'building_frame': 'frame', 'multi_storey_frame': 'frame', 'rectangular_frame': 'frame',
    'warren': 'truss_warren', 'howe': 'truss_howe', 'pratt': 'truss_pratt',
    'long': 'truss_long', 'town': 'truss_town', 'kingpost': 'truss_kingpost',
    'king_post': 'truss_kingpost', 'king_post_truss': 'truss_kingpost',
    'warren_truss': 'truss_warren', 'howe_truss': 'truss_howe',
    'pratt_truss': 'truss_pratt', 'kingpost_truss': 'truss_kingpost',
    'town_truss': 'truss_town', 'long_truss': 'truss_long',
    'shear_wall': 'wall', 'wall_panel': 'wall', 'wall_panel_fixed': 'wall',
    'rectangular_slab': 'slab', 'slab_panel': 'slab', 'plate': 'slab',
    'deep_beam': 'wall_beam', 'wall_deep_beam': 'wall_beam',
    'wall_and_frame': 'wall_frame', 'frame_and_wall': 'wall_frame',
    'frame_wall': 'wall_frame', 'mixed': 'wall_frame', 'wall_plus_frame': 'wall_frame',
    'ribbed_slab': 'slab_ribbed', 'rib_slab': 'slab_ribbed', 'rib_stiffened_slab': 'slab_ribbed',
    'flat': 'flat_slab', 'mushroom_slab': 'flat_slab', 'flat_slab_on_columns': 'flat_slab',
    'winkler': 'slab_on_grade', 'slab_winkler': 'slab_on_grade', 'ground_slab': 'slab_on_grade',
    'circular_sector': 'slab_sector', 'sector': 'slab_sector', 'circular_slab': 'slab_sector',
    'grid': 'beam_grid', 'grillage': 'beam_grid', 'beam_grillage': 'beam_grid',
    'bridge': 'bridge_grillage', 'bridge_deck': 'bridge_grillage',
}

# Section and material parameters shared by the linear kinds. One 'material'
# name serves concrete, steel and timber alike -- whatever Structure2D's own
# _auto_material_from_class recognises (a concrete class, a steel grade or a
# timber grade) -- rather than a separate parameter per family. A catalogue
# 'profile' still needs its own axis: it replaces b/h with a real rolled
# section's shape, but its grade is 'material' too, not a fourth parameter.
_BAR_SECTION = {
    'b': (0.30, "section width [m]"),
    'h': (0.50, "section height [m]"),
    'material': ('C30/37', "material class: a concrete, steel or timber "
                "Eurocode grade (e.g. 'C30/37', 'S275', 'C24')"),
    'profile': (None, "steel catalogue profile, e.g. 'IPE300': replaces b "
               "and h; material is then its steel grade"),
}
_AREA_MATERIAL = {
    'material': ('C30/37', "material class: a concrete, steel or timber "
                "Eurocode grade (e.g. 'C30/37', 'S275', 'C24')"),
}

# kind -> label, family ('bar' or 'area'), what it is, parameters (name ->
# (default, meaning)). The default is what the call uses when the name is left
# out. Read by the validation, by the docstring of create_from_template and by
# the assistant's step that lists the kinds.
TEMPLATE_KINDS: dict = {
    'beam': dict(
        label='Beam', family='bar',
        doc="one span, cut into bars; support is 'simply' (pin and roller), "
            "'cantilever' (fixed at the left end), 'fixed' (both ends) or "
            "'propped' (fixed left, roller right)",
        params={'span': (6.0, "length [m]"), 'support': ('simply', "end conditions"),
                'divisions': (4, "bars along the span"), **_BAR_SECTION}),
    'continuous_beam': dict(
        label='Continuous beam', family='bar',
        doc="spans in a line, one bar per span and a node at every support; "
            "spans is the list of span lengths (or their number, with span "
            "their common length); left, right and intermediate are 'free', "
            "'pin' or 'fixed', by default 'pin' (a support at every node)",
        params={'spans': (2, "list of span lengths [m], or their number"),
                'span': (6.0, "common span length [m], when spans is a number"),
                'left': ('pin', "support at the left end"),
                'intermediate': ('pin', "support at the inner nodes"),
                'right': ('pin', "support at the right end"), **_BAR_SECTION}),
    'portal_frame': dict(
        label='Portal frame', family='bar',
        doc="two columns and a beam, fixed at the bases",
        params={'span': (6.0, "beam span [m]"), 'height': (4.0, "column height [m]"),
                **_BAR_SECTION}),
    'frame': dict(
        label='Frame', family='bar',
        doc="a multi-bay, multi-storey frame, fixed at the bases; bay_width and "
            "floor_height are a length or a list of them",
        params={'bays': (1, "number of bays"), 'floors': (1, "number of storeys"),
                'bay_width': (5.0, "bay width [m] or list"),
                'floor_height': (3.0, "storey height [m] or list"), **_BAR_SECTION}),
    'truss_warren': dict(
        label='Warren truss', family='bar', doc="a Warren truss, pinned at both ends",
        params={'panels': (6, "number of panels"), 'span': (16.0, "span [m]"),
                'depth': (2.0, "depth [m]"), **_BAR_SECTION}),
    'truss_howe': dict(
        label='Howe truss', family='bar', doc="a Howe truss, pinned at both ends",
        params={'panels': (6, "number of panels"), 'span': (16.0, "span [m]"),
                'depth': (2.0, "depth [m]"), **_BAR_SECTION}),
    'truss_pratt': dict(
        label='Pratt truss', family='bar', doc="a Pratt truss, pinned at both ends",
        params={'panels': (6, "number of panels"), 'span': (16.0, "span [m]"),
                'depth': (2.0, "depth [m]"), **_BAR_SECTION}),
    'truss_long': dict(
        label='Long truss', family='bar', doc="an X-braced (Long) truss",
        params={'panels': (6, "number of panels"), 'span': (16.0, "span [m]"),
                'depth': (2.0, "depth [m]"), **_BAR_SECTION}),
    'truss_kingpost': dict(
        label='King post truss', family='bar', doc="a king post roof truss",
        params={'panels': (4, "number of panels"), 'span': (12.0, "span [m]"),
                'depth': (2.0, "rise [m]"), **_BAR_SECTION}),
    'truss_town': dict(
        label='Town lattice truss', family='bar', doc="a Town lattice truss",
        params={'span': (16.0, "span [m]"), 'depth': (2.0, "depth [m]"),
                'pitch': (1.0, "diagonal pitch [m]"), 'k': (4, "number of lattices"),
                **_BAR_SECTION}),
    'arch': dict(
        label='Arch', family='bar', doc="a parabolic two-hinged arch, in straight bars",
        params={'segments': (8, "number of bars"), 'span': (20.0, "span [m]"),
                'rise': (4.0, "rise [m]"), **_BAR_SECTION}),
    'wall': dict(
        label='Wall', family='area',
        doc="a wall panel of width x height, one region meshed into quads by "
            "default (or triangles), fixed along the base; its corners are "
            "R.p0 (bottom left), R.p1, R.p2 (top right) and R.p3, the object "
            "is 'R'",
        params={'width': (4.0, "width [m]"), 'height': (3.0, "height [m]"),
                'thickness': (0.20, "thickness [m]"),
                'mesh_size': (0.5, "target element size [m]"),
                # 26/09/2026 (Matias, "'quad' e o novo default do
                # programa"): every mesh xdfem2d builds should default to
                # quads now -- pass quads=False for the old triangle mesh.
                'quads': (True, "mesh with quads (False for triangles)"),
                **_AREA_MATERIAL}),
    'slab': dict(
        label='Slab', family='area',
        doc="a rectangular slab Lx x Ly (plate domain), one region 'R' meshed "
            "into quads by default (or triangles); edges is 'simply', "
            "'clamped' or 'free' for all four sides, or a dict {'bottom', "
            "'right', 'top', 'left': ...}; pressure is a downward load "
            "[kN/m2] in its own case 'Q' (none by default)",
        params={'Lx': (5.0, "size in x [m]"), 'Ly': (5.0, "size in y [m]"),
                'thickness': (0.20, "thickness [m]"), 'edges': ('simply', "edge supports"),
                'pressure': (None, "uniform pressure [kN/m2], downward"),
                'mesh_size': (0.5, "target element size [m]"),
                'quads': (True, "mesh with quads (False for triangles)"),
                **_AREA_MATERIAL}),
    'wall_beam': dict(
        label='Deep beam', family='area',
        doc="a simply supported deep beam (a wall panel loaded in its plane): "
            "width is the span, height the depth; one region 'R' meshed into quads "
            "by default (or triangles), pinned at its bottom-left corner R.p0 and "
            "on a roller at the bottom-right corner R.p1",
        params={'width': (6.0, "span [m]"), 'height': (3.0, "depth [m]"),
                'thickness': (0.30, "thickness [m]"),
                'mesh_size': (0.5, "target element size [m]"),
                'quads': (True, "mesh with quads (False for triangles)"),
                **_AREA_MATERIAL}),
    'wall_frame': dict(
        label='Wall + frame', family='bar',
        doc="a multi-bay, multi-storey frame with a full-height shear wall at one "
            "end (wall_side 'left' or 'right'), every base fixed; the wall shares "
            "the frame's joint at each floor; bay_width and floor_height are a "
            "length or a list of them",
        params={'bays': (1, "number of bays"), 'floors': (1, "number of storeys"),
                'bay_width': (5.0, "bay width [m] or list"),
                'floor_height': (3.0, "storey height [m] or list"),
                'wall_width': (3.0, "width of the wall [m]"),
                'wall_side': ('left', "'left' or 'right' of the frame"),
                'thickness': (0.20, "wall thickness [m]"),
                'mesh_size': (0.5, "target wall element size [m]"),
                'quads': (True, "mesh the wall with quads (False for triangles)"),
                **_BAR_SECTION}),
    'slab_ribbed': dict(
        label='Rib-stiffened slab', family='area',
        doc="a rectangular slab Lx x Ly (plate domain) framed by four edge ribs, "
            "resting on its four corners; pressure is a downward load [kN/m2] in "
            "its own case 'Q' (none by default)",
        params={'Lx': (5.0, "size in x [m]"), 'Ly': (5.0, "size in y [m]"),
                'thickness': (0.15, "slab thickness [m]"),
                'rib_b': (0.30, "rib width [m]"), 'rib_h': (0.50, "rib height [m]"),
                'pressure': (None, "uniform pressure [kN/m2], downward"),
                'mesh_size': (0.5, "target element size [m]"),
                'quads': (True, "mesh with quads (False for triangles)"),
                **_AREA_MATERIAL}),
    'flat_slab': dict(
        label='Flat slab on columns', family='area',
        doc="a flat slab (plate domain), one panel per bay, carried only on point "
            "columns at the panel corners (no edge beams); Lx and Ly are the span "
            "of each bay, or a list of spans, and bays_x/bays_y the number of bays",
        params={'Lx': (8.0, "bay span in x [m] or list"),
                'Ly': (8.0, "bay span in y [m] or list"),
                'bays_x': (2, "number of bays in x"), 'bays_y': (2, "number of bays in y"),
                'thickness': (0.22, "slab thickness [m]"),
                'pressure': (None, "uniform pressure [kN/m2], downward"),
                'mesh_size': (0.5, "target element size [m]"),
                'quads': (True, "mesh with quads (False for triangles)"),
                **_AREA_MATERIAL}),
    'slab_on_grade': dict(
        label='Slab on grade', family='area',
        doc="a rectangular slab Lx x Ly (plate domain) on an elastic (Winkler) "
            "foundation: the springs alone carry it, there are no other supports",
        params={'Lx': (6.0, "size in x [m]"), 'Ly': (6.0, "size in y [m]"),
                'thickness': (0.25, "slab thickness [m]"),
                'kz': (30000.0, "subgrade modulus [kN/m3]"),
                'pressure': (None, "uniform pressure [kN/m2], downward"),
                'mesh_size': (0.5, "target element size [m]"),
                'quads': (True, "mesh with quads (False for triangles)"),
                **_AREA_MATERIAL}),
    'slab_sector': dict(
        label='Circular slab sector', family='area',
        doc="a circular slab sector (plate domain): rmin = 0 is solid, angle 360 a "
            "full disc; the outer arc, the inner arc and the two radial sides each "
            "take 'free', 'pin' or 'fixed'",
        params={'rmin': (0.0, "inner radius [m]"), 'rmax': (5.0, "outer radius [m]"),
                'angle': (90.0, "opening angle [degrees], up to 360"),
                'thickness': (0.20, "thickness [m]"),
                'mesh_size': (0.5, "target element size [m]"),
                'pressure': (None, "uniform pressure [kN/m2], downward"),
                'outer': ('pin', "support of the outer arc"),
                'inner': ('free', "support of the inner arc"),
                'radial': ('free', "support of the two radial sides"),
                **_AREA_MATERIAL}),
    'beam_grid': dict(
        label='Beam grid', family='area',
        doc="an orthogonal grillage Lx x Ly of beams (plate domain), simply "
            "supported on the perimeter, under self-weight",
        params={'Lx': (6.0, "size in x [m]"), 'Ly': (6.0, "size in y [m]"),
                'bays_x': (3, "number of bays in x"), 'bays_y': (3, "number of bays in y"),
                'b': (0.30, "beam width [m]"), 'h': (0.50, "beam height [m]"),
                **_AREA_MATERIAL}),
    'bridge_grillage': dict(
        label='Bridge deck', family='area',
        doc="a bridge-deck grillage (plate domain): longitudinal girders tied by "
            "transverse members, simply supported at both abutments, under "
            "self-weight",
        params={'length': (20.0, "span between abutments [m]"),
                'width': (10.0, "deck width [m]"),
                'girders': (4, "number of longitudinal girders (2 or more)"),
                'stations': (8, "number of transverse bays along the span"),
                'girder_b': (0.40, "girder width [m]"), 'girder_h': (1.20, "girder height [m]"),
                'cross_b': (0.30, "transverse member width [m]"),
                'cross_h': (0.60, "transverse member height [m]"),
                **_AREA_MATERIAL}),
}

_SUPPORTS_BEAM = ('simply', 'cantilever', 'fixed', 'propped')
_SUPPORTS_END = ('free', 'pin', 'fixed')
_SUPPORTS_EDGE = ('simply', 'clamped', 'free', 'pin', 'fixed')


def template_kinds() -> list[str]:
    return list(TEMPLATE_KINDS)


def normalize_kind(kind) -> str:
    """The kind *kind* names, or ValueError listing the valid ones."""
    key = str(kind or '').strip().lower().replace('-', '_').replace(' ', '_')
    key = _ALIASES.get(key, key)
    if key not in TEMPLATE_KINDS:
        raise ValueError(
            f"create_from_template: unknown kind {kind!r}. The kinds are: "
            f"{', '.join(TEMPLATE_KINDS)}.")
    return key


def kind_signature(kind: str) -> str:
    """``kind(name=default, ...)`` as text, for a message or a reference."""
    spec = TEMPLATE_KINDS[kind]
    parts = []
    for name, (default, _doc) in spec['params'].items():
        parts.append(f"{name}={default!r}")
    return f"{kind}({', '.join(parts)})"


def _num(kind, name, value, *, positive=True, integer=False):
    try:
        v = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"create_from_template({kind!r}): {name} must be a number, "
                         f"not {value!r}") from None
    if not math.isfinite(v) or (positive and v <= 0):
        raise ValueError(f"create_from_template({kind!r}): {name} must be a positive "
                         f"number, not {value!r}")
    if integer:
        if v != int(v):
            raise ValueError(f"create_from_template({kind!r}): {name} must be a whole "
                             f"number, not {value!r}")
        return int(v)
    return v


def _lengths(kind, name, value):
    """A length, or a list of them, as floats (a number stays a number)."""
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"create_from_template({kind!r}): {name} is an empty list")
        return [_num(kind, name, v) for v in value]
    return _num(kind, name, value)


def check_params(kind: str, params: dict) -> dict:
    """*params* completed with the defaults, or ValueError saying what is wrong.

    Only names and values are looked at: nothing is built. An unknown name is
    the common mistake (a parameter of another kind, or of the low-level
    template), so the message lists what the kind takes.
    """
    kind = normalize_kind(kind)
    spec = TEMPLATE_KINDS[kind]
    takes = spec['params']
    unknown = [k for k in params if k not in takes]
    if unknown:
        raise ValueError(
            f"create_from_template({kind!r}): unknown parameter "
            f"{unknown[0]!r}. It takes: {', '.join(takes)}.")
    p = {name: default for name, (default, _d) in takes.items()}
    p.update(params)

    for name in ('b', 'h'):
        if name in p:
            p[name] = _num(kind, name, p[name])
    if 'material' in p:
        p['material'] = str(p['material'])
    if p.get('profile') is not None:
        p['profile'] = str(p['profile'])

    if kind == 'beam':
        p['span'] = _num(kind, 'span', p['span'])
        p['divisions'] = _num(kind, 'divisions', p['divisions'], integer=True)
        p['support'] = str(p['support']).lower()
        if p['support'] not in _SUPPORTS_BEAM:
            raise ValueError(f"create_from_template('beam'): support must be one of "
                             f"{', '.join(_SUPPORTS_BEAM)}, not {params.get('support')!r}")
    elif kind == 'continuous_beam':
        s = p['spans']
        if isinstance(s, (list, tuple)):
            p['spans'] = _lengths(kind, 'spans', s)
        else:
            p['spans'] = _num(kind, 'spans', s, integer=True)
        p['span'] = _lengths(kind, 'span', p['span'])
        for name in ('left', 'intermediate', 'right'):
            p[name] = str(p[name]).lower()
            if p[name] not in _SUPPORTS_END:
                raise ValueError(
                    f"create_from_template('continuous_beam'): {name} must be one of "
                    f"{', '.join(_SUPPORTS_END)}, not {params.get(name)!r}")
    elif kind == 'portal_frame':
        p['span'] = _num(kind, 'span', p['span'])
        p['height'] = _num(kind, 'height', p['height'])
    elif kind == 'frame':
        p['bays'] = _num(kind, 'bays', p['bays'], integer=True)
        p['floors'] = _num(kind, 'floors', p['floors'], integer=True)
        p['bay_width'] = _lengths(kind, 'bay_width', p['bay_width'])
        p['floor_height'] = _lengths(kind, 'floor_height', p['floor_height'])
    elif kind in ('truss_warren', 'truss_howe', 'truss_pratt', 'truss_long',
                  'truss_kingpost'):
        p['panels'] = _num(kind, 'panels', p['panels'], integer=True)
        p['span'] = _num(kind, 'span', p['span'])
        p['depth'] = _num(kind, 'depth', p['depth'])
    elif kind == 'truss_town':
        for name in ('span', 'depth', 'pitch'):
            p[name] = _num(kind, name, p[name])
        p['k'] = _num(kind, 'k', p['k'], integer=True)
    elif kind == 'arch':
        p['segments'] = _num(kind, 'segments', p['segments'], integer=True)
        p['span'] = _num(kind, 'span', p['span'])
        p['rise'] = _num(kind, 'rise', p['rise'])
    elif kind == 'wall':
        for name in ('width', 'height', 'thickness', 'mesh_size'):
            p[name] = _num(kind, name, p[name])
        p['quads'] = bool(p['quads'])
    elif kind == 'slab':
        for name in ('Lx', 'Ly', 'thickness', 'mesh_size'):
            p[name] = _num(kind, name, p[name])
        p['quads'] = bool(p['quads'])
        if p['pressure'] is not None:
            p['pressure'] = _num(kind, 'pressure', p['pressure'], positive=False)
        e = p['edges']
        names = [e] if isinstance(e, str) else (
            list(e.values()) if isinstance(e, dict) else None)
        if names is None:
            raise ValueError("create_from_template('slab'): edges is 'simply', "
                             "'clamped' or 'free', or a dict of the sides "
                             "bottom, right, top and left")
        if isinstance(e, dict):
            bad = [k for k in e if k not in ('bottom', 'right', 'top', 'left')]
            if bad:
                raise ValueError(f"create_from_template('slab'): {bad[0]!r} is not a "
                                 "side: bottom, right, top or left")
        for v in names:
            if str(v).lower() not in _SUPPORTS_EDGE:
                raise ValueError(f"create_from_template('slab'): an edge is "
                                 f"'simply', 'clamped' or 'free', not {v!r}")
    elif kind == 'wall_beam':
        for name in ('width', 'height', 'thickness', 'mesh_size'):
            p[name] = _num(kind, name, p[name])
        p['quads'] = bool(p['quads'])
    elif kind == 'wall_frame':
        p['bays'] = _num(kind, 'bays', p['bays'], integer=True)
        p['floors'] = _num(kind, 'floors', p['floors'], integer=True)
        p['bay_width'] = _lengths(kind, 'bay_width', p['bay_width'])
        p['floor_height'] = _lengths(kind, 'floor_height', p['floor_height'])
        for name in ('wall_width', 'thickness', 'mesh_size'):
            p[name] = _num(kind, name, p[name])
        p['quads'] = bool(p['quads'])
        p['wall_side'] = str(p['wall_side']).lower()
        if p['wall_side'] not in ('left', 'right'):
            raise ValueError("create_from_template('wall_frame'): wall_side is "
                             f"'left' or 'right', not {params.get('wall_side')!r}")
    elif kind in ('slab_ribbed', 'slab_on_grade'):
        for name in ('Lx', 'Ly', 'thickness', 'mesh_size'):
            p[name] = _num(kind, name, p[name])
        if kind == 'slab_ribbed':
            p['rib_b'] = _num(kind, 'rib_b', p['rib_b'])
            p['rib_h'] = _num(kind, 'rib_h', p['rib_h'])
        else:
            p['kz'] = _num(kind, 'kz', p['kz'])
        p['quads'] = bool(p['quads'])
        if p['pressure'] is not None:
            p['pressure'] = _num(kind, 'pressure', p['pressure'], positive=False)
    elif kind == 'flat_slab':
        p['Lx'] = _lengths(kind, 'Lx', p['Lx'])
        p['Ly'] = _lengths(kind, 'Ly', p['Ly'])
        p['bays_x'] = _num(kind, 'bays_x', p['bays_x'], integer=True)
        p['bays_y'] = _num(kind, 'bays_y', p['bays_y'], integer=True)
        p['thickness'] = _num(kind, 'thickness', p['thickness'])
        p['mesh_size'] = _num(kind, 'mesh_size', p['mesh_size'])
        p['quads'] = bool(p['quads'])
        if p['pressure'] is not None:
            p['pressure'] = _num(kind, 'pressure', p['pressure'], positive=False)
    elif kind == 'slab_sector':
        p['rmin'] = _num(kind, 'rmin', p['rmin'], positive=False)
        if p['rmin'] < 0:
            raise ValueError("create_from_template('slab_sector'): rmin must be "
                             f"zero or more, not {params.get('rmin')!r}")
        for name in ('rmax', 'angle', 'thickness', 'mesh_size'):
            p[name] = _num(kind, name, p[name])
        if p['rmax'] <= p['rmin']:
            raise ValueError("create_from_template('slab_sector'): rmax must be "
                             "larger than rmin")
        if p['angle'] > 360:
            raise ValueError("create_from_template('slab_sector'): angle is in "
                             f"degrees, 360 at most, not {params.get('angle')!r}")
        if p['pressure'] is not None:
            p['pressure'] = _num(kind, 'pressure', p['pressure'], positive=False)
        for name in ('outer', 'inner', 'radial'):
            p[name] = str(p[name]).lower()
            if p[name] not in _SUPPORTS_END:
                raise ValueError(
                    f"create_from_template('slab_sector'): {name} must be one of "
                    f"{', '.join(_SUPPORTS_END)}, not {params.get(name)!r}")
    elif kind == 'beam_grid':
        for name in ('Lx', 'Ly'):
            p[name] = _num(kind, name, p[name])
        p['bays_x'] = _num(kind, 'bays_x', p['bays_x'], integer=True)
        p['bays_y'] = _num(kind, 'bays_y', p['bays_y'], integer=True)
    elif kind == 'bridge_grillage':
        for name in ('length', 'width', 'girder_b', 'girder_h', 'cross_b',
                     'cross_h'):
            p[name] = _num(kind, name, p[name])
        p['girders'] = _num(kind, 'girders', p['girders'], integer=True)
        p['stations'] = _num(kind, 'stations', p['stations'], integer=True)
        if p['girders'] < 2:
            raise ValueError("create_from_template('bridge_grillage'): girders "
                             "is 2 or more")
    return p


def has_geometry(struc) -> bool:
    """True when *struc* already has nodes, bars, area elements or objects."""
    for attr in ('nodes', 'bar_elements', 'tri_elements', 'quad_elements',
                 'geometry_objects'):
        if getattr(struc, attr, None):
            return True
    return False


def domain_mismatch(kind: str, struc) -> str:
    """'' unless *struc* is an empty plate-domain model (a grillage or slab
    started from New Model) and *kind* is a bar kind, else what is wrong.

    Every bar kind (beam, continuous_beam, ...) builds a plane-domain model,
    and ``fill_in_place`` replaces the target's whole state, domain included --
    so without this an empty grillage model (``Structure2D(domain='plate')``,
    no geometry yet, indistinguishable from a fresh one by ``has_geometry``
    alone) would silently become a plane one. The other direction is not a
    mismatch: a brand new ``Structure2D()`` is 'plane' by default whether or
    not the user meant a slab, so 'slab' is free to turn it into a plate
    model -- that default carries no signal the way an already-set 'plate'
    domain does.
    """
    kind = normalize_kind(kind)
    if (str(getattr(struc, 'domain', 'plane') or 'plane') == 'plate'
            and TEMPLATE_KINDS[kind]['family'] != 'area'):
        return (f"create_from_template({kind!r}) builds a plane-domain model, "
               "but this one is already a plate-domain one (a grillage or "
               "slab): there is no bar template for that domain yet -- build "
               "it with the create_* calls instead.")
    return ''


def _resolve_material(material: str):
    """(E, unit weight, poisson, material_type, design) for *material* -- a
    concrete, steel or timber Eurocode class, one string. Goes through
    Structure2D._auto_material_from_class, the exact classification the
    single-element API (create_bar_section) already uses for the same
    strings -- not a second concrete/steel/timber grade table here.
    ValueError (unrecognised class) for anything none of the three know."""
    from xdfem2d.structure import Structure2D
    tmp = Structure2D()
    name = tmp._auto_material_from_class(material)
    if name is None:
        raise ValueError(
            f"create_from_template: unknown material {material!r} -- not a "
            "recognised concrete, steel or timber class")
    m = tmp.materials[name]
    return (m.elastic_modulus, m.unit_weight, getattr(m, 'poisson', 0.2),
            m.material_type, m.design)


def _apply_profile(s, profile: str, grade: str):
    """Give every bar section of *s* the catalogue *profile*'s exact numbers
    (area, inertia, moduli, buckling curves, ...) -- _tpl_*'s shape formula
    from b/h/tw/tf is exact for a plain rectangle but only approximate for a
    real rolled section, and this is the only place a Section's ~15 override
    fields are threaded through, since the _tpl_* builders take plain
    dimensions, not a full catalogue Section. The material itself needs no
    fixing here -- _bar_kwargs already built it correctly, in the same pass
    as everything else, so this only ever touches sections."""
    from xdfem2d.structure import Structure2D
    tmp = Structure2D()
    sec = tmp.create_steel_section('X', profile, grade)
    mat_name = next(iter(s.materials))   # the one material a bar kind built
    for name in list(s.sections):
        new = copy.deepcopy(sec)
        new.name = name
        new.material_name = mat_name
        s.sections[name] = new


def _bar_kwargs(p):
    """b, h, E, g, material_type, shape, tw, tf, design for the bar builders.

    Concrete, steel (as a plain rectangle) and timber all resolve through
    _resolve_material -- one call, whatever the class -- so build_template
    makes the final material in a single pass and never builds one only to
    replace it with another. A catalogue profile (b, h, tw, tf, shape) comes
    from create_steel_section instead, since that is where a rolled
    section's true dimensions come from; profile always means steel, and its
    grade is still 'material' -- not a second, separate parameter.
    """
    E, g, _nu, material_type, design = _resolve_material(p['material'])
    if not p.get('profile'):
        return dict(b=p['b'], h=p['h'], E=E, g=g, material_type=material_type,
                    design=design)
    from xdfem2d.structure import Structure2D
    tmp = Structure2D()
    sec = tmp.create_steel_section('X', p['profile'], p['material'])
    return dict(b=sec.b, h=sec.h, E=E, g=g, material_type=material_type,
                shape=sec.shape, tw=sec.tw, tf=sec.tf, design=design)


def _stamp_material(s, material_type, design) -> None:
    """Give the material(s) of a built model the type and the design data of the
    class asked for (fck, fy...): the area and grillage builders take only E, g
    and nu, and a reinforcement design needs the class."""
    for m in s.materials.values():
        m.material_type = material_type
        m.design = copy.deepcopy(design)


def _pressure_pz(p) -> float:
    return -p['pressure'] if p.get('pressure') else 0.0


def _mesh_count(length: float, size: float) -> int:
    return max(1, math.ceil(length / size - 1e-9))


def _build_more(kind: str, p: dict, T):
    """The kinds that came after wall and slab: the deep beam, the frame with a
    wall, and the special slabs and grillages. Each is the application's own
    builder, with the material as a name and the mesh as a target size."""
    if kind == 'wall_frame':
        kw = _bar_kwargs(p)
        design = kw.pop('design')
        nu = _resolve_material(p['material'])[2]
        bw, fh = p['bay_width'], p['floor_height']
        bays = len(bw) if isinstance(bw, list) else p['bays']
        floors = len(fh) if isinstance(fh, list) else p['floors']
        s = T.wall_frame_obj(
            bays, floors, bw, fh, p['wall_width'],
            _mesh_count(p['wall_width'], p['mesh_size']), thickness=p['thickness'],
            nu=nu, wall_side=p['wall_side'], prefer_quad=p['quads'], **kw)
        _stamp_material(s, kw['material_type'], design)
        return s
    E, g, nu, material_type, design = _resolve_material(p['material'])
    if kind == 'wall_beam':
        s = T.wall_beam_obj(
            p['width'], p['height'], _mesh_count(p['width'], p['mesh_size']),
            _mesh_count(p['height'], p['mesh_size']), p['thickness'], E, g, nu,
            prefer_quad=p['quads'], material_type=material_type)
    elif kind == 'slab_ribbed':
        s = T.slab_ribbed_obj(
            p['Lx'], p['Ly'], t=p['thickness'],
            nx=_mesh_count(p['Lx'], p['mesh_size']),
            ny=_mesh_count(p['Ly'], p['mesh_size']), b=p['rib_b'], h=p['rib_h'],
            E=E, g=g, nu=nu, pz=_pressure_pz(p), prefer_quad=p['quads'])
    elif kind == 'flat_slab':
        lx, ly = p['Lx'], p['Ly']
        ncx = len(lx) if isinstance(lx, list) else p['bays_x']
        ncy = len(ly) if isinstance(ly, list) else p['bays_y']
        s = T.flat_slab_obj(lx, ly, ncx, ncy, t=p['thickness'],
                            max_size=p['mesh_size'], E=E, g=g, nu=nu,
                            pz=_pressure_pz(p), prefer_quad=p['quads'])
    elif kind == 'slab_on_grade':
        s = T.slab_on_grade_obj(
            p['Lx'], p['Ly'], t=p['thickness'],
            nx=_mesh_count(p['Lx'], p['mesh_size']),
            ny=_mesh_count(p['Ly'], p['mesh_size']), kz=p['kz'], E=E, g=g, nu=nu,
            pz=_pressure_pz(p), prefer_quad=p['quads'])
    elif kind == 'slab_sector':
        s = T.slab_sector_obj(
            p['rmin'], p['rmax'], p['angle'], p['mesh_size'], t=p['thickness'],
            E=E, g=g, nu=nu, pz=_pressure_pz(p), outer=p['outer'],
            inner=p['inner'], radial=p['radial'])
    elif kind == 'beam_grid':
        s = T.beam_grid_obj(p['Lx'], p['Ly'], p['bays_x'], p['bays_y'],
                            b=p['b'], h=p['h'], E=E, g=g, nu=nu)
    else:                                                    # bridge_grillage
        s = T.bridge_grillage_obj(
            p['length'], p['width'], p['girders'], p['stations'],
            b=p['girder_b'], h=p['girder_h'], bt=p['cross_b'], ht=p['cross_h'],
            E=E, g=g, nu=nu)
    _stamp_material(s, material_type, design)
    return s


def build_template(kind: str, **params):
    """A new :class:`Structure2D` for *kind* with *params* (see TEMPLATE_KINDS).

    The model is built, not applied: :meth:`Structure2D.create_from_template`
    puts it in place of an empty one. Raises ValueError for an unknown kind, an
    unknown parameter or a value that cannot be used.
    """
    from xdfem2d import templates as T
    kind = normalize_kind(kind)
    p = check_params(kind, params)
    family = TEMPLATE_KINDS[kind]['family']

    if kind == 'beam':
        s = T.beam(p['support'], p['span'], n_seg=p['divisions'], **_bar_kwargs(p))
    elif kind == 'continuous_beam':
        n = p['spans']
        lengths = (n if isinstance(n, list)
                   else p['span'] if isinstance(p['span'], list) else None)
        count = len(lengths) if lengths else int(n)
        span_len = lengths if lengths else p['span']
        s = T.continuous_beam(count, span_len, 0.0,
                              left=p['left'], intermediate=p['intermediate'],
                              right=p['right'], **_bar_kwargs(p))
    elif kind == 'portal_frame':
        s = T.portal_frame(p['span'], p['height'], **_bar_kwargs(p))
    elif kind == 'frame':
        bw, fh = p['bay_width'], p['floor_height']
        bays = len(bw) if isinstance(bw, list) else p['bays']
        floors = len(fh) if isinstance(fh, list) else p['floors']
        s = T.frame(bays, floors, bw, fh, **_bar_kwargs(p))
    elif kind in ('truss_warren', 'truss_howe', 'truss_pratt', 'truss_long',
                  'truss_kingpost'):
        s = getattr(T, kind)(p['panels'], p['span'], p['depth'], **_bar_kwargs(p))
    elif kind == 'truss_town':
        s = T.truss_town(p['span'], p['depth'], p['pitch'], p['k'], **_bar_kwargs(p))
    elif kind == 'arch':
        s = T.arch(p['segments'], p['span'], p['rise'], **_bar_kwargs(p))
    elif kind == 'wall':
        E, g, nu, material_type, design = _resolve_material(p['material'])
        nx = max(1, math.ceil(p['width'] / p['mesh_size'] - 1e-9))
        ny = max(1, math.ceil(p['height'] / p['mesh_size'] - 1e-9))
        s = T.wall_obj(p['width'], p['height'], nx, ny, p['thickness'], E, g, nu,
                       prefer_quad=p['quads'], material_type=material_type,
                       design=design)
    elif kind == 'slab':
        E, g, nu, material_type, design = _resolve_material(p['material'])
        nx = max(1, math.ceil(p['Lx'] / p['mesh_size'] - 1e-9))
        ny = max(1, math.ceil(p['Ly'] / p['mesh_size'] - 1e-9))
        pz = -p['pressure'] if p['pressure'] else 0.0
        s = T.slab_edges_obj(T._normalize_edges(p['edges']), p['Lx'], p['Ly'],
                             p['thickness'], nx, ny, E, g, nu, pz=pz,
                             prefer_quad=p['quads'], material_type=material_type,
                             design=design)
    else:
        s = _build_more(kind, p, T)

    # Everything above already built the right material and section in one
    # pass (_bar_kwargs / _resolve_material) -- the only thing still needed
    # is a catalogue profile's exact section numbers, which the _tpl_*
    # builders cannot take directly (see _apply_profile).
    if family == 'bar' and p.get('profile'):
        _apply_profile(s, p['profile'], p['material'])
    if not s.project_info.get('Project name'):
        s.project_info['Project name'] = TEMPLATE_KINDS[kind]['label']
    return s


def fill_in_place(target, built) -> None:
    """Make *target* (an empty model) the model *built*, keeping its object.

    A script's ``model`` is one object the caller holds on to, so the new state
    has to go into that object. The project information the target already has
    is kept.
    """
    info = {k: v for k, v in (target.project_info or {}).items() if v}
    target.__dict__.clear()
    target.__dict__.update(built.__dict__)
    target.project_info.update(info)
