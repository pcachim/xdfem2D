"""A model as JSON: the minimal shape, and whether a given one is complete.

Written to test a hypothesis. Asked to build a beam, five language models
produced five scripts and four of them were unsolvable — not because they got
the API names wrong, but because they left steps out. Four of five called
``add_support`` and never ``assign_support``; three referred to a load case
they never created.

The idea, which is the user's: a JSON template makes the missing step visible.
``support_assignments`` is a key you can see sitting empty next to a filled
``supports``. A sequence of calls has no such surface — nothing in
``add_support(...)`` says a second call is owed.

Two other properties come free, and they matter as much:

**It is data.** Loading JSON runs nothing, so the whole "the assistant must
never execute code" constraint does not apply. A model handed back as JSON can
go straight into the application, which a script cannot.

**It can be checked for completeness.** :func:`missing_pieces` answers "is
anything needed absent" for the whole model at once, which no line-by-line
reading of a script can.

What it does *not* do, and this is the honest half: it says nothing about
whether the model is right. A load in the wrong direction, a support that
restrains nothing, a section too small — all of those pass. The engineering
judgement stays where it was.
"""
from __future__ import annotations

import json
from typing import Any

# The smallest model that solves: a simply supported beam under a uniform load.
# Every key present is one a model must fill; the point of the template is that
# nothing needed is invisible.
#
# Values are real and consistent — 6 m, C30/37, 0.3x0.5, 20 kN/m downward — so
# it can be loaded and solved as it stands. A template that does not work is a
# template nobody trusts.
BEAM: dict[str, Any] = {
    "nodes": [
        {"id": "N1", "x": 0.0, "y": 0.0},
        {"id": "N2", "x": 6.0, "y": 0.0},
    ],
    "materials": [
        {"name": "C30/37", "elastic_modulus": 33000000.0, "unit_weight": 25.0},
    ],
    "sections": [
        {"name": "S1", "material_name": "C30/37", "b": 0.3, "h": 0.5},
    ],
    "bar_elements": [
        {"id": "E1", "node_i": "N1", "node_j": "N2", "section_name": "S1"},
    ],
    # Two steps, and this is the pair that gets forgotten: 'supports' describes
    # the restraints, 'support_assignments' puts them on nodes. A model with the
    # first and not the second is a mechanism.
    "supports": [
        {"name": "PIN", "ux": True, "uy": True, "tz": False},
        {"name": "ROLLER_X", "ux": False, "uy": True, "tz": False},
    ],
    "support_assignments": [
        {"node_id": "N1", "support_name": "PIN"},
        {"node_id": "N2", "support_name": "ROLLER_X"},
    ],
    "load_cases": [
        {"id": "G", "self_weight_factor": 1.0, "action_type": "Permanent"},
    ],
    # fye/fyd are the load at each end, in kN/m. Negative is downward, and
    # equal values mean a uniform load — different ones a trapezoidal one.
    "distributed_loads": [
        {"element_id": "E1", "load_case_id": "G",
         "fxe": 0.0, "fxd": 0.0, "fye": -20.0, "fyd": -20.0,
         "coord_sys": "global"},
    ],
    "analysis_cases": [
        {"id": "G", "analysis_type": "Linear", "coefficients": {"G": 1.0}},
    ],
}

# The wall counterpart: a 4 x 3 m shear wall, 0.2 m thick, fixed along the base
# and pushed sideways by 100 kN at the top. Two templates and not one because a
# wall shares almost no keys with a beam — 'tri_sections' instead of 'sections',
# a thickness instead of b and h, three node ids per element instead of two —
# and a model handed the beam template and asked for a wall has to invent every
# one of those. It was asked to, and it invented 'sections' with a 'thickness'.
#
# Meshed by hand into four triangles, which is the smallest mesh that is not
# degenerate. Real walls are not written out this way: see 'geometry_objects',
# which describes the region and lets the solver mesh it. This template is the
# explicit form because it shows what a triangle element actually needs.
#
# It solves, and its equilibrium is exact: the base reactions sum to -100 kN
# horizontally against the 100 kN applied.
WALL: dict[str, Any] = {
    "nodes": [
        {"id": "N1", "x": 0.0, "y": 0.0},
        {"id": "N2", "x": 2.0, "y": 0.0},
        {"id": "N3", "x": 4.0, "y": 0.0},
        {"id": "N4", "x": 0.0, "y": 3.0},
        {"id": "N5", "x": 2.0, "y": 3.0},
        {"id": "N6", "x": 4.0, "y": 3.0},
    ],
    "materials": [
        {"name": "C30/37", "elastic_modulus": 33000000.0, "unit_weight": 25.0},
    ],
    # A wall section is a thickness. There is no b and h — the other two
    # dimensions are the geometry of the triangles themselves.
    "tri_sections": [
        {"name": "W1", "material_name": "C30/37", "thickness": 0.2},
    ],
    # Three nodes each, anticlockwise. 'section_name' here names a tri_section,
    # not one of 'sections' — the same field name, a different vocabulary.
    "tri_elements": [
        {"id": "T1", "node_i": "N1", "node_j": "N2", "node_k": "N5",
         "section_name": "W1"},
        {"id": "T2", "node_i": "N1", "node_j": "N5", "node_k": "N4",
         "section_name": "W1"},
        {"id": "T3", "node_i": "N2", "node_j": "N3", "node_k": "N6",
         "section_name": "W1"},
        {"id": "T4", "node_i": "N2", "node_j": "N6", "node_k": "N5",
         "section_name": "W1"},
    ],
    # Triangles carry no rotation, so tz is false: restraining it would restrain
    # a degree of freedom the element does not have — the same pattern the
    # pin() facade produces on a plane domain (translation fixed, rotation
    # free), so named "PIN" here too, not "FIX" (which would mean tz fixed
    # as well, a DOF this element does not have to give).
    "supports": [
        {"name": "PIN", "ux": True, "uy": True, "tz": False},
    ],
    "support_assignments": [
        {"node_id": "N1", "support_name": "PIN"},
        {"node_id": "N2", "support_name": "PIN"},
        {"node_id": "N3", "support_name": "PIN"},
    ],
    "load_cases": [
        {"id": "G", "self_weight_factor": 1.0, "action_type": "Permanent"},
    ],
    # Loads on a wall go on its nodes: a triangle has no 'distributed_loads'
    # the way a bar does. A load spread along an edge is 'tri_edge_loads'.
    "point_loads": [
        {"node_id": "N4", "load_case_id": "G", "fx": 50.0, "fy": 0.0,
         "mz": 0.0},
        {"node_id": "N6", "load_case_id": "G", "fx": 50.0, "fy": 0.0,
         "mz": 0.0},
    ],
    "analysis_cases": [
        {"id": "G", "analysis_type": "Linear", "coefficients": {"G": 1.0}},
    ],
}

def _slab_template() -> dict:
    """A minimal solvable DKT slab (plate domain): a 5 x 4 m panel, 0.20 m
    thick, meshed 2 x 2 into eight triangles, simply supported (w restrained)
    on every edge node, under self-weight plus a uniform pressure pz.

    Built in code rather than written out because a slab needs an interior node
    to deflect at all — a single quad held at four corners has none, and a
    template that solves to zero everywhere teaches the wrong thing. The keys
    are the plate ones: 'domain': 'plate', 'tri_sections' with 'formulation':
    'MITC3' (the default shear-deformable plate element; 'DKT' is the thin-plate
    alternative), a support that restrains only w, and 'tri_area_loads' (a
    pressure), which is where a slab's load lives — not 'distributed_loads',
    which is a bar's."""
    nx, ny, Lx, Ly = 2, 2, 5.0, 4.0
    nodes, ids = [], {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            nid = f"N{i}{j}"
            ids[(i, j)] = nid
            nodes.append({"id": nid, "x": round(Lx * i / nx, 3),
                          "y": round(Ly * j / ny, 3)})
    tris, tri_ids = [], []
    e = 1
    for j in range(ny):
        for i in range(nx):
            a, b = ids[(i, j)], ids[(i + 1, j)]
            c, d = ids[(i + 1, j + 1)], ids[(i, j + 1)]
            for (u, v, w) in ((a, b, c), (a, c, d)):
                tid = f"T{e}"
                tri_ids.append(tid)
                tris.append({"id": tid, "node_i": u, "node_j": v,
                             "node_k": w, "section_name": "Slab"})
                e += 1
    edge = [nid for (i, j), nid in ids.items()
            if i in (0, nx) or j in (0, ny)]
    return {
        "domain": "plate",
        "nodes": nodes,
        "materials": [
            {"name": "C30/37", "elastic_modulus": 33000000.0,
             "unit_weight": 25.0, "poisson": 0.2},
        ],
        # A plate section is a thickness and a formulation. MITC3 is the default
        # (shear-deformable) plate bending element; there is no b and h.
        "tri_sections": [
            {"name": "Slab", "material_name": "C30/37", "thickness": 0.20,
             "formulation": "MITC3"},
        ],
        "tri_elements": tris,
        # Simple support restrains only w (the first slot). Restraining the
        # rotations too would clamp the edge — a different boundary condition.
        # Same pattern the pin() facade produces on a plate domain, so named
        # "PIN" here too, not the old "SS" ("simply supported") — the facade
        # never uses that name, and matching it is the point.
        "supports": [
            {"name": "PIN", "ux": True, "uy": False, "tz": False},
        ],
        "support_assignments": [
            {"node_id": nid, "support_name": "PIN"} for nid in edge
        ],
        "load_cases": [
            {"id": "G", "self_weight_factor": 1.0, "action_type": "Permanent"},
        ],
        # A slab's load is a pressure per triangle [kN/m²], negative downward —
        # not a bar's distributed_loads.
        "tri_area_loads": [
            {"tri_id": tid, "load_case_id": "G", "pz": -5.0}
            for tid in tri_ids
        ],
        "analysis_cases": [
            {"id": "G", "analysis_type": "Linear", "coefficients": {"G": 1.0}},
        ],
    }


def _grillage_template() -> dict:
    """A minimal solvable grillage (plate domain): an orthogonal grid of beams
    over a 6 x 6 m bay, 2 x 2 panels, simply supported (w restrained) on the
    perimeter nodes, under self-weight. Bars carry bending, shear and torsion
    out of plane; the section is a normal b x h beam."""
    nx, ny, Lx, Ly = 2, 2, 6.0, 6.0
    nodes, ids = [], {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            nid = f"N{i}{j}"
            ids[(i, j)] = nid
            nodes.append({"id": nid, "x": round(Lx * i / nx, 3),
                          "y": round(Ly * j / ny, 3)})
    bars = []
    e = 1
    for j in range(ny + 1):                      # beams along x
        for i in range(nx):
            bars.append({"id": f"B{e}", "node_i": ids[(i, j)],
                         "node_j": ids[(i + 1, j)], "section_name": "Beam"})
            e += 1
    for i in range(nx + 1):                      # beams along y
        for j in range(ny):
            bars.append({"id": f"B{e}", "node_i": ids[(i, j)],
                         "node_j": ids[(i, j + 1)], "section_name": "Beam"})
            e += 1
    edge = [nid for (i, j), nid in ids.items()
            if i in (0, nx) or j in (0, ny)]
    return {
        "domain": "plate",
        "nodes": nodes,
        "materials": [
            {"name": "C30/37", "elastic_modulus": 33000000.0,
             "unit_weight": 25.0, "poisson": 0.2},
        ],
        "sections": [
            {"name": "Beam", "material_name": "C30/37", "b": 0.30, "h": 0.50},
        ],
        "bar_elements": bars,
        # Same pattern the pin() facade produces on a plate domain (restrains
        # only w) — named "PIN" for the same reason as the slab template.
        "supports": [
            {"name": "PIN", "ux": True, "uy": False, "tz": False},
        ],
        "support_assignments": [
            {"node_id": nid, "support_name": "PIN"} for nid in edge
        ],
        "load_cases": [
            {"id": "G", "self_weight_factor": 1.0, "action_type": "Permanent"},
        ],
        "analysis_cases": [
            {"id": "G", "analysis_type": "Linear", "coefficients": {"G": 1.0}},
        ],
    }


#: The templates by name, so a caller can ask for one without knowing the
#: module's variables.
TEMPLATES = {'beam': BEAM, 'wall': WALL,
             'slab': _slab_template(), 'grillage': _grillage_template()}

#: What every solvable model needs, whatever it is made of. Keys, not counts —
#: the check is "is this present at all", which is the failure that keeps
#: happening.
#:
#: 'supports'/'support_assignments' used to be required outright — but a slab
#: on grade (add_area_spring, no supports at all) solves perfectly well
#: restrained only by its Winkler springs, and was being reported incomplete
#: for having no 'supports', the same beam/wall-shaped assumption that once
#: made the wall template look incomplete (see ONE_OF's own history below).
#: The restraint check now lives in RESTRAINT_ONE_OF instead.
REQUIRED = (
    'nodes', 'materials', 'load_cases', 'analysis_cases',
)

#: Alternatives, one of each family. Written as families and not as a flat list
#: because the first version demanded 'sections' and 'bar_elements', and a wall
#: has neither: it is triangles with a thickness. A perfectly solvable wall was
#: reported incomplete, and the button to open it never appeared — the same
#: beam-shaped assumption that produced six beam fixtures and no wall.
#:
#: A model may be bars, or triangles, or geometry objects that become triangles
#: at solve time, or any mixture. Each line is "at least one of these".
ONE_OF = (
    (('sections', 'tri_sections'),
     "a section: 'sections' for bars (b, h) or 'tri_sections' for walls "
     "(thickness)"),
    (('bar_elements', 'tri_elements', 'geometry_objects'),
     "something to analyse: 'bar_elements', 'tri_elements', or "
     "'geometry_objects' (regions meshed into triangles when the model is "
     "solved)"),
)

#: What can restrain a model — a real support (assigned to a node) is the
#: usual answer, but an elastic foundation is a legitimate alternative: a
#: slab on grade rests entirely on 'tri_area_springs'/'quad_area_springs'/
#: 'surface_area_springs', no 'supports' at all, and that is correct, not
#: incomplete. Node/element/line springs are listed too for the same reason,
#: even though a model relying on one of those alone (no support, no area
#: spring) would be an unusual thing to actually build.
RESTRAINT_ONE_OF = (
    'support_assignments', 'node_springs', 'element_springs',
    'line_element_springs', 'tri_area_springs', 'quad_area_springs',
    'surface_area_springs',
)


def template(kind: str = 'beam', as_text: bool = True):
    """A minimal model of the given kind, ready to be edited.

    ``kind`` is one of 'beam' (in-plane bars), 'wall' (in-plane triangles),
    'slab' (plate-domain triangles) or 'grillage' (plate-domain bars). Anything
    else falls back to the beam rather than raising: this is called with
    whatever a language model wrote in a tool argument, and a KeyError there
    costs the user their answer over a word.

    Returned as text by default, because that is how it reaches a model and
    how it is compared: an indented, stable rendering diffs cleanly against
    what comes back.
    """
    data = TEMPLATES.get(str(kind).strip().lower(), BEAM)
    if as_text:
        return json.dumps(data, indent=2, ensure_ascii=False)
    return json.loads(json.dumps(data))             # a fresh copy


def completeness_pieces(data: dict) -> list[str]:
    """Presence-only checks: what a solvable model must *have*, said without any
    reference to whether its cross-references resolve.

    The completeness half of :func:`missing_pieces`, split out so the resilient
    loader can fold it into a report (as warnings) alongside
    ``Structure2D.reference_problems`` — which covers the dangling references on
    the built model — without the two overlapping. ``missing_pieces`` calls this
    for its own head, so the wording is shared and a reader sees one vocabulary.
    """
    if not isinstance(data, dict):
        return ["the model is not a JSON object"]

    out: list[str] = []
    for key in REQUIRED:
        if not data.get(key):
            out.append(f"'{key}' is empty or absent")

    for keys, what in ONE_OF:
        if not any(data.get(k) for k in keys):
            out.append(f"the model has none of {', '.join(keys)} — it needs "
                       f"{what}")

    # A geometry object with an explicit edge_supports list (see
    # GeoRectangle.edge_supports in models.py) creates its own
    # support_assignments dynamically at solve time, from nothing recorded
    # here — the same class of "restraint that only exists after expansion"
    # tri_area_springs/surface_area_springs already are, just discovered
    # later (2026-08, when this template was added). Treated the same way:
    # a valid restraint source on its own.
    has_edge_supports = any(
        isinstance(g, dict) and g.get('edge_supports')
        for g in data.get('geometry_objects') or [])

    if not (any(data.get(k) for k in RESTRAINT_ONE_OF) or has_edge_supports):
        out.append(
            "the model has no restraint at all — none of "
            f"{', '.join(RESTRAINT_ONE_OF)}, and no geometry object with "
            "edge_supports; it needs 'support_assignments' (the usual "
            "case), an area spring ('tri_area_springs'/'quad_area_springs'/"
            "'surface_area_springs') for a foundation model, or a geometry "
            "object with per-edge restraint")

    # The pair, spelled out rather than left to the reader: 'supports' present
    # and 'support_assignments' empty is the single most common way a generated
    # model turns out to be a mechanism — UNLESS a geometry object's own
    # edge_supports is what is going to assign them, at solve time.
    if (data.get('supports') and not data.get('support_assignments')
            and not has_edge_supports):
        out.append(
            "supports are defined but assigned to no node — "
            "'support_assignments' links a support name to a node id, and "
            "without it the structure floats")
    return out


def missing_pieces(data: dict) -> list[str]:
    """Sentences naming what a model needs and does not have.

    Deliberately about presence, not correctness — the distinction this whole
    module rests on. It finds a model that cannot be solved; it has nothing to
    say about one that solves and is wrong.
    """
    out: list[str] = completeness_pieces(data)
    if not isinstance(data, dict):
        return out

    # References that point at nothing. Same class as the dangling load case
    # in a script, and cheaper to find here because everything is in one place.
    #
    # Per collection, not per field name: 'section_name' means a bar section
    # inside bar_elements and a wall section inside tri_elements — the same
    # word, two vocabularies. A single global mapping looked up a wall's
    # section among the bar sections and reported a real one as missing.
    # Only dict rows carry a name/id. A malformed model — e.g. a node that came
    # back as a bare number instead of an object — would otherwise crash the set
    # comprehension on `.get`; here it is simply skipped, and the reference
    # checks below report it as dangling, which is the useful message.
    def _ids(rows, key):
        # skip non-dict rows (malformed model output) — see comment above
        return {r.get(key) for r in rows or [] if isinstance(r, dict)}

    names = {
        'sections': _ids(data.get('sections'), 'name'),
        'tri_sections': _ids(data.get('tri_sections'), 'name'),
        'materials': _ids(data.get('materials'), 'name'),
        'supports': _ids(data.get('supports'), 'name'),
        'load_cases': _ids(data.get('load_cases'), 'id'),
        'nodes': _ids(data.get('nodes'), 'id'),
        'elements': _ids((data.get('bar_elements') or [])
                         + (data.get('tri_elements') or []), 'id'),
    }
    # collection -> {field: which set it must be found in}
    refs = {
        'bar_elements': {'node_i': 'nodes', 'node_j': 'nodes',
                         'section_name': 'sections'},
        'tri_elements': {'node_i': 'nodes', 'node_j': 'nodes',
                         'node_k': 'nodes', 'section_name': 'tri_sections'},
        # geometry_objects are not in this table: which field they carry
        # depends on their kind, and they are handled below.
        'sections': {'material_name': 'materials'},
        'tri_sections': {'material_name': 'materials'},
        'support_assignments': {'node_id': 'nodes',
                                'support_name': 'supports'},
        'distributed_loads': {'element_id': 'elements',
                              'load_case_id': 'load_cases'},
        'point_loads': {'node_id': 'nodes', 'load_case_id': 'load_cases'},
        'tri_edge_loads': {'load_case_id': 'load_cases'},
        'nodal_masses': {'node_id': 'nodes'},
    }
    # Every field above is one the row cannot work without, so absent is as
    # wrong as dangling — and it was the case not covered. granite wrote four
    # tri_elements with no 'section_name' at all: this said the model was
    # complete and the loader died with KeyError 'material_name', which tells
    # the user nothing about what to fix.
    #
    # The same hole had been found and patched the day before for a geometry
    # object with no 'node_ids'. It was patched there and only there, which is
    # the mistake this project keeps making: fixing one path and not the class
    # it belongs to. Hence one loop over one table, and no special cases.
    for collection, fields in refs.items():
        for row in data.get(collection) or []:
            if not isinstance(row, dict):
                continue
            for field, source in fields.items():
                value = row.get(field)
                if value in (None, ''):
                    out.append(f"{collection}: {row.get('id', '?')!r} has no "
                               f"'{field}' — it must name one of {source}")
                elif not isinstance(value, str):
                    # A real Section/Material/etc. object here (instead of its
                    # name) means the caller passed the live object where a
                    # name string belongs — a bug in whoever built `data`, not
                    # in the model. Reported, not raised: `value not in
                    # names[source]` below would crash on an unhashable
                    # dataclass instance before this function got to say why.
                    out.append(f"{collection}: {row.get('id', '?')!r} has a "
                               f"non-string '{field}'={value!r} "
                               f"({type(value).__name__}) — it must name one "
                               f"of {source} by its name, not the object "
                               f"itself")
                elif value not in names[source]:
                    out.append(f"{collection}: {field}={value!r} is not in "
                               f"{source}")
            # A geometry object carries the ids of the points that define it.
            for node in row.get('node_ids') or []:
                if node not in names['nodes']:
                    out.append(f"{collection}: {node!r} is not a node")

    # Geometry objects, which are two families and not one. I wrote the first
    # version as though they were all regions, and a shipped example with an
    # arc in it was told its arc had no wall section — an arc has no wall
    # section, it becomes bars:
    #
    #   line, arc, polyline    -> bar elements, so 'section_name'     (sections)
    #   rectangle, surface     -> triangles,    so 'tri_section_name' (tri_sections)
    #
    # The number of defining nodes is what separates a real object from a
    # leftover: a rectangle needs its four corners, a line two, an arc three.
    # Checked because a rectangle carrying two corners meshes into nothing at
    # all, and everything else passes it — there is nothing dangling to find.
    # The example that failed this rule turned out to contain exactly that: an
    # object producing no elements, which nobody had noticed.
    # kind -> (section field, its source, min nodes). The object kinds were
    # renamed (line->segment, polyline->multisegment, surface->polygon); both
    # the new and old spellings are accepted so files saved before the rename
    # still validate.
    GEO = {
        'segment': ('section_name', 'sections', 2),
        'line': ('section_name', 'sections', 2),
        'arc': ('section_name', 'sections', 3),
        'multisegment': ('section_name', 'sections', 2),
        'polyline': ('section_name', 'sections', 2),
        'rectangle': ('tri_section_name', 'tri_sections', 4),
        'polygon': ('tri_section_name', 'tri_sections', 3),
        'surface': ('tri_section_name', 'tri_sections', 3),
    }
    for obj in data.get('geometry_objects') or []:
        if not isinstance(obj, dict):
            continue
        oid = obj.get('id', '?')
        kind = str(obj.get('kind', '')).strip().lower()
        if kind not in GEO:
            out.append(f"geometry_objects: {oid!r} has kind={kind!r}, which is "
                       f"not one of {', '.join(sorted(GEO))}")
            continue
        field, source, least = GEO[kind]
        value = obj.get(field)
        if value in (None, ''):
            out.append(f"geometry_objects: {kind} {oid!r} has no {field!r} — "
                       f"it must name one of {source}")
        elif not isinstance(value, str):
            # See the matching guard in the collection loop above: the same
            # object-instead-of-name mistake, here for a geometry object.
            out.append(f"geometry_objects: {kind} {oid!r} has a non-string "
                       f"{field!r}={value!r} ({type(value).__name__}) — it "
                       f"must name one of {source} by its name, not the "
                       f"object itself")
        elif value not in names[source]:
            out.append(f"geometry_objects: {field}={value!r} is not in "
                       f"{source}")
        ids = obj.get('node_ids') or []
        if not ids:
            out.append(
                f"geometry_objects: {oid!r} has no 'node_ids' — an object "
                f"names the nodes that define it, which must exist as nodes "
                f"too. It does not carry its own coordinates")
        elif len(ids) < least:
            out.append(f"geometry_objects: {kind} {oid!r} is defined by "
                       f"{len(ids)} node(s) and needs at least {least} — as it "
                       f"stands it produces no elements")
        for node in ids:
            if node not in names['nodes']:
                out.append(f"geometry_objects: {node!r} is not a node")

    # Constraints (multi-point / linear DOF coupling). Optional — a model
    # without any is complete — so this only fires for constraints that are
    # present but incomplete or dangling. The shapes differ by kind, so unlike
    # the flat 'refs' table above each is checked on its own.
    _COMPS = ('ux', 'uy', 'tz')
    for c in data.get('constraints') or []:
        if not isinstance(c, dict):
            continue
        cid = c.get('id', '?')
        kind = str(c.get('kind', 'equation')).strip()
        if kind == 'rigid_link':
            master = c.get('master')
            slaves = c.get('slaves') or []
            # The pair, like supports/support_assignments: slaves but no master
            # is the common way a generated rigid link goes nowhere.
            if slaves and not master:
                out.append(f"constraints: rigid link {cid!r} has slaves but no "
                           f"'master' node to tie them to")
            elif master and master not in names['nodes']:
                out.append(f"constraints: {cid!r} master={master!r} is not a node")
            if not slaves:
                out.append(f"constraints: rigid link {cid!r} has no 'slaves'")
            for s in slaves:
                if s not in names['nodes']:
                    out.append(f"constraints: {cid!r} slave {s!r} is not a node")
        elif kind == 'equal_dof':
            cnodes = c.get('nodes') or []
            comps = c.get('components') or []
            if len(cnodes) < 2:
                out.append(f"constraints: equal-DOF {cid!r} needs at least two "
                           f"'nodes'")
            for n in cnodes:
                if n not in names['nodes']:
                    out.append(f"constraints: {cid!r} node {n!r} is not a node")
            if not comps:
                out.append(f"constraints: equal-DOF {cid!r} names no "
                           f"'components' (any of ux, uy, tz)")
            for comp in comps:
                if comp not in _COMPS:
                    out.append(f"constraints: {cid!r} component {comp!r} is not "
                               f"one of ux, uy, tz")
        else:   # 'equation'
            terms = c.get('terms') or []
            if not terms:
                out.append(f"constraints: equation {cid!r} has no 'terms' "
                           f"(each is [node, comp, coef])")
            for t in terms:
                if not (isinstance(t, (list, tuple)) and len(t) == 3):
                    out.append(f"constraints: {cid!r} has a malformed term {t!r} "
                               f"— expected [node, comp, coef]")
                    continue
                node, comp, _coef = t
                if node not in names['nodes']:
                    out.append(f"constraints: {cid!r} term node {node!r} is not "
                               f"a node")
                if comp not in _COMPS:
                    out.append(f"constraints: {cid!r} term component {comp!r} is "
                               f"not one of ux, uy, tz")

    return out


def _strip_comments(text: str) -> str:
    """Remove // and /* */ comments from JSON that should not have them.

    JSON has no comments and this is not an attempt to invent them. It is that
    a language model asked for a commented model writes them anyway — granite
    produced a model with `/* Load case ... */` in the middle — and json.loads
    then rejects the whole thing with a character offset. The intent is
    unambiguous and the alternative is the user seeing nothing work.

    String contents are left alone, so an id containing // survives.
    """
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch == '"':                     # copy the string whole
            j = i + 1
            while j < n:
                if text[j] == '\\':
                    j += 2
                    continue
                if text[j] == '"':
                    break
                j += 1
            out.append(text[i:j + 1])
            i = j + 1
        elif text.startswith('//', i):
            i = text.find('\n', i)
            if i < 0:
                break
        elif text.startswith('/*', i):
            end = text.find('*/', i + 2)
            i = n if end < 0 else end + 2
        else:
            out.append(ch)
            i += 1
    return ''.join(out)


def _try_fix(text: str, e) -> str | None:
    """Return *text* with one heuristic repair applied at the error, or None.

    Used by :func:`parse_tolerant` to read past a syntax error a language model
    or a hand-edit left behind: a stray or leading comma, a missing delimiter,
    an illegal character.
    """
    pos, msg = e.pos, e.msg
    # A stray comma before a closing brace/bracket, or a leading/double comma:
    # json reports "Expecting property name…" (object) or "Expecting value"
    # (array). Walk back over whitespace to the comma and drop it.
    if msg.startswith("Expecting property name") or msg.startswith("Expecting value"):
        j = pos - 1
        while j >= 0 and text[j] in " \t\r\n":
            j -= 1
        if j >= 0 and text[j] == ',':
            return text[:j] + text[j + 1:]
    # A missing separator between two values: insert the one json expected.
    if msg.startswith("Expecting ',' delimiter"):
        return text[:pos] + ',' + text[pos:]
    if msg.startswith("Expecting ':' delimiter"):
        return text[:pos] + ':' + text[pos:]
    # Anything else (a stray or illegal character): remove it and move on.
    if 0 <= pos < len(text):
        return text[:pos] + text[pos + 1:]
    return None


def parse_tolerant(text, *, on_syntax_error=None, max_fixes: int = 60):
    """Parse JSON, tolerating what people and language models actually write.

    The single tolerant parser both entry points share — a file being opened
    and a model lifted out of an assistant reply. It applies, in order:

    1. a fast strict parse (the common case: the text is already valid JSON);
    2. comment stripping (``//`` and ``/* */``) and trailing-comma removal, both
       invalid JSON and both common;
    3. a bounded recovery loop that reads past a remaining syntax error with one
       small heuristic edit at its position (drop a stray comma, insert a
       missing delimiter, remove an illegal character) and retries.

    ``on_syntax_error(msg, line, column)`` is called once per error met in the
    loop, so a caller collecting a report (the file loader) can record each;
    a caller that only wants the result (the assistant path) passes ``None``.
    Returns the parsed object, or raises the final ``JSONDecodeError`` when the
    text cannot be recovered within *max_fixes*.
    """
    import re as _re
    try:
        return json.loads(text)
    except ValueError:
        pass
    working = _strip_comments(text)
    working = _re.sub(r',(\s*[}\]])', r'\1', working)
    last = None
    for _ in range(max_fixes + 1):
        try:
            return json.loads(working)
        except json.JSONDecodeError as e:
            last = e
            if on_syntax_error is not None:
                on_syntax_error(e.msg, e.lineno, e.colno)
            fixed = _try_fix(working, e)
            if fixed is None or fixed == working:
                raise
            working = fixed
    if on_syntax_error is not None:
        on_syntax_error("too many syntax errors — stopped trying to recover",
                        None, None)
    raise last


def parse(text: str) -> dict:
    """JSON text as a dict, tolerating what models actually write.

    A thin wrapper over :func:`parse_tolerant` with no error reporting: the
    assistant path only needs the result — a block parses or it does not.
    Comments, a trailing comma, a missing delimiter and a stray character are
    all invalid and all common; recovering them here turns "Expecting property
    name at line 31" — where the reader gave up, not where the mistake is —
    into a model that loads.
    """
    return parse_tolerant(text)


def load(data) -> 'object':
    """Build a Structure2D from a model dict or JSON text.

    Nothing here executes anything: it is `json.loads` and the same
    ``_from_dict`` the file loader uses. That is the property that makes this
    usable directly from an assistant reply, where a script is not.
    """
    from xdfem2d.structure_io import _from_dict
    if isinstance(data, (bytes, bytearray)):
        data = data.decode('utf-8')
    if isinstance(data, str):
        data = parse(data)
    return _from_dict(data)


def load_checked(data):
    """Build a Structure2D from a model dict or JSON text, resiliently.

    The tolerant counterpart to :func:`load`. Where ``load`` uses the strict
    ``_from_dict`` and raises on the first bad row, this delegates to
    ``structure_io_checked.build_checked``: it reads to the end, builds every
    valid entry, and returns a report of everything wrong. This is what lets a
    model lifted out of an assistant reply be built with the same tolerance as a
    file opened from disk — the same builder, the same ``LoadReport``.

    Accepts a dict (already parsed) or JSON text (parsed here with the shared
    tolerant parser, syntax errors recorded into the report). Returns
    ``(struc, report)``.
    """
    from xdfem2d.structure_io_checked import LoadReport, build_checked
    if isinstance(data, (bytes, bytearray)):
        data = data.decode('utf-8')
    if isinstance(data, str):
        report = LoadReport()
        try:
            data = parse_tolerant(
                data,
                on_syntax_error=lambda m, l, c: report.add(
                    'syntax', 'json', m, line=l, column=c))
        except ValueError:
            data = None
        return build_checked(data, report=report)
    return build_checked(data)
