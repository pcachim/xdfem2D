"""Check a model-building script without running it.

The point is one specific mistake. Writing a test in this project, with the
source open alongside, I called ``add_distributed_load(wy=-10.0)``; the
parameters are ``fye`` and ``fyd``. A language model working from its memory of
other FEM libraries makes that mistake constantly, and it is invisible: the
script reads correctly, the name is plausible, and nothing says otherwise until
the user runs the file and gets a TypeError from inside a library they did not
write.

So every ``struc.method(...)`` call is matched against the real signature,
taken from the running class. Nothing here executes the script, imports it, or
evaluates any part of it — it is parsed, walked, and compared. That is the
whole reason this is worth having: a check that cannot be turned into a way of
running code the user has not read.

**What it does not do.** It knows nothing about values, units, or whether the
structure makes sense. A script that passes this can still build a mechanism,
load it in the wrong direction, or reference a section that was never created.
Those need the model, and the model needs the script to have run.
"""
from __future__ import annotations

import ast
import inspect
import re
from typing import Any

from xdfem2d.structure import Structure2D


def is_build_mode(source: str) -> bool:
    """Whether *source* is a whole-model script (defines ``build()``) rather
    than a snippet meant to land on a ``model`` already provided.

    The single source of truth for that split — :func:`xdfem2d.script_runner
    .run_script` used to make this same decision with its own inline regex,
    the application's script-editor dialog made it with
    none at all (always ran the whole-script check, :func:`check`, even on a
    plain ``model.`` snippet — which :func:`check` always reports as
    'no Structure2D is built', so a script that ``run_script`` executes fine
    was refused before it ever got there). Both now call this.
    """
    return bool(re.search(r'^\s*def\s+build\s*\(', source or '', re.MULTILINE))

# Typographic punctuation a model's own decoding sometimes substitutes for the
# ASCII a Python parser expects — non-breaking/en/em hyphens and the unicode
# minus sign in place of '-' (a hexagon's vertices typed as "-1.5" with a
# U+2011 non-breaking hyphen is the transcript that found this), curly quotes
# in place of straight ones, and a non-breaking space in place of a plain one.
#
# the application's ``tidy_reply`` (assistant) already maps the hyphen/minus pair back to
# ASCII, but only in the prose half of a reply: a fenced code block is
# deliberately left untouched there (its own history: a LaTeX-subscript
# cleanup once ran over code too and silently turned add_material into
# addmaterial). That exclusion is right for the substitutions that can mangle
# an identifier; it is not right for these, which are ASCII either way and
# never appear inside a real xdfem2D name — so the code side of a reply never
# got them, and one of these characters inside a fenced block reached
# ast.parse unmodified and reported only "invalid character", with nothing
# pointing at what to fix.
#
# Explicit \u escapes throughout, deliberately: several of these look
# identical to their ASCII counterpart (or to each other) in most editors and
# terminals, and a literal glyph typed here would be exactly the kind of
# mistake this function exists to correct in someone else's text.
_SMART_PUNCT = {
    '\u2010': '-',   # hyphen
    '\u2011': '-',   # non-breaking hyphen (the transcript that found this)
    '\u2012': '-',   # figure dash
    '\u2013': '-',   # en dash
    '\u2014': '-',   # em dash
    '\u2212': '-',   # minus sign
    '\u2018': "'",   # left single quotation mark
    '\u2019': "'",   # right single quotation mark
    '\u201c': '"',   # left double quotation mark
    '\u201d': '"',   # right double quotation mark
    '\u00a0': ' ',   # non-breaking space
}


def normalize_source(text: str) -> str:
    """Map the typographic punctuation in :data:`_SMART_PUNCT` back to ASCII.

    Applied once, ahead of every ``ast.parse``/``compile`` a model-authored
    script or increment goes through (:func:`check`, :func:`check_increment`,
    :func:`check_editor`, and :func:`xdfem2d.script_runner.run_script`), so
    the check and the actual run agree on what the source "is" before either
    reads it.
    """
    if not text:
        return text
    for bad, good in _SMART_PUNCT.items():
        text = text.replace(bad, good)
    return text


def _data_attrs() -> frozenset:
    """Real instance attributes of a Structure2D — 'model.bar_elements',
    'model.domain', 'model.nodes[...]' and the like, none of which are
    method calls.

    Computed from a fresh instance rather than hand-maintained: the
    previous hard-coded list here had drifted (missing 'domain', 'cuts',
    'tri_edge_loads', 'surface_edge_loads', and about a dozen others —
    confirmed by diffing it against `vars(Structure2D())`), which would
    have made _unknown_attributes below cry wolf on real, legitimate
    code the moment it was added. Constructing an instance is cheap (no
    I/O, just dict/list init), same cost class as _methods()'s
    unconditional inspect.getmembers() call on every check().
    """
    return frozenset(a for a in vars(Structure2D()) if not a.startswith('_'))


def _methods() -> dict:
    return {name: fn for name, fn in
            inspect.getmembers(Structure2D, inspect.isfunction)
            if not name.startswith('_')}


def _has_var_keyword(sig) -> bool:
    return any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())


import functools
import textwrap


@functools.lru_cache(maxsize=None)
def _forward_attr(name: str) -> str | None:
    """The method a ``**kw`` wrapper hands its extra keywords to.

    A sugar method like ``add_plate_section(self, name, material_name,
    thickness=..., formulation=..., **kw)`` ends in ``return
    self.add_tri_section(..., **kw)``. ``inspect.signature`` sees only the
    ``**kw`` and so ``sig.bind`` accepts *any* keyword — which is exactly how
    ``add_plate_section(x=5)`` passed the check and then raised a TypeError at
    apply time, the ``x`` reaching ``add_tri_section`` which rejects it. This
    reads the wrapper's source (statically, nothing executed) and returns the
    ``self.<attr>`` it forwards ``**kw`` into, so the residual keywords can be
    validated there. None when there is no such single forward — a genuinely
    open ``**kw`` whose keys cannot be judged without running the method.
    """
    fn = _methods().get(name)
    if fn is None:
        return None
    try:
        tree = ast.parse(textwrap.dedent(inspect.getsource(fn)))
    except (OSError, TypeError, SyntaxError):
        return None
    target = None
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and any(isinstance(k, ast.keyword) and k.arg is None
                        for k in node.keywords)):
            f = node.func
            if (isinstance(f, ast.Attribute) and isinstance(f.value, ast.Name)
                    and f.value.id == 'self'):
                target = f.attr          # last one wins: the actual forward
    return target


@functools.lru_cache(maxsize=None)
def _accepted_keywords(name: str, _seen: tuple = ()) -> tuple:
    """(accepted keyword names, open) for a method, following ``**kw`` forwards.

    ``accepted`` is every keyword the call chain will take by name; ``open`` is
    True when some link forwards ``**kw`` to a destination this cannot resolve
    (or a cycle), meaning an unlisted keyword still might be valid, so nothing
    should be flagged. When ``open`` is False the set is exhaustive: a keyword
    outside it is genuinely unexpected and would raise at run time.
    """
    methods = _methods()
    if name in _seen or name not in methods:
        return (frozenset(), True)
    sig = inspect.signature(methods[name])
    named = frozenset(p.name for p in sig.parameters.values()
                      if p.kind in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY))
    if not _has_var_keyword(sig):
        return (named, False)
    target = _forward_attr(name)
    if target is None:
        return (named, True)             # open **kw — cannot judge extras
    sub, sub_open = _accepted_keywords(target, _seen + (name,))
    return (named | sub, sub_open)


def _forwarded_kw_problem(name: str, kwargs: dict, line: int,
                          sig, render) -> dict | None:
    """Flag a keyword that ``sig.bind`` waved through into a ``**kw`` but the
    forward destination will reject at run time. None when the method takes no
    ``**kw``, its forward is unresolvable, or every keyword is accepted."""
    if not _has_var_keyword(sig):
        return None
    accepted, is_open = _accepted_keywords(name)
    if is_open:
        return None
    bad = [k for k in kwargs if k not in accepted]
    if not bad:
        return None
    target = _forward_attr(name)
    where = f" (forwarded to {target}(), which does not take it)" if target else ""
    return _problem(line, 'arguments',
                    f"{name}(): unexpected keyword argument '{bad[0]}'"
                    f"{where}. It takes {render(name, sig)}")


def _none_for_str_problem(name: str, sig, node, line: int) -> dict | None:
    """Flag a literal ``None`` given to a parameter annotated plain ``str``.

    ``bind`` accepts it, so ``model.add_node(id=None, x=6.0, y=6.0)`` came back
    "all valid" from the checker, and then failed (or made a node with no id)
    when the user ran it. Only an annotation that is exactly ``str`` counts:
    ``str | None``, a union with an object type, or a parameter whose default is
    None all allow it. Nothing is evaluated: the argument has to be the literal
    ``None`` in the source.
    """
    if any(isinstance(a, ast.Starred) for a in node.args):
        return None
    if any(kw.arg is None for kw in node.keywords):
        return None
    try:
        bound = sig.bind(None, *node.args,
                         **{kw.arg: kw.value for kw in node.keywords})
    except TypeError:
        return None                    # reported by the caller's own bind
    for pname, value in bound.arguments.items():
        param = sig.parameters[pname]
        if pname == 'self' or param.default is None:
            continue
        ann = param.annotation
        plain_str = ann is str or (isinstance(ann, str)
                                   and ann.strip().strip("'\"") == 'str')
        if (plain_str and isinstance(value, ast.Constant)
                and value.value is None):
            return _problem(
                line, 'arguments',
                f"{name}(): '{pname}' cannot be None, it must be a string. "
                f"It takes {_render(name, sig)}")
    return None


class _Problem(dict):
    """A finding: line, kind, and a sentence a person can act on."""


def _problem(line: int, kind: str, msg: str) -> dict:
    return {'line': line, 'kind': kind, 'msg': msg}


def _structure_names(tree: ast.AST) -> set:
    """Variables holding a Structure2D, by the ways one is normally made.

    Deliberately shallow: an assignment from ``Structure2D()``, from
    ``build()``, or a parameter annotated with the class. Anything cleverer
    would be guessing, and a checker that guesses wrong reports a mistake that
    is not there — which is worse than missing one, because the reader learns
    to ignore it.
    """
    names: set = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            fn = node.value.func
            called = getattr(fn, 'id', None) or getattr(fn, 'attr', None)
            if called in ('Structure2D', 'build', 'load_x2d',
                          'load_structure_json'):
                for tgt in node.targets:
                    if isinstance(tgt, ast.Name):
                        names.add(tgt.id)
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
            ann = getattr(node.annotation, 'id', None)
            if ann == 'Structure2D':
                names.add(node.target.id)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for arg in list(node.args.args) + list(node.args.kwonlyargs):
                if getattr(arg.annotation, 'id', None) == 'Structure2D':
                    names.add(arg.arg)
    return names


def _placeholder(node) -> Any:
    """A stand-in for an argument value. Only its presence is checked."""
    return object()


# Which argument of which method *creates* an id, and which ones *refer* to
# one. Position and keyword both, because scripts use either.
#
# Read off the API deliberately narrowly: only the pairs where a name is
# clearly a reference to something that must already exist. A wider net would
# report ids built at run time — `f'N{i}'` in a loop — as missing, and a
# checker that cries wolf is one whose true findings get skipped.
_CREATES = {
    'add_node': ('id', 0), 'add_bar_element': ('id', 0),
    'add_material': ('name', 0), 'add_section': ('name', 0),
    'add_support': ('name', 0), 'add_load_case': ('id', 0),
    'add_analysis_case': ('id', 0), 'add_load_combination': ('id', 0),
    'create_load_combination': ('id', 0),
    'add_spectral_function': ('id', 0), 'add_tri_section': ('name', 0),
    # add_quad_section had no _CREATES entry at all — a latent gap that
    # add_quad_element's new node_l/section_name _REFERS coverage below
    # would otherwise turn into a false "dangling section" report for
    # every quad section a script (or the exporter) creates the quad way.
    'add_quad_section': ('name', 0),
    # A plate section is created the same way; register it so a slab that
    # names it is not reported as referring to a section that was never made.
    'add_plate_section': ('name', 0),
    # create_load_case takes id in the same position and creates the same
    # thing — without this, a script that creates a load case the
    # create_* way and then references it through a fundamental add_*
    # call (add_point_load, add_distributed_load, ...) got a false
    # "dangling reference" on an id the script plainly defines one line
    # above.
    'create_load_case': ('id', 0),
    # The rest of the create_* layer had the exact same gap as
    # create_load_case above, just never registered: found for real when a
    # user's script wrote create_bar_element(..., id='E1') and referenced
    # 'E1' from add_distributed_load two lines later — a genuine
    # "unknown_id" false positive on an id the script plainly creates.
    # Every create_* that takes a single literal id/name argument is
    # registered here, at its actual keyword position.
    'create_node': ('id', 2),
    # create_bar_element(section, node_i, node_j, id=None, **kw) — id sits at
    # the same position as before the section-first reorder (section took
    # node_i's old slot, node_i/node_j shifted down one each, id unmoved).
    'create_bar_element': ('id', 3),
    # create_area_element(section, node_i, node_j, node_k, node_l=None,
    # id=None, **kw) replaces create_tri_element; id moved from 4 to 5 to
    # make room for node_l, which a triangle call omits but a quad call
    # doesn't.
    'create_area_element': ('id', 5),
    'create_analysis_case': ('id', 0),
    'create_rc_material': ('name', 2),
    'create_steel_material': ('name', 1),
    'create_timber_material': ('name', 1),
    'create_bar_section': ('name', 0),
    'create_area_section': ('name', 0),
    'create_rc_section': ('name', 0),
    'create_steel_section': ('name', 0),
    'create_timber_bar_section': ('name', 0),
    'create_support': ('name', 7),
    # create_polygon(outline, thickness, material=..., target_size=..., id=None)
    # — id made optional and moved to the end (required args first), like
    # create_bar_element/create_area_element above.
    'create_polygon': ('id', 4),
    'create_panel': ('id', 4),
    # Geometry-object creators, registered so an edge method (support_edge,
    # pin_edge, …) that names an object is checked against the ids the script
    # actually makes, instead of a wrong id (object_id=1 for a 'Laje1') passing
    # as valid until it KeyErrors at apply time. Only the surface objects an
    # edge method can take are needed; each takes its id at position 0.
    'add_geo_rectangle': ('id', 0),
    'add_geo_polygon': ('id', 0),
    # create_node_list is NOT registered here on purpose (see
    # _node_list_created_ids below): it creates several nodes at once from
    # an id_prefix, not one literal id per call, which this single-argument
    # (method -> (argname, position)) mechanism has no way to express. It is
    # handled as a special case in _missing_ids/_missing_ids_incremental
    # instead, simulating create_node_list's own id_prefix + auto_name
    # sequential numbering.
}


def _node_list_created_ids(call: ast.Call, existing: set) -> "list | None":
    """The ids a literal ``create_node_list(outline, id_prefix=...)`` call
    will produce, so ``_missing_ids``/``_missing_ids_incremental`` know
    "P1"/"P2"/"P3" exist without ever running the script -- the gap the
    _CREATES comment above used to just accept (confirmed for real on
    qwen3.5:4b: a script that built three nodes this way and then wired
    them up by id got a false "dangling reference", even though the model
    ran fine -- create_bar_element's node_i/node_j take an id OR a Node
    object, as_node_id resolves either the same way).

    Simulates create_node_list's own id generation (structure.py):
    ``auto_name(id_prefix, self.nodes)`` per point, in order, which walks
    "{prefix}1", "{prefix}2", ... skipping whatever is already in
    *existing* -- and "N1", "N2", ... (create_node's own default) when
    id_prefix is omitted, ``None``, or ``''`` (all falsy, so
    ``create_node_list``'s ``if id_prefix else None`` takes the same branch
    as leaving it out).

    Returns ``None`` -- not ``[]`` -- when the call's shape isn't one this
    can count without executing ``_compat.as_points``'s full dict/two-column
    logic (a computed outline, a non-literal id_prefix, ...); the caller
    then leaves *existing* untouched, same as today for those calls.
    """
    prefix_arg = _arg(call, 'id_prefix', 1)
    if prefix_arg is None:
        prefix = 'N'
    else:
        lit = _literal(prefix_arg)
        if lit is None or lit == '':
            prefix = 'N'
        elif isinstance(lit, str):
            prefix = lit
        else:
            return None
    outline = _arg(call, 'outline', 0)
    if not isinstance(outline, (ast.List, ast.Tuple)) or not outline.elts:
        return None
    elts = outline.elts
    if isinstance(elts[0], (ast.List, ast.Tuple)):
        count = len(elts)                          # [(x, y), (x, y), ...]
    elif all(isinstance(e, ast.Constant)
             and isinstance(e.value, (int, float))
             and not isinstance(e.value, bool) for e in elts):
        if len(elts) % 2:
            return None
        count = len(elts) // 2                      # [x1, y1, x2, y2, ...]
    else:
        return None                                  # dicts / two columns / computed
    pool = set(existing)
    ids = []
    for _ in range(count):
        n = 1
        while f"{prefix}{n}" in pool:
            n += 1
        nid = f"{prefix}{n}"
        pool.add(nid)
        ids.append(nid)
    return ids


_REFERS = {
    'add_bar_element': [('node_i', 1), ('node_j', 2), ('section_name', 3)],
    # create_bar_element(section, node_i, node_j, id=None, **kw) had the same
    # gap add_bar_element above closes: node_i/node_j/section are always
    # statically, explicitly created (never a runtime-mesh id the way a
    # tri/quad id legitimately can be), so a dangling reference here is a
    # genuine mistake, not a false positive waiting to happen.
    'create_bar_element': [('node_i', 1), ('node_j', 2), ('section', 0)],
    # create_support(nodes, ...) assigns a support to every id in `nodes` —
    # a list literal, not a single scalar like every other entry here. Each
    # element is checked individually (see _literal_values below), so
    # create_support(nodes=['N1', 'N2']) flags whichever of 'N1'/'N2' (if
    # any) was never created, not the call as a whole.
    'create_support': [('nodes', 0)],
    # add_area_element(id, node_i, node_j, node_k, node_l=None, *,
    # section_name) — same mechanism as add_bar_element above. node_l is
    # legitimately None for a triangle call; _literal(None) returns None
    # (it is neither a str nor a non-bool int), so _literal_values yields
    # nothing for it and no false "dangling reference" is ever reported for
    # the omitted 4th node.
    'add_area_element': [('node_i', 1), ('node_j', 2), ('node_k', 3),
                         ('node_l', 4), ('section_name', 5)],
    # add_tri_element(id, node_i, node_j, node_k, section_name) and
    # add_quad_element(id, node_i, node_j, node_k, node_l, section_name) —
    # the two shape-specific entry points add_area_element dispatches to.
    # Registered too so a script calling either directly (not just through
    # the create_*/add_area_element facade) gets the same coverage.
    'add_tri_element': [('node_i', 1), ('node_j', 2), ('node_k', 3),
                        ('section_name', 4)],
    'add_quad_element': [('node_i', 1), ('node_j', 2), ('node_k', 3),
                         ('node_l', 4), ('section_name', 5)],
    # create_area_element(section, node_i, node_j, node_k, node_l=None,
    # id=None, **kw) — the create_* facade over add_area_element above, same
    # node_l=None-for-a-triangle non-issue.
    'create_area_element': [('node_i', 1), ('node_j', 2), ('node_k', 3),
                            ('node_l', 4), ('section', 0)],
    # create_bar_point_load(elements, ...) — elements is a bar id, an
    # object, or a list of either (see its docstring); _literal_values
    # handles the list case the same way it does for create_support's nodes.
    'create_bar_point_load': [('load_case', 1), ('elements', 0)],
    # create_node_load(nodes, ...) — same shape as create_support's nodes.
    'create_node_load': [('load_case', 1), ('nodes', 0)],
    # create_support_settlement(nodes, ...) — same shape as create_support's
    # nodes.
    'create_support_settlement': [('load_case', 1), ('nodes', 0)],
    # create_edge_load(object_id, edge, ...) — object_id must name a
    # geometry object that exists, same as support_edge/pin_edge/fix_edge/
    # roller_edge/symm_edge/free_edge below.
    'create_edge_load': [('load_case', 2), ('object_id', 0)],
    'add_section': [('material_name', 1)],
    'assign_support': [('node_id', 0), ('support_name', 1)],
    'add_distributed_load': [('element_id', 0), ('load_case_id', 1)],
    'add_point_load': [('node_id', 0), ('load_case_id', 1)],
    'add_nodal_mass': [('node_id', 0), ('mass_case_id', 1)],
    # A slab's pressure load, the plate analogue of create_uniform_load: only the
    # load case is checked (the target triangle is often a mesh id built at run
    # time, which must not be reported as missing).
    'add_area_load': [('load_case_id', 1)],
    # The create_* loads facade had no coverage here at all — a load_case_id
    # left dangling by a typo went unreported even though the fundamental
    # equivalents above already catch it. Only load_case_id/load_case is
    # checked (not elements/nodes/targets/object_id): those accept a
    # geometry-object id too, and geometry-object creators are not
    # registered in _CREATES, so checking them would report a real,
    # legitimate reference as dangling. load_case_id has no such ambiguity
    # — it is always a load case, always created by add_load_case or
    # create_load_case, both already in _CREATES above.
    'create_uniform_load': [('load_case', 1)],
    'create_bar_distributed_load': [('load_case', 1)],
    # create_self_weight actually raises KeyError at run time for a
    # not-yet-created literal load_case (self.load_cases_by_id[lc_id], no
    # get-or-create for an explicit non-None case) — an even stronger
    # reason to catch this one before the script runs.
    'create_self_weight': [('load_case', 1)],
    # create_temperature's parameter is spelled load_case, not
    # load_case_id — same idea, different keyword.
    'create_temperature': [('load_case', 3)],
    # The temperature-load fundamentals had the same gap create_* loads had
    # above (and add_area_load had before it): a dangling load_case_id here
    # went unreported even though the mechanical loads' equivalents already
    # catch it. add_temperature_load's target is a bar element, always a
    # literal id the script itself assigns, so it is checked too, same as
    # add_distributed_load/create_uniform_load/add_point_load above. The other
    # four take a tri/quad/surface target instead — a mesh id built at run
    # time (tri/quad) or a geometry-object id create_*'s facade may not
    # register in _CREATES (surface) — so, like add_area_load, only
    # load_case_id is checked for those.
    'add_temperature_load': [('element_id', 0), ('load_case_id', 1)],
    'add_tri_temperature_load': [('load_case_id', 1)],
    'add_quad_temperature_load': [('load_case_id', 1)],
    'add_area_temperature_load': [('load_case_id', 1)],
    'add_line_temperature_load': [('load_case_id', 1)],
    # Edge supports/removals: object_id must name a geometry object that
    # exists (created in this increment, or already in the open model). Catches
    # object_id=1 for a 'Laje1' before it KeyErrors at apply time. Only the
    # object id is checked — edge/kind/support_name are values, not ids.
    'support_edge': [('object_id', 0)],
    'support_object_edge': [('object_id', 0)],
    'pin_edge': [('object_id', 0)],
    'fix_edge': [('object_id', 0)],
    'roller_edge': [('object_id', 0)],
    'symm_edge': [('object_id', 0)],
    'free_edge': [('object_id', 0)],
    'delete_area': [('object_id', 0)],
}


def _literal(node) -> str | None:
    """The string an argument holds, when it is a plain literal.

    Anything computed — an f-string, a variable, a loop index — returns None
    and is not checked. That is the whole safety margin: this may only report
    what it can see written down.
    """
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    # An id written as a bare number — object_id=1 for a rectangle whose id is
    # 'Laje1' — is a wrong-type id, not a computed value. Read it as its string
    # form so the missing-id check sees it is not a real id instead of skipping
    # it as unknowable (bool is an int subclass, so exclude True/False).
    if (isinstance(node, ast.Constant) and isinstance(node.value, int)
            and not isinstance(node.value, bool)):
        return str(node.value)
    return None


def _literal_values(node) -> list[str]:
    """Every literal id string in *node* — a single scalar literal (see
    :func:`_literal`), or a ``[...]``/``(...)`` literal, each element of
    which is read the same way and checked on its own.

    ``create_support(nodes=['N1', 'N2'], ...)`` needs this: ``nodes`` is
    always a list, never a scalar, so treating the whole call as
    "computed, not checked" the way :func:`_literal` alone would (a List is
    not an ast.Constant) meant every create_support call went unchecked
    regardless of how plainly its ids were written out. An element that is
    itself computed (a variable, an f-string, ...) is silently skipped, same
    as anywhere else in this module — only the list's shape is new here, not
    the safety margin.
    """
    if isinstance(node, (ast.List, ast.Tuple)):
        return [v for v in (_literal(elt) for elt in node.elts) if v is not None]
    v = _literal(node)
    return [v] if v is not None else []


def _arg(call, name: str, pos: int):
    for kw in call.keywords:
        if kw.arg == name:
            return kw.value
    if len(call.args) > pos and not any(isinstance(a, ast.Starred)
                                        for a in call.args):
        return call.args[pos]
    return None


def _missing_ids(calls, names: set) -> list[dict]:
    """Ids a script uses without ever creating them.

    The failure this exists for: a model writes create_uniform_load(..., 'G') and
    never writes add_load_case('G'). The call is valid, the arguments are
    valid, the script runs — and the load belongs to a case that does not
    exist, so nothing is loaded. Every layer below this is happy.
    """
    created, used = set(), []
    for node in calls:
        fn = node.func
        if not isinstance(fn, ast.Attribute) or not isinstance(fn.value, ast.Name):
            continue
        if fn.value.id not in names:
            continue
        method = fn.attr
        if method == 'create_node_list':
            ids = _node_list_created_ids(node, created)
            if ids:
                created.update(ids)
        elif method in _CREATES:
            v = _literal(_arg(node, *_CREATES[method]))
            if v is not None:
                created.add(v)
        for argname, pos in _REFERS.get(method, ()):
            for v in _literal_values(_arg(node, argname, pos)):
                used.append((node.lineno, method, argname, v))

    # A script that creates nothing is a fragment, not a model: `s.add_node('N1')`
    # on its own refers to ids it was never going to define, and saying so is
    # noise. Only worth reporting once the script is evidently building
    # something.
    if not created:
        return []

    problems = []
    for line, method, argname, value in used:
        if value in created:
            continue
        kind = argname.removesuffix('_id').removesuffix('_name').replace('_', ' ')
        problems.append(_problem(line, 'unknown_id',
                         f"{method}() refers to {argname}={value!r} — no {kind} {value!r} "
                         f"is ever created in this script. Create it first. Nothing will "
                         f"fail when it runs — the reference is simply dangling, and "
                         f"whatever depends on it is silently absent"))
    return problems


def _domain_reassignment_problems(tree: ast.AST, names: set, struc) -> list[dict]:
    """``<model>.domain = ...`` is a plain attribute assignment (like
    ``section_name`` or ``edge_supports``), and the checker does not flag
    it on its own -- ``domain`` is a real Structure2D attribute
    (:func:`_data_attrs`), so :func:`_unknown_attributes` has nothing to
    say about it.

    But unlike ``section_name``, ``domain`` is not a free-standing property:
    plane and plate models use different DOFs per node (ux/uy/tz vs
    w/tx/ty), so reassigning it on a model that already has real content
    (:func:`~xdfem2d.template_api.has_geometry`) leaves every existing
    node/element meaning something different than it did the line before,
    silently -- the same risk :func:`~xdfem2d.template_api.domain_mismatch`
    and Structure2D.create_from_template's own ``has_geometry`` guard
    already refuse for a *template* call switching domain; this is the same
    refusal for the direct-assignment route those two miss entirely
    (27/09/2026, Matias: "para discutir. não alterar" -- discussed across
    a few turns, this half of it having a real gap, the create_from_template
    half turning out already covered)."""
    from .template_api import has_geometry
    if struc is None or not has_geometry(struc):
        return []
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == 'domain'
                and isinstance(node.targets[0].value, ast.Name)
                and node.targets[0].value.id in names):
            continue
        out.append(_problem(
            node.lineno, 'domain_reassignment',
            "the model already has geometry, and its domain cannot be "
            "changed under it: plane and plate models use different "
            "degrees of freedom per node, so every existing node/element "
            "would silently mean something else. Build the other domain's "
            "model separately (a fresh Structure2D(domain=...), or "
            "create_from_template on an empty model)"))
    return out


def _unknown_section_assignments(tree: ast.AST, struc) -> list[dict]:
    """``<element>.section_name = 'X'`` where no section 'X' exists.

    Pointing an element at another section is a plain attribute assignment
    (there is no ``set_section()``), so nothing validates it: the script
    runs, the element now names a section that is not there, and the solve
    fails later, far from the cause. Found live (03/10/2026): "adiciona
    material aço S355" came back as four ``section_name = 'S355'``
    assignments and a check that said "all valid", with no section 'S355'
    anywhere.

    A section counts as existing when the open model has it (bar, triangle or
    quad sections) or when the script calls a ``*_section`` creator with that
    literal as its name. Reported only with a model to compare against, and
    only for a literal name: a computed one is not checked."""
    if struc is None:
        return []
    known = (set(getattr(struc, 'sections', {}) or {})
             | set(getattr(struc, 'tri_sections', {}) or {})
             | set(getattr(struc, 'quad_sections', {}) or {}))
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and 'section' in node.func.attr):
            v = _literal(_arg(node, 'name', 0))
            if v:
                known.add(v)
    out = []
    for node in ast.walk(tree):
        # create_bar_element('Sec', n1, n2): the section is nowhere. The
        # reference check in _missing_ids_incremental sees this only when the
        # script also creates something under a literal id; with auto-numbered
        # nodes (the usual case) it treats the script as a fragment and says
        # nothing, so the same fact is checked here, and the caller drops the
        # duplicate when both speak.
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr in ('create_bar_element',
                                       'create_area_element')):
            v = _literal(_arg(node, 'section', 0))
            if v and v not in known:
                out.append(_problem(
                    node.lineno, 'unknown_id',
                    f"{node.func.attr}(): there is no section {v!r} in the model "
                    f"and none is created in this script. Create it first "
                    f"(create_rc_section, create_steel_section, ...), or use the "
                    f"name of one that exists"))
            continue
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == 'section_name'):
            continue
        v = _literal(node.value)
        if v is None or v in known:
            continue
        out.append(_problem(
            node.lineno, 'unknown_id',
            f"section_name = {v!r}: there is no section {v!r} in the model and "
            f"none is created in this script. Create it first "
            f"(create_rc_section, create_steel_section, ...), then point the "
            f"element at it. Nothing fails when this runs; the element just "
            f"names a section that is not there"))
    return out


def _edge_supports_names(tree: ast.AST) -> set:
    """Support names referenced by any ``<obj>.edge_supports = [...]``
    assignment in the script — the direct per-edge restraint on a geometry
    object (see ``GeoRectangle.edge_supports`` in models.py), a second way
    a support gets attached to the structure that ``assign_support`` calls
    do not cover: the assignment only takes effect when the object expands
    at solve time, so a script using it legitimately never calls
    ``assign_support`` at all. Without this, ``_completeness`` and
    ``_defined_but_never_used`` both misread a perfectly restrained
    edge_supports slab as a floating mechanism (found live, 2026-08, on
    every ``_tpl_slab_edges_obj``-based script)."""
    names: set = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Attribute)
                and node.targets[0].attr == 'edge_supports'):
            continue
        if not isinstance(node.value, (ast.List, ast.Tuple)):
            continue        # computed at run time — nothing to see here
        for elt in node.value.elts:
            v = _literal(elt)
            if v:
                names.add(v)
    return names


def _defined_but_never_used(calls, names: set,
                            edge_supports_names: frozenset = frozenset()
                            ) -> list[dict]:
    """Things a script creates and then never attaches to anything.

    One pair matters above the rest. ``add_support`` defines a support —
    a set of restraints with a name — and ``assign_support`` puts it on a
    node. Four of five language models asked to build a beam called the first
    and not the second, and so did every one of them that got everything else
    right.

    It is not their mistake so much as the API's shape: `add_support(name,
    ux=True, uy=True)` reads like it supports something. Nothing in the call
    says a node is still missing, the script runs, and the model comes out a
    mechanism — which fails much later as a singular matrix, if it fails
    visibly at all.
    """
    made, attached, first = {}, set(), {}
    for node in calls:
        fn = node.func
        if not isinstance(fn, ast.Attribute) or not isinstance(fn.value, ast.Name):
            continue
        if fn.value.id not in names:
            continue
        if fn.attr == 'add_support':
            v = _literal(_arg(node, 'name', 0))
            if v is not None:
                made[v] = node.lineno
        elif fn.attr == 'assign_support':
            v = _literal(_arg(node, 'support_name', 1))
            if v is not None:
                attached.add(v)
            first.setdefault('any', node.lineno)

    attached |= edge_supports_names

    # Only when *nothing* is assigned. A script that assigns some supports and
    # leaves another defined is carrying an unused definition, which is what a
    # model edited in the GUI looks like when exported — example-tension.x2d
    # has four supports and two assignments, legitimately. Assigning none is
    # the different thing: the author did not know the second step exists, and
    # the structure floats.
    if attached or not made:
        return []

    return [_problem(
        line, 'unattached_support',
        f"support {name!r} is defined and never assigned to a node, so it "
        f"restrains nothing. add_support() only describes the restraints; "
        f"assign_support(node_id, {name!r}) is what puts it on the structure. "
        f"Without it the model is a mechanism and the solve fails or, worse, "
        f"does not")
        for name, line in sorted(made.items(), key=lambda kv: kv[1])
        if name not in attached]


def _unknown_attributes(tree: ast.AST, names: set, methods: dict,
                        data_attrs: frozenset) -> list[dict]:
    """Attribute access on a Structure2D that names neither a method nor a
    real data attribute — the case the per-call loop in :func:`check`
    cannot see, because it only walks a ``Call``'s own ``.func``.

    Found for real: a script wrote ``for e in model.mesh['bars']:`` —
    ``model`` has no ``mesh`` attribute at all (bar elements live at
    ``model.bar_elements``/``model.bar_elements_by_id``); the checker gave
    no error because ``model.mesh`` is never a ``Call``, so it was never
    visited by the existing loop. This exists specifically for that shape:
    attribute access used as a *value* — subscripted, iterated, assigned,
    passed as an argument — not called.

    Deliberately does not re-check a ``Call``'s own ``.func`` (skipped via
    ``call_funcs``): that case already has its own message, with a
    closest-match suggestion scoped to method names only. Mixing the two
    candidate pools there would suggest a data attribute for what is
    obviously meant to be a call, or vice versa.
    """
    call_funcs = {id(n.func) for n in ast.walk(tree) if isinstance(n, ast.Call)}
    candidates = {**methods, **{a: None for a in data_attrs}}
    out = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or id(node) in call_funcs:
            continue
        if not isinstance(node.value, ast.Name) or node.value.id not in names:
            continue
        attr = node.attr
        if attr.startswith('__') or attr in methods or attr in data_attrs:
            continue
        near = _closest(attr, candidates)
        out.append(_problem(
            node.lineno, 'unknown_attribute',
            f"Structure2D has no attribute '{attr}'"
            + (f" — did you mean '{near}'?" if near else "")))
    return out


def _id_dict_attrs() -> frozenset:
    """Attributes of a Structure2D that are dictionaries keyed by id."""
    return frozenset(a for a, v in vars(Structure2D()).items()
                     if not a.startswith('_') and isinstance(v, dict))


def _integer_index_problems(tree: ast.AST, names: set) -> list[dict]:
    """``model.nodes[0]``: an integer used to index a dictionary keyed by id.

    ``model.nodes``, ``model.bar_elements_by_id``, ``model.sections`` and the
    like are dicts keyed by the id as text ('N1', 'E0'), not lists, so an
    integer index is a ``KeyError`` when the script runs, and the check said
    nothing (found live, 03/10/2026: ``create_bar_element(section='Sec',
    node_i=model.nodes[0], node_j=model.nodes[1])``, "checks out", then "not
    applied — KeyError: 0"). Only a literal integer is reported: a variable
    may well hold an id."""
    dicts = _id_dict_attrs()
    out = []
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Subscript)
                and isinstance(node.value, ast.Attribute)
                and isinstance(node.value.value, ast.Name)
                and node.value.value.id in names
                and node.value.attr in dicts):
            continue
        idx = node.slice
        if (isinstance(idx, ast.UnaryOp) and isinstance(idx.op, ast.USub)):
            idx = idx.operand
        if not (isinstance(idx, ast.Constant) and isinstance(idx.value, int)
                and not isinstance(idx.value, bool)):
            continue
        attr = node.value.attr
        out.append(_problem(
            node.lineno, 'integer_index',
            f"model.{attr} is a dict keyed by id as text ('N1', 'E0'), not a "
            f"list: model.{attr}[{ast.unparse(node.slice)}] raises KeyError. "
            f"Keep what the create_* call returns (n1 = model.create_node(...)) "
            f"and pass that, or index with the id as a string"))
    return out


# ---------------------------------------------------------------------------
# Security checks for the script editor
# ---------------------------------------------------------------------------

#: Modules the editor sandbox allows to import.
ALLOWED_IMPORTS: frozenset[str] = frozenset({
    'xdfem2d', 'math', 'numpy', 'itertools', 'functools',
})

#: Top-level callable names that must never appear in a user script.
_FORBIDDEN_CALLS: frozenset[str] = frozenset({
    'open', 'exec', 'eval', 'compile', '__import__',
    'breakpoint', 'input',
})


def _main_guard_lines(tree: ast.AST) -> frozenset:
    """Line numbers of nodes inside ``if __name__ == '__main__':`` blocks.

    Imports inside such a guard are never reachable when a script is exec-ed
    by the editor (because ``__name__`` is ``'<script>'``, not ``'__main__'``).
    Flagging them as forbidden would cause every exported script to fail the
    security check, so they are excluded.
    """
    guarded: set[int] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        # Pattern:  __name__ == '__main__'  or  '__main__' == __name__
        if isinstance(test, ast.Compare) and len(test.ops) == 1:
            left, right = test.left, test.comparators[0]
            is_name_eq = (
                isinstance(test.ops[0], ast.Eq) and (
                    (isinstance(left,  ast.Name) and left.id == '__name__' and
                     isinstance(right, ast.Constant) and right.value == '__main__') or
                    (isinstance(right, ast.Name) and right.id == '__name__' and
                     isinstance(left,  ast.Constant) and left.value == '__main__')
                )
            )
            if is_name_eq:
                for child in ast.walk(node):
                    if hasattr(child, 'lineno'):
                        guarded.add(child.lineno)
    return frozenset(guarded)


def _security_checks(tree: ast.AST) -> list[dict]:
    """AST-level safety checks for scripts run inside the editor.

    These are not about correctness — they are about the threat model stated
    in the design doc: protection against *accidental* escapes (dunder tricks,
    shell access via open/exec), not against a determined attacker who knows
    Python internals. The three rules together close the obvious surface.
    """
    out: list[dict] = []
    guarded = _main_guard_lines(tree)

    for node in ast.walk(tree):

        # 1. Import allowlist -----------------------------------------------
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            if node.lineno in guarded:
                continue   # inside if __name__ == '__main__': — never reached
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split('.')[0]
                if top not in ALLOWED_IMPORTS:
                    out.append(_problem(
                        node.lineno, 'forbidden_import',
                        f"import of '{alias.name}' is not allowed in the "
                        f"script editor. Allowed modules: "
                        + ', '.join(sorted(ALLOWED_IMPORTS))))
        elif isinstance(node, ast.ImportFrom):
            top = (node.module or '').split('.')[0]
            if top not in ALLOWED_IMPORTS:
                out.append(_problem(
                    node.lineno, 'forbidden_import',
                    f"'from {node.module} import …' is not allowed in the "
                    f"script editor. Allowed modules: "
                    + ', '.join(sorted(ALLOWED_IMPORTS))))

        # 2. Forbidden builtins ---------------------------------------------
        elif isinstance(node, ast.Call):
            fn = node.func
            name = None
            if isinstance(fn, ast.Name):
                name = fn.id
            elif isinstance(fn, ast.Attribute):
                name = fn.attr
            if name in _FORBIDDEN_CALLS:
                out.append(_problem(
                    node.lineno, 'forbidden_builtin',
                    f"calling '{name}' is not permitted in the script editor"))

        # 3. Dunder attribute access ----------------------------------------
        elif isinstance(node, ast.Attribute):
            if node.attr.startswith('__') and node.attr.endswith('__'):
                out.append(_problem(
                    node.lineno, 'dunder_access',
                    f"access to '{node.attr}' is not permitted in the script "
                    f"editor (dunder attributes are a common escape vector)"))

    return out


def check_editor(source: str, struc=None) -> list[dict]:
    """Like :func:`check` or :func:`check_increment` — whichever *source*'s
    own shape calls for — plus the security checks for the editor.

    Call this instead of ``check()``/``check_increment()`` directly when the
    source comes from the :class:`ScriptEditorDialog` — the extra security
    rules are not appropriate for the AI assistant or for offline validation
    of exported files, which call ``check``/``check_increment`` themselves.

    *struc* is the model the editor's "Mode 2 — snippet" would land on
    (``self._get_struc()`` there) — ignored when *source* is build-mode.
    Passing it, or not, used to be the only choice: the editor called
    ``check()`` unconditionally, whether or not the source built its own
    ``Structure2D()``. ``run_script`` already executes both shapes correctly
    (a plain ``model.`` snippet gets ``model`` injected as a copy of *struc*),
    but ``check()`` always reports 'no Structure2D is built in this script'
    for that same snippet and this ``_run()`` refused to run it — a script
    that "Open" (incremental mode) had staged from a working assistant reply
    could never be run from the editor, however correct it was.
    """
    source = normalize_source(source)
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [_problem(e.lineno or 0, 'syntax',
                         f"the file is not valid Python: {e.msg}")]

    security = _security_checks(tree)
    if security:
        # Return security findings immediately — do not run further checks on
        # a script that tries to escape the sandbox.
        return sorted(security, key=lambda p: p['line'])

    # All the existing correctness checks — for the shape this source is.
    if is_build_mode(source):
        return check(source)
    return check_increment(source, struc=struc)


def check(source: str) -> list[dict]:
    """Problems found in *source*, in the order they appear.

    An empty list means every call named a real method and could have been
    made with the arguments given. It does not mean the script is right.
    """
    source = normalize_source(source)
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [_problem(e.lineno or 0, 'syntax',
                         f"the file is not valid Python: {e.msg}")]

    methods = _methods()
    data_attrs = _data_attrs()
    names = _structure_names(tree)
    out: list[dict] = []

    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]

    # A script that never builds a Structure2D cannot be checked call by call,
    # and staying quiet about it is the worst thing this can do. Measured on
    # granite4.1:8b, the failure was not a wrong parameter but a wholly
    # invented API — ``import xdfem2d as struc`` and then ``struc.Material(…)``,
    # ``struc.Node(…)``, ``struc.solve()``. Not one call was on a Structure2D,
    # so there was nothing to compare and the reply came back "no problems
    # found", which the model passed on to the user as having verified it.
    #
    # This is the one finding that does not need a guess: whether the script
    # constructs the class is a fact about the source.
    if calls and not names:
        out.append(_problem(
            calls[0].lineno, 'no_model',
            "no Structure2D is built in this script, so none of its calls "
            "could be checked. A model is made with `s = Structure2D()` and "
            "filled in with its methods (s.add_node, s.add_material, …); "
            "there are no free functions and no Node or Material classes to "
            "construct"))

    for node in calls:
        fn = node.func
        # Tool names belong to the conversation and do not exist in any
        # script. The same run wrote `struc.model_check_tool(kind='', limit=10)`
        # into the file, which is a category error rather than a typo, and the
        # reader has no way to guess what was meant.
        if (isinstance(fn, (ast.Attribute, ast.Name))
                and str(getattr(fn, 'attr', None) or getattr(fn, 'id', ''))
                .endswith('_tool')):
            out.append(_problem(
                node.lineno, 'tool_in_script',
                f"'{getattr(fn, 'attr', None) or fn.id}' is one of the "
                "assistant's tools, not part of xdfem2D. Tools are called in "
                "the conversation; a script has only the library"))
            continue
        if not isinstance(fn, ast.Attribute) or not isinstance(fn.value, ast.Name):
            continue
        if fn.value.id not in names:
            continue
        name, line = fn.attr, node.lineno

        if name in data_attrs:
            continue
        if name not in methods:
            near = _closest(name, methods)
            out.append(_problem(
                line, 'unknown_method', _unknown_method_msg(name, near)))
            continue

        sig = inspect.signature(methods[name])
        args = [_placeholder(a) for a in node.args]
        starred = any(isinstance(a, ast.Starred) for a in node.args)
        kwargs = {}
        double_starred = False
        for kw in node.keywords:
            if kw.arg is None:
                double_starred = True          # **something: unknowable here
                continue
            kwargs[kw.arg] = _placeholder(kw.value)
        if starred or double_starred:
            # The call is assembled at run time; nothing can be said about it
            # without evaluating, and evaluating is the one thing off limits.
            continue
        try:
            sig.bind(None, *args, **kwargs)    # None stands for self
        except TypeError as e:
            out.append(_problem(line, 'arguments',
                                f"{name}(): {e}. It takes "
                                f"{_render(name, sig)}"))
        else:
            # bind() accepts any keyword once the signature has **kw; catch the
            # one it forwards to a stricter callee (add_plate_section(x=...)).
            fwd = (_forwarded_kw_problem(name, kwargs, line, sig, _render)
                   or _none_for_str_problem(name, sig, node, line))
            if fwd:
                out.append(fwd)
    edge_supports_names = _edge_supports_names(tree)
    out += _missing_ids(calls, names)
    out += _defined_but_never_used(calls, names, edge_supports_names)
    out += _unknown_attributes(tree, names, methods, data_attrs)
    out += _integer_index_problems(tree, names)
    # Completeness only applies to a *whole* model-building script (the shape the
    # assistant is asked for: a build() that returns the model, or one that
    # calculate()s it) — not to a fragment or a single-call snippet.
    whole = any(isinstance(n, ast.FunctionDef) and n.name == 'build'
                for n in ast.walk(tree)) or any(
        isinstance(n.func, ast.Attribute) and n.func.attr == 'calculate'
        for n in calls)
    if whole:
        out += _completeness(calls, names, bool(edge_supports_names))
    return sorted(out, key=lambda p: p['line'])


# The methods that assign a support to a node. A model with none is a
# mechanism — the single most common way a generated model is unsolvable, and
# the one completeness finding robust enough to assert against every exported
# script (the exporter builds geometry/elements at a lower level than a method
# name, so an element-presence check would misfire; supports always go through
# one of these calls).
_SUPPORT_METHODS = frozenset({
    'assign_support', 'pin', 'fix', 'roller', 'symm', 'support_edge',
    # create_support does add_support + assign_support in one call (see
    # Structure2D.create_support) — a script built from it never calls
    # assign_support separately, so it needs its own entry here or a
    # perfectly restrained model reads as a floating mechanism.
    'create_support',
})

# An elastic foundation is a legitimate alternative restraint — a slab on
# grade (add_area_spring, no supports at all) is correct, not incomplete;
# model_json.RESTRAINT_ONE_OF makes the same allowance on the JSON side. A
# script calling only one of these with no _SUPPORT_METHODS call is unusual
# but not wrong the way a script with neither is.
_SPRING_METHODS = frozenset({
    'add_node_spring', 'add_element_spring', 'add_line_element_spring',
    'add_area_spring',
})


def _completeness(calls, names: set,
                  has_edge_supports: bool = False) -> list[dict]:
    """Presence-only warning: a whole model with no support assigned and no
    spring either is a mechanism. Says nothing about whether the model is
    *right* — only that a restraint is absent, the script twin of
    :func:`xdfem2d.model_json.completeness_pieces`' mechanism check.

    ``has_edge_supports``: an ``<obj>.edge_supports = [...]`` assignment
    (see :func:`_edge_supports_names`) restrains the model on its own, at
    solve time, without any ``assign_support``-family call ever appearing
    in the script — a legitimate third way to be complete, alongside
    ``_SUPPORT_METHODS`` and ``_SPRING_METHODS``."""
    struct_calls = [n for n in calls
                    if isinstance(n.func, ast.Attribute)
                    and isinstance(n.func.value, ast.Name)
                    and n.func.value.id in names]
    if not struct_calls:
        return []
    if has_edge_supports:
        return []
    called = {n.func.attr for n in struct_calls}
    if not (called & (_SUPPORT_METHODS | _SPRING_METHODS)):
        return [_problem(
            min(n.lineno for n in struct_calls), 'incomplete',
            "no support is assigned — the structure floats (a mechanism). "
            "Assign one with pin/fix/roller/symm/support_edge, or "
            "assign_support after add_support (or, for a foundation model, "
            "an area spring via add_area_spring)")]
    return []


def _render(name: str, sig: inspect.Signature) -> str:
    parts = []
    for pname, p in sig.parameters.items():
        if pname == 'self':
            continue
        if p.default is inspect.Parameter.empty:
            parts.append(pname)
        else:
            parts.append(f"{pname}={p.default!r}")
    return f"{name}({', '.join(parts)})"


def _closest(name: str, methods: dict) -> str:
    """The nearest real method name, when there is an obvious one.

    Scored on the words, not the characters. Character similarity gets this
    exactly backwards on the names that matter: 'add_dist_load' scores 0.81
    against add_point_load and only 0.79 against add_distributed_load, so the
    obvious implementation sends the reader to the wrong method with
    confidence. Split on underscores and 'dist' is plainly a shortened
    'distributed', while 'point' is a different word.

    Nothing is suggested unless one candidate is clearly the best. A confident
    wrong suggestion is worse than none: it is read as knowledge.
    """
    import difflib

    def tokens(s):
        return [w for w in s.split('_') if w]

    def subseq(short, long):
        """Whether *short* is a token-subsequence of *long* (order kept, words
        may be skipped in long, prefix counts). 'add_rectangle' this way is a
        subsequence of 'add_geo_rectangle' — every word present, in order, with
        only 'geo' left out — which the positional match above misses because
        the inserted word shifts everything after it out of alignment."""
        if len(short) < 2:
            return False                             # 'add' alone matches all
        i = 0
        for w in long:
            if i < len(short):
                s = short[i]
                if s == w or s.startswith(w) or w.startswith(s):
                    i += 1
        return i == len(short)

    want = tokens(name)
    scored = []
    for cand in methods:
        have = tokens(cand)
        matched = 0
        for a, b in zip(want, have):
            if a == b or a.startswith(b) or b.startswith(a):
                matched += 1
        by_word = matched / max(len(want), len(have))
        by_char = difflib.SequenceMatcher(None, name, cand).ratio()
        score = max(by_word, by_char if by_char >= 0.9 else 0.0)
        # One name every word of the other, in order, with a word left out
        # (add_rectangle → add_geo_rectangle): strong, but below an exact match,
        # so the tie-breaker below still discards it when two candidates qualify.
        if subseq(want, have) or subseq(have, want):
            score = max(score, 0.9)
        scored.append((score, cand))
    scored.sort(reverse=True)
    if not scored or scored[0][0] < 0.8:
        return ''
    if len(scored) > 1 and scored[1][0] >= scored[0][0]:
        return ''                                # a tie is not a suggestion
    return scored[0][1]


# Real geometry-object primitives xdfem2D actually has — every one of them
# is node-driven (see models.py); named here only to describe them to a
# person reading the message below, not read live off any class, since the
# list itself is small and stable.
_GEO_PRIMITIVES = ("segment (straight, 2 nodes)",
                   "arc (circular, through 3 nodes)",
                   "multisegment (open/closed polyline, any number of nodes)",
                   "rectangle", "polygon")


def _unknown_method_msg(name: str, near: str) -> str:
    """Message for a call that names no real Structure2D method.

    A close-name suggestion (*near*) is the common, useful case — a typo or a
    superseded name — and is used whenever :func:`_closest` found one.
    ``add_geo_<shape>()`` for a shape xdfem2D simply does not have (a
    parabola, an ellipse, a spline, ...) has no close real name to suggest —
    every geometry object is one of a handful of primitives, none named after
    the curve or surface someone actually wants — so this used to just say
    "no such method" and stop there, leaving the reader nowhere to go
    (confirmed live: an assistant asked for a parabolic arc invented
    ``add_geo_parabola`` and got exactly that bare message back). This adds
    the one thing that IS true and actionable, for whichever family the name
    suggests: there is no dedicated curve or surface type for anything but
    the primitives below, but either can be approximated — a curve (parabola,
    ellipse arc, spline, ...) by computing points along it and handing them
    to add_geo_multisegment / add_geo_polyline as vertices; a bounded surface
    of any other outline (ellipse, rounded corners, ...) the same way, as the
    ordered boundary vertices of add_geo_polygon.
    """
    msg = f"Structure2D has no method '{name}'"
    if near:
        return msg + f" — did you mean '{near}'?"
    if 'support' in name and 'edge' not in name:
        # A per-node support call under some other name ('support_node',
        # 'set_support', 'apply_support', ...) — _closest has no confident
        # match to offer (the real names share no word with 'support_node':
        # pin/fix/roller/symm name the restraint, not the act of applying
        # one, and assign_support's own token overlap with 'support_node'
        # scores only 0.5, short of the 0.8 _closest requires), so this
        # would otherwise fall through to the bare "no such method" message
        # that made add_geo_parabola a dead end before _unknown_method_msg
        # existed. Named support methods are their own edge-support family
        # (support_edge, pin_edge, ...), already reachable via _closest's
        # word-overlap match when the name says 'edge' — this only fires for
        # the node case, so the two hints never both fire on the same name.
        return (msg + ". A support on a NODE is pin/fix/roller/symm "
               "(e.g. model.pin('N3')), or create_support for a custom "
               "combination — or, on an already-defined named support, "
               "assign_support(node_id, support_name). An EDGE of a "
               "surface object uses the separate pin_edge/fix_edge/"
               "roller_edge/symm_edge/support_edge family instead")
    if name.startswith('add_geo_') or name.startswith('create_geo_'):
        # A surface-sounding name gets pointed at the polygon approximation
        # first, a curve-sounding one at multisegment first — same two
        # sentences either way, just reordered so the likelier fit is read
        # first; neither is omitted, since the name alone is a guess.
        surface_hint = ("a bounded surface of any other outline (ellipse, "
                        "rounded corners, ...) as the ordered boundary "
                        "vertices of add_geo_polygon")
        curve_hint = ("a curve (parabola, ellipse arc, spline, ...) by "
                     "computing points along it and handing them to "
                     "add_geo_multisegment / add_geo_polyline as vertices")
        looks_surface = any(w in name for w in
                            ('surface', 'area', 'panel', 'plate', 'slab',
                             'ellipse', 'circle', 'oval'))
        order = (surface_hint, curve_hint) if looks_surface else (curve_hint, surface_hint)
        return (msg + ". xdfem2D's geometry objects are only "
               + ", ".join(_GEO_PRIMITIVES) + " — there is no dedicated type "
               "for anything else. Approximate it instead: " + order[0]
               + "; or " + order[1])
    return msg


def summary(problems: list[dict]) -> str:
    """The findings as lines of text, or a sentence saying there are none."""
    if not problems:
        return ("No problems found in the calls. This checks names and "
                "arguments only — not values, units, or whether the structure "
                "makes sense.")
    return "\n".join(f"line {p['line']}: {p['msg']}" for p in problems)


# ---------------------------------------------------------------------------
# Incremental checking — a snippet applied on top of a model already open,
# rather than a whole build()/Structure2D() script checked in isolation.
#
# Added for the assistant's incremental authoring mode: instead of writing
# the whole model every turn, one part at a time (geometry, then supports,
# then loads, ...) on top of the model already open in the application. The
# functions below are the difference between checking a fragment (which
# check() above deliberately treats with suspicion — see _missing_ids) and
# checking an increment, which is only a fragment when read alone; read
# together with the model it is meant to land on, it is complete.
# ---------------------------------------------------------------------------

def known_ids_from_struc(struc) -> set:
    """Every id that already exists in *struc*, across every category
    _CREATES/_REFERS knows about — the seed an incremental check() needs so
    a reference to something built in an *earlier* increment is not reported
    as unknown_id.

    Elements and load cases are lists on Structure2D, not dicts (nodes,
    materials and sections are) — their ids live in the matching
    ``*_by_id`` mapping, not in the list itself. Reading straight off
    ``bar_elements``/``tri_elements``/``load_cases`` here would silently
    see nothing and every increment referencing an existing element would
    read as a dangling reference.

    ``load_combinations`` is the same shape (a list, ``LoadCombination``
    objects carrying their own ``.id``) but has no ``*_by_id`` twin at all —
    it was missing here entirely, so every combination the model has ever
    created read as unknown the moment anything asked "does this id already
    exist". ``add_load_combination``/``create_load_combination`` only ever
    *create* a combination id (see ``_CREATES``), never reference an
    existing one, so this never broke an incremental check() in practice —
    but ``ai_context.hallucinated_ids_in_reply`` also seeds its "known ids"
    set from this same function, and there a combination id is very much
    referenced: ``model_overview_tool`` lists every combination by name, and
    an answer that just repeats one back (a normal, correct answer to "what
    combinations exist") was reported as inventing it.

    ``struc is None`` (nothing open yet — the very first increment of a new
    model) returns an empty set, same as if nothing had ever been created.
    """
    if struc is None:
        return set()
    ids: set = set()
    for coll in (struc.nodes, struc.materials, struc.sections,
                struc.tri_sections, struc.quad_sections, struc.supports,
                struc.bar_elements_by_id, struc.tri_elements_by_id,
                struc.quad_elements_by_id, struc.load_cases_by_id,
                struc.analysis_cases_by_id, struc.geometry_objects):
        ids |= set(coll.keys())
    ids |= {c.id for c in (struc.load_combinations or [])}
    return ids


# The element/target argument of every load whose target has a *fixed* kind,
# so a load can be caught landing on the wrong sort of thing. A bar/line load
# on a surface (a slab region), or an area load on a bar, both check out under
# the existence test above — the id is real — yet do nothing useful: the reason
# the assistant reached for add_distributed_load on a rectangle is that it did
# not register the rectangle is an area. Existence is not enough; kind is the
# second half of "does this id mean what the call needs it to mean".
_BAR_LOAD_TARGET = {
    'add_distributed_load': ('element_id', 0),
    'add_element_point_load': ('element_id', 0),
    # A line object (segment/arc/open polyline), not a bar element, but the
    # mismatch we catch — a surface passed here — is the same one.
    'add_line_distributed_load': ('object_id', 0),
}
_AREA_LOAD_TARGET = {
    'add_area_load': ('target_id', 0),
}


def _geo_is_surface(obj) -> bool:
    """Whether a geometry object encloses an area (a slab region) rather than
    being a line. Rectangles and polygons always do; a multisegment does only
    when closed; segments and arcs never do."""
    cls = type(obj).__name__
    if cls in ('GeoRectangle', 'GeoPolygon'):
        return True
    if cls == 'GeoMultisegment':
        return bool(getattr(obj, 'closed', False))
    return False


def _typed_ids(struc, calls, names) -> tuple[set, set]:
    """Two sets of ids — surface (area) and bar/line — drawn from the open
    model and from any geometry this increment creates itself. Only the ids
    whose kind is knowable are included; a mesh id built at run time is in
    neither set and so is never flagged."""
    area, line = set(), set()
    if struc is not None:
        area |= set(getattr(struc, 'tri_elements_by_id', {}))
        area |= set(getattr(struc, 'quad_elements_by_id', {}))
        line |= set(getattr(struc, 'bar_elements_by_id', {}))
        for oid, obj in getattr(struc, 'geometry_objects', {}).items():
            (area if _geo_is_surface(obj) else line).add(oid)
    # Geometry the increment creates before it loads it (create-then-load in
    # one increment) is classified from the call itself.
    for node in calls:
        fn = node.func
        if not isinstance(fn, ast.Attribute) or not isinstance(fn.value, ast.Name):
            continue
        if fn.value.id not in names:
            continue
        m = fn.attr
        if m in ('add_geo_rectangle', 'add_geo_polygon'):
            v = _literal(_arg(node, 'id', 0))
            if v is not None:
                area.add(v)
        elif m in ('add_geo_segment', 'add_geo_arc', 'add_geo_arc_3pts'):
            v = _literal(_arg(node, 'id', 0))
            if v is not None:
                line.add(v)
        elif m == 'add_geo_multisegment':
            v = _literal(_arg(node, 'id', 0))
            closed = _arg(node, 'closed', 2)
            if v is not None:
                is_closed = (isinstance(closed, ast.Constant)
                             and closed.value is True)
                (area if is_closed else line).add(v)
    return area, line


def _kind_mismatches(calls, names: set, area_ids: set,
                     line_ids: set) -> list[dict]:
    """Loads landing on the wrong kind of target: a bar/line load on a
    surface, or an area load on a bar/line. Only fires on an id whose kind is
    known (see :func:`_typed_ids`); an unclassifiable target is left alone."""
    out = []
    for node in calls:
        fn = node.func
        if not isinstance(fn, ast.Attribute) or not isinstance(fn.value, ast.Name):
            continue
        if fn.value.id not in names:
            continue
        method = fn.attr
        spec = _BAR_LOAD_TARGET.get(method)
        if spec is not None:
            v = _literal(_arg(node, *spec))
            if v is not None and v in area_ids:
                out.append(_problem(
                    node.lineno, 'wrong_kind',
                    f"{method}() puts a line/bar load on {spec[0]}={v!r}, but "
                    f"{v!r} is a surface (area) object, not a bar or line. A "
                    f"surface takes an area load — use add_area_load, or the "
                    f"create_uniform_load facade, instead"))
            continue
        spec = _AREA_LOAD_TARGET.get(method)
        if spec is not None:
            v = _literal(_arg(node, *spec))
            if v is not None and v in line_ids:
                out.append(_problem(
                    node.lineno, 'wrong_kind',
                    f"{method}() puts an area load on {spec[0]}={v!r}, but "
                    f"{v!r} is a bar or line, not a surface. A bar or line "
                    f"takes a line/bar load — use create_uniform_load or "
                    f"add_distributed_load (or the create_ facade) instead"))
    return out


# Every edge method that accepts the bottom/right/top/left vocabulary, and
# where its (object_id, edge) arguments sit — support_edge's own facade
# (pin_edge/fix_edge/roller_edge/symm_edge/free_edge) all share the same
# (object_id, edge) positions, and support_object_edge (what they all end up
# calling) does too.
_NAMED_EDGE_METHODS = {
    'support_edge': (('object_id', 0), ('edge', 1)),
    'support_object_edge': (('object_id', 0), ('edge', 1)),
    'pin_edge': (('object_id', 0), ('edge', 1)),
    'fix_edge': (('object_id', 0), ('edge', 1)),
    'roller_edge': (('object_id', 0), ('edge', 1)),
    'symm_edge': (('object_id', 0), ('edge', 1)),
    'free_edge': (('object_id', 0), ('edge', 1)),
}
_NAMED_EDGES = {'bottom', 'right', 'top', 'left'}


def _edge_object_corners(struc, calls, names: set) -> dict:
    """``{object_id: corner_count}`` for every rectangle/polygon knowable
    from the open model or from this increment's own ``add_geo_*`` calls —
    what :meth:`~xdfem2d.structure.Structure2D._edge_segments` actually
    counts before deciding whether a named edge (bottom/right/top/left) is
    even meaningful for that object. A rectangle is always 4 corners
    (``add_geo_rectangle``/``GeoRectangle`` — two *diagonal* corners are
    stored, but the shape itself always has four); a polygon is however
    many vertices it was given. An id whose count cannot be read off (a
    computed vertex list, an object of some other/unknown kind) is left out
    — silence, not a guess, same discipline as the rest of this module."""
    corners: dict = {}
    if struc is not None:
        for oid, obj in getattr(struc, 'geometry_objects', {}).items():
            cls = type(obj).__name__
            if cls == 'GeoRectangle':
                corners[oid] = 4
            elif cls == 'GeoPolygon':
                corners[oid] = len(getattr(obj, 'node_ids', []) or [])
    for node in calls:
        fn = node.func
        if not isinstance(fn, ast.Attribute) or not isinstance(fn.value, ast.Name):
            continue
        if fn.value.id not in names:
            continue
        method = fn.attr
        if method == 'add_geo_rectangle':
            v = _literal(_arg(node, 'id', 0))
            if v is not None:
                corners[v] = 4
        elif method == 'add_geo_polygon':
            v = _literal(_arg(node, 'id', 0))
            verts = _arg(node, 'vertices', 1)
            if v is not None and isinstance(verts, (ast.List, ast.Tuple)):
                corners[v] = len(verts.elts)
    return corners


def _named_edge_kind_mismatches(calls, names: set, corners: dict,
                                line_ids: set) -> list[dict]:
    """Two distinct ways an edge-support call can name a target that does
    not accept the ``edge`` it was given — both real ``ValueError``s from
    :meth:`~xdfem2d.structure.Structure2D`, caught here instead of only
    after the increment already "checked out" and then failed to apply:

    - the target is a known LINE object (a segment/arc/open multisegment —
      see :func:`_typed_ids`'s *line_ids*), which :meth:`support_object_edge`
      rejects outright ("is not a surface (rectangle/polygon)") regardless
      of what ``edge`` is — a name, an index, or ``'all'``, none of it
      matters when there is no perimeter of edges to select from at all.
    - the target IS a rectangle/polygon, but a bottom/right/top/left name
      is used on one that is not exactly a 4-corner shape (see
      :func:`_edge_object_corners`).

    Both only fire when the target's kind/corner count is actually known —
    an unclassifiable object, or one this increment neither creates nor an
    earlier one did, is left alone rather than guessed at."""
    out = []
    for node in calls:
        fn = node.func
        if not isinstance(fn, ast.Attribute) or not isinstance(fn.value, ast.Name):
            continue
        if fn.value.id not in names:
            continue
        spec = _NAMED_EDGE_METHODS.get(fn.attr)
        if spec is None:
            continue
        (oid_name, oid_pos), (edge_name, edge_pos) = spec
        oid = _literal(_arg(node, oid_name, oid_pos))
        if oid is None:
            continue
        if oid in line_ids:
            out.append(_problem(
                node.lineno, 'wrong_kind',
                f"{fn.attr}() targets object_id={oid!r}, but {oid!r} is a "
                f"line (segment/arc/multisegment), not a surface — edge "
                f"methods (support_edge, pin_edge, fix_edge, roller_edge, "
                f"symm_edge, support_object_edge, free_edge) only work on a "
                f"rectangle or polygon. For a line object's endpoints, use "
                f"pin/fix/roller/symm/create_support/assign_support on its "
                f"node ids instead"))
            continue
        edge_val = _arg(node, edge_name, edge_pos)
        if not (isinstance(edge_val, ast.Constant)
                and isinstance(edge_val.value, str)):
            continue
        edge = edge_val.value.lower()
        if edge not in _NAMED_EDGES:
            continue
        n = corners.get(oid)
        if n is not None and n != 4:
            out.append(_problem(
                node.lineno, 'wrong_kind',
                f"{fn.attr}() names edge {edge_val.value!r} on "
                f"object_id={oid!r}, but {oid!r} has {n} corners, not 4 — "
                f"named edges (bottom/right/top/left) only work on a "
                f"rectangle or a 4-vertex polygon. Use an edge index "
                f"(0-based, perimeter order) or 'all' instead"))
    return out


def _missing_ids_incremental(calls, names: set, known: set) -> list[dict]:
    """Like :func:`_missing_ids`, seeded with ids that already exist in the
    model an increment is landing on, instead of starting from nothing.

    The one behavioural difference from :func:`_missing_ids` that matters:
    that function stays silent when the script creates nothing at all — a
    fragment is not worth flagging, since it was never going to define what
    it refers to. Here, "creates nothing new" is the *normal* shape of a
    pure-loads or pure-supports increment — everything it needs was already
    created by an earlier one — so the same silence would defeat the point.
    Silence is kept only for the one case it still means the same thing: a
    genuinely empty model (``known`` empty, this increment creates nothing
    either) checked with nothing to check against.
    """
    created, used = set(known), []
    for node in calls:
        fn = node.func
        if not isinstance(fn, ast.Attribute) or not isinstance(fn.value, ast.Name):
            continue
        if fn.value.id not in names:
            continue
        method = fn.attr
        if method == 'create_node_list':
            ids = _node_list_created_ids(node, created)
            if ids:
                created.update(ids)
        elif method in _CREATES:
            v = _literal(_arg(node, *_CREATES[method]))
            if v is not None:
                created.add(v)
        for argname, pos in _REFERS.get(method, ()):
            for v in _literal_values(_arg(node, argname, pos)):
                used.append((node.lineno, method, argname, v))

    if not created:
        return []

    problems = []
    for line, method, argname, value in used:
        if value in created:
            continue
        kind = argname.removesuffix('_id').removesuffix('_name').replace('_', ' ')
        problems.append(_problem(line, 'unknown_id',
                         f"{method}() refers to {argname}={value!r} — no {kind} {value!r} "
                         f"exists yet: not created earlier in this increment, and not "
                         f"already in the model. Create it first. Nothing will fail when "
                         f"this runs — the reference is simply dangling, and whatever "
                         f"depends on it is silently absent"))
    return problems


# Calls that name a bar as the target of a load. create_bar_point_load's own
# `elements` is already covered by _REFERS; these two were left out on purpose
# because their target may also be a geometry object or a mesh id (see the
# comment in _REFERS), so they are checked only where that cannot be so.
_BAR_TARGET_ARGS = {
    'create_uniform_load': ('targets', 0),
    'create_bar_distributed_load': ('elements', 0),
}

# A call that makes a bar or an area, or a region, with an automatic id the
# checker cannot see: after one of these a literal id may be a real one.
_AUTO_ID_CREATORS = frozenset({
    'create_bar_element', 'create_area_element', 'create_polygon',
    'add_bar_element', 'add_tri_element', 'add_quad_element',
    'add_area_element', 'add_geo_rectangle', 'add_geo_polygon',
    'add_geo_segment', 'add_geo_arc', 'add_geo_arc_3pts',
    'add_geo_multisegment', 'add_panel'})


def _dangling_bar_targets(calls, names: set, struc) -> list[dict]:
    """A load put on a bar id the open model does not have.

    ``create_uniform_load('B1', ...)`` on a model whose bar is ``E1`` runs the
    check without a word and fails only when it is applied. Checked in a model
    made of bars alone, and only when the increment creates no bar, area or
    region of its own: in any other case the id may be an automatic one
    (``create_bar_element`` numbers itself) or a mesh id that only exists at
    run time, and the checker cannot tell.
    """
    if struc is None:
        return []
    bars = set(getattr(struc, 'bar_elements_by_id', {}) or {})
    if (not bars or getattr(struc, 'geometry_objects', None)
            or getattr(struc, 'tri_elements_by_id', None)
            or getattr(struc, 'quad_elements_by_id', None)):
        return []
    for node in calls:
        fn = node.func
        if (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                and fn.value.id in names and fn.attr in _AUTO_ID_CREATORS):
            return []
    out = []
    for node in calls:
        fn = node.func
        if not isinstance(fn, ast.Attribute) or not isinstance(fn.value, ast.Name):
            continue
        if fn.value.id not in names or fn.attr not in _BAR_TARGET_ARGS:
            continue
        argname, pos = _BAR_TARGET_ARGS[fn.attr]
        for v in _literal_values(_arg(node, argname, pos)):
            if v in bars:
                continue
            listed = ', '.join(sorted(bars)[:6]) + (', ...' if len(bars) > 6 else '')
            out.append(_problem(
                node.lineno, 'unknown_id',
                f"{fn.attr}() refers to {argname}={v!r} — the model has no bar "
                f"{v!r}; its bars are {listed}. Read the ids with geometry_tool "
                f"before naming a bar: they are not always B1, B2"))
    return out


def _undefined_variables(tree, names: set, calls) -> list[dict]:
    """A variable passed to a model call that this increment never defines.

    An increment runs on its own, so ``b1`` from an earlier reply does not
    exist in it: the call raises ``NameError`` when it is applied. The checker
    used to let it through, and the model was told nothing until the user saw
    the error. Only a bare name (or a list of them) given directly to a call on
    the model is looked at, so the rule stays narrow.
    """
    import builtins
    defined = set(dir(builtins)) | set(names)
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, (ast.Store, ast.Del)):
            defined.add(node.id)
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            defined.add(node.name)
            if not isinstance(node, ast.ClassDef):
                a = node.args
                for arg in a.posonlyargs + a.args + a.kwonlyargs:
                    defined.add(arg.arg)
                for arg in (a.vararg, a.kwarg):
                    if arg is not None:
                        defined.add(arg.arg)
        elif isinstance(node, ast.Lambda):
            a = node.args
            for arg in a.posonlyargs + a.args + a.kwonlyargs:
                defined.add(arg.arg)
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            for alias in node.names:
                defined.add((alias.asname or alias.name).split('.')[0])
        elif isinstance(node, ast.ExceptHandler) and node.name:
            defined.add(node.name)
    out, seen = [], set()
    for node in calls:
        fn = node.func
        if not isinstance(fn, ast.Attribute) or not isinstance(fn.value, ast.Name):
            continue
        if fn.value.id not in names:
            continue
        values = list(node.args) + [kw.value for kw in node.keywords]
        flat = []
        for v in values:
            flat += list(v.elts) if isinstance(v, (ast.List, ast.Tuple)) else [v]
        for v in flat:
            if (isinstance(v, ast.Name) and v.id not in defined
                    and (node.lineno, v.id) not in seen):
                seen.add((node.lineno, v.id))
                out.append(_problem(
                    node.lineno, 'undefined_name',
                    f"{fn.attr}() uses the variable {v.id!r}, which this "
                    f"increment never defines: variables from an earlier reply "
                    f"do not exist here. Name the object by its id string (read "
                    f"the ids with geometry_tool), or create it in this increment"))
    return out


def _unknown_coefficient_cases(calls, names: set, struc) -> list[dict]:
    """``coefficients`` keys of an analysis case that name no load case.

    ``create_analysis_case('AC1', coefficients={'G': 1.0})`` runs without
    complaint even when no load case 'G' exists: the case is stored, points at
    nothing, and the analysis solves an unloaded model. A small model saw this
    happen with keys such as 'Permanent' or 'Dead' that it invented. Only a
    literal dict with literal keys is read, and nothing is said when the
    increment creates a load case without a literal id (auto-numbered), since
    then the key may well be that case.
    """
    known = set(getattr(struc, 'load_cases_by_id', {}) or {})
    auto = False
    for node in calls:
        fn = node.func
        if (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                and fn.value.id in names
                and fn.attr in ('create_load_case', 'add_load_case')):
            v = _literal(_arg(node, 'id', 0))
            if isinstance(v, str):
                known.add(v)
            else:
                auto = True
    if auto:
        return []
    out = []
    for node in calls:
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)
                and fn.value.id in names
                and fn.attr in ('create_analysis_case', 'add_analysis_case')):
            continue
        kind = _literal(_arg(node, 'analysis_type', 1))
        if kind is not None and kind not in ('Linear', 'NonLinear', 'Mass'):
            continue
        coeffs = _arg(node, 'coefficients', 2)
        if not isinstance(coeffs, ast.Dict):
            continue
        for key in coeffs.keys:
            k = _literal(key) if key is not None else None
            if isinstance(k, str) and k not in known:
                have = ', '.join(sorted(map(str, known))) or 'none yet'
                out.append(_problem(
                    node.lineno, 'unknown_id',
                    f"{fn.attr}() weighs the load case {k!r}, which does not "
                    f"exist (load cases: {have}). The keys of coefficients are "
                    f"load case ids: read them with model_overview_tool, or "
                    f"create the load case first"))
    return out


_NOT_PLAIN = object()


def _plain_value(node):
    """The value a literal in the source writes (a number, text, True/False/
    None, a list, tuple or dict of those), or _NOT_PLAIN. Read from the syntax
    tree: nothing is evaluated, this module never runs what it checks."""
    if isinstance(node, ast.Constant):
        return node.value
    if (isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd))
            and isinstance(node.operand, ast.Constant)
            and isinstance(node.operand.value, (int, float))
            and not isinstance(node.operand.value, bool)):
        return -node.operand.value if isinstance(node.op, ast.USub) else node.operand.value
    if isinstance(node, (ast.List, ast.Tuple)):
        items = [_plain_value(e) for e in node.elts]
        return _NOT_PLAIN if any(i is _NOT_PLAIN for i in items) else items
    if isinstance(node, ast.Dict):
        keys = [_plain_value(k) if k is not None else _NOT_PLAIN for k in node.keys]
        vals = [_plain_value(v) for v in node.values]
        if any(x is _NOT_PLAIN for x in keys + vals):
            return _NOT_PLAIN
        return dict(zip(keys, vals))
    return _NOT_PLAIN


def _template_effect(calls, names: set, struc):
    """(struc the rest of the increment is read against, problems) for a
    ``create_from_template`` call.

    The call puts a whole model in place of an empty one, so the ids the next
    lines use (``'E0'``, ``'R.p2'``, the load case ``'SW'``) exist only after it
    has run. Read against the model as it is now, every one of them would be
    reported as missing. So the template is built here, from the literal
    arguments, and the rest of the increment is checked against that model.
    A keyword that is not a literal is left at its default: the ids the
    template gives do not depend on most of them, and a wrong guess costs a
    missed report, never a false one on a real id.

    Also reported: a kind or parameter it does not have (with what it takes),
    a value it cannot use, a second call, and a model that already has
    geometry, all of which would raise when the increment runs.
    """
    tcalls = [n for n in calls
              if isinstance(n.func, ast.Attribute)
              and isinstance(n.func.value, ast.Name)
              and n.func.value.id in names
              and n.func.attr == 'create_from_template']
    if not tcalls:
        return struc, []
    from xdfem2d.template_api import build_template, domain_mismatch, has_geometry
    out = []
    for extra in tcalls[1:]:
        out.append(_problem(
            extra.lineno, 'template_twice',
            "create_from_template() is called more than once: it fills an "
            "empty model, and after the first call the model is not empty"))
    node = tcalls[0]
    if struc is not None and has_geometry(struc):
        out.append(_problem(
            node.lineno, 'template_not_empty',
            "create_from_template() only fills an empty model, and this one "
            "already has geometry: build on it with create_node, "
            "create_bar_element, create_polygon and the other create_* calls"))
        return struc, out
    kind = _arg(node, 'kind', 0)
    kind = kind.value if isinstance(kind, ast.Constant) else None
    if not isinstance(kind, str):
        return struc, out
    mismatch = domain_mismatch(kind, struc) if struc is not None else ''
    if mismatch:
        out.append(_problem(node.lineno, 'template_domain', mismatch))
        return struc, out
    kwargs = {}
    for kw in node.keywords:
        if kw.arg is None or kw.arg == 'kind':
            continue
        value = _plain_value(kw.value)
        if value is not _NOT_PLAIN:
            kwargs[kw.arg] = value
    try:
        return build_template(kind, **kwargs), out
    except ValueError as e:
        out.append(_problem(node.lineno, 'arguments', str(e)))
        return struc, out


def check_increment(source: str, struc=None) -> list[dict]:
    """Like :func:`check`, for one increment of the assistant's incremental
    authoring mode — plain ``model.`` calls meant to land on *struc* (the
    model already open), never a ``Structure2D()``/``def build():`` of its
    own.

    Two differences from :func:`check`, both because an increment is read
    together with the model it targets rather than in isolation:

    - ``model`` is always accepted as a valid receiver name, whether or not
      this source constructs one — :func:`script_runner.run_script` injects
      it (a deep copy of *struc*, or a fresh ``Structure2D()`` when *struc*
      is ``None``) before the increment ever runs, so it exists by the time
      any of these calls execute even though nothing here creates it.
    - dangling-reference checking (:func:`_missing_ids_incremental`) is
      seeded with :func:`known_ids_from_struc`, so a reference to something
      an *earlier* increment created is not mistaken for one this increment
      was going to define and never did.

    Deliberately does not run :func:`_defined_but_never_used` or
    :func:`_completeness` — a support defined in one increment and assigned
    in the next is incomplete only until the *last* increment of the
    sequence, and checking that from an isolated fragment of source would
    misread every such sequence as broken partway through. See
    :func:`completeness_from_struc`, which asks the model itself instead of
    trying to follow that across several separate pieces of text.
    """
    source = normalize_source(source)
    try:
        tree = ast.parse(source)
    except SyntaxError as e:
        return [_problem(e.lineno or 0, 'syntax',
                         f"the file is not valid Python: {e.msg}")]

    methods = _methods()
    data_attrs = _data_attrs()
    names = _structure_names(tree) | {'model'}
    out: list[dict] = []

    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    struc, template_problems = _template_effect(calls, names, struc)
    out += template_problems

    for node in calls:
        fn = node.func
        if (isinstance(fn, (ast.Attribute, ast.Name))
                and str(getattr(fn, 'attr', None) or getattr(fn, 'id', ''))
                .endswith('_tool')):
            out.append(_problem(
                node.lineno, 'tool_in_script',
                f"'{getattr(fn, 'attr', None) or fn.id}' is one of the "
                "assistant's tools, not part of xdfem2D. Tools are called in "
                "the conversation; a script has only the library"))
            continue
        if not isinstance(fn, ast.Attribute) or not isinstance(fn.value, ast.Name):
            continue
        if fn.value.id not in names:
            continue
        name, line = fn.attr, node.lineno

        if name in data_attrs:
            continue
        if name not in methods:
            near = _closest(name, methods)
            out.append(_problem(
                line, 'unknown_method', _unknown_method_msg(name, near)))
            continue

        sig = inspect.signature(methods[name])
        args = [_placeholder(a) for a in node.args]
        starred = any(isinstance(a, ast.Starred) for a in node.args)
        kwargs = {}
        double_starred = False
        for kw in node.keywords:
            if kw.arg is None:
                double_starred = True
                continue
            kwargs[kw.arg] = _placeholder(kw.value)
        if starred or double_starred:
            continue
        try:
            sig.bind(None, *args, **kwargs)
        except TypeError as e:
            out.append(_problem(line, 'arguments',
                                f"{name}(): {e}. It takes "
                                f"{_render(name, sig)}"))
        else:
            fwd = (_forwarded_kw_problem(name, kwargs, line, sig, _render)
                   or _none_for_str_problem(name, sig, node, line))
            if fwd:
                out.append(fwd)

    out += _missing_ids_incremental(calls, names, known_ids_from_struc(struc))
    _area_ids, _line_ids = _typed_ids(struc, calls, names)
    out += _kind_mismatches(calls, names, _area_ids, _line_ids)
    out += _named_edge_kind_mismatches(
        calls, names, _edge_object_corners(struc, calls, names), _line_ids)
    out += _dangling_bar_targets(calls, names, struc)
    out += _undefined_variables(tree, names, calls)
    out += _unknown_coefficient_cases(calls, names, struc)
    out += _unknown_attributes(tree, names, methods, data_attrs)
    out += _integer_index_problems(tree, names)
    out += _domain_reassignment_problems(tree, names, struc)
    for extra in _unknown_section_assignments(tree, struc):
        # the reference check may have said the same about this call already
        if not any(q['line'] == extra['line'] and q['kind'] == 'unknown_id'
                   and 'section' in q['msg'] and extra['msg'].split("'")[1:2]
                   and f"'{extra['msg'].split(chr(39))[1]}'" in q['msg']
                   for q in out):
            out.append(extra)
    return sorted(out, key=lambda p: p['line'])


def completeness_from_struc(struc) -> list[dict]:
    """The whole-model "does this float" check, asked of the model itself
    rather than followed through several increments of source text.

    The script-editor equivalent (:func:`_completeness`, together with
    :func:`_defined_but_never_used`) reads this off the AST because that is
    all it has — a script checked before it ever runs. An incremental
    sequence has something better by the time the last increment is about
    to be offered: the actual :class:`~xdfem2d.structure.Structure2D` every
    earlier increment has already been applied to. Asking it directly means
    a support defined in one increment and assigned two increments later is
    never misread as incomplete on the increments in between — there is
    nothing to misread, because this is only ever called once, at the end
    of a module sequence, not after each increment.
    """
    problems: list[dict] = []
    defined = set(struc.supports.keys())
    attached = {sa.support_name for sa in struc.support_assignments}
    for name in sorted(defined - attached):
        problems.append(_problem(
            0, 'unattached_support',
            f"support {name!r} is defined and never assigned to a node, so "
            f"it restrains nothing"))
    has_springs = bool(struc.node_springs or struc.element_springs
                       or struc.tri_area_springs or struc.quad_area_springs)
    has_edge_supports = any(getattr(o, 'edge_supports', None)
                            for o in struc.geometry_objects.values())
    if not attached and not has_springs and not has_edge_supports:
        problems.append(_problem(
            0, 'incomplete',
            "no support is assigned anywhere in the model — it floats (a "
            "mechanism)"))
    return problems


def increment_has_template(code: str) -> bool:
    """True when *code* calls ``create_from_template`` on the model."""
    try:
        tree = ast.parse(normalize_source(code))
    except SyntaxError:
        return False
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
               and n.func.attr == 'create_from_template'
               for n in ast.walk(tree))


def _insert_increment(struc, code, insert_at, join, join_tol) -> dict:
    """Build *code* on an empty model and merge it into *struc* at *insert_at*."""
    from xdfem2d.structure import Structure2D
    from xdfem2d.import_io import import_extend
    from xdfem2d.script_runner import run_script

    def fail(**kw):
        return {'ok': False, 'struc': struc, 'problems': [], 'error': None,
                'stdout': '', **kw}

    empty = Structure2D()
    problems = check_increment(code, struc=empty)
    if problems:
        return fail(problems=problems)
    try:
        built, stdout = run_script(code, struc=empty)
        if built.nodes:
            dx = insert_at[0] - min(n.x for n in built.nodes.values())
            dy = insert_at[1] - min(n.y for n in built.nodes.values())
            for n in built.nodes.values():
                n.x += dx
                n.y += dy
        # tol < 0 welds nothing (no distance is <= it); colliding ids are
        # renumbered by the merge either way.
        unified, report = import_extend(
            struc, built, weld='coords', tol=join_tol if join else -1.0)
    except Exception as e:                                     # noqa: BLE001
        return fail(error=f"{type(e).__name__}: {e}")

    def new(attr_old, attr_new):
        return set(attr_new) - set(attr_old)
    inserted = {
        'nodes': new(struc.nodes, unified.nodes),
        'elements': (new(struc.bar_elements_by_id, unified.bar_elements_by_id)
                     | new(struc.tri_elements_by_id, unified.tri_elements_by_id)
                     | new(struc.quad_elements_by_id, unified.quad_elements_by_id)),
        'objects': new(struc.geometry_objects, unified.geometry_objects),
    }
    return {'ok': True, 'struc': unified, 'problems': [], 'error': None,
            'stdout': stdout, 'inserted': inserted, 'report': report}


def check_increment_for_apply(source: str, struc=None) -> list[dict]:
    """:func:`check_increment` as the assistant's Apply button will run it.

    A template over a model that already has geometry is not an error there:
    Apply asks whether to insert it or replace the model, and either way the
    template is built on an empty model. So such a source is checked against
    an empty one; every other source is checked against *struc* as before.
    """
    from xdfem2d.template_api import has_geometry
    if (struc is not None and has_geometry(struc)
            and increment_has_template(source)):
        from xdfem2d.structure import Structure2D
        struc = Structure2D()
    return check_increment(source, struc=struc)


def apply_increment(struc, code: str, replace_model: bool = False,
                    insert_at=None, join: bool = True,
                    join_tol: float = 1e-3,
                    weld_tol: float | None = 1e-3) -> dict:
    """Validate-then-run one increment, without side effects on *struc* when
    either step fails.

    Two lines of defence, in order: :func:`check_increment` first, so a
    problem visible from the source alone (an unknown method, a dangling
    reference) never reaches ``exec`` at all; then
    :func:`~xdfem2d.script_runner.run_script`, which deep-copies *struc*
    before running the increment against the copy — so an exception partway
    through (a duplicate id, for instance: ``add_node`` raises when the id
    is already taken, and nothing upstream of it validates that in
    advance) never leaves *struc* itself half-mutated. Either failure
    returns *struc* unchanged, under the same key (``'struc'``), so a
    caller does not need two different ways to recover the model to keep
    working with.

    Returns a dict with ``ok`` (bool), ``struc`` (the result on success, the
    original *struc* unchanged on failure), ``problems`` (from
    :func:`check_increment`, when that is why it failed), ``error`` (the
    exception text, when running is why it failed) and ``stdout`` (anything
    the increment printed, on success).

    A template only fills an empty model, so over one that already has
    geometry the caller picks what to do (it asked the user):
    *replace_model* runs the increment on a fresh empty model, replacing
    *struc*; *insert_at* ``(x, y)`` builds the increment on a fresh empty
    model, moves its lower-left corner to that point and merges it into
    *struc* (:func:`~xdfem2d.import_io.import_extend`), welding nodes within
    *join_tol* of an existing one when *join* is true. On a successful insert
    the result carries ``inserted`` — ``{'nodes': set, 'elements': set,
    'objects': set}`` of the ids that are new in the merged model — and
    ``report``, the :class:`~xdfem2d.import_io.WeldReport`.

    *weld_tol* (metres; the Preferences' coincident-node tolerance, ``None`` to
    switch it off): a ``create_node`` that lands within it of a node already in
    the model returns that node, so the bars of the increment connect to the
    model instead of sitting on a duplicate. The result's ``welded`` lists the
    ids that were reused.
    """
    if insert_at is not None and increment_has_template(code):
        return _insert_increment(struc, code, insert_at, join, join_tol)
    base = struc
    if replace_model and increment_has_template(code):
        # A template fills an empty model and replaces its whole state; to
        # apply one over an existing model the caller (who asked the user)
        # opts in and the increment runs on a fresh empty one instead.
        from xdfem2d.structure import Structure2D
        base = Structure2D()
    problems = check_increment(code, struc=base)
    if problems:
        return {'ok': False, 'struc': struc, 'problems': problems,
               'error': None, 'stdout': ''}

    from xdfem2d.script_runner import run_script
    # run_script deep-copies *base*, so the weld settings travel with the copy;
    # they are taken off the original at once and off the result below.
    base._script_weld_tol = weld_tol
    base._script_weld_log = []
    try:
        new_struc, stdout = run_script(code, struc=base)
    except Exception as e:                                     # noqa: BLE001
        return {'ok': False, 'struc': struc, 'problems': [],
               'error': f"{type(e).__name__}: {e}", 'stdout': '', 'welded': []}
    finally:
        del base._script_weld_tol, base._script_weld_log
    welded = list(getattr(new_struc, '_script_weld_log', []))
    for attr in ('_script_weld_tol', '_script_weld_log'):
        if hasattr(new_struc, attr):
            delattr(new_struc, attr)

    return {'ok': True, 'struc': new_struc, 'problems': [], 'error': None,
           'stdout': stdout, 'welded': welded}
