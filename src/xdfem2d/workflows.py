"""
Headless workflows for variants & phasing (Phase 5 — controller layer).

The GUI (PySide6) binds to these functions; they contain no Qt and are fully
testable. They wrap the Phase 1–4 engines (variants, import_io, phasing) and the
structure's registered overlays so the UI only has to gather inputs and render
results.
"""
from __future__ import annotations

from .models import Variant
from .variants import solve_variants, combine_across_variants, CombTerm
from .phasing import solve_sequence, solve_phase
from .import_io import import_model


# ---------------------------------------------------------------------------
# Variants (Mode A)
# ---------------------------------------------------------------------------

def run_variant_combination(struc, terms, op,
                            missing: str = 'zero',
                            allow_geometry_mismatch: bool = False,
                            precomputed: dict = None) -> dict:
    """Solve the structure's registered variants involved in *terms* and combine.

    terms : list of (variant_id, case_id, factor) tuples or CombTerm objects.
    precomputed : optional ``{variant_id: VariantResult}`` of already-solved
        variants (e.g. from the Run that just solved base + all variants);
        those are reused instead of being solved a second time. The caller is
        responsible for their freshness (solve AFTER syncing with the base).
    Returns the combined result dict.
    """
    norm = [t if isinstance(t, CombTerm) else CombTerm(*t) for t in terms]
    needed = {t.variant_id for t in norm}
    variants = [struc.variants[vid] for vid in needed if vid in struc.variants]
    # Never combine stale models: reconcile each involved variant with the
    # base first (guarded — a no-op when nothing changed).
    for v in variants:
        sync_variant_with_base(struc, v)
    vres = {vid: r for vid, r in (precomputed or {}).items() if vid in needed}
    # A bare 'BASE' id (not registered) means the unmodified structure.
    extra = [Variant(id=vid) for vid in needed
             if vid not in struc.variants and vid not in vres]
    to_solve = [v for v in variants if v.id not in vres] + extra
    if to_solve:
        vres.update(solve_variants(struc, to_solve, struc.support_sets))
    return combine_across_variants(vres, norm, op, missing=missing,
                                   allow_geometry_mismatch=allow_geometry_mismatch)


def combination_validity(struc, variant_ids, op) -> dict:
    """Report whether *op* is valid for the given variants (for UI warnings).

    Returns ``{'same_stiffness': bool, 'same_geometry': bool, 'linear_ok': bool,
    'message': str}`` without solving load results (geometry/stiffness only).
    """
    from .variants import _geometry_sig
    hashes, geoms = set(), set()
    for vid in variant_ids:
        v = struc.variants.get(vid)
        if v is not None and getattr(v, 'model', None) is not None:
            # The model is authoritative — hash it directly, no deep copy
            # (derive_variant would clone the whole structure per call; this
            # runs on every keystroke of the Combine editor).
            sync_variant_with_base(struc, v)   # guarded no-op when in sync
            sub = v.model
        elif (v is None or (getattr(v, 'active_elements', None) is None
                            and not getattr(v, 'support_set_id', None))):
            # 'BASE' or a config-less variant: the unmodified structure.
            sub = struc
        else:
            # Declared config without a model: derive (copies, but rare).
            sub = struc.derive_variant(v, struc.support_sets)
        hashes.add(sub.stiffness_hash())
        geoms.add(_geometry_sig(sub))
    same_k = len(hashes) <= 1
    same_g = len(geoms) <= 1
    # Bar internal forces are always combinable; only displacements/reactions are
    # not meaningful when the stiffness differs.
    if not same_k:
        msg = ("OK for bar forces (N/V/M). Different stiffness → deformed shape "
               "and reactions are not meaningful for this combination.")
    elif not same_g:
        msg = ("OK for bar forces. Variants have different geometry; bars present "
               "in only some variants are handled per the 'missing' rule.")
    else:
        msg = "OK — same stiffness, all views valid."
    return {'same_stiffness': same_k, 'same_geometry': same_g,
            'linear_ok': True, 'message': msg}


# ---------------------------------------------------------------------------
# Import (Phase 2)
# ---------------------------------------------------------------------------

def extend_base_with_file(base, other, **kw):
    """Extend *base* with a model from *other* (path or Structure2D). Returns
    ``(unified, report)``. See :func:`xdfem2d.import_io.import_extend`."""
    from .import_io import import_extend
    from .structure import Structure2D
    if isinstance(other, str):
        other = Structure2D.load(other)
    return import_extend(base, other, **kw)


def import_file_as_variant(struc, other, overlay_id: str, **kw):
    """Import *other* (path or Structure2D) into a COPY of struc, registering the
    resulting variant and support set. Returns ``(unified, variant, report)``.

    The unified structure is returned (caller decides whether to adopt it as the
    working model); the variant/support-set are registered on it.
    """
    unified, variant, sset, report = import_model(struc, other, overlay_id, **kw)
    unified.add_support_set(sset)
    unified.add_variant(variant)
    return unified, variant, report


# ---------------------------------------------------------------------------
# Phasing (Modes B / sequences)
# ---------------------------------------------------------------------------

def run_sequence(struc, seq_id: str) -> dict:
    """Run a registered construction sequence by id."""
    seq = struc.construction_sequences.get(seq_id)
    if seq is None:
        raise KeyError(f"Construction sequence '{seq_id}' not found.")
    return solve_sequence(struc, seq, struc.support_sets)


def _strip_overlays(m):
    m.variants = {}
    m.support_sets = {}
    m.construction_sequences = {}
    m.variant_combinations = {}


def _clear_actions(m):
    m.load_cases = []
    m.load_cases_by_id = {}
    m.point_loads = []
    m.distributed_loads = []
    m.element_point_loads = []
    m.support_settlements = []
    m.temperature_loads = []
    m.analysis_cases = []
    m.analysis_cases_by_id = {}
    m.load_combinations = []
    m.nodal_masses = []


def make_variant_model(base, inherit: bool = True,
                       active_elements=None, support_set=None):
    """Build a variant's authoritative model — a Structure2D that already
    reflects the variant's **geometry** (only ``active_elements``) and
    **supports** (``support_set``), plus its loads/cases/combinations.

    inherit=True  → keep the base's loads/cases/combinations;
    inherit=False → clean (geometry & supports kept, no actions).

    The returned model is the single source of truth for the variant: what you
    see when it is active, what is solved, and what is saved.
    """
    from .models import Variant
    ssid = None
    ssets = {}
    if support_set is not None:
        ssid = support_set.id
        ssets[ssid] = support_set
    seed = Variant(id="__seed__",
                   active_elements=(set(active_elements) if active_elements else None),
                   support_set_id=ssid)
    # derive_variant (seed has no model) filters geometry + applies the supports.
    m = base.derive_variant(seed, ssets)
    _strip_overlays(m)
    if not inherit:
        _clear_actions(m)
    return m


def _copy_section_into(base, model, section_name):
    """Copy (replace) *section_name* and its material from base into model.

    Sections/materials are base-owned properties: a deepcopy replace keeps the
    variant in sync regardless of which fields changed."""
    import copy
    sec = base.sections.get(section_name)
    if sec is None:
        return
    mat = base.materials.get(sec.material_name)
    if mat is not None:
        model.materials[mat.name] = copy.deepcopy(mat)
    model.sections[sec.name] = copy.deepcopy(sec)


def _copy_tri_section_into(base, model, tri_section_name):
    """Copy (replace) *tri_section_name* and its material from base into
    model — the geometry-object twin of :func:`_copy_section_into`."""
    import copy
    ts = base.tri_sections.get(tri_section_name)
    if ts is None:
        return
    mat = base.materials.get(ts.material_name)
    if mat is not None:
        model.materials[mat.name] = copy.deepcopy(mat)
    model.tri_sections[ts.name] = copy.deepcopy(ts)


def apply_variant_config(base, model, active_elements=None, support_set=None,
                         restore_loads=True, prune=True,
                         restore_base_supports=True):
    """Reconcile an existing variant *model* with the base and a config,
    **preserving the variant's own loads/cases/combinations** on surviving
    elements.

    Two reconciliation regimes share this function:

    - **Explicit / destructive** (defaults; used by the workspace's "Save
      variant" and "Sync with base"): the declared config wins — bars outside
      ``active_elements`` (or gone from the base) are removed with their
      loads, orphan nodes are dropped, and when no ``support_set`` is given
      the supports/springs are restored from the base.
    - **Automatic / non-destructive** (``prune=False,
      restore_base_supports=False``; used by the guarded auto-sync on
      selection / Run / combinations): only ADDITIVE and property changes are
      applied — new base bars are copied in, shared entities re-synced
      (coordinates, sections/materials, hinges, connectivity) — but nothing
      the user may have edited directly on the variant's model is deleted:
      bars are never removed, and supports/springs are left untouched unless
      the variant references a support set (set-owned). This is what keeps a
      variant's directly-edited supports/geometry safe from silent wipes.

    In both regimes bars copied from the base get their base element loads
    restored (``restore_loads``) for the load cases the variant model has,
    and a referenced ``support_set`` always rebuilds the supports (that is
    the propagation path when the set itself is edited).
    """
    import copy
    from .models import SupportAssignment
    base_ids = set(base.bar_elements_by_id)
    active = (set(active_elements) & base_ids if active_elements
              else set(base_ids))

    # 1) Remove inactive bars (and bars that no longer exist in the base),
    #    together with every load bound to them. Destructive regime only —
    #    the auto-sync must never delete what the user may have added.
    if prune:
        for e in list(model.bar_elements):
            if e.id not in active:
                eid = e.id
                model.remove_element(eid)
                model.element_point_loads = [
                    l for l in model.element_point_loads if l.element_id != eid]
                model.temperature_loads = [
                    l for l in model.temperature_loads if l.element_id != eid]

    # 2) Add missing active bars from the base.
    added = []
    for eid in sorted(active):
        if eid in model.bar_elements_by_id:
            continue
        be = base.bar_elements_by_id[eid]
        for nid in (be.node_i, be.node_j):
            if nid not in model.nodes and nid in base.nodes:
                n = base.nodes[nid]
                model.add_node(nid, n.x, n.y)
        _copy_section_into(base, model, be.section_name)
        model.add_bar_element(eid, be.node_i, be.node_j, be.section_name,
                              hinge_i=be.hinge_i, hinge_j=be.hinge_j)
        added.append(eid)

    # 2b) Restore the base's element loads for the bars just (re)added, for the
    #     load cases the variant model actually has.
    if restore_loads and added:
        added_set = set(added)
        cases = set(getattr(model, 'load_cases_by_id', {}))
        for attr in ('distributed_loads', 'element_point_loads',
                     'temperature_loads'):
            for l in getattr(base, attr, []):
                if l.element_id in added_set and l.load_case_id in cases:
                    getattr(model, attr).append(copy.deepcopy(l))

    # 2c) Sync geometry objects (rectangles/polygons meshed into tri_elements
    #     only at solve time) with the base. These have no per-variant
    #     active-set selection yet (unlike bars), so every variant always
    #     carries the base's full set: added when new to the base, updated
    #     when the base edits one (corners, section, target size, edge
    #     modes...), removed when deleted from the base (destructive regime
    #     only — mirrors the bar-removal step above). Without this, a variant
    #     model's own copy of a plate/wall region could silently drift from
    #     the base (or, if ever pruned as an "orphan", lose corners and fail
    #     to mesh) with nothing to repair it short of rebuilding the variant.
    if prune:
        for oid in list(model.geometry_objects):
            if oid not in base.geometry_objects:
                model.remove_geo_object(oid)
    for oid, obj in base.geometry_objects.items():
        for nid in getattr(obj, 'node_ids', []) or []:
            if nid not in model.nodes and nid in base.nodes:
                bn = base.nodes[nid]
                model.add_node(nid, bn.x, bn.y)
        tri_section_name = getattr(obj, 'tri_section_name', '')
        if tri_section_name:
            _copy_tri_section_into(base, model, tri_section_name)
        model.geometry_objects[oid] = copy.deepcopy(obj)

    # 3) Re-sync surviving entities with the base (geometry & properties are
    #    base-owned; only actions belong to the variant).
    for nid, n in model.nodes.items():
        bn = base.nodes.get(nid)
        if bn is not None:
            n.x, n.y = bn.x, bn.y
    for e in model.bar_elements:
        be = base.bar_elements_by_id.get(e.id)
        if be is None:
            continue
        for nid in (be.node_i, be.node_j):     # base may have rewired the bar
            if nid not in model.nodes and nid in base.nodes:
                bn = base.nodes[nid]
                model.add_node(nid, bn.x, bn.y)
        e.node_i, e.node_j = be.node_i, be.node_j
        e.hinge_i, e.hinge_j = be.hinge_i, be.hinge_j
        e.section_name = be.section_name
        _copy_section_into(base, model, be.section_name)

    # 4) Supports & springs: replaced by the support set when given (the set
    #    is their owner — this is how editing a set propagates); restored from
    #    the base only in the destructive regime; otherwise LEFT ALONE — a
    #    variant's directly-edited supports belong to the variant.
    if support_set is not None and getattr(support_set, 'restraints', None):
        # Canonical, shared definitions (PIN/FIXED/ROLLER-X/... or, in the
        # plate domain, SIMPLE/CLAMPED/...) via support_for()/assign_support()
        # — the same mechanism the canvas "Add support" flow uses — instead
        # of one bespoke Support named after the node itself. A support set
        # used to give every restrained node its own private definition, so a
        # 40-node "No prop" set defined 40 different names for what was really
        # 2 or 3 distinct restraint patterns; now it reuses one definition per
        # pattern, same as everywhere else a support is drawn.
        from .models import CANONICAL_SUPPORTS, CANONICAL_SUPPORTS_PLATE
        model.support_assignments = []
        for nid, r in support_set.restraints.items():
            ux, uy, tz = bool(r[0]), bool(r[1]), bool(r[2])
            name = model.support_for(ux, uy, tz)
            if name is not None:
                model.assign_support(nid, name)
        # Drop canonical definitions nothing uses any more (e.g. the previous
        # restraint pattern at a node no longer applies) — mirrors
        # MainWindow._prune_supports_we_made, without the Qt dependency;
        # anything under a name we didn't generate is left alone either way.
        canonical = set(CANONICAL_SUPPORTS.values()) | \
            set(CANONICAL_SUPPORTS_PLATE.values())
        ours = {n for n in model.supports
                if n in canonical
                or (n.rsplit('-', 1)[0] in canonical
                    and n.rsplit('-', 1)[-1].isdigit())}
        model.prune_unused_supports(only=ours)
        model.node_springs = {k: copy.deepcopy(v)
                              for k, v in support_set.node_springs.items()}
        model.element_springs = {k: copy.deepcopy(v)
                                 for k, v in support_set.element_springs.items()}
    elif restore_base_supports:
        # Restore the base supports (adding any supported node the model lacks,
        # e.g. a deliberately free supported node).
        model.support_assignments = []
        for a in base.support_assignments:
            if a.node_id not in model.nodes and a.node_id in base.nodes:
                bn = base.nodes[a.node_id]
                model.add_node(a.node_id, bn.x, bn.y)
            model.support_assignments.append(
                SupportAssignment(node_id=a.node_id,
                                  support_name=a.support_name))
            if (a.support_name in base.supports
                    and a.support_name not in model.supports):
                sp = base.supports[a.support_name]
                model.add_support(sp.name, ux=sp.ux, uy=sp.uy, tz=sp.tz)
        model.node_springs = {k: copy.deepcopy(v)
                              for k, v in base.node_springs.items()
                              if k in model.nodes}
        model.element_springs = {k: copy.deepcopy(v)
                                 for k, v in base.element_springs.items()
                                 if k in model.bar_elements_by_id}

    # 5) Drop orphan nodes (no active bar and not supported) — destructive
    #    regime only.
    if prune:
        supported = {a.node_id for a in model.support_assignments}
        used = set()
        for e in model.bar_elements:
            used.add(e.node_i); used.add(e.node_j)
        for e in model.tri_elements:
            used.add(e.node_i); used.add(e.node_j); used.add(e.node_k)
        # Quad nodes must survive the prune too (dev/refactor_area_path.md
        # Phase 2) — a node used only by a hand-added quad would otherwise be
        # removed as an orphan.
        for e in getattr(model, 'quad_elements', []):
            used.add(e.node_i); used.add(e.node_j)
            used.add(e.node_k); used.add(e.node_l)
        # A plate/wall region is often still a GeoRectangle/GeoPolygon (meshed
        # into tri_elements only at solve time) rather than actual elements
        # yet — its corner/vertex nodes must survive the prune too, or the
        # object silently loses corners and fails to mesh later.
        for obj in model.geometry_objects.values():
            used.update(getattr(obj, 'node_ids', []) or [])
        for nid in list(model.nodes):
            if nid not in used and nid not in supported:
                model.remove_node(nid)
    model._node_dof_index = None
    return model


# ---------------------------------------------------------------------------
# Base ↔ variant synchronisation (Fase 2)
# ---------------------------------------------------------------------------

def _variant_sync_sig(base, variant, support_set) -> str:
    """Signature of everything a variant model derives from: the base's
    stiffness-relevant state, the declared active set and the support set's
    content. Equal signatures ⇒ a reconcile would be a no-op."""
    import hashlib
    parts = [base.stiffness_hash()]
    ae = getattr(variant, 'active_elements', None)
    parts.append('*' if ae is None else '|'.join(sorted(ae)))
    if support_set is not None:
        from .structure_io import _support_set_to_dict
        parts.append(repr(_support_set_to_dict(support_set)))
    return hashlib.sha256('\n'.join(parts).encode('utf-8')).hexdigest()


def sync_variant_with_base(base, variant, support_sets=None,
                           force: bool = False,
                           destructive: bool = False) -> bool:
    """Reconcile *variant.model* with the current base (see
    :func:`apply_variant_config`). Guarded by a signature of the base state +
    variant config, so calling it repeatedly is O(hash) when nothing changed.

    By default the reconcile is NON-DESTRUCTIVE (additive + property sync
    only): it never deletes bars from the variant's model nor touches its
    supports/springs unless the variant references a support set. Pass
    ``destructive=True`` only from explicit user actions ("Sync with base",
    "Save variant") where the declared config is meant to win.

    Returns True when a reconcile actually ran; False for a guarded no-op or a
    variant without its own model (nothing to sync — it derives on demand).
    """
    model = getattr(variant, 'model', None)
    if model is None:
        return False
    ssets = (support_sets if support_sets is not None
             else getattr(base, 'support_sets', {}) or {})
    sid = getattr(variant, 'support_set_id', None)
    sset = ssets.get(sid) if sid else None
    sig = _variant_sync_sig(base, variant, sset)
    # A model missing bars, or missing a geometry object's corner/vertex nodes,
    # while the base has them, is damage (older builds could save either kind),
    # not a state the guard may preserve: the additive sync below repopulates
    # it from the base. Checking bar_elements alone missed a plate/wall model
    # (geometry objects only, no bars at all) whose object had silently lost
    # nodes — the guard's signature match then hid the damage forever.
    missing_obj_nodes = any(
        nid not in model.nodes
        for obj in base.geometry_objects.values()
        for nid in getattr(obj, 'node_ids', []) or [])
    damaged = ((not model.bar_elements and base.bar_elements)
              or missing_obj_nodes)
    if (not force and not damaged
            and getattr(variant, '_synced_base_sig', None) == sig):
        return False
    apply_variant_config(base, model,
                         active_elements=variant.active_elements,
                         support_set=sset,
                         prune=destructive,
                         restore_base_supports=destructive)
    variant._synced_base_sig = sig
    return True


def variant_base_diff(base, variant, support_sets=None) -> dict:
    """What a sync would change on *variant.model* — pure, no mutation.

    Returns ``{'in_sync': bool, 'add': [elem ids], 'remove': [elem ids],
    'supports_changed': bool}``. ``in_sync`` is the signature guard's verdict
    (False also when only properties — sections, coordinates, hinges, springs —
    changed, even if add/remove/supports report nothing).
    """
    model = getattr(variant, 'model', None)
    if model is None:
        return {'in_sync': True, 'add': [], 'remove': [],
                'supports_changed': False}
    ssets = (support_sets if support_sets is not None
             else getattr(base, 'support_sets', {}) or {})
    sid = getattr(variant, 'support_set_id', None)
    sset = ssets.get(sid) if sid else None
    sig = _variant_sync_sig(base, variant, sset)
    in_sync = (getattr(variant, '_synced_base_sig', None) == sig)

    base_ids = set(base.bar_elements_by_id)
    ae = getattr(variant, 'active_elements', None)
    active = (set(ae) & base_ids) if ae else base_ids
    have = set(model.bar_elements_by_id)

    # Expected supports: the set's restraints, or the base's.
    if sset is not None and getattr(sset, 'restraints', None):
        expected = {nid: tuple(bool(x) for x in r)
                    for nid, r in sset.restraints.items()
                    if any(r)}
    else:
        expected = current_restraints(base)
    actual = current_restraints(model)

    return {'in_sync': in_sync,
            'add': sorted(active - have),
            'remove': sorted(have - active),
            'supports_changed': expected != actual}


def sync_all_variants(base) -> dict:
    """Sync every registered variant. Returns ``{variant_id: True | False |
    'error: …'}`` — never raises, so a broken variant cannot block the rest."""
    out = {}
    for vid, v in getattr(base, 'variants', {}).items():
        try:
            out[vid] = sync_variant_with_base(base, v)
        except Exception as e:
            out[vid] = f"error: {e}"
    return out


def current_restraints(struc) -> dict:
    """Return the structure's current supports as ``{node_id: (ux, uy, tz)}``.

    Handy for seeding a SupportSet editor from the model's actual supports.
    """
    out = {}
    for a in struc.support_assignments:
        sp = struc.supports.get(a.support_name)
        if sp is not None:
            out[a.node_id] = (bool(sp.ux), bool(sp.uy), bool(sp.tz))
    return out


def phases_from_scenes(scenes, valid_elem_ids, cumulative: bool = True) -> list:
    """Generate ConstructionPhases from an ordered list of scenes.

    scenes : iterable of ``{'name': str, 'elements': set}`` (the GUI's named
        working subsets), in construction order.
    valid_elem_ids : element ids that exist in the base (others are ignored).
    cumulative : phase k activates the union of scenes 1..k (monotone growth —
        the regime the incremental engine handles natively). When False each
        phase activates exactly its scene's elements (removal is resolved by
        the engine's release forces, but review the result).

    No load cases are assigned — the caller/user picks each phase's increment.
    """
    from .models import ConstructionPhase
    valid = set(valid_elem_ids)
    phases, cum = [], set()
    for sc in scenes:
        elems = set(sc.get('elements', set())) & valid
        if cumulative:
            cum |= elems
            active = set(cum)
        else:
            active = elems
        phases.append(ConstructionPhase(id=sc['name'], active_elements=active))
    return phases


def _stageable_elements(struc):
    """Every element that can carry a construction stage: bars AND area
    elements (triangles and quads alike — a plate/wall region is staged exactly
    like a bar frame, see TriElement.stage / QuadElement.stage). Geometry
    objects (unmeshed rectangles/polygons) have no stage of their own; they are
    staged via the area elements they expand into.

    dev/refactor_area_path.md Phase 1: quads were previously omitted here, so a
    quad-meshed region could not be staged at all — now included via
    ``area_elements()`` when the model exposes it (older/duck-typed models
    without it fall back to bars + triangles)."""
    if hasattr(struc, 'area_elements'):
        return list(struc.bar_elements) + list(struc.area_elements())
    return list(struc.bar_elements) + list(getattr(struc, 'tri_elements', []))


def assign_stage(struc, element_ids, stage: int) -> int:
    """Set the construction stage of the given elements (bars, triangles or
    quads). Returns how many elements were actually updated (unknown ids are
    ignored). dev/refactor_area_path.md Phase 1: quads are now resolvable too,
    via ``area_element_by_id`` — before, a quad id was silently skipped."""
    stage = int(stage)
    if stage < 1:
        raise ValueError("stage must be >= 1")
    n = 0
    area_by_id = getattr(struc, 'area_element_by_id', None)
    for eid in element_ids:
        e = struc.bar_elements_by_id.get(eid)
        if e is None and area_by_id is not None:
            e = area_by_id(eid)
        elif e is None:
            e = getattr(struc, 'tri_elements_by_id', {}).get(eid)
        if e is not None:
            e.stage = stage
            n += 1
    return n


def stage_numbers(struc) -> list:
    """The sorted list of stage numbers present in the model (bars + area
    elements: triangles and quads)."""
    return sorted({int(getattr(e, 'stage', 1) or 1)
                   for e in _stageable_elements(struc)})


def phases_from_stages(struc, cases_by_stage: dict = None) -> list:
    """Generate ConstructionPhases from the elements' ``stage`` attribute
    (bars and area elements — triangles and quads — alike).

    Phase k activates every element with ``stage <= k`` — monotone growth by
    construction, the regime the incremental engine handles natively. One
    phase is generated per stage number present in the model, in order, with
    id ``"Stage k"``.

    cases_by_stage : optional ``{stage_number: [case ids]}`` — each phase's
        applied increment; phases without an entry get no cases (the user
        assigns them afterwards).
    """
    from .models import ConstructionPhase
    cases_by_stage = cases_by_stage or {}
    phases = []
    for k in stage_numbers(struc):
        active = {e.id for e in _stageable_elements(struc)
                  if int(getattr(e, 'stage', 1) or 1) <= k}
        phases.append(ConstructionPhase(
            id=f"Stage {k}",
            active_elements=active,
            applied_cases=list(cases_by_stage.get(k, []))))
    return phases


def list_overlays(struc) -> dict:
    """Summarise the registered overlays for a UI list."""
    return {
        'support_sets': sorted(struc.support_sets),
        'variants': [{'id': v.id, 'support_set_id': v.support_set_id,
                      'n_active': (None if v.active_elements is None
                                   else len(v.active_elements))}
                     for v in struc.variants.values()],
        'sequences': [{'id': s.id, 'n_phases': len(s.phases),
                       'method': s.displacement_method}
                      for s in struc.construction_sequences.values()],
    }
