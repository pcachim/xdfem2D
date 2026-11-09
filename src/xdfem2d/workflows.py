"""
Headless workflows for variants & phasing (Phase 5 — controller layer).

The GUI (PySide6) binds to these functions; they contain no Qt and are fully
testable. They wrap the Phase 1–4 engines (variants, import_io, phasing) and the
structure's registered overlays so the UI only has to gather inputs and render
results.
"""
from __future__ import annotations

import traceback
from dataclasses import dataclass, field

from .models import Variant
from .variants import (solve_variants, combine_across_variants, CombTerm,
                       VariantResult, _geometry_sig)
from .phasing import solve_sequence
from .import_io import import_model


# ---------------------------------------------------------------------------
# Run preconditions
# ---------------------------------------------------------------------------

def validate_for_analysis(struc) -> list:
    """What stops *struc* from being analysed, as ``[{'code', 'msg'}, ...]``
    (empty when it can run), in the order the GUI reports them. The GUI shows
    the first one; a headless caller gets them all.

    Geometry-object models have their nodes/bars/supports generated at solve
    time, so the editable-model geometry checks are skipped for them.
    """
    problems = []
    has_objects = bool(getattr(struc, "geometry_objects", None))
    if not has_objects:
        if not struc.nodes:
            problems.append({'code': 'no_nodes', 'msg': "No nodes defined."})
        if not struc.has_elements():
            problems.append({'code': 'no_elements',
                             'msg': "No elements defined."})
        if not (struc.support_assignments or struc.node_springs
                or struc.element_springs):
            problems.append({
                'code': 'no_supports',
                'msg': "No supports assigned. Assign a support or a spring "
                       "(node or element) to restrain the structure."})
    if not struc.load_cases:
        problems.append({'code': 'no_load_cases',
                         'msg': "No load cases defined."})
    if not struc.analysis_cases:
        problems.append({'code': 'no_analysis_cases',
                         'msg': "No analysis cases are defined."})
    return problems


# ---------------------------------------------------------------------------
# Running an analysis (no Qt: the GUI runs these on a worker thread)
# ---------------------------------------------------------------------------

@dataclass
class VariantRun:
    """Outcome of :func:`solve_all_variants`.

    rcache : ``{"base": res, ("var", id): res, ("combo", id): res}``.
    variant_errors : ``{variant_id: message}`` for variants that failed.
    force_only : ids of cross-variant combinations valid for forces only.
    combo_errors : ``{combination_id: message}`` for combinations that failed.
    """
    rcache: dict = field(default_factory=dict)
    variant_errors: dict = field(default_factory=dict)
    force_only: set = field(default_factory=set)
    combo_errors: dict = field(default_factory=dict)


def solve_all_variants(base) -> VariantRun:
    """Solve the base, every variant and every cross-variant combination.

    The cross-variant combinations are **injected into the base results** as
    ordinary combinations, so they appear in the normal Case selector when
    viewing the base and render on the base geometry. A variant or combination
    that fails to solve is recorded, never raised: the base and the remaining
    variants are still computed. Only a failure of the base itself raises.
    """
    base_res = base.calculate()
    base_res.setdefault("combinations", {})
    base_res.setdefault("combo_distribution", {})
    run = VariantRun(rcache={"base": base_res})
    empty = {"displacements": {}, "reactions": {},
             "element_forces": {}, "spring_forces": {}}
    # Everything solved here is collected as VariantResult so the
    # cross-variant combinations below reuse it instead of solving the same
    # variants a second time.
    precomputed = {"BASE": VariantResult(
        variant_id="BASE", stiffness_hash=base.stiffness_hash(),
        results=base_res, geometry_sig=_geometry_sig(base))}
    for vid, v in getattr(base, "variants", {}).items():
        try:
            # Reconcile with the base before solving (guarded no-op).
            sync_variant_with_base(base, v)
            sub = base.derive_variant(v)
            res = sub.calculate()
            # The variant is solved on a copy, so the cases of its own model
            # are marked here: left pending, the Case selector hid them while
            # their results were there (only the combinations showed).
            for ac in getattr(getattr(v, 'model', None), 'analysis_cases', []) or []:
                ac.solved = True
            run.rcache[("var", vid)] = res
            precomputed[vid] = VariantResult(
                variant_id=vid, stiffness_hash=sub.stiffness_hash(),
                results=res, geometry_sig=_geometry_sig(sub))
        except Exception as e:
            run.variant_errors[vid] = str(e)
    for cid, d in getattr(base, "variant_combinations", {}).items():
        try:
            cr = run_variant_combination(
                base, d.get("terms", []), d.get("op", "Envelope"),
                allow_geometry_mismatch=True, precomputed=precomputed)
            run.rcache[("combo", cid)] = cr        # for the Combine tab table
            base_res["combinations"][cid] = {
                k: v for k, v in cr.items()
                if k not in ("combo_distribution", "_same_stiffness")}
            base_res["combo_distribution"][cid] = cr.get("combo_distribution", {})
            if not cr.get("_same_stiffness", True):
                run.force_only.add(cid)
        except Exception as e:
            run.rcache[("combo", cid)] = {"error": str(e)}
            run.combo_errors[cid] = str(e)
            base_res["combinations"][cid] = dict(empty)
            base_res["combo_distribution"][cid] = {}
    return run


def run_sequences(struc, results, cache: dict = None, on_start=None) -> dict:
    """Run every construction sequence and inject the phase results into
    *results*, under ``results['analysis_cases']['seq__<seq>__<phase>']``.

    cache : optional dict; a sequence already in it (key ``('seq', id)``) is not
        solved again, and a newly solved one is stored there.
    on_start : optional ``callable(seq_id)`` called before each sequence that
        is actually solved (a progress hook for the GUI).

    Returns ``{seq_id: message}`` for sequences that failed (the others are
    still injected).
    """
    errors = {}
    seqs = getattr(struc, 'construction_sequences', {})
    if not seqs or results is None:
        return errors
    if cache is None:
        cache = {}
    ac_dict = results.setdefault('analysis_cases', {})
    for seq_id in seqs:
        out = cache.get(('seq', seq_id))
        if out is None:
            try:
                if on_start is not None:
                    on_start(seq_id)
                out = run_sequence(struc, seq_id)
            except Exception as e:
                errors[seq_id] = f"{e}\n\n{traceback.format_exc()}"
                continue
            cache[('seq', seq_id)] = out
        # Strip 'initial_state' (ElementInitialState objects: not
        # JSON-serializable, only used internally for phase chaining).
        for phase_id, phase_res in out.get('phases', {}).items():
            ac_dict[f"seq__{seq_id}__{phase_id}"] = {
                k: v for k, v in phase_res.items() if k != 'initial_state'}
        seq_ac = struc.analysis_cases_by_id.get(seq_id)
        if seq_ac is not None:
            seq_ac.solved = True
    return errors


@dataclass
class AnalysisOutcome:
    """Result of :func:`run_analysis`. ``results`` is the base model's results
    (with sequence phases and cross-variant combinations injected); the other
    fields are what the GUI keeps next to it."""
    results: dict
    rcache: dict = field(default_factory=dict)
    variant_errors: dict = field(default_factory=dict)
    combo_errors: dict = field(default_factory=dict)
    sequence_errors: dict = field(default_factory=dict)
    force_only: set = field(default_factory=set)
    incremental: bool = False


def run_analysis(struc, scope: str = "active", results: dict = None) -> AnalysisOutcome:
    """Run *struc* to completion, with no dialogs: the headless counterpart of
    the GUI's Run.

    scope : ``"active"`` solves the model as it is; ``"all"`` also solves every
        variant and cross-variant combination (:func:`solve_all_variants`).
    results : the previous results of a solved model. When some analysis case
        is still pending (``solved`` False) only those are computed
        (``solve_pending_cases``); otherwise, or if that cannot be done, a full
        solve runs.

    Check :func:`validate_for_analysis` first. A failure to solve raises.
    """
    from .solver import solve_pending_cases
    out = AnalysisOutcome(results=None)
    if scope == "all":
        run = solve_all_variants(struc)
        out.rcache, out.variant_errors = run.rcache, run.variant_errors
        out.combo_errors, out.force_only = run.combo_errors, run.force_only
        out.results = run.rcache["base"]
    else:
        res = None
        if results is not None and any(
                not getattr(ac, 'solved', True) for ac in struc.analysis_cases):
            res = solve_pending_cases(struc, results)
            out.incremental = res is not None
        if res is None:
            res = struc.calculate()
        out.results = res
        out.rcache = {"base": res}
    out.sequence_errors = run_sequences(struc, out.results, out.rcache)
    return out


DESIGN_KINDS = ("concrete", "bars", "areas", "crack", "punching", "steel", "timber")


def vertical_support_nodes(struc) -> list:
    """Node ids whose transverse deflection is restrained: the nodes a punching column can sit
    on. In the plate domain w is stored in the ux slot, so a support with ``ux`` fixed restrains
    it. Order kept, no repeats."""
    out, seen = [], set()
    for a in getattr(struc, 'support_assignments', []):
        sp = struc.supports.get(a.support_name)
        if sp is not None and getattr(sp, 'ux', False) and a.node_id not in seen:
            seen.add(a.node_id)
            out.append(a.node_id)
    return out


def available_combinations(struc, results) -> list:
    """The combination ids a design can use: the model's load combinations, then
    any that only the results carry (envelopes), in that order."""
    out = []
    for lc in getattr(struc, "load_combinations", []):
        if lc.id not in out:
            out.append(lc.id)
    for cid in (results or {}).get("combinations", {}):
        if cid not in out:
            out.append(cid)
    return out


def run_design(struc, results: dict, kind: str, prefs: dict = None,
               combinations=None, sls_combinations=None) -> dict:
    """Design the members of *kind* from the analysis *results* and store the
    outcome in them, where ``get design`` reads it: the headless counterpart of
    the GUI's Design menu.

    kind : ``"concrete"`` (the bars: EC2 reinforcement of the beams, into
        ``results['reinforcement']``, and the checks of the columns, into
        ``results['column_design']``), ``"areas"`` (slabs of a plate model and
        walls of a plane model, meshed in triangles: ``results['tri_reinforcement']``),
        ``"steel"`` (EC3, ``results['steel_design']``) or ``"timber"`` (EC5,
        ``results['timber_design']``) or ``"crack"`` (EC2 7.3.4 crack widths of the
        beams, into ``results['crack']``; needs the ``"concrete"`` design first, whose
        reinforcement it uses), ``"bars"`` (the bars of each beam by zone, from the concrete
        design's reinforcement, into ``results['beam_bars']``: a proposal, the detail stored in
        the model is not touched) or ``"punching"`` (EC2 6.4 of the slab at the punching columns
        the model defines, into ``results['punching']``; plate domain). Without ``sls_combinations`` the crack width
        takes the combinations whose id starts with ``SLS`` (the auto-generated ones), or all.
    prefs : the design preferences (partial safety factors, code editions);
        what is missing takes the engine's defaults. The ones saved in a model's
        view (``view['design']['prefs']``) are what the GUI used.
    combinations : ids to design for; by default all that are available.
    sls_combinations : the same for ``"crack"``, which looks at serviceability ones.

    Returns ``{'kind', 'designed', 'failing', 'combinations'}`` (plus ``columns``
    for ``"concrete"``); ``designed`` is 0 when the model has no member of that
    material, or none of the combinations gave it forces. Raises ``ValueError`` for an
    unknown *kind*, results missing, or no combination to design for.
    """
    if kind not in DESIGN_KINDS:
        raise ValueError(f"kind is one of {', '.join(DESIGN_KINDS)}")
    if not results:
        raise ValueError("run the analysis first")
    combos = available_combinations(struc, results)
    wanted = sls_combinations if kind == "crack" else combinations
    if kind == "crack" and not wanted:
        # a crack width belongs to a serviceability combination: the ones the auto
        # generator names SLS-... when there are some (the menu takes all of them)
        wanted = [c for c in combos if c.startswith("SLS")] or None
    if wanted:
        combos = [c for c in combos if c in wanted]
    if not combos:
        raise ValueError("no combinations available for design: add some "
                         "(e.g. add combination ULS = 1.35 G + 1.5 Q)")
    prefs = dict(prefs or {})
    # The mesh the results are keyed by: the compiled one for a model built from
    # geometry objects, the model itself otherwise.
    if getattr(struc, "geometry_objects", None):
        from .geo_expand import expand_geometry
        mesh, _trace = expand_geometry(struc)
    else:
        mesh = struc
    out = {'kind': kind, 'designed': 0, 'failing': 0, 'combinations': combos}
    gc, gs, acc = (prefs.get("gamma_c", 1.5), prefs.get("gamma_s", 1.15),
                   prefs.get("alpha_cc", 1.0))
    if kind == "bars":
        from .rc_design import design_beam_bars
        rc = results.get("reinforcement") or {}
        if not rc.get("rows"):
            raise ValueError("run the concrete design first (run_design concrete): the bars are "
                             "chosen from the reinforcement it finds")
        inputs: list = []
        bars = design_beam_bars(mesh, rc["rows"], results, gamma_c=gc, gamma_s=gs, alpha_cc=acc,
                                prefs=prefs, inputs_out=inputs)
        out['designed'] = len({b["beam"] for b in bars})
        out['failing'] = sum(1 for b in bars if b.get("As_bot_prov") is None)
        if bars:
            results["beam_bars"] = {
                "rows": bars, "inputs": [i.to_dict() for i in inputs],
                "header": f"{len(bars)} zone(s) in {out['designed']} beam(s)"}
        else:
            results.pop("beam_bars", None)
        return out
    if kind == "punching":
        from .punching import design_punching_and_store
        if getattr(struc, "domain", "plane") != "plate":
            raise ValueError("punching applies to a plate model (slabs): this one is plane")
        if not getattr(struc, "punch_columns", None):
            raise ValueError("no punching columns: add_punch_column first (a column sits on a "
                             "node that has a support)")
        stored = design_punching_and_store(
            mesh, results, combinations=combos, gamma_c=gc, gamma_s=gs, alpha_cc=acc,
            gamma_v=prefs.get("gamma_v", 1.4), eta_sys=prefs.get("eta_sys", 1.5),
            dmax=prefs.get("dmax", 20.0), edition=prefs.get("ec2_edition", "2004"),
            annex=prefs.get("ec2_annex") or None, prefs=prefs)
        rows = (stored or {}).get("rows") or []
        out['designed'] = len({r.get("column") for r in rows})
        out['failing'] = sum(1 for r in rows if r.get("governing") and r.get("needs_reinf"))
        if rows:
            out['worst'] = max(rows, key=lambda r: r.get("utilization", 0.0)).get("column")
        return out
    if kind == "crack":
        from .rc_design import crack_width_sections
        rc = results.get("reinforcement") or {}
        if not rc.get("rows"):
            raise ValueError("run the concrete design first (run_design concrete): the crack "
                             "width is computed with the reinforcement it finds")
        sls_combo = prefs.get("sls_combo", "Quasi-permanent")
        k_t = 0.4 if sls_combo == "Quasi-permanent" else 0.6
        wk_limit = prefs.get("wk_max", 0.3)
        rows = crack_width_sections(mesh, results, rc["rows"], combinations=combos, k_t=k_t,
                                    wk_limit=wk_limit, rc_envelope=rc.get("envelope"))
        out['designed'] = len(rows)
        if rows:
            results["crack"] = {"rows": rows, "prefs": {"sls_combo": sls_combo, "k_t": k_t,
                                                         "wk_max": wk_limit}}
            out['failing'] = sum(1 for r in rows if r.get("governing") and r.get("ok") is False)
        else:
            results.pop("crack", None)
        return out
    if kind == "concrete":
        from .rc_design import design_and_store, design_concrete_columns
        stored = design_and_store(mesh, results, gamma_c=gc, gamma_s=gs,
                                  alpha_cc=acc, combinations=combos, prefs=prefs)
        out['designed'] = len((stored or {}).get("rows") or [])
        # Members flagged is_column get the EC2 5.8 / 6.1 check instead, in the
        # same run (as the menu does).
        cols = design_concrete_columns(mesh, results, prefs=prefs,
                                       combinations=combos)
        members, ratios = cols.get("members", []), cols.get("elements", {})
        out['columns'] = len(members)
        out['failing'] = sum(1 for r in members if not r.get("passed", True))
        if members or ratios:
            results["column_design"] = {
                "members": members, "elements": ratios,
                "code": str(prefs.get("ec2_edition", "2004")),
                "header": (f"{len(members)} column(s), {len(combos)} combination(s), "
                           + (f"{out['failing']} not verified." if out['failing']
                              else "all verified."))}
        else:
            results.pop("column_design", None)
        return out
    if kind == "areas":
        from .rc_design import design_tri_and_store
        stored = design_tri_and_store(mesh, results, gamma_c=gc, gamma_s=gs,
                                      alpha_cc=acc, combinations=combos, prefs=prefs)
        out['designed'] = len((stored or {}).get("rows") or [])
        return out
    if kind == "steel":
        from .steel_design import design_steel_members as design
        key, edition = "steel_design", prefs.get("ec3_edition", "2005")
    else:
        from .timber_design import design_timber_members as design
        key, edition = "timber_design", prefs.get("ec5_edition", "2004")
    res = design(mesh, results, prefs=prefs, combinations=combos)
    rows, ratios = res.get("members", []), res.get("elements", {})
    out['designed'] = len(rows)
    out['failing'] = sum(1 for r in rows if not r.get("passed", True))
    if rows or ratios:
        results[key] = {
            "members": rows, "elements": ratios, "code": str(edition),
            "header": (f"{len(rows)} member(s), {len(combos)} combination(s), "
                       + (f"{out['failing']} not verified." if out['failing']
                          else "all verified."))}
    else:
        results.pop(key, None)
    return out


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
        vres.update(solve_variants(struc, to_solve))
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
        elif v is None or getattr(v, 'active_elements', None) is None:
            # 'BASE' or a config-less variant: the unmodified structure.
            sub = struc
        else:
            # Declared config without a model: derive (copies, but rare).
            sub = struc.derive_variant(v)
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
    resulting variant (its supports are the imported model's). Returns
    ``(unified, variant, report)``.

    The unified structure is returned (caller decides whether to adopt it as the
    working model); the variant is registered on it.
    """
    unified, variant, report = import_model(struc, other, overlay_id, **kw)
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
    reflects the variant's **geometry** (only ``active_elements``) and its
    supports, plus its loads/cases/combinations.

    The supports are the base's, changed by ``support_set`` (a SupportSet) when
    one is given to seed them from: it is used once, here, and not remembered —
    from then on the supports are the model's own.

    inherit=True  → keep the base's loads/cases/combinations;
    inherit=False → clean (geometry & supports kept, no actions).

    The returned model is the single source of truth for the variant: what you
    see when it is active, what is solved, and what is saved.
    """
    seed = Variant(id="__seed__",
                   active_elements=(set(active_elements) if active_elements else None))
    # derive_variant (seed has no model) filters geometry + applies the supports.
    m = base.derive_variant(seed, support_set)
    _strip_overlays(m)
    if not inherit:
        _clear_actions(m)
    from .structure import attach_family
    attach_family(base, m)
    return m


def new_variant(base, id: str, description: str = "", active_elements=None,
                inherit: bool = True, support_set=None) -> Variant:
    """A ready variant: its model, and the record of the base's supports it was
    made from (so a later sync can tell what the user changed)."""
    v = Variant(id=id, description=description,
                active_elements=(set(active_elements) if active_elements else None))
    v.model = make_variant_model(base, inherit=inherit,
                                 active_elements=active_elements,
                                 support_set=support_set)
    v.base_state = base_state(base)
    v.inherit_actions = bool(inherit)        # a clean variant defines its own
    return v


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


def _copy_area_section_into(base, model, name):
    """Copy the triangle and the quad section called *name* (the panel dialog
    makes them in pairs), with their material, from base into model."""
    import copy
    for coll in ('tri_sections', 'quad_sections'):
        sec = getattr(base, coll, {}).get(name)
        if sec is None:
            continue
        mat = base.materials.get(sec.material_name)
        if mat is not None:
            model.materials[mat.name] = copy.deepcopy(mat)
        getattr(model, coll)[sec.name] = copy.deepcopy(sec)


def _copy_nodes_into(base, model, node_ids):
    for nid in node_ids:
        if nid not in model.nodes and nid in base.nodes:
            n = base.nodes[nid]
            model.add_node(nid, n.x, n.y)


def _area_nodes(ent):
    """The node ids an area element or a geometry object stands on."""
    if hasattr(ent, 'node_ids'):
        return list(ent.node_ids or [])
    return [getattr(ent, f) for f in ('node_i', 'node_j', 'node_k', 'node_l')
            if getattr(ent, f, None)]


def area_selection(base, active_elements):
    """Which areas a variant's ``active_elements`` keep, as
    ``(tris, quads, objects)`` sets of ids, or ``None`` for all of them (no
    set: the variant follows the whole base).

    The set lists bars, triangles, quads and geometry objects in one set, as a
    construction phase does: an area is kept only if its id is in it."""
    if not active_elements:
        return None
    active = set(active_elements)
    return (active & set(base.tri_elements_by_id),
            active & set(base.quad_elements_by_id),
            active & set(base.geometry_objects))


# ---------------------------------------------------------------------------
# A variant's supports and springs
# ---------------------------------------------------------------------------

_SUPPORT_KINDS = ('supports', 'node_springs', 'element_springs')
# What else a variant may change on its own model and keep: the nodes'
# coordinates, the bars' properties (hinges, section, end nodes, ...) and the
# sections and materials. One entry per FIELD, keyed ``"<id>|<field>"``, so
# the user's change to a bar's hinge and the base's change to its section both
# survive a sync.
_ENTITY_KINDS = ('nodes', 'bars', 'sections', 'materials',
                 'tris', 'quads', 'objects')
# The variant's actions: its load cases, analysis cases and combinations (one
# entry per field, as above, and ``"<id>|__exists__"`` for the case itself),
# and its loads, grouped by what they act on and the case they belong to.
_ACTION_ENTITY_KINDS = ('load_cases', 'analysis_cases', 'combinations')
_LOADS_KIND = 'loads'
_ACTION_KINDS = _ACTION_ENTITY_KINDS + (_LOADS_KIND,)
_PLAN_KINDS = _SUPPORT_KINDS + _ENTITY_KINDS + _ACTION_KINDS
# Every list of loads on a structure (each record has a ``load_case_id``,
# except the masses), and the fields that say what it acts on.
_LOAD_LISTS = ('point_loads', 'distributed_loads', 'element_point_loads',
               'temperature_loads', 'support_settlements',
               'area_temperature_loads', 'line_distributed_loads',
               'line_temperature_loads', 'quad_area_loads', 'quad_edge_loads',
               'quad_temperature_loads', 'surface_area_loads',
               'surface_edge_loads', 'tri_area_loads', 'tri_edge_loads',
               'tri_temperature_loads', 'nodal_masses')
_TARGET_FIELDS = ('node_id', 'element_id', 'tri_id', 'quad_id', 'object_id', 'id')


def _entities(struc, kind) -> dict:
    if kind == 'combinations':
        return {c.id: c for c in struc.load_combinations}
    return {'nodes': struc.nodes, 'bars': struc.bar_elements_by_id,
            'sections': struc.sections, 'materials': struc.materials,
            'tris': struc.tri_elements_by_id, 'quads': struc.quad_elements_by_id,
            'objects': struc.geometry_objects,
            'load_cases': struc.load_cases_by_id,
            'analysis_cases': struc.analysis_cases_by_id}[kind]


def _group_key(attr, rec) -> str:
    """The group a load belongs to: its list, what it acts on and its case."""
    target = '|'.join(str(getattr(rec, f)) for f in _TARGET_FIELDS
                      if hasattr(rec, f))
    return f"{attr}|{target}|{getattr(rec, 'load_case_id', '')}"


def _grouped_loads(struc) -> dict:
    """``{group key: [record, ...]}`` of the structure's loads, as objects."""
    out = {}
    for attr in _LOAD_LISTS:
        for rec in getattr(struc, attr, []) or []:
            out.setdefault(_group_key(attr, rec), []).append(rec)
    return out


def _plain(v):
    """A value as plain, comparable data (what is kept in ``base_state``)."""
    import dataclasses
    if dataclasses.is_dataclass(v) and not isinstance(v, type):
        return {f.name: _plain(getattr(v, f.name)) for f in dataclasses.fields(v)}
    if isinstance(v, dict):
        return {str(k): _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if isinstance(v, (set, frozenset)):
        return sorted((_plain(x) for x in v), key=repr)
    if hasattr(v, 'value') and not isinstance(v, (int, float, str, bool)):
        return _plain(v.value)                       # an Enum
    return v


def _flat_entities(struc, kind) -> dict:
    import dataclasses
    out = {}
    for eid, obj in _entities(struc, kind).items():
        for f in dataclasses.fields(obj):
            if f.name in ('id', 'name') or (kind == 'analysis_cases'
                                           and f.name == 'solved'):
                continue          # ``solved`` says what has been run, not what the case is
            out[f"{eid}|{f.name}"] = _plain(getattr(obj, f.name))
        if kind in _ACTION_ENTITY_KINDS:
            out[f"{eid}|__exists__"] = True
    return out


def base_state(struc) -> dict:
    """What a variant has to compare against the base: the supports and
    springs of *struc*, and its nodes, bars, sections and materials, as plain
    comparable data::

        {'supports': {node: [ux, uy, tz]},
         'node_springs': {node: {...}}, 'element_springs': {element: {...}},
         'nodes': {'N1|x': 0.0, ...}, 'bars': {'E1|hinge_j': False, ...},
         'sections': {...}, 'materials': {...}}

    A variant keeps the base's at its last sync (``Variant.base_state``): with
    the base's now and the variant's own, a sync can tell what the base changed
    from what the user changed."""
    from dataclasses import asdict
    out = {
        'supports': {n: list(r) for n, r in current_restraints(struc).items()},
        'node_springs': {k: asdict(v) for k, v in struc.node_springs.items()},
        'element_springs': {k: asdict(v)
                            for k, v in struc.element_springs.items()},
    }
    for kind in _ENTITY_KINDS + _ACTION_ENTITY_KINDS:
        out[kind] = _flat_entities(struc, kind)
    out[_LOADS_KIND] = {k: [_plain(r) for r in recs]
                        for k, recs in _grouped_loads(struc).items()}
    return out


def _support_plan(old: dict, new: dict, cur: dict, kinds=_PLAN_KINDS):
    """Three-way comparison, per node (or element) and kind. Returns
    ``(follow, overrides, conflicts)``: ``follow`` is what the base changed on
    something the variant left alone — ``{(kind, key): value or None}`` to
    apply (None: remove); ``overrides`` the keys where the variant differs from
    the base now; ``conflicts`` those the base changed AND the variant had
    changed differently (the variant's value stays)."""
    follow, overrides, conflicts = {}, [], []
    for kind in kinds:
        o, n_, c = old.get(kind), new.get(kind, {}), cur.get(kind, {})
        if o is None:                    # a record from before this kind was kept
            o = n_
        for k in set(o) | set(n_) | set(c):
            b0, b1, m = o.get(k), n_.get(k), c.get(k)
            if m != b1:
                overrides.append((kind, k))
            if m == b0:
                if b1 != m:
                    follow[(kind, k)] = b1
            elif b1 != b0 and b1 != m:
                conflicts.append((kind, k))
    return follow, sorted(overrides), sorted(conflicts)


def _prune_our_supports(model):
    """Drop the canonical support definitions nothing uses any more (mirrors
    MainWindow._prune_supports_we_made, without the Qt dependency); anything
    under a name we did not generate is left alone."""
    from .models import CANONICAL_SUPPORTS, CANONICAL_SUPPORTS_PLATE
    canonical = set(CANONICAL_SUPPORTS.values()) | \
        set(CANONICAL_SUPPORTS_PLATE.values())
    ours = {n for n in model.supports
            if n in canonical
            or (n.rsplit('-', 1)[0] in canonical
                and n.rsplit('-', 1)[-1].isdigit())}
    model.prune_unused_supports(only=ours)


def _set_entity_field(model, source, kind, key) -> None:
    """Take one field of a node / bar / section / material from *source*."""
    import copy
    eid, field = key.rsplit('|', 1)
    mine = _entities(model, kind).get(eid)
    theirs = _entities(source, kind).get(eid)
    if mine is None or theirs is None:
        return
    setattr(mine, field, copy.deepcopy(getattr(theirs, field)))
    if kind in ('tris', 'quads', 'objects'):
        # an area that now stands on nodes, or uses a section, the model lacks
        _copy_nodes_into(source, model, _area_nodes(mine))
        sec = getattr(mine, 'section_name', None) or \
            getattr(mine, 'tri_section_name', '')
        if sec:
            _copy_area_section_into(source, model, sec)
        return
    if kind != 'bars':
        return
    # a bar that now ends on a node, or uses a section, the model lacks
    for nid in (mine.node_i, mine.node_j):
        if nid not in model.nodes and nid in source.nodes:
            n = source.nodes[nid]
            model.add_node(nid, n.x, n.y)
    if field == 'section_name' and mine.section_name not in model.sections:
        _copy_section_into(source, model, mine.section_name)


def _set_support_state(model, source, items) -> None:
    """Make *model*'s supports/springs at ``items`` (``[(kind, key), ...]``)
    those of *source*. A node or element the model does not have is skipped."""
    import copy
    src = base_state(source)
    for kind, key in items:
        if kind in _ENTITY_KINDS:
            _set_entity_field(model, source, kind, key)
        elif kind == 'supports':
            if key not in model.nodes:
                continue
            model.support_assignments = [a for a in model.support_assignments
                                         if a.node_id != key]
            r = src['supports'].get(key)
            if r is not None:
                name = model.support_for(bool(r[0]), bool(r[1]), bool(r[2]))
                if name is not None:
                    model.assign_support(key, name)
        elif kind == 'node_springs':
            if key not in model.nodes:
                continue
            if key in source.node_springs:
                model.node_springs[key] = copy.deepcopy(source.node_springs[key])
            else:
                model.node_springs.pop(key, None)
        else:
            if key not in model.bar_elements_by_id:
                continue
            if key in source.element_springs:
                model.element_springs[key] = copy.deepcopy(source.element_springs[key])
            else:
                model.element_springs.pop(key, None)
    _prune_our_supports(model)
    model._node_dof_index = None


def _plan_kinds(actions: bool):
    return _PLAN_KINDS if actions else _SUPPORT_KINDS + _ENTITY_KINDS


def merge_supports(base, model, old_state, actions: bool = True):
    """Bring what the base changed since *old_state* to the variant *model*,
    wherever the variant has not changed the same thing: its supports and
    springs, its nodes, bars, sections and materials (field by field) and,
    with *actions*, its load cases, analysis cases, combinations and loads.
    ``old_state=None`` (no sync yet): the model is taken as it is. Returns
    ``(follow, overrides, conflicts)`` as :func:`_support_plan`."""
    new = base_state(base)
    old = old_state if old_state is not None else new
    cur = base_state(model)
    follow, overrides, conflicts = _support_plan(old, new, cur,
                                                 _plan_kinds(actions))
    items = list(follow)
    _set_support_state(model, base,
                       [i for i in items if i[0] not in _ACTION_KINDS])
    _apply_actions(model, base, follow, old, cur,
                   [i for i in items if i[0] in _ACTION_KINDS])
    return follow, overrides, conflicts


def _target_exists(model, rec) -> bool:
    for f, coll in (('node_id', model.nodes),
                    ('element_id', model.bar_elements_by_id),
                    ('tri_id', model.tri_elements_by_id),
                    ('quad_id', model.quad_elements_by_id),
                    ('object_id', model.geometry_objects)):
        if hasattr(rec, f) and getattr(rec, f) not in coll:
            return False
    return True


def _apply_actions(model, base, follow, old, cur, items) -> None:
    """Apply to the variant *model* what the base changed in the actions:
    cases (and analysis cases, combinations) added, edited and removed, and
    the loads of every group (what they act on, and the case) that the base
    changed and the variant did not. In that order, so a load never meets a
    case that is not there yet."""
    import copy
    if not items:
        return
    exists = [(k, key) for k, key in items if key.endswith('|__exists__')]
    # 1) cases the base added
    for kind, key in exists:
        if follow[(kind, key)] is not True:
            continue
        eid = key.rsplit('|', 1)[0]
        if eid in _entities(model, kind) or eid not in _entities(base, kind):
            continue
        obj = copy.deepcopy(_entities(base, kind)[eid])
        if kind == 'load_cases':
            model.load_cases.append(obj); model.load_cases_by_id[eid] = obj
        elif kind == 'analysis_cases':
            model.analysis_cases.append(obj); model.analysis_cases_by_id[eid] = obj
        else:
            model.load_combinations.append(obj)
    # 2) fields of cases that exist on both sides
    for kind, key in items:
        if kind in _ACTION_ENTITY_KINDS and not key.endswith('|__exists__'):
            _set_entity_field(model, base, kind, key)
    # 3) loads, a group at a time
    base_groups = _grouped_loads(base)
    for kind, key in items:
        if kind != _LOADS_KIND:
            continue
        attr = key.split('|', 1)[0]
        lst = getattr(model, attr)
        lst[:] = [r for r in lst if _group_key(attr, r) != key]
        for rec in base_groups.get(key, []):
            if getattr(rec, 'load_case_id', None) not in (None, *model.load_cases_by_id) \
                    and rec.load_case_id not in model.load_cases_by_id:
                continue
            if _target_exists(model, rec):
                lst.append(copy.deepcopy(rec))
    # 4) cases the base removed — only one the variant left as it was
    for kind, key in exists:
        if follow[(kind, key)] is not None:
            continue
        eid = key.rsplit('|', 1)[0]
        if eid not in _entities(model, kind):
            continue
        prefix = f"{eid}|"
        if any(cur[kind].get(k) != old.get(kind, {}).get(k)
               for k in cur[kind] if k.startswith(prefix)):
            continue                                  # the variant changed it: it stays
        if kind == 'load_cases':
            model.remove_load_case(eid)
        elif kind == 'analysis_cases':
            model.remove_analysis_case(eid)
        else:
            model.remove_load_combination(eid)


def reset_variant_supports(base, variant) -> None:
    """Discard the variant's own supports and springs: they become the base's,
    and follow it again."""
    model = variant.model
    bs, ms = base_state(base), base_state(model)
    items = [(k, key) for k in _SUPPORT_KINDS for key in set(bs[k]) | set(ms[k])]
    _set_support_state(model, base, items)
    # only the supports are discarded: the record of the rest stays, so what
    # the user changed on the bars is still the user's
    rec = dict(variant.base_state or bs)
    for k in _SUPPORT_KINDS:
        rec[k] = bs[k]
    variant.base_state = rec


def copy_variant_supports(variant, source) -> None:
    """Give the variant the supports and springs of *source* (a Structure2D:
    the base, or another variant's model). They are the variant's own from
    then on; what differs from the base counts as the variant's choice."""
    model = variant.model
    ss, ms = base_state(source), base_state(model)
    items = [(k, key) for k in _SUPPORT_KINDS for key in set(ss[k]) | set(ms[k])]
    _set_support_state(model, source, items)


def apply_variant_config(base, model, active_elements=None,
                        restore_loads=True, prune=True, old_state=None,
                        actions=True):
    """Reconcile an existing variant *model* with the base and a config,
    **preserving the variant's own loads/cases/combinations and supports**.

    Two regimes share this function:

    - **Explicit / destructive** (``prune=True``, the default; used by the
      workspace's "Save variant" and "Sync with base"): bars outside
      ``active_elements`` (or gone from the base) are removed with their loads,
      and orphan nodes are dropped.
    - **Automatic / non-destructive** (``prune=False``; used by the guarded
      auto-sync on selection / Run / combinations): only ADDITIVE and property
      changes are applied — new base bars are copied in, shared entities
      re-synced (coordinates, sections/materials, hinges, connectivity) —
      nothing the user may have edited directly on the variant's model is
      deleted: bars are never removed.

    In both, the **supports and springs are the variant's own**: what the base
    changed since the last sync (``old_state``, see :func:`base_state`) reaches
    the nodes the variant did not change, and a node the user changed on the
    variant keeps its value (see :func:`merge_supports`). Discarding the
    variant's supports is explicit: :func:`reset_variant_supports`.

    In both regimes bars copied from the base get their base element loads
    restored (``restore_loads``) for the load cases the variant model has.
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

    sel = area_selection(base, active_elements)
    tri_keep = set(base.tri_elements_by_id) if sel is None else sel[0]
    quad_keep = set(base.quad_elements_by_id) if sel is None else sel[1]
    obj_keep = set(base.geometry_objects) if sel is None else sel[2]
    if prune:
        for t in list(model.tri_elements):
            if t.id not in tri_keep:
                model.remove_tri_element(t.id)
        for q in list(model.quad_elements):
            if q.id not in quad_keep:
                model.remove_quad_element(q.id)

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

    # 2c) The base's triangles, quads and geometry objects the variant keeps
    #     and does not have yet are copied in, like the bars. Objects are
    #     meshed at solve time, on the variant's own model. What the base
    #     changes on the ones the variant already has (corners, section, target
    #     size, edge modes, nodes of a triangle...) reaches it by the field-wise
    #     merge of step 4, so an edit made on the variant stays; removal of what
    #     the base lost is the destructive regime's (prune) alone.
    if prune:
        for oid in list(model.geometry_objects):
            if oid not in obj_keep:
                model.remove_geo_object(oid)
    for tid in sorted(tri_keep):
        t = base.tri_elements_by_id.get(tid)
        if t is None or tid in model.tri_elements_by_id:
            continue
        _copy_nodes_into(base, model, _area_nodes(t))
        _copy_area_section_into(base, model, t.section_name)
        try:
            model.add_tri_element(tid, t.node_i, t.node_j, t.node_k,
                                  t.section_name)
        except ValueError:
            continue            # the id is the variant's own: variant_id_collisions
    for qid in sorted(quad_keep):
        q = base.quad_elements_by_id.get(qid)
        if q is None or qid in model.quad_elements_by_id:
            continue
        _copy_nodes_into(base, model, _area_nodes(q))
        _copy_area_section_into(base, model, q.section_name)
        try:
            model.add_quad_element(qid, q.node_i, q.node_j, q.node_k,
                                   q.node_l, q.section_name)
        except ValueError:
            continue
    for oid in sorted(obj_keep):
        obj = base.geometry_objects.get(oid)
        if obj is None or oid in model.geometry_objects:
            continue
        _copy_nodes_into(base, model, _area_nodes(obj))
        if getattr(obj, 'tri_section_name', ''):
            _copy_area_section_into(base, model, obj.tri_section_name)
        model.geometry_objects[oid] = copy.deepcopy(obj)
    # corner/vertex nodes an object already in the model lost (older builds
    # could save that) come back from the base.
    for oid, obj in model.geometry_objects.items():
        if oid in base.geometry_objects:
            _copy_nodes_into(base, model, _area_nodes(base.geometry_objects[oid]))

    # 3) Nodes a base bar was rewired to must exist in the model. What the
    #    base changed on the entities the model already has (coordinates, a
    #    bar's hinges / section / end nodes, section and material properties)
    #    is merged field by field in step 4: it reaches what the variant has
    #    not changed.
    for e in model.bar_elements:
        be = base.bar_elements_by_id.get(e.id)
        if be is None:
            continue
        for nid in (be.node_i, be.node_j):
            if nid not in model.nodes and nid in base.nodes:
                bn = base.nodes[nid]
                model.add_node(nid, bn.x, bn.y)

    # 4) The variant's own edits stay. Only what the BASE changed since the
    #    last sync, where the variant had not changed the same thing, is taken
    #    over: supports and springs, nodes, bars, sections and materials.
    merge_supports(base, model, old_state, actions)

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

def _variant_sync_sig(base, variant) -> str:
    """Signature of everything a variant model derives from: the base's
    stiffness-relevant state and supports, and the declared active set. Equal
    signatures ⇒ a reconcile would be a no-op."""
    import hashlib
    import json
    parts = [base.stiffness_hash()]
    ae = getattr(variant, 'active_elements', None)
    parts.append('*' if ae is None else '|'.join(sorted(ae)))
    parts.append(json.dumps(base_state(base), sort_keys=True))
    return hashlib.sha256('\n'.join(parts).encode('utf-8')).hexdigest()


def sync_variant_with_base(base, variant, force: bool = False,
                           destructive: bool = False) -> bool:
    """Reconcile *variant.model* with the current base (see
    :func:`apply_variant_config`). Guarded by a signature of the base state +
    variant config, so calling it repeatedly is O(hash) when nothing changed.

    By default the reconcile is NON-DESTRUCTIVE (additive + property sync
    only): it never deletes bars from the variant's model. Pass
    ``destructive=True`` only from explicit user actions ("Sync with base",
    "Save variant") where the declared config is meant to win. Neither ever
    touches a support or spring the user changed on the variant.

    Returns True when a reconcile actually ran; False for a guarded no-op or a
    variant without its own model (nothing to sync — it derives on demand).
    """
    model = getattr(variant, 'model', None)
    if model is None:
        return False
    from .structure import attach_family
    attach_family(base, model)
    sig = _variant_sync_sig(base, variant)
    # A model missing bars, or missing a geometry object's corner/vertex nodes,
    # while the base has them, is damage (older builds could save either kind),
    # not a state the guard may preserve: the additive sync below repopulates
    # it from the base. Checking bar_elements alone missed a plate/wall model
    # (geometry objects only, no bars at all) whose object had silently lost
    # nodes — the guard's signature match then hid the damage forever.
    missing_obj_nodes = any(
        nid not in model.nodes
        for oid, obj in base.geometry_objects.items()
        if oid in model.geometry_objects
        for nid in getattr(obj, 'node_ids', []) or [])
    damaged = ((not model.bar_elements and base.bar_elements)
              or missing_obj_nodes)
    if (not force and not damaged
            and getattr(variant, '_synced_base_sig', None) == sig):
        return False
    apply_variant_config(base, model,
                         active_elements=variant.active_elements,
                         prune=destructive,
                         old_state=variant.base_state,
                         actions=variant.inherit_actions)
    variant.base_state = base_state(base)
    variant._synced_base_sig = sig
    return True


def variant_base_diff(base, variant) -> dict:
    """What a sync would change on *variant.model* — pure, no mutation.

    Returns ``{'in_sync': bool, 'add': [elem ids], 'remove': [elem ids],
    'overrides': [(kind, key)], 'follow': [(kind, key)], 'conflicts':
    [(kind, key)]}``. ``in_sync`` is the signature guard's verdict (False also
    when only properties — sections, coordinates, hinges — changed). The last
    three are about the supports and springs (``kind`` is ``'supports'``,
    ``'node_springs'`` or ``'element_springs'``, ``key`` the node or element):
    ``overrides`` where the variant differs from the base now, ``follow`` what
    a sync would take from the base, and ``conflicts`` where both changed.
    """
    model = getattr(variant, 'model', None)
    if model is None:
        return {'in_sync': True, 'add': [], 'remove': [], 'overrides': [],
                'follow': [], 'conflicts': []}
    sig = _variant_sync_sig(base, variant)
    in_sync = (getattr(variant, '_synced_base_sig', None) == sig)

    base_ids = set(base.bar_elements_by_id)
    ae = getattr(variant, 'active_elements', None)
    active = (set(ae) & base_ids) if ae else base_ids
    have = set(model.bar_elements_by_id)
    sel = area_selection(base, ae)
    for base_c, mine_c, i in (
            (base.tri_elements_by_id, model.tri_elements_by_id, 0),
            (base.quad_elements_by_id, model.quad_elements_by_id, 1),
            (base.geometry_objects, model.geometry_objects, 2)):
        want = set(base_c) if sel is None else sel[i]
        active |= want
        have |= set(mine_c)

    new = base_state(base)
    old = variant.base_state if variant.base_state is not None else new
    follow, overrides, conflicts = _support_plan(
        old, new, base_state(model), _plan_kinds(variant.inherit_actions))
    return {'in_sync': in_sync,
            'add': sorted(active - have),
            'remove': sorted(have - active),
            'overrides': overrides,
            'follow': sorted(follow),
            'conflicts': conflicts,
            'id_collisions': variant_id_collisions(base, variant)}


def variant_id_collisions(base, variant) -> list:
    """Ids the variant's own model gave to a node, bar, triangle, quad or
    geometry object of its own that the base has since used for another one
    (``[(kind, id), ...]``). Such an id names two different things, and the
    base's does not reach the variant (the sync keeps the variant's). New ids
    are taken from the whole family (:meth:`Structure2D._family_keys`), so this
    only happens with ids typed by hand or from files that predate that."""
    model = getattr(variant, 'model', None)
    old = getattr(variant, 'base_state', None)
    if model is None or old is None:
        return []
    out = []
    for kind in ('nodes', 'bars', 'tris', 'quads', 'objects'):
        was = {k.rsplit('|', 1)[0] for k in old.get(kind) or {}}
        if not was and kind not in old:
            continue                  # a record from before this kind was kept
        now = set(_entities(base, kind))
        for eid in sorted(set(_entities(model, kind)) - was):
            if eid in now:
                out.append((kind, eid))
    return out


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
        'variants': [{'id': v.id,
                      'n_active': (None if v.active_elements is None
                                   else len(v.active_elements))}
                     for v in struc.variants.values()],
        'sequences': [{'id': s.id, 'n_phases': len(s.phases),
                       'method': s.displacement_method}
                      for s in struc.construction_sequences.values()],
    }
