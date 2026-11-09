"""Eurocode 5 design of timber bar members (EN 1995-1-1).

Ties the xdfem2d model to the eurocodepy timber checks and produces, for every
timber-section element, the utilization ratios of the grouped cross-section /
member verification (:func:`eurocodepy.ec5.uls.eurocode5_section_check`):

* **bending + axial** (N + My + Mz) — including the compression (``k_c``) and
  lateral-torsional (``k_m``) stability factors;
* **shear** (Vy, Vz) and their interaction;
* **torsion** (T) and the shear + torsion interaction.

The **buckling lengths** are obtained exactly like the steel design
(:mod:`steel_design`): the collinear bars of a column/beam are grouped into a
*physical member* (:func:`member_utils.identify_members`) and the member length
is multiplied by the per-member ``K`` factors (``sd_ky`` / ``sd_kz`` / ``sd_klt``,
or the global defaults) to give ``l_0y`` / ``l_0z`` / ``l_0m`` fed to the EC5
stability factors.

Both structural domains are handled:

* ``plane`` (frames): the bar carries axial ``N`` (tension +), in-plane bending
  ``M`` and shear ``V``;
* ``plate`` (grillages): the bar's ``N`` slot is the St-Venant torsion ``T``; it
  carries out-of-plane bending and shear, with no axial force.

The in-plane actions are resolved onto the section's principal axes with the
section orientation angle θ (``My = M·cosθ``, ``Mz = M·sinθ`` …), as for steel.

Forces are enveloped over each element's critical points and over the selected
combinations (conservative). Sign is preserved for the axial force, so both the
tension and the compression governing points are checked (the EC5 rules differ).

Bars carrying a ``beam`` tag are designed as tagged beams (:mod:`timber_beams`):
the member is the whole tagged chain, the buckling length that of the span
between supports / hinges / joining bars (``lateral_spacing`` on the beam, when
set, replaces ``K_LT·L``).

The result is ``{"members": [...member rows...], "elements": {id: ratios}}``
where each element's ``ratios`` dict has ``bending``, ``shear``, ``torsion`` and
``combined`` (the governing maximum).
"""
from __future__ import annotations

import math

from .models import SectionShape, SectionType, section_type_of
from .member_utils import identify_members

# Check keys exposed to the view / table.
CHECK_KEYS = ("combined", "bending", "shear", "torsion")


def _timber_grade(material) -> str | None:
    """Timber grade label (e.g. 'C24', 'GL24h') from the material design dict."""
    if material is None:
        return None
    grade = (getattr(material, "design", {}) or {}).get("class")
    return str(grade) if grade else None


def _gamma_m_key(grade: str) -> str:
    """Code-preferences γM key for a timber grade label."""
    g = (grade or "").upper()
    if g.startswith(("C", "D")):
        return "timber_gammaM_solid"
    if g.startswith("GL"):
        return "timber_gammaM_glulam"
    if g.startswith("LVL"):
        return "timber_gammaM_lvl"
    if g.startswith("CLT"):
        return "timber_gammaM_clt"
    return "timber_gammaM_panels"


def _build_timber(material, grade, prefs):
    """Build a eurocodepy Timber object for *grade*, then overlay the values the
    user edited in the GUI so the design is computed with **those** properties:

    * the characteristic strengths ``fmk`` / ``fvk`` / ``fc0k`` / ``ft0k`` from
      the material dialog (when present) override the database values;
    * the partial factor ``γM`` comes from the Code preferences (per material
      family) rather than the eurocodepy default.

    The elastic constants ``E0,05`` / ``G`` (used only by the buckling factors)
    keep the grade values — they are not editable as 5%-fractiles in the GUI.
    """
    from eurocodepy.ec5.materials import TimberClass
    edition = str((prefs or {}).get("ec5_edition", "2004"))
    try:
        # the edition selects k_mod, k_def, k_h, the service classes (SC4 is
        # EN 1995-1-1:2025 only) and the partial factors of the database
        timber = (TimberClass(grade, edition="2025") if edition == "2025"
                  else TimberClass(grade))
    except Exception:       # a grade eurocodepy does not know
        return None
    design = (getattr(material, "design", {}) or {})
    for key in ("fmk", "fvk", "fc0k", "ft0k"):
        val = design.get(key)
        if val in (None, ""):
            continue
        try:
            setattr(timber, key, float(val))
        except (TypeError, ValueError):
            pass
    gm = (prefs or {}).get(_gamma_m_key(grade))
    try:
        if gm not in (None, ""):
            timber.safety = float(gm)
    except (TypeError, ValueError):
        pass
    return timber


# EN 1995-1-1:2004 Table 2.1 — action → load-duration class, and the rank used
# to pick the shortest-duration action present in a combination (§2.3.1.2(2)P:
# the k_mod for the action with the *shortest* duration governs; shortest = the
# smallest rank = the largest k_mod).
_ACTION_LOAD_DURATION = {
    "G": "Permanent", "Q": "MediumDuration", "W": "ShortDuration",
    "E": "Instantaneous", "T": "ShortDuration",
    "S": "ShortDuration", "C": "MediumDuration", "A": "Instantaneous",
    "F": "Instantaneous", "O": "Permanent",
}
_DURATION_RANK = {
    "Instantaneous": 0, "ShortDuration": 1, "MediumDuration": 2,
    "LongDuration": 3, "Permanent": 4,
}


def _combo_load_cases(struc, combo_id):
    """The LoadCase objects present in *combo_id*.

    Resolves the combination's load cases directly and through its analysis
    cases (each analysis case is itself a combination of load cases).
    """
    combo = None
    for c in getattr(struc, "load_combinations", []):
        if c.id == combo_id:
            combo = c
            break
    if combo is None:
        return []
    lc_ids: set[str] = set()
    for lc_id, coeff in (combo.coefficients or {}).items():
        if coeff:
            lc_ids.add(lc_id)
    for ac_id, coeff in (getattr(combo, "analysis_coefficients", {}) or {}).items():
        if not coeff:
            continue
        ac = struc.analysis_cases_by_id.get(ac_id)
        if ac is not None:
            for lc_id, f in (ac.coefficients or {}).items():
                if f:
                    lc_ids.add(lc_id)
    return [struc.load_cases_by_id[i] for i in sorted(lc_ids)
            if i in struc.load_cases_by_id]


def _case_load_duration(lc):
    """Load-duration class name of one load case: the one set on the case, else
    the one its action type implies (Table 2.1). None when neither is known."""
    explicit = getattr(lc, "load_duration", "") or ""
    if explicit:
        return explicit
    letter = getattr(lc.action_type, "value", str(lc.action_type))
    return _ACTION_LOAD_DURATION.get(letter)


def _combo_load_duration(struc, combo_id, default_ld):
    """Governing (shortest-duration) LoadDuration for *combo_id* from the load
    durations of its cases (EN 1995-1-1 §2.3.1.2); falls back to *default_ld*
    when unresolved."""
    from eurocodepy.ec5.materials import LoadDuration
    classes = [_case_load_duration(lc) for lc in _combo_load_cases(struc, combo_id)]
    classes = [c for c in classes if c]
    if not classes:
        return default_ld
    governing = min(classes, key=lambda c: _DURATION_RANK.get(c, 2))
    try:
        return LoadDuration[governing]
    except KeyError:
        return default_ld


def _cross_section(sec):
    """A eurocodepy CrossSection (m units) for the timber section, or None."""
    from eurocodepy.utils import CircularCrossSection, RectangularCrossSection
    if sec.shape in (SectionShape.CIRCULAR, SectionShape.CIRCULAR_HOLLOW):
        return CircularCrossSection(sec.b)
    # Rectangular / generic: width = b, height (depth, major axis) = h.
    if sec.b > 0 and sec.h > 0:
        return RectangularCrossSection(sec.b, sec.h)
    return None


def _service_class(sec):
    from eurocodepy.ec5.materials import ServiceClass
    label = str(getattr(sec, "timber_service_class", "SC1") or "SC1")
    try:
        return ServiceClass[label]
    except KeyError:
        return ServiceClass.SC1


def _load_duration(prefs):
    """Fallback load-duration class, used only when a combination's actions
    cannot be resolved (per-combination duration is derived from the action
    types — see ``_combo_load_duration``)."""
    from eurocodepy.ec5.materials import LoadDuration
    return LoadDuration.MediumDuration


def _combo_cases(results, combinations):
    out = {}
    for combo_id, cd in results.get("combinations", {}).items():
        ef = cd.get("element_forces", {})
        if isinstance(ef, dict) and "max" in ef:
            out[combo_id] = [("max", ef.get("max", {})), ("min", ef.get("min", {}))]
        else:
            out[combo_id] = [("", ef)]
    if combinations is not None:
        wanted = set(combinations)
        out = {k: v for k, v in out.items() if k in wanted}
    return out


def _critical(ef, dist=None):
    from .rc_design import _critical_points
    return _critical_points(ef, dist)


def _span_dist(results, combo_id, sub_label, elem_id):
    """The (N, V, M, x) span distribution for one element in one combo, or
    ``None`` when unavailable — either because the combo is an envelope
    sub-case (``sub_label`` set: max/min carry no single span distribution,
    same restriction as :func:`rc_design.design_concrete_sections`) or
    because the solver has no entry for it.

    Without this, ``_critical`` falls back to the element's own end forces
    only — correct for a member whose critical section really is at a node,
    but silently wrong (missed mid-span M/V peak) for e.g. a simply
    supported beam modelled as a single bar element, where the true M_max
    sits mid-span, not at either end."""
    if sub_label:
        return None
    return results.get("combo_distribution", {}).get(combo_id, {}).get(elem_id)


def _is_timber(struc, sec):
    return (sec is not None
            and section_type_of(struc.materials.get(sec.material_name))
            == SectionType.TIMBER
            and sec.shape in (SectionShape.RECTANGULAR, SectionShape.GENERIC,
                              SectionShape.CIRCULAR))


def _where_text(loc):
    """'E0 · x=2.50 m · ULS-1 (max)' for a governing point."""
    pt = loc.get("point", "")
    pt = {"i": "end i", "j": "end j"}.get(pt, pt)
    case = f" ({loc['case']})" if loc.get("case") else ""
    return f"{loc.get('element', '')} · {pt} · {loc.get('combination', '')}{case}"


def _pt_force(pts, key, attr):
    """A force component at the point governing check *key* (0 when none)."""
    pt = (pts or {}).get(key)
    return float(getattr(pt[2], attr, 0.0)) if pt else 0.0


_CHECK_SECTIONS = (("bending", "Bending + axial", "Bending"),
                   ("shear", "Shear", "Shear"),
                   ("torsion", "Shear + torsion", "Torsion"))


def _sections_at_governing_points(rd, pts, mid, CalcReport, check):
    """Rebuild the report so every check is shown with the forces of ITS OWN
    governing point, and says where that point is.

    A single trace is made at one point -- for a beam usually mid-span bending,
    where V ~ 0 -- so its shear section read V_Ed = 0, U_V = 0 although the
    shear ratio in the table came from the support. Each check (bending/axial,
    shear, torsion) is therefore traced at the point that governs it, and its
    own "Inputs" and result sections are labelled with element, position and
    combination. The original single-point Inputs / shear sections are dropped.
    Sections that don't depend on the point (strengths, stresses, stability,
    combined) are kept from the main trace."""
    if not pts:
        return
    traces = {}
    for key, (_u, inp, forces, loc) in pts.items():
        rep = CalcReport(title=f"Timber member {mid}", meta={})
        try:
            check(inp, forces, trace=rep)
        except TypeError:
            return
        traces[key] = ({s_["title"]: s_ for s_ in rep.to_dict().get("sections", [])},
                       loc)
    wanted = {"bending": lambda t: t.startswith("Bending"),
              "shear": lambda t: t.startswith("Shear"),
              "torsion": lambda t: t.startswith("Shear")}
    point_dep = lambda t: (t == "Inputs" or t.startswith("Shear")   # noqa: E731
                           or t.startswith("Bending"))
    kept = [s_ for s_ in rd.get("sections", []) if not point_dep(s_["title"])]
    # Order: strengths, stresses, stability (as kept), one block per check,
    # then the combined utilisation.
    tail = [s_ for s_ in kept if s_["title"].startswith("Combined")]
    head = [s_ for s_ in kept if not s_["title"].startswith("Combined")]
    body = []
    for key, label, _short in _CHECK_SECTIONS:
        if key not in traces:
            continue
        secs, loc = traces[key]
        if key == "torsion":
            # Torsion shares the shear section; only worth its own pair when its
            # point differs from the shear one and torsion actually exists.
            if pts[key][0] <= 0.0 or loc == traces.get("shear", ({}, None))[1]:
                continue
        tag = f" — {label.lower()}: {_where_text(loc)}"
        if "Inputs" in secs:
            body.append(dict(secs["Inputs"], title="Inputs" + tag))
        for t, sec in secs.items():
            if wanted[key](t):
                body.append(dict(sec, title=t + tag))
    rd["sections"] = head + body + tail


def _add_special_sections(rd, mid, special, calls, CalcReport, elem_geom):
    """Append the EN 1995-1-1 §6.4 sections to the member report *rd*: what was
    read from the bars, then each check at the point that governs it (a point
    that governs several utilisations, e.g. the apex, is traced once)."""
    if not special:
        return
    from .timber_beams import geometry_section
    secs = []
    first = next((g for b_, g in elem_geom.items() if g[0].id == mid), None)
    if first is not None:
        secs.append(geometry_section(first[0], first[1]))
    done = set()
    for key in sorted(special, key=lambda k: -special[k]):
        got = calls.get((mid, key))
        if got is None or id(got[0]) in done:
            continue
        done.add(id(got[0]))
        recipe, where = got
        rep = CalcReport(title="", meta={})
        recipe(rep)
        for sec in rep.to_dict()["sections"]:
            secs.append(dict(sec, title=f"{sec['title']} — {where}"))
    at = next((i for i, s_ in enumerate(rd.get("sections", []))
               if s_["title"].startswith("Combined")), len(rd.get("sections", [])))
    rd["sections"] = rd.get("sections", [])[:at] + secs + rd["sections"][at:]


_PROFILE_STATIONS = 15


def _inp_at(inp, eg, elem, label, sec):
    """*inp* with the section at the design point: for a beam of varying depth
    the depth there, not the constant one of the bar."""
    if eg is None:
        return inp
    from .timber_beams import local_depth
    h = local_depth(eg[1], elem, label)
    if h is None or sec.b <= 0:
        return inp
    import dataclasses
    from eurocodepy.utils import RectangularCrossSection
    return dataclasses.replace(inp, section=RectangularCrossSection(sec.b, h))


def _station_profile(elem, bg, dist_of, combo_cases, combo_ld, default_ld, inp_of,
                     xs_c, domain, geom, ec5uls, timber, svc, special_point_checks,
                     sec_):
    """Utilisation along one bar of a tagged beam: ``[(s, {bending, shear,
    taper, apex})]``, enveloped over the combinations, at up to
    ``_PROFILE_STATIONS`` stations of the span distribution (the bar's ends
    when a combination has none). ``bending`` / ``shear`` already include the
    §6.4 checks, as in the member table."""
    TimberForces = ec5uls.TimberForces
    check = ec5uls.eurocode5_section_check
    c, s_ = xs_c
    acc: dict[float, dict] = {}
    for combo_id, subcases in combo_cases.items():
        inp = inp_of(combo_id)
        for sub, ef_map in subcases:
            ef = ef_map.get(elem.id)
            if ef is None:
                continue
            dist = dist_of(combo_id, sub, elem.id)
            if dist is not None:
                import numpy as np
                x_ = np.asarray(dist["x"], dtype=float)
                idx = sorted(set(np.linspace(0, len(x_) - 1,
                                             min(_PROFILE_STATIONS, len(x_)))
                                 .astype(int).tolist()))
                stations = [(float(x_[i]), float(dist["N"][i]),
                             float(dist["V"][i]), float(dist["M"][i]))
                            for i in idx]
            else:
                stations = [(0.0, ef["i"][0], ef["i"][1], ef["i"][2]),
                            (bg.length, -ef["j"][0], -ef["j"][1], ef["j"][2])]
            for x, n, v, m_val in stations:
                label = ("i" if x <= 1e-9 else "j" if x >= bg.length - 1e-9
                         else f"x={x:.6f}m")
                if domain == "plate":
                    n_ed, t_ed = 0.0, abs(n)
                else:
                    n_ed, t_ed = n, 0.0
                forces = TimberForces(n_ed=n_ed, my_ed=m_val * c, mz_ed=m_val * s_,
                                      vy_ed=v * s_, vz_ed=v * c, t_ed=t_ed)
                res = check(_inp_at(inp, (None, geom), elem, label, sec_), forces)
                sp = special_point_checks(
                    ec5uls, geom, elem, label, v, m_val, timber, svc,
                    combo_ld.get(combo_id, default_ld))
                row = {"bending": max(res.util_bending_axial, sp.get("taper", 0.0),
                                      sp.get("apex_bending", 0.0),
                                      sp.get("curve", 0.0)),
                       "shear": max(res.util_shear, sp.get("apex_t90", 0.0),
                                    sp.get("apex_inter", 0.0),
                                    sp.get("apex_rf", 0.0)),
                       "taper": sp.get("taper", 0.0),
                       "apex": max(sp.get("apex_bending", 0.0),
                                   sp.get("apex_t90", 0.0),
                                   sp.get("apex_inter", 0.0),
                                   sp.get("apex_rf", 0.0))}
                key = round(bg.s_at(x), 4)
                cur = acc.setdefault(key, dict.fromkeys(row, 0.0))
                for k_, u_ in row.items():
                    cur[k_] = max(cur[k_], u_)
    return sorted(acc.items())


def design_timber_members(struc, results, prefs=None, combinations=None,  # noqa: C901, PLR0912, PLR0915
                          with_reports=False, with_profile=False):
    """Run the EN 1995-1-1 ULS check on every timber-section member.

    Returns ``{"members": rows, "elements": {elem_id: ratios}}`` (see the module
    docstring). Requires the eurocodepy package.

    With ``with_profile`` every tagged beam with a geometry also gets, in its
    member row, ``profile`` (utilisation along the beam, for the elevation) and
    ``outline`` (the beam drawn as it is).

    With ``with_reports`` the governing case of each member is re-run once with a
    calculation trace, returned under ``"reports"`` ({member_id: report_dict});
    recording only, not persisted. Both the 2004 and 2025 editions' checks now
    take ``trace=`` (the ``except TypeError`` below is only a defensive
    fallback in case a future edition still lacks it).
    """
    # Pick the EC5 edition: the 2004 checks live in ec5.uls, the second-
    # generation §8 ones in ec5.uls2025. Both expose the same names and share
    # the force/input/result dataclasses, so the grouped check is swappable.
    edition = str((prefs or {}).get("ec5_edition", "2004"))
    if edition == "2025":
        from eurocodepy.ec5 import uls2025 as _ec5uls
    else:
        from eurocodepy.ec5 import uls as _ec5uls
    TimberForces = _ec5uls.TimberForces
    TimberSectionInput = _ec5uls.TimberSectionInput
    eurocode5_section_check = _ec5uls.eurocode5_section_check

    prefs = prefs or {}
    ky_def = float(prefs.get("steel_ky", 1.0))
    kz_def = float(prefs.get("steel_kz", 1.0))
    klt_def = float(prefs.get("steel_klt", 1.0))
    default_ld = _load_duration(prefs)
    domain = getattr(struc, "domain", "plane")

    members, by_bar = identify_members(struc)
    from .member_utils import (_incident_bar_count, _constraint_nodes,
                                cantilever_k_for)
    _incident = _incident_bar_count(struc)
    _constrained = _constraint_nodes(struc)
    combo_cases = _combo_cases(results, combinations)
    # Load-duration class per combination, from its action types (§2.3.1.2).
    combo_ld = {cid: _combo_load_duration(struc, cid, default_ld)
                for cid in combo_cases}

    # Per-element physical-member length and K factors (same procedure as steel).
    elem_member = {}
    for m in members:
        b0 = struc.bar_elements_by_id[m.bar_ids[0]]
        _cant = cantilever_k_for(struc, m, prefs, _incident, _constrained)
        ky_d = _cant if _cant is not None else ky_def
        kz_d = _cant if _cant is not None else kz_def
        ky = b0.sd_ky if b0.sd_ky is not None else ky_d
        kz = b0.sd_kz if b0.sd_kz is not None else kz_d
        klt = b0.sd_klt if b0.sd_klt is not None else klt_def
        for bid in m.bar_ids:
            elem_member[bid] = (m, ky, kz, klt)

    # Tagged beams (Structure2D.beams): the whole tagged chain is the member
    # reported, but the buckling length is that of the span between supports /
    # hinges / joining bars, and a change of section or direction does not end
    # it. They replace the per-member grouping for their bars.
    from .timber_beams import resolve_timber_beams
    tbeams, beam_warnings = resolve_timber_beams(
        struc, lambda b: _is_timber(struc, struc.sections.get(b.section_name)))
    from .timber_beams import beam_geometry, local_depth, special_point_checks
    step_warn = float((prefs or {}).get("timber_taper_step_warn", 0.10) or 0.10)
    elem_geom: dict[str, tuple] = {}          # bar id → (beam, BeamGeometry)
    elem_span_len: dict[str, float] = {}      # bar id → span length [m]
    elem_l0m_fixed: dict[str, float] = {}     # bar id → l_0m [mm] set on the beam
    _special = hasattr(_ec5uls, "check_apex_zone")
    for tb in tbeams:
        geom = beam_geometry(struc, tb, step_warn) if _special else None
        if geom is not None:
            beam_warnings.extend(geom.warnings)
            if geom.apex is not None:
                b0_ = struc.bar_elements_by_id[tb.bar_ids[0]]
                sec0 = struc.sections.get(b0_.section_name)
                g0 = (_timber_grade(struc.materials.get(sec0.material_name))
                      or "") if sec0 else ""
                if not g0.upper().startswith(("GL", "LVL")):
                    beam_warnings.append(
                        f"Beam {tb.id}: EN 1995-1-1:2025 §8.3.4 applies only "
                        "to softwood glued laminated timber and double tapered "
                        f"LVL (grade {g0 or '?'})." if edition == "2025" else
                        f"Beam {tb.id}: EN 1995-1-1 §6.4.3 applies only to "
                        f"glued laminated timber and LVL (grade {g0 or '?'}).")
            if geom.reinforced and geom.apex is not None and edition != "2025":
                beam_warnings.append(
                    f"Beam {tb.id}: 'reinforced' is ignored — EN 1995-1-1:2004 "
                    "has no rule that exempts a reinforced apex from the tension "
                    "perpendicular to the grain check (EN 1995-1-1:2025 "
                    "§8.3.4.2(1) does).")
            if (edition == "2025" and geom.taper_edge and any(
                    bg.alpha > math.radians(15.0) + 1e-9
                    for bg in geom.bars.values())):
                beam_warnings.append(
                    f"Beam {tb.id}: the taper exceeds 15°; EN 1995-1-1:2025 "
                    "Eq. 8.57 (σ = 6M/(bh²)) is given up to 15° only.")
            for bid in tb.bar_ids:
                elem_geom[bid] = (tb, geom)
        elif not _special and (tb.params or len(tb.bar_ids) > 1):
            beam_warnings.append(
                f"Beam {tb.id}: this eurocodepy has no EN 1995-1-1 §6.4 "
                "checks (tapered / curved / pitched beams); update it.")
        for span in tb.spans:
            b0 = struc.bar_elements_by_id[span.bar_ids[0]]
            _cant = cantilever_k_for(struc, span, prefs, _incident, _constrained)
            ky_d = _cant if _cant is not None else ky_def
            kz_d = _cant if _cant is not None else kz_def
            ky = b0.sd_ky if b0.sd_ky is not None else ky_d
            kz = b0.sd_kz if b0.sd_kz is not None else kz_d
            klt = b0.sd_klt if b0.sd_klt is not None else klt_def
            for bid in span.bar_ids:
                elem_member[bid] = (tb.member, ky, kz, klt)
                elem_span_len[bid] = span.length
                if tb.lateral_spacing is not None:
                    elem_l0m_fixed[bid] = tb.lateral_spacing * 1e3

    element_ratios: dict[str, dict] = {}
    # `member_env[mid]` is the true governing ratio per CHECK TYPE across
    # every element of the physical member — independently enveloped, same
    # treatment as the per-point envelope above (a member built of several
    # bar elements can have its worst bending in one element and its worst
    # shear in another, e.g. a simply supported beam split at midspan: the
    # support element governs shear, the midspan element governs bending).
    member_env: dict[str, dict] = {}
    # `member_meta[mid]` carries only the narrative fields (N_Ed/My_Ed/
    # combination/section/…) from whichever single element governs overall
    # — informational, not a source of any reported ratio.
    member_meta: dict[str, dict] = {}
    member_report_src: dict[str, tuple] = {}   # member_id → (inp, forces)
    # §6.4 utilisations per member (taper, apex_bending, apex_t90, ...) and
    # where each one governs.
    member_special: dict[str, dict] = {}
    member_special_at: dict[tuple, str] = {}
    member_special_calls: dict[tuple, tuple] = {}   # (member, key) → (recipe, where)
    member_profile: dict[str, dict] = {}            # member → {s: utilisations}
    member_reinforce: dict[str, str] = {}           # member → why (2025 §8.3.1)
    member_rf: dict[str, dict] = {}                 # member → reinforcement force
    member_rf_check: dict[str, tuple] = {}          # member → (util, mode, n)
    member_rf_notes: dict[str, list] = {}           # member → limits not met
    # member_id → {check: (util, inp, forces, location)}: the point governing
    # each check on its own (shear peaks at the supports, bending at mid-span).
    member_pts: dict[str, dict] = {}

    for elem in struc.bar_elements:
        sec = struc.sections.get(elem.section_name)
        if not _is_timber(struc, sec):
            continue
        material = struc.materials.get(sec.material_name)
        grade = _timber_grade(material)
        timber = _build_timber(material, grade, prefs)
        xs = _cross_section(sec)
        if timber is None or xs is None:
            continue

        mm = elem_member.get(elem.id)
        if mm is not None:
            member, ky, kz, klt = mm
            lmm = elem_span_len.get(elem.id, member.length) * 1e3
        else:
            member, ky, kz, klt = None, ky_def, kz_def, klt_def
            lmm = 0.0
        l0y, l0z, l0m = ky * lmm, kz * lmm, klt * lmm
        if elem.id in elem_l0m_fixed:
            l0m = elem_l0m_fixed[elem.id]
        svc = _service_class(sec)
        try:                    # no k_mod for this grade in this service class?
            for _cid in combo_cases:
                timber.k_mod(svc, combo_ld.get(_cid, default_ld))
        except KeyError as exc:
            beam_warnings.append(
                f"Element {elem.id}: {exc.args[0]} It was not designed.")
            continue
        angle = getattr(sec, "angle", 0.0) or 0.0
        th = math.radians(angle)
        c, s = math.cos(th), math.sin(th)

        # `env` is the true governing ratio per CHECK TYPE, each one taken
        # from the (combination, critical-point) pair that actually produces
        # it — never a mix of the worst N from one point with the worst M
        # from another, since bending+axial (k_c) and shear+torsion are
        # interaction checks that only mean something for forces that
        # actually occur together. So: run the full EC5 check at every
        # (combination, critical point) — as before — but instead of
        # keeping only the single point with the highest *combined* ratio
        # (which silently dropped whichever check type didn't win there),
        # envelope bending/shear/torsion independently across every point
        # actually checked. `best`/`best_meta` still track the single
        # overall-governing point too, but now purely for the report
        # narrative (N_Ed/My_Ed/combination shown in the table) — they no
        # longer feed the stored utilizations.
        env = {"bending": 0.0, "shear": 0.0, "torsion": 0.0}
        best = None
        best_inp = best_forces = None
        # The point that governs SHEAR on this element: it is generally not
        # the one that governs bending (V peaks at the supports, M at mid-span),
        # so its forces and report are tracked on their own.
        pts = {}     # check key -> (util, inp, forces, location) of its worst point
        for _combo_id, subcases in combo_cases.items():
            # k_mod follows the shortest-duration action in this combination.
            inp = TimberSectionInput(
                section=xs, timber=timber, service_class=svc,
                load_duration=combo_ld.get(_combo_id, default_ld),
                l_0y=l0y, l_0z=l0z, l_0m=l0m)
            for _sub, ef_map in subcases:
                ef = ef_map.get(elem.id)
                if ef is None:
                    continue
                dist = _span_dist(results, _combo_id, _sub, elem.id)
                for _l, _r, n, v, m_val in _critical(ef, dist):
                    _where = {"element": elem.id, "point": _l,
                              "combination": _combo_id, "case": _sub}
                    if domain == "plate":
                        n_ed, t_ed = 0.0, abs(n)
                    else:
                        n_ed, t_ed = n, 0.0          # tension +, compression −
                    my = m_val * c
                    mz = m_val * s
                    vz = v * c
                    vy = v * s
                    forces = TimberForces(
                        n_ed=n_ed, my_ed=my, mz_ed=mz,
                        vy_ed=vy, vz_ed=vz, t_ed=t_ed)
                    eg = elem_geom.get(elem.id)
                    inp_pt = _inp_at(inp, eg, elem, _l, sec)
                    res = eurocode5_section_check(inp_pt, forces)
                    cand = {"bending": res.util_bending_axial,
                            "shear": res.util_shear,
                            "torsion": res.util_torsion}
                    if eg is not None and domain != "plate":
                        _calls, _info = {}, {}
                        sp = special_point_checks(
                            _ec5uls, eg[1], elem, _l, v, m_val, timber, svc,
                            combo_ld.get(_combo_id, default_ld), calls=_calls,
                            info=_info)
                        if _info.get("reinforce") and member is not None:
                            member_reinforce.setdefault(
                                member.id, _info["reinforce"])
                        if _info.get("rf_notes") and member is not None:
                            member_rf_notes.setdefault(member.id, [])
                            for n_ in _info["rf_notes"]:
                                if n_ not in member_rf_notes[member.id]:
                                    member_rf_notes[member.id].append(n_)
                        if _info.get("rf_util") is not None and member is not None:
                            member_rf_check[member.id] = max(
                                member_rf_check.get(member.id, (0.0, "", 0))[0],
                                _info["rf_util"]), _info["rf_governing"], _info["rf_n"]
                        if _info.get("rf_per_m") is not None and member is not None:
                            cur_rf = member_rf.setdefault(
                                member.id, {"per_m": 0.0, "force": None})
                            cur_rf["per_m"] = max(cur_rf["per_m"], _info["rf_per_m"])
                            if _info.get("rf_force") is not None:
                                cur_rf["force"] = max(cur_rf["force"] or 0.0,
                                                      _info["rf_force"])
                        if sp:
                            cand["bending"] = max(
                                cand["bending"], sp.get("taper", 0.0),
                                sp.get("apex_bending", 0.0), sp.get("curve", 0.0))
                            cand["shear"] = max(
                                cand["shear"], sp.get("apex_t90", 0.0),
                                sp.get("apex_inter", 0.0), sp.get("apex_rf", 0.0))
                            if member is not None:
                                ms = member_special.setdefault(member.id, {})
                                for k_, u_ in sp.items():
                                    if u_ > ms.get(k_, 0.0):
                                        ms[k_] = u_
                                        member_special_at[(member.id, k_)] = \
                                            _where_text(_where)
                                        rk = "apex" if k_.startswith("apex") else k_
                                        if rk in _calls:
                                            member_special_calls[(member.id, k_)] = (
                                                _calls[rk], _where_text(_where))
                    for key, val in cand.items():
                        env[key] = max(env[key], val)
                    _ld = combo_ld.get(_combo_id, default_ld)
                    for key, val in cand.items():
                        if key not in pts or val > pts[key][0]:
                            pts[key] = (val, inp_pt, forces, _where)
                    if best is None or max(cand.values()) > max(best.values()):
                        best = cand
                        best_inp, best_forces = inp_pt, forces
                        best_meta = {
                            "k_c": res.k_c, "k_m": res.k_m,
                            "N_Ed": n_ed, "My_Ed": my, "Mz_Ed": mz,
                            "load_duration": getattr(_ld, "name", str(_ld)),
                            "combination": _combo_id}
        if best is None:
            continue
        # A beam that is not straight and prismatic can peak between the
        # critical sections of the bar (the depth changes along it): sample
        # stations along the span and fold the worst into the utilisation.
        eg_ = elem_geom.get(elem.id)
        if (eg_ is not None and member is not None
                and (eg_[1].kind != "straight" or eg_[1].depth_kind != "constant")):
            for s_pos, vals in _station_profile(
                    elem, eg_[1].bars[elem.id],
                    lambda cid, sub, eid: _span_dist(results, cid, sub, eid),
                    combo_cases, combo_ld, default_ld,
                    lambda cid: TimberSectionInput(
                        section=xs, timber=timber, service_class=svc,
                        load_duration=combo_ld.get(cid, default_ld),
                        l_0y=l0y, l_0z=l0z, l_0m=l0m),
                    (c, s), domain, eg_[1], _ec5uls, timber, svc,
                    special_point_checks, sec):
                env["bending"] = max(env["bending"], vals["bending"])
                env["shear"] = max(env["shear"], vals["shear"])
                cur = member_profile.setdefault(member.id, {}).setdefault(
                    s_pos, dict.fromkeys(vals, 0.0))
                for k_, u_ in vals.items():
                    cur[k_] = max(cur[k_], u_)
        element_ratios.setdefault(elem.id, {}).update(env)

        # Aggregate into the member row. `member_env[mid]` envelopes each
        # check type independently across every element of the member (this
        # element's `env` folds in here); `member_meta[mid]` only tracks the
        # narrative fields from whichever single ELEMENT governs the member
        # overall — never a source of a reported ratio, same split as at the
        # point level above.
        if member is not None:
            mid = member.id
            menv = member_env.setdefault(mid, {"bending": 0.0, "shear": 0.0,
                                                "torsion": 0.0})
            for key, val in env.items():
                menv[key] = max(menv[key], val)

            mp = member_pts.setdefault(mid, {})
            for key, pt in pts.items():
                if key not in mp or pt[0] > mp[key][0]:
                    mp[key] = pt

            gov = max(env.values())
            meta_row = member_meta.get(mid)
            if meta_row is None or gov > meta_row["_gov"]:
                member_meta[mid] = {
                    "_gov": gov,
                    "elements": ", ".join(member.bar_ids),
                    "section": sec.name, "grade": grade or "—",
                    "edition": edition,
                    "length": member.length,
                    "combination": best_meta.get("combination", ""),
                    "load_duration": best_meta.get("load_duration", ""),
                    "service_class": getattr(sec, "timber_service_class", "SC1"),
                    "k_cy": best_meta["k_c"][0], "k_cz": best_meta["k_c"][1],
                    "k_m": best_meta["k_m"],
                    "N_Ed": best_meta["N_Ed"], "My_Ed": best_meta["My_Ed"],
                    "Mz_Ed": best_meta["Mz_Ed"],
                    "section_geom": (
                        {"shape": "rect", "b": xs.width, "h": xs.height}
                        if getattr(xs, "width", None)
                        and getattr(xs, "height", None)
                        else ({"shape": "CHS", "d": xs.diameter}
                              if getattr(xs, "diameter", None) else None))}
                member_report_src[mid] = (best_inp, best_forces)

    # Combined governing ratio per element.
    for r in element_ratios.values():
        r["combined"] = max((r.get(k, 0.0) for k in CHECK_KEYS if k != "combined"),
                            default=0.0)

    member_rows = []
    for mid, menv in member_env.items():
        meta = member_meta[mid]
        gov = max(menv.values())
        member_rows.append({
            "member": mid, "elements": meta["elements"],
            "section": meta["section"], "grade": meta["grade"],
            "edition": meta["edition"], "length": meta["length"],
            "combination": meta["combination"],
            "load_duration": meta["load_duration"],
            "service_class": meta["service_class"],
            "k_cy": meta["k_cy"], "k_cz": meta["k_cz"], "k_m": meta["k_m"],
            "N_Ed": meta["N_Ed"], "My_Ed": meta["My_Ed"], "Mz_Ed": meta["Mz_Ed"],
            "Vy_Ed": _pt_force(member_pts.get(mid), "shear", "vy_ed"),
            "Vz_Ed": _pt_force(member_pts.get(mid), "shear", "vz_ed"),
            "locations": {k: _where_text(pt[3])
                          for k, pt in (member_pts.get(mid) or {}).items()
                          if k != "torsion" or menv["torsion"] > 0.0},
            "util_bending": menv["bending"], "util_shear": menv["shear"],
            "util_torsion": menv["torsion"],
            "utilization": gov, "passed": gov <= 1.0,
            "section_geom": meta["section_geom"]})
        sp = member_special.get(mid)
        if sp:
            member_rows[-1]["special"] = {
                k: {"util": u, "where": member_special_at.get((mid, k), "")}
                for k, u in sp.items()}
        tgeom = next((g_[1] for b_, g_ in elem_geom.items()
                      if g_[0].id == mid), None)
        if tgeom is not None and tgeom.kind != "straight":
            member_rows[-1]["beam_kind"] = tgeom.kind
        if mid in member_rf:
            rf = member_rf[mid]
            member_rows[-1]["reinforced"] = {
                "F_t90_rf_per_m": rf["per_m"], "F_t90_rf_Ed": rf["force"]}
            if mid in member_rf_check:
                u_rf, mode, n_rf = member_rf_check[mid]
                member_rows[-1]["reinforced"].update(
                    util=u_rf, governing=mode, elements=n_rf)
            beam_warnings.append(
                f"Beam {mid}: reinforced — Eqs. 8.67 and 8.70 are not applied "
                "(§8.3.4.2(1)). The reinforcement must carry "
                f"{rf['per_m']:.1f} kN per metre of beam axis"
                + (f" ({rf['force']:.1f} kN per element, a1 given)"
                   if rf["force"] is not None else "")
                + " (Eq. 8.76); "
                + ("its resistance is verified (Eq. 8.48): "
                   f"{member_rf_check[mid][0]:.2f}."
                   if mid in member_rf_check else
                   "its resistance (Eq. 8.48) is not verified here (give the "
                   "reinforcement data to verify it)."))
        for note in member_rf_notes.get(mid, []):
            beam_warnings.append(f"Beam {mid}: reinforcement — {note}.")
        if mid in member_reinforce:
            member_rows[-1]["reinforcement_advised"] = member_reinforce[mid]
            beam_warnings.append(
                f"Beam {mid}: reinforcement against tension perpendicular to "
                f"the grain is optional but advised ({member_reinforce[mid]}). "
                "It is checked here without it (Eqs. 8.67, 8.70); with "
                "reinforcement carrying the full tension those two may be "
                "disregarded (§8.3.4.2(1)). Its design is not done here.")
        if with_profile and mid in member_profile:
            from .timber_beams import beam_outline
            tb_ = next(g_[0] for g_ in elem_geom.values() if g_[0].id == mid)
            member_rows[-1]["profile"] = [
                dict(s=s_pos, **vals)
                for s_pos, vals in sorted(member_profile[mid].items())]
            member_rows[-1]["outline"] = beam_outline(struc, tb_, tgeom)
    member_rows.sort(key=lambda x: x["member"])
    out = {"members": member_rows, "elements": element_ratios}
    if beam_warnings:
        out["warnings"] = beam_warnings
    if with_reports:
        from eurocodepy.calc_report import CalcReport
        reports: dict[str, dict] = {}
        ed_label = ("EN 1995-1-1:2025" if edition == "2025"
                    else "EN 1995-1-1:2004")
        row_by_id = {r["member"]: r for r in member_rows}
        for mid, (inp, forces) in member_report_src.items():
            if inp is None or forces is None:
                continue
            row = row_by_id.get(mid, {})
            rep = CalcReport(
                title=f"Timber member {mid} — {ed_label}",
                meta={"id": mid, "member": mid, "elements": row.get("elements", ""),
                      "section": row.get("section", ""),
                      "grade": row.get("grade", ""), "code": ed_label,
                      "combination": row.get("combination", ""),
                      "load_duration": row.get("load_duration", ""),
                      "service_class": row.get("service_class", ""),
                      "section_geom": row.get("section_geom"),
                      "utilization": row.get("utilization"), "ok": bool(row.get("passed"))})
            try:
                eurocode5_section_check(inp, forces, trace=rep)
            except TypeError:
                continue     # this EC5 edition's check is not instrumented yet
            rd = rep.to_dict()
            _add_special_sections(rd, mid, member_special.get(mid),
                                  member_special_calls, CalcReport,
                                  elem_geom)
            _sections_at_governing_points(
                rd, member_pts.get(mid), mid, CalcReport,
                eurocode5_section_check)
            reports[mid] = rd
        out["reports"] = reports
    return out
