"""Eurocode 3 design of steel bar members (EN 1993-1-1 §6.2 + §6.3.3).

Ties the xdfem2d model to the eurocodepy steel checks and produces, for every
steel-section element, the utilization ratios of:

* the **cross-section resistance** (§6.2) — N, My, Mz, Vy, Vz and T, split into
  the *bending* (N+M interaction), *shear* and *torsion* ratios;
* the **member stability** (§6.3.3) — flexural + lateral-torsional buckling,
  computed once per *physical member* (:func:`member_utils.identify_members`)
  and shared by its elements.

Both structural domains are handled:

* ``plane`` (frames): the bar carries axial ``N``, in-plane bending ``M`` and
  shear ``V``; member flexural buckling applies.
* ``plate`` (grillages): the bar's ``N`` slot is the St-Venant torsion ``T``;
  it carries out-of-plane bending ``M`` and shear ``V`` with no axial force, so
  only lateral-torsional buckling is relevant.

The in-plane actions are resolved onto the section's principal axes with the
section orientation angle θ (``My = M·cosθ``, ``Mz = M·sinθ`` …).

Model sign convention: axial force is tension-positive, so the design
compression is ``-N``. Forces per element are enveloped over the element's
critical points and over the selected combinations (conservative).

The result is ``{"members": [...member rows...], "elements": {id: ratios}}``
where each element's ``ratios`` dict has ``bending``, ``shear``, ``torsion``,
``buckling`` and ``combined`` (the governing maximum).
"""
from __future__ import annotations

import math
from types import SimpleNamespace

from .models import SectionShape, SectionType, section_type_of
from .member_utils import identify_members

# xdfem2d section shape → eurocodepy classification profile ``type``.
_SHAPE_TO_TYPE = {
    SectionShape.I: "I",
    SectionShape.RECTANGULAR_HOLLOW: "RHS",
    SectionShape.CIRCULAR_HOLLOW: "CHS",
}

# Check keys exposed to the view / table.
CHECK_KEYS = ("combined", "bending", "shear", "torsion", "buckling")


def _steel_fy(material) -> float | None:
    if material is None:
        return None
    fy = (getattr(material, "design", {}) or {}).get("fy")
    try:
        return float(fy) if fy else None
    except (TypeError, ValueError):
        return None


# xdfem2d section shape → design-report figure shape string (see
# design_report.section_figure_png).
_SHAPE_TO_FIGURE = {
    SectionShape.I: "I",
    SectionShape.RECTANGULAR_HOLLOW: "RHS",
    SectionShape.CIRCULAR_HOLLOW: "CHS",
}


def _steel_section_geom(sec) -> dict:
    """A ``section_geom`` dict (metres) for the report's cross-section drawing.

    Maps the xdfem2d section shape to the figure's shape string and carries the
    outline dimensions (``b``, ``h``, ``tw``, ``tf`` for an I / RHS; ``d`` for a
    CHS). Falls back to a plain rectangle for any other shape."""
    fig = _SHAPE_TO_FIGURE.get(sec.shape)
    if fig == "RHS" and abs(sec.b - sec.h) < 1e-9:
        fig = "SHS"
    geom = {"shape": fig or "rect", "b": sec.b, "h": sec.h,
            "tw": sec.tw, "tf": sec.tf}
    if fig == "CHS":
        geom["d"] = sec.h            # outer diameter
    return geom


def _classification_profile(sec):
    """Duck-typed profile (cm units) for eurocodepy classification, or None when
    the shape is not a standard thin-walled steel profile."""
    kind = _SHAPE_TO_TYPE.get(sec.shape)
    if kind is None:
        return None
    if kind == "RHS" and abs(sec.b - sec.h) < 1e-9:
        kind = "SHS"
    return SimpleNamespace(
        type=kind,
        h=sec.h * 100.0, b=sec.b * 100.0,
        tw=sec.tw * 100.0, tf=sec.tf * 100.0, r=0.0,
        A=sec.area * 1e4, Iy=sec.inertia_major * 1e8, Iz=sec.inertia_minor * 1e8)


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


def _member_actions(struc, member, ef_map, domain, results=None, combo_id=None,
                    sub_label=None):
    """Envelope (N_compression [kN], M [kNm]) of a member's bars for buckling."""
    n_comp = m_max = 0.0
    for bid in member.bar_ids:
        ef = ef_map.get(bid)
        if ef is None:
            continue
        dist = (_span_dist(results, combo_id, sub_label, bid)
                if results is not None else None)
        for _l, _r, n, _v, m in _critical(ef, dist):
            if domain != "plate":
                n_comp = max(n_comp, -n)
            m_max = max(m_max, abs(m))
    return n_comp, m_max


def _point_actions(n, v, m, domain, angle_deg):
    """Resolve ONE critical point's (N, V, M) onto the section axes.

    Returns a dict with n (compression, kN), t (torsion, kNm), my, mz (kNm),
    vy, vz (kN) — for this point only. Deliberately does not envelope across
    points: the §6.2 cross-section check combines N+My+Mz and Vy+Vz+T with
    genuine interaction (N reduces the plastic moment, V reduces it further,
    T reduces the shear resistance), so running the check with the worst N
    from one point mixed with the worst M from another would check a force
    state that may never actually occur — see design_steel_members's §6.2
    loop, which now runs the check separately at every critical point and
    every combination and envelopes each utilization ratio independently."""
    if domain == "plate":
        n_comp, t_abs = 0.0, abs(n)             # the N slot carries torsion
    else:
        n_comp, t_abs = -n, 0.0                 # compression positive
    m_abs, v_abs = abs(m), abs(v)
    th = math.radians(angle_deg or 0.0)
    c, s = math.cos(th), math.sin(th)
    return {"n": n_comp, "t": t_abs,
            "my": m_abs * c, "mz": m_abs * s,
            "vz": v_abs * c, "vy": v_abs * s}


def _section_input(sec, fy, cls, prof, ec3cls, n_ed, use_class4):  # noqa: ANN001, PLR0913
    """Build a eurocodepy :class:`SectionResistanceInput` from an xdfem2d
    section (m → mm) for the given class."""
    from eurocodepy.ec3.uls import cross_section as ec3cs

    kind = prof.type
    area = sec.area * 1e6
    hw = ((sec.h - 2 * sec.tf) if kind == "I" else (sec.h - 2 * sec.tw)) * 1e3
    d_my = 0.0
    weff_y = weff_z = 0.0
    area_eff = area
    if cls == 4 and use_class4:
        eff = ec3cls.effective_properties(prof, fy)
        area_eff = eff.A_eff
        weff_y = eff.W_eff_y
        weff_z = sec.wel_z * 1e9
        d_my = n_ed * eff.e_Ny / 1e3
    return ec3cs.SectionResistanceInput(
        kind=kind, section_class=cls, fy=fy, gamma_M0=1.0,
        area=area, area_eff=area_eff,
        b=sec.b * 1e3, h=sec.h * 1e3, tw=sec.tw * 1e3, tf=sec.tf * 1e3, hw=hw,
        eps=ec3cls.epsilon(fy),
        wpl_y=sec.wpl_y * 1e9, wpl_z=sec.wpl_z * 1e9,
        wel_y=sec.wel_y * 1e9, wel_z=sec.wel_z * 1e9,
        weff_y=weff_y, weff_z=weff_z,
        wt=sec.torsion_modulus * 1e9,
        av_y=sec.av_y * 1e6, av_z=sec.av_z * 1e6, d_my=d_my)


def design_steel_members(struc, results, prefs=None, combinations=None,  # noqa: C901, PLR0912, PLR0915
                         with_reports=False):
    """Run the EC3 §6.2 + §6.3.3 checks on every steel member.

    Returns ``{"members": rows, "elements": {elem_id: ratios}}`` (see the module
    docstring). Requires the eurocodepy package.

    When ``with_reports`` is True, the governing member-buckling case of each
    member is re-run once with a calculation trace and the full step-by-step
    :class:`eurocodepy.calc_report.CalcReport` is returned under an extra
    ``"reports"`` key (``{member_id: report_dict}``). The re-run only records —
    the design values are unchanged — and the reports are **not** attached to the
    persisted results, so they never bloat the ``.x2d``.
    """
    from eurocodepy.ec3 import classification as ec3cls
    from eurocodepy.ec3.uls import cross_section as ec3cs
    from eurocodepy.ec3.uls import member_buckling as ec3mem

    prefs = prefs or {}
    gm1 = float(prefs.get("gamma_M1", 1.0))
    ky_def = float(prefs.get("steel_ky", 1.0))
    kz_def = float(prefs.get("steel_kz", 1.0))
    klt_def = float(prefs.get("steel_klt", 1.0))
    rolled = bool(prefs.get("steel_ltb_rolled", True))
    use_class4 = bool(prefs.get("steel_class4_effective", True))
    # Interaction factors kij: "B" = Annex B (Method 2), "A" = Annex A (Method 1).
    # Both are allowed by the Portuguese National Annex.
    method = str(prefs.get("steel_interaction_method", "B")).upper()
    # Global (default) equivalent-moment factors and elastic constants.
    cmy = float(prefs.get("steel_cmy", 0.9))
    cmz = float(prefs.get("steel_cmz", 0.9))
    cm_lt = float(prefs.get("steel_cmlt", 0.9))
    c1 = float(prefs.get("steel_c1", 1.0))
    e_mod = float(prefs.get("steel_E", 210000.0))
    g_mod = float(prefs.get("steel_G", 81000.0))
    domain = getattr(struc, "domain", "plane")

    members, by_bar = identify_members(struc)
    # Precompute topology for the free-end (cantilever) auto K.
    from .member_utils import (_incident_bar_count, _constraint_nodes,
                                cantilever_k_for)
    _incident = _incident_bar_count(struc)
    _constrained = _constraint_nodes(struc)
    combo_cases = _combo_cases(results, combinations)
    element_ratios: dict[str, dict] = {}
    member_rows = []
    reports: dict[str, dict] = {}

    def _is_steel(sec):
        return (sec is not None
                and section_type_of(struc.materials.get(sec.material_name))
                == SectionType.STEEL
                and sec.shape in _SHAPE_TO_TYPE)

    # ── 1) member buckling (§6.3.3), once per physical member ──
    for m in members:
        sec = struc.sections.get(m.section_name)
        if not _is_steel(sec):
            continue
        fy = _steel_fy(struc.materials.get(sec.material_name))
        prof = _classification_profile(sec)
        if fy is None or prof is None:
            continue
        b0 = struc.bar_elements_by_id[m.bar_ids[0]]
        # A free-end (cantilever) member takes the auto K for Ky/Kz unless the
        # user set an explicit override; K_LT keeps the global default.
        _cant = cantilever_k_for(struc, m, prefs, _incident, _constrained)
        ky_d = _cant if _cant is not None else ky_def
        kz_d = _cant if _cant is not None else kz_def
        ky = b0.sd_ky if b0.sd_ky is not None else ky_d
        kz = b0.sd_kz if b0.sd_kz is not None else kz_d
        klt = b0.sd_klt if b0.sd_klt is not None else klt_def
        # Closed hollow sections (RHS/SHS/CHS) are not susceptible to torsional
        # deformations / LTB (EN 1993-1-1 §6.3.2.1(2)): Table B.1 (not B.2) applies.
        ltb = bool(getattr(b0, "sd_ltb", True)) and sec.shape == SectionShape.I
        lmm = m.length * 1e3
        th = math.radians(getattr(sec, "angle", 0.0) or 0.0)
        cy, cz, clt = sec.buckling_curves(rolled=rolled)

        best = None
        best_inp = None
        for combo_id, subcases in combo_cases.items():
            for sub, ef_map in subcases:
                n_comp, m_in = _member_actions(struc, m, ef_map, domain,
                                               results=results, combo_id=combo_id,
                                               sub_label=sub)
                n_ed = 0.0 if domain == "plate" else n_comp
                my, mz = m_in * math.cos(th), m_in * math.sin(th)
                cls = int(ec3cls.classify_section(prof, fy, n_ed=n_ed,
                                                  m_ed=my).section_class)
                area_mm = sec.area * 1e6
                if cls <= 2:
                    w_y, w_z, a_eff, d_my = (sec.wpl_y * 1e9, sec.wpl_z * 1e9,
                                             area_mm, 0.0)
                elif cls == 3 or not use_class4:
                    w_y, w_z, a_eff, d_my = (sec.wel_y * 1e9, sec.wel_z * 1e9,
                                             area_mm, 0.0)
                else:
                    eff = ec3cls.effective_properties(prof, fy)
                    a_eff, w_y, w_z = eff.A_eff, eff.W_eff_y, sec.wel_z * 1e9
                    d_my = n_ed * eff.e_Ny / 1e3
                inp = ec3mem.MemberInput(
                    n_ed=n_ed, my_ed=my, mz_ed=mz,
                    area=area_mm, area_eff=a_eff, w_y=w_y, w_z=w_z,
                    iy=sec.inertia_major * 1e12, iz=sec.inertia_minor * 1e12,
                    it=sec.torsion * 1e12, iw=sec.warping * 1e18,
                    lcr_y=ky * lmm, lcr_z=kz * lmm, l_lt=klt * lmm,
                    curve_y=cy, curve_z=cz, curve_lt=clt,
                    c1=c1, cmy=cmy, cmz=cmz, cm_lt=cm_lt,
                    fy=fy, e_mod=e_mod, g_mod=g_mod,
                    gamma_m1=gm1, section_class=cls,
                    susceptible_lt=ltb, rolled_lt=rolled, d_my=d_my,
                    method=method,
                    wpl_y=sec.wpl_y * 1e9, wpl_z=sec.wpl_z * 1e9,
                    wel_y=sec.wel_y * 1e9, wel_z=sec.wel_z * 1e9)
                res = ec3mem.eurocode3_member_check(inp)
                combo_name = f"{combo_id} ({sub})" if sub else combo_id
                if best is None or res.utilization > best["utilization"]:
                    best = {"member": m.id, "elements": ", ".join(m.bar_ids),
                            "section": sec.name, "profile": sec.profile_name or "custom",
                            "length": m.length, "combination": combo_name,
                            "section_class": cls, "N_Ed": n_ed, "My_Ed": my,
                            "Mz_Ed": mz, "chi_y": res.chi_y, "chi_z": res.chi_z,
                            "chi_LT": res.chi_lt, "util_6_61": res.util_6_61,
                            "util_6_62": res.util_6_62, "utilization": res.utilization,
                            "method": method,
                            "passed": res.passed}
                    best_inp = inp
        if best is not None:
            member_rows.append(best)
            for bid in m.bar_ids:
                element_ratios.setdefault(bid, {})["buckling"] = best["utilization"]
            # On-demand: re-run the governing case once with a trace to build the
            # full step-by-step report (recording only; design unchanged).
            if with_reports and best_inp is not None:
                from eurocodepy.calc_report import CalcReport
                rep = CalcReport(
                    title=f"Steel member {m.id} — EN 1993-1-1 §6.3.3",
                    meta={"id": m.id, "member": m.id, "elements": best["elements"],
                          "section": sec.name, "profile": best["profile"],
                          "combination": best["combination"],
                          "K_y": ky, "K_z": kz, "K_LT": klt,
                          "cantilever_auto_K": _cant, "gamma_M1": gm1,
                          "interaction_method": method,
                          "section_geom": _steel_section_geom(sec),
                          "utilization": best["utilization"], "ok": bool(best["passed"])})
                ec3mem.eurocode3_member_check(best_inp, trace=rep)
                reports[m.id] = rep.to_dict()

    # ── 2) cross-section resistance (§6.2), per element ──
    for elem in struc.bar_elements:
        sec = struc.sections.get(elem.section_name)
        if not _is_steel(sec):
            continue
        fy = _steel_fy(struc.materials.get(sec.material_name))
        prof = _classification_profile(sec)
        if fy is None or prof is None:
            continue
        angle = getattr(sec, "angle", 0.0)
        # `env` is the true governing ratio per CHECK TYPE, each taken from
        # the (combination, critical-point) pair that actually produces it
        # — never a mix of the worst N from one point with the worst M from
        # another (see _point_actions). Runs the full §6.2 check at every
        # point/combination and envelopes bending/shear/torsion
        # independently, same treatment as design_timber_members.
        env = {"bending": 0.0, "shear": 0.0, "torsion": 0.0}
        checked_any = False
        # On-demand only (see the buckling loop above): track the single
        # (combo, point) pair with the highest COMBINED ratio (max of the
        # three) so it can be re-run once with a trace. The persisted `env`
        # ratios stay the true independent envelope per check type —
        # unchanged by this — the report is only a representative worked
        # example at the point that is overall worst, not a synthetic mix.
        best_combined = -1.0
        best_a = best_inp = None
        best_combo_name = best_label = None
        for _combo_id, subcases in combo_cases.items():
            for _sub, ef_map in subcases:
                ef = ef_map.get(elem.id)
                if ef is None:
                    continue
                dist = _span_dist(results, _combo_id, _sub, elem.id)
                for _label, _reason, n, v, m in _critical(ef, dist):
                    checked_any = True
                    a = _point_actions(n, v, m, domain, angle)
                    cls = int(ec3cls.classify_section(
                        prof, fy, n_ed=a["n"], m_ed=a["my"]).section_class)
                    inp = _section_input(sec, fy, cls, prof, ec3cls, a["n"], use_class4)
                    res = ec3cs.eurocode3_section_check(inp, ec3cs.SectionForces(
                        n_ed=a["n"], my_ed=a["my"], mz_ed=a["mz"],
                        vy_ed=a["vy"], vz_ed=a["vz"], t_ed=a["t"]))
                    cand = {"bending": res.util_bending_axial,
                            "shear": max(res.util_shear_y, res.util_shear_z),
                            "torsion": res.util_torsion}
                    for key, val in cand.items():
                        env[key] = max(env[key], val)
                    if with_reports:
                        combined = max(cand.values())
                        if combined > best_combined:
                            best_combined = combined
                            best_a, best_inp = a, inp
                            _sub_name = f"{_combo_id} ({_sub})" if _sub else _combo_id
                            best_combo_name, best_label = _sub_name, _label
        if checked_any:
            element_ratios.setdefault(elem.id, {}).update(env)
            # On-demand: re-run the governing point once with a trace to build
            # the full step-by-step report, including the §5.5 classification
            # derivation (recording only; the env ratios above are unchanged).
            if with_reports and best_a is not None:
                from eurocodepy.calc_report import CalcReport
                rep = CalcReport(
                    title=f"Steel section {elem.id} — EN 1993-1-1 §6.2",
                    meta={"id": elem.id, "element": elem.id, "section": sec.name,
                          "combination": best_combo_name, "location": best_label,
                          "section_geom": _steel_section_geom(sec),
                          "utilization": max(env.values()), "ok": max(env.values()) <= 1.0})
                ec3cls.classify_section(prof, fy, n_ed=best_a["n"],
                                        m_ed=best_a["my"], trace=rep)
                ec3cs.eurocode3_section_check(best_inp, ec3cs.SectionForces(
                    n_ed=best_a["n"], my_ed=best_a["my"], mz_ed=best_a["mz"],
                    vy_ed=best_a["vy"], vz_ed=best_a["vz"], t_ed=best_a["t"]),
                    trace=rep)
                reports[elem.id] = rep.to_dict()

    # ── 3) combined governing ratio per element ──
    for r in element_ratios.values():
        r["combined"] = max((r.get(k, 0.0) for k in CHECK_KEYS if k != "combined"),
                            default=0.0)

    out = {"members": member_rows, "elements": element_ratios}
    if with_reports:
        out["reports"] = reports
    return out
