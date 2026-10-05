"""
Simplified Eurocode 2 (EN 1992-1-1) RC beam reinforcement design.

Covers:
  - fck lookup for common concrete classes (C16/20 … C50/60)
  - fyk lookup for common steel classes (A400, A500, A500NR)
  - Flexural reinforcement (As,top and As,bottom) for rectangular sections
  - Shear reinforcement (Asw/s) using the variable-angle truss model

All quantities in SI units unless otherwise noted:
  Forces  [kN], Moments [kNm], Areas [m²], Lengths [m]
  Stresses automatically converted to kPa = kN/m²
"""
from __future__ import annotations
import math

from .detailing import (cutoff_points as _cutoff_points,
                        merge_short_zones as _merge_short_zones)


# ---------------------------------------------------------------------------
# Material databases
# ---------------------------------------------------------------------------

# fck [MPa] keyed by class string (e.g. 'C25/30')
_FCK: dict[str, float] = {
    'C12/15': 12, 'C16/20': 16, 'C20/25': 20, 'C25/30': 25,
    'C30/37': 30, 'C35/45': 35, 'C40/50': 40, 'C45/55': 45,
    'C50/60': 50,
}

# fyk [MPa] keyed by steel class string
_FYK: dict[str, float] = {
    'A400': 400, 'A400NR': 400,
    'A500': 500, 'A500NR': 500, 'A500EL': 500,
    'B500': 500, 'S500': 500,
    'S400': 400, 'B400': 400,
}


def _fck(concrete_class: str) -> float:
    """Return fck [MPa] for a given concrete class."""
    key = concrete_class.upper().strip()
    if key in _FCK:
        return _FCK[key]
    # Try to parse numeric prefix (e.g. 'C30')
    import re
    m = re.match(r'C(\d+)', key)
    if m:
        return float(m.group(1))
    raise ValueError(f"Unknown concrete class '{concrete_class}'")


def _fyk(steel_class: str) -> float:
    """Return fyk [MPa] for a given steel class."""
    key = steel_class.upper().strip()
    if key in _FYK:
        return _FYK[key]
    import re
    m = re.match(r'[AB](\d+)', key)
    if m:
        return float(m.group(1))
    raise ValueError(f"Unknown steel class '{steel_class}'")


# ---------------------------------------------------------------------------
# Single-section design
# ---------------------------------------------------------------------------

def _flexure_note(r: dict) -> str:
    """User-facing note for a :func:`calc_asl_nm` result: only warnings (steel
    above 4 % Ac, no solution). Which method produced the result is not noise
    worth a note on every row — ``calc_asl_nm(method="auto")`` picks the
    simplified block method for flexure with little axial compression and
    strain compatibility otherwise."""
    return r.get('note', '')


class RCSection:
    """EC2 rectangular beam section for ULS design."""

    def __init__(self, b: float, h: float, cover: float,
                 fck_mpa: float, fyk_mpa: float,
                 gamma_c=1.5, gamma_s=1.15, alpha_cc=1.0):
        """
        b, h, cover  [m]
        fck_mpa, fyk_mpa  [MPa]
        """
        self.b       = b
        self.h       = h
        self.d       = h - cover        # effective depth [m]
        self.cover   = cover
        # Design strengths [kPa = kN/m²]
        self.fcd = alpha_cc * fck_mpa * 1e3 / gamma_c    # [kN/m²]
        self.fyd = fyk_mpa * 1e3 / gamma_s               # [kN/m²]
        self.fck_mpa = fck_mpa
        self.fyk_mpa = fyk_mpa
        self.gamma_c = gamma_c
        self.gamma_s = gamma_s
        self.alpha_cc = alpha_cc

    # ------------------------------------------------------------------
    # Flexural reinforcement
    # ------------------------------------------------------------------

    def flexural_reinforcement(self, Med: float, Ned: float = 0.0,
                               trace=None) -> dict:
        """
        Design flexural reinforcement for a rectangular section (combined M-N).

        Med  [kNm] — design bending moment (positive = sagging)
        Ned  [kN]  — design axial force (positive = tension, negative = compression)

        The actual design is delegated to eurocodepy
        (:func:`eurocodepy.ec2.uls.calc_asl_nm`) so the EC2 formulae live in a
        single place. eurocodepy uses the opposite axial-sign convention
        (compression positive), hence ``ned = -Ned`` below.

        Sign convention: a positive (sagging) design moment puts the tension
        steel on the **bottom** face; a negative (hogging) moment puts it on the
        **top** face. The axial force shifts the design moment to the tension
        steel (``med_s`` from eurocodepy), and it is the sign of *that* moment
        that decides which face is in tension — so the steel is always placed on
        the correct side.

        Returns dict with:
          As_bot  [m²]  — bottom-face reinforcement
          As_top  [m²]  — top-face reinforcement
          As_tension/As_comp [m²] — same areas labelled by structural role
          mu_Ed         — normalised (transferred) moment (None when the
                          section was solved by strain compatibility)
          note          — non-empty when the design needs the engineer's
                          attention (steel above 4 % Ac / no solution, or the
                          old simplified-method small-eccentricity flag)
        """
        from eurocodepy.ec2.uls import calc_asl_nm

        # method="auto": the simplified block method (checkable by hand) for
        # flexure with little axial compression, nu <= 0.05 — tension included —
        # and strain compatibility above that, where the simplified method
        # drifts (up to ~2x too much steel, or a few % too little) and, at small
        # eccentricity, breaks down. Older eurocodepy releases have no ``method``
        # and keep the simplified result (with its note, surfaced in the rows).
        try:
            r = calc_asl_nm(
                self.b, self.h, self.cover, self.cover,
                Med, -Ned,                   # eurocodepy: compression positive
                self.fck_mpa, self.fyk_mpa,
                gamma_c=self.gamma_c, gamma_s=self.gamma_s,
                alpha_cc=self.alpha_cc, trace=trace, method="auto",
            )
        except TypeError:
            r = calc_asl_nm(
                self.b, self.h, self.cover, self.cover,
                Med, -Ned,
                self.fck_mpa, self.fyk_mpa,
                gamma_c=self.gamma_c, gamma_s=self.gamma_s,
                alpha_cc=self.alpha_cc, trace=trace,
            )
        as_tension = r['As1'] * 1e-4          # cm² → m² (tension face)
        as_comp = r['As2'] * 1e-4             # compression face
        # med_s > 0 → sagging → tension at the bottom; med_s < 0 → hogging →
        # tension at the top (compression steel, if any, goes to the bottom).
        if r.get('med_s', Med) >= 0.0:
            as_bot, as_top = as_tension, as_comp
        else:
            as_bot, as_top = as_comp, as_tension
        return {
            'As_bot': as_bot,
            'As_top': as_top,
            'As_tension': as_tension,
            'As_comp': as_comp,
            'mu_Ed': r['mu'],
            'note': _flexure_note(r),
        }

    # ------------------------------------------------------------------
    # Shear reinforcement
    # ------------------------------------------------------------------

    def shear_reinforcement(self, Ved: float, Ned: float = 0.0,
                            cotg_theta: float = 1.0,
                            alpha_s: float = 90.0,
                            As_long: float = 0.0,
                            min_shear: bool = True,
                            trace=None) -> dict:
        """
        Design shear reinforcement (variable-angle truss model, EC2 §6.2.3).

        Ved        [kN]   — design shear force
        Ned        [kN]   — design axial force
        cotg_theta        — cot(θ), angle of concrete strut (1.0–2.5)
        alpha_s    [deg]  — stirrup inclination (90 = vertical)
        As_long    [m²]   — provided longitudinal reinforcement
        min_shear  [bool] — when True (default) the EC2 §9.2.2 minimum stirrups
                            (ρw,min·b) are enforced; when False the minimum is
                            NOT applied, so members where the concrete alone
                            carries the shear return Asw/s = 0 and any truss
                            value below the minimum is kept as computed
                            (e.g. box-culvert slabs designed without stirrups).

        Returns dict with:
          Asw_s  [m²/m]  — stirrup area per unit length (Asw/s)
          VRd_max [kN]   — max shear resistance (strut crushing)
          VRd_s  [kN]    — shear resistance provided by Asw_s
          mode           — 'stirrups' or 'no_shear_reinf'

        The whole EC2 §6.2 shear procedure — VRd,c, the variable-angle truss
        (Asw/s, VRd,max) and the crushing verdict — is delegated to the composite
        :func:`eurocodepy.ec2.uls.eurocode2_shear_check`, so it has a single
        source of truth (the app no longer re-implements the cot θ sweep). The
        delegated path assumes vertical stirrups (``alpha_s = 90``) and omits the
        small axial-stress bonus to VRd,c, slightly conservative in compression.
        """
        from eurocodepy.ec2.uls import ShearInput, eurocode2_shear_check

        inp = ShearInput(b=self.b, d=self.d,
                         fck=self.fck_mpa, fyk=self.fyk_mpa,
                         gamma_c=self.gamma_c, gamma_s=self.gamma_s,
                         as_long=As_long, min_shear=min_shear)
        r = eurocode2_shear_check(inp, Ved, trace=trace)
        return {
            'Asw_s': r.asw_s,
            'VRd_max': r.vrd_max,
            # VRd,c is reported as VRd_s only in the no-reinforcement branch,
            # matching the previous contract.
            'VRd_s': r.vrd_c if r.mode == 'no_shear_reinf' else None,
            'mode': r.mode,
            'crushing': r.crushing,
            'cot': r.cot,
        }

    # ------------------------------------------------------------------
    # Torsion reinforcement (grillage bars — plate domain)
    # ------------------------------------------------------------------

    def torsion_reinforcement(self, Ted: float, cotg_theta: float = 1.0,
                              trace=None) -> dict:
        """Design St-Venant torsion (EC2 §6.3) for the rectangular section,
        delegating the thin-walled closed-section formulae to
        :func:`eurocodepy.ec2.uls.calc_torsion`.

        Ted [kNm] — design torsion moment. Returns the eurocodepy dict with
        ``t_ef, A_k, u_k, TRd_max [kNm], Asw_tor_s [m²/m per leg], Asl_tor [m²],
        util`` — see that function. The mechanical cover bounds the wall
        thickness. ``trace`` is an optional eurocodepy ``CalcReport`` that the
        torsion design records its steps into (report content lives in
        eurocodepy)."""
        from eurocodepy.ec2.uls import calc_torsion
        return calc_torsion(Ted, self.b, self.h, self.fck_mpa, self.gamma_c,
                            self.fyk_mpa, self.gamma_s, cotg_theta,
                            cover=self.cover, alpha_cc=self.alpha_cc,
                            trace=trace)

    # ------------------------------------------------------------------
    # Combined shear + torsion (grillage bars — plate domain)
    # ------------------------------------------------------------------

    def shear_torsion_reinforcement(self, Ved: float, Ted: float,
                                    alpha_s: float = 90.0,
                                    As_long: float = 0.0,
                                    min_shear: bool = True,
                                    distribution_mode: str = "top_bottom",
                                    trace=None) -> dict:
        """Combined EC2 §6.2 shear + §6.3 torsion check for one section.

        Delegates *entirely* to
        :func:`eurocodepy.ec2.uls.eurocode2_shear_torsion_check` — this
        method owns none of the combination logic itself (no stirrup
        summation, no interaction formula, no cot θ bookkeeping). See
        ``dev/GRILLAGE_DESIGN.md`` §3/§5.1/§5.2: that logic used to live only
        here, duplicated and untested outside xdfem2D; it now lives in
        eurocodepy as a reusable, unit-tested composite, mirroring
        :func:`eurocodepy.ec5.uls.shear.check_shear_with_torsion` for timber.

        Ved [kN] — design shear force. Ted [kNm] — design torsion moment.
        alpha_s [deg] — accepted for signature symmetry with
        :meth:`shear_reinforcement`; unused (the eurocodepy shear+torsion
        composite assumes vertical stirrups, same as the plain shear path).

        **Behaviour change from the pre-migration ``torsion_reinforcement``
        call** (dev/GRILLAGE_DESIGN.md §2.3/§7): this no longer accepts a
        ``cotg_theta`` argument at all. The old code passed
        ``elem.rc_cotg_theta`` straight into the torsion formula while shear
        silently ignored it (``eurocode2_shear_check`` always sweeps its own
        optimum) — the two resistances combined in Eq. 6.29 could therefore
        be computed at genuinely different strut angles, which violates EC2
        §6.3.2(3). The eurocodepy composite fixes this by construction: it
        runs the shear check first and reuses whatever cot θ it picked for
        the torsion design too. So the torsion-side numbers (``TRd_max``,
        ``Asw_tor_s``, ``Asl_tor``) computed here will differ from the old
        ``rc_design.py`` output whenever ``elem.rc_cotg_theta`` did not
        happen to coincide with the shear check's own optimum — this is the
        intended fix, not a regression (see
        ``tests/tests_engine/test_rc_design.py``,
        ``TestDesignConcreteSectionsGrillage`` for the Phase-1 baseline that
        first exposed the bug, and the parity/divergence tests added for
        this migration).

        Returns a dict with (units: m²/m for stirrups, m² for longitudinal
        steel, kN/kN·m for resistances): ``Asw_shear_s``, ``Asw_tor_s``,
        ``Asw_s`` (combined total, per the existing field name consumers
        already read), ``Asl_tor`` (total), ``Asl_tor_by_face`` (new —
        ``{"top","bottom","side_left","side_right"}``), ``VRd_max``,
        ``TRd_max``, ``cot``, ``v_ratio``, ``t_ratio``, ``interaction``,
        ``crushing``, ``passed``.
        """
        from eurocodepy.ec2.uls import ShearTorsionInput, eurocode2_shear_torsion_check

        inp = ShearTorsionInput(b=self.b, h=self.h, cover=self.cover,
                                fck=self.fck_mpa, fyk=self.fyk_mpa,
                                gamma_c=self.gamma_c, gamma_s=self.gamma_s,
                                alpha_cc=self.alpha_cc, as_long=As_long,
                                min_shear=min_shear)
        r = eurocode2_shear_torsion_check(inp, Ved, Ted,
                                          distribution_mode=distribution_mode,
                                          trace=trace)
        return {
            'Asw_shear_s': r.asw_shear_s,
            'Asw_tor_s': r.asw_tor_s,
            'Asw_s': r.asw_total_s,
            'Asl_tor': r.asl_tor_total,
            'Asl_tor_by_face': r.asl_tor_by_face,
            'VRd_max': r.vrd_max,
            'TRd_max': r.trd_max,
            'cot': r.cot,
            'v_ratio': r.v_ratio,
            't_ratio': r.t_ratio,
            'interaction': r.interaction,
            'crushing': r.crushing,
            'passed': r.passed,
        }


# ---------------------------------------------------------------------------
# Critical-point helper (shared)
# ---------------------------------------------------------------------------

def _critical_points(ef, dist):
    """Return a list of (label, reason, N, V, M) design sections.

    The member is designed at its critical sections: the two ends, plus — from
    the span force distribution — the sections of maximum M (governs the bottom
    steel), minimum M (governs the top steel) and maximum |V| (governs the
    stirrups). The ``reason`` field says why each section is listed.

    Forces are returned with a consistent *internal* sign convention (tension
    positive, sagging M positive). When the span distribution is available the
    end forces are taken from its endpoints; otherwise they come from the
    element end-force vector, whose j-end stores the axial and shear negated
    (the nodal convention), so those are flipped here. Coincident sections
    (e.g. an end that is also the max-M station) are de-duplicated so the same
    (N, V, M) is not designed — and reported — twice.
    """
    def _dedup(points):
        out, seen = [], set()
        for p in points:
            key = (round(p[2], 6), round(p[3], 6), round(p[4], 6))
            if key not in seen:
                seen.add(key)
                out.append(p)
        return out

    if dist is not None:
        try:
            import numpy as np
            N = np.asarray(dist['N'], dtype=float)
            V = np.asarray(dist['V'], dtype=float)
            M = np.asarray(dist['M'], dtype=float)
            x = np.asarray(dist['x'], dtype=float)
            pts = [('i', 'i', float(N[0]), float(V[0]), float(M[0])),
                   ('j', 'j', float(N[-1]), float(V[-1]), float(M[-1]))]
            for idx, reason in ((int(np.argmax(M)), 'Asxb,max'),
                                (int(np.argmin(M)), 'Asxt,max'),
                                (int(np.argmax(np.abs(V))), 'Asws,max')):
                pts.append((f'x={x[idx]:.2f}m', reason,
                            float(N[idx]), float(V[idx]), float(M[idx])))
            return _dedup(pts)
        except Exception:
            pass
    # No distribution (e.g. envelope sub-cases): use the end-force vector and
    # correct the j-end sign convention (axial/shear are stored negated there).
    pts = [('i', 'i', ef['i'][0], ef['i'][1], ef['i'][2]),
           ('j', 'j', -ef['j'][0], -ef['j'][1], ef['j'][2])]
    return _dedup(pts)


# ---------------------------------------------------------------------------
# Concrete-section design (by section type, using Code-preferences materials)
# ---------------------------------------------------------------------------

def _effective_is_column(elem, sec) -> bool:
    """Resolve is_column for one bar: its own explicit override (True/False)
    wins; ``None`` (the default) falls back to the section's own is_column.
    Shared by design_concrete_sections (which excludes column bars — they
    are designed by design_concrete_columns instead) and
    design_concrete_columns/suggest_column_reinforcement (which select only
    column bars) — see dev/BUCKLING_COLUMN_PERSISTENCE.md ("element vs
    secção")."""
    override = getattr(elem, 'is_column', None)
    return bool(override) if override is not None else bool(
        getattr(sec, 'is_column', False))


def _is_generic_concrete_section(sec) -> bool:
    """True when *sec* is defined by A/I overrides rather than real b×h (or
    circular diameter) geometry -- i.e. ``area_override``/``inertia_override``
    are set (the same fields the steel-profile library populates for a
    catalogue profile). ``sec.shape`` is NOT the right signal here: it
    defaults to ``SectionShape.GENERIC`` for any section that never bothered
    to set it explicitly (including perfectly ordinary rectangular b×h
    sections), so gating on shape alone silently stopped designing existing
    models with no overrides at all. RC bar/column design needs *some* real
    fibre geometry to integrate against; a section known only by A and I has
    none, so it is skipped rather than treated as a b×h rectangle it may not
    be."""
    return sec.area_override is not None or sec.inertia_override is not None


def _column_section_and_placeholder_rebar(ec2mod, sec):
    """Build the ``(CrossSection, RebarLayout)`` pair for *sec*'s column
    check, dispatching on ``sec.shape`` — rectangular (default, ``b``x``h``)
    or circular (``sec.shape == SectionShape.CIRCULAR``, diameter ``b``,
    ``h`` ignored — see ``models.Section``/``section_area_inertia``, where
    the same convention already applies to the elastic properties).

    T/L sections are not supported by the underlying fibre integrator
    (``eurocodepy.ec2.uls.column._section_forces``) yet — see dev/
    RC_COLUMN_DESIGN.md Fase 5 item 7 — so they fall back to the rectangular
    ``b``x``h`` envelope, same as before circular-section support was added,
    rather than silently mis-designing a shape nobody asked to support.

    Returns ``None`` when :func:`_is_generic_concrete_section` says *sec* is
    defined by A/I overrides rather than real geometry — the caller must
    skip that member instead of building a fibre section from meaningless
    ``b``/``h`` values.

    The returned ``RebarLayout`` is only a placeholder (the caller's own
    layout/diameter search replaces it) — its only job here is to satisfy
    ``ColumnInput.rebar``, which must not be empty.

    Bar count: circular sections use ``sec.rc_n_bars`` (a single count
    evenly spaced on the circle — there is no face direction to split
    over). Rectangular sections use ``sec.rc_n_bars_y``/``sec.rc_n_bars_z``
    instead, via ``RebarLayout.symmetric_rectangular_biaxial`` — an
    independent bar count per face direction, needed to support
    skew/biaxial bending (flexão desviada) economically rather than
    always spreading bars symmetrically regardless of the governing
    M_y/M_z demand. See dev/RC_COLUMN_DESIGN.md §14.
    """
    from xdfem2d.models import SectionShape
    from eurocodepy.utils.crosssection import (
        CircularCrossSection, RectangularCrossSection,
    )

    if _is_generic_concrete_section(sec):
        return None

    if sec.shape == SectionShape.CIRCULAR:
        section = CircularCrossSection(diameter=sec.b)
        n_bars = max(int(getattr(sec, 'rc_n_bars', 4)), 3)
        rebar = ec2mod.RebarLayout.symmetric_circular(
            section, sec.rc_cover, n_bars, sec.rc_bar_phi)
    else:
        section = RectangularCrossSection(width=sec.b, height=sec.h)
        n_y = max(int(getattr(sec, 'rc_n_bars_y', 2)), 2)
        n_z = max(int(getattr(sec, 'rc_n_bars_z', 2)), 2)
        rebar = ec2mod.RebarLayout.symmetric_rectangular_biaxial(
            section, sec.rc_cover, n_y, n_z, sec.rc_bar_phi)
    return section, rebar


def _section_strengths(struc, sec):
    """Return (fck, fyk) [MPa] from the section's material design properties,
    or None when the material lacks usable concrete/steel strengths.

    The numeric ``fck``/``fyk`` are used when present; otherwise they are derived
    from the strength-class names (``class_conc`` / ``class_reinf``) via the
    standard tables, so a material that carries only its class still designs."""
    mat = struc.materials.get(sec.material_name)
    d = getattr(mat, 'design', {}) or {}

    def _val(numeric_key, class_key, table):
        try:
            v = float(d.get(numeric_key))
            if v > 0:
                return v
        except (TypeError, ValueError):
            pass
        return table.get(str(d.get(class_key, '')).strip())

    fck = _val('fck', 'class_conc', _FCK)
    fyk = _val('fyk', 'class_reinf', _FYK)
    if not fck or not fyk or fck <= 0 or fyk <= 0:
        return None
    return fck, fyk


def design_concrete_sections(struc, results: dict,
                             gamma_c: float = 1.5, gamma_s: float = 1.15,
                             alpha_cc: float = 1.0,
                             combinations=None, with_reports: bool = False) -> list:
    """Run EC2 reinforcement design for every element whose section is of
    type ``Concrete``, using a single set of material properties (taken from
    the Design ▸ Code preferences dialog).

    Unlike :func:`design_reinforcement` this does not depend on the per-element
    ``rc_design`` flag or on ``concrete_materials`` entries — it selects members
    purely by ``section.section_type == SectionType.CONCRETE`` and applies the
    supplied fck/fyk and partial safety factors to all of them.

    Every selected combination is evaluated at every critical section (both
    ends plus the span extremes of M and V) and one result row is produced for
    each, so the full design table is returned rather than only the envelope.

    Parameters
    ----------
    struc    : Structure2D
    results  : dict returned by ``Structure2D.calculate()``
    fck_mpa  : characteristic concrete strength [MPa]
    fyk_mpa  : characteristic reinforcement strength [MPa]
    gamma_c, gamma_s, alpha_cc : EC2 partial safety factors / coefficient
    combinations : iterable of combination ids to design for. ``None`` (the
                   default) uses every available combination.

    Returns
    -------
    list of dict, each with keys: element, combination, location,
        M_Ed, N_Ed, V_Ed, As_bot, As_top, Asw_s, VRd_max, mu_Ed, governing.
        ``governing`` is True for the row with the largest total steel area of
        each element (the design-driving section).
    """
    from .models import SectionType, section_type_of

    # In the plate domain a bar is a grillage member: it bends out of plane and
    # carries St-Venant torsion. Its end-force "N" slot is the torsion T, not an
    # axial force, so the flexure/shear use N = 0 and the torsion is designed to
    # EC2 §6.3 and combined with the shear (Eq. 6.29).
    is_grillage = getattr(struc, 'domain', 'plane') == 'plate'

    # Collect available combinations. Envelope-type combos (Envelope / seismic)
    # store element_forces as {'max': {...}, 'min': {...}}; these are normalised
    # into two pseudo-cases ('max'/'min') so every combination is designed.
    combo_cases = {}   # {combo_id: [(sub_label, ef_map), ...]}
    for combo_id, cd in results.get('combinations', {}).items():
        ef = cd.get('element_forces', {})
        if isinstance(ef, dict) and 'max' in ef:
            combo_cases[combo_id] = [('max', ef.get('max', {})),
                                     ('min', ef.get('min', {}))]
        else:
            combo_cases[combo_id] = [('', ef)]

    if combinations is not None:
        wanted = set(combinations)
        combo_cases = {k: v for k, v in combo_cases.items() if k in wanted}

    rows = []
    for elem in struc.bar_elements:
        sec = struc.sections.get(elem.section_name)
        if sec is None or section_type_of(
                struc.materials.get(sec.material_name)) != SectionType.CONCRETE:
            continue
        # Column bars (effective is_column) are designed by
        # design_concrete_columns instead (EC2 §5.8/§6.1 N-M interaction +
        # slenderness) — the GUI runs both from the same 'Concrete members
        # (ULS)' action and lets the model decide which applies per member.
        if _effective_is_column(elem, sec):
            continue
        if _is_generic_concrete_section(sec):
            # Defined by A/I overrides, not real b×h geometry -- nothing to
            # design a rectangular section's flexure/shear against.
            continue
        st = _section_strengths(struc, sec)
        if st is None:
            continue   # material has no fck/fyk → cannot design
        fck_mpa, fyk_mpa = st

        rcs = RCSection(sec.b, sec.h, sec.rc_cover, fck_mpa, fyk_mpa,
                        gamma_c, gamma_s, alpha_cc)
        elem_rows = []
        for combo_id, sub_cases in combo_cases.items():
            for sub_label, ef_map in sub_cases:
                ef = ef_map.get(elem.id)
                if ef is None:
                    continue
                # Span distribution only exists for plain (non-envelope) combos.
                dist = (results.get('combo_distribution', {})
                        .get(combo_id, {}).get(elem.id)) if not sub_label else None
                combo_name = f"{combo_id} ({sub_label})" if sub_label else combo_id
                for label, reason, N, V, M in _critical_points(ef, dist):
                    # Grillage: the "N" slot is torsion; there is no axial force.
                    T = N if is_grillage else 0.0
                    N_axial = 0.0 if is_grillage else N
                    flex = rcs.flexural_reinforcement(M, N_axial)

                    if is_grillage:
                        # Combined EC2 §6.2 shear + §6.3 torsion is delegated
                        # entirely to eurocodepy now (dev/GRILLAGE_DESIGN.md
                        # §3/§5.1/§5.2) -- no stirrup summation, interaction
                        # formula or cot-theta bookkeeping happens in this
                        # file anymore; see RCSection.shear_torsion_reinforcement.
                        st = rcs.shear_torsion_reinforcement(
                            V, T, alpha_s=sec.rc_alpha_s,
                            As_long=flex.get('As_tension', flex['As_bot']),
                            min_shear=getattr(sec, 'rc_shear_min', True),
                            distribution_mode=getattr(
                                sec, 'rc_torsion_distribution', 'top_bottom'))
                        row = {
                            'element': elem.id,
                            'combination': combo_name,
                            'location': label,
                            'reason': reason,
                            'M_Ed': M, 'N_Ed': N_axial, 'V_Ed': V,
                            'T_Ed': T,
                            'As_bot': flex['As_bot'] + st['Asl_tor'] / 2.0,
                            'As_top': flex['As_top'] + st['Asl_tor'] / 2.0,
                            'Asw_s': st['Asw_s'],
                            'Asw_tor_s': st['Asw_tor_s'],
                            'Asl_tor': st['Asl_tor'],
                            'Asl_tor_by_face': st['Asl_tor_by_face'],
                            'cot': st['cot'],
                            'crushing': st['crushing'],
                            'VRd_max': st['VRd_max'],
                            'TRd_max': st['TRd_max'],
                            'interaction': st['interaction'],
                            'mu_Ed': flex['mu_Ed'],
                            'note': flex['note'],
                            'kind': 'grillage',
                            'governing': False,
                        }
                    else:
                        shear = rcs.shear_reinforcement(V, N_axial,
                                                        cotg_theta=elem.rc_cotg_theta,
                                                        alpha_s=sec.rc_alpha_s,
                                                        As_long=flex.get('As_tension', flex['As_bot']),
                                                        min_shear=getattr(sec, 'rc_shear_min', True))
                        row = {
                            'element': elem.id,
                            'combination': combo_name,
                            'location': label,
                            'reason': reason,
                            'M_Ed': M, 'N_Ed': N_axial, 'V_Ed': V,
                            'As_bot': flex['As_bot'],
                            'As_top': flex['As_top'],
                            'Asw_s':  shear['Asw_s'],
                            'cot': shear.get('cot'),
                            'crushing': shear.get('crushing', False),
                            'VRd_max': shear.get('VRd_max'),
                            'mu_Ed':  flex['mu_Ed'],
                            'note': flex['note'],
                            'governing': False,
                        }
                    elem_rows.append(row)
        if elem_rows:
            gov = max(elem_rows, key=lambda d: d['As_bot'] + d['As_top'])
            gov['governing'] = True
            # On-demand: build the step-by-step report for the governing section
            # by re-running flexure + shear (+ torsion for a grillage member)
            # once with a shared trace (recording only; the design values are
            # unchanged). The report is attached to the governing row.
            if with_reports:
                from eurocodepy.calc_report import CalcReport
                rep = CalcReport(
                    title=f"Concrete section {elem.id} — EN 1992-1-1",
                    meta={"id": elem.id, "element": elem.id, "section": sec.name,
                          "combination": gov["combination"],
                          "location": gov["location"],
                          "fck": fck_mpa, "fyk": fyk_mpa,
                          "gamma_c": gamma_c, "gamma_s": gamma_s,
                          "alpha_cc": alpha_cc,
                          "section_geom": {"shape": "rect",
                                           "b": sec.b, "h": sec.h},
                          "utilization": None,
                          "ok": not gov.get("crushing", False),
                          # Reinforcement-design summary fields (see
                          # design_report.summary_rows' concrete branch) --
                          # As_bot/As_top let a General-tab summary show the
                          # governing member's actual steel demand instead of
                          # a blank "Utilization" column, which has no real
                          # meaning for a reinforcement-sizing check.
                          "As_bot": gov.get("As_bot"), "As_top": gov.get("As_top"),
                          "Asw_s": gov.get("Asw_s"), "reason": gov.get("reason", "")})
                _flex = rcs.flexural_reinforcement(
                    gov["M_Ed"], gov["N_Ed"], trace=rep)
                if is_grillage and gov.get("T_Ed") is not None:
                    rcs.shear_torsion_reinforcement(
                        gov["V_Ed"], gov["T_Ed"], alpha_s=sec.rc_alpha_s,
                        As_long=_flex.get("As_tension", _flex["As_bot"]),
                        min_shear=getattr(sec, "rc_shear_min", True),
                        distribution_mode=getattr(
                            sec, "rc_torsion_distribution", "top_bottom"),
                        trace=rep)
                else:
                    rcs.shear_reinforcement(
                        gov["V_Ed"], gov["N_Ed"],
                        cotg_theta=elem.rc_cotg_theta, alpha_s=sec.rc_alpha_s,
                        As_long=_flex.get("As_tension", _flex["As_bot"]),
                        min_shear=getattr(sec, "rc_shear_min", True), trace=rep)
                gov["report"] = rep.to_dict()
            rows.extend(elem_rows)

    return rows


# ---------------------------------------------------------------------------
# Triangle (CST / Allman) membrane reinforcement — Wood/Baumann method
# ---------------------------------------------------------------------------

def design_concrete_planes(struc, results: dict,
                              gamma_c: float = 1.5, gamma_s: float = 1.15,
                              alpha_cc: float = 1.0,
                              combinations=None, with_reports: bool = False) -> list:
    """Run EC2 membrane reinforcement design for every triangle (CST or
    Allman) whose CST section is of type ``Concrete``, delegating the
    Wood/Baumann membrane equations to
    :func:`eurocodepy.ec2.uls.calc_reinf_plane`.

    Triangles in this app are pure membrane elements — even the Allman
    element's drilling DOF is a numerical device, not real plate-bending
    stiffness (see tri_elements_allman.py) — so there is no bending moment
    to design for, only the constant in-plane forces per unit length:

        n_xx = sx·t,  n_yy = sy·t,  n_xy = txy·t   [kN/m]

    (sx/sy/txy from ``tri_stress``, already kN/m²; t = section thickness).

    ``calc_reinf_plane`` returns (asx, asy, asc, theta) as *forces* per unit
    length, not areas, and does not itself know about fck/fyk — this
    function does the rest of the EC2 bookkeeping:
      - steel area As = max(force, 0) / fyd  (fyd = fyk·1e3/γs, [kN/m²])
      - a concrete crushing check: asc vs. fcd·t (fcd = αcc·fck·1e3/γc);
        this is a simplified check (no EC2 §6.109 cracked-concrete
        reduction factor ν), at the same level of rigor as the beam shear
        crushing check elsewhere in this module.

    Per request, each direction's design force is split evenly between a
    top and a bottom mesh layer (Asx_top = Asx_bot = Asx/2, and likewise
    for y) rather than reported as a single mid-depth layer — this matches
    how orthogonal slab/wall reinforcement is normally detailed, even
    though the element itself has no bending to justify two *different*
    layers.

    Known limitation: ``tri_stress`` is only computed for combinations
    whose displacement field is a plain vector; envelope-type combinations
    (max/min sub-cases, e.g. seismic 'Envelope') do not carry a
    ``tri_stress`` entry (see solver.py) and are silently skipped here
    until that gap is closed.

    Also designs every Q4/QM6 quad of Concrete type (dev/IMPLEMENT_QUAD.md
    Phase 7): a quad is a membrane element in exactly the same sense a CST/
    Allman triangle is, and its ``tri_stress`` entry carries the same
    sx/sy/txy keys (see ``quad_elements.quad_stresses``), so the row is built
    the same way — only the section lookup (``quad_sections`` instead of
    ``tri_sections``) and the ``node_i..node_l`` geometry differ, and this
    function never touches element geometry directly (it only reads the
    already-recovered stress). DKT4/MITC4 quads are plate elements and are
    designed by :func:`design_concrete_slabs` instead, so they are excluded
    here by formulation — unlike the pre-existing triangle loop below, which
    does not filter by formulation and is left as-is (out of scope for this
    phase).

    Returns
    -------
    list of dict, each with keys: triangle, combination, formulation,
        n_xx, n_yy, n_xy [kN/m], Asx, Asy [m²/m] (the full per-direction
        area, before the top/bottom split), Asx_bot, Asx_top, Asy_bot,
        Asy_top [m²/m], Nc [kN/m] (concrete compression force per unit length,
        as returned by eurocodepy -- not a reinforcement area, despite
        living beside Asx/Asy), sigma_c [kN/m²] (= Nc/t, the concrete
        stress the crushing check actually uses), fcd [kN/m²], crushing
        (bool), theta, governing (bool — the row with the largest Asx+Asy
        for each element). The ``'triangle'`` key holds the element id
        whether the row comes from a triangle or a quad — kept unrenamed on
        purpose (dev/IMPLEMENT_QUAD.md Phase 7): ``punching._rho_l_at`` and
        ``design_report.py`` (``reports_from``/``report_base_id``) already
        read rows by that key name, and this codebase's save/report-format
        convention is additive, never rename-with-migration.
    """
    from .models import SectionType, section_type_of
    from eurocodepy.ec2.uls import MembraneInput, eurocode2_membrane_check

    combo_cases = {}
    for combo_id, cd in results.get('combinations', {}).items():
        ts = cd.get('tri_stress')
        if ts:
            combo_cases[combo_id] = ts

    if combinations is not None:
        wanted = set(combinations)
        combo_cases = {k: v for k, v in combo_cases.items() if k in wanted}

    rows: list = []
    for tri in getattr(struc, 'tri_elements', []):
        sec = struc.tri_sections.get(tri.section_name)
        if sec is None or section_type_of(
                struc.materials.get(sec.material_name)) != SectionType.CONCRETE:
            continue
        st = _section_strengths(struc, sec)
        if st is None:
            continue   # material has no fck/fyk → cannot design
        fck_mpa, fyk_mpa = st
        t = sec.thickness

        elem_rows = []
        inp_by_combo = {}
        for combo_id, tri_stress in combo_cases.items():
            d = tri_stress.get(tri.id)
            if d is None:
                continue
            n_xx = d.get('sx', 0.0) * t
            n_yy = d.get('sy', 0.0) * t
            n_xy = d.get('txy', 0.0) * t
            # The membrane design (Wood/Baumann reinforcement + concrete crushing)
            # is owned by eurocodepy's composite; the app only feeds forces and
            # materials and reads the result back into the row.
            inp = MembraneInput(
                n_xx=n_xx, n_yy=n_yy, n_xy=n_xy, fck=fck_mpa, fyk=fyk_mpa,
                thickness=t, gamma_c=gamma_c, gamma_s=gamma_s, alpha_cc=alpha_cc)
            r = eurocode2_membrane_check(inp)
            inp_by_combo[combo_id] = inp
            elem_rows.append({
                'triangle': tri.id,
                'combination': combo_id,
                'formulation': d.get('formulation', getattr(sec, 'formulation', 'CST')),
                'n_xx': n_xx, 'n_yy': n_yy, 'n_xy': n_xy,
                'Asx': r.asx, 'Asy': r.asy,
                'Asx_bot': r.asx_bot, 'Asx_top': r.asx_top,
                'Asy_bot': r.asy_bot, 'Asy_top': r.asy_top,
                'Nc': r.nc,
                'sigma_c': r.sigma_c,                    # [kN/m²]
                'fcd': r.fcd,                            # [kN/m²]
                'crushing': r.crushing,
                'theta': r.theta,
                'governing': False,
            })
        if elem_rows:
            gov = max(elem_rows, key=lambda r: r['Asx'] + r['Asy'])
            gov['governing'] = True
            # On-demand: re-run the governing combination once with a trace to
            # record the full step-by-step report (recording only; the design is
            # unchanged). Content/clauses/LaTeX come from the eurocodepy trace.
            if with_reports:
                from eurocodepy.calc_report import CalcReport
                rep = CalcReport(
                    title=f"Concrete membrane {tri.id} — EN 1992-1-1 (Wood/Baumann)",
                    meta={"id": tri.id, "triangle": tri.id, "section": sec.name,
                          "combination": gov["combination"],
                          "formulation": gov["formulation"],
                          "fck": fck_mpa, "fyk": fyk_mpa, "gamma_c": gamma_c,
                          "gamma_s": gamma_s, "alpha_cc": alpha_cc,
                          "thickness": t,
                          "utilization": None,
                          "ok": not gov.get("crushing", False),
                          # See the bar-section block's own comment above.
                          "Asx_bot": gov.get("Asx_bot"), "Asx_top": gov.get("Asx_top"),
                          "Asy_bot": gov.get("Asy_bot"), "Asy_top": gov.get("Asy_top")})
                eurocode2_membrane_check(inp_by_combo[gov["combination"]], trace=rep)
                gov["report"] = rep.to_dict()
            rows.extend(elem_rows)

    # Q4/QM6 quads (dev/IMPLEMENT_QUAD.md Phase 7): the same membrane design,
    # against quad_sections instead of tri_sections and filtered to the
    # membrane formulations — DKT4/MITC4 are plate elements, designed by
    # design_concrete_slabs instead (see the docstring above).
    for quad in getattr(struc, 'quad_elements', []):
        sec = struc.quad_sections.get(quad.section_name)
        if sec is None or getattr(sec, 'formulation', 'MITC4') not in (
                'Q4', 'QM6'):
            continue
        if section_type_of(
                struc.materials.get(sec.material_name)) != SectionType.CONCRETE:
            continue
        st = _section_strengths(struc, sec)
        if st is None:
            continue   # material has no fck/fyk → cannot design
        fck_mpa, fyk_mpa = st
        t = sec.thickness

        elem_rows = []
        inp_by_combo = {}
        for combo_id, tri_stress in combo_cases.items():
            d = tri_stress.get(quad.id)
            if d is None:
                continue
            n_xx = d.get('sx', 0.0) * t
            n_yy = d.get('sy', 0.0) * t
            n_xy = d.get('txy', 0.0) * t
            inp = MembraneInput(
                n_xx=n_xx, n_yy=n_yy, n_xy=n_xy, fck=fck_mpa, fyk=fyk_mpa,
                thickness=t, gamma_c=gamma_c, gamma_s=gamma_s, alpha_cc=alpha_cc)
            r = eurocode2_membrane_check(inp)
            inp_by_combo[combo_id] = inp
            elem_rows.append({
                'triangle': quad.id,
                'combination': combo_id,
                'formulation': d.get('formulation', getattr(sec, 'formulation', 'Q4')),
                'n_xx': n_xx, 'n_yy': n_yy, 'n_xy': n_xy,
                'Asx': r.asx, 'Asy': r.asy,
                'Asx_bot': r.asx_bot, 'Asx_top': r.asx_top,
                'Asy_bot': r.asy_bot, 'Asy_top': r.asy_top,
                'Nc': r.nc,
                'sigma_c': r.sigma_c,
                'fcd': r.fcd,
                'crushing': r.crushing,
                'theta': r.theta,
                'governing': False,
            })
        if elem_rows:
            gov = max(elem_rows, key=lambda r: r['Asx'] + r['Asy'])
            gov['governing'] = True
            if with_reports:
                from eurocodepy.calc_report import CalcReport
                rep = CalcReport(
                    title=f"Concrete membrane {quad.id} — EN 1992-1-1 (Wood/Baumann)",
                    meta={"id": quad.id, "triangle": quad.id, "section": sec.name,
                          "combination": gov["combination"],
                          "formulation": gov["formulation"],
                          "fck": fck_mpa, "fyk": fyk_mpa, "gamma_c": gamma_c,
                          "gamma_s": gamma_s, "alpha_cc": alpha_cc,
                          "thickness": t,
                          "utilization": None,
                          "ok": not gov.get("crushing", False),
                          # See the bar-section block's own comment above.
                          "Asx_bot": gov.get("Asx_bot"), "Asx_top": gov.get("Asx_top"),
                          "Asy_bot": gov.get("Asy_bot"), "Asy_top": gov.get("Asy_top")})
                eurocode2_membrane_check(inp_by_combo[gov["combination"]], trace=rep)
                gov["report"] = rep.to_dict()
            rows.extend(elem_rows)

    return rows


def _fctm(fck_mpa: float) -> float:
    """Mean tensile strength fctm [MPa] (EC2 Table 3.1)."""
    if fck_mpa <= 50.0:
        return 0.30 * fck_mpa ** (2.0 / 3.0)
    return 2.12 * math.log(1.0 + (fck_mpa + 8.0) / 10.0)


# ---------------------------------------------------------------------------
# Slab (plate DKT / MITC3) flexural reinforcement — Wood-Armer + EC2 flexure
# ---------------------------------------------------------------------------

def design_concrete_slabs(struc, results: dict,
                          gamma_c: float = 1.5, gamma_s: float = 1.15,
                          alpha_cc: float = 1.0,
                          combinations=None, with_reports: bool = False) -> list:
    """EC2 flexural reinforcement of every plate (DKT / MITC3) triangle whose
    section is of Concrete type, from the **Wood-Armer design moments** already
    recovered per element (``mx_bot, my_bot, mx_top, my_top`` [kNm/m]).

    Each of the four orthogonal reinforcements — bottom/top × x/y — is sized as a
    1 m strip with :func:`eurocodepy.ec2.uls.calc_asl`, using its own effective
    depth ``d = t − cover`` from the section's per-face/direction covers
    (:meth:`TriSection.resolved_covers`; ``cover`` is the mechanical cover to the
    bar, matching the beam design's ``d = h − cover``). The EC2 §9.3.1.1 minimum
    slab reinforcement is applied as a floor to each tensioned direction.

    Envelope combinations do not carry per-element moments (no ``tri_stress``)
    and are silently skipped, as in the membrane design.

    Limitation: the design uses the **per-element (centroid) moments**, which on
    a triangular mesh carry a mesh-dependent twisting component ``mxy``. Because
    Wood-Armer adds ``|mxy|`` to both faces, this shows up as some top steel even
    in a purely sagging slab. Smoothing the moments to the nodes before designing
    would remove most of that noise and is a worthwhile future refinement; the
    per-element values are used here for consistency with the membrane design and
    because they are the (conservative) raw finite-element result.

    Also designs every DKT4/MITC4 quad of Concrete type
    (dev/IMPLEMENT_QUAD.md Phase 7): the quad moment recovery
    (``quad_elements_dkt4.dkt4_moment_entry``/
    ``quad_elements_mitc4.mitc4_moment_entry``) already reports the same
    Wood-Armer ``mx_bot/my_bot/mx_top/my_top`` keys the triangle recovery
    does, via the shared ``plate_common.plate_moment_result``, so the row is
    built identically — only the section lookup (``quad_sections``) and
    covers (``QuadSection.resolved_covers``) differ. Q4/QM6 quads are
    membrane elements and are designed by :func:`design_concrete_planes`
    instead, so they are excluded here by formulation, mirroring the
    pre-existing triangle DKT/MITC3 filter below.

    Returns a list of dict, one row per (element, combination), with the
    moments and the four areas ``Asx_bot, Asx_top, Asy_bot, Asy_top`` [m²/m]
    (×1e4 → cm²/m in the report). ``kind='slab'`` tags the row; ``governing`` is
    the row with the largest total area for each element. The ``'triangle'``
    key holds the element id whether the row comes from a triangle or a quad
    — see :func:`design_concrete_planes`'s docstring for why this is not
    renamed.
    """
    from .models import SectionType, section_type_of
    from eurocodepy.ec2.uls import SlabInput, eurocode2_slab_check

    combo_cases = {}
    for combo_id, cd in results.get('combinations', {}).items():
        ts = cd.get('tri_stress')
        if ts:
            combo_cases[combo_id] = ts
    if combinations is not None:
        wanted = set(combinations)
        combo_cases = {k: v for k, v in combo_cases.items() if k in wanted}

    rows: list = []
    for tri in getattr(struc, 'tri_elements', []):
        sec = struc.tri_sections.get(tri.section_name)
        if sec is None or getattr(sec, 'formulation', 'CST') not in (
                'DKT', 'MITC3'):
            continue
        if section_type_of(
                struc.materials.get(sec.material_name)) != SectionType.CONCRETE:
            continue
        st = _section_strengths(struc, sec)
        if st is None:
            continue
        fck_mpa, fyk_mpa = st
        t = sec.thickness
        covers = sec.resolved_covers()

        def _slab_input(mxb, myb, mxt, myt):
            return SlabInput(
                mx_bot=mxb, my_bot=myb, mx_top=mxt, my_top=myt,
                fck=fck_mpa, fyk=fyk_mpa, thickness=t,
                cover_bot_x=covers['bot_x'], cover_bot_y=covers['bot_y'],
                cover_top_x=covers['top_x'], cover_top_y=covers['top_y'],
                gamma_c=gamma_c, gamma_s=gamma_s, alpha_cc=alpha_cc)

        elem_rows = []
        inp_by_combo = {}
        for combo_id, tri_stress in combo_cases.items():
            d = tri_stress.get(tri.id)
            if d is None:
                continue
            mxb = d.get('mx_bot', 0.0); myb = d.get('my_bot', 0.0)
            mxt = d.get('mx_top', 0.0); myt = d.get('my_top', 0.0)
            # The slab flexural design (four faces, EC2 §6.1 + §9.3.1.1 minimum)
            # is owned by eurocodepy's composite; the app feeds the Wood-Armer
            # design moments and geometry and reads the areas back.
            inp = _slab_input(mxb, myb, mxt, myt)
            r = eurocode2_slab_check(inp)
            inp_by_combo[combo_id] = inp
            elem_rows.append({
                'triangle': tri.id,
                'combination': combo_id,
                'formulation': d.get('formulation',
                                     getattr(sec, 'formulation', 'DKT')),
                'kind': 'slab',
                'mx': d.get('mx', 0.0), 'my': d.get('my', 0.0),
                'mxy': d.get('mxy', 0.0),
                'mx_bot': mxb, 'my_bot': myb, 'mx_top': mxt, 'my_top': myt,
                'Asx_bot': r.asx_bot, 'Asy_bot': r.asy_bot,
                'Asx_top': r.asx_top, 'Asy_top': r.asy_top,
                'governing': False,
            })
        if elem_rows:
            gov = max(elem_rows, key=lambda r: (r['Asx_bot'] + r['Asx_top']
                                                + r['Asy_bot'] + r['Asy_top']))
            gov['governing'] = True
            # On-demand: re-run the governing combination once with a trace to
            # record the report (recording only; the design is unchanged).
            if with_reports:
                from eurocodepy.calc_report import CalcReport
                rep = CalcReport(
                    title=f"Concrete slab {tri.id} — EN 1992-1-1 (Wood-Armer)",
                    meta={"id": tri.id, "triangle": tri.id, "section": sec.name,
                          "combination": gov["combination"],
                          "formulation": gov["formulation"],
                          "fck": fck_mpa, "fyk": fyk_mpa, "gamma_c": gamma_c,
                          "gamma_s": gamma_s, "alpha_cc": alpha_cc,
                          "thickness": t,
                          "Asx_bot": gov.get("Asx_bot"), "Asx_top": gov.get("Asx_top"),
                          "Asy_bot": gov.get("Asy_bot"), "Asy_top": gov.get("Asy_top"),
                          "utilization": None, "ok": True})
                eurocode2_slab_check(inp_by_combo[gov["combination"]], trace=rep)
                gov["report"] = rep.to_dict()
            rows.extend(elem_rows)

    # DKT4/MITC4 quads (dev/IMPLEMENT_QUAD.md Phase 7): the same flexural
    # design, against quad_sections instead of tri_sections and filtered to
    # the plate formulations — Q4/QM6 are membrane elements, designed by
    # design_concrete_planes instead (see the docstring above).
    for quad in getattr(struc, 'quad_elements', []):
        sec = struc.quad_sections.get(quad.section_name)
        if sec is None or getattr(sec, 'formulation', 'MITC4') not in (
                'DKT4', 'MITC4'):
            continue
        if section_type_of(
                struc.materials.get(sec.material_name)) != SectionType.CONCRETE:
            continue
        st = _section_strengths(struc, sec)
        if st is None:
            continue
        fck_mpa, fyk_mpa = st
        t = sec.thickness
        covers = sec.resolved_covers()

        def _slab_input_q(mxb, myb, mxt, myt):
            return SlabInput(
                mx_bot=mxb, my_bot=myb, mx_top=mxt, my_top=myt,
                fck=fck_mpa, fyk=fyk_mpa, thickness=t,
                cover_bot_x=covers['bot_x'], cover_bot_y=covers['bot_y'],
                cover_top_x=covers['top_x'], cover_top_y=covers['top_y'],
                gamma_c=gamma_c, gamma_s=gamma_s, alpha_cc=alpha_cc)

        elem_rows = []
        inp_by_combo = {}
        for combo_id, tri_stress in combo_cases.items():
            d = tri_stress.get(quad.id)
            if d is None:
                continue
            mxb = d.get('mx_bot', 0.0); myb = d.get('my_bot', 0.0)
            mxt = d.get('mx_top', 0.0); myt = d.get('my_top', 0.0)
            inp = _slab_input_q(mxb, myb, mxt, myt)
            r = eurocode2_slab_check(inp)
            inp_by_combo[combo_id] = inp
            elem_rows.append({
                'triangle': quad.id,
                'combination': combo_id,
                'formulation': d.get('formulation',
                                     getattr(sec, 'formulation', 'MITC4')),
                'kind': 'slab',
                'mx': d.get('mx', 0.0), 'my': d.get('my', 0.0),
                'mxy': d.get('mxy', 0.0),
                'mx_bot': mxb, 'my_bot': myb, 'mx_top': mxt, 'my_top': myt,
                'Asx_bot': r.asx_bot, 'Asy_bot': r.asy_bot,
                'Asx_top': r.asx_top, 'Asy_top': r.asy_top,
                'governing': False,
            })
        if elem_rows:
            gov = max(elem_rows, key=lambda r: (r['Asx_bot'] + r['Asx_top']
                                                + r['Asy_bot'] + r['Asy_top']))
            gov['governing'] = True
            if with_reports:
                from eurocodepy.calc_report import CalcReport
                rep = CalcReport(
                    title=f"Concrete slab {quad.id} — EN 1992-1-1 (Wood-Armer)",
                    meta={"id": quad.id, "triangle": quad.id, "section": sec.name,
                          "combination": gov["combination"],
                          "formulation": gov["formulation"],
                          "fck": fck_mpa, "fyk": fyk_mpa, "gamma_c": gamma_c,
                          "gamma_s": gamma_s, "alpha_cc": alpha_cc,
                          "thickness": t,
                          "Asx_bot": gov.get("Asx_bot"), "Asx_top": gov.get("Asx_top"),
                          "Asy_bot": gov.get("Asy_bot"), "Asy_top": gov.get("Asy_top"),
                          "utilization": None, "ok": True})
                eurocode2_slab_check(inp_by_combo[gov["combination"]], trace=rep)
                gov["report"] = rep.to_dict()
            rows.extend(elem_rows)

    return rows


def store_tri_reinforcement(results: dict, rows, *,
                            gamma_c: float = 1.5, gamma_s: float = 1.15,
                            alpha_cc: float = 1.0,
                            prefs: dict | None = None):
    """Embed an already-computed triangle (shell) reinforcement design into
    ``results['tri_reinforcement']`` — the triangle analogue of
    :func:`store_reinforcement`. Passing empty ``rows`` removes the key.
    Returns the stored payload dict (or ``None`` when nothing was stored)."""
    if not rows:
        results.pop('tri_reinforcement', None)
        return None
    payload_prefs = {'gamma_c': gamma_c, 'gamma_s': gamma_s, 'alpha_cc': alpha_cc}
    if prefs:
        payload_prefs.update(prefs)
    payload = {'rows': rows, 'prefs': payload_prefs}
    results['tri_reinforcement'] = payload
    return payload


def design_tri_and_store(struc, results: dict, *,
                         gamma_c: float = 1.5, gamma_s: float = 1.15,
                         alpha_cc: float = 1.0, combinations=None,
                         prefs: dict | None = None, with_reports: bool = False):
    """Design every Concrete triangle **and** store the result in
    ``results['tri_reinforcement']`` in one step — the triangle analogue of
    :func:`design_and_store`. Dispatches by domain: a plate model designs its
    slabs for bending (Wood-Armer + EC2 flexure); a plane model designs its
    membranes (Wood/Baumann). Returns the stored payload dict (or ``None`` if
    nothing was designed).

    ``with_reports`` mirrors the bar-section design's own flag (see
    :func:`design_concrete_sections`): the governing row of each triangle gets
    a re-recorded :class:`eurocodepy.calc_report.CalcReport` under its
    ``"report"`` key. The caller is responsible for not persisting those
    reports into ``results['tri_reinforcement']`` (they are *not* meant to
    survive a .x2d save/reload, same as the steel/timber member reports) —
    strip ``row["report"]`` from the returned payload's rows after reading it,
    if it was requested only for a one-off cache/report build."""
    if getattr(struc, 'domain', 'plane') == 'plate':
        rows = design_concrete_slabs(
            struc, results, gamma_c=gamma_c, gamma_s=gamma_s,
            alpha_cc=alpha_cc, combinations=combinations,
            with_reports=with_reports)
    else:
        rows = design_concrete_planes(
            struc, results, gamma_c=gamma_c, gamma_s=gamma_s,
            alpha_cc=alpha_cc, combinations=combinations,
            with_reports=with_reports)
    return store_tri_reinforcement(results, rows, gamma_c=gamma_c,
                                   gamma_s=gamma_s, alpha_cc=alpha_cc,
                                   prefs=prefs)


# ---------------------------------------------------------------------------
# Serviceability — crack-width verification (delegated to eurocodepy)
# ---------------------------------------------------------------------------

def _as_min(b, h, cover, fck_mpa, fyk_mpa):
    """EC2 §9.2.1.1 minimum tension reinforcement [m²]."""
    fctm = 0.30 * fck_mpa ** (2.0 / 3.0)
    d_eff = h - cover
    return max(0.26 * fctm / fyk_mpa * b * d_eff, 0.0013 * b * d_eff)


def crack_width_sections(struc, results: dict, rc_rows: list,
                         combinations=None, phi_mm: float = 16.0,
                         k_t: float = 0.4, wk_limit: float | None = None,
                         rc_envelope: dict | None = None) -> list:
    """EC2 crack-width (SLS) verification, EN 1992-1-1 §7.3.4.

    The crack opening is computed with ``eurocodepy.ec2.sls.crack_opening``.
    The reinforcement is taken **station by station** from ``rc_envelope`` (with
    an As,min floor), exactly like :func:`crack_width_envelope`, so the table
    and the diagram are consistent. For each element × combination the two ends
    and the governing (maximum-wk) station are reported; the element's governing
    row therefore matches the peak of the diagram. When no envelope is supplied
    the governing As from ``rc_rows`` is used as a constant fallback.

    Returns
    -------
    list of dict with keys: element, combination, location, M_Ed, N_Ed,
        cracked, wk [mm], wk_limit, ok, governing. Combined bending + axial is
        accounted for (axial negated to eurocodepy's compression-positive
        convention); k_t comes from the Code preferences.
    """
    from .models import SectionType, section_type_of
    import numpy as np
    from eurocodepy.ec2.sls import crack_opening, is_cracked
    from eurocodepy.ec2.materials import Concrete

    prov: dict[str, dict] = {}
    for r in rc_rows or []:
        eid = r['element']
        cur = prov.get(eid)
        if (r.get('governing') or cur is None
                or (r['As_bot'] + r['As_top']) > (cur['As_bot'] + cur['As_top'])):
            prov[eid] = r

    rc_envelope = rc_envelope or {}
    combo_dist = results.get('combo_distribution', {})
    combo_ids = list(combo_dist.keys())
    if combinations is not None:
        wanted = set(combinations)
        combo_ids = [c for c in combo_ids if c in wanted]

    rows = []
    for elem in struc.bar_elements:
        sec = struc.sections.get(elem.section_name)
        if sec is None or section_type_of(
                struc.materials.get(sec.material_name)) != SectionType.CONCRETE:
            continue
        st = _section_strengths(struc, sec)
        if st is None:
            continue
        pr = prov.get(elem.id)
        env = rc_envelope.get(elem.id)
        if pr is None and env is None:
            continue
        fck_mpa, fyk_mpa = st
        b, h, cover = sec.b, sec.h, sec.rc_cover
        phi = getattr(sec, 'rc_bar_phi', phi_mm) / 1000.0
        conc = Concrete.from_fck(fck_mpa)
        As_min = _as_min(b, h, cover, fck_mpa, fyk_mpa)
        env_x = np.asarray(env['x'], float) if env is not None else None
        env_pair = ((np.asarray(env['As_bot'], float),
                     np.asarray(env['As_top'], float)) if env is not None else None)

        # A tension face is "minimum-reinforced" when its required steel is at
        # (≈) As,min; a small relative tolerance absorbs the rounding difference
        # between this As,min and the one embedded in the ULS envelope.
        as_min_thr = As_min * 1.03

        def _raw_as(k, use_env):
            if use_env:
                return float(env_pair[0][k]), float(env_pair[1][k])
            return (float(pr['As_bot']) if pr else 0.0,
                    float(pr['As_top']) if pr else 0.0)

        def _wk_at(M, N, asb_raw, ast_raw):
            N_ec = -N
            # Tension face follows the moment sign.
            t_raw = asb_raw if M >= 0 else ast_raw
            # When the tension face only carries minimum reinforcement, crack
            # control is deemed satisfied by the As,min detailing → wk = 0.
            if t_raw <= as_min_thr:
                return 0.0, False, True
            if M >= 0:
                At, Ac = max(asb_raw, As_min), max(ast_raw, As_min)
            else:
                At, Ac = max(ast_raw, As_min), max(asb_raw, As_min)
            ds, dsc = h - cover, cover
            Ata = np.array([At]); dsa = np.array([ds])
            Aca = np.array([Ac]); dsca = np.array([dsc])
            Ma = np.array([abs(M)]); Na = np.array([N_ec])
            cracked = bool(is_cracked(b, h, Ata, dsa, Aca, dsca,
                                      conc, "A500NR", Ma, Na))
            wk = 0.0 if not cracked else float(np.max(crack_opening(
                b, h, phi, Ata, dsa, Aca, dsca, conc, "A500NR", Ma, Na, k_t)))
            return wk, cracked, False

        elem_rows = []
        for cid in combo_ids:
            d = combo_dist.get(cid, {}).get(elem.id)
            if d is None:
                continue
            xs = np.asarray(d['x'], float)
            Nd = np.asarray(d['N'], float)
            Md = np.asarray(d['M'], float)
            use_env = (env_x is not None and len(env_x) == len(xs))
            n = len(xs)
            if n == 0:
                continue
            # Find the governing (max-wk) station for this combination.
            wk_all = np.empty(n)
            for k in range(n):
                asb, ast = _raw_as(k, use_env)
                wk_all[k], _, _ = _wk_at(float(Md[k]), float(Nd[k]), asb, ast)
            kg = int(np.argmax(wk_all))
            # Report the two ends plus the governing station (de-duplicated).
            seen = set()
            for k, label in ((0, 'i'), (n - 1, 'j'), (kg, f'x={xs[kg]:.2f}m')):
                if k in seen:
                    continue
                seen.add(k)
                asb, ast = _raw_as(k, use_env)
                wk, cracked, asmin = _wk_at(float(Md[k]), float(Nd[k]), asb, ast)
                elem_rows.append({
                    'element': elem.id, 'combination': cid, 'location': label,
                    'M_Ed': float(Md[k]), 'N_Ed': float(Nd[k]),
                    'wk': wk, 'cracked': cracked, 'asmin': asmin,
                    'wk_limit': wk_limit,
                    'ok': None if asmin else (
                        (wk <= wk_limit) if wk_limit is not None else None),
                    'governing': False,
                })
        if elem_rows:
            gov = max(elem_rows, key=lambda d: d['wk'])
            gov['governing'] = True
            rows.extend(elem_rows)

    return rows


def crack_width_envelope(struc, results: dict, rc_rows: list,
                         combinations=None, phi_mm: float = 16.0,
                         k_t: float = 0.4, rc_envelope: dict | None = None) -> dict:
    """Crack-width (wk) envelope along every concrete element.

    For each station of the span force distribution the crack opening is
    evaluated for every selected SLS combination, keeping the per-station
    maximum — i.e. the envelope of wk along the member, suitable for a diagram.

    The reinforcement is taken **station by station** from ``rc_envelope`` (the
    required-As envelope, which shares the same stations as the force
    distribution) when available, so the steel varies continuously along the
    member. This avoids the spurious wk jumps at internal nodes of a member
    chain that arise from using a single (governing) As per element. When no
    envelope is supplied it falls back to the governing As from ``rc_rows``.

    Returns
    -------
    dict {element_id: {'x': [m], 'wk': [mm], 'side': [+1/-1]}}
    where ``side`` is the tension face that governs wk at each station: +1 for a
    sagging moment (tension on the bottom / right-of-travel face) and -1 for a
    hogging moment (tension on the top face). It lets a diagram be drawn on the
    cracked side, and may flip along a member near a point of inflection.
    """
    from .models import SectionType, section_type_of
    import numpy as np
    from eurocodepy.ec2.sls import crack_opening
    from eurocodepy.ec2.materials import Concrete

    prov: dict[str, dict] = {}
    for r in rc_rows or []:
        eid = r['element']
        cur = prov.get(eid)
        if (r.get('governing') or cur is None
                or (r['As_bot'] + r['As_top']) > (cur['As_bot'] + cur['As_top'])):
            prov[eid] = r

    rc_envelope = rc_envelope or {}
    combo_dist = results.get('combo_distribution', {})
    combo_ids = list(combo_dist.keys())
    if combinations is not None:
        wanted = set(combinations)
        combo_ids = [c for c in combo_ids if c in wanted]

    out = {}
    for elem in struc.bar_elements:
        sec = struc.sections.get(elem.section_name)
        if sec is None or section_type_of(
                struc.materials.get(sec.material_name)) != SectionType.CONCRETE:
            continue
        st = _section_strengths(struc, sec)
        if st is None:
            continue
        pr = prov.get(elem.id)
        env = rc_envelope.get(elem.id)
        if pr is None and env is None:
            continue
        fck_mpa, fyk_mpa = st
        b, h, cover = sec.b, sec.h, sec.rc_cover
        phi = getattr(sec, 'rc_bar_phi', phi_mm) / 1000.0
        conc = Concrete.from_fck(fck_mpa)
        # EC2 minimum tension steel (§9.2.1.1). Real detailing provides at least
        # this on each face, so floor the per-station As here — otherwise a face
        # with zero *required* steel (singly-reinforced region) but tension under
        # some combination gives ρ≈0 and an exploding, meaningless wk.
        As_min = _as_min(b, h, cover, fck_mpa, fyk_mpa)
        env_x = np.asarray(env['x'], float) if env is not None else None
        env_asb = np.asarray(env['As_bot'], float) if env is not None else None
        env_ast = np.asarray(env['As_top'], float) if env is not None else None

        xs_ref = None
        wk_env = None
        side_env = None
        for cid in combo_ids:
            d = combo_dist.get(cid, {}).get(elem.id)
            if d is None:
                continue
            xs = np.asarray(d['x'], dtype=float)
            Nd = np.asarray(d['N'], dtype=float)
            Md = np.asarray(d['M'], dtype=float)
            if xs_ref is None:
                xs_ref = xs
                wk_env = np.zeros_like(xs)
                side_env = np.ones_like(xs)
            if len(xs) != len(xs_ref):
                continue
            use_env = (env_x is not None and len(env_x) == len(xs))
            as_min_thr = As_min * 1.03
            for k in range(len(xs)):
                M = float(Md[k]); N_ec = -float(Nd[k])
                if use_env:
                    asb_raw, ast_raw = float(env_asb[k]), float(env_ast[k])
                elif pr:
                    asb_raw, ast_raw = float(pr['As_bot']), float(pr['As_top'])
                else:
                    asb_raw = ast_raw = 0.0
                # Tension face only at As,min → crack controlled by detailing.
                t_raw = asb_raw if M >= 0 else ast_raw
                if t_raw <= as_min_thr:
                    continue
                As_bot = max(asb_raw, As_min); As_top = max(ast_raw, As_min)
                if M >= 0:
                    As_t, ds, As_c, dsc = As_bot, h - cover, As_top, cover
                else:
                    As_t, ds, As_c, dsc = As_top, h - cover, As_bot, cover
                At = np.array([As_t]); dsa = np.array([ds])
                Ac = np.array([As_c]); dsca = np.array([dsc])
                Ma = np.array([abs(M)]); Na = np.array([N_ec])
                # No hard cracked/uncracked gate here: crack_opening tapers
                # smoothly to 0 as M decreases (eps_sm → 0.6·σs/Es → 0), giving
                # a continuous diagram instead of a cliff at the cracking onset.
                wk = float(np.max(crack_opening(
                    b, h, phi, At, dsa, Ac, dsca,
                    conc, "A500NR", Ma, Na, k_t)))
                if wk > wk_env[k]:
                    wk_env[k] = wk
                    side_env[k] = 1.0 if M >= 0 else -1.0

        if xs_ref is not None:
            out[elem.id] = {'x': xs_ref, 'wk': wk_env, 'side': side_env}
    return out


def reinforcement_envelope(struc, results: dict,
                           gamma_c: float = 1.5, gamma_s: float = 1.15,
                           alpha_cc: float = 1.0,
                           combinations=None) -> dict:
    """Required-reinforcement envelopes along every Concrete element.

    For each station of the span force diagrams (``combo_distribution``) the
    flexural (As_bot / As_top) and shear (Asw/s) reinforcement is designed for
    every selected combination, and the per-station maximum is kept — i.e. the
    envelope of required steel along the member.

    Returns
    -------
    dict {element_id: {'x':   [m] along the element,
                       'As_bot': [m²] tension-face envelope,
                       'As_top': [m²] compression-face envelope,
                       'Asw_s':  [m²/m] stirrup envelope}}
    Only elements with a Concrete section and available distributions appear.
    """
    from .models import SectionType, section_type_of
    import numpy as np

    combo_dist = results.get('combo_distribution', {})
    combo_ids = list(combo_dist.keys())
    if combinations is not None:
        wanted = set(combinations)
        combo_ids = [c for c in combo_ids if c in wanted]

    out = {}
    for elem in struc.bar_elements:
        sec = struc.sections.get(elem.section_name)
        if sec is None or section_type_of(
                struc.materials.get(sec.material_name)) != SectionType.CONCRETE:
            continue
        st = _section_strengths(struc, sec)
        if st is None:
            continue
        fck_mpa, fyk_mpa = st
        rcs = RCSection(sec.b, sec.h, sec.rc_cover, fck_mpa, fyk_mpa,
                        gamma_c, gamma_s, alpha_cc)

        xs_ref = None
        as_bot = as_top = asw = crush = None
        for cid in combo_ids:
            d = combo_dist.get(cid, {}).get(elem.id)
            if d is None:
                continue
            xs = np.asarray(d['x'], dtype=float)
            N = np.asarray(d['N'], dtype=float)
            V = np.asarray(d['V'], dtype=float)
            M = np.asarray(d['M'], dtype=float)
            if xs_ref is None:
                xs_ref = xs
                as_bot = np.zeros_like(xs)
                as_top = np.zeros_like(xs)
                asw = np.zeros_like(xs)
                crush = np.zeros_like(xs, dtype=bool)
            if len(xs) != len(xs_ref):
                continue
            for k in range(len(xs)):
                flex = rcs.flexural_reinforcement(float(M[k]), float(N[k]))
                shear = rcs.shear_reinforcement(float(V[k]), float(N[k]),
                                                cotg_theta=elem.rc_cotg_theta,
                                                alpha_s=sec.rc_alpha_s,
                                                As_long=flex.get('As_tension', flex['As_bot']),
                                                min_shear=getattr(sec, 'rc_shear_min', True))
                if flex['As_bot'] > as_bot[k]:
                    as_bot[k] = flex['As_bot']
                if flex['As_top'] > as_top[k]:
                    as_top[k] = flex['As_top']
                if shear['Asw_s'] > asw[k]:
                    asw[k] = shear['Asw_s']
                if shear.get('crushing'):
                    crush[k] = True

        if xs_ref is not None:
            out[elem.id] = {'x': xs_ref, 'As_bot': as_bot, 'As_top': as_top,
                            'Asw_s': asw, 'crushing': crush}
    return out


# ---------------------------------------------------------------------------
# Structure-level RC design
# ---------------------------------------------------------------------------

def design_reinforcement(struc, results: dict) -> dict:
    """
    Run EC2 reinforcement design for all elements flagged with rc_design=True.

    Iterates over all load combinations and computes As,bot / As,top / Asw/s
    at both element ends. Returns envelope (critical) values per element.

    Parameters
    ----------
    struc   : Structure2D
    results : dict returned by Structure2D.calculate()

    Returns
    -------
    dict  {element_id: {
              'combination': combo_id,
              'end': 'i' or 'j',
              'M_Ed': float [kNm],
              'N_Ed': float [kN],
              'V_Ed': float [kN],
              'As_bot': float [m²],
              'As_top': float [m²],
              'Asw_s':  float [m²/m],
              'VRd_max': float [kN],
              'mu_Ed':   float,
           }, ...}
    """
    rc_results = {}

    # Build combined set of cases + combinations to check
    all_cases = list(results['element_forces'].keys())
    combo_cases = {}
    for combo_id, cd in results['combinations'].items():
        ef = cd.get('element_forces', {})
        # Skip envelope (has sub-dict max/min)
        if isinstance(ef, dict) and 'max' in ef:
            continue
        combo_cases[combo_id] = ef

    def _critical_points(ef, dist):
        """
        Return a list of (label, N, V, M) sections to design.

        Always includes the two element ends. When a span force distribution is
        available it also adds the sections of maximum |M| (governs flexure) and
        maximum |V| (governs shear). This is essential for members with end
        hinges (or pinned ends), whose critical moment occurs within the span,
        not at the ends.
        """
        pts = [('i', ef['i'][0], ef['i'][1], ef['i'][2]),
               ('j', ef['j'][0], ef['j'][1], ef['j'][2])]
        if dist is not None:
            try:
                import numpy as np
                N = np.asarray(dist['N'], dtype=float)
                V = np.asarray(dist['V'], dtype=float)
                M = np.asarray(dist['M'], dtype=float)
                x = np.asarray(dist['x'], dtype=float)
                # Both extreme moments (sagging and hogging) and peak shear.
                for idx in (int(np.argmax(M)), int(np.argmin(M)),
                            int(np.argmax(np.abs(V)))):
                    pts.append((f'x={x[idx]:.2f}m',
                                float(N[idx]), float(V[idx]), float(M[idx])))
            except Exception:
                pass
        return pts

    def _check_elem(elem, case_id, ef, dist=None):
        sec  = struc.sections[elem.section_name]
        mat  = struc.materials[sec.material_name]
        cm   = struc.concrete_materials.get(sec.material_name)
        if cm is None:
            return None
        fck = _fck(cm.concrete_class)
        fyk = _fyk(cm.steel_class)
        rcs = RCSection(sec.b, sec.h, sec.rc_cover, fck, fyk,
                        cm.gamma_c, cm.gamma_s, cm.alpha_cc)
        best = None
        for label, N, V, M in _critical_points(ef, dist):
            flex = rcs.flexural_reinforcement(M, N)
            shear = rcs.shear_reinforcement(V, N,
                                            cotg_theta=elem.rc_cotg_theta,
                                            alpha_s=sec.rc_alpha_s,
                                            As_long=flex.get('As_tension', flex['As_bot']),
                                            min_shear=getattr(sec, 'rc_shear_min', True))
            candidate = {
                'combination': case_id,
                'end': label,
                'M_Ed': M, 'N_Ed': N, 'V_Ed': V,
                'As_bot': flex['As_bot'],
                'As_top': flex['As_top'],
                'Asw_s':  shear['Asw_s'],
                'VRd_max': shear.get('VRd_max'),
                'mu_Ed':  flex['mu_Ed'],
                'note': flex['note'],
            }
            if best is None or (candidate['As_bot'] + candidate['As_top'] >
                                best['As_bot'] + best['As_top']):
                best = candidate
        return best

    for elem in struc.bar_elements:
        if not elem.rc_design:
            continue
        if struc.sections[elem.section_name].material_name not in struc.concrete_materials:
            continue

        worst = None

        # Check individual load cases (use span distribution for critical sections)
        for case_id in all_cases:
            ef = results['element_forces'][case_id].get(elem.id)
            if ef is None:
                continue
            dist = (results.get('element_distribution', {})
                    .get(case_id, {}).get(elem.id))
            cand = _check_elem(elem, case_id, ef, dist)
            if cand and (worst is None or
                         cand['As_bot'] + cand['As_top'] > worst['As_bot'] + worst['As_top']):
                worst = cand

        # Check combinations
        for combo_id, ef_map in combo_cases.items():
            ef = ef_map.get(elem.id)
            if ef is None:
                continue
            dist = (results.get('combo_distribution', {})
                    .get(combo_id, {}).get(elem.id))
            cand = _check_elem(elem, combo_id, ef, dist)
            if cand and (worst is None or
                         cand['As_bot'] + cand['As_top'] > worst['As_bot'] + worst['As_top']):
                worst = cand

        if worst:
            rc_results[elem.id] = worst

    return rc_results


# ---------------------------------------------------------------------------
# Persistence — the single shared mechanism for embedding the design in results
# ---------------------------------------------------------------------------

def store_reinforcement(results: dict, rows, envelope=None, *,
                        gamma_c: float = 1.5, gamma_s: float = 1.15,
                        alpha_cc: float = 1.0, prefs: dict | None = None):
    """Embed an already-computed reinforcement design into
    ``results['reinforcement']`` in the canonical shape
    ``{'rows', 'envelope', 'prefs'}``.

    This is the single source of truth for the payload written to a ``.x2d``:
    the GUI (which already has ``rows``/``envelope`` for its tables) and any
    external exporter both call this so the persisted structure is identical.
    ``save_x2d`` serialises whatever is in ``results`` and ``load_x2d`` restores
    it, so storing here is what makes the reinforcement survive a reload.

    Passing empty ``rows`` and ``envelope`` removes the key. Returns the stored
    payload dict (or ``None`` when nothing was stored).
    """
    if not rows and not envelope:
        results.pop('reinforcement', None)
        return None
    payload_prefs = {'gamma_c': gamma_c, 'gamma_s': gamma_s,
                     'alpha_cc': alpha_cc}
    if prefs:
        payload_prefs.update(prefs)
    payload = {'rows': rows or None, 'envelope': envelope or None,
               'prefs': payload_prefs}
    results['reinforcement'] = payload
    return payload


def design_and_store(struc, results: dict, *,
                     gamma_c: float = 1.5, gamma_s: float = 1.15,
                     alpha_cc: float = 1.0, combinations=None,
                     prefs: dict | None = None, with_envelope: bool = True):
    """Design every Concrete element (EC2) **and** store the result in
    ``results['reinforcement']`` in one step.

    This is the shared high-level entry point: call it and the reinforcement is
    both computed and persisted (so a subsequent ``save_x2d`` carries it). It
    composes the pure functions :func:`design_concrete_sections` and
    :func:`reinforcement_envelope` and then :func:`store_reinforcement` — the
    pure functions stay side-effect free for other callers (e.g. crack-width),
    while this wrapper provides the "calculate = persist" behaviour.

    Returns the stored payload dict (or ``None`` if no concrete elements were
    designed).
    """
    rows = design_concrete_sections(
        struc, results, gamma_c=gamma_c, gamma_s=gamma_s,
        alpha_cc=alpha_cc, combinations=combinations)
    envelope = (reinforcement_envelope(
        struc, results, gamma_c=gamma_c, gamma_s=gamma_s,
        alpha_cc=alpha_cc, combinations=combinations)
        if with_envelope else None)
    return store_reinforcement(results, rows, envelope, gamma_c=gamma_c,
                               gamma_s=gamma_s, alpha_cc=alpha_cc, prefs=prefs)


def store_crack(results: dict, rows, prefs: dict | None = None):
    """Embed an already-computed crack-width (SLS) check into
    ``results['crack']`` in the canonical shape ``{'rows', 'prefs'}`` — the same
    payload the GUI persists, so ``save_x2d`` keeps it and ``load_x2d`` restores
    it. Passing empty rows removes the key. Returns the stored payload or None.
    """
    if not rows:
        results.pop('crack', None)
        return None
    payload = {'rows': rows, 'prefs': prefs or {}}
    results['crack'] = payload
    return payload


def crack_and_store(struc, results: dict, rc_rows, *, combinations=None,
                    k_t: float = 0.4, wk_limit: float = 0.3,
                    phi_mm: float = 16.0, rc_envelope=None,
                    sls_combo: str = 'Quasi-permanent'):
    """Run the EC2 §7.3.4 crack-width check for the given SLS ``combinations``
    **and** store it in ``results['crack']`` in one step (shared entry point,
    mirroring :func:`design_and_store`).

    ``rc_rows`` / ``rc_envelope`` are the ULS reinforcement design (rows and
    per-station envelope) that provide the steel used in the crack calculation.
    Returns the stored payload dict (or ``None`` if nothing was computed).
    """
    rows = crack_width_sections(
        struc, results, rc_rows, combinations=combinations,
        phi_mm=phi_mm, k_t=k_t, wk_limit=wk_limit, rc_envelope=rc_envelope)
    return store_crack(results, rows,
                       {'sls_combo': sls_combo, 'k_t': k_t, 'wk_max': wk_limit})


# ---------------------------------------------------------------------------
# RC column design (EC2 §5.8 slenderness + 2nd order, §6.1 M-N interaction)
# ---------------------------------------------------------------------------

def _reinforcement_grade_for_fyk(fyk_mpa: float) -> str:
    """Map a numeric fyk [MPa] to an eurocodepy reinforcement grade name.

    eurocodepy's ``Reinforcement`` materials are built from grade-name
    strings validated against its database (e.g. "B500B"), not raw fyk.
    xdfem2D only carries the numeric fyk, so this guesses the ductility-B
    grade at the nearest standard fyk, falling back to "B500B" (the most
    common European grade) when nothing matches.
    """
    from eurocodepy import dbase
    fyk_i = int(round(fyk_mpa))
    for candidate in (f"B{fyk_i}B", f"A{fyk_i}NR", f"B{fyk_i}C", f"B{fyk_i}A"):
        if candidate in dbase.ReinforcementGrades:
            return candidate
    return "B500B"


def design_concrete_columns(struc, results: dict, prefs: dict | None = None,
                            combinations=None, with_reports: bool = False) -> dict:
    """Run the EC2 §5.8 (slenderness + 2nd-order) and §6.1 (M-N interaction)
    column checks on every physical member flagged with ``is_column``.

    Mirrors :func:`design_concrete_sections` (section selection, envelope
    handling) and :func:`~xdfem2d.steel_design.design_steel_members` (physical
    -member grouping via :func:`~xdfem2d.member_utils.identify_members`,
    return shape). Delegates every normative calculation to
    ``eurocodepy.ec2.uls.column`` (or ``ec2.uls2023.column`` when the 2023
    edition is selected in preferences) — see dev/RC_COLUMN_DESIGN.md.

    xdfem2D is a 2D (plane-frame/grillage) program: only one in-plane bending
    moment is available per element, so it is mapped onto the column's
    ``my_ed``/``m02_y`` (major-axis) actions; the out-of-plane moment
    (``mz_ed``/``m02_z``) is always zero but is still checked against the
    EC2 minimum-eccentricity requirement through the normal biaxial check.

    Parameters
    ----------
    struc, results : as returned by ``Structure2D.calculate()``
    prefs : Code-preferences dict. Recognised keys: ``gamma_c`` (1.5),
        ``gamma_s`` (1.15), ``alpha_cc`` (1.0), ``ec2_edition``
        ("2004"/"2023", default "2004").
    combinations : iterable of combination ids to design for, or ``None``
        for every available combination.
    with_reports : when True, attach a step-by-step trace for the governing
        case of each member under an extra ``"reports"`` key.

    Returns
    -------
    dict with keys ``"members"`` (list of result rows, one per physical
    column member) and ``"elements"`` (``{bar_id: utilization}``), mirroring
    ``design_steel_members``'s return shape.
    """
    from .member_utils import identify_members
    from .models import SectionType, section_type_of

    prefs = prefs or {}
    gamma_c = float(prefs.get('gamma_c', 1.5))
    gamma_s = float(prefs.get('gamma_s', 1.15))
    alpha_cc = float(prefs.get('alpha_cc', 1.0))
    edition = str(prefs.get('ec2_edition', '2004'))

    if edition == '2023':
        from eurocodepy.ec2 import uls2023 as ec2mod
    else:
        from eurocodepy.ec2 import uls as ec2mod
    from eurocodepy.ec2.materials import Concrete, Reinforcement

    combo_cases = {}
    for combo_id, cd in results.get('combinations', {}).items():
        ef = cd.get('element_forces', {})
        if isinstance(ef, dict) and 'max' in ef:
            combo_cases[combo_id] = [('max', ef.get('max', {})),
                                     ('min', ef.get('min', {}))]
        else:
            combo_cases[combo_id] = [('', ef)]
    if combinations is not None:
        wanted = set(combinations)
        combo_cases = {k: v for k, v in combo_cases.items() if k in wanted}

    members, _by_bar = identify_members(struc)

    def _is_column(m):
        b0 = struc.bar_elements_by_id[m.bar_ids[0]]
        sec = struc.sections.get(m.section_name)
        if sec is None or section_type_of(
                struc.materials.get(sec.material_name)) != SectionType.CONCRETE:
            return None
        if not _effective_is_column(b0, sec):
            return None
        return b0, sec

    member_rows = []
    element_ratios: dict[str, float] = {}
    reports: dict[str, dict] = {}

    for m in members:
        flagged = _is_column(m)
        if flagged is None:
            continue
        b0, sec = flagged
        st = _section_strengths(struc, sec)
        if st is None:
            continue
        fck_mpa, fyk_mpa = st

        concrete = Concrete.from_fck(fck_mpa)
        reinforcement = Reinforcement(_reinforcement_grade_for_fyk(fyk_mpa))
        built = _column_section_and_placeholder_rebar(
            ec2mod, sec)
        if built is None:
            continue   # A/I-only section — no fibre geometry to check
        section, rebar = built

        ky = b0.sd_ky if b0.sd_ky is not None else 1.0
        kz = b0.sd_kz if b0.sd_kz is not None else 1.0

        best = None
        best_inp = None
        for combo_id, sub_cases in combo_cases.items():
            for sub_label, ef_map in sub_cases:
                n_ed = m0_ed = 0.0
                for bid in m.bar_ids:
                    ef = ef_map.get(bid)
                    if ef is None:
                        continue
                    for _l, _r, n, _v, mm in _critical_points(ef, None):
                        # eurocodepy uses N_Ed compression-positive; xdfem2D
                        # uses tension-positive internally.
                        n_ed = max(n_ed, -n)
                        m0_ed = max(m0_ed, abs(mm))
                combo_name = f"{combo_id} ({sub_label})" if sub_label else combo_id

                inp = ec2mod.ColumnInput(
                    section=section, concrete=concrete,
                    reinforcement=reinforcement, rebar=rebar,
                    cover_m=sec.rc_cover, alpha_cc=alpha_cc,
                    length_m=m.length, k_y=ky, k_z=kz,
                    m01_y=m0_ed, m02_y=m0_ed, m01_z=0.0, m02_z=0.0,
                    phi_ef=float(getattr(sec, 'rc_phi_ef', 2.0)),
                    n_ed=n_ed, my_ed=m0_ed, mz_ed=0.0,
                    second_order_method=getattr(
                        sec, 'rc_second_order_method', 'nominal_curvature'))
                res = ec2mod.eurocode2_column_check(inp)
                if best is None or res.utilization > best['utilization']:
                    best = {
                        'member': m.id, 'elements': ', '.join(m.bar_ids),
                        'section': sec.name, 'length': m.length,
                        'combination': combo_name,
                        'N_Ed': n_ed, 'My_Ed': m0_ed, 'Mz_Ed': 0.0,
                        'lambda_y': res.lambda_y, 'lambda_z': res.lambda_z,
                        'slender_y': res.slender_y, 'slender_z': res.slender_z,
                        'method_used': res.method_used,
                        'MRd_y': res.mrd_y, 'MRd_z': res.mrd_z,
                        'utilization': res.utilization,
                        'passed': res.passed,
                    }
                    best_inp = inp
        if best is not None:
            member_rows.append(best)
            for bid in m.bar_ids:
                element_ratios[bid] = best['utilization']
            if with_reports and best_inp is not None:
                from eurocodepy.calc_report import CalcReport
                rep = CalcReport(
                    title=f"Concrete column {m.id} — EN 1992-1-1",
                    meta={'id': m.id, 'member': m.id, 'section': sec.name,
                          'combination': best['combination'],
                          'fck': fck_mpa, 'fyk': fyk_mpa,
                          'gamma_c': gamma_c, 'gamma_s': gamma_s,
                          'alpha_cc': alpha_cc, 'edition': edition,
                          'utilization': best['utilization'],
                          'ok': bool(best['passed'])})
                ec2mod.eurocode2_column_check(best_inp, trace=rep)
                # M-N interaction curve (EN 1992-1-1 §6.1): sample MRd_y over
                # a range of N from 0 to the squash load, for the report to
                # plot/tabulate the full envelope, not just the governing
                # point.
                try:
                    n_hi = section.area * 1000.0 * (alpha_cc * concrete.fcd)
                    rep.section("M-N interaction curve (axis y)")
                    n_points = 11
                    for i in range(n_points):
                        n_i = n_hi * i / (n_points - 1)
                        mrd_i = ec2mod.uniaxial_moment_resistance(
                            best_inp, n_i, "y")
                        rep.step(f"N{i}", n_i, "kN",
                                clause="EN 1992-1-1 §6.1")
                        rep.step(f"MRd_y{i}", mrd_i, "kNm",
                                clause="EN 1992-1-1 §6.1")
                except Exception:
                    # Best-effort only — the governing-point check above is
                    # already in the report; a failed sweep must not drop it.
                    pass
                reports[m.id] = rep.to_dict()

    out = {'members': member_rows, 'elements': element_ratios}
    if with_reports:
        out['reports'] = reports
    return out


# ---------------------------------------------------------------------------
# RC column automatic reinforcement design (Fase 4 — dev/RC_COLUMN_DESIGN.md)
# ---------------------------------------------------------------------------

def suggest_column_reinforcement(struc, results: dict, prefs: dict | None = None,
                                 combinations=None) -> dict:
    """Suggest the lightest symmetric reinforcement layout for every physical
    member flagged ``is_column``, using the governing (max-utilization) load
    case exactly as :func:`design_concrete_columns` selects it, then handing
    that single case to ``eurocodepy.ec2.uls.column.design_column_reinforcement``
    (or its 2023-edition counterpart).

    Unlike :func:`design_concrete_columns` (a *verification* of a
    user-specified ``rc_n_bars``/``rc_bar_phi`` layout), this is *design*: the
    reinforcement itself is the output. ``rc_n_bars``/``rc_bar_phi`` on the
    member's section are only used as the starting point for reporting; the
    returned layout may use a different bar count/diameter.

    Returns ``{"members": rows, "elements": {bar_id: utilization}}``, mirroring
    :func:`design_concrete_columns`. Each row additionally carries
    ``n_bars_suggested`` and ``diameter_mm_suggested``. A member for which no
    layout in the default search range satisfies EC2 gets a row with
    ``passed=False`` and ``error`` set, rather than being silently dropped —
    so a failed column is still visible to the user, not hidden.
    """
    from .member_utils import identify_members
    from .models import SectionType, section_type_of

    prefs = prefs or {}
    alpha_cc = float(prefs.get('alpha_cc', 1.0))
    edition = str(prefs.get('ec2_edition', '2004'))

    if edition == '2023':
        from eurocodepy.ec2 import uls2023 as ec2mod
    else:
        from eurocodepy.ec2 import uls as ec2mod
    from eurocodepy.ec2.materials import Concrete, Reinforcement

    combo_cases = {}
    for combo_id, cd in results.get('combinations', {}).items():
        ef = cd.get('element_forces', {})
        if isinstance(ef, dict) and 'max' in ef:
            combo_cases[combo_id] = [('max', ef.get('max', {})),
                                     ('min', ef.get('min', {}))]
        else:
            combo_cases[combo_id] = [('', ef)]
    if combinations is not None:
        wanted = set(combinations)
        combo_cases = {k: v for k, v in combo_cases.items() if k in wanted}

    members, _by_bar = identify_members(struc)
    member_rows = []
    element_ratios: dict[str, float] = {}

    for m in members:
        b0 = struc.bar_elements_by_id[m.bar_ids[0]]
        sec = struc.sections.get(m.section_name)
        if sec is None or section_type_of(
                struc.materials.get(sec.material_name)) != SectionType.CONCRETE:
            continue
        if not _effective_is_column(b0, sec):
            continue
        st = _section_strengths(struc, sec)
        if st is None:
            continue
        fck_mpa, fyk_mpa = st

        concrete = Concrete.from_fck(fck_mpa)
        reinforcement = Reinforcement(_reinforcement_grade_for_fyk(fyk_mpa))
        built = _column_section_and_placeholder_rebar(
            ec2mod, sec)
        if built is None:
            # A/I-only section — no fibre geometry to suggest a layout for.
            # Visible, not silently dropped: same convention as an
            # impossible layout below (passed=False + error).
            member_rows.append({
                'member': m.id, 'elements': ', '.join(m.bar_ids),
                'section': sec.name, 'length': m.length,
                'combination': '', 'N_Ed': 0.0, 'My_Ed': 0.0,
                'passed': False, 'utilization': float('inf'),
                'error': ("Section is defined by area/inertia overrides, "
                         "not real geometry — no rectangular/circular "
                         "shape to design a layout for."),
                'n_bars_suggested': None, 'diameter_mm_suggested': None,
                'As_suggested_cm2': None,
                'n_bars_y_suggested': None, 'n_bars_z_suggested': None,
            })
            continue
        section, placeholder = built
        ky = b0.sd_ky if b0.sd_ky is not None else 1.0
        kz = b0.sd_kz if b0.sd_kz is not None else 1.0

        # Governing (max N0 + M0) case across the selected combinations,
        # same envelope logic as design_concrete_columns.
        n_ed = m0_ed = 0.0
        combo_name = ''
        best_load = -1.0
        for combo_id, sub_cases in combo_cases.items():
            for sub_label, ef_map in sub_cases:
                n_try = m_try = 0.0
                for bid in m.bar_ids:
                    ef = ef_map.get(bid)
                    if ef is None:
                        continue
                    for _l, _r, n, _v, mm in _critical_points(ef, None):
                        n_try = max(n_try, -n)
                        m_try = max(m_try, abs(mm))
                score = n_try + m_try
                if score > best_load:
                    best_load = score
                    n_ed, m0_ed = n_try, m_try
                    combo_name = f"{combo_id} ({sub_label})" if sub_label else combo_id

        inp = ec2mod.ColumnInput(
            section=section, concrete=concrete, reinforcement=reinforcement,
            rebar=placeholder, cover_m=sec.rc_cover, alpha_cc=alpha_cc,
            length_m=m.length, k_y=ky, k_z=kz,
            m01_y=m0_ed, m02_y=m0_ed, m01_z=0.0, m02_z=0.0,
            phi_ef=float(getattr(sec, 'rc_phi_ef', 2.0)),
            n_ed=n_ed, my_ed=m0_ed, mz_ed=0.0,
            second_order_method=getattr(
                sec, 'rc_second_order_method', 'nominal_curvature'))

        row = {
            'member': m.id, 'elements': ', '.join(m.bar_ids),
            'section': sec.name, 'length': m.length,
            'combination': combo_name, 'N_Ed': n_ed, 'My_Ed': m0_ed,
        }
        try:
            trial, res = ec2mod.design_column_reinforcement(inp)
        except ValueError as e:
            row.update({'passed': False, 'utilization': float('inf'),
                       'error': str(e), 'n_bars_suggested': None,
                       'diameter_mm_suggested': None,
                       'As_suggested_cm2': None,
                       'n_bars_y_suggested': None, 'n_bars_z_suggested': None})
        else:
            from eurocodepy.utils.crosssection import CircularCrossSection
            n_total = len(trial.rebar.diameters_mm)
            n_y_suggested = n_z_suggested = None
            if not isinstance(section, CircularCrossSection):
                # positions_m holds (y, z); the top/bottom faces (n_y each)
                # sit at the extreme |z|, the left/right faces (n_z each) at
                # the extreme |y| — recover n_y/n_z from the layout actually
                # returned rather than assuming the section's configured
                # rc_n_bars_y/rc_n_bars_z (the search may have chosen a
                # different pair to satisfy EC2).
                zs = [z for _y, z in trial.rebar.positions_m]
                ys = [y for y, _z in trial.rebar.positions_m]
                z_max = max(zs)
                y_max = max(ys)
                n_y_suggested = sum(1 for _y, z in trial.rebar.positions_m
                                    if abs(z - z_max) < 1e-9 or abs(z + z_max) < 1e-9) // 2
                n_z_suggested = sum(1 for y, _z in trial.rebar.positions_m
                                    if abs(y - y_max) < 1e-9 or abs(y + y_max) < 1e-9) // 2
            row.update({
                'passed': res.passed, 'utilization': res.utilization,
                'MRd_y': res.mrd_y, 'MRd_z': res.mrd_z,
                'n_bars_suggested': n_total,
                'diameter_mm_suggested': trial.rebar.diameters_mm[0],
                'As_suggested_cm2': trial.rebar.total_area_cm2(),
                'n_bars_y_suggested': n_y_suggested,
                'n_bars_z_suggested': n_z_suggested,
            })
            element_ratios.update({bid: res.utilization for bid in m.bar_ids})
        member_rows.append(row)

    return {'members': member_rows, 'elements': element_ratios}


# ---------------------------------------------------------------------------
# Beam bars — required areas (design_concrete_sections rows) -> bar layouts
# ---------------------------------------------------------------------------

def _chain_beams_with_start(struc, members: list) -> list:
    """Group *members* (spans) into continuous beams, oriented and ordered.

    *members* are the beam spans only (concrete, not columns), so bars of
    columns / walls never count at a joint: a beam-column node with two
    collinear beam spans and a column is a continuity node.

    At every node where spans end, the collinear pairs are joined when both
    use the same section, neither end is hinged and the turn is within
    ``member_utils._ANGLE_TOL`` (1°). A pair is chosen greedily by smallest
    deviation, so three beams meeting in a T join only the collinear pair and
    the third starts its own beam; a section change or a hinge is a break of
    continuity (the spans stay in different beams).

    Returns ``[(start_xy, [(member, flip), …]), …]`` — each beam oriented
    left → right (smaller x first; vertical beams bottom → top) — ordered
    bottom → top and left → right by the starting point (y within 1 mm counts
    as one level). ``flip`` is True when the member's own node order runs
    against the beam.
    """
    from .member_utils import _ANGLE_TOL, _hinge_at, _unit_from

    bars = {b.id: b for b in struc.bar_elements}
    ends: dict = {}                   # node id -> [(member, bar id at end)]
    for m in members:
        ends.setdefault(m.node_ids[0], []).append((m, m.bar_ids[0]))
        ends.setdefault(m.node_ids[-1], []).append((m, m.bar_ids[-1]))

    nxt: dict = {}                    # member id -> {node id: member}
    for node, lst in ends.items():
        cand = []
        for i in range(len(lst)):
            for j in range(i + 1, len(lst)):
                (m1, b1), (m2, b2) = lst[i], lst[j]
                if m1 is m2 or m1.section_name != m2.section_name:
                    continue
                bar1, bar2 = bars[b1], bars[b2]
                if _hinge_at(bar1, node) or _hinge_at(bar2, node):
                    continue
                u1 = _unit_from(struc, node, bar1)
                u2 = _unit_from(struc, node, bar2)
                dot = max(-1.0, min(1.0, u1[0] * u2[0] + u1[1] * u2[1]))
                dev = math.pi - math.acos(dot)
                if dot < 0.0 and dev <= _ANGLE_TOL:
                    cand.append((dev, i, j))
        used = set()
        for dev, i, j in sorted(cand):
            if i in used or j in used:
                continue
            used.update((i, j))
            m1, m2 = lst[i][0], lst[j][0]
            nxt.setdefault(m1.id, {})[node] = m2
            nxt.setdefault(m2.id, {})[node] = m1

    seen, chains = set(), []
    for m in members:
        if m.id in seen or len(nxt.get(m.id, {})) == 2:
            continue                  # start only from a beam end
        chain, cur, prev_node = [m], m, None
        seen.add(m.id)
        while True:
            options = [(n, o) for n, o in nxt.get(cur.id, {}).items()
                       if n != prev_node and o.id not in seen]
            if not options:
                break
            prev_node, cur = options[0]
            seen.add(cur.id)
            chain.append(cur)
        chains.append(chain)
    for m in members:                 # closed loops (no free end): as is
        if m.id not in seen:
            seen.add(m.id)
            chains.append([m])

    def node_xy(nid):
        n = struc.nodes[nid]
        return (n.x, n.y)

    oriented = []
    for chain in chains:
        # entry node of the chain: the end of the first member that is not
        # shared with the second one
        if len(chain) == 1:
            entry = chain[0].node_ids[0]
        else:
            e0 = {chain[0].node_ids[0], chain[0].node_ids[-1]}
            free = e0 - {chain[1].node_ids[0], chain[1].node_ids[-1]}
            entry = next(iter(free)) if len(free) == 1 else chain[0].node_ids[0]
        flips, cur_in = [], entry
        for m in chain:
            flip = m.node_ids[-1] == cur_in and m.node_ids[0] != cur_in
            flips.append(flip)
            cur_in = m.node_ids[0] if flip else m.node_ids[-1]
        start, end = node_xy(entry), node_xy(cur_in)
        if (start[0], start[1]) > (end[0], end[1]):       # run left -> right
            chain = chain[::-1]
            flips = [not f for f in flips[::-1]]
            start = end
        oriented.append((start, list(zip(chain, flips))))

    return _sort_bottom_up_left_right(oriented)


def _sort_bottom_up_left_right(items: list) -> list:
    """Sort ``(start_xy, payload, …)`` tuples bottom → top (y within 1 mm =
    same level), then left → right."""
    items = sorted(items, key=lambda t: t[0][1])
    levels, cur = [], []
    for item in items:
        if cur and item[0][1] - cur[-1][0][1] > 1e-3:
            levels.append(cur)
            cur = []
        cur.append(item)
    if cur:
        levels.append(cur)
    out = []
    for lvl in levels:
        out += sorted(lvl, key=lambda t: t[0][0])
    return out


def _chain_beam_spans(struc, members: list) -> list:
    """:func:`_chain_beams_with_start` without the starting points:
    ``[[(member, flip), …], …]``."""
    return [pairs for _, pairs in _chain_beams_with_start(struc, members)]


def _is_beam_member(struc, m) -> bool:
    """A member made only of concrete, non-column, real-geometry bars."""
    from .models import SectionType, section_type_of
    for bid in m.bar_ids:
        el = struc.bar_elements_by_id.get(bid)
        sec = struc.sections.get(el.section_name) if el else None
        if sec is None or section_type_of(
                struc.materials.get(sec.material_name)) != SectionType.CONCRETE:
            return False
        if _effective_is_column(el, sec) or _is_generic_concrete_section(sec):
            return False
    return True


#: Automatic beam detection only considers members whose slope to the
#: horizontal is at most this [degrees]; steeper ones (columns, walls, braces)
#: are never proposed as beams. An explicit ``beam`` tag is not limited by it.
AUTO_BEAM_MAX_SLOPE_DEG = 45.0


def _is_near_horizontal(struc, m, max_deg: float = None) -> bool:
    """True when every bar of member *m* slopes at most *max_deg* degrees
    (default :data:`AUTO_BEAM_MAX_SLOPE_DEG`) from the horizontal."""
    limit = AUTO_BEAM_MAX_SLOPE_DEG if max_deg is None else max_deg
    for bid in m.bar_ids:
        b = struc.bar_elements_by_id.get(bid)
        ni = struc.nodes.get(b.node_i) if b else None
        nj = struc.nodes.get(b.node_j) if b else None
        if ni is None or nj is None:
            return False
        slope = math.degrees(math.atan2(abs(nj.y - ni.y), abs(nj.x - ni.x)))
        if slope > limit + 1e-6:
            return False
    return True


def _shear_station_points(struc, results, elem, sec, st, gamma_c, gamma_s,
                          alpha_cc):
    """Required shear reinforcement Asw/s [m²/m] at every station of the span
    diagrams of *elem* (envelope over the plain combinations):
    ``[(x, Asw_s)]``, with the same longitudinal steel and the minimum-shear
    rule the section design uses."""
    rcs = RCSection(sec.b, sec.h, sec.rc_cover, st[0], st[1],
                    gamma_c, gamma_s, alpha_cc)
    acc: dict = {}
    cache: dict = {}
    for cd in results.get('combo_distribution', {}).values():
        dist = cd.get(elem.id)
        if dist is None:
            continue
        try:
            for x, N, V, M in zip(dist['x'], dist['N'], dist['V'], dist['M']):
                key = (round(float(M), 6), round(float(N), 6),
                       round(float(V), 6))
                if key not in cache:
                    fl = rcs.flexural_reinforcement(float(M), float(N))
                    sh = rcs.shear_reinforcement(
                        float(V), float(N), cotg_theta=elem.rc_cotg_theta,
                        alpha_s=sec.rc_alpha_s,
                        As_long=fl.get('As_tension', fl['As_bot']),
                        min_shear=getattr(sec, 'rc_shear_min', True))
                    cache[key] = float(sh['Asw_s'])
                a = acc.setdefault(round(float(x), 6), [float(x), 0.0])
                a[1] = max(a[1], cache[key])
        except (KeyError, TypeError, ValueError):
            continue
    return [tuple(v) for v in sorted(acc.values())]


def compiled_beam_index(struc):
    """``(mesh, obj_of, obj_bars)`` to define beams from bar ids of the
    *compiled mesh* (the ids ``design_beam_bars`` / ``suggest_beams`` return):
    the mesh, which line object generated each of its bars, and each object's
    bars. A model without geometry objects is its own mesh."""
    trace = {}
    if getattr(struc, 'geometry_objects', None):
        from .geo_expand import expand_geometry
        mesh, trace = expand_geometry(struc)
    else:
        mesh = struc
    obj_of = {b: oid for oid, tr in trace.items()
              for b in tr.get('elems', [])
              if hasattr(struc.geometry_objects.get(oid), 'beam')}
    obj_bars: dict = {}
    for b, oid in obj_of.items():
        obj_bars.setdefault(oid, set()).add(b)
    return mesh, obj_of, obj_bars


def beam_assignment(index, bar_ids):
    """``(real bar ids, line object ids)`` that define the beam made of the
    compiled-mesh bars *bar_ids*, or ``None`` when it cannot be tagged as a
    whole: a tag belongs to a whole object, so an object that also generates
    bars outside *bar_ids* (one object spanning several beams) blocks it."""
    _mesh, obj_of, obj_bars = index
    ids = set(bar_ids)
    objs = {obj_of[b] for b in ids if b in obj_of}
    if any(not obj_bars[o] <= ids for o in objs):
        return None
    return [b for b in bar_ids if b not in obj_of], sorted(objs)


def assign_detected_beam(struc, bar_ids, index=None, tag=None, name=None):
    """Define (tag) the beam made of compiled-mesh bars *bar_ids* in the model
    *struc* and return its tag, or ``None`` if it cannot be tagged as a whole
    (see :func:`beam_assignment`). *index* (:func:`compiled_beam_index`) can be
    passed to avoid rebuilding the mesh for several beams."""
    index = index or compiled_beam_index(struc)
    got = beam_assignment(index, bar_ids)
    if got is None or not (got[0] or got[1]):
        return None
    real, objs = got
    return struc.assign_beam(real, tag=tag, name=name, object_ids=objs)


def promote_auto_beam(struc, rows: list, beam_name: str, index=None):
    """The beam tag for the beam named *beam_name* in the design *rows*: its
    existing tag, or — for an automatically detected beam — a new one assigned
    now (``None`` if it cannot be tagged as a whole). This is how an automatic
    beam becomes editable: the detailing is always attached to a tag."""
    mine = [r for r in rows if r['beam'] == beam_name]
    if not mine:
        raise KeyError(f"beam '{beam_name}' not found in the design rows")
    if mine[0].get('beam_tag'):
        return mine[0]['beam_tag']
    bars = [b for r in mine for b in r['elements']]
    return assign_detected_beam(struc, bars, index)


def _is_beam_object(struc, obj) -> bool:
    """A line object (segment/arc/multisegment) whose bars are concrete,
    non-column beams (its own ``is_column`` override, else its section's)."""
    from .models import SectionType, section_type_of
    if not hasattr(obj, 'beam'):
        return False
    sec = struc.sections.get(getattr(obj, 'section_name', ''))
    if sec is None or section_type_of(
            struc.materials.get(sec.material_name)) != SectionType.CONCRETE:
        return False
    is_col = getattr(obj, 'is_column', None)
    if is_col is None:
        is_col = bool(getattr(sec, 'is_column', False))
    return not is_col and not _is_generic_concrete_section(sec)


def suggest_beams(struc) -> list:
    """Beams the automatic detection would form, as lists of bar ids ordered
    along each beam (left → right), the beams themselves ordered bottom → top
    and left → right. Uses the concrete non-column bars that slope at most
    :data:`AUTO_BEAM_MAX_SLOPE_DEG` from the horizontal (columns, walls and
    braces are never proposed), ignoring the existing ``beam`` tags — meant as
    a *suggestion* for the user to accept.
    """
    from .member_utils import identify_members
    members, _ = identify_members(struc)
    members = [m for m in members
               if _is_beam_member(struc, m) and _is_near_horizontal(struc, m)]
    out = []
    for pairs in _chain_beam_spans(struc, members):
        ids = []
        for m, flip in pairs:
            ids += m.bar_ids[::-1] if flip else m.bar_ids
        out.append(ids)
    return out


def _member_tag(struc, m):
    """Beam tag of a member: the common tag of its bars (first non-None one
    when they disagree)."""
    tags = [struc.bar_elements_by_id[b].beam for b in m.bar_ids
            if b in struc.bar_elements_by_id]
    tags = [t for t in tags if t]
    return tags[0] if tags else None


def _row_x(label: str, length: float):
    """Station [m] along an element from a design-row ``location`` label."""
    if label == 'i':
        return 0.0
    if label == 'j':
        return length
    if label.startswith('x='):
        try:
            return float(label[2:].rstrip('m'))
        except ValueError:
            return None
    return None


def _station_points(struc, results, elem, sec, st, gamma_c, gamma_s, alpha_cc):
    """Flexural steel [m²] at every station of the span diagrams of *elem*.

    Returns ``[(x, As_bot, As_top)]`` — for each station the envelope over all
    plain (non-envelope) combinations that have a ``combo_distribution``.
    """
    rcs = RCSection(sec.b, sec.h, sec.rc_cover, st[0], st[1],
                    gamma_c, gamma_s, alpha_cc)
    acc: dict = {}                       # rounded x -> [x, As_bot, As_top]
    cache: dict = {}
    for cd in results.get('combo_distribution', {}).values():
        dist = cd.get(elem.id)
        if dist is None:
            continue
        try:
            xs, Ns, Ms = dist['x'], dist['N'], dist['M']
            for x, N, M in zip(xs, Ns, Ms):
                key = (round(float(M), 6), round(float(N), 6))
                if key not in cache:
                    f = rcs.flexural_reinforcement(float(M), float(N))
                    cache[key] = (f['As_bot'], f['As_top'])
                ab, at = cache[key]
                a = acc.setdefault(round(float(x), 6), [float(x), 0.0, 0.0])
                a[1], a[2] = max(a[1], ab), max(a[2], at)
        except (KeyError, TypeError, ValueError):
            continue
    return [tuple(v) for v in sorted(acc.values())]


def _group_beams(struc, members: list) -> list:
    """Beams formed by *members*: ``[(start_xy, [(member, flip), …], name, tag,
    n_segments), …]`` ordered bottom → top and left → right.

    Members sharing a ``beam`` tag form that beam (any slope — the user chose
    it), split into continuous segments ``V1.1``, ``V1.2`` … at a section
    change, a hinge or a kink. Untagged members are auto-detected (``B1``,
    ``B2`` …) only when they slope at most :data:`AUTO_BEAM_MAX_SLOPE_DEG`
    from the horizontal.
    """
    registry = getattr(struc, 'beams', None) or {}
    tagged, untagged = {}, []
    for m in members:
        t = _member_tag(struc, m)
        if t:
            tagged.setdefault(t, []).append(m)
        elif _is_near_horizontal(struc, m):
            untagged.append(m)
    groups = []                        # (start_xy, pairs, name, tag, n_seg)
    taken = set()
    for tag, ms in tagged.items():
        disp = (registry.get(tag) or {}).get('name') or tag
        chains = _chain_beams_with_start(struc, ms)
        for k, (start, pairs) in enumerate(chains, 1):
            name = disp if len(chains) == 1 else f"{disp}.{k}"
            taken.add(name)
            groups.append((start, pairs, name, tag, len(chains)))
    n_auto = 0
    for start, pairs in _chain_beams_with_start(struc, untagged):
        n_auto += 1
        while f"B{n_auto}" in taken:
            n_auto += 1
        groups.append((start, pairs, f"B{n_auto}", None, 1))
    return _sort_bottom_up_left_right(groups)


def beam_labels(struc) -> dict:
    """``{bar id: {"beam": name, "tag": tag or None, "span": "T<n>"}}`` for
    every bar that belongs to a beam (tagged, or auto-detected), with the
    names :func:`design_beam_bars` uses: beams ``B1``… / their tag name, spans
    ``T1``, ``T2`` … counted left → right inside each beam segment."""
    from .member_utils import identify_members
    members, _ = identify_members(struc)
    members = [m for m in members if _is_beam_member(struc, m)]
    out = {}
    for _start, pairs, name, tag, _n in _group_beams(struc, members):
        for i, (m, _flip) in enumerate(pairs, 1):
            for bid in m.bar_ids:
                out[bid] = {"beam": name, "tag": tag, "span": f"T{i}"}
    return out


def design_beam_bars(struc, rows: list, results: dict | None = None,
                     gamma_c: float = 1.5, gamma_s: float = 1.15,
                     alpha_cc: float = 1.0, params=None, prefs=None,
                     inputs_out: list | None = None) -> list:
    """Turn the required areas of :func:`design_concrete_sections` into bars.

    **Parameters come from one place only**
    (:class:`~xdfem2d.beam_bars_params.BeamBarParams`): *prefs* — the model's
    design preferences (``beam_*`` keys, ``dmax``) — with the section
    overrides and the beam overrides of ``struc.beams``, resolved per beam by
    :func:`~xdfem2d.beam_bars_params.resolve_beam_params`; or *params*, one
    :class:`BeamBarParams` used for every beam (section and beam overrides are
    then ignored). There are no loose keyword arguments: that is what makes
    the detailing window, which only knows these layers, reproduce exactly
    what the design produced. To try a value, put it in *prefs* or build a
    ``BeamBarParams`` (``dataclasses.replace(BeamBarParams(), n_through=4)``).

    Physical members (:func:`~xdfem2d.member_utils.identify_members`) are the
    beam *spans*; spans that meet collinearly at a support or at a column
    are chained into one continuous beam (``_chain_beam_spans``: columns are
    ignored at the joint; a section change or a hinge ends the beam). Beams
    are oriented left → right and numbered ``B1``, ``B2``… bottom → top, then
    left → right. Each span is split into *zones* and
    :func:`~xdfem2d.detailing.propose_segment` chooses the bars of every zone
    of the beam, with As,min, uniform diameters and through bars (present in
    **all** zones) per face.

    Zones (per span) follow ``zone_mode``:

    * ``"cutoff"`` (default) — **cutoff points of the envelope**: the
      required-steel envelope of each face is quantised into ``cutoff_levels``
      steps of its peak; a zone boundary is placed wherever the bottom or top
      envelope changes step (at the station of the lower step, so the higher
      step keeps covering the interval). Zones shorter than
      ``min_zone_length`` [m] are merged. A span with a flat envelope stays a
      single zone.
    * ``"fixed"`` — the fractions ``zones`` of the span length
      (e.g. ``(0.25, 0.5, 0.25)``); ``(1.0,)`` = one zone per span.

    The requirement of a zone is the envelope (max) of As_bot / As_top over the
    stations inside it. With *results* those are all the stations of the
    span force diagrams (``combo_distribution``), designed for every plain
    combination (same ``gamma_c`` / ``gamma_s`` / ``alpha_cc`` as
    :func:`design_concrete_sections`), plus the rows themselves (element ends,
    envelope combinations); without *results* only the rows are used.
    ``shift_d`` shifts the envelope along the *beam* (across supports) by that
    many effective depths at both sides (EC2 §9.2.1.3 shift rule,
    a_l ≈ 0.45–1.0·d; 0 = off). Grillage models use the rows only (their
    torsion steel is not in the diagrams). The end zones of a span get at
    least ``support_bottom_ratio`` of the span's largest bottom requirement
    (EC2 §9.2.1.4(1): 25 %).

    ``inputs_out``: when a list is given, one
    :class:`~xdfem2d.detailing.SegmentInputs` per beam segment (the spans, the
    required-steel stations, the section and the effective parameters — what
    the detailing window needs to verify edits) is appended to it.

    Returns one dict per zone: beam, span (``T1``, ``T2`` … left → right
    inside the beam segment; ``member`` = the internal member id), zone (1..n), x0/x1 [m along the
    span], elements, length [m], As_bot_req / As_top_req / As_bot_prov /
    As_top_prov [m²], bottom / top (text, e.g. ``"3Ø16 + 2Ø12"``),
    bottom_layers / top_layers (``[(n, Ø mm), …]``), through_bot /
    through_top (Ø mm) and ``notes`` (warnings or an error).
    """
    from .member_utils import _bar_length, identify_members
    from .beam_bars_params import resolve_beam_params

    if params is not None:
        params.validate()

    def params_for(sec, overrides=None):
        return (params if params is not None
                else resolve_beam_params(prefs, sec, overrides))

    bars = {b.id: b for b in struc.bar_elements}
    pts: dict = {}          # element id -> [(x_local, As_bot, As_top)] [m, m²]
    for r in rows:
        el = bars.get(r['element'])
        if el is None:
            continue
        x = _row_x(str(r.get('location', '')), _bar_length(struc, el))
        if x is None:
            continue
        pts.setdefault(r['element'], []).append(
            (x, r.get('As_bot', 0.0) or 0.0, r.get('As_top', 0.0) or 0.0))

    ptsv: dict = {}         # element id -> [(x_local, Asw_s [m²/m])]
    for r in rows:
        el = bars.get(r['element'])
        x = (_row_x(str(r.get('location', '')), _bar_length(struc, el))
             if el is not None else None)
        if x is not None:
            ptsv.setdefault(r['element'], []).append(
                (x, r.get('Asw_s', 0.0) or 0.0))

    if results is not None and getattr(struc, 'domain', 'plane') != 'plate':
        for el in struc.bar_elements:
            if el.id not in pts:
                continue
            sec_e = struc.sections.get(el.section_name)
            st_e = _section_strengths(struc, sec_e) if sec_e else None
            if st_e is None:
                continue
            pts[el.id] += _station_points(struc, results, el, sec_e, st_e,
                                          gamma_c, gamma_s, alpha_cc)
            ptsv.setdefault(el.id, []).extend(_shear_station_points(
                struc, results, el, sec_e, st_e, gamma_c, gamma_s, alpha_cc))

    members, _ = identify_members(struc)
    members = [m for m in members if all(b in pts for b in m.bar_ids)]

    registry = getattr(struc, 'beams', None) or {}
    groups = _group_beams(struc, members)

    from .detailing import (layers_area, layers_text, propose_segment,
                            segment_inputs_from_design)
    out = []
    for start, oriented, beam_id, beam_tag, n_seg in groups:
        chain = [m for m, _ in oriented]
        flips = [f for _, f in oriented]
        span_name = {m.id: f"T{i}" for i, m in enumerate(chain, 1)}
        member_of = {span_name[m.id]: m for m in chain}
        sec = struc.sections.get(chain[0].section_name)
        st = _section_strengths(struc, sec) if sec is not None else None
        p = params_for(sec, (registry.get(beam_tag) or {}).get('overrides')
                       if beam_tag else None)

        # stations in beam coordinates: (X, As_bot, As_top) / (X, Asw_s) with
        # X measured along the beam
        spans_x, sta, sta_v, off_beam = [], [], [], 0.0
        for m, flip in zip(chain, flips):
            seq, off = list(m.node_ids), 0.0
            mp, mv = [], []
            for k, bid in enumerate(m.bar_ids):
                b, Lb = bars[bid], _bar_length(struc, bars[bid])
                fwd = b.node_i == seq[k]
                mp += [((off + (x if fwd else Lb - x)), ab, at)
                       for x, ab, at in pts[bid]]
                mv += [((off + (x if fwd else Lb - x)), v)
                       for x, v in ptsv.get(bid, [])]
                off += Lb
            if flip:
                mp = [(m.length - x, ab, at) for x, ab, at in mp]
                mv = [(m.length - x, v) for x, v in mv]
            spans_x.append((m, off_beam, off_beam + m.length))
            sta += [(off_beam + x, ab, at) for x, ab, at in mp]
            sta_v += [(off_beam + x, v) for x, v in mv]
            off_beam += m.length
        sta.sort()
        sta_v.sort()
        a_l = p.shift_d * (sec.h - sec.rc_cover) if sec is not None else 0.0

        notes, prop = [], None
        if n_seg > 1:
            notes.append(f"beam {beam_tag} split into {n_seg} continuous "
                         "segments (section change, hinge or non-collinear bars)")
        if st is None:
            notes.append("no fck/fyk for the section material")
        else:
            inp = segment_inputs_from_design(
                beam_id, beam_tag or beam_id,
                [(span_name[m.id], m.length) for m in chain],
                [(x, b * 1e6, t * 1e6) for x, b, t in sta],
                {"b": sec.b * 1e3, "h": sec.h * 1e3,
                 "cover": (sec.rc_cover * 1e3 - p.stirrup_diameter_mm
                           - p.d_bar_est / 2),
                 "rc_cover": sec.rc_cover * 1e3,
                 "stirrup": p.stirrup_diameter_mm, "d_bar_est": p.d_bar_est,
                 "fck": st[0], "fyk": st[1], "dg": p.dg,
                 "gamma_c": float(gamma_c), "gamma_s": float(gamma_s),
                 "alpha_s": float(sec.rc_alpha_s),
                 "shear_min": bool(getattr(sec, 'rc_shear_min', True))},
                p, a_l, shear=[(x, v * 1e6) for x, v in sta_v],
                section_name=sec.name)
            if inputs_out is not None:
                inputs_out.append(inp)
            prop = propose_segment(inp)          # the same function the
            notes.extend(prop.notes)             # detailing window uses

        if prop is not None:
            zones = [z for sp in prop.segment.spans for z in sp.zones]
            for zp, zd in zip(prop.zones, zones):
                m = member_of[zp.span]
                ok = prop.error is None
                out.append({
                    'beam': beam_id, 'beam_tag': beam_tag,
                    'span': zp.span, 'member': m.id, 'zone': zp.k + 1,
                    'x0': float(zp.x0), 'x1': float(zp.x1),
                    'elements': list(m.bar_ids),
                    'length': float(zp.x1 - zp.x0),
                    'As_bot_req': float(zp.req_bottom * 1e-6),
                    'As_top_req': float(zp.req_top * 1e-6),
                    'As_bot_prov': (float(layers_area(zd.bottom) * 1e-6)
                                    if ok else None),
                    'As_top_prov': (float(layers_area(zd.top) * 1e-6)
                                    if ok else None),
                    'bottom': layers_text(zd.bottom) if ok else '',
                    'top': layers_text(zd.top) if ok else '',
                    'bottom_layers': [[n, float(d)] for n, d in zd.bottom],
                    'top_layers': [[n, float(d)] for n, d in zd.top],
                    'through_bot': prop.through.get('bottom') if ok else None,
                    'through_top': prop.through.get('top') if ok else None,
                    'notes': list(notes)})
            continue
        # no strengths: one zone per span with the raw requirement, no bars
        for m, S0, S1 in spans_x:
            inside = [(b, t) for x, b, t in sta if S0 - 1e-9 <= x <= S1 + 1e-9]
            out.append({
                'beam': beam_id, 'beam_tag': beam_tag,
                'span': span_name[m.id], 'member': m.id, 'zone': 1,
                'x0': 0.0, 'x1': float(m.length),
                'elements': list(m.bar_ids), 'length': float(m.length),
                'As_bot_req': float(max((b for b, _ in inside), default=0.0)),
                'As_top_req': float(max((t for _, t in inside), default=0.0)),
                'As_bot_prov': None, 'As_top_prov': None,
                'bottom': '', 'top': '', 'bottom_layers': [],
                'top_layers': [], 'through_bot': None, 'through_top': None,
                'notes': list(notes)})
    return out
