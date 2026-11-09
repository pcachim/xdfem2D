"""
Export a xdfem2D model to the SAP2000 text interchange format (``.s2k``).

The 2-D model (X horizontal, Y vertical) is mapped to a planar SAP2000 model in
the global **X-Z plane** (SAP Y = 0): xdfem2D X -> SAP X, xdfem2D Y -> SAP Z.
The out-of-plane DOFs (UY, RX, RZ) are deactivated globally via the
``ACTIVE DEGREES OF FREEDOM`` table, so only the in-plane DOFs (UX, UZ, RY)
participate; the in-plane rotation maps to SAP R2 (about Y). Because the
out-of-plane axis is reversed by that mapping, in-plane moments are negated
(SAP M2 = -mz).

xdfem2D objects map to SAP as:
  * load case   -> a load pattern (SAP auto-creates its Linear Static case)
  * analysis case -> a SAP load case: Linear -> LinStatic,
    NonLinear (unilateral springs) -> NonStatic (GeoNonLin None),
    GeometricNonlinear -> NonStatic (P-Delta), Modal -> LinModal,
    Spectrum -> LinRespSpec (with its response-spectrum function and a mass
    source built from the Mass case loads)
  * load combination (LinearSum) -> a SAP combination
  * node spring -> a joint spring (uncoupled; bilateral only)
  * element spring -> a frame line spring (along the element local axes); the
    tension-/compression-only mode maps to the SAP "SimpleType"
  * concrete materials -> EC2 (Eurocode 2-2004) RC design data: concrete Fc and
    a rebar material (Fy), per-frame design procedure, section cover, and the
    EC2 design preferences (γc, γs, αcc); the concrete design code is set to
    Eurocode 2-2004

Import reads these back: LinStatic/NonStatic load cases become Linear /
NonLinear / GeometricNonlinear analysis cases (with their load coefficients),
frame springs become element springs (with their unilateral mode), and the
concrete design tables rebuild the concrete materials and RC design flags.

!!! note
    SAP joint (node) springs are linear/bilateral, so a node spring's
    tension-/compression-only mode is **not** exported; model unilateral node
    supports with a compression/tension-only element spring or a nonlinear
    link in SAP.

Units are kN, m, C (consistent with xdfem2D). This is a best-effort exporter —
open the file in SAP2000 (File > Import > SAP2000 .s2k Text File) and verify a
simple case (especially moment/temperature sign conventions).

GUI-free and testable; the GUI's "Export to SAP2000" action calls
:func:`save_s2k`.
"""
from __future__ import annotations

from pathlib import Path

_G = 9.80665  # gravity, for UnitMass = UnitWeight / g

# Unilateral spring mode <-> SAP frame-spring "SimpleType".
_SPRING_SIMPLE_TYPE = {
    'both':        "Tension and Compression",
    'compression': "Compression Only",
    'tension':     "Tension Only",
}
_SIMPLE_TYPE_TO_MODE = {v.lower(): k for k, v in _SPRING_SIMPLE_TYPE.items()}


def _num(v: float) -> str:
    """Format a number the way SAP2000 .s2k files do (dot decimals)."""
    if v == 0:
        return "0"
    return f"{v:.10g}"


def _q(name) -> str:
    return '"' + str(name) + '"'


def _row(**fields) -> str:
    return "   " + "   ".join(f"{k}={v}" for k, v in fields.items())


def _table(title: str, rows: list[str]) -> list[str]:
    if not rows:
        return []
    return [f'TABLE:  "{title}"', *rows, ""]


def _yn(flag: bool) -> str:
    return "Yes" if flag else "No"


def _sap_frame_shape(s):
    """Map a xdfem2D section's geometric *shape* to a SAP2000 frame shape.

    Returns ``(sap_shape, extra_fields)`` where *sap_shape* is the SAP
    ``Shape`` / section-type name (e.g. ``"Box/Tube"``, ``"I/Wide Flange"``)
    and *extra_fields* the wall/flange thickness fields SAP needs to rebuild
    the section. Returns ``(None, {})`` for generic sections (no real shape),
    which the caller exports as ``Shape=General`` with explicit properties.
    """
    val = getattr(getattr(s, "shape", None), "value",
                  getattr(s, "shape", None))
    tw = getattr(s, "tw", 0.0) or 0.0
    tf = getattr(s, "tf", 0.0) or 0.0
    if val == "Rectangular":
        return "Rectangular", {}
    if val == "Circular":
        return "Circle", {}
    if val == "Rectangular hollow":
        # RHS: uniform wall -> SAP flange (tf) and web (tw) walls both = tw.
        return "Box/Tube", {"tf": _num(tw), "tw": _num(tw)}
    if val == "Circular hollow":
        return "Pipe", {"tw": _num(tw)}
    if val == "I":
        return "I/Wide Flange", {
            "tf": _num(tf), "tw": _num(tw),
            "t2b": _num(getattr(s, "b", 0.0)), "tfb": _num(tf),
            "FilletRadius": "0"}
    if val == "T":
        return "Tee", {"tf": _num(tf), "tw": _num(tw)}
    return None, {}


def _xd_section_from_sap(shape, t2, t3, tw, tf, row):
    """Map a SAP2000 frame ``Shape`` back to a xdfem2D section.

    Returns ``(shape_value, b, h, tw, tf, area_override, inertia_override)``.
    Real shapes (I, box, pipe, …) become parametric sections defined by their
    wall/flange thicknesses; ``General`` (and anything unrecognised, e.g. a
    Section-Designer shape) keeps SAP's explicit Area / I33."""
    sl = str(shape or "").strip().lower()
    if "wide flange" in sl or sl in ("i", "i/wide flange"):
        return ("I", t2, t3, tw, tf, None, None)
    if "box" in sl or "tube" in sl:
        return ("Rectangular hollow", t2, t3, (tw or tf), 0.0, None, None)
    if "pipe" in sl:
        return ("Circular hollow", t3, t3, tw, 0.0, None, None)
    if sl in ("circle", "circular"):
        return ("Circular", t3, t3, 0.0, 0.0, None, None)
    if sl in ("rectangular", "solid rectangle"):
        return ("Rectangular", t2, t3, 0.0, 0.0, None, None)
    if sl in ("tee", "t"):
        return ("T", t2, t3, tw, tf, None, None)
    area = _f(row, "Area")
    i33 = _f(row, "I33")
    return ("Generic", t2, t3, 0.0, 0.0,
            area if area > 0 else None, i33 if i33 > 0 else None)


def _fit_linear_expr(pts) -> str:
    """Fit ``value ≈ a·x + b·y + c`` to *pts* = [(x, y, value), …] and return it
    as a field expression string (``"x"``, ``"2*x + 3"``, ``"-0.5*y + 1"``, …).

    SAP joint patterns are almost always defined by a coordinate gradient, so a
    linear fit reproduces them exactly; a genuinely non-linear pattern is
    approximated by its best-fit plane."""
    import numpy as _np
    xs = _np.array([p[0] for p in pts], float)
    ys = _np.array([p[1] for p in pts], float)
    vs = _np.array([p[2] for p in pts], float)
    A = _np.column_stack([xs, ys, _np.ones(len(pts))])
    try:
        (a, b, c), *_ = _np.linalg.lstsq(A, vs, rcond=None)
    except Exception:
        a = b = 0.0
        c = float(vs.mean()) if len(vs) else 0.0

    def _r(v):
        v = float(v)
        return 0.0 if abs(v) < 1e-9 else round(v, 6)

    a, b, c = _r(a), _r(b), _r(c)
    terms = []
    for coef, var in ((a, "x"), (b, "y")):
        if coef == 0.0:
            continue
        terms.append(var if coef == 1.0 else
                     (f"-{var}" if coef == -1.0 else f"{coef:g}*{var}"))
    expr = " + ".join(terms).replace("+ -", "- ")
    if c != 0.0 or not expr:
        expr = (f"{expr} + {c:g}" if expr and c > 0 else
                f"{expr} - {abs(c):g}" if expr else f"{c:g}")
    return expr


def _steel_fy_fu(mat) -> tuple[float, float]:
    """Return (fy, fu) in MPa for a steel material, from its design dict or,
    failing that, from a S<nnn> name (S275 -> 275) with an EN-ish ultimate."""
    d = getattr(mat, "design", {}) or {}
    fy = d.get("fy") or d.get("fyk")
    fu = d.get("fu")
    if not fy:
        m = _re.match(r"[Ss]\s*(\d{3})", getattr(mat, "name", "") or "")
        fy = float(m.group(1)) if m else 235.0
    fy = float(fy)
    fu = float(fu) if fu else fy + 155.0   # S235->360, S275->430, S355->510
    return fy, fu


def to_s2k(struc) -> str:
    """Return the SAP2000 ``.s2k`` text for *struc* (a Structure2D)."""
    # Geometry objects have no equivalent in .s2k — export the compiled mesh.
    if getattr(struc, "geometry_objects", None):
        from .geo_expand import expand_geometry
        struc = expand_geometry(struc)[0]
    lines: list[str] = []

    # Domain-aware orientation (dev/sap2000_domain_io.md): a plane (frame/wall)
    # model is written in SAP's vertical X–Z plane (Y=0, active UX/UZ/RY); a
    # plate (slab) model in the horizontal X–Y plane (Z=0, active UZ/RX/RY), so
    # the transverse deflection w is UZ and the two bending rotations are RX/RY.
    # The engine stores a plate node's DOFs in the ux/uy/tz slots aliased as
    # w/θx/θy (see add_support / add_point_load), so the exporter reads the same
    # three components and re-labels them by domain. The plane path below is
    # byte-for-byte unchanged.
    plate = getattr(struc, "domain", "plane") == "plate"

    # ── Used sections / materials ─────────────────────────────────────
    # Only export sections and materials that some element actually references,
    # so helper entries (e.g. Rigid / Dummy) are omitted when unused.
    used_sec_names = {e.section_name for e in struc.bar_elements}
    for te in getattr(struc, "tri_elements", []) or []:
        sn = getattr(te, "section_name", None)
        if sn:
            used_sec_names.add(sn)
    sections = {n: s for n, s in struc.sections.items() if n in used_sec_names}
    used_mat_names = {s.material_name for s in sections.values()}
    for te in getattr(struc, "tri_elements", []) or []:
        ts = getattr(struc, "tri_sections", {}).get(
            getattr(te, "section_name", None))
        if ts is not None and getattr(ts, "material_name", None):
            used_mat_names.add(ts.material_name)
    for qe in getattr(struc, "quad_elements", []) or []:
        qs = getattr(struc, "quad_sections", {}).get(
            getattr(qe, "section_name", None))
        if qs is not None and getattr(qs, "material_name", None):
            used_mat_names.add(qs.material_name)
    materials = {n: m for n, m in struc.materials.items()
                 if n in used_mat_names}

    # ── Concrete (EC2) design data ────────────────────────────────────
    # Resolve fck / fyk for each concrete material so RC design info can be
    # exported. A separate SAP "rebar" material is created per steel class.
    from .rc_design import _fck as _fck_of, _fyk as _fyk_of
    conc_design = {}     # material_name -> (fck_MPa, fyk_MPa, rebar_mat, cm)
    rebar_mats = {}      # rebar material name -> fyk_MPa
    for cm in getattr(struc, "concrete_materials", {}).values():
        if cm.material_name not in used_mat_names:
            continue
        try:
            fck = _fck_of(cm.concrete_class)
        except ValueError:
            continue
        try:
            fyk = _fyk_of(cm.steel_class)
        except ValueError:
            fyk = 500.0
        rebar = f"Rebar_{cm.steel_class}"
        rebar_mats[rebar] = fyk
        conc_design[cm.material_name] = (fck, fyk, rebar, cm)
    # A material is concrete exactly when ``section_type_of`` says so
    # (material_type == CONCRETE) — the same test the RC design uses — with the
    # strengths taken from its ``material.design`` dict (fck/fyk). This keeps the
    # SAP export consistent with the reinforcement design: any concrete material
    # is exported as a rectangular concrete section set up for RC design.
    from .models import section_type_of, SectionType
    for m in materials.values():
        if m.name in conc_design:
            continue
        if section_type_of(m) != SectionType.CONCRETE:
            continue
        d = getattr(m, "design", {}) or {}
        try:
            fck = float(d.get("fck")); fyk = float(d.get("fyk"))
        except (TypeError, ValueError):
            continue
        if fck <= 0 or fyk <= 0:
            continue
        rebar = f"Rebar_A{int(round(fyk))}"
        rebar_mats[rebar] = fyk
        conc_design[m.name] = (fck, fyk, rebar, None)   # no concrete_material obj
    has_conc = bool(conc_design)

    # ── Steel design materials ────────────────────────────────────────
    # A material is exported as a SAP steel-design material when it is typed
    # Steel, or when it carries a non-generic (real-shape) frame section and is
    # not already a concrete-design material. This lets SAP run EC3 steel design
    # on the exported profiles (Box/Tube, I/Wide Flange, …).
    steel_mats: dict = {}   # material name -> (fy_MPa, fu_MPa)
    _steel_shaped = set()   # material names used by a non-generic section
    for s in sections.values():
        if s.material_name in conc_design:
            continue
        sap_shape, _ = _sap_frame_shape(s)
        if sap_shape is not None:
            _steel_shaped.add(s.material_name)
    for m in materials.values():
        if m.name in conc_design:
            continue
        mt = getattr(getattr(m, "material_type", None), "value",
                     getattr(m, "material_type", None))
        if mt == "Steel" or (m.name in _steel_shaped
                             and mt not in ("Timber", "Concrete")):
            steel_mats[m.name] = _steel_fy_fu(m)
    has_steel = bool(steel_mats)

    # ── Program control ───────────────────────────────────────────────
    # Use the Eurocode 2 frame-design code when the model carries concrete
    # design data, so SAP runs RC design under EC2.
    conc_code = "Eurocode 2-2004" if has_conc else "ACI 318-19"
    steel_code = ("EN 1993-1-1:2005/A1:2014" if has_steel else "AISC 360-16")
    lines += _table("PROGRAM CONTROL", [_row(
        ProgramName="SAP2000", Version="26.3.0", CurrUnits=_q("KN, m, C"),
        SteelCode=_q(steel_code), ConcCode=_q(conc_code),
        ConcSCode=_q("Eurocode 2-2004"))])

    # Active DOFs identify the domain on re-import (from_s2k reads this table):
    #   plane  → UX/UZ/RY (in-plane translations + in-plane rotation), X–Z plane;
    #   plate  → UZ/RX/RY (transverse translation + two bending rotations), X–Y.
    if plate:
        lines += _table("ACTIVE DEGREES OF FREEDOM", [_row(
            UX="No", UY="No", UZ="Yes", RX="Yes", RY="Yes", RZ="No")])
    else:
        lines += _table("ACTIVE DEGREES OF FREEDOM", [_row(
            UX="Yes", UY="No", UZ="Yes", RX="No", RY="Yes", RZ="No")])

    # ── Materials ─────────────────────────────────────────────────────
    # The SAP material type follows the xdfem2D material_type (Concrete / Steel
    # / …), so the export is consistent with how the material is defined and
    # designed. SAP has no "Timber" type → mapped to "Other".
    _SAP_MAT_TYPE = {"Concrete": "Concrete", "Steel": "Steel",
                     "Timber": "Other", "Other": "Other"}
    mat_general, mat_mech = [], []
    for m in materials.values():
        nu = 0.2
        g12 = m.elastic_modulus / (2.0 * (1.0 + nu))
        mtype = _SAP_MAT_TYPE.get(
            getattr(getattr(m, "material_type", None), "value", "Concrete"),
            "Concrete")
        if m.name in steel_mats:
            mtype = "Steel"
        # Concrete grade (e.g. C30/37): from the concrete_material record or the
        # material.design 'class_conc'.
        grade = None
        if m.name in conc_design:
            cm = conc_design[m.name][3]
            grade = (getattr(cm, "concrete_class", None)
                     or (getattr(m, "design", {}) or {}).get("class_conc"))
        if grade:
            mat_general.append(_row(Material=_q(m.name), Type=mtype,
                                    Grade=_q(grade), SymType="Isotropic",
                                    TempDepend="No"))
        else:
            mat_general.append(_row(Material=_q(m.name), Type=mtype,
                                    SymType="Isotropic", TempDepend="No"))
        mat_mech.append(_row(
            Material=_q(m.name),
            UnitWeight=_num(m.unit_weight),
            UnitMass=_num(m.unit_weight / _G),
            E1=_num(m.elastic_modulus),
            G12=_num(g12),
            U12=_num(nu),
            A1=_num(m.alpha)))
    # Rebar materials (one per steel class referenced by concrete design).
    for rebar, fyk in rebar_mats.items():
        mat_general.append(_row(Material=_q(rebar), Type="Rebar",
                                SymType="Uniaxial", TempDepend="No"))
        Es = 200e6  # kN/m²  (EC2 reinforcement modulus)
        mat_mech.append(_row(
            Material=_q(rebar), UnitWeight=_num(78.5),
            UnitMass=_num(78.5 / _G), E1=_num(Es), G12="0", U12="0",
            A1=_num(1.2e-5)))
    lines += _table("MATERIAL PROPERTIES 01 - GENERAL", mat_general)
    lines += _table("MATERIAL PROPERTIES 02 - BASIC MECHANICAL PROPERTIES",
                    mat_mech)

    # ── Steel design strengths (EC3) ──────────────────────────────────
    # Fy / Fu in current units (kN/m²) = fy/fu [MPa] × 1000.
    if has_steel:
        steel_rows = [_row(
            Material=_q(name), Fy=_num(fy * 1e3), Fu=_num(fu * 1e3),
            EffFy=_num(fy * 1e3), EffFu=_num(fu * 1e3),
            SSCurveOpt="Simple", SSHysType="Kinematic",
            SHard="0.015", SMax="0.11", SRup="0.17", FinalSlope="-0.1",
            CoupModType=_q("Von Mises"))
            for name, (fy, fu) in steel_mats.items()]
        lines += _table("MATERIAL PROPERTIES 03A - STEEL DATA", steel_rows)

    # ── Concrete / rebar design strengths (EC2) ───────────────────────
    # Fc / Fy are in current units (kN/m²): fck/fyk [MPa] × 1000.
    if has_conc:
        conc_rows = [_row(Material=_q(name), Fc=_num(fck * 1e3),
                          eFc=_num(fck * 1e3), LtWtConc="No",
                          SSCurveOpt="Mander", SSHysType="Takeda",
                          SFc="0,00181818", SCap="0,005", FinalSlope="-0,1",
                          FAngle="0", DAngle="0",
                          CoupModType=_q("Modified Darwin-Pecknold"))
                     for name, (fck, _fyk, _rb, _cm) in conc_design.items()]
        lines += _table("MATERIAL PROPERTIES 03B - CONCRETE DATA", conc_rows)
        rebar_rows = [_row(Material=_q(rb), Fy=_num(fyk * 1e3),
                           Fu=_num(fyk * 1e3),
                           EffFy=_num(fyk * 1e3),
                           EffFu=_num(fyk * 1e3),
                           SSCurveOpt="Simple", SSHysType="Kinematic",
                           SHard="0,01", SCap="0,09", FinalSlope="-0,1",
                           UseCTDef="No", CoupModType=_q("Von Mises"))
                      for rb, fyk in rebar_mats.items()]
        lines += _table("MATERIAL PROPERTIES 03E - REBAR DATA", rebar_rows)

    # ── Frame sections ────────────────────────────────────────────────
    # Concrete-material sections are exported as rectangular concrete sections
    # (Shape=Rectangular, ConcBeam=Yes) so SAP runs RC design on them; SAP
    # recomputes the section properties from t2/t3. Other sections keep the
    # generic form (explicit Area + in-plane inertia, no shear deformation).
    sec_rows = []
    sec_shape = {}   # section name -> SAP SectionType for the assignments table
    for s in sections.values():
        if s.material_name in conc_design:
            sec_shape[s.name] = "Rectangular"
            sec_rows.append(_row(
                SectionName=_q(s.name), Material=_q(s.material_name),
                Shape="Rectangular", t3=_num(s.h), t2=_num(s.b),
                ConcBeam="Yes"))
            continue
        sap_shape, extra = _sap_frame_shape(s)
        if sap_shape is not None:
            # Real SAP shape (Box/Tube, I/Wide Flange, …): SAP recomputes the
            # section properties from t3/t2 and the wall/flange thicknesses.
            sec_shape[s.name] = sap_shape
            fields = {"SectionName": _q(s.name), "Material": _q(s.material_name),
                      "Shape": _q(sap_shape) if "/" in sap_shape else sap_shape,
                      "t3": _num(s.h), "t2": _num(s.b)}
            fields.update(extra)
            fields["ConcBeam"] = "No"
            sec_rows.append(_row(**fields))
        else:
            sec_shape[s.name] = "General"
            sec_rows.append(_row(
                SectionName=_q(s.name), Material=_q(s.material_name),
                Shape="General", t3=_num(s.h), t2=_num(s.b),
                Area=_num(s.area),
                TorsConst=_num(max(s.inertia * 1e-3, 1e-9)),
                I33=_num(s.inertia), I22=_num(s.inertia),
                AS2="0", AS3="0",   # 0 shear area -> Euler-Bernoulli
                ConcBeam="No"))
    lines += _table("FRAME SECTION PROPERTIES 01 - GENERAL", sec_rows)

    # ── Joint coordinates ─────────────────────────────────────────────
    # plane: (x, y) → X–Z plane (Y=0); plate: (x, y) → X–Y plane (Z=0).
    if plate:
        lines += _table("JOINT COORDINATES", [
            _row(Joint=_q(n.id), CoordSys="GLOBAL", CoordType="Cartesian",
                 XorR=_num(n.x), Y=_num(n.y), Z="0")
            for n in struc.nodes.values()])
    else:
        lines += _table("JOINT COORDINATES", [
            _row(Joint=_q(n.id), CoordSys="GLOBAL", CoordType="Cartesian",
                 XorR=_num(n.x), Y="0", Z=_num(n.y))
            for n in struc.nodes.values()])

    # ── Fields → SAP joint patterns ───────────────────────────────────
    # A xdfem2D field is an expression f(x, y); SAP's equivalent is a joint
    # pattern, a scalar value per joint. Each field is exported as a pattern
    # whose value at every joint is the field evaluated at that node.
    fields = getattr(struc, "fields", {}) or {}
    if fields:
        from .models import evaluate_field
        lines += _table("JOINT PATTERN DEFINITIONS",
                        [_row(Pattern=_q(name)) for name in fields])
        patt_rows = []
        for name, fld in fields.items():
            expr = getattr(fld, "expression", "0.0")
            for n in struc.nodes.values():
                patt_rows.append(_row(
                    Joint=_q(n.id), Pattern=_q(name),
                    Value=_num(evaluate_field(expr, n.x, n.y))))
        lines += _table("JOINT PATTERN ASSIGNMENTS", patt_rows)

    # ── Frame connectivity ────────────────────────────────────────────
    lines += _table("CONNECTIVITY - FRAME", [
        _row(Frame=_q(e.id), JointI=_q(e.node_i), JointJ=_q(e.node_j))
        for e in struc.bar_elements])

    # ── Frame section assignments ─────────────────────────────────────
    def _sectype(name):
        st = sec_shape.get(name, "General")
        return _q(st) if "/" in st else st
    lines += _table("FRAME SECTION ASSIGNMENTS", [
        _row(Frame=_q(e.id), SectionType=_sectype(e.section_name),
             AutoSelect="N.A.", AnalSect=_q(e.section_name),
             DesignSect=_q(e.section_name), MatProp="Default")
        for e in struc.bar_elements])

    # ── Area sections and elements (triangles AND quads) ──────────────
    # xdfem2D area elements → SAP thin-shell area objects. A triangle is a
    # 3-joint area, a quad a 4-joint one (dev/sap2000_domain_io.md Phase 2).
    # The section's thickness and material carry over; SAP recomputes the area
    # properties. Geometry-object meshes are already expanded above, so explicit
    # elements and object regions are both exported here.
    tri_elems = getattr(struc, "tri_elements", []) or []
    quad_elems = getattr(struc, "quad_elements", []) or []
    _tsecs = getattr(struc, "tri_sections", {}) or {}
    _qsecs = getattr(struc, "quad_sections", {}) or {}

    def _area_sec(name):
        """The section object for *name*, triangle table first then quad."""
        return _tsecs.get(name) or _qsecs.get(name)

    if tri_elems or quad_elems:
        # Section names used by either kind, in a stable order (triangles'
        # first, then quad-only names) — a paired "Panel" shares one name across
        # tri_sections and quad_sections, so it is emitted once (a SAP area
        # section is not tri/quad specific — thickness + material is all it
        # needs). Deterministic order keeps the export byte-stable.
        used_names, _seen = [], set()
        for e in list(tri_elems) + list(quad_elems):
            if e.section_name not in _seen:
                _seen.add(e.section_name); used_names.append(e.section_name)
        def _shell_type(name):
            """SAP shell type from the section formulation: MITC3/MITC4 are
            thick-plate (transverse shear) elements → "Shell-Thick"; DKT/DKT4
            are thin-plate (Kirchhoff) → "Shell-Thin". Membrane formulations
            (CST/QM6/…) keep the thin default."""
            forms = {getattr(s, "formulation", None)
                     for s in (_tsecs.get(name), _qsecs.get(name))
                     if s is not None}
            if forms & {"MITC3", "MITC4"}:
                return "Shell-Thick"
            return "Shell-Thin"

        asec_rows, adesign_rows = [], []
        for n in used_names:
            sec = _area_sec(n)
            if sec is None:
                continue
            asec_rows.append(_row(
                Section=_q(n), Material=_q(sec.material_name), MatAngle="0",
                AreaType="Shell", Type=_shell_type(n), DrillDOF="Yes",
                Thickness=_num(sec.thickness), BendThick=_num(sec.thickness)))
            adesign_rows.append(_row(
                Section=_q(n), RebarMat="None", RebarOpt="Default"))
        lines += _table("AREA SECTION PROPERTIES", asec_rows)
        lines += _table("AREA SECTION PROPERTY DESIGN PARAMETERS", adesign_rows)

        conn_rows, assign_rows, adproc_rows = [], [], []

        def _emit_area(aid, joints, section_name):
            conn = {"Area": _q(aid), "NumJoints": str(len(joints))}
            for k, j in enumerate(joints, start=1):
                conn[f"Joint{k}"] = _q(j)
            conn_rows.append(_row(**conn))
            assign_rows.append(_row(
                Area=_q(aid), Section=_q(section_name), MatProp="Default"))
            sec = _area_sec(section_name)
            mt = None
            if sec is not None:
                m = struc.materials.get(sec.material_name)
                mt = getattr(getattr(m, "material_type", None), "value", None)
            adproc_rows.append(_row(
                Area=_q(aid),
                DesignProc=_q("Concrete Shell Design" if mt == "Concrete"
                              else "No Design")))

        for t in tri_elems:
            _emit_area(t.id, (t.node_i, t.node_j, t.node_k), t.section_name)
        for q in quad_elems:
            _emit_area(q.id, (q.node_i, q.node_j, q.node_k, q.node_l),
                       q.section_name)
        lines += _table("CONNECTIVITY - AREA", conn_rows)
        lines += _table("AREA SECTION ASSIGNMENTS", assign_rows)
        lines += _table("AREA DESIGN PROCEDURES", adproc_rows)

    # ── Joint restraints (in-plane from supports, out-of-plane fixed) ─
    supp_by_node = {}
    for a in struc.support_assignments:
        sup = struc.supports.get(a.support_name)
        if sup is not None:
            supp_by_node[a.node_id] = sup
    # Only the in-plane DOFs need restraining; the out-of-plane ones (U2, R1,
    # R3) are inactive globally (see ACTIVE DEGREES OF FREEDOM above).
    # The three stored flags (ux/uy/tz slots) mean (ux, uy, θz) in the plane
    # domain and (w, θx, θy) in the plate domain — mapped to the active SAP DOFs:
    #   plane: ux→U1, uy→U3, θz→R2;   plate: w→U3, θx→R1, θy→R2.
    res_rows = []
    for n in struc.nodes.values():
        sup = supp_by_node.get(n.id)
        if sup is None:
            continue
        if plate:
            res_rows.append(_row(
                Joint=_q(n.id), U1="No", U2="No", U3=_yn(sup.ux),
                R1=_yn(sup.uy), R2=_yn(sup.tz), R3="No"))
        else:
            res_rows.append(_row(
                Joint=_q(n.id), U1=_yn(sup.ux), U2="No", U3=_yn(sup.uy),
                R1="No", R2=_yn(sup.tz), R3="No"))
    lines += _table("JOINT RESTRAINT ASSIGNMENTS", res_rows)

    # ── Joint (node) springs ──────────────────────────────────────────
    # kx/ky/kt slots mean (kx, ky, kθz) in plane and (kz, kθx, kθy) in plate:
    #   plane: kx→U1, ky→U3, kt→R2;   plate: kz→U3, kθx→R1, kθy→R2.
    spr_rows = []
    for sp in struc.node_springs.values():
        if plate:
            spr_rows.append(_row(
                Joint=_q(sp.node_id), CoordSys="GLOBAL",
                U1="0", U2="0", U3=_num(sp.kx),
                R1=_num(sp.ky), R2=_num(sp.kt), R3="0"))
        else:
            spr_rows.append(_row(
                Joint=_q(sp.node_id), CoordSys="GLOBAL",
                U1=_num(sp.kx), U2="0", U3=_num(sp.ky),
                R1="0", R2=_num(sp.kt), R3="0"))
    lines += _table("JOINT SPRING ASSIGNMENTS 1 - UNCOUPLED", spr_rows)

    # ── Frame (element / Winkler) line springs ───────────────────────
    # SAP frame springs act along the element's local axes (Dir = 1 axial,
    # 2 in-plane transverse). xdfem2D local springs map directly; global
    # springs are mapped to the local axis for axis-aligned elements
    # (X->1, Y->2 for horizontal members; swapped for vertical).
    el_spr_rows = []
    for es in struc.element_springs.values():
        if es.coord_sys == "local":
            dir_x, dir_y = "1", "2"
        else:
            elem = struc.bar_elements_by_id.get(es.element_id)
            horiz = True
            if elem is not None:
                ni = struc.nodes.get(elem.node_i)
                nj = struc.nodes.get(elem.node_j)
                if ni is not None and nj is not None:
                    horiz = abs(nj.x - ni.x) >= abs(nj.y - ni.y)
            dir_x, dir_y = ("1", "2") if horiz else ("2", "1")
        for k, d, mode in ((es.kx, dir_x, getattr(es, 'mode_x', 'both')),
                           (es.ky, dir_y, getattr(es, 'mode_y', 'both'))):
            if k:
                el_spr_rows.append(_row(
                    Frame=_q(es.element_id), Type="Simple", Stiffness=_num(k),
                    SimpleType=_q(_SPRING_SIMPLE_TYPE[mode]),
                    Dir1Type=_q("Object Axes"), Dir=d))
    lines += _table("FRAME SPRING ASSIGNMENTS", el_spr_rows)

    # ── Frame moment releases (hinges) ────────────────────────────────
    rel_rows = [_row(Frame=_q(e.id), M3I=_yn(e.hinge_i), M3J=_yn(e.hinge_j))
                for e in struc.bar_elements if e.hinge_i or e.hinge_j]
    lines += _table("FRAME RELEASE ASSIGNMENTS 1 - GENERAL", rel_rows)

    # ── Per-frame design procedure (Concrete / Steel / No Design) ─────
    if has_conc or has_steel:
        proc_rows = []
        for e in struc.bar_elements:
            sec = struc.sections.get(e.section_name)
            is_conc = bool(getattr(e, "rc_design", False)) or (
                sec is not None and sec.material_name in conc_design)
            is_steel = sec is not None and sec.material_name in steel_mats
            proc = "Concrete" if is_conc else ("Steel" if is_steel
                                               else "No Design")
            proc_rows.append(_row(Frame=_q(e.id), DesignProc=_q(proc)))
        lines += _table("FRAME DESIGN PROCEDURES", proc_rows)

    # ── Concrete (EC2) design assignments ─────────────────────────────
    if has_conc:
        # Concrete section rebar data: cover and reinforcing material (as a
        # beam section). Table name/fields match SAP's own .s2k output.
        crebar_rows = []
        for sname, sec in sections.items():
            if sec.material_name not in conc_design:
                continue
            rebar = conc_design[sec.material_name][2]
            cover = _num(getattr(sec, "rc_cover", 0.045))
            crebar_rows.append(_row(
                SectionName=_q(sname), RebarMatL=_q(rebar), RebarMatC=_q(rebar),
                TopCover=cover, BotCover=cover,
                TopLeftArea="0", TopRghtArea="0",
                BotLeftArea="0", BotRghtArea="0"))
        lines += _table("FRAME SECTION PROPERTIES 03 - CONCRETE BEAM",
                        crebar_rows)

        # EC2 design preferences (partial safety factors, αcc). Use the first
        # concrete material's factors, or EC2 defaults when the material only
        # carries a design dict (no concrete_material record → cm is None).
        cm0 = next(iter(conc_design.values()))[3]
        lines += _table("PREFERENCES - CONCRETE DESIGN - EUROCODE 2-2004", [_row(
            THETA0="0.01",
            GammaS=_num(getattr(cm0, "gamma_s", 1.15)),
            GammaC=_num(getattr(cm0, "gamma_c", 1.5)),
            AlphaCC=_num(getattr(cm0, "alpha_cc", 1.0)), AlphaCT="1",
            Combos=_q("Auto"), NumCurves="24", NumPoints="11")])

    # ── Load patterns (xdfem2D load cases) ────────────────────────────
    lines += _table("LOAD PATTERN DEFINITIONS", [
        _row(LoadPat=_q(lc.id), DesignType="Other",
             SelfWtMult=_num(getattr(lc, "self_weight_factor", 0.0) or 0.0))
        for lc in struc.load_cases])

    # ── Analysis cases -> SAP load cases ──────────────────────────────
    # A SAP LinStatic load case is emitted for EVERY analysis case, including the
    # load-case twins (one linear case per load pattern). On .s2k import SAP does
    # NOT auto-create a load case per pattern, so combinations that reference a
    # pattern name would otherwise fail with "Case name not recognized". Emitting
    # the twins explicitly makes those combinations resolve (any "already exists"
    # message on re-import is harmless).
    case_defs, static_assign, modal_rows = [], [], []
    nl_param_rows, nl_app_rows = [], []
    rs_general, rs_assign = [], []          # response-spectrum cases
    _DIR_TO_SAP = {"X": ["U1"], "Y": ["U3"], "XY": ["U1", "U3"]}
    first_node = next(iter(struc.nodes), None)

    for ac in getattr(struc, "analysis_cases", []):
        at = ac.analysis_type
        if at in ("Linear", "NonLinear", "GeometricNonlinear"):
            # NonLinear (unilateral springs) and GeometricNonlinear (P-Delta)
            # are both nonlinear static cases in SAP; they differ in the
            # geometric-nonlinearity setting (NonLinear = None; the nonlinearity
            # comes from the tension-/compression-only springs).
            is_nl = at in ("NonLinear", "GeometricNonlinear")
            case_defs.append(_row(
                Case=_q(ac.id), Type=("NonStatic" if is_nl else "LinStatic"),
                InitialCond="Zero", DesignType="Other", RunCase="Yes",
                CaseStatus=_q("Not Run")))
            for pat, sf in ac.coefficients.items():
                static_assign.append(_row(
                    Case=_q(ac.id), LoadType=_q("Load pattern"),
                    LoadName=_q(pat), LoadSF=_num(sf)))
            if is_nl:
                geo = "P-Delta" if at == "GeometricNonlinear" else "None"
                nl_param_rows.append(_row(
                    Case=_q(ac.id), GeoNonLin=geo,
                    ResultsSave=_q("Final State"),
                    SolScheme=_q("Iterative Events"),
                    MaxTotal="200", MaxNull="50"))
                if first_node is not None:
                    nl_app_rows.append(_row(
                        Case=_q(ac.id), LoadApp=_q("Full Load"),
                        MonitorDOF="U1", MonitorJt=_q(first_node)))
        elif at == "Modal":
            case_defs.append(_row(
                Case=_q(ac.id), Type="LinModal", InitialCond="Zero",
                DesignType="Other", RunCase="Yes", CaseStatus=_q("Not Run")))
            modal_rows.append(_row(
                Case=_q(ac.id), ModeType="Eigen",
                MaxNumModes=_num(getattr(ac, "num_modes", 12)),
                MinNumModes="1"))
        elif at == "Spectrum":
            case_defs.append(_row(
                Case=_q(ac.id), Type="LinRespSpec",
                ModalCase=_q(ac.modal_case_id), DesignType="Quake",
                RunCase="Yes", CaseStatus=_q("Not Run")))
            rs_general.append(_row(
                Case=_q(ac.id),
                ModalCombo=("CQC" if ac.combination_rule == "CQC" else "SRSS"),
                DirCombo="SRSS", MotionType="Acceleration",
                DampingType="Constant", ConstDamp=_num(ac.damping)))
            for u in _DIR_TO_SAP.get(ac.direction, ["U1"]):
                rs_assign.append(_row(
                    Case=_q(ac.id), LoadType="Acceleration", LoadName=u,
                    CoordSys="GLOBAL", Function=_q(ac.spectrum_id),
                    Angle="0", TransAccSF="1"))
        # 'Mass' cases are represented via the MASS SOURCE table below.

    lines += _table("LOAD CASE DEFINITIONS", case_defs)
    lines += _table("CASE - STATIC 1 - LOAD ASSIGNMENTS", static_assign)
    lines += _table("CASE - STATIC 2 - NONLINEAR LOAD APPLICATION", nl_app_rows)
    lines += _table("CASE - STATIC 4 - NONLINEAR PARAMETERS", nl_param_rows)
    lines += _table("CASE - MODAL 1 - GENERAL", modal_rows)
    lines += _table("CASE - RESPONSE SPECTRUM 1 - GENERAL", rs_general)
    lines += _table("CASE - RESPONSE SPECTRUM 2 - LOAD ASSIGNMENTS", rs_assign)

    # ── Response-spectrum functions (period/accel pairs) ──────────────
    func_rows = []
    for sf in getattr(struc, "spectral_functions", {}).values():
        for i, pt in enumerate(sf.points):
            T, Sa = pt[0], pt[1]
            if i == 0:
                func_rows.append(_row(Name=_q(sf.id), Period=_num(T),
                                      Accel=_num(Sa), FuncDamp=_num(sf.damping)))
            else:
                func_rows.append(_row(Name=_q(sf.id), Period=_num(T),
                                      Accel=_num(Sa)))
    lines += _table("FUNCTION - RESPONSE SPECTRUM - USER", func_rows)

    # ── Mass source (from Mass analysis cases: loads -> mass) ─────────
    acases = list(getattr(struc, "analysis_cases", []))
    mass_assign = [(lc, f) for ac in acases if ac.analysis_type == "Mass"
                   for lc, f in ac.coefficients.items()]
    needs_mass = any(ac.analysis_type in ("Mass", "Modal", "Spectrum")
                     for ac in acases)
    if needs_mass:
        mass_rows = []
        if mass_assign:
            for i, (lc, f) in enumerate(mass_assign):
                if i == 0:
                    mass_rows.append(_row(
                        MassSource="MSSSRC1", Elements="No", Masses="No",
                        Loads="Yes", IsDefault="Yes",
                        LoadPat=_q(lc), Multiplier=_num(f)))
                else:
                    mass_rows.append(_row(
                        MassSource="MSSSRC1", LoadPat=_q(lc), Multiplier=_num(f)))
        else:
            mass_rows.append(_row(
                MassSource="MSSSRC1", Elements="Yes", Masses="Yes",
                Loads="No", IsDefault="Yes"))
        lines += _table("MASS SOURCE", mass_rows)

    # ── Joint loads - force ────────────────────────────────────────────
    # fx/fy/mz slots mean (Fx, Fy, Mz) in plane and (Fz, Mx, My) in plate:
    #   plane: fx→F1, fy→F3, mz→M2 (=-mz, X–Z sign);  plate: fz→F3, mx→M1, my→M2.
    if plate:
        lines += _table("JOINT LOADS - FORCE", [
            _row(Joint=_q(pl.node_id), LoadPat=_q(pl.load_case_id),
                 CoordSys="GLOBAL", F1="0", F2="0", F3=_num(pl.fx),
                 M1=_num(pl.fy), M2=_num(pl.mz), M3="0")
            for pl in struc.point_loads])
    else:
        lines += _table("JOINT LOADS - FORCE", [
            _row(Joint=_q(pl.node_id), LoadPat=_q(pl.load_case_id),
                 CoordSys="GLOBAL", F1=_num(pl.fx), F2="0", F3=_num(pl.fy),
                 M1="0", M2=_num(-pl.mz), M3="0")
            for pl in struc.point_loads])

    # ── Joint ground displacements (support settlements) ──────────────
    if plate:
        lines += _table("JOINT LOADS - GROUND DISPLACEMENT", [
            _row(Joint=_q(ss.node_id), LoadPat=_q(ss.load_case_id),
                 CoordSys="GLOBAL", U1="0", U2="0", U3=_num(ss.ux),
                 R1=_num(ss.uy), R2=_num(ss.tz), R3="0")
            for ss in getattr(struc, "support_settlements", [])])
    else:
        lines += _table("JOINT LOADS - GROUND DISPLACEMENT", [
            _row(Joint=_q(ss.node_id), LoadPat=_q(ss.load_case_id),
                 CoordSys="GLOBAL", U1=_num(ss.ux), U2="0", U3=_num(ss.uy),
                 R1="0", R2=_num(-ss.tz), R3="0")
            for ss in getattr(struc, "support_settlements", [])])

    # ── Frame distributed loads (global X/Y -> X/Z, or local 1/2) ─────
    dl_rows = []
    for dl in struc.distributed_loads:
        local = dl.coord_sys == "local"
        csys = "Local" if local else "GLOBAL"
        if dl.fxe or dl.fxd:
            dl_rows.append(_row(
                Frame=_q(dl.element_id), LoadPat=_q(dl.load_case_id),
                CoordSys=csys, Type="Force", Dir=("1" if local else "X"),
                DistType="RelDist", RelDistA="0", RelDistB="1",
                FOverLA=_num(dl.fxe), FOverLB=_num(dl.fxd)))
        if dl.fye or dl.fyd:
            dl_rows.append(_row(
                Frame=_q(dl.element_id), LoadPat=_q(dl.load_case_id),
                CoordSys=csys, Type="Force", Dir=("2" if local else "Z"),
                DistType="RelDist", RelDistA="0", RelDistB="1",
                FOverLA=_num(dl.fye), FOverLB=_num(dl.fyd)))
    lines += _table("FRAME LOADS - DISTRIBUTED", dl_rows)

    # ── Frame point loads (concentrated load at relative distance a/L) ─
    pt_rows = []
    for epl in getattr(struc, "element_point_loads", []):
        elem = struc.bar_elements_by_id.get(epl.element_id)
        if elem is None:
            continue
        ni = struc.nodes.get(elem.node_i); nj = struc.nodes.get(elem.node_j)
        rel = "0"
        if ni is not None and nj is not None:
            import math as _math
            L = _math.hypot(nj.x - ni.x, nj.y - ni.y)
            rel = _num(min(max(epl.a / L, 0.0), 1.0)) if L > 0 else "0"
        local = epl.coord_sys == "local"
        csys = "Local" if local else "GLOBAL"
        if epl.fx:
            pt_rows.append(_row(
                Frame=_q(epl.element_id), LoadPat=_q(epl.load_case_id),
                CoordSys=csys, Type="Force", Dir=("1" if local else "X"),
                DistType="RelDist", RelDist=rel, Force=_num(epl.fx)))
        if epl.fy:
            pt_rows.append(_row(
                Frame=_q(epl.element_id), LoadPat=_q(epl.load_case_id),
                CoordSys=csys, Type="Force", Dir=("2" if local else "Z"),
                DistType="RelDist", RelDist=rel, Force=_num(epl.fy)))
        if epl.mz:
            pt_rows.append(_row(
                Frame=_q(epl.element_id), LoadPat=_q(epl.load_case_id),
                CoordSys=csys, Type="Moment", Dir=("3" if local else "Y"),
                DistType="RelDist", RelDist=rel, Moment=_num(-epl.mz)))
    lines += _table("FRAME LOADS - POINT", pt_rows)

    # ── Frame temperature loads ───────────────────────────────────────
    tmp_rows = []
    for tl in getattr(struc, "temperature_loads", []):
        if getattr(tl, "delta_t_uniform", 0.0):
            tmp_rows.append(_row(
                Frame=_q(tl.element_id), LoadPat=_q(tl.load_case_id),
                Type="Temperature", Temp=_num(tl.delta_t_uniform),
                JtPattern="None"))
        if getattr(tl, "delta_t_gradient", 0.0):
            tmp_rows.append(_row(
                Frame=_q(tl.element_id), LoadPat=_q(tl.load_case_id),
                Type="Gradient3", TempGrad3=_num(tl.delta_t_gradient),
                JtPattern="None"))
    lines += _table("FRAME LOADS - TEMPERATURE", tmp_rows)

    # ── Area temperature loads ────────────────────────────────────────
    # A CST only responds to the mean of its three nodal ΔT, so each triangle
    # is written as one uniform area temperature (JtPattern=None). A field
    # temperature is already resolved into per-triangle values by the mesh
    # expansion, so its spatial variation is carried by the per-area values.
    atmp_rows = []
    for tl in getattr(struc, "tri_temperature_loads", []) or []:
        atmp_rows.append(_row(
            Area=_q(tl.tri_id), LoadPat=_q(tl.load_case_id),
            Type="Temperature", Temp=_num(tl.dt_mean), JtPattern="None"))
    for tl in getattr(struc, "quad_temperature_loads", []) or []:
        atmp_rows.append(_row(
            Area=_q(tl.quad_id), LoadPat=_q(tl.load_case_id),
            Type="Temperature", Temp=_num(tl.dt_mean), JtPattern="None"))
    lines += _table("AREA LOADS - TEMPERATURE", atmp_rows)

    # ── Load combinations (linear add; reference the per-load-case cases)
    cb_rows = []
    for combo in struc.load_combinations:
        if combo.combo_type != "LinearSum":
            continue  # SAP linear-add only
        # Combinations reference analysis cases (the load-case twin sharing its
        # id, or explicit analysis cases); those are exported as SAP load cases
        # above. Combinations referencing other combinations are flattened to
        # their underlying analysis cases.
        # Concrete design usage of this combo: ULS → Strength, SLS → Service,
        # any other → Strength (all treated as design combos). 'None' when the
        # model has no concrete design.
        cid_up = str(combo.id).upper()
        if not has_conc:
            conc_dsgn = "None"
        elif cid_up.startswith("SLS"):
            conc_dsgn = "Service"
        else:
            conc_dsgn = "Strength"
        first = True
        for case_id, factor in struc.expand_combination_coefficients(combo).items():
            if first:
                cb_rows.append(_row(
                    ComboName=_q(combo.id), ComboType=_q("Linear Add"),
                    AutoDesign="No", CaseType=_q("Linear Static"),
                    CaseName=_q(case_id), ScaleFactor=_num(factor),
                    SteelDesign="None", ConcDesign=conc_dsgn,
                    AlumDesign="None", ColdDesign="None"))
                first = False
            else:
                cb_rows.append(_row(
                    ComboName=_q(combo.id), CaseType=_q("Linear Static"),
                    CaseName=_q(case_id), ScaleFactor=_num(factor)))
    lines += _table("COMBINATION DEFINITIONS", cb_rows)

    # Turn OFF SAP's auto-generation of concrete design combinations, so the
    # explicit combos above (marked ConcDesign=Strength/Service) are the ones
    # used for RC design instead of SAP's own auto combos.
    if has_conc:
        lines += _table("AUTO COMBINATION OPTION DATA 01 - GENERAL",
                        [_row(DesignType="Concrete", AutoGen="No")])

    # ── Nodal masses (JOINT ADDED MASS ASSIGNMENTS) ───────────────────
    # xdfem2D: mx [t], my [t], mtz [t·m²]
    # SAP2000 X-Z plane: mx -> U1, my -> U3, mtz -> R2; U2, R1, R3 = 0
    nm_rows = []
    for nm in getattr(struc, "nodal_masses", []):
        nm_rows.append(_row(
            Joint=_q(nm.node_id), CoordSys="GLOBAL",
            U1=_num(nm.mx), U2="0", U3=_num(nm.my),
            R1="0", R2=_num(nm.mtz), R3="0"))
    lines += _table("JOINT ADDED MASS ASSIGNMENTS", nm_rows)

    lines.append("END TABLE DATA")
    return "\n".join(lines) + "\n"


def save_s2k(struc, path: str | Path):
    """Write *struc* to a SAP2000 ``.s2k`` text file at *path*."""
    Path(path).write_text(to_s2k(struc), encoding="utf-8")


# ---------------------------------------------------------------------------
# Import (read a SAP2000 .s2k text file into a Structure2D)
# ---------------------------------------------------------------------------

import re as _re


def _parse_s2k_tables(text: str) -> dict:
    """Parse a .s2k file into {table_title: [ {field: value}, ... ]}.

    The .s2k format is line-based: a ``TABLE:  "Title"`` header followed by data
    rows of ``Key=Value`` tokens (values may be quoted and contain spaces).
    """
    tables: dict[str, list] = {}
    cur = None
    tok = _re.compile(r'(\w+)=("[^"]*"|\S+)')
    # SAP2000 wraps long data rows, ending each continued physical line with a
    # trailing " _"; join them back into one logical line before parsing.
    logical: list[str] = []
    pending = ""
    for phys in text.splitlines():
        chunk = phys.rstrip()
        if chunk.endswith(" _") or chunk == "_":
            pending += chunk[:-1] + " "
            continue
        logical.append(pending + phys)
        pending = ""
    if pending:
        logical.append(pending)
    for raw in logical:
        s = raw.strip()
        if not s:
            continue
        if s.upper().startswith("END TABLE DATA"):
            break
        if s.startswith("TABLE:"):
            m = _re.match(r'TABLE:\s*"([^"]*)"', s)
            cur = m.group(1) if m else None
            if cur:
                tables.setdefault(cur, [])
            continue
        if cur is None:
            continue
        row = {}
        for k, v in tok.findall(s):
            row[k] = v[1:-1] if v.startswith('"') else v
        if row:
            tables[cur].append(row)
    return tables


def _f(row, key, default=0.0):
    v = row.get(key, default)
    try:
        return float(v)
    except (TypeError, ValueError):
        # SAP2000 in a comma-decimal locale writes numbers like "6,25".
        try:
            return float(str(v).replace(",", "."))
        except (TypeError, ValueError):
            return default


def _yn_true(v) -> bool:
    return str(v).strip().lower() in ("yes", "true", "1")


class AmbiguousDomainError(ValueError):
    """Raised by :func:`from_s2k` when the SAP model's ACTIVE DEGREES OF FREEDOM
    do not identify plane vs plate (e.g. a full 3-D shell with all six DOFs, or
    the table absent). The caller must pass an explicit ``domain=``."""


def _infer_domain(T: dict):
    """'plane' | 'plate' | None from the ACTIVE DEGREES OF FREEDOM table.

    plane ⇒ UX/UZ/RY active (in-plane), UY/RX/RZ off.
    plate ⇒ UZ/RX/RY active (out-of-plane bending), UX/UY/RZ off.
    Anything else (all six active = a full shell, or the table missing) ⇒ None.
    """
    rows = T.get("ACTIVE DEGREES OF FREEDOM", [])
    if not rows:
        return None
    r = rows[0]
    on = {d: str(r.get(d, "")).strip().lower() in ("yes", "true", "1")
          for d in ("UX", "UY", "UZ", "RX", "RY", "RZ")}
    active = frozenset(d for d, v in on.items() if v)
    if active == frozenset({"UX", "UZ", "RY"}):
        return "plane"
    if active == frozenset({"UZ", "RX", "RY"}):
        return "plate"
    return None


def from_s2k(text: str, domain: str = None):
    """Build a :class:`Structure2D` from SAP2000 ``.s2k`` text.

    The domain (plane vs plate) is taken from *domain* if given, else inferred
    from the ACTIVE DEGREES OF FREEDOM table (see :func:`_infer_domain`); if
    neither identifies it, :class:`AmbiguousDomainError` is raised so the caller
    (e.g. the GUI) can ask. Joints → nodes (plane: X→x, Z→y; plate: X→x, Y→y),
    frames → bar elements, 3-joint areas → triangles and 4-joint areas → quads,
    sections, materials, restraints → supports, joint springs, load patterns →
    load cases (with a matching Linear analysis case), joint and frame loads, and
    linear load combinations, all mapped to the resolved domain's DOFs.
    """
    from .structure import Structure2D

    T = _parse_s2k_tables(text)
    resolved = domain or _infer_domain(T)
    if resolved not in ("plane", "plate"):
        raise AmbiguousDomainError(
            "SAP model's ACTIVE DEGREES OF FREEDOM do not identify plane vs "
            "plate — pass domain='plane' or domain='plate'.")
    plate = resolved == "plate"
    s = Structure2D(domain=resolved)

    # ── Materials ──────────────────────────────────────────────
    mech = {r.get("Material"): r
            for r in T.get("MATERIAL PROPERTIES 02 - BASIC MECHANICAL PROPERTIES", [])}
    mat_types = {r.get("Material"): r.get("Type", "Concrete")
                 for r in T.get("MATERIAL PROPERTIES 01 - GENERAL", [])}
    for name, r in mech.items():
        if not name:
            continue
        mt = mat_types.get(name, "Concrete")
        mt = mt if mt in ("Concrete", "Steel", "Timber") else "Other"
        s.add_material(name, _f(r, "E1", 30e6), _f(r, "UnitWeight", 0.0),
                       alpha=_f(r, "A1", 1.0e-5), material_type=mt)
    if not s.materials:
        s.add_material("MAT", 30e6, 25.0, material_type="Concrete")
    _default_mat = next(iter(s.materials))

    # ── Sections (map the SAP Shape → parametric I / box / pipe / …) ──
    for r in T.get("FRAME SECTION PROPERTIES 01 - GENERAL", []):
        name = r.get("SectionName")
        if not name:
            continue
        mat = r.get("Material") if r.get("Material") in s.materials else _default_mat
        kind, b, h, sw, sf, ao, io = _xd_section_from_sap(
            r.get("Shape"), _f(r, "t2", 0.3), _f(r, "t3", 0.5),
            _f(r, "tw"), _f(r, "tf"), r)
        s.add_section(name, mat, b, h, shape=kind, tw=sw, tf=sf,
                      area_override=ao, inertia_override=io)
    if not s.sections:
        s.add_section("SEC", _default_mat, 0.3, 0.5)
    _default_sec = next(iter(s.sections))

    # ── Nodes (plane: X→x, Z→y; plate: X→x, Y→y) ───────────────
    for r in T.get("JOINT COORDINATES", []):
        jid = r.get("Joint")
        if jid is None:
            continue
        x = _f(r, "XorR", _f(r, "X"))
        y = _f(r, "Y") if plate else _f(r, "Z")
        s.add_node(str(jid), x, y)

    # ── Elements (connectivity + section assignment) ───────────
    sec_of = {r.get("Frame"): r.get("AnalSect")
              for r in T.get("FRAME SECTION ASSIGNMENTS", [])}
    for r in T.get("CONNECTIVITY - FRAME", []):
        fid = r.get("Frame")
        ni, nj = r.get("JointI"), r.get("JointJ")
        if fid is None or ni is None or nj is None:
            continue
        sec = sec_of.get(fid)
        sec = sec if sec in s.sections else _default_sec
        try:
            s.add_bar_element(str(fid), str(ni), str(nj), sec)
        except (ValueError, KeyError):
            continue

    # ── Area sections → paired tri + quad sections (by domain) ─────────
    # Each SAP area section becomes both a triangle section and a quad section
    # under the same name, so a 3-joint area can use the triangle and a 4-joint
    # area the quad. In the plane domain the formulation is ES-FEM / QM6. In the
    # plate domain it follows the SAP shell type (inverse of the export map):
    # "Shell-Thick" ⇒ MITC3 / MITC4 (thick plate), "Shell-Thin" ⇒ DKT / DKT4
    # (thin plate); an unspecified type defaults to the thick MITC elements.
    def _plate_forms(shell_type):
        t = str(shell_type or "").lower()
        if "thin" in t:
            return "DKT", "DKT4"
        return "MITC3", "MITC4"

    for r in T.get("AREA SECTION PROPERTIES", []):
        name = r.get("Section")
        if not name:
            continue
        mat = r.get("Material") if r.get("Material") in s.materials else _default_mat
        thk = _f(r, "Thickness", 0.1)
        shell_type = r.get("Type", "")
        plane_strain = "strain" in str(shell_type).lower()
        if plate:
            _tri_form, _quad_form = _plate_forms(shell_type)
        else:
            _tri_form, _quad_form = "ES-FEM", "QM6"
        if str(name) not in s.tri_sections:
            try:
                s.add_tri_section(str(name), mat, thickness=thk,
                                  plane_strain=plane_strain,
                                  formulation=_tri_form)
            except (ValueError, KeyError):
                pass
        if str(name) not in s.quad_sections:
            try:
                s.add_quad_section(str(name), mat, thickness=thk,
                                   plane_strain=plane_strain,
                                   formulation=_quad_form)
            except (ValueError, KeyError):
                pass
    _default_tsec = next(iter(s.tri_sections), None)
    _default_qsec = next(iter(s.quad_sections), None)

    # ── Area connectivity → triangle / quad elements ──────────────────
    # A 3-joint area → a triangle; a 4-joint area → a real quad element
    # (dev/sap2000_domain_io.md Phase 3, replacing the old two-triangle split);
    # a 5+-joint polygon (rare) is still fanned from its first vertex into
    # (n-2) triangles.
    area_sec_of = {r.get("Area"): r.get("Section")
                   for r in T.get("AREA SECTION ASSIGNMENTS", [])}
    area_tris: dict = {}     # SAP area id -> [created area element ids]
    for r in T.get("CONNECTIVITY - AREA", []):
        aid = r.get("Area")
        if aid is None:
            continue
        nj = int(_f(r, "NumJoints", 3))
        js = [r.get(f"Joint{k}") for k in range(1, nj + 1)]
        js = [str(j) for j in js if j is not None and str(j) in s.nodes]
        sec = area_sec_of.get(aid)
        if len(js) == 4:
            qsec = sec if sec in s.quad_sections else _default_qsec
            if qsec is None:
                continue
            try:
                s.add_quad_element(str(aid), js[0], js[1], js[2], js[3], qsec)
                area_tris[str(aid)] = [str(aid)]
            except (ValueError, KeyError):
                continue
            continue
        tsec = sec if sec in s.tri_sections else _default_tsec
        if tsec is None or len(js) < 3:
            continue
        faces = [(0, i, i + 1) for i in range(1, len(js) - 1)]
        tri_ids = []
        for n, (a, b, c) in enumerate(faces):
            tid = str(aid) if n == 0 else f"{aid}_{n + 1}"
            try:
                s.add_tri_element(tid, js[a], js[b], js[c], tsec)
                tri_ids.append(tid)
            except (ValueError, KeyError):
                continue
        if tri_ids:
            area_tris[str(aid)] = tri_ids

    # ── Frame moment releases (hinges): M3I→hinge_i, M3J→hinge_j ─
    for r in T.get("FRAME RELEASE ASSIGNMENTS 1 - GENERAL",
                   T.get("FRAME RELEASE ASSIGNMENTS", [])):
        fid = r.get("Frame")
        if fid is None or str(fid) not in s.bar_elements_by_id:
            continue
        el = s.bar_elements_by_id[str(fid)]
        if _yn_true(r.get("M3I")):
            el.hinge_i = True
        if _yn_true(r.get("M3J")):
            el.hinge_j = True

    # ── Restraints → supports ──────────────────────────────────
    # plane: U1→ux, U3→uy, R2→θz;  plate: U3→w, R1→θx, R2→θy
    #   (all stored in the same ux/uy/tz slots, re-labelled by domain).
    for r in T.get("JOINT RESTRAINT ASSIGNMENTS", []):
        jid = r.get("Joint")
        if jid is None or str(jid) not in s.nodes:
            continue
        if plate:
            a, b, c = (_yn_true(r.get("U3")), _yn_true(r.get("R1")),
                       _yn_true(r.get("R2")))
        else:
            a, b, c = (_yn_true(r.get("U1")), _yn_true(r.get("U3")),
                       _yn_true(r.get("R2")))
        if not (a or b or c):
            continue
        sup_name = f"S_{int(a)}{int(b)}{int(c)}"
        if sup_name not in s.supports:
            s.add_support(sup_name, ux=a, uy=b, tz=c)
        s.assign_support(str(jid), sup_name)

    # ── Joint springs ──────────────────────────────────────────
    # plane: U1→kx, U3→ky, R2→kθz;  plate: U3→kz, R1→kθx, R2→kθy.
    for r in T.get("JOINT SPRING ASSIGNMENTS 1 - UNCOUPLED", []):
        jid = r.get("Joint")
        if jid is None or str(jid) not in s.nodes:
            continue
        if plate:
            s.add_node_spring(str(jid), kx=_f(r, "U3"), ky=_f(r, "R1"),
                              kt=_f(r, "R2"))
        else:
            s.add_node_spring(str(jid), kx=_f(r, "U1"), ky=_f(r, "U3"),
                              kt=_f(r, "R2"))

    # ── Frame (Winkler) line springs → element springs ────────────
    # SAP frame springs are along the element local axes (Dir 1 axial, 2
    # transverse). SimpleType carries the unilateral behaviour.
    es_acc: dict = {}   # element_id -> {'kx':.., 'ky':.., 'mode_x':.., 'mode_y':..}
    for r in T.get("FRAME SPRING ASSIGNMENTS", []):
        fid = r.get("Frame")
        if fid is None or str(fid) not in s.bar_elements_by_id:
            continue
        d = str(r.get("Dir", "")).strip()
        k = _f(r, "Stiffness")
        mode = _SIMPLE_TYPE_TO_MODE.get(
            str(r.get("SimpleType", "")).strip().lower(), 'both')
        acc = es_acc.setdefault(str(fid),
                                {'kx': 0.0, 'ky': 0.0,
                                 'mode_x': 'both', 'mode_y': 'both'})
        if d in ("1", "U1", "X"):
            acc['kx'] = k; acc['mode_x'] = mode
        elif d in ("2", "U3", "Z", "Y"):
            acc['ky'] = k; acc['mode_y'] = mode
    for fid, acc in es_acc.items():
        s.add_element_spring(fid, kx=acc['kx'], ky=acc['ky'], coord_sys="local",
                             mode_x=acc['mode_x'], mode_y=acc['mode_y'])

    # ── Load patterns → load cases (+ matching Linear analysis case) ─
    for r in T.get("LOAD PATTERN DEFINITIONS", []):
        lc = r.get("LoadPat")
        if not lc:
            continue
        sw = _f(r, "SelfWtMult", 0.0)
        if lc not in s.load_cases_by_id:
            s.add_load_case(lc, self_weight_factor=sw)
            if lc not in s.analysis_cases_by_id:
                s.add_analysis_case(lc, "Linear", coefficients={lc: 1.0})

    def _ensure_lc(lc):
        if lc and lc not in s.load_cases_by_id:
            s.add_load_case(lc)
            s.add_analysis_case(lc, "Linear", coefficients={lc: 1.0})

    # ── Joint loads (F1→fx, F3→fy, M2→ -mz) ────────────────────
    for r in T.get("JOINT LOADS - FORCE", []):
        jid, lc = r.get("Joint"), r.get("LoadPat")
        if jid is None or str(jid) not in s.nodes or not lc:
            continue
        _ensure_lc(lc)
        # plane: F1→fx, F3→fy, M2→-mz;  plate: F3→fz, M1→mx, M2→my.
        if plate:
            s.add_point_load(str(jid), lc, fx=_f(r, "F3"), fy=_f(r, "M1"),
                             mz=_f(r, "M2"))
        else:
            s.add_point_load(str(jid), lc, fx=_f(r, "F1"), fy=_f(r, "F3"),
                             mz=-_f(r, "M2"))

    # ── Frame distributed loads (global X/Z or local 1/2) ──────
    # Accumulate per (frame, pattern, dir) the start/end intensities.
    for r in T.get("FRAME LOADS - DISTRIBUTED", []):
        fid, lc = r.get("Frame"), r.get("LoadPat")
        if fid is None or str(fid) not in s.bar_elements_by_id or not lc:
            continue
        _ensure_lc(lc)
        local = str(r.get("CoordSys", "GLOBAL")).lower() == "local"
        d = str(r.get("Dir", "")).upper()
        a = _f(r, "FOverLA"); b = _f(r, "FOverLB", a)
        is_x = d in ("1", "X")
        kw = (dict(fxe=a, fxd=b) if is_x else dict(fye=a, fyd=b))
        s.add_distributed_load(str(fid), lc, coord_sys=("local" if local else "global"), **kw)

    # ── Frame point loads (concentrated at relative distance a/L) ─
    # Export writes one row per component (fx/fy/mz); accumulate them per
    # (frame, pattern, coord-sys, rel-distance) and rebuild one point load.
    import math as _math
    epl_acc: dict = {}
    for r in T.get("FRAME LOADS - POINT", []):
        fid, lc = r.get("Frame"), r.get("LoadPat")
        if fid is None or str(fid) not in s.bar_elements_by_id or not lc:
            continue
        _ensure_lc(lc)
        local = str(r.get("CoordSys", "GLOBAL")).lower() == "local"
        rel = _f(r, "RelDist", _f(r, "RelDistA"))
        key = (str(fid), lc, local, round(rel, 6))
        acc = epl_acc.setdefault(key, {"fx": 0.0, "fy": 0.0, "mz": 0.0})
        d = str(r.get("Dir", "")).upper()
        typ = str(r.get("Type", "Force")).lower()
        if typ == "moment" or d in ("3", "Y"):
            acc["mz"] += -_f(r, "Moment", _f(r, "Force"))   # SAP M = -mz
        elif d in ("1", "X"):
            acc["fx"] += _f(r, "Force")
        elif d in ("2", "Z"):
            acc["fy"] += _f(r, "Force")
    for (fid, lc, local, rel), acc in epl_acc.items():
        el = s.bar_elements_by_id.get(fid)
        ni = s.nodes.get(el.node_i) if el else None
        nj = s.nodes.get(el.node_j) if el else None
        if ni is None or nj is None:
            continue
        L = _math.hypot(nj.x - ni.x, nj.y - ni.y)
        a = rel * L
        s.add_element_point_load(fid, lc, a=a, fx=acc["fx"], fy=acc["fy"],
                                 mz=acc["mz"],
                                 coord_sys=("local" if local else "global"))

    # ── Joint patterns (fields) → per-joint scalar values ─────────────
    # A SAP joint pattern carries a scalar value per joint; a temperature that
    # references it varies as ΔT(joint) = Temp × pattern(joint).
    jpatt: dict = {}     # pattern name -> {joint id: value}
    for r in T.get("JOINT PATTERN ASSIGNMENTS", []):
        p, j = r.get("Pattern"), r.get("Joint")
        if p and j is not None:
            jpatt.setdefault(str(p), {})[str(j)] = _f(r, "Value")

    # Each joint pattern becomes a xdfem2D field: fit value ≈ a·x + b·y + c to
    # the per-joint values (exact for the usual coordinate-based patterns) and
    # store it as a field expression of (x, y).
    for pname, vals in jpatt.items():
        if pname in s.fields:
            continue
        pts = [(s.nodes[j].x, s.nodes[j].y, v)
               for j, v in vals.items() if j in s.nodes]
        if pts:
            s.add_field(str(pname), _fit_linear_expr(pts))

    # ── Frame temperature loads (uniform + gradient) ───────────
    # Type=Temperature → uniform (axial) ΔT; Type=Gradient 2-2/3-3 → a
    # through-depth gradient. A joint-pattern (field) temperature varies along
    # the bar; a bar carries a single uniform ΔT, so it is taken as Temp × the
    # mean pattern value at the bar's two nodes.
    tl_acc: dict = {}
    for r in T.get("FRAME LOADS - TEMPERATURE", []):
        fid, lc = r.get("Frame"), r.get("LoadPat")
        if fid is None or str(fid) not in s.bar_elements_by_id or not lc:
            continue
        _ensure_lc(lc)
        acc = tl_acc.setdefault((str(fid), lc), {"u": 0.0, "g": 0.0})
        typ = str(r.get("Type", "")).lower()
        if typ.startswith("gradient"):
            acc["g"] += _f(r, "TempGrad3", _f(r, "TempGrad2", _f(r, "Temp")))
        else:
            temp = _f(r, "Temp")
            pv = jpatt.get(str(r.get("JtPattern", "None")))
            if pv:
                el = s.bar_elements_by_id[str(fid)]
                scale = 0.5 * (pv.get(el.node_i, 0.0) + pv.get(el.node_j, 0.0))
                temp *= scale
            acc["u"] += temp
    for (fid, lc), acc in tl_acc.items():
        s.add_temperature_load(fid, lc, dt_uniform=acc["u"],
                               dt_gradient=acc["g"])

    # ── Area temperature loads → tri temperature loads ────────────────
    # SAP writes one ΔT per area, optionally scaled by a joint pattern (field).
    # Uniform (JtPattern=None) → the same ΔT at the three corners. Pattern → the
    # per-node ΔT = Temp × pattern value at each corner joint, so a field
    # temperature imports its actual spatial variation.
    for r in T.get("AREA LOADS - TEMPERATURE", []):
        aid, lc = r.get("Area"), r.get("LoadPat")
        if aid is None or not lc:
            continue
        if str(r.get("Type", "Temperature")).lower() not in ("", "temperature"):
            continue
        _ensure_lc(lc)
        temp = _f(r, "Temp")
        pv = jpatt.get(str(r.get("JtPattern", "None")))
        for tid in area_tris.get(str(aid), []):
            tri = s.tri_elements_by_id.get(tid)
            if pv and tri is not None:
                di = temp * pv.get(tri.node_i, 0.0)
                dj = temp * pv.get(tri.node_j, 0.0)
                dk = temp * pv.get(tri.node_k, 0.0)
            else:
                di = dj = dk = temp
            s.add_tri_temperature_load(tid, lc, dt_i=di, dt_j=dj, dt_k=dk)

    # ── Joint ground displacements → support settlements ──────
    # U1→ux, U3→uy, R2→ -tz (out-of-plane axis reversal, mirrors the export).
    for r in T.get("JOINT LOADS - GROUND DISPLACEMENT", []):
        jid, lc = r.get("Joint"), r.get("LoadPat")
        if jid is None or str(jid) not in s.nodes or not lc:
            continue
        _ensure_lc(lc)
        s.create_support_settlement(str(jid), lc, ux=_f(r, "U1"), uy=_f(r, "U3"),
                                 tz=-_f(r, "R2"))

    # ── Analysis cases (LinStatic / NonStatic) with coefficients ──
    # Coefficients come from the static load assignments; the geometric
    # nonlinearity flag distinguishes NonLinear (springs) from P-Delta.
    static_loads: dict = {}
    for r in T.get("CASE - STATIC 1 - LOAD ASSIGNMENTS", []):
        cid = r.get("Case"); pat = r.get("LoadName")
        if not cid or not pat:
            continue
        static_loads.setdefault(cid, {})[pat] = _f(r, "LoadSF", 1.0)
    geo_of = {r.get("Case"): str(r.get("GeoNonLin", "None"))
              for r in T.get("CASE - STATIC 4 - NONLINEAR PARAMETERS", [])}
    for r in T.get("LOAD CASE DEFINITIONS", []):
        cid = r.get("Case"); ctype = str(r.get("Type", "")).strip()
        if not cid:
            continue
        if ctype == "LinStatic":
            atype = "Linear"
        elif ctype == "NonStatic":
            atype = ("GeometricNonlinear"
                     if geo_of.get(cid, "None").lower() == "p-delta"
                     else "NonLinear")
        else:
            continue   # Modal / RespSpec / etc. not imported here
        coeffs = {p: f for p, f in static_loads.get(cid, {}).items()}
        for p in coeffs:
            _ensure_lc(p)
        if cid in s.analysis_cases_by_id:
            ac = s.analysis_cases_by_id[cid]
            ac.analysis_type = atype
            if coeffs:
                ac.coefficients = coeffs
        else:
            s.add_analysis_case(cid, atype, coefficients=coeffs or {})

    # ── Response-spectrum functions (period/accel pairs) ───────
    sf_pts: dict[str, list] = {}
    sf_damp: dict[str, float] = {}
    for r in T.get("FUNCTION - RESPONSE SPECTRUM - USER", []):
        name = r.get("Name")
        if not name:
            continue
        sf_pts.setdefault(name, []).append([_f(r, "Period"), _f(r, "Accel")])
        if "FuncDamp" in r:
            sf_damp[name] = _f(r, "FuncDamp", 0.05)
    for name, pts in sf_pts.items():
        if name not in s.spectral_functions:
            s.add_spectral_function(name, damping=sf_damp.get(name, 0.05),
                                    points=pts)

    # ── Mass source (loads → mass) → a Mass analysis case ──────
    mass_coeffs: dict[str, float] = {}
    for r in T.get("MASS SOURCE", []):
        lp = r.get("LoadPat")
        if lp:
            mass_coeffs[lp] = _f(r, "Multiplier", 1.0)
    if mass_coeffs:
        for lp in mass_coeffs:
            _ensure_lc(lp)
        mass_case = next((ac.id for ac in s.analysis_cases
                          if ac.analysis_type == "Mass"), None)
        if mass_case is None:
            s.add_analysis_case("MASS", "Mass", coefficients=mass_coeffs)
        else:
            s.analysis_cases_by_id[mass_case].coefficients = mass_coeffs

    # ── Modal & response-spectrum analysis cases ───────────────
    modal_general = {r.get("Case"): r
                     for r in T.get("CASE - MODAL 1 - GENERAL", [])}
    rs_general = {r.get("Case"): r
                  for r in T.get("CASE - RESPONSE SPECTRUM 1 - GENERAL", [])}
    rs_assign: dict[str, list] = {}
    for r in T.get("CASE - RESPONSE SPECTRUM 2 - LOAD ASSIGNMENTS", []):
        rs_assign.setdefault(r.get("Case"), []).append(r)
    for r in T.get("LOAD CASE DEFINITIONS", []):
        cid = r.get("Case")
        ctype = str(r.get("Type", "")).strip()
        if not cid or cid in s.analysis_cases_by_id:
            continue
        if ctype == "LinModal":
            mg = modal_general.get(cid, {})
            s.add_analysis_case(cid, "Modal",
                                num_modes=int(_f(mg, "MaxNumModes", 12)))
        elif ctype == "LinRespSpec":
            g = rs_general.get(cid, {})
            rule = ("CQC" if str(g.get("ModalCombo", "SRSS")).upper() == "CQC"
                    else "SRSS")
            dirs, func = set(), ""
            for a in rs_assign.get(cid, []):
                ln = str(a.get("LoadName", "")).upper()
                if ln in ("U1", "UX", "X"):
                    dirs.add("X")
                elif ln in ("U3", "UZ", "Z", "U2", "Y"):
                    dirs.add("Y")
                func = a.get("Function") or func
            direction = ("XY" if dirs == {"X", "Y"}
                         else ("Y" if dirs == {"Y"} else "X"))
            s.add_analysis_case(
                cid, "Spectrum", modal_case_id=(r.get("ModalCase") or ""),
                spectrum_id=(func or ""), combination_rule=rule,
                direction=direction, damping=_f(g, "ConstDamp", 0.05))

    # ── Concrete (EC2) design data ─────────────────────────────
    # Reverse fck [MPa] -> class string for the closest standard class.
    from .rc_design import _FCK as _FCK_MAP
    _fck_to_class = {v: k for k, v in _FCK_MAP.items()}
    # Steel class from the first rebar material's Fy, if present.
    steel_class = "A500"
    for r in T.get("MATERIAL PROPERTIES 03E - REBAR DATA", []):
        fy = _f(r, "Fy") / 1e3      # kN/m² -> MPa
        steel_class = f"A{int(round(fy))}" if fy else steel_class
        break
    ec2 = next(iter(T.get("PREFERENCES - CONCRETE DESIGN - EUROCODE 2-2004", [])),
               {})
    gamma_c = _f(ec2, "GammaC", 1.5) or 1.5
    gamma_s = _f(ec2, "GammaS", 1.15) or 1.15
    alpha_cc = _f(ec2, "AlphaCC", 1.0) or 1.0
    for r in T.get("MATERIAL PROPERTIES 03B - CONCRETE DATA", []):
        mat = r.get("Material")
        if not mat or mat not in s.materials:
            continue
        fck = round(_f(r, "Fc") / 1e3)      # kN/m² -> MPa
        cls = _fck_to_class.get(fck, f"C{fck}")
        s.add_concrete_material(mat, cls, steel_class,
                                gamma_c, gamma_s, alpha_cc)
    # Per-frame concrete design flag ('PROCEDURES' is SAP's table name; the old
    # singular form is accepted for backward compatibility). SAP writes either
    # an explicit "Concrete" or "From Material" (design follows the section's
    # material); the latter is RC design when that material is a concrete one.
    _conc_mats = set(getattr(s, "concrete_materials", {}))
    for r in (T.get("FRAME DESIGN PROCEDURES")
              or T.get("FRAME DESIGN PROCEDURE", [])):
        fid = r.get("Frame")
        if not fid or str(fid) not in s.bar_elements_by_id:
            continue
        proc = str(r.get("DesignProc", "")).strip().lower()
        el = s.bar_elements_by_id[str(fid)]
        sec = s.sections.get(el.section_name)
        mat_is_conc = sec is not None and sec.material_name in _conc_mats
        if proc == "concrete" or (proc == "from material" and mat_is_conc):
            el.rc_design = True

    # ── Concrete section cover (rc_cover) ──────────────────────
    # Current export writes 'FRAME SECTION PROPERTIES 03 - CONCRETE BEAM' with
    # TopCover/BotCover; accept the older '06'/'Cover' variants too.
    for r in (T.get("FRAME SECTION PROPERTIES 03 - CONCRETE BEAM")
              or T.get("FRAME SECTION PROPERTIES 06 - CONCRETE BEAM", [])):
        sname = r.get("SectionName")
        if not sname or sname not in s.sections:
            continue
        cover = _f(r, "TopCover", _f(r, "BotCover", _f(r, "Cover", 0.0)))
        if cover > 0:
            s.sections[sname].rc_cover = cover

    # ── Linear load combinations ───────────────────────────────
    combo_acc: dict[str, dict] = {}
    for r in T.get("COMBINATION DEFINITIONS", []):
        cid = r.get("ComboName") or r.get("Combo")
        case = r.get("CaseName") or r.get("Case")
        if not cid or not case:
            continue
        sf = _f(r, "ScaleFactor", 1.0)
        combo_acc.setdefault(cid, {})[case] = sf
    for cid, coeffs in combo_acc.items():
        # Only keep references to known analysis cases / load cases.
        coeffs = {k: v for k, v in coeffs.items()
                  if k in s.analysis_cases_by_id or k in s.load_cases_by_id}
        if coeffs:
            try:
                s.add_load_combination(cid, coeffs, combo_type="LinearSum")
            except ValueError:
                # e.g. a NonLinear case may not be combined with others — skip.
                continue

    # ── Joint added masses (U1→mx, U3→my, R2→mtz) ──────────────
    # xdfem2D nodal masses belong to a Mass analysis case; SAP's added-mass
    # table is case-independent, so attach them to a single Mass case (created
    # on demand) named 'MASS'.
    added_mass = [r for r in T.get("JOINT ADDED MASS ASSIGNMENTS", [])
                  if str(r.get("Joint", "")) in s.nodes
                  and (_f(r, "U1") or _f(r, "U3") or _f(r, "R2"))]
    if added_mass:
        mass_case = next((ac.id for ac in getattr(s, "analysis_cases", [])
                          if ac.analysis_type == "Mass"), None)
        if mass_case is None:
            mass_case = "MASS"
            if mass_case not in s.analysis_cases_by_id:
                s.add_analysis_case(mass_case, "Mass", coefficients={})
        for r in added_mass:
            s.add_nodal_mass(str(r.get("Joint")), mass_case,
                             mx=_f(r, "U1"), my=_f(r, "U3"), mtz=_f(r, "R2"))

    s.normalize_combinations_to_analysis_cases()
    return s


def load_s2k(path: str | Path, domain: str = None):
    """Read a SAP2000 ``.s2k`` file and return a :class:`Structure2D`.

    *domain* ('plane'/'plate') is forwarded to :func:`from_s2k`; leave it None to
    infer from the file's active DOFs (raises :class:`AmbiguousDomainError` if
    the file does not identify the domain)."""
    return from_s2k(Path(path).read_text(encoding="utf-8", errors="replace"),
                    domain=domain)
