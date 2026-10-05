"""
Data structures for xdfem2D structural analysis.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class SectionType(str, Enum):
    CONCRETE = "Concrete"
    STEEL    = "Steel"
    TIMBER   = "Timber"
    GENERIC  = "Other"


class SectionShape(str, Enum):
    """Geometric shape of a section, driving the area / inertia formulas.

    Dimension mapping (b = width or diameter, h = height,
    tw = web / wall thickness, tf = flange thickness):
      GENERIC          — area / inertia taken from overrides (or b·h fallback)
      RECTANGULAR      — full rectangle b × h
      CIRCULAR         — solid circle, diameter b
      RECTANGULAR_HOLLOW — box b × h with uniform wall thickness tw
      CIRCULAR_HOLLOW  — tube, outer diameter b, wall thickness tw
      I                — symmetric I: flange width b, total height h, tf, tw
      T                — T: flange width b, total height h, tf, tw
    """
    GENERIC            = "Generic"
    RECTANGULAR        = "Rectangular"
    CIRCULAR           = "Circular"
    RECTANGULAR_HOLLOW = "Rectangular hollow"
    CIRCULAR_HOLLOW    = "Circular hollow"
    I                  = "I"
    T                  = "T"


def section_area_inertia(shape, b, h, tw=0.0, tf=0.0):
    """Return (area, inertia) [m², m⁴] for a section shape and dimensions.

    The inertia is about the horizontal centroidal axis (in-plane bending).
    Invalid / degenerate dimensions fall back to the solid rectangle b·h.
    """
    import math
    rect = (b * h, b * h ** 3 / 12.0)
    try:
        if shape == SectionShape.RECTANGULAR:
            return rect
        if shape == SectionShape.CIRCULAR:
            d = b
            return (math.pi * d ** 2 / 4.0, math.pi * d ** 4 / 64.0)
        if shape == SectionShape.RECTANGULAR_HOLLOW:
            bi, hi = b - 2 * tw, h - 2 * tw
            if bi <= 0 or hi <= 0:
                return rect
            return (b * h - bi * hi, (b * h ** 3 - bi * hi ** 3) / 12.0)
        if shape == SectionShape.CIRCULAR_HOLLOW:
            d, di = b, b - 2 * tw
            if di <= 0:
                return (math.pi * d ** 2 / 4.0, math.pi * d ** 4 / 64.0)
            return (math.pi * (d ** 2 - di ** 2) / 4.0,
                    math.pi * (d ** 4 - di ** 4) / 64.0)
        if shape == SectionShape.I:
            hw = h - 2 * tf
            if hw <= 0 or tw <= 0 or tf <= 0:
                return rect
            A = 2 * b * tf + hw * tw
            I = b * h ** 3 / 12.0 - (b - tw) * hw ** 3 / 12.0
            return (A, I)
        if shape == SectionShape.T:
            hw = h - tf
            if hw <= 0 or tw <= 0 or tf <= 0:
                return rect
            Af, Aw = b * tf, hw * tw
            A = Af + Aw
            yf, yw = tf / 2.0, tf + hw / 2.0            # from top
            yb = (Af * yf + Aw * yw) / A
            I = (b * tf ** 3 / 12.0 + Af * (yb - yf) ** 2
                 + tw * hw ** 3 / 12.0 + Aw * (yw - yb) ** 2)
            return (A, I)
    except (ZeroDivisionError, ValueError):
        return rect
    return rect


def section_inertia_minor(shape, b, h, tw=0.0, tf=0.0):
    """Return the minor-axis inertia Iz [m⁴] (bending about the *vertical*
    centroidal axis) for a section shape and dimensions — the companion of
    :func:`section_area_inertia`, which gives the major-axis Iy.

    Used to rotate a steel profile in-plane: a section turned by an angle θ has
    an in-plane bending inertia Iy·cos²θ + Iz·sin²θ. Invalid / degenerate
    dimensions fall back to the solid rectangle about the vertical axis.
    """
    import math
    rect = h * b ** 3 / 12.0
    try:
        if shape == SectionShape.RECTANGULAR:
            return rect
        if shape == SectionShape.CIRCULAR:
            return math.pi * b ** 4 / 64.0          # symmetric: Iz = Iy
        if shape == SectionShape.RECTANGULAR_HOLLOW:
            bi, hi = b - 2 * tw, h - 2 * tw
            if bi <= 0 or hi <= 0:
                return rect
            return (h * b ** 3 - hi * bi ** 3) / 12.0
        if shape == SectionShape.CIRCULAR_HOLLOW:
            d, di = b, b - 2 * tw
            if di <= 0:
                return math.pi * d ** 4 / 64.0
            return math.pi * (d ** 4 - di ** 4) / 64.0
        if shape == SectionShape.I:
            hw = h - 2 * tf
            if hw <= 0 or tw <= 0 or tf <= 0:
                return rect
            # two flanges bending about the vertical axis + the thin web
            return 2.0 * (tf * b ** 3 / 12.0) + hw * tw ** 3 / 12.0
        if shape == SectionShape.T:
            hw = h - tf
            if hw <= 0 or tw <= 0 or tf <= 0:
                return rect
            # symmetric about the vertical axis: flange plate + web plate
            return tf * b ** 3 / 12.0 + hw * tw ** 3 / 12.0
    except (ZeroDivisionError, ValueError):
        return rect
    return rect


def section_plastic_moduli(shape, b, h, tw=0.0, tf=0.0):
    """Return (Wpl_y, Wpl_z) [m³] — the plastic section moduli about the major
    and minor axes. Used as the computed fallback for a manually-defined section
    (a catalogue profile carries its own tabulated values). Degenerate
    dimensions fall back to the solid rectangle.
    """
    rect = (b * h ** 2 / 4.0, h * b ** 2 / 4.0)
    try:
        if shape == SectionShape.RECTANGULAR:
            return rect
        if shape == SectionShape.CIRCULAR:
            return (b ** 3 / 6.0, b ** 3 / 6.0)
        if shape == SectionShape.RECTANGULAR_HOLLOW:
            bi, hi = b - 2 * tw, h - 2 * tw
            if bi <= 0 or hi <= 0:
                return rect
            return ((b * h ** 2 - bi * hi ** 2) / 4.0,
                    (h * b ** 2 - hi * bi ** 2) / 4.0)
        if shape == SectionShape.CIRCULAR_HOLLOW:
            d, di = b, b - 2 * tw
            if di <= 0:
                return (d ** 3 / 6.0, d ** 3 / 6.0)
            return ((d ** 3 - di ** 3) / 6.0, (d ** 3 - di ** 3) / 6.0)
        if shape == SectionShape.I:
            hw = h - 2 * tf
            if hw <= 0 or tw <= 0 or tf <= 0:
                return rect
            wply = b * tf * (h - tf) + tw * hw ** 2 / 4.0
            wplz = tf * b ** 2 / 2.0 + hw * tw ** 2 / 4.0
            return (wply, wplz)
        if shape == SectionShape.T:
            hw = h - tf
            if hw <= 0 or tw <= 0 or tf <= 0:
                return rect
            af, aw = b * tf, hw * tw
            a = af + aw
            # Plastic neutral axis at half-area, measured from the top face.
            if af >= a / 2.0:                         # PNA within the flange
                yp = a / (2.0 * b)
            else:                                     # PNA within the web
                yp = tf + (a / 2.0 - af) / tw
            # Wpl,y = first moment of the two equal halves about the PNA.
            #   top part (flange + web above yp), bottom part (web below yp).
            top_flange = af * abs(yp - tf / 2.0)
            if yp <= tf:
                top = b * yp * (yp / 2.0) + 0.0       # only flange above
                top = b * yp ** 2 / 2.0
                bot = (b * (tf - yp) ** 2 / 2.0
                       + aw * abs((tf + hw / 2.0) - yp))
            else:
                web_above = tw * (yp - tf)
                top = top_flange + web_above * (yp - tf) / 2.0
                web_below = tw * (h - yp)
                bot = web_below * (h - yp) / 2.0
            wply = top + bot
            wplz = tf * b ** 2 / 4.0 + hw * tw ** 2 / 4.0
            return (wply, wplz)
    except (ZeroDivisionError, ValueError):
        return rect
    return rect


def section_shear_areas(shape, b, h, tw=0.0, tf=0.0):
    """Return (Av_y, Av_z) [m²] — shear areas for forces parallel to the y (in
    the flanges) and z (in the web) axes, following the EN 1993-1-1 §6.2.6
    approximations (root radii ignored). Computed fallback for a manual section.
    """
    import math
    area, _ = section_area_inertia(shape, b, h, tw, tf)
    try:
        if shape == SectionShape.RECTANGULAR:
            return (5.0 / 6.0 * area, 5.0 / 6.0 * area)
        if shape == SectionShape.CIRCULAR:
            return (0.9 * area, 0.9 * area)
        if shape == SectionShape.CIRCULAR_HOLLOW:
            return (2.0 * area / math.pi, 2.0 * area / math.pi)
        if shape == SectionShape.RECTANGULAR_HOLLOW:
            if b + h <= 0:
                return (area / 2.0, area / 2.0)
            return (area * b / (b + h), area * h / (b + h))
        if shape == SectionShape.I:
            hw = h - 2 * tf
            if hw <= 0 or tw <= 0 or tf <= 0:
                return (area, area)
            av_z = area - 2.0 * b * tf + tw * tf     # web (V parallel to z)
            av_y = 2.0 * b * tf                       # flanges (V parallel to y)
            return (av_y, av_z)
        if shape == SectionShape.T:
            hw = h - tf
            if hw <= 0 or tw <= 0 or tf <= 0:
                return (area, area)
            return (b * tf, hw * tw)
    except (ZeroDivisionError, ValueError):
        return (area, area)
    return (area, area)


def section_warping_constant(shape, b, h, tw=0.0, tf=0.0) -> float:
    """Return the warping constant Iw [m⁶] for a section shape.

    For a doubly-symmetric I the standard ``Iw = Iz·(h − tf)²/4`` is used (Iz the
    minor-axis inertia, h − tf the distance between flange centroids). Closed
    sections (RHS/SHS/CHS) and solid shapes warp negligibly → Iw ≈ 0. Computed
    fallback for a manual section; a catalogue profile carries its own value.
    """
    try:
        if shape == SectionShape.I:
            hw = h - 2 * tf
            if hw <= 0 or tw <= 0 or tf <= 0:
                return 0.0
            iz = section_inertia_minor(shape, b, h, tw, tf)
            hs = h - tf                       # between flange centroids
            return iz * hs ** 2 / 4.0
    except (ZeroDivisionError, ValueError):
        return 0.0
    return 0.0


def section_torsion_modulus(shape, b, h, tw=0.0, tf=0.0) -> float:
    """Return the torsional section modulus Wt [m³] (τt = T/Wt).

    Open sections (I/T): Wt = J/t_max. Closed hollow sections use the thin-wall
    Bredt modulus 2·A_m·t; CHS uses 2·J/d; solids a rough J/(short side). Computed
    fallback for a manual section; a catalogue profile carries its own value.
    """
    import math
    j = section_torsion_constant(shape, b, h, tw, tf)
    try:
        if shape in (SectionShape.I, SectionShape.T):
            t = max(tw, tf)
            return j / t if t > 0 else 0.0
        if shape == SectionShape.RECTANGULAR_HOLLOW:
            if b > tw and h > tw:
                return 2.0 * (b - tw) * (h - tw) * tw   # 2·A_m·t (Bredt)
            return 0.0
        if shape == SectionShape.CIRCULAR_HOLLOW:
            return 2.0 * j / b if b > 0 else 0.0
        if shape == SectionShape.CIRCULAR:
            return math.pi * b ** 3 / 16.0
        if shape == SectionShape.RECTANGULAR:
            c = min(b, h)
            return j / c if c > 0 else 0.0
    except (ZeroDivisionError, ValueError):
        return 0.0
    return 0.0


def section_buckling_curves(shape, b, h, tf=0.0, *, rolled=True):
    """Return (curve_y, curve_z, curve_LT) per EN 1993-1-1 Tables 6.2 / 6.5.

    Auto-selection for the flexural buckling curves about the two axes and the
    lateral-torsional buckling curve, from the shape and (for I sections) the
    h/b ratio and flange thickness (m). Grade-460 refinements (curve a0) are not
    applied. Computed fallback for a manual section; a catalogue profile carries
    its own y/z curves.
    """
    if shape == SectionShape.I:
        ratio = (h / b) if b > 0 else 0.0
        if ratio > 1.2:
            cy, cz = ("a", "b") if tf <= 0.040 else ("b", "c")
        elif tf <= 0.100:
            cy, cz = "b", "c"
        else:
            cy, cz = "d", "d"
        clt = "b" if ratio <= 2.0 else "c"
        return cy, cz, clt
    if shape in (SectionShape.RECTANGULAR_HOLLOW, SectionShape.CIRCULAR_HOLLOW):
        # Hot-finished hollow sections → curve a on both axes (assumed).
        return "a", "a", "b"
    if shape == SectionShape.CIRCULAR:
        return "a", "a", "b"
    # Generic / rectangular / T — use a conservative middle curve.
    return "c", "c", "d"


def section_torsion_constant(shape, b, h, tw=0.0, tf=0.0) -> float:
    """Return the St-Venant torsion constant J [m⁴] for a section shape.

    Used by the grillage (plate-domain bar) element: k_torsion = G·J/L. The
    formulas are the standard engineering approximations; a section whose J
    matters more precisely can set ``torsion_override``.

    Rectangle (a = long side, c = short side):
        J = a·c³·(1/3 − 0.21·(c/a)·(1 − c⁴/(12·a⁴)))
    Circular (solid / hollow): polar inertia π·d⁴/32 (minus the hole).
    Rectangular hollow (closed thin-wall, Bredt): J = 4·A₀²·t / p₀.
    I / T (open thin-wall): J = Σ (bᵢ·tᵢ³)/3.
    Invalid / degenerate dimensions fall back to the solid rectangle.
    """
    import math

    def _rect(bb, hh):
        a, c = (bb, hh) if bb >= hh else (hh, bb)
        if a <= 0.0 or c <= 0.0:
            return 0.0
        return a * c ** 3 * (1.0 / 3.0 - 0.21 * (c / a)
                             * (1.0 - c ** 4 / (12.0 * a ** 4)))

    rect = _rect(b, h)
    try:
        if shape == SectionShape.CIRCULAR:
            d = b
            return math.pi * d ** 4 / 32.0
        if shape == SectionShape.CIRCULAR_HOLLOW:
            d, di = b, b - 2 * tw
            if di <= 0:
                return math.pi * d ** 4 / 32.0
            return math.pi * (d ** 4 - di ** 4) / 32.0
        if shape == SectionShape.RECTANGULAR_HOLLOW:
            bi, hi = b - 2 * tw, h - 2 * tw
            if bi <= 0 or hi <= 0 or tw <= 0:
                return rect
            a0 = (b - tw) * (h - tw)          # area enclosed by the midline
            p0 = 2.0 * ((b - tw) + (h - tw))  # midline perimeter
            return 4.0 * a0 * a0 * tw / p0
        if shape == SectionShape.I:
            hw = h - 2 * tf
            if hw <= 0 or tw <= 0 or tf <= 0:
                return rect
            return (2.0 * b * tf ** 3 + hw * tw ** 3) / 3.0
        if shape == SectionShape.T:
            hw = h - tf
            if hw <= 0 or tw <= 0 or tf <= 0:
                return rect
            return (b * tf ** 3 + hw * tw ** 3) / 3.0
    except (ZeroDivisionError, ValueError):
        return rect
    return rect


class MaterialType(str, Enum):
    """Physical material class, used for stress calculations."""
    CONCRETE = "Concrete"
    STEEL    = "Steel"
    TIMBER   = "Timber"
    OTHER    = "Other"


def default_design_for(material_type, given: dict | None = None) -> dict:
    """The design strengths a Concrete, Steel or Timber material needs
    (fck/fyk, fy/fu, or the EN 338 strengths), from the eurocodepy database
    (``databases.material_from_grade``).

    The class is the one named in *given* (``class_conc``/``class_reinf`` for
    concrete, ``class`` for steel and timber), so a design that names a class
    but not its strengths gets that class's strengths -- never another's.
    Without one, the type's default: C30/37 with A500NR, S275, C24. A class
    the database does not know falls back to the default's strengths (the
    checked loader and add_material are what flag a bad class). Empty for
    OTHER/None. Callers fill only the missing keys, so any value the user
    supplied is preserved."""
    mt = getattr(material_type, "value", material_type)
    default = {MaterialType.CONCRETE.value: "C30/37",
               MaterialType.STEEL.value: "S275",
               MaterialType.TIMBER.value: "C24"}.get(mt)
    if default is None:
        return {}
    from .databases import material_from_grade
    given = given or {}
    if mt == MaterialType.CONCRETE.value:
        grade, reinf = given.get("class_conc"), given.get("class_reinf")
    else:
        grade, reinf = given.get("class"), None
    for g, r in ((grade or default, reinf), (default, None)):
        try:
            return material_from_grade(mt, g, reinforcement=r)["design"]
        except ValueError:
            continue
    return {}


def section_type_of(material) -> "SectionType":
    """Design (section) type derived from a Material's physical type.

    The section no longer stores its own type — it always follows the material
    (Concrete / Steel / Timber / Other). Returns ``SectionType.GENERIC`` when the
    material is missing or its type is unknown.
    """
    if material is None:
        return SectionType.GENERIC
    mt = getattr(material, "material_type", None)
    val = mt.value if hasattr(mt, "value") else str(mt)
    try:
        return SectionType(val)
    except ValueError:
        return SectionType.GENERIC


class ActionType(str, Enum):
    """The Eurocode action a load case belongs to.

    The value is the stored datum and nothing else: a single ASCII letter,
    which is what goes into a .x2d and what anyone writing one by hand — or
    any program generating one — would guess. :attr:`label` is what the
    interface shows.

    They used to be the same string, "G – Permanent", with an en dash. That
    put presentation in the file format, and the consequences were not
    cosmetic: the dash comes out of json.dumps as \u2013, nothing can guess
    it, and a language model asked to write a load case produced
    action_type="Permanent" and was rejected by an error naming neither the
    valid values nor the reason.
    """

    G = "G"
    Q = "Q"
    W = "W"
    E = "E"
    T = "T"
    S = "S"
    C = "C"
    A = "A"
    F = "F"
    O = "O"

    @property
    def label(self) -> str:
        """For the interface. Never written to a file."""
        return {
            'G': "G – Permanent", 'Q': "Q – Live", 'W': "W – Wind",
            'E': "E – Seismic", 'T': "T – Temperature",
            'S': "S – Snow", 'C': "C – Construction",
            'A': "A – Accident", 'F': "F – Fire", 'O': "O – Other",
        }[self.value]

    @classmethod
    def coerce(cls, value) -> 'ActionType':
        """Whatever someone meant, as an ActionType.

        Accepts the letter, the word, the old dashed label, the same with a
        plain hyphen, in any case. Files written before the value changed
        still load, and a model that writes "permanent" is understood rather
        than refused.
        """
        if isinstance(value, cls):
            return value
        text = str(value or '').strip()
        if not text:
            return cls.G
        # The letter on its own, or the letter before a separator.
        head = text.replace('\u2013', '-').split('-')[0].strip().upper()
        if head in cls.__members__:
            return cls[head]
        # The American letters. The Eurocode says G and Q; ASCE says D and L,
        # and that is what most of the writing in the world uses \u2014 a model
        # asked for a beam wrote action_type "D" for dead and the file was
        # refused. The word "Dead" was already understood; the letter for it
        # was not, which is the wrong way round.
        if head in ('D', 'DL'):
            return cls.G
        if head in ('L', 'LL'):
            return cls.Q
        words = {'PERMANENT': cls.G, 'DEAD': cls.G, 'LIVE': cls.Q,
                 'IMPOSED': cls.Q, 'VARIABLE': cls.Q, 'WIND': cls.W,
                 'SEISMIC': cls.E, 'EARTHQUAKE': cls.E, 'TEMPERATURE': cls.T,
                 'THERMAL': cls.T, 'SNOW': cls.S,
                 'CONSTRUCTION': cls.C, 'EXECUTION': cls.C,
                 'ACCIDENT': cls.A, 'ACCIDENTAL': cls.A,
                 'FIRE': cls.F, 'OTHER': cls.O}
        for word, member in words.items():
            if word in text.upper():
                return member
        raise ValueError(
            f"unknown action type {value!r}. Use one of "
            f"{', '.join(m.value for m in cls)} — or the word: "
            f"permanent, live, wind, seismic, temperature. "
            f"D and L are read as G and Q")


#: How a LoadCombination combines its inputs. Renamed from Portuguese to
#: English (26/09/2026, Matias: "tudo devia estar em inglês. os combo types
#: estão mal") -- the whole engine/app surface is English-facing and this was
#: the one enum left over in Portuguese, which a model asked for "the
#: envelope type" would naturally spell "Envelope" and have rejected.
#:   LinearSum — linear superposition: Sum coeff_i * case_i
#:   Envelope  — envelope: max and min of each component across cases
#:   AbsSum    — sum of absolutes: Sum |coeff_i * case_i|
#:   SRSS      — square root of sum of squares: sqrt(Sum (coeff_i * case_i)^2)
#: NonLinearCombo / SequenceCombo were already English and are unaffected.
_LEGACY_COMBO_TYPES = {
    'SomaLinear': 'LinearSum',
    'Envolvente': 'Envelope',
    'SomaModulo': 'AbsSum',
    'RaizSoma':   'SRSS',
    # The old misspelling that structure_io used to special-case on its own.
    'SomaSodulo': 'AbsSum',
}


def coerce_combo_type(value) -> str:
    """Whatever someone meant by *value*, as the canonical combo_type string.

    Accepts the current English value in any case, or the old Portuguese
    value a .x2d written before 26/09/2026 may still hold (also
    case-insensitively) -- so an old file loads without error and, once the
    model is saved again, is written back out with the English value.
    ``NonLinearCombo`` and ``SequenceCombo`` were already English; a caller
    spelling one of them (or a current value) in a different case is
    normalised to the listed spelling below, same treatment as a case id
    (see Structure2D._find_load_case_id_ci) -- 'envelope'/'ENVELOPE' should
    not be rejected just because it is not spelled exactly like the enum.
    Any value this function does not recognise at all passes through
    unchanged -- the caller's own validation (add_load_combination's
    ``valid`` tuple) is what rejects a genuinely wrong value; this only
    normalises a known spelling, old or current.
    """
    text = str(value or '').strip()
    if text in _LEGACY_COMBO_TYPES:
        return _LEGACY_COMBO_TYPES[text]
    upper = text.upper()
    for old, new in _LEGACY_COMBO_TYPES.items():
        if upper == old.upper():
            return new
    for current in ('LinearSum', 'Envelope', 'AbsSum', 'SRSS',
                    'NonLinearCombo', 'SequenceCombo'):
        if upper == current.upper():
            return current
    return text



#: Analysis domains a model can live in. One per model, never mixed:
#:   "plane" — in-plane behaviour (frames + membranes). Nodal DOFs, in storage
#:             order: (ux, uy, tz).
#:   "plate" — out-of-plane behaviour (grillages + plate bending). Nodal DOFs,
#:             in storage order: (w, tx, ty) — transverse deflection along
#:             global Z and rotations about global X and Y (right-hand rule).
#: Both domains use exactly three DOFs per node, so the DOF numbering, the
#: shapes of K/F/U and everything that operates on them positionally is shared;
#: only the *meaning* of the three components changes.
DOMAINS = ("plane", "plate")

#: Per-domain names of the three nodal DOF components, in storage order
#: (component 0, 1, 2). The single source for every label the engine emits.
DOF_LABELS = {
    "plane": ("ux", "uy", "tz"),
    "plate": ("w", "tx", "ty"),
}

# ── Result vocabulary per domain ───────────────────────────────────────────
# The single source for how a result quantity is *named* in each domain, shared
# by the application's GUI domain adapter and its assistant. Storage keys never change with the domain — a
# plate deflection is still stored in the ``ux`` slot — only the label/unit the
# reader sees does. Keeping these here (and not in ``gui/``) lets the headless
# assistant read one model correctly without importing any GUI code.

#: Per-triangle result components as (label, key, unit). ``key`` indexes the
#: stored per-triangle result dict — ``tri_stress`` (membrane σ) in plane,
#: ``dkt_moments`` (bending moments, shears, Wood-Armer) in plate. First entry
#: is the natural default.
TRI_RESULT_COMPONENTS = {
    "plane": (
        ("σx", "sx", "kN/m²"),
        ("σy", "sy", "kN/m²"),
        ("τxy", "txy", "kN/m²"),
        ("von Mises", "vm", "kN/m²"),
        ("σ1", "s1", "kN/m²"),
        ("σ2", "s2", "kN/m²"),
    ),
    "plate": (
        ("mx", "mx", "kNm/m"),
        ("my", "my", "kNm/m"),
        ("mxy", "mxy", "kNm/m"),
        ("m1", "m1", "kNm/m"),
        ("m2", "m2", "kNm/m"),
        ("vx", "vx", "kN/m"),
        ("vy", "vy", "kN/m"),
        ("m*x,bot (W-A)", "mx_bot", "kNm/m"),
        ("m*y,bot (W-A)", "my_bot", "kNm/m"),
        ("m*x,top (W-A)", "mx_top", "kNm/m"),
        ("m*y,top (W-A)", "my_top", "kNm/m"),
    ),
}

#: Nodal displacement components as (label, storage-key). The storage key
#: (ux/uy/rz) is fixed; its meaning is (ux, uy, θz) or (w, θx, θy) by domain.
DISP_COMPONENTS = {
    "plane": (("u_x", "ux"), ("u_y", "uy"), ("θ_z", "rz")),
    "plate": (("w", "ux"), ("θx", "uy"), ("θy", "rz")),
}

#: Bar end-force / diagram views as {view: (label, unit)}. The axial slot is the
#: axial force N in plane, the torsion T in a plate grillage.
BAR_DIAGRAMS = {
    "plane": {"bending": ("M", "kNm"), "shear": ("V", "kN"),
              "axial": ("N", "kN")},
    "plate": {"bending": ("M", "kNm"), "shear": ("V", "kN"),
              "axial": ("T", "kNm")},
}

#: Reaction component labels+units, in storage-slot order.
REACTION_LABELS = {
    "plane": (("Rx", "kN"), ("Ry", "kN"), ("Mz", "kNm")),
    "plate": (("Rz", "kN"), ("Mx", "kNm"), ("My", "kNm")),
}

#: True when the deformed shape is out-of-plane (a w colour field) rather than
#: drawn by moving the nodes in-plane.
DEFORMED_IS_FIELD = {"plane": False, "plate": True}


def result_vocab(domain: str = "plane") -> dict:
    """All domain-dependent result vocabulary in one dict, defaulting to plane.

    Consumed by both the GUI adapter and the assistant so the two can never name
    the same quantity differently."""
    d = domain if domain in DOF_LABELS else "plane"
    return {
        "dof_labels": DOF_LABELS[d],
        "tri_result_components": TRI_RESULT_COMPONENTS[d],
        "disp_components": DISP_COMPONENTS[d],
        "bar_diagrams": BAR_DIAGRAMS[d],
        "reaction_labels": REACTION_LABELS[d],
        "deformed_is_field": DEFORMED_IS_FIELD[d],
    }


@dataclass
class Node:
    id: str
    x: float
    y: float


@dataclass
class Material:
    name: str
    elastic_modulus: float        # E  [kN/m² or consistent units]
    unit_weight: float             # γ  [kN/m³]
    alpha: float = 1.0e-5         # thermal expansion coefficient [1/°C]
    unit_mass: Optional[float] = None  # ρ  [t/m³]; if None, derived as unit_weight / g (g=9.81)
    material_type: MaterialType = MaterialType.CONCRETE  # physical class for stress calc
    poisson: float = 0.2          # Poisson's ratio ν (plane elements)
    # Type-specific design strengths (MPa, kg/m³). Keys depend on material_type:
    #   concrete: class_conc, fck, class_reinf, fyk
    #   steel:    class, fy, fu
    #   timber:   class, fmk, fvk, fc0k, ft0k, rhok
    design: dict = field(default_factory=dict)

    @property
    def shear_modulus(self) -> float:
        """G = E / 2(1+ν) — used by the grillage element's torsional stiffness."""
        return self.elastic_modulus / (2.0 * (1.0 + self.poisson))


@dataclass
class Section:
    name: str
    material_name: str
    b: float    # width  (b2)
    h: float    # height (b3)
    # Optional overrides for non-rectangular sections (e.g. steel profiles).
    # When set these take precedence over the b×h formulas.
    area_override:    Optional[float] = None  # A  [m²]
    inertia_override: Optional[float] = None  # Iy [m⁴]
    inertia_minor_override: Optional[float] = None  # Iz [m⁴] (rotated sections)
    torsion_override: Optional[float] = None  # J  [m⁴] (grillage torsion)
    # Design section moduli / shear areas — populated from the catalogue when a
    # commercial profile is applied, left None for a manually-defined section
    # (where they are computed from the shape). SI units: moduli m³, areas m².
    wel_y_override:   Optional[float] = None  # Wel,y [m³]
    wpl_y_override:   Optional[float] = None  # Wpl,y [m³]
    wel_z_override:   Optional[float] = None  # Wel,z [m³]
    wpl_z_override:   Optional[float] = None  # Wpl,z [m³]
    av_y_override:    Optional[float] = None  # shear area A_v,y [m²]
    av_z_override:    Optional[float] = None  # shear area A_v,z [m²]
    warping_override: Optional[float] = None  # warping constant Iw [m⁶]
    wt_override:      Optional[float] = None  # torsional modulus Wt [m³]
    # Buckling curves (EN 1993-1-1 Table 6.2). None → auto from the shape.
    curve_y_override: Optional[str]   = None
    curve_z_override: Optional[str]   = None
    profile_name:     Optional[str]   = None  # e.g. 'IPE200'
    # Profile orientation in the frame plane [degrees]. Meaningful for steel
    # profiles: 0° places the section's strong (y-y) axis to resist in-plane
    # bending — the natural orientation, with the largest inertia in the plane —
    # and 90° turns it onto its weak axis. The in-plane bending inertia is then
    # Iy·cos²θ + Iz·sin²θ.
    angle:            float = 0.0
    # NOTE: the design "type" (concrete/steel/...) is no longer stored here —
    # it is derived from the section's material (see section_type_of).
    # Geometric shape and its extra thickness dimensions (see SectionShape).
    shape:            SectionShape    = SectionShape.GENERIC
    tw:               float = 0.0     # web / wall thickness [m]
    tf:               float = 0.0     # flange thickness [m]
    # RC design parameters (used when the material is concrete)
    rc_cover:         float = 0.045   # mechanical cover [m]
    rc_alpha_s:       float = 90.0    # stirrup inclination [degrees]
    rc_bar_phi:       float = 16.0    # main longitudinal bar diameter [mm]
    # When False, the EC2 minimum shear reinforcement (ρw,min·b) is NOT enforced
    # for this section: members where the concrete alone carries the shear get
    # Asw/s = 0 instead of the minimum stirrups (e.g. box-culvert slabs).
    rc_shear_min:     bool  = True
    # How EC2 §6.3.2(3) torsion longitudinal steel (Asl,tor) is distributed
    # for a grillage (plate-domain) member — "top_bottom" (default) splits it
    # 50/50 onto the flexural top/bottom faces (the pre-existing xdfem2D
    # simplification); "perimeter" distributes it proportionally around the
    # equivalent thin-walled section's centre-line, adding steel to the side
    # faces too. Kept off by default so enabling it is an explicit,
    # non-silent decision (dev/GRILLAGE_DESIGN.md §2.4/§5.1/Phase 3) — it
    # typically means MORE side-face steel than an existing project already
    # assumed, not less.
    rc_torsion_distribution: str = "top_bottom"   # "top_bottom" | "perimeter"
    # Beam-bars design overrides for every beam using this section:
    # ``{BeamBarParams field: value}`` limited to ``beam_bars_params.
    # SECTION_PARAMS`` (diameters, stirrup Ø, max layers); ``None`` = inherit
    # the model preferences. dev/BEAM_BARS_DEFINITION.md §6.
    beam_overrides: Optional[dict] = None
    # RC column design (EC2 §5.8) defaults for bars using this section — see
    # dev/RC_COLUMN_DESIGN.md and dev/BUCKLING_COLUMN_PERSISTENCE.md §"element
    # vs secção". ``is_column`` marks this section as a physical column by
    # default; a bar can override it explicitly via ``BarElement.is_column``
    # (Design ▸ Buckling lengths…) — ``None`` there means "follow the
    # section". ``rc_phi_ef``/``rc_n_bars``/``rc_n_bars_y``/``rc_n_bars_z``/
    # ``rc_second_order_method`` configure the column check itself and have
    # **no** per-bar override: a different rebar count/creep ratio/method
    # describes a different column *type*, i.e. a different section, not an
    # exception on one bar (unlike ``is_column``, which is a yes/no
    # classification that can legitimately differ bar-by-bar even within the
    # same section). ``rc_n_bars`` is used for CIRCULAR sections (single bar
    # count evenly spaced on the circle -- there is no face direction to
    # split over). ``rc_n_bars_y``/``rc_n_bars_z`` are used for RECTANGULAR
    # sections instead, to support skew/biaxial bending (flexão desviada):
    # ``rc_n_bars_y`` bars on each of the two faces perpendicular to z
    # (top/bottom, spanning width b), ``rc_n_bars_z`` bars on each of the two
    # faces perpendicular to y (left/right, spanning height h), corners
    # shared/counted once -- see RebarLayout.symmetric_rectangular_biaxial in
    # eurocodepy and dev/RC_COLUMN_DESIGN.md §14.
    is_column: bool = False
    rc_phi_ef: float = 2.0
    rc_n_bars: int = 4
    rc_n_bars_y: int = 2
    rc_n_bars_z: int = 2
    rc_second_order_method: str = "nominal_curvature"
    # Timber design parameter (used when the material is timber). EN 1995-1-1
    # service class: "SC1" (dry), "SC2", "SC3" (wet) — drives kmod / kdef.
    timber_service_class: str = "SC1"

    @property
    def area(self) -> float:
        if self.area_override is not None:
            return self.area_override
        return section_area_inertia(self.shape, self.b, self.h, self.tw, self.tf)[0]

    @property
    def inertia_major(self) -> float:
        """Major-axis (strong, y-y) second moment Iy [m⁴]."""
        if self.inertia_override is not None:
            return self.inertia_override
        return section_area_inertia(self.shape, self.b, self.h, self.tw, self.tf)[1]

    @property
    def inertia_minor(self) -> float:
        """Minor-axis (weak, z-z) second moment Iz [m⁴]."""
        if self.inertia_minor_override is not None:
            return self.inertia_minor_override
        return section_inertia_minor(self.shape, self.b, self.h, self.tw, self.tf)

    @property
    def inertia(self) -> float:
        """In-plane bending inertia used by the solver [m⁴].

        With ``angle == 0`` this is the major-axis Iy (unchanged behaviour). A
        non-zero orientation rotates the profile: Iy·cos²θ + Iz·sin²θ.
        """
        iy = self.inertia_major
        if not self.angle:
            return iy
        import math
        iz = self.inertia_minor
        th = math.radians(self.angle)
        c, s = math.cos(th), math.sin(th)
        return iy * c * c + iz * s * s

    @property
    def torsion(self) -> float:
        """St-Venant torsion constant J [m⁴] — used only in the plate domain
        (grillage bars). Derived from the shape unless ``torsion_override``."""
        if self.torsion_override is not None:
            return self.torsion_override
        return section_torsion_constant(self.shape, self.b, self.h,
                                        self.tw, self.tf)

    # ── Design section moduli / shear areas (EC3) ─────────────────────────
    # Each returns the catalogue value when a commercial profile was applied,
    # otherwise the value computed from the section's shape and dimensions.
    @property
    def wel_y(self) -> float:
        """Elastic section modulus about the major axis Wel,y [m³]."""
        if self.wel_y_override is not None:
            return self.wel_y_override
        return self.inertia_major / (self.h / 2.0) if self.h else 0.0

    @property
    def wel_z(self) -> float:
        """Elastic section modulus about the minor axis Wel,z [m³]."""
        if self.wel_z_override is not None:
            return self.wel_z_override
        return self.inertia_minor / (self.b / 2.0) if self.b else 0.0

    @property
    def wpl_y(self) -> float:
        """Plastic section modulus about the major axis Wpl,y [m³]."""
        if self.wpl_y_override is not None:
            return self.wpl_y_override
        return section_plastic_moduli(self.shape, self.b, self.h,
                                      self.tw, self.tf)[0]

    @property
    def wpl_z(self) -> float:
        """Plastic section modulus about the minor axis Wpl,z [m³]."""
        if self.wpl_z_override is not None:
            return self.wpl_z_override
        return section_plastic_moduli(self.shape, self.b, self.h,
                                      self.tw, self.tf)[1]

    @property
    def av_y(self) -> float:
        """Shear area for a force parallel to the y axis A_v,y [m²]."""
        if self.av_y_override is not None:
            return self.av_y_override
        return section_shear_areas(self.shape, self.b, self.h,
                                   self.tw, self.tf)[0]

    @property
    def av_z(self) -> float:
        """Shear area for a force parallel to the z axis A_v,z [m²]."""
        if self.av_z_override is not None:
            return self.av_z_override
        return section_shear_areas(self.shape, self.b, self.h,
                                   self.tw, self.tf)[1]

    @property
    def radius_gyration_y(self) -> float:
        """Radius of gyration about the major axis iy = √(Iy/A) [m]."""
        a = self.area
        return (self.inertia_major / a) ** 0.5 if a > 0 else 0.0

    @property
    def radius_gyration_z(self) -> float:
        """Radius of gyration about the minor axis iz = √(Iz/A) [m]."""
        a = self.area
        return (self.inertia_minor / a) ** 0.5 if a > 0 else 0.0

    @property
    def warping(self) -> float:
        """Warping constant Iw [m⁶] — catalogue value or computed from shape."""
        if self.warping_override is not None:
            return self.warping_override
        return section_warping_constant(self.shape, self.b, self.h,
                                        self.tw, self.tf)

    @property
    def torsion_modulus(self) -> float:
        """Torsional section modulus Wt [m³] — catalogue value or from shape."""
        if self.wt_override is not None:
            return self.wt_override
        return section_torsion_modulus(self.shape, self.b, self.h,
                                       self.tw, self.tf)

    def buckling_curves(self, rolled: bool = True):
        """Return (curve_y, curve_z, curve_LT) — the section's buckling curves,
        using the stored y/z overrides (from a catalogue profile) where present
        and auto-selecting the rest from the shape."""
        cy, cz, clt = section_buckling_curves(self.shape, self.b, self.h,
                                              self.tf, rolled=rolled)
        return (self.curve_y_override or cy,
                self.curve_z_override or cz,
                clt)

    @property
    def is_catalogue_profile(self) -> bool:
        """True when this section was filled from a commercial catalogue
        profile (it carries a designation), False for a manual section."""
        return bool(self.profile_name)


@dataclass
class BarElement:
    id: str
    node_i: str           # left node ID
    node_j: str           # right node ID
    section_name: str
    # End releases (moment hinges). When True the bending moment is released
    # (set to zero) at that end of the element, decoupling the member's end
    # rotation from the node rotation.
    hinge_i:       bool  = False   # moment release at node_i end
    hinge_j:       bool  = False   # moment release at node_j end
    # RC design flags (optional)
    rc_design:     bool  = False
    rc_cover:      float = 0.05   # mechanical cover [m]
    rc_cotg_theta: float = 2.5     # cot(θ) for shear (EC2 truss model)
    rc_alpha_s:    float = 90.0    # stirrup angle [degrees]
    # Steel design (EC3) — buckling-length factors applied to the *physical
    # member* length (see member_utils). None → use the global default from
    # the Code preferences. ``sd_ltb`` marks the member as susceptible to
    # lateral-torsional buckling (unchecked when it is laterally restrained).
    sd_ky:  Optional[float] = None   # K for flexural buckling about y (major)
    sd_kz:  Optional[float] = None   # K for flexural buckling about z (minor)
    sd_klt: Optional[float] = None   # K for the lateral-torsional length
    sd_ltb: bool = True
    # RC column design (EC2 §5.8) — per-bar override of Section.is_column
    # (buckling/2nd-order design purposes, reusing sd_ky/sd_kz above for the
    # effective-length factors). ``None`` (the default) means "follow the
    # section's is_column"; an explicit True/False overrides it for this bar
    # only — e.g. a bar using a "column section" that actually acts as a
    # strut/brace here, or vice versa. ``rc_phi_ef``/``rc_n_bars``/
    # ``rc_second_order_method`` live on ``Section`` only (dev/
    # BUCKLING_COLUMN_PERSISTENCE.md) — no per-bar override for those.
    is_column: Optional[bool] = None
    # Beam membership (dev/BEAM_BARS_DEFINITION.md): the tag of the continuous
    # beam this bar belongs to, or None (= not assigned; the beam-bars design
    # then auto-detects). Persisted with the bar and copied to every sub-bar a
    # split creates; the beam's name/overrides live in ``Structure2D.beams``.
    beam: Optional[str] = None
    # Construction stage the element is built in (staged analysis). Stage k of
    # a stage-generated sequence activates every element with ``stage <= k``,
    # so growth is monotone by construction. Default 1 = present from the
    # start; the attribute travels with the element (rename/edit safe).
    stage: int = 1


@dataclass
class TriSection:
    """Section for CST plane (membrane) elements — the analogue of :class:`Section`
    for triangles. Its only geometric property is the thickness; the plane
    stress/strain assumption and the RC design parameters live here too, so that
    triangles reference a section exactly like bars do.
    """
    name: str
    material_name: str
    thickness: float = 0.1        # t [m] — the only geometry parameter
    plane_strain: bool = False    # False = plane stress (default)
    # Element formulation used by every TriElement referencing this section:
    #   "CST"    — constant-strain triangle, 2 DOF/node (ux, uy). Fast; the
    #              node's rotational DOF (tz) is left untouched (auto-restrained
    #              by the solver if the node is membrane-only).
    #   "Allman" — CST + drilling DOF (in-plane rotation) at each node, 3
    #              DOF/node (ux, uy, tz). Needed to transfer moment through a
    #              node shared with a BarElement, and more accurate in bending
    #              on coarse meshes. The tz stiffness is a numerical "drilling"
    #              artifact, not a real plate-bending stiffness — see
    #              tri_elements_allman.py.
    #   "DKT"    — plate bending (plate domain only): the Discrete Kirchhoff
    #              Triangle, thin-plate, 3 DOF/node (w, tx, ty). See
    #              tri_elements_dkt.py.
    #   "MITC3"  — plate bending (plate domain only): the shear-deformable
    #              (Mindlin-Reissner) MITC3 triangle, same 3 DOF/node, valid for
    #              thin and thick slabs — the default plate element. See
    #              tri_elements_mitc3.py.
    formulation: str = "CST"   # "CST" | "Allman" | "ES-FEM" | "DKT" | "MITC3"
    # RC design parameters (used when the material is concrete) — mirror Section.
    rc_cover:     float = 0.045   # mechanical cover [m] — the single/legacy value
    rc_alpha_s:   float = 90.0    # reinforcement inclination [degrees]
    rc_bar_phi:   float = 16.0    # main bar diameter [mm]
    rc_shear_min: bool  = True    # enforce EC2 minimum shear reinforcement
    # Per-face, per-direction covers for a slab's orthogonal reinforcement:
    # a slab is reinforced top and bottom, each in x and y, and the two
    # directions sit at different depths (the bars are layered). Each is
    # optional and falls back to ``rc_cover`` when unset — so a plane model (or
    # any model that only sets the single cover) keeps its behaviour, and only a
    # plate model needs the four. See :meth:`resolved_covers`.
    rc_cover_top_x: Optional[float] = None   # [m]
    rc_cover_top_y: Optional[float] = None   # [m]
    rc_cover_bot_x: Optional[float] = None   # [m]
    rc_cover_bot_y: Optional[float] = None   # [m]

    def resolved_covers(self) -> dict:
        """The four effective covers (top_x, top_y, bot_x, bot_y) in metres.

        Each per-face/direction cover if set, otherwise the single ``rc_cover``
        propagated to all four — so a section with only ``rc_cover`` behaves as
        four equal covers, and a slab section can override each independently."""
        c = self.rc_cover
        return {
            'top_x': self.rc_cover_top_x if self.rc_cover_top_x is not None else c,
            'top_y': self.rc_cover_top_y if self.rc_cover_top_y is not None else c,
            'bot_x': self.rc_cover_bot_x if self.rc_cover_bot_x is not None else c,
            'bot_y': self.rc_cover_bot_y if self.rc_cover_bot_y is not None else c,
        }


@dataclass
class TriElement:
    """3-node triangular plane element, plane stress (or plane strain).

    A membrane element. Whether it carries only the translational DOFs (ux,
    uy) of its three nodes ("CST") or also a drilling rotation DOF (tz,
    "Allman") is decided by the referenced :class:`TriSection`'s
    ``formulation`` field, not by this class — every TriElement referencing a
    given section uses that section's formulation. For a CST section, the
    rotational DOF (tz) at a node is untouched (fed by any frame member there,
    or auto-restrained by the solver when a node is membrane-only). Geometry,
    material and design come from the referenced :class:`TriSection`.
    """
    id: str
    node_i: str
    node_j: str
    node_k: str
    section_name: str
    # Construction stage the triangle is built in (staged analysis) — the
    # triangle twin of Element.stage, so a plate/wall region can be staged
    # exactly like a bar frame. Default 1 = present from the start.
    stage: int = 1


@dataclass
class QuadSection:
    """Section for 4-node quadrilateral elements — the analogue of
    :class:`TriSection` for quads (dev/IMPLEMENT_QUAD.md §3). Its own
    dataclass rather than a generalisation of ``TriSection``, matching how
    this codebase already keeps CST/Allman/DKT as separate modules instead of
    a generic N-node element type (dev/IMPLEMENT_QUAD.md §2).
    """
    name: str
    material_name: str
    thickness: float = 0.1        # t [m] — the only geometry parameter
    plane_strain: bool = False    # False = plane stress (default)
    # Element formulation used by every QuadElement referencing this section:
    #   "Q4"    — plain bilinear, 2 DOF/node (ux, uy). Kept for patch-test
    #             reference and comparison; known to lock in bending, not
    #             recommended as the default. See quad_elements.py.
    #   "QM6"   — Q4 + Wilson incompatible modes, same 2 DOF/node — the
    #             recommended membrane default. See quad_elements.py.
    #   "DKT4"  — plate bending (plate domain only): thin (Kirchhoff),
    #             3 DOF/node (w, tx, ty). See quad_elements_dkt4.py — a
    #             composite of 4 condensed DKT triangles fanned from the
    #             quad's centroid, not the literal Batoz-Tahar DKQ. Named
    #             "DKT4" rather than "DKQ" on purpose: it is not that
    #             formulation, and calling it DKQ would misrepresent what a
    #             user picking it actually gets — see that module's
    #             docstring for the full story.
    #   "MITC4" — plate bending (plate domain only): the shear-deformable
    #             (Mindlin-Reissner) MITC4 element, same 3 DOF/node, valid
    #             for thin and thick slabs — the default plate element. See
    #             quad_elements_mitc4.py.
    formulation: str = "MITC4"   # "Q4" | "QM6" | "DKT4" | "MITC4"
    # RC design parameters (used when the material is concrete) — duplicated
    # verbatim from TriSection rather than shared: same slab, same design
    # code, matching how TriSection itself duplicates Section's bar fields
    # (dev/IMPLEMENT_QUAD.md §3).
    rc_cover:     float = 0.045   # mechanical cover [m] — the single/legacy value
    rc_alpha_s:   float = 90.0    # reinforcement inclination [degrees]
    rc_bar_phi:   float = 16.0    # main bar diameter [mm]
    rc_shear_min: bool  = True    # enforce EC2 minimum shear reinforcement
    # Per-face, per-direction covers for a slab's orthogonal reinforcement —
    # see TriSection.resolved_covers for the same rationale.
    rc_cover_top_x: Optional[float] = None   # [m]
    rc_cover_top_y: Optional[float] = None   # [m]
    rc_cover_bot_x: Optional[float] = None   # [m]
    rc_cover_bot_y: Optional[float] = None   # [m]

    def resolved_covers(self) -> dict:
        """The four effective covers (top_x, top_y, bot_x, bot_y) in metres —
        identical logic to :meth:`TriSection.resolved_covers`."""
        c = self.rc_cover
        return {
            'top_x': self.rc_cover_top_x if self.rc_cover_top_x is not None else c,
            'top_y': self.rc_cover_top_y if self.rc_cover_top_y is not None else c,
            'bot_x': self.rc_cover_bot_x if self.rc_cover_bot_x is not None else c,
            'bot_y': self.rc_cover_bot_y if self.rc_cover_bot_y is not None else c,
        }


@dataclass
class QuadElement:
    """4-node quadrilateral element (membrane or plate bending, per its
    referenced :class:`QuadSection`'s ``formulation``) — the analogue of
    :class:`TriElement`.

    Nodes must be wound counter-clockwise, ``node_i → node_j → node_k →
    node_l``. This is checked *eagerly*, at ``Structure2D.add_quad_element``
    time (dev/IMPLEMENT_QUAD.md Phase 3, §5 point 2: "err toward falling back
    ... rather than accepting a bad quad silently" — here that means
    rejecting with a clear message rather than silently reordering the
    nodes, since a silent reorder could disagree with what the caller
    intended and mask a real modelling mistake)."""
    id: str
    node_i: str
    node_j: str
    node_k: str
    node_l: str
    section_name: str
    # Construction stage the quad is built in (staged analysis) — the quad
    # twin of TriElement.stage.
    stage: int = 1


@dataclass
class TriEdgeLoad:
    """Uniform edge load on one edge (node_a→node_b) of a CST triangle,
    distributed to the two edge nodes as consistent nodal loads.

    ``coord_sys='global'`` (default): ``fx``/``fy`` are force per unit length
    [kN/m] along the global X/Y axes.
    ``coord_sys='local'`` (future): ``pn``/``pt`` are pressures [kN/m²] along the
    edge's right-hand normal and its tangent (a→b), multiplied by the thickness.
    """
    id: str
    tri_id: str
    node_a: str
    node_b: str
    load_case_id: str
    fx: float = 0.0
    fy: float = 0.0
    coord_sys: str = "global"
    pn: float = 0.0
    pt: float = 0.0


@dataclass
class QuadEdgeLoad:
    """Uniform edge load on one edge (node_a→node_b) of a quad element — the
    4-node analogue of :class:`TriEdgeLoad`, distributed to the two edge nodes
    as consistent nodal loads (½·L each). Same convention: ``coord_sys='global'``
    uses ``fx``/``fy`` [kN/m]; plate domain uses ``fx`` as the transverse line
    load fz [kN/m] on the two w DOFs.
    """
    id: str
    quad_id: str
    node_a: str
    node_b: str
    load_case_id: str
    fx: float = 0.0
    fy: float = 0.0
    coord_sys: str = "global"
    pn: float = 0.0
    pt: float = 0.0


@dataclass
class SurfaceEdgeLoad:
    """Uniform edge load on one boundary side (node_a→node_b, two consecutive
    boundary vertices) of a surface object. At solve time it is distributed as
    consistent nodal loads over the boundary mesh nodes along that side.

    Same convention as :class:`TriEdgeLoad`: ``coord_sys='global'`` uses
    ``fx``/``fy`` [kN/m]; ``'local'`` (future) uses ``pn``/``pt``.
    """
    id: str
    object_id: str
    node_a: str
    node_b: str
    load_case_id: str
    fx: float = 0.0
    fy: float = 0.0
    coord_sys: str = "global"
    pn: float = 0.0
    pt: float = 0.0


@dataclass
class TriAreaLoad:
    """Uniform transverse pressure on one plate (DKT) triangle.

    ``pz`` [kN/m²] acts along global Z (negative downward) and is lumped to
    the three vertices as pz·A/3 on the w DOF. Plate domain only.
    """
    tri_id: str
    load_case_id: str
    pz: float = 0.0


@dataclass
class SurfaceAreaLoad:
    """Uniform transverse pressure on a surface object (a slab region).

    At solve time the surface is meshed and the pressure becomes one
    :class:`TriAreaLoad` per generated triangle. Plate domain only.
    """
    object_id: str
    load_case_id: str
    pz: float = 0.0


@dataclass
class QuadAreaLoad:
    """Uniform transverse pressure on one plate (DKT4/MITC4) quadrilateral —
    the 4-node analogue of :class:`TriAreaLoad` (dev/IMPLEMENT_QUAD.md
    Phase 4).

    ``pz`` [kN/m²] acts along global Z (negative downward) and is lumped to
    the four vertices as pz·A/4 on the w DOF. Plate domain only.
    """
    quad_id: str
    load_case_id: str
    pz: float = 0.0


@dataclass
class Support:
    """Boundary condition attached to a node."""
    name: str
    ux: bool = False   # restrain x-translation
    uy: bool = False   # restrain y-translation
    tz: bool = False   # restrain z-rotation


#: A name for each way of restraining a node, so the same boundary condition
#: is called the same thing wherever it was made.
#:
#: The two ways of writing a model had drifted apart. Written by hand or by an
#: assistant, a model has a couple of shared definitions — PIN, ROLLER — put on
#: many nodes. Drawn in the application, it had one definition per supported
#: node, named after that node, so a wall base produced forty near-identical
#: entries and the idea of a support *type* meant nothing. The two describe the
#: same structure and do not look alike, and one consequence was concrete:
#: propagating a support along the edge of a meshed region matches by name, so
#: it fired on models an assistant wrote and never on models anyone drew.
#:
#: The suffix is the direction the node can still MOVE in — ROLLER-X rolls
#: along x, so ux is free and uy is held. Named for the movement rather than
#: for the restraint because that is how the support is pictured, and a name
#: that has to be guessed is the failure this project keeps having.
CANONICAL_SUPPORTS = {
    #  (ux,    uy,    tz)      restrained = True
    (True,  True,  False): 'PIN',        # both translations, free to rotate
    (True,  True,  True):  'FIXED',      # everything: encastré
    (False, True,  False): 'ROLLER-X',   # rolls along x
    (True,  False, False): 'ROLLER-Y',   # rolls along y
    (False, True,  True):  'GUIDED-X',   # slides along x, cannot rotate
    (True,  False, True):  'GUIDED-Y',   # slides along y, cannot rotate
    (False, False, True):  'BLOCK',      # rotation only; free to translate
}

#: The plate domain reuses the three DOF slots for (w, θx, θy), so the same
#: restraint triple means something entirely different and needs its own names.
#: Slabs/grillages are pictured as edges, not point machines, so the vocabulary
#: is the slab-edge one: SIMPLE (w held, free to rotate), CLAMPED (encastré),
#: CLAMP-X/Y (a supported edge that also holds one rotation), SYM-X/Y (a
#: symmetry line: w free, one rotation held), BLOCK (both rotations, w free).
CANONICAL_SUPPORTS_PLATE = {
    #  (w,     θx,    θy)       restrained = True
    (True,  False, False): 'SIMPLE',    # deflection held, free to rotate
    (True,  True,  True):  'CLAMPED',   # everything: encastré
    (True,  True,  False): 'CLAMP-X',   # supported edge + θx held
    (True,  False, True):  'CLAMP-Y',   # supported edge + θy held
    (False, True,  True):  'BLOCK',     # both rotations held; free to deflect
    (False, True,  False): 'SYM-X',     # symmetry line: w free, θx held
    (False, False, True):  'SYM-Y',     # symmetry line: w free, θy held
}


def canonical_support_name(ux: bool, uy: bool, tz: bool,
                           domain: str = 'plane') -> str | None:
    """The standard name for a set of restraints, or None for no restraint.

    The three slots are (ux, uy, tz) in the plane domain and (w, θx, θy) in the
    plate domain — same storage, different meaning — so the name depends on the
    domain. None rather than a name for (False, False, False): that is not a
    support, and giving it one would put an entry in the table for a free node.
    """
    table = CANONICAL_SUPPORTS_PLATE if domain == 'plate' else CANONICAL_SUPPORTS
    return table.get((bool(ux), bool(uy), bool(tz)))


@dataclass
class SupportAssignment:
    node_id: str
    support_name: str


@dataclass
class NodeSpring:
    """Translational/rotational spring attached directly to a node.

    The ``mode_*`` fields control unilateral (tension-/compression-only)
    behaviour of the two translational components and are honoured **only by
    NonLinear analysis cases**. Linear cases always treat the spring as
    bilateral (as if every mode were 'both'), preserving existing behaviour.

    Sign convention (per global axis): 'tension' means the spring is active
    only when the node displacement along that axis is positive (the spring
    is stretched); 'compression' means active only when the displacement is
    negative; 'both' is the classic bilateral spring.
    """
    node_id: str          # key — same as the node it belongs to
    kx: float = 0.0      # global X stiffness [kN/m]
    ky: float = 0.0      # global Y stiffness [kN/m]
    kt: float = 0.0      # rotational stiffness [kNm/rad]
    mode_x: str = 'both'  # 'both' | 'tension' | 'compression'
    mode_y: str = 'both'  # 'both' | 'tension' | 'compression'


@dataclass
class ElementSpring:
    """Distributed (Winkler) foundation spring attached directly to an element.

    The ``mode_*`` fields control unilateral (tension-/compression-only)
    behaviour and are honoured **only by NonLinear analysis cases** (Linear
    cases always treat the spring as bilateral). The unilateral state is
    evaluated per end-node, consistent with the lumped ``k·L/2`` model used in
    assembly. For ``coord_sys='global'`` the modes apply to the X (kx) and Y
    (ky) components; for ``coord_sys='local'`` they apply to the axial (kx) and
    transverse (ky) components.

    Sign convention: 'tension' is active when the end-node displacement
    projected on the component direction is positive; 'compression' when it is
    negative; 'both' is bilateral.

    Plate domain: the spring resists the transverse deflection w only (beam on
    elastic foundation); ``ky`` is the transverse stiffness per unit length
    [kN/m²] — the same slot that is transverse for a 'local' spring in the
    plane domain — and ``kx`` is ignored. Unilateral modes are not available
    in the plate domain yet (see Structure2D.domain_problems).
    """
    element_id: str          # key — same as the element it belongs to
    kx: float = 0.0          # stiffness per unit length, X [kN/m²]
    ky: float = 0.0          # stiffness per unit length, Y [kN/m²]
    coord_sys: str = "global"  # 'global' (X,Y) or 'local' (axial, transverse)
    mode_x: str = 'both'     # 'both' | 'tension' | 'compression'  (X / axial)
    mode_y: str = 'both'     # 'both' | 'tension' | 'compression'  (Y / transverse)


@dataclass
class TriAreaSpring:
    """Winkler (elastic-foundation) area spring on one plate (DKT) triangle.

    ``kz`` [kN/m³] is the modulus of subgrade reaction resisting the transverse
    deflection w — a slab-on-grade support. It is lumped kz·A/3 onto the w DOF
    of each of the triangle's three vertices, exactly like the pz area load, so
    the two share the same A/3 tributary rule. Plate domain only.
    """
    tri_id: str
    kz: float = 0.0


@dataclass
class QuadAreaSpring:
    """Winkler (elastic-foundation) area spring on one plate (DKT4/MITC4)
    quadrilateral — the 4-node analogue of :class:`TriAreaSpring`
    (dev/IMPLEMENT_QUAD.md Phase 6, added after Phase 6's investigation flagged
    the missing model).

    ``kz`` [kN/m³] is lumped kz·A/4 onto the w DOF of each of the quad's four
    vertices, the same tributary rule :class:`QuadAreaLoad` uses for pz. Plate
    domain only.
    """
    quad_id: str
    kz: float = 0.0


@dataclass
class SurfaceAreaSpring:
    """Winkler area spring on a surface object (a slab-on-grade region).

    At solve time the surface is meshed and the modulus becomes one
    :class:`TriAreaSpring` per generated triangle. Plate domain only.
    """
    object_id: str
    kz: float = 0.0


@dataclass
class PunchColumn:
    """A column / concentrated support under a slab, for the EC2 §6.4 punching
    check (plate domain only).

    The slab FEM does not know a column's size or shape, so the punching check
    needs it as explicit data: which node the column sits under, its plan shape
    and dimensions, its position relative to the slab edge, and where the
    punching force comes from.

    - ``node_id``  : the slab node the column reaction acts on.
    - ``shape``    : 'rectangular' or 'circular'.
    - ``bx``, ``by`` : plan dimensions [m] (``bx`` is the diameter when circular;
                     ``by`` is then ignored).
    - ``position`` : 'center', 'edgex', 'edgey' or 'corner' — the control
                     perimeter differs at a free edge or corner.
    - ``force``    : design punching force [kN]; ``None`` uses the support
                     reaction at ``node_id`` for the designed combination.
    - ``dx``, ``dy`` : distance from the column to the slab edge [m] (edge/corner).
    """
    id: str
    node_id: str
    shape: str = "rectangular"      # 'rectangular' | 'circular'
    bx: float = 0.30                # [m] (diameter if circular)
    by: float = 0.30                # [m]
    position: str = "center"        # 'center' | 'edgex' | 'edgey' | 'corner'
    force: Optional[float] = None   # [kN]; None → use the reaction at node_id
    dx: float = 0.0                 # [m] distance to x edge
    dy: float = 0.0                 # [m] distance to y edge
    # Minimum β (load-eccentricity factor). When a support transfers no moment
    # the computed β falls to its floor (~1.05); EC2 §6.4.3(6) allows the
    # simplified values 1.15 (interior), 1.4 (edge) and 1.5 (corner) instead.
    # The design uses max(computed β, beta_min); 1.0 leaves the computed value.
    beta_min: float = 1.0


# EN 1995-1-1 Table 2.1 load-duration classes, shortest first. The stored
# value of LoadCase.load_duration is one of these, or "" for automatic.
LOAD_DURATIONS = ("Instantaneous", "ShortDuration", "MediumDuration",
                  "LongDuration", "Permanent")


def coerce_load_duration(value) -> str:
    """A load-duration class name from whatever someone meant: the class
    itself, or the short word ("short", "medium", "long", "perm", ...), in any
    case. None / "" / "auto" mean automatic and give "". Anything else raises
    ValueError naming the valid values."""
    if value is None:
        return ""
    text = str(value).strip().lower().replace(" ", "").replace("-", "")
    text = text.replace("_", "").replace("term", "").replace("duration", "")
    if text in ("", "auto", "automatic", "none"):
        return ""
    for name in LOAD_DURATIONS:
        key = name.lower().replace("duration", "")
        if text == key or (len(text) >= 4 and key.startswith(text)):
            return name
    raise ValueError(
        f"Unknown load duration {value!r}; use one of "
        f"{', '.join(LOAD_DURATIONS)} (or '' for automatic).")


@dataclass
class LoadCase:
    id: str
    self_weight_factor: float = 0.0   # PESO_PROP
    action_type: ActionType = ActionType.G
    # Timber load-duration class (EN 1995-1-1 Table 2.1) fixing k_mod. ""
    # means automatic: derived from action_type. Read only by the timber design.
    load_duration: str = ""
    # EN 1990 factors. ``category`` picks the row of the defaults table (the
    # EN 1991-1-1 usage category "A".."J" of a live load, the snow altitude
    # variant "<=1000"/">1000"); "" is the generic row of the action type.
    # ``psi0/psi1/psi2`` override the default for this case; None keeps it.
    # The defaults themselves come from eurocodepy (see combinations.py).
    category: str = ""
    psi0: Optional[float] = None
    psi1: Optional[float] = None
    psi2: Optional[float] = None

    # Hashable by id, so the object can be a key of a coefficients dict
    # ({lc: 1.35}) as the id can; a dataclass with eq and no __hash__ is not.
    def __hash__(self):
        return hash(self.id)


@dataclass
class PointLoad:
    """Nodal concentrated load."""
    node_id: str
    load_case_id: str
    fx: float = 0.0
    fy: float = 0.0
    mz: float = 0.0


@dataclass
class DistributedLoad:
    """Trapezoidal distributed load on a bar element."""
    element_id: str
    load_case_id: str
    fxe: float = 0.0        # axial load at start (left)
    fxd: float = 0.0        # axial load at end  (right)
    fye: float = 0.0        # transverse load at start
    fyd: float = 0.0        # transverse load at end
    coord_sys: str = 'global'  # 'global' or 'local'


@dataclass
class ElementPointLoad:
    """Concentrated load applied on a bar element at distance ``a`` from node i.

    ``a`` is the distance [m] measured from the i-end along the element axis
    (0 < a < L). Forces are given in global axes by default, or in the element
    local axes (x' = axis i→j, y' = perpendicular) when ``coord_sys='local'``.
    """
    element_id: str
    load_case_id: str
    a: float = 0.0          # distance from i-end along the element [m]
    fx: float = 0.0         # force along global X (or local x' if coord_sys='local')
    fy: float = 0.0         # force along global Y (or local y')
    mz: float = 0.0         # concentrated moment about Z [kNm]
    coord_sys: str = 'global'  # 'global' or 'local'


@dataclass
class LoadCombination:
    id: str
    coefficients: dict[str, float] = field(default_factory=dict)  # {load_case_id: coeff}
    analysis_coefficients: dict[str, float] = field(default_factory=dict)  # {analysis_case_id: coeff}
    combo_type: str = 'LinearSum'  # LinearSum | Envelope | AbsSum | SRSS | NonLinearCombo

    # Hashable by id, so the object can key a coefficients dict as its id can.
    def __hash__(self):
        return hash(self.id)


@dataclass
class Cut:
    """User-defined section cut: a straight segment ``(x1,y1) -> (x2,y2)``
    across the model, used to report the resultant of the internal forces
    (bars + areas) transferred across it. See dev/CUT_PLAN.md."""
    id:   str
    name: str = ""
    x1:   float = 0.0
    y1:   float = 0.0
    x2:   float = 0.0
    y2:   float = 0.0


@dataclass
class AnalysisCase:
    """
    Defines a type of analysis to be performed on the structure.

    analysis_type:
      'Linear'             — linear static combination of load cases
      'NonLinear'          — nonlinear case; same inputs as Linear (currently
                             solved identically to Linear). Subject to special
                             combination rules: a combination containing a
                             NonLinear case may hold only that single case with
                             factor 1.0 and no other analysis cases.
      'Mass'               — mass definition for modal analysis (combination of load cases)
      'Modal'              — eigenvalue / free-vibration analysis  (not yet implemented)
      'Spectrum'           — response-spectrum seismic analysis    (not yet implemented)
      'GeometricNonlinear' — P-Delta / large-displacement          (not yet implemented)
      'Sequence'           — one phase of a construction sequence. Subject to the
                             same single-case combination rule as NonLinear: may
                             only appear in a SequenceCombo with factor 1.0.
                             sequence_id and phase_id identify the source phase.
    """
    id: str
    analysis_type: str = 'Linear'

    # ── Load case coefficients (Linear and Mass) ──────────────────────
    # {load_case_id: factor}
    coefficients: dict = field(default_factory=dict)

    # ── Modal parameters ──────────────────────────────────────────────
    num_modes:        int   = 10
    modal_case_id:    str   = ''         # ID of the Modal AnalysisCase to use
    spectrum_id:      str   = ''         # ID of the SpectralFunction
    combination_rule: str   = 'SRSS'    # 'SRSS' | 'CQC'
    direction:        str   = 'XY'      # 'X' | 'Y' | 'XY'
    damping:          float = 0.05      # damping ratio for CQC correlation

    # ── Geometric nonlinear parameters ────────────────────────────────
    nl_method:               str   = 'P-Delta'
    max_iterations:          int   = 100
    tolerance:               float = 1.0e-6
    # Stiffness reduction factors for ANLG (fraction of elastic EI)
    beam_stiffness_factor:   float = 1.0   # beams: inclination < 45°
    column_stiffness_factor: float = 1.0   # columns: inclination ≥ 45°

    # ── Stored stiffness (for Linear cases that reuse an ANLG matrix) ─
    stored_stiffness_id: str = ''   # ID of GeometricNonlinear case whose K to reuse

    # ── Sequence back-reference (analysis_type == 'Sequence' only) ───────
    # Identifies which construction sequence and phase this AC represents.
    sequence_id: str = ''
    phase_id:    str = ''   # '__final__' for the accumulated final state

    # Whether this case has current results. A new or edited case starts
    # unsolved; the solver sets it True. It lets a solved (locked) model gain a
    # case that is run on its own — the results dropdowns show only the solved
    # ones, and Run computes just the pending ones (see solver.solve_pending).
    solved: bool = False

    # Hashable by id, so the object can key a coefficients dict as its id can.
    def __hash__(self):
        return hash(self.id)


@dataclass
class SpectralFunction:
    """Response spectrum Sa(T) defined by a list of (period, acceleration) pairs."""
    id:          str
    description: str   = ""
    damping:     float = 0.05          # reference damping ratio (used as label/doc)
    points:      list  = field(default_factory=list)  # [[T [s], Sa [g or m/s²]], ...]


@dataclass
class NodalMass:
    """Concentrated mass added directly to a node for a specific Mass analysis case."""
    node_id:      str
    mass_case_id: str          # ID of the Mass AnalysisCase
    mx:   float = 0.0          # translational mass X [t]
    my:   float = 0.0          # translational mass Y [t]
    mtz:  float = 0.0          # rotational inertia   [t·m²]


@dataclass
class SupportSettlement:
    """Prescribed displacement/rotation at a support node."""
    node_id: str
    load_case_id: str
    ux: float = 0.0   # prescribed x-displacement [m]
    uy: float = 0.0   # prescribed y-displacement [m]
    tz: float = 0.0   # prescribed rotation [rad]


@dataclass
class TemperatureLoad:
    """Thermal load on a bar element."""
    element_id: str
    load_case_id: str
    delta_t_uniform: float = 0.0    # uniform temperature change ΔT [°C]
    delta_t_gradient: float = 0.0   # linear gradient (T_top - T_bottom) [°C]


@dataclass
class TriTemperatureLoad:
    """Thermal load on a CST triangle, given as ΔT at each of its three nodes.

    In the plane domain the three nodal values are an *in-plane* temperature: a
    bar carries a through-thickness *gradient* that bends it, and a membrane
    triangle has no bending, so ``dt_i/dt_j/dt_k`` drive only the in-plane
    expansion (a CST feels their mean). A uniform rise is the case
    dt_i = dt_j = dt_k.

    In the plate domain the DKT *does* bend, so the same class carries the
    through-thickness gradient ``dt_gradient`` = T_top − T_bottom [°C], which
    produces a free thermal curvature κ₀ = α·ΔT/t (the direct plate analogue of
    the bar's ``delta_t_gradient``). The in-plane ``dt_i/dt_j/dt_k`` are
    stress-free in a plate (a mean rise stretches the mid-surface, which the
    plate domain does not model) and are ignored there.

    The per-node in-plane values allow a temperature that varies across a mesh
    (each triangle taking the values at its own corners). Within a *single* CST
    the element only feels their mean, because its strain is constant — the
    effect of a gradient appears only once the mesh is fine enough to resolve
    it. The per-node form is kept so a finer mesh, or a future higher-order
    element, reads exactly the same data.
    """
    tri_id: str
    load_case_id: str
    dt_i: float = 0.0
    dt_j: float = 0.0
    dt_k: float = 0.0
    dt_gradient: float = 0.0   # plate domain: T_top − T_bottom [°C] (bends DKT)

    @property
    def dt_mean(self) -> float:
        """The mean of the three nodal values — all a CST responds to."""
        return (self.dt_i + self.dt_j + self.dt_k) / 3.0


@dataclass
class QuadTemperatureLoad:
    """Thermal load on a Q4/QM6/DKT4/MITC4 quad, given as ΔT at each of its
    four nodes — the 4-node analogue of :class:`TriTemperatureLoad`
    (dev/IMPLEMENT_QUAD.md Phase 6, added after Phase 6's own investigation
    flagged "quads carry no thermal-load feature at all" as a gap).

    In the plane domain (Q4/QM6) the four nodal values are an *in-plane*
    temperature, exactly like a triangle's: Q4/QM6 has no bending, so
    ``dt_i/dt_j/dt_k/dt_l`` drive only the in-plane expansion. Unlike a CST's
    exactly-constant strain, a Q4/QM6's strain-displacement matrix varies over
    the element, so the equivalent nodal load is Gauss-integrated rather than
    a single area×mean-strain product (see ``quad_elements.q4_thermal_load``/
    ``qm6_thermal_load``) — but the *free strain field itself* is still taken
    as uniform over the element, equal to the mean of the four corner values
    (``dt_mean``), the same simplification ``TriTemperatureLoad`` makes for a
    CST: this class stores one ΔT value per element in the ``thermal`` dict
    used throughout stress/moment recovery, not a spatially varying field.

    In the plate domain (DKT4/MITC4) the same class carries the
    through-thickness gradient ``dt_gradient`` = T_top − T_bottom [°C], the
    direct analogue of ``TriTemperatureLoad``'s plate-domain field — DKT4 (a
    condensed fan of 4 DKT triangles) and MITC4 both derive their free
    thermal curvature the same isotropic way a DKT triangle does. The
    in-plane ``dt_i/dt_j/dt_k/dt_l`` are ignored in the plate domain,
    mirroring ``TriTemperatureLoad``.
    """
    quad_id: str
    load_case_id: str
    dt_i: float = 0.0
    dt_j: float = 0.0
    dt_k: float = 0.0
    dt_l: float = 0.0
    dt_gradient: float = 0.0   # plate domain: T_top − T_bottom [°C] (bends DKT4/MITC4)

    @property
    def dt_mean(self) -> float:
        """The mean of the four nodal values — see the class docstring for
        why only the mean (not the per-node field) drives the element."""
        return (self.dt_i + self.dt_j + self.dt_k + self.dt_l) / 4.0


@dataclass
class AreaTemperatureLoad:
    """A temperature on an *area object* (rectangle or surface), applied to the
    triangles it meshes into at solve time.

    The object's triangles do not exist until :func:`expand_geometry` runs, so
    the load is stored against the object and expanded there — the same shape
    as a surface edge load. ``field_name`` empty means a uniform
    ``dt_uniform`` at every node; a field name means the temperature is the
    field sampled at each generated mesh node, which is where a gradient
    actually resolves, because the mesh is as fine as the object's target size.

    In the plate domain the object also carries a through-thickness gradient
    ``dt_gradient`` = T_top − T_bottom [°C] (a fixed value, or ``grad_field_name``
    sampled per triangle), passed straight to each generated DKT triangle — the
    area-object counterpart of :class:`LineTemperatureLoad`'s ``dt_gradient``.
    """
    object_id: str
    load_case_id: str
    dt_uniform: float = 0.0
    field_name: str = ""
    dt_gradient: float = 0.0
    grad_field_name: str = ""


@dataclass
class LineTemperatureLoad:
    """A temperature on a *line object* (line / arc / polyline), applied to the
    bars it subdivides into at solve time.

    The line-object counterpart of :class:`AreaTemperatureLoad`. A bar carries
    two thermal effects and both are here: ``dt_uniform`` (axial) and
    ``dt_gradient`` = T_top - T_bottom (curvature). Either may be a fixed value
    or come from a field, independently — the same field or different ones. A
    field is a function of (x, y), sampled at each generated bar's midpoint: for
    the uniform part it is the axial temperature there; for the gradient it is
    how the gradient *magnitude* varies in space (the through-depth profile is
    always linear, its size is what the field sets). An empty ``field_name`` /
    ``grad_field_name`` means the corresponding ``dt_*`` value is used instead.
    """
    object_id: str
    load_case_id: str
    dt_uniform: float = 0.0
    dt_gradient: float = 0.0
    field_name: str = ""
    grad_field_name: str = ""


@dataclass
class LineDistributedLoad:
    """A distributed load on a *line object* (line / arc / polyline), applied to
    the bars it subdivides into at solve time.

    The counterpart of :class:`DistributedLoad` for an object. Each direction is
    a constant value or a field, independently — the same field or two different
    ones. A constant makes a uniform load (both ends equal); a field is sampled
    at each generated bar's two end nodes, so the trapezoid follows the field
    along the object. ``coord_sys`` is 'global' (fx, fy) or 'local' (axial,
    transverse), exactly as for a bar. An empty field name uses the numeric
    value.
    """
    object_id: str
    load_case_id: str
    fx: float = 0.0
    fy: float = 0.0
    fx_field: str = ""
    fy_field: str = ""
    coord_sys: str = "global"


@dataclass
class LineElementSpring:
    """A foundation spring on a *line object* (line / arc / polyline), applied to
    the bars it subdivides into at solve time.

    The object counterpart of :class:`ElementSpring`. kx and ky are stiffness
    per unit length, each a constant or a field (the field's value at the bar
    midpoint — a Winkler modulus that varies along the object). No load case: a
    spring is a structural property, not a load. coord_sys and the unilateral
    modes carry through unchanged.
    """
    object_id: str
    kx: float = 0.0
    ky: float = 0.0
    kx_field: str = ""
    ky_field: str = ""
    coord_sys: str = "global"
    mode_x: str = "both"
    mode_y: str = "both"


#: The three DOF component names of a node, in the order the solver numbers them
#: (ux -> +0, uy -> +1, tz -> +2). A constraint term names a component by one of
#: these strings so a model written by hand or by an assistant reads the same way
#: it solves. In the plate domain the same three slots are named
#: (w -> +0, tx -> +1, ty -> +2); both spellings are accepted everywhere a
#: component is named, so a plate model does not have to call its deflection
#: 'ux' to be understood.
CONSTRAINT_COMPONENTS = ("ux", "uy", "tz", "w", "tx", "ty")

#: Offset added to a node's base DOF index to reach each component. The plate
#: names alias the same offsets as the plane names — the storage is positional.
COMPONENT_OFFSET = {"ux": 0, "uy": 1, "tz": 2,
                    "w": 0, "tx": 1, "ty": 2}


@dataclass
class Constraint:
    """A linear multi-point constraint (MPC) relating several nodal DOFs.

    The canonical, internal form is a single linear equation::

        sum(coef_i * dof_i  for i in terms) = value

    where each *term* is ``(node_id, component, coef)`` and *component* is one of
    :data:`CONSTRAINT_COMPONENTS` ('ux', 'uy', 'tz'). Every high-level kind is
    expanded into one or more equations of this form by
    :func:`xdfem2d.structure.expand_constraints`; the solver never sees anything
    but equations.

    ``kind`` records how the constraint was authored so the interface can show it
    back the way it was made (the same idea as :data:`CANONICAL_SUPPORTS`):

      'equation'   — ``terms`` / ``value`` used directly (advanced / AI-written).
      'rigid_link' — ``master`` node and ``slaves`` nodes move as one rigid body;
                     the equations are generated from the nodal coordinates.
      'equal_dof'  — every node in ``nodes`` shares the same value of each
                     component in ``components``.

    (A rigid-diaphragm shortcut is a planned future ``kind`` that expands to
    equations exactly like ``rigid_link``; it needs no change to this class.)

    A disabled constraint (``enabled=False``) is stored but ignored by the solver.
    """
    id: str
    kind: str = "equation"            # 'equation' | 'rigid_link' | 'equal_dof'
    # canonical form: sum(coef * dof) = value
    terms: list = field(default_factory=list)   # list[(node_id, component, coef)]
    value: float = 0.0
    # convenience fields for the high-level shortcuts:
    master: str = ""                  # rigid_link master node id
    slaves: list = field(default_factory=list)  # rigid_link slave node ids
    nodes: list = field(default_factory=list)   # equal_dof node ids
    components: list = field(default_factory=list)  # equal_dof components
    enabled: bool = True


@dataclass
class ConcreteMaterial:
    """EC2 concrete/steel design parameters linked to a material."""
    material_name: str
    concrete_class: str   # e.g. 'C25/30'
    steel_class:    str   # e.g. 'A500'
    gamma_c:  float = 1.5   # concrete partial safety factor
    gamma_s:  float = 1.15  # steel partial safety factor
    alpha_cc: float = 1.0   # long-term strength reduction factor


@dataclass
class SupportSet:
    """A named alternative set of boundary conditions for a structure.

    In xdfem2D a support is defined *per node* by its restraints (ux, uy, tz);
    there are no shared, user-named support types. A SupportSet therefore stores
    restraints keyed by node id::

        restraints = {node_id: (ux, uy, tz)}   # booleans

    When a variant/phase using this set is derived, the structure's supports are
    rebuilt from these restraints (replace semantics: the set lists *all* the
    supported nodes for that scenario; nodes not listed are free).

    ``assignments`` (referencing named supports in ``struc.supports``) is kept for
    backward compatibility / programmatic use; when ``restraints`` is non-empty it
    takes precedence. ``node_springs`` / ``element_springs`` replace the base
    springs when the variant is derived.
    """
    id: str
    description: str = ""
    restraints:      dict = field(default_factory=dict)   # {node_id: (ux,uy,tz)}
    assignments:     list = field(default_factory=list)   # list[SupportAssignment]
    node_springs:    dict = field(default_factory=dict)   # {node_id: NodeSpring}
    element_springs: dict = field(default_factory=dict)   # {element_id: ElementSpring}


@dataclass
class ElementInitialState:
    """Inherited (frozen) member end forces of an element, local convention
    ``[N, V, M]`` at each end, matching ``element_forces`` in compute_results.

    Used by construction phases (Mode B): an element built in an earlier phase
    carries this locked-in state; the new phase's increment is added to it.
    """
    i: tuple = (0.0, 0.0, 0.0)
    j: tuple = (0.0, 0.0, 0.0)


class OpAction(str, Enum):
    """What an :class:`Operation` does to its target."""
    ADD = "add"
    REMOVE = "remove"


class OpTarget(str, Enum):
    """What kind of thing an :class:`Operation` acts on.

    There is deliberately no SPRINGS target: a :class:`SupportSet` already
    carries its own ``node_springs``/``element_springs`` alongside
    ``restraints`` (applied together by ``Structure2D.derive_variant``), so
    a support set is "the whole support condition for this phase" — fixed
    restraints, springs, or both. An ADD/REMOVE SUPPORTS operation now covers
    springs too; there is nothing left for a separate target to do.
    """
    ELEMENTS = "elements"    # a named GROUP of bars/triangles — see group_kind
    SUPPORTS = "supports"    # a SupportSet (whole-set add/remove; may carry springs)
    LOADS = "loads"          # a load case's contribution (applied_cases)


@dataclass
class Operation:
    """One authored step within a :class:`ConstructionPhase` — "add Stage 2",
    "remove support set X", "add load case Y".

    A phase's ``active_elements``/``support_set_id``/``applied_cases`` are no
    longer authored directly; they are *resolved* by replaying a phase's
    ``operations`` on top of the previous phase's resolved state (see
    :func:`xdfem2d.phasing.resolve_phase_state`). This makes a phase an
    explicit, ordered log of what changed, instead of a bare declaration of
    the resulting state.

    Elements can only be added/removed by GROUP — a construction Stage or a
    (GUI-only) Scene — never by picking individual ids, so an operation stays
    a legible, reusable statement ("add Stage 2") instead of an opaque list
    of bar ids.

    id         : internal, unique within the phase (e.g. "op_1").
    action     : ADD or REMOVE.
    target     : ELEMENTS / SUPPORTS / LOADS — what ``ids`` refers to.
    group_kind : only meaningful for ELEMENTS — "stage" or "scene" (empty
        for SUPPORTS/LOADS, which are already a single named group).
        - "stage": ``ids`` is a 1-tuple with the stage NUMBER (as a string).
          Resolved dynamically against the model's current ``stage``
          attribute every time the phase is replayed (by
          :func:`xdfem2d.phasing.resolve_phase_state`) — if elements are
          re-tagged to that stage later, this operation picks them up
          automatically, no re-editing needed.
        - "scene": a scene is GUI/view state (saved in view.json, not part
          of the engine model — see MainWindow._scenes), so the *engine*
          cannot resolve a scene name on its own. ``ids`` holds the
          element ids the scene contained at the moment this operation was
          authored/last edited in the GUI (a snapshot, not a live query);
          re-opening the phase in the editor offers to refresh it against
          the scene's current membership.
    ids        : the affected ids — see ``group_kind`` for ELEMENTS; the
        support-set id (1-tuple) for SUPPORTS; load-case ids for LOADS.
    factor     : only meaningful for LOADS/ADD — a multiplier applied to
        that case's contribution when it is superposed into this phase's
        increment (see ``phasing._superpose_cases``). Lets a phase apply,
        say, 50% of a live-load case during an intermediate construction
        stage instead of only ever "all of it or none of it". Defaults to
        1.0 (full magnitude, the previous, implicit behaviour). Purely a
        post-solve linear-combination factor — it does not change which
        loads exist in the case, only how much of the unit-case result is
        added in.
    note       : optional free text, shown in the phase's operation log.
    """
    id: str
    action: OpAction = OpAction.ADD
    target: OpTarget = OpTarget.ELEMENTS
    group_kind: str = ""
    ids: tuple = ()
    factor: float = 1.0
    note: str = ""


@dataclass
class ConstructionPhase:
    """One construction phase over a shared entity space (Mode B).

    operations : ordered list[Operation] — the authored log of what this
        phase adds/removes (elements, the support set, springs, load
        cases). This is now the source of truth; ``active_elements`` and
        ``support_set_id`` below are the *resolved* cache computed from it
        (see :func:`xdfem2d.phasing.resolve_phase_state`), kept as plain
        fields only so every existing reader (the solver, the GUI preview,
        ``_seq_reload_phases``, ...) keeps working unchanged.
    active_elements : resolved — bars/tris/objects active in this phase.
    applied_cases   : load-case ids applied *in this phase* (the increment).
        A LOADS operation only records the *intent* to add/remove a case
        from the phase's applied set; ``applied_cases`` is the resolved list.
    support_set_id  : resolved — optional SupportSet for provisional supports.
    inherits_from   : id of the previous phase (documentation / chaining).
    reapply_self_weight : if False, old elements' self-weight is assumed already
        carried in ``initial_state`` and is not reapplied (caller controls this
        by not listing the old self-weight case in ``applied_cases``).
    release_supports : if True (default), a support present in the previous
        phase but absent in this one has its accumulated reaction released as
        an equivalent load on the remaining structure (the escora/prumo
        case: the reaction redistributes onto the structure once the support
        is struck). If False, the reaction simply stops being carried (no
        redistribution) — see ``dev/SUPPORT_RELEASE_PLAN.md``.
    time            : indicative AGE of this phase (whatever time unit the
        model author chooses — days, typically), e.g. for a concrete pour
        that will later gain stiffness/strength as it cures. Purely
        informative for now — no solver reads it yet; it exists so
        sequences can already record it ahead of future time-dependent
        (creep/shrinkage/strength-gain) behaviour.
    initial_state   : {element_id: ElementInitialState} carried from before.
    """
    id: str
    operations: list = field(default_factory=list)   # list[Operation]
    active_elements: Optional[set] = None
    applied_cases: list = field(default_factory=list)
    support_set_id: Optional[str] = None
    inherits_from: Optional[str] = None
    reapply_self_weight: bool = False
    release_supports: bool = True
    time: int = 0   # indicative phase age in days (see GUI: Sequences tab)
    # Resolved cache (mirrors active_elements/support_set_id/applied_cases —
    # see this class's own docstring above): {case_id: factor}, only
    # meaningful for ids present in applied_cases. A case with no entry here
    # defaults to 1.0 (full magnitude) — see phasing._superpose_cases and
    # Operation.factor, which is the AUTHORED source of this value for a
    # phase built with Operations.
    case_factors: dict = field(default_factory=dict)
    initial_state: dict = field(default_factory=dict)


@dataclass
class ConstructionSequence:
    """An ordered chain of construction phases over one shared entity space.

    phases : in construction order; each phase inherits the accumulated state of
        all previous phases.
    displacement_method : 'increment' (Method 1 — report Δu per phase) or
        'cumulative' (Method 2 — report u0 + Δu).
    """
    id: str
    phases: list = field(default_factory=list)        # list[ConstructionPhase]
    displacement_method: str = "increment"


@dataclass
class Variant:
    """An overlay on a structure's shared entity space.

    A variant is resolved into a derived :class:`~xdfem2d.structure.Structure2D`
    by :meth:`Structure2D.derive_variant`. It never mutates the base structure.

    active_elements : set of bar-element ids that are active; ``None`` keeps the
        full geometry.
    support_set_id : id of a :class:`SupportSet` to apply instead of the base
        supports; ``None`` keeps the base supports.

    Scope (Phase 1): variants that differ only in supports and/or loads share the
    same active geometry, so their stiffness hash matches and linear sums are
    exact. Combining variants with *different* active geometry is a comparison-
    only ("what-if") operation — the mechanically correct treatment of added
    geometry is staged/incremental analysis (Mode B), not result combination.
    """
    id: str
    description: str = ""
    active_elements: Optional[set] = None
    support_set_id:  Optional[str] = None
    # Optional own action model (loads / load cases / analysis cases /
    # combinations). When None, the variant inherits the base model's actions.
    # When set, it is a full Structure2D whose geometry mirrors the base but whose
    # loads/cases/combinations are the variant's own (seeded as a copy of the base
    # when "inherit" is chosen, or clean otherwise).
    model: object = None


# ---------------------------------------------------------------------------
# Objects (geometry elements) — parametric geometry that is discretised
# into concrete nodes + bar elements only at analysis time. See
# dev/geometry_objects_plan.md. The mesh is generated by xdfem2d.geo_expand and
# is NOT part of the editable model; only the obj definitions below are.
# ---------------------------------------------------------------------------

# All geometry objects are **node-driven**: their geometry is derived from real
# nodes (``node_ids``) so moving a node reshapes the object. Objects store no
# coordinates of their own — only the node references and meshing parameters.

@dataclass
class GeoSegment:
    """Straight line between two real nodes (``node_ids = [start, end]``),
    subdivided into ``divisions`` equal segments (or by ``max_chord``) at solve
    time. Handy for a multi-storey column or a continuous beam."""
    id:           str
    section_name: str = ""
    divisions:    int = 1
    max_chord:    float = 0.0
    loads:         list = field(default_factory=list)
    intersect_crossings: bool = False
    is_column:    Optional[bool] = None   # None = follow the section's is_column
    sd_ky:        Optional[float] = None
    sd_kz:        Optional[float] = None
    sd_klt:       Optional[float] = None
    sd_ltb:       bool = True
    # Beam-bars design: tag of the continuous beam these bars belong to (every
    # bar the object generates inherits it); see BarElement.beam.
    beam:         Optional[str] = None
    node_ids:      list = field(default_factory=list)   # [start, end]


@dataclass
class GeoArc:
    """Circular arc through three real nodes (``node_ids = [start, mid, end]``).
    The circle / centre / sweep are derived from the node positions, so moving
    any node reshapes the arc. The three points must not be collinear."""
    id:           str
    section_name: str = ""
    divisions:    int = 8               # segments for the actual sweep
    max_chord:    float = 0.0
    loads:         list = field(default_factory=list)
    intersect_crossings: bool = False
    is_column:    Optional[bool] = None   # None = follow the section's is_column
    sd_ky:        Optional[float] = None
    sd_kz:        Optional[float] = None
    sd_klt:       Optional[float] = None
    sd_ltb:       bool = True
    beam:         Optional[str] = None    # beam tag (see GeoSegment.beam)
    node_ids:      list = field(default_factory=list)   # [start, mid, end]


@dataclass
class GeoMultisegment:
    """Open polyline / closed polygon through real nodes (``node_ids`` = the
    ordered vertices). Each span is subdivided by ``divisions`` or ``max_chord``.

    ``span_divisions`` optionally overrides ``divisions`` per span instead of
    sharing one count across the whole curve: entry *i* is the division count
    for the span from ``node_ids[i]`` to ``node_ids[i + 1]`` (plus, when
    ``closed``, one more entry for the closing span back to ``node_ids[0]``).
    ``None`` (the default) or a list whose length does not match the current
    number of spans falls back to the uniform ``divisions`` for every span —
    the same fallback a file saved before this field existed gets, and the
    same one a vertex added/removed after ``span_divisions`` was set gets too,
    rather than silently misapplying a now-mismatched list. ``max_chord``,
    when set, still overrides both on a per-span basis, exactly as before."""
    id:           str
    closed:       bool = False
    section_name: str = ""
    divisions:    int = 1
    max_chord:    float = 0.0
    loads:         list = field(default_factory=list)
    intersect_crossings: bool = False
    is_column:    Optional[bool] = None   # None = follow the section's is_column
    sd_ky:        Optional[float] = None
    sd_kz:        Optional[float] = None
    sd_klt:       Optional[float] = None
    sd_ltb:       bool = True
    beam:         Optional[str] = None    # beam tag (see GeoSegment.beam)
    node_ids:      list = field(default_factory=list)   # [v0, v1, ...]
    span_divisions: list | None = None   # [divisions_0_1, divisions_1_2, ...]


# ── Surface objects — discretised into triangles (and, since dev/
# IMPLEMENT_QUAD.md Phase 6, quads) rather than bars ─────────────────────────
# ``tri_section_name`` names a TriSection OR a QuadSection (the field keeps
# its original name — see below — rather than gaining a second one) and a
# target element size; at solve time xdfem2d.geo_expand meshes the object
# into nodes + TriElements/QuadElements.
#
# Why the field is not renamed/duplicated (Phase 6 decision, mirroring the
# ``tri_stress`` results-key decision in Phase 5): ``tri_section_name`` is a
# persisted ``.x2d`` field with ~44 references across the codebase, and this
# codebase's established save-format rule is additive/absent-key-compatible,
# never rename-with-migration. Widening what an existing field may resolve to
# — rather than adding a parallel ``quad_section_name`` a surface would have
# to pick between — keeps every existing model file valid unchanged, and
# keeps "what section does this surface use" a single question with a single
# answer, resolved by :func:`xdfem2d.geo_expand._expand_surface` by lookup
# (TriSection first, then QuadSection) rather than by which field is set.

# Edge-restraint propagation modes for a meshed surface (see
# xdfem2d.geo_expand._propagate_edge_supports / _propagate_edge_springs):
#   supports: "none"   — no restraint on the generated edge nodes
#             "common" — the DOFs restrained at BOTH corners (their intersection),
#                        named by CANONICAL_SUPPORTS (default)
#             "equal"  — only when both corners restrain the same DOF set
#   springs:  "none"   — no spring on the generated edge nodes
#             "linear" — kx/ky/kt interpolated linearly between the two corners
EDGE_SUPPORT_MODES = ("none", "common", "equal")
EDGE_SPRING_MODES = ("none", "linear")


@dataclass
class GeoRectangle:
    """Axis-aligned rectangle defined by two opposite (diagonal) corner nodes
    (``node_ids = [corner_a, corner_c]``). Meshed at the given target element
    size, into quads where ``tri_section_name`` names a QuadSection and the
    per-cell shape is good enough (else triangles for that cell), or into
    triangles throughout when it names a TriSection — see
    ``xdfem2d.geo_expand._expand_surface`` (dev/IMPLEMENT_QUAD.md Phase 6).

    ``prefer_quad`` (Phase 8, GUI item 2b revision): the GUI's "Panel
    sections" dialog now always creates a same-named TriSection AND
    QuadSection pair together, so ``tri_section_name`` alone can no longer
    signal which kind to mesh into (a TriSection under that name always
    exists too). ``prefer_quad=True`` makes ``_expand_surface`` check
    ``quad_sections`` first instead of ``tri_sections`` first. Defaults to
    False so files saved before this field existed keep meshing into
    triangles exactly as before — the pre-Phase-8-revision behaviour is
    still reachable by leaving a QuadSection unpaired (no same-named
    TriSection): ``_expand_surface`` still falls back to whichever section
    kind actually resolves, same as it always has."""
    id:              str
    tri_section_name: str = ""   # a TriSection OR a QuadSection name — see above
    target_size:     float = 0.5        # target element edge length [m]
    prefer_quad:     bool = False       # see docstring above
    node_ids:        list = field(default_factory=list)   # [corner_a, corner_c]
    edge_support_mode: str = "common"   # see EDGE_SUPPORT_MODES
    edge_spring_mode:  str = "linear"   # see EDGE_SPRING_MODES
    # Explicit per-edge restraint, one entry per edge in perimeter order
    # (node_ids[0]-node_ids[1], [1]-[2], [2]-[3], [3]-[0]): a support name,
    # or None/"free" for no restraint on that edge. Overrides
    # edge_support_mode for THIS object when set (None keeps the old
    # corner-intersection behaviour unchanged).
    #
    # Why this exists: edge_support_mode="common" derives each edge's
    # restraint from the intersection of its two corners' own restraints —
    # which cannot express an alternating pattern (e.g. clamped / simply /
    # clamped / simply around the four sides), because a corner shared by a
    # clamped edge and a simply-supported edge must be promoted to
    # "clamped" for the clamped edge to work, and that promoted corner then
    # also clamps its simply-supported neighbour through the same
    # intersection — confirmed by direct test (2026-08), every edge came
    # out clamped. edge_supports assigns each edge's restraint directly
    # instead of deriving it, so mixed/alternating per-edge conditions on a
    # rectangle object are possible at all.
    edge_supports:   list | None = None


@dataclass
class GeoPolygon:
    """Closed polygon surface through real vertex nodes (``node_ids`` = ordered
    vertices). Meshed at the given target element size: quads are only
    possible when the outline is itself a convex quadrilateral (the
    structured branch — see ``xdfem2d.meshing.mesh_quad_structured_cells``,
    dev/IMPLEMENT_QUAD.md Phase 6) and ``tri_section_name`` names a
    QuadSection; every other outline, or a TriSection name, meshes into
    triangles (Delaunay, target element size), same as before Phase 6.

    ``prefer_quad`` — see GeoRectangle.prefer_quad's docstring (same field,
    same reasoning, dev/IMPLEMENT_QUAD.md Phase 8 GUI item 2b revision)."""
    id:              str
    tri_section_name: str = ""   # a TriSection OR a QuadSection name — see above
    target_size:     float = 0.5
    prefer_quad:     bool = False       # see GeoRectangle.prefer_quad
    node_ids:        list = field(default_factory=list)   # [v0, v1, ...]
    edge_support_mode: str = "common"   # see EDGE_SUPPORT_MODES
    edge_spring_mode:  str = "linear"   # see EDGE_SPRING_MODES
    # Explicit per-edge restraint, one entry per edge in perimeter order
    # (node_ids[0]-node_ids[1], [1]-[2], ..., [-1]-[0]): a support name, or
    # None/"free" for no restraint on that edge. Overrides edge_support_mode
    # for THIS object when set (None keeps the old corner-intersection
    # behaviour unchanged). Same rationale and mechanics as
    # GeoRectangle.edge_supports — see its docstring — generalised to N
    # vertices/edges instead of a fixed 4.
    edge_supports:   list | None = None


# ── Scalar fields  value = f(x, y)  ─────────────────────────────────────────

@dataclass
class Field:
    """A named scalar field defined by a Python expression of ``x`` and ``y``
    (e.g. ``"x*y"``, ``"sin(x) + 0.5*y"``). Evaluated at node coordinates for
    visualisation on bars or triangles."""
    name:       str
    expression: str = "0.0"


# Math namespace exposed to field expressions (no builtins → safe-ish eval).
import math as _math
_FIELD_NS = {k: getattr(_math, k) for k in (
    "sin", "cos", "tan", "asin", "acos", "atan", "atan2", "exp", "log",
    "log10", "sqrt", "hypot", "sinh", "cosh", "tanh", "floor", "ceil",
    "degrees", "radians", "pi", "e")}
_FIELD_NS.update({"abs": abs, "min": min, "max": max, "pow": pow})


def evaluate_field(expression: str, x: float, y: float) -> float:
    """Evaluate a field expression at (x, y). Returns 0.0 on error."""
    try:
        return float(eval(expression, {"__builtins__": {}},
                          {**_FIELD_NS, "x": float(x), "y": float(y)}))
    except Exception:
        return 0.0
