"""Beam-bars detailing core — the data contract and the editing logic (no Qt).

The beam-bars design (:func:`xdfem2d.rc_design.design_beam_bars`) *proposes*
bars per zone. A designer then refines that proposal. This module holds
everything needed for that, independent of the GUI and of the model, so a
detailing window — in this process or, later, a separate application — only
has to draw it and call these functions:

``SegmentInputs``  what is *derived* (never edited): the spans, the required
                   steel along the beam (stations), the section, the effective
                   parameters. Built by the design (``design_beam_bars(...,
                   inputs_out=)``) or read from a package file.
``BeamDetail``     what the *user decides*: per segment → span → zone, the
                   layers ``(n, Ø)`` of the bottom and top face, and — in its
                   own zoning — the stirrups ``legs / Ø / spacing``; each
                   zone marked ``auto`` (from the proposal) or ``manual``.
``DetailDocument`` the ``beam_detail.json`` payload: ``{tag: BeamDetail}``.
``DetailEditor``   edit operations with undo / redo.
``check_segment``  live verification of a detail against its inputs.
``reconcile_*``    merge a new proposal into the existing detail: manual zones
                   are kept (and flagged if they no longer comply), auto zones
                   follow the proposal, nothing is erased silently.
``BeamPackage``    inputs + detail in one self-contained JSON (the contract a
                   standalone window can open).

Units: lengths [m], areas [mm²], bar diameters [mm]. A zone's ``x0``/``x1`` are
measured from the start of its span; ``stations`` are measured along the
segment (the segment is the continuous run of spans of one beam). See
dev/BEAM_BARS_DEFINITION.md and the contract discussion in the session notes.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, field, replace

from .beam_bars_params import (
    PREF_KEYS, BeamBarParams, SECTION_PARAMS, BEAM_PARAMS, resolve_beam_params,
)
from .beam_rebar import (
    Beam, BarGroup, Layout, Span, _counts_ok, bar_area, beam_reinforcement,
    layouts_compatible, max_bars_per_layer,
)

SCHEMA_VERSION = 1
ORIGINS = ("auto", "manual")
FACES = ("bottom", "top")


class DetailError(ValueError):
    """A malformed detail / package, or an invalid edit."""


def _tol(length: float) -> float:
    return 1e-6 * max(length, 1.0)


def _natural(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(name))]


def _layers(raw, what="layers") -> list:
    """``[[n, d], …]`` → ``[(n, d), …]`` with n int >= 1 and d > 0."""
    out = []
    for item in raw or []:
        try:
            n, d = item
            n, d = int(n), float(d)
        except (TypeError, ValueError):
            raise DetailError(f"{what}: bad layer {item!r}") from None
        if n < 1 or d <= 0:
            raise DetailError(f"{what}: layer {item!r} needs n >= 1 and Ø > 0")
        out.append((n, d))
    return out


def layers_area(layers) -> float:
    """Steel area [mm²] of a list of ``(n, Ø)`` layers."""
    return sum(n * bar_area(d) for n, d in layers)


def layers_text(layers) -> str:
    return " + ".join(f"{n}Ø{d:g}" for n, d in layers) or "-"


# ======================================================================
# The user's decisions: BeamDetail
# ======================================================================

@dataclass
class ZoneDetail:
    x0: float                       # [m] from the start of the span
    x1: float
    origin: str = "auto"            # "auto" | "manual"
    bottom: list = field(default_factory=list)   # [(n, Ø), …]
    top: list = field(default_factory=list)

    def layers(self, face: str) -> list:
        if face not in FACES:
            raise DetailError(f"face must be one of {FACES}")
        return self.bottom if face == "bottom" else self.top

    def area(self, face: str) -> float:
        return layers_area(self.layers(face))

    def to_dict(self) -> dict:
        return {"x0": self.x0, "x1": self.x1, "origin": self.origin,
                "bottom": [list(l) for l in self.bottom],
                "top": [list(l) for l in self.top]}

    @classmethod
    def from_dict(cls, d: dict) -> "ZoneDetail":
        try:
            z = cls(float(d["x0"]), float(d["x1"]),
                    str(d.get("origin", "auto")),
                    _layers(d.get("bottom"), "bottom"),
                    _layers(d.get("top"), "top"))
        except (KeyError, TypeError, ValueError) as e:
            if isinstance(e, DetailError):
                raise
            raise DetailError(f"bad zone {d!r}: {e}") from None
        if z.origin not in ORIGINS:
            raise DetailError(f"zone origin must be one of {ORIGINS}")
        if z.x1 <= z.x0:
            raise DetailError(f"zone [{z.x0}, {z.x1}] has no length")
        return z


@dataclass
class StirrupZone:
    """Stirrups of one stretch of a span (its own zoning, independent of the
    bar zones): ``legs`` legs of diameter ``diameter`` [mm] every ``spacing``
    [mm]."""
    x0: float
    x1: float
    origin: str = "auto"
    legs: int = 2
    diameter: float = 8.0
    spacing: float = 150.0

    def asw_s(self) -> float:
        """Shear steel per unit length Asw/s [mm²/m]."""
        return self.legs * bar_area(self.diameter) / (self.spacing / 1000.0)

    def to_dict(self) -> dict:
        return {"x0": self.x0, "x1": self.x1, "origin": self.origin,
                "legs": self.legs, "diameter": self.diameter,
                "spacing": self.spacing}

    @classmethod
    def from_dict(cls, d: dict) -> "StirrupZone":
        try:
            z = cls(float(d["x0"]), float(d["x1"]),
                    str(d.get("origin", "auto")), int(d["legs"]),
                    float(d["diameter"]), float(d["spacing"]))
        except (KeyError, TypeError, ValueError) as e:
            raise DetailError(f"bad stirrup zone {d!r}: {e}") from None
        if z.origin not in ORIGINS:
            raise DetailError(f"zone origin must be one of {ORIGINS}")
        if z.x1 <= z.x0:
            raise DetailError(f"stirrup zone [{z.x0}, {z.x1}] has no length")
        if z.legs < 1 or z.diameter <= 0 or z.spacing <= 0:
            raise DetailError(f"stirrup zone needs legs >= 1, Ø > 0 and "
                              f"spacing > 0 ({d!r})")
        return z


@dataclass
class SpanDetail:
    id: str
    length: float
    zones: list = field(default_factory=list)
    stirrups: list = field(default_factory=list)    # [StirrupZone]; [] = none

    def to_dict(self) -> dict:
        d = {"id": self.id, "length": self.length,
             "zones": [z.to_dict() for z in self.zones]}
        if self.stirrups:                  # absent = a detail without stirrups
            d["stirrups"] = [z.to_dict() for z in self.stirrups]
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SpanDetail":
        try:
            sp = cls(str(d["id"]), float(d["length"]),
                     [ZoneDetail.from_dict(z) for z in d.get("zones", [])],
                     [StirrupZone.from_dict(z) for z in d.get("stirrups", [])])
        except (KeyError, TypeError, ValueError) as e:
            if isinstance(e, DetailError):
                raise
            raise DetailError(f"bad span {d!r}: {e}") from None
        return sp


@dataclass
class SegmentDetail:
    name: str
    spans: list = field(default_factory=list)

    def span(self, span_id: str) -> SpanDetail:
        for sp in self.spans:
            if sp.id == span_id:
                return sp
        raise DetailError(f"span {span_id!r} not found")

    def to_dict(self) -> dict:
        return {"name": self.name, "spans": [s.to_dict() for s in self.spans]}

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentDetail":
        try:
            return cls(str(d["name"]),
                       [SpanDetail.from_dict(s) for s in d.get("spans", [])])
        except (KeyError, TypeError) as e:
            raise DetailError(f"bad segment {d!r}: {e}") from None


@dataclass
class BeamDetail:
    tag: str
    segments: list = field(default_factory=list)
    based_on: str = ""              # signature of the inputs it was built from

    def segment(self, name: str) -> SegmentDetail:
        for s in self.segments:
            if s.name == name:
                return s
        raise DetailError(f"segment {name!r} not found")

    def to_dict(self) -> dict:
        return {"based_on": self.based_on,
                "segments": [s.to_dict() for s in self.segments]}

    @classmethod
    def from_dict(cls, tag: str, d: dict) -> "BeamDetail":
        try:
            return cls(str(tag),
                       [SegmentDetail.from_dict(s)
                        for s in d.get("segments", [])],
                       str(d.get("based_on", "")))
        except (AttributeError, TypeError) as e:
            raise DetailError(f"bad beam {tag!r}: {e}") from None


@dataclass
class DetailDocument:
    """The ``beam_detail.json`` payload: the user's decisions per beam tag."""
    beams: dict = field(default_factory=dict)       # tag -> BeamDetail
    version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {"version": self.version,
                "beams": {t: b.to_dict() for t, b in self.beams.items()}}

    @classmethod
    def from_dict(cls, d: dict | None) -> "DetailDocument":
        """Tolerant of an absent document (old file); strict about what is
        present: a newer version, or a structure that does not hold together,
        raises :class:`DetailError` rather than being half-loaded."""
        if not d:
            return cls()
        if not isinstance(d, dict):
            raise DetailError("detail document must be an object")
        ver = d.get("version", SCHEMA_VERSION)
        if not isinstance(ver, int) or ver > SCHEMA_VERSION:
            raise DetailError(
                f"detail document version {ver!r} is newer than this "
                f"program understands ({SCHEMA_VERSION})")
        beams = {str(t): BeamDetail.from_dict(t, b)
                 for t, b in (d.get("beams") or {}).items()}
        doc = cls(beams, ver)
        for b in doc.beams.values():
            for seg in b.segments:
                issues = validate_structure(seg)
                if issues:
                    raise DetailError(
                        f"beam {b.tag!r} segment {seg.name!r}: "
                        f"{issues[0].message}")
        return doc

    def has_manual(self, key: str) -> bool:
        b = self.beams.get(key)
        return bool(b) and any(
            z.origin == "manual" for seg in b.segments for sp in seg.spans
            for z in list(sp.zones) + list(sp.stirrups))

    def prune_auto(self) -> list:
        """Drop the beams that hold no manual zone (they are just copies of an
        automatic proposal, regenerated by every design run). Returns the keys
        removed."""
        gone = [k for k in self.beams if not self.has_manual(k)]
        for k in gone:
            del self.beams[k]
        return gone

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def loads(cls, text: str) -> "DetailDocument":
        try:
            return cls.from_dict(json.loads(text))
        except json.JSONDecodeError as e:
            raise DetailError(f"not valid JSON: {e}") from None


# ======================================================================
# What is derived: SegmentInputs
# ======================================================================

@dataclass
class SpanInputs:
    id: str
    length: float


@dataclass
class SegmentInputs:
    """Everything the checks need about one continuous segment of a beam."""
    name: str                       # segment name, e.g. "V1" or "V1.2"
    tag: str                        # beam key (tag, or the automatic name)
    spans: list                     # [SpanInputs]
    stations: list                  # [(X [m], As_bot [mm²], As_top [mm²])]
    section: dict                   # b h cover stirrup d_bar_est fck fyk dg
    params: dict                    # BeamBarParams as a dict
    a_l: float = 0.0                # envelope shift [m] (EC2 9.2.1.3)
    shear: list = field(default_factory=list)   # [(X [m], Asw/s [mm²/m])]
    section_name: str = ""          # for "all beams of this section"

    # -- derived -------------------------------------------------------
    @property
    def params_obj(self) -> BeamBarParams:
        return BeamBarParams(**{
            k: (tuple(v) if isinstance(v, list) else v)
            for k, v in self.params.items()})

    def beam(self) -> Beam:
        s = self.section
        return Beam(width=s["b"], height=s["h"], cover=s["cover"],
                    stirrup_diameter=s["stirrup"], fck=s["fck"], fyk=s["fyk"],
                    spans=[], dg=s.get("dg", 20.0),
                    d_bar_est=s.get("d_bar_est", 16.0))

    @property
    def as_min(self) -> float:
        return self.beam().as_min

    @property
    def as_max(self) -> float:
        return self.beam().as_max

    def offsets(self) -> dict:
        out, x = {}, 0.0
        for sp in self.spans:
            out[sp.id] = x
            x += sp.length
        return out

    def length(self) -> float:
        return sum(sp.length for sp in self.spans)

    def signature(self) -> str:
        """Hash of what the proposal depends on, to tell whether a detail was
        built from the current calculation."""
        payload = {
            "spans": [[sp.id, round(sp.length, 4)] for sp in self.spans],
            "stations": [[round(x, 4), round(b, 1), round(t, 1)]
                         for x, b, t in self.stations],
            "section": {k: (round(v, 4) if isinstance(v, float) else v)
                        for k, v in sorted(self.section.items())},
            "params": {k: (list(v) if isinstance(v, tuple) else v)
                       for k, v in sorted(self.params.items())},
            "a_l": round(self.a_l, 4),
            "shear": [[round(x, 4), round(v, 1)] for x, v in self.shear]}
        return hashlib.sha1(
            json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]

    # -- (de)serialisation --------------------------------------------
    def to_dict(self) -> dict:
        return {"name": self.name, "tag": self.tag,
                "spans": [{"id": s.id, "length": s.length} for s in self.spans],
                "stations": [list(t) for t in self.stations],
                "section": dict(self.section),
                "params": {k: (list(v) if isinstance(v, tuple) else v)
                           for k, v in self.params.items()},
                "a_l": self.a_l,
                "shear": [list(t) for t in self.shear],
                "section_name": self.section_name}

    @classmethod
    def from_dict(cls, d: dict) -> "SegmentInputs":
        try:
            inp = cls(
                str(d["name"]), str(d["tag"]),
                [SpanInputs(str(s["id"]), float(s["length"]))
                 for s in d["spans"]],
                [(float(x), float(b), float(t)) for x, b, t in d["stations"]],
                dict(d["section"]), dict(d["params"]), float(d.get("a_l", 0.0)),
                [(float(x), float(v)) for x, v in d.get("shear", [])],
                str(d.get("section_name", "")))
            for key in ("b", "h", "cover", "stirrup", "fck", "fyk"):
                if key not in inp.section:
                    raise KeyError(f"section.{key}")
            inp.params_obj.validate()
        except (KeyError, TypeError, ValueError) as e:
            raise DetailError(f"bad inputs {d.get('name', '?')!r}: {e}") from None
        return inp


def segment_inputs_from_design(name, tag, spans, stations, section, params,
                               a_l, shear=(), section_name="") -> SegmentInputs:
    """Build :class:`SegmentInputs` from the pieces ``design_beam_bars`` has
    at hand (areas in mm², lengths in m)."""
    pd = asdict(params) if not isinstance(params, dict) else dict(params)
    return SegmentInputs(
        name, tag, [SpanInputs(i, float(L)) for i, L in spans],
        [(float(x), float(b), float(t)) for x, b, t in stations],
        dict(section),
        {k: (list(v) if isinstance(v, tuple) else v) for k, v in pd.items()},
        float(a_l), [(float(x), float(v)) for x, v in shear],
        str(section_name))


# ======================================================================
# Zoning helpers (shared with the design)
# ======================================================================

def cutoff_points(xs, vals, n_levels, floor):
    """Stations where the steel envelope *vals* changes level.

    The envelope is quantised into *n_levels* equal steps of its peak (never
    below *floor*); between two consecutive stations of different level the
    cutoff is placed at the station of the **lower** level, so the higher
    level keeps covering the whole interval (conservative).
    """
    peak = max(vals, default=0.0)
    if peak <= 0.0 or n_levels < 1:
        return []
    step = peak / n_levels
    lev = [math.ceil(max(v, floor) / step - 1e-9) for v in vals]
    return [xs[i] if lev[i] < lev[i + 1] else xs[i + 1]
            for i in range(len(xs) - 1) if lev[i] != lev[i + 1]]


def merge_short_zones(edges, min_len):
    """Drop interior breakpoints until every zone is >= *min_len* long."""
    edges = list(edges)
    while len(edges) > 2:
        lens = [edges[i + 1] - edges[i] for i in range(len(edges) - 1)]
        k = min(range(len(lens)), key=lens.__getitem__)
        if lens[k] >= min_len - 1e-9:
            break
        # remove the interior edge bounding the shortest zone (the one that
        # leaves the shorter neighbour, i.e. merge with the shorter side)
        if k == 0:
            del edges[1]
        elif k == len(lens) - 1:
            del edges[-2]
        else:
            del edges[k if lens[k - 1] <= lens[k + 1] else k + 1]
    return edges


# ======================================================================
# Required steel per zone
# ======================================================================

def zone_requirements(stations, intervals, a_l: float = 0.0,
                      support_bottom_ratio: float = 0.0) -> list:
    """Required steel of each zone: ``[(bottom, top), …]`` in the units of
    *stations* (``[(X, bottom, top)]``).

    *intervals* are the zone limits ``[(lo, hi), …]`` on the same axis as the
    stations, in order along ONE span. The requirement of a zone is the
    maximum over the stations in ``[lo - a_l, hi + a_l]`` — the envelope
    shifted by *a_l* (EC2 9.2.1.3). With several zones, the first and last get
    at least *support_bottom_ratio* of the largest bottom requirement of the
    span (EC2 9.2.1.4(1)).
    """
    reqs = []
    for lo, hi in intervals:
        tol = _tol(hi - lo)
        inside = [(b, t) for x, b, t in stations
                  if lo - a_l - tol <= x <= hi + a_l + tol]
        reqs.append([max((b for b, _ in inside), default=0.0),
                     max((t for _, t in inside), default=0.0)])
    if len(reqs) > 1 and support_bottom_ratio > 0:
        mx = max(r[0] for r in reqs) * support_bottom_ratio
        for z in (0, len(reqs) - 1):
            reqs[z][0] = max(reqs[z][0], mx)
    return [tuple(r) for r in reqs]


def required_areas(inputs: SegmentInputs, span_id: str, zones) -> list:
    """Steel required in each of *zones* of span *span_id* [mm²], with the
    floors the design applies: bottom >= As,min, top >= top_min_ratio·As,min.
    Returns ``[(bottom, top), …]``."""
    p = inputs.params_obj
    off = inputs.offsets()[span_id]
    raw = zone_requirements(
        inputs.stations, [(off + z.x0, off + z.x1) for z in zones],
        inputs.a_l, p.support_bottom_ratio)
    am = inputs.as_min
    return [(max(b, am), max(t, p.top_min_ratio * am)) for b, t in raw]


def envelope_profile(inputs: SegmentInputs, face: str) -> list:
    """The required-steel envelope along the segment for the drawing:
    ``[(X [m], area [mm²]), …]`` at every station, shifted by ``a_l``
    (EC2 9.2.1.3) — the demand a zone has to cover at that point."""
    if face not in FACES:
        raise DetailError(f"face must be one of {FACES}")
    col = 1 if face == "bottom" else 2
    st = sorted(inputs.stations)
    out = []
    for x, *_ in st:
        w = [t[col] for t in st if abs(t[0] - x) <= inputs.a_l + 1e-9]
        out.append((x, max(w)))
    return out


def zone_profile(segment: SegmentDetail, inputs: SegmentInputs) -> list:
    """Per zone, required and provided steel for the chart: a list of dicts
    ``{span, k, x0, x1 (m along the segment), req_bottom, req_top,
    prov_bottom, prov_top, origin}`` in mm² (required = the same value
    :func:`check_segment` uses, floors included). Spans the inputs do not know
    (a detail from another calculation) are skipped."""
    off = inputs.offsets()
    out = []
    for sp in segment.spans:
        if sp.id not in off:
            continue
        reqs = required_areas(inputs, sp.id, sp.zones)
        for k, (z, (rb, rt)) in enumerate(zip(sp.zones, reqs)):
            out.append({"span": sp.id, "k": k, "x0": off[sp.id] + z.x0,
                        "x1": off[sp.id] + z.x1, "req_bottom": rb,
                        "req_top": rt, "prov_bottom": z.area("bottom"),
                        "prov_top": z.area("top"), "origin": z.origin})
    return out


# ======================================================================
# Stirrups: required steel, proposal
# ======================================================================

def min_shear_floor(inputs: SegmentInputs) -> float:
    """Minimum shear reinforcement Asw/s [mm²/m]: ρw,min·b·sinα with
    ρw,min = 0.08·√fck/fyk (EC2 9.2.2(5)); 0 when the section disables it."""
    sec = inputs.section
    if not sec.get("shear_min", True):
        return 0.0
    alpha = math.radians(sec.get("alpha_s", 90.0))
    rho = 0.08 * math.sqrt(sec["fck"]) / sec["fyk"]
    return rho * sec["b"] * math.sin(alpha) * 1000.0


def stirrup_s_max(inputs: SegmentInputs) -> float:
    """Maximum longitudinal spacing of stirrups [mm]: 0.75·d·(1 + cotα)
    (EC2 9.2.2(6))."""
    alpha = math.radians(inputs.section.get("alpha_s", 90.0))
    cot = 1.0 / math.tan(alpha) if abs(math.tan(alpha)) > 1e-9 else 0.0
    return 0.75 * inputs.beam().d_eff * (1.0 + cot)


def stirrup_requirements(inputs: SegmentInputs, span_id: str, zones) -> list:
    """Asw/s required in each of the stirrup *zones* of span *span_id*
    [mm²/m]: the maximum of the shear stations inside it (no shift for shear),
    never below the minimum."""
    off = inputs.offsets()[span_id]
    floor = min_shear_floor(inputs)
    out = []
    for z in zones:
        lo, hi = off + z.x0, off + z.x1
        tol = _tol(hi - lo)
        v = max((a for x, a in inputs.shear if lo - tol <= x <= hi + tol),
                default=0.0)
        out.append(max(v, floor))
    return out


def propose_stirrups(inputs: SegmentInputs) -> dict:
    """Automatic stirrups ``{span id: [StirrupZone]}`` from the shear stations
    (empty when the inputs carry none).

    The zones follow the cutoff points of the Asw/s envelope (its steps are
    ``params.cutoff_levels`` of the peak, shorter zones than
    ``min_zone_length`` merged). One diameter is used along the segment — the
    one that spends the least steel; diameters up to the one assumed for the
    cover are tried first — with ``legs`` legs, and in every zone the largest
    allowed spacing (at most :func:`stirrup_s_max`) that covers the
    requirement. If nothing covers it the closest effort is returned and the
    checks flag it.
    """
    if not inputs.shear:
        return {}
    p = inputs.params_obj
    floor = min_shear_floor(inputs)
    smax = stirrup_s_max(inputs)
    st = sorted(inputs.shear)
    xs, vals = [x for x, _ in st], [v for _, v in st]
    cut = sorted(set(cutoff_points(xs, vals, p.cutoff_levels, floor)))
    off = inputs.offsets()

    zones = []                                   # (span id, x0, x1, req, len)
    for sp in inputs.spans:
        s0, s1 = off[sp.id], off[sp.id] + sp.length
        tol = _tol(sp.length)
        inner = [c for c in cut if s0 + tol < c < s1 - tol]
        edges = merge_short_zones([s0] + inner + [s1], p.min_zone_length)
        for lo, hi in zip(edges, edges[1:]):
            sec_tol = _tol(hi - lo)
            req = max((v for x, v in st if lo - sec_tol <= x <= hi + sec_tol),
                      default=0.0)
            zones.append((sp.id, lo - s0, hi - s0, max(req, floor), hi - lo))

    spacings = sorted((float(s) for s in p.stirrup_spacings), reverse=True)
    allowed = [s for s in spacings if s <= smax + 1e-9] or [min(spacings)]
    diams = sorted(float(d) for d in p.stirrup_diameters)
    assumed = inputs.section.get("stirrup", 8.0)
    order = [[d for d in diams if d <= assumed + 1e-9], diams]

    def pick(d, req):
        """Largest allowed spacing covering *req*, or None."""
        a = p.stirrup_legs * bar_area(d)
        for s in allowed:                        # descending: fewest stirrups
            if a / (s / 1000.0) + 1e-9 >= req:
                return s
        return None

    best = None
    for group in order:
        for d in group:
            sp_pick = [pick(d, z[3]) for z in zones]
            if any(s is None for s in sp_pick):
                continue
            cost = sum(z[4] * p.stirrup_legs * bar_area(d) / (s / 1000.0)
                       for z, s in zip(zones, sp_pick))
            if best is None or cost < best[0] - 1e-9:
                best = (cost, d, sp_pick)
        if best:
            break
    if best is None:                              # nothing covers it: closest
        d = diams[-1]
        best = (0.0, d, [pick(d, z[3]) or min(allowed) for z in zones])
    _cost, d, picks = best
    out: dict = {}
    for (sid, x0, x1, _r, _l), s in zip(zones, picks):
        out.setdefault(sid, []).append(
            StirrupZone(x0, x1, "auto", p.stirrup_legs, d, s))
    return out


def stirrup_profile(segment: SegmentDetail, inputs: SegmentInputs) -> list:
    """Per stirrup zone, for the chart: ``{span, k, x0, x1 (m along the
    segment), required, provided (Asw/s [mm²/m]), legs, diameter, spacing,
    s_max, origin}``. Spans without stirrups give nothing."""
    off = inputs.offsets()
    smax = stirrup_s_max(inputs)
    out = []
    for sp in segment.spans:
        if sp.id not in off or not sp.stirrups:
            continue
        reqs = stirrup_requirements(inputs, sp.id, sp.stirrups)
        for k, (z, r) in enumerate(zip(sp.stirrups, reqs)):
            out.append({"span": sp.id, "k": k, "x0": off[sp.id] + z.x0,
                        "x1": off[sp.id] + z.x1, "required": r,
                        "provided": z.asw_s(), "legs": z.legs,
                        "diameter": z.diameter, "spacing": z.spacing,
                        "s_max": smax, "origin": z.origin})
    return out


# ======================================================================
# Checks
# ======================================================================

@dataclass(frozen=True)
class Issue:
    code: str
    severity: str                   # "error" | "warning"
    message: str
    span: str | None = None
    zone: int | None = None
    face: str | None = None
    layer: str = "bars"             # "bars" | "stirrups" (own zone numbering)


@dataclass
class CheckReport:
    issues: list = field(default_factory=list)

    @property
    def errors(self) -> list:
        return [i for i in self.issues if i.severity == "error"]

    @property
    def warnings(self) -> list:
        return [i for i in self.issues if i.severity == "warning"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self) -> set:
        return {i.code for i in self.issues}

    def for_zone(self, span: str, zone: int, layer: str = "bars") -> list:
        return [i for i in self.issues
                if i.span == span and i.zone == zone and i.layer == layer]


def validate_structure(seg: SegmentDetail) -> list:
    """Do the zones of every span tile ``[0, length]`` without gaps or
    overlaps? (Structure only — no design rule.)"""
    out = []
    for sp in seg.spans:
        tol = _tol(sp.length)
        if sp.length <= 0:
            out.append(Issue("ZONES", "error",
                             f"span {sp.id} has no length", sp.id))
            continue
        if not sp.zones:
            out.append(Issue("ZONES", "error",
                             f"span {sp.id} has no zones", sp.id))
            continue
        for zones, layer, label in ((sp.zones, "bars", "zone"),
                                    (sp.stirrups, "stirrups", "stirrup zone")):
            if not zones:
                continue
            pos = 0.0
            for k, z in enumerate(zones):
                if abs(z.x0 - pos) > tol:
                    what = "gap" if z.x0 > pos else "overlap"
                    out.append(Issue(
                        "ZONES", "error",
                        f"span {sp.id}: {what} before {label} {k + 1} "
                        f"(at {pos:.3f} m)", sp.id, k, layer=layer))
                pos = z.x1
            if abs(pos - sp.length) > tol:
                out.append(Issue(
                    "ZONES", "error",
                    f"span {sp.id}: {label}s end at {pos:.3f} m, span is "
                    f"{sp.length:.3f} m", sp.id, layer=layer))
    return out


def _check_layers(layers, face, beam: Beam, p: BeamBarParams,
                  span, zone) -> list:
    out = []

    def add(code, msg, sev="error"):
        out.append(Issue(code, sev, f"{face}: {msg}", span, zone, face))

    if not layers:
        add("NO_BARS", "no bars")
        return out
    if len(layers) > p.max_layers:
        add("MAX_LAYERS", f"{len(layers)} layers, max {p.max_layers}")
    n_tot = sum(n for n, _ in layers)
    if n_tot < p.min_bars:
        add("MIN_BARS", f"{n_tot} bars, min {p.min_bars}")
    for k, (n, d) in enumerate(layers):
        tag = f"layer {k + 1} ({n}Ø{d:g})"
        if not _counts_ok(n, p.symmetry):
            add("LAYER_COUNT", f"{tag}: {n} bars breaks the "
                f"'{p.symmetry}' symmetry rule")
        if n > max_bars_per_layer(beam, d):
            add("SPACING", f"{tag}: does not fit the width "
                f"(max {max_bars_per_layer(beam, d)} bars of Ø{d:g})")
        if k and n > layers[k - 1][0]:
            add("LAYER_ORDER", f"{tag}: more bars than the layer before")
        if not any(abs(d - x) < 1e-9 for x in p.diameters):
            add("DIAMETER_UNLISTED", f"{tag}: Ø{d:g} is not in the available "
                "diameters", "warning")
    return out


def _check_stirrups(sp: SpanDetail, inputs: SegmentInputs,
                    p: BeamBarParams) -> list:
    out = []
    if not sp.stirrups:
        return out
    smax = stirrup_s_max(inputs)
    assumed = inputs.section.get("stirrup", 8.0)
    reqs = stirrup_requirements(inputs, sp.id, sp.stirrups)
    for k, (z, req) in enumerate(zip(sp.stirrups, reqs)):
        def add(code, msg, sev="error"):
            out.append(Issue(code, sev, f"stirrups: {msg}", sp.id, k,
                             layer="stirrups"))
        prov = z.asw_s()
        if z.legs < 2:
            add("STIRRUP_LEGS", f"{z.legs} leg(s); at least 2 are needed")
        if prov + 1e-6 < req:
            add("ASW_LOW", f"Asw/s {prov:.0f} mm²/m provided < "
                f"{req:.0f} mm²/m required")
        if z.spacing > smax + 1e-6:
            add("S_MAX", f"spacing {z.spacing:g} mm exceeds the maximum "
                f"{smax:.0f} mm (0.75·d·(1+cotα))")
        if not any(abs(z.diameter - d) < 1e-9 for d in p.stirrup_diameters):
            add("STIRRUP_DIAMETER", f"Ø{z.diameter:g} is not in the available "
                "stirrup diameters", "warning")
        if not any(abs(z.spacing - s) < 1e-9 for s in p.stirrup_spacings):
            add("STIRRUP_SPACING", f"{z.spacing:g} mm is not in the available "
                "spacings", "warning")
        if z.diameter > assumed + 1e-9:
            add("STIRRUP_COVER", f"Ø{z.diameter:g} is larger than the Ø"
                f"{assumed:g} assumed for the cover of the bars", "warning")
    return out


def check_segment(seg: SegmentDetail, inputs: SegmentInputs) -> CheckReport:
    """Verify *seg* against its *inputs*: structure, layer rules, area vs
    required steel (with As,min / As,max), symmetric curtailment between
    adjacent zones, through bars and number of diameters."""
    rep = CheckReport(list(validate_structure(seg)))
    # topology: the detail must describe the spans of the inputs
    want = [(s.id, s.length) for s in inputs.spans]
    have = [(s.id, s.length) for s in seg.spans]
    if (len(want) != len(have)
            or any(a[0] != b[0] or abs(a[1] - b[1]) > _tol(b[1])
                   for a, b in zip(have, want))):
        rep.issues.append(Issue(
            "TOPOLOGY", "error",
            "the spans of the detail differ from those of the calculation "
            "(the beam changed)"))
        return rep

    p = inputs.params_obj
    beam = inputs.beam()
    as_max = beam.as_max
    zones_flat = []                  # (span id, index, zone)
    for sp in seg.spans:
        reqs = required_areas(inputs, sp.id, sp.zones)
        for k, (z, (rb, rt)) in enumerate(zip(sp.zones, reqs)):
            zones_flat.append((sp.id, k, z))
            for face, req in (("bottom", rb), ("top", rt)):
                lay = z.layers(face)
                rep.issues += _check_layers(lay, face, beam, p, sp.id, k)
                prov = layers_area(lay)
                if prov + 1e-6 < req:
                    rep.issues.append(Issue(
                        "AS_LOW", "error",
                        f"{face}: {prov:.0f} mm² provided < {req:.0f} mm² "
                        "required", sp.id, k, face))
                if prov > as_max:
                    rep.issues.append(Issue(
                        "AS_MAX", "error",
                        f"{face}: {prov:.0f} mm² exceeds As,max "
                        f"{as_max:.0f} mm²", sp.id, k, face))

    # stirrups (their own zones)
    if inputs.shear:
        for sp in seg.spans:
            rep.issues += _check_stirrups(sp, inputs, p)

    # symmetric curtailment between adjacent zones (whole segment)
    if p.symmetry == "rule":
        for (s1, k1, z1), (s2, k2, z2) in zip(zones_flat, zones_flat[1:]):
            for face in FACES:
                a = Layout([BarGroup(n, d) for n, d in z1.layers(face)])
                b = Layout([BarGroup(n, d) for n, d in z2.layers(face)])
                if a.layers and b.layers and not layouts_compatible(a, b):
                    rep.issues.append(Issue(
                        "CURTAILMENT", "error",
                        f"{face}: {layers_text(z1.layers(face))} → "
                        f"{layers_text(z2.layers(face))} is not a symmetric "
                        "curtailment", s2, k2, face))

    # through bars and number of diameters, per face
    for face in FACES:
        firsts = [(s, k, z.layers(face)[0]) for s, k, z in zones_flat
                  if z.layers(face)]
        if p.n_through > 0 and firsts:
            ds = [round(f[1], 6) for _, _, f in firsts]
            ref = max(set(ds), key=ds.count)
            for s, k, (n, d) in firsts:
                if abs(d - ref) > 1e-6 or n < p.n_through:
                    rep.issues.append(Issue(
                        "THROUGH", "error",
                        f"{face}: the outer layer should hold at least "
                        f"{p.n_through} through bars of Ø{ref:g}", s, k, face))
        diams = {d for _, _, z in zones_flat for _, d in z.layers(face)}
        if len(diams) > p.max_diameters_beam:
            rep.issues.append(Issue(
                "DIAMETERS", "warning",
                f"{face}: {len(diams)} different diameters along the beam "
                f"(max {p.max_diameters_beam})", None, None, face))
    return rep


# ======================================================================
# From the design proposal
# ======================================================================

def details_from_rows(rows: list, inputs: list | None = None) -> dict:
    """The proposal as ``{beam key: BeamDetail}`` from the rows of
    ``design_beam_bars`` (one row per zone). The key is the beam tag, or the
    automatic name for an untagged beam (the application *promotes* such a
    beam to a tag before saving — see :func:`rekey`). Every zone is ``auto``.
    """
    beams: dict = {}
    for r in rows:
        key = r.get("beam_tag") or r["beam"]
        b = beams.setdefault(key, {})
        seg = b.setdefault(r["beam"], {})
        seg.setdefault(r["span"], []).append(r)
    out = {}
    for key, segs in beams.items():
        segments = []
        for sname in sorted(segs, key=_natural):
            spans = []
            for sid, zrows in segs[sname].items():
                zrows = sorted(zrows, key=lambda z: z["x0"])
                zones = [ZoneDetail(
                    float(z["x0"]), float(z["x1"]), "auto",
                    _layers(z.get("bottom_layers")),
                    _layers(z.get("top_layers"))) for z in zrows]
                spans.append(SpanDetail(sid, max(z["x1"] for z in zrows),
                                        zones))
            segments.append(SegmentDetail(sname, spans))
        out[key] = BeamDetail(key, segments)
    if inputs:
        attach_stirrups(out, inputs)
    return out


def attach_stirrups(details: dict, inputs: list) -> None:
    """Fill the automatic stirrups of *details* (``{key: BeamDetail}``) from
    the shear stations of *inputs*; segments are matched by position. Spans
    without shear data are left without stirrups."""
    for key, det in details.items():
        ins = sorted((i for i in inputs if i.tag == key),
                     key=lambda i: _natural(i.name))
        for n, seg in enumerate(det.segments):
            if n >= len(ins):
                continue
            prop = propose_stirrups(ins[n])
            for sp in seg.spans:
                sp.stirrups = [copy.deepcopy(z) for z in prop.get(sp.id, [])]


def stamp(detail: BeamDetail, inputs: list) -> BeamDetail:
    """Set ``based_on`` from the inputs of the beam's segments."""
    sigs = [i.signature() for i in sorted(
        (i for i in inputs if i.tag == detail.tag), key=lambda i: _natural(i.name))]
    detail.based_on = hashlib.sha1("|".join(sigs).encode()).hexdigest()[:16]
    return detail


def is_outdated(detail: BeamDetail, inputs: list) -> bool:
    """True when the calculation changed since *detail* was built."""
    probe = stamp(BeamDetail(detail.tag), inputs)
    return bool(detail.based_on) and detail.based_on != probe.based_on


def rekey(doc: DetailDocument, old: str, new: str) -> None:
    """Rename a beam key (promoting an automatic beam to a tag)."""
    if old not in doc.beams:
        raise DetailError(f"beam {old!r} not found")
    if new in doc.beams and new != old:
        raise DetailError(f"beam {new!r} already exists")
    b = doc.beams.pop(old)
    b.tag = new
    doc.beams[new] = b


# ======================================================================
# Editing
# ======================================================================

class DetailEditor:
    """Edit one segment with undo / redo.

    Every operation snapshots the segment first, marks what it touches
    ``manual`` and raises :class:`DetailError` (leaving the segment as it was)
    on invalid arguments. ``proposal`` — the automatic segment — is what
    ``reset_*`` restores; ``inputs`` (optional) enable :meth:`checks`.
    """

    def __init__(self, segment: SegmentDetail,
                 inputs: SegmentInputs | None = None,
                 proposal: SegmentDetail | None = None):
        self._seg = copy.deepcopy(segment)
        self.inputs = inputs
        self.proposal = copy.deepcopy(proposal) if proposal else None
        self._undo: list = []
        self._redo: list = []
        self.last_label = ""

    # -- state ---------------------------------------------------------
    @property
    def segment(self) -> SegmentDetail:
        return self._seg

    @property
    def can_undo(self) -> bool:
        return bool(self._undo)

    @property
    def can_redo(self) -> bool:
        return bool(self._redo)

    def undo(self) -> bool:
        if not self._undo:
            return False
        label, snap = self._undo.pop()
        self._redo.append((label, self._seg))
        self._seg = snap
        self.last_label = f"undo {label}"
        return True

    def redo(self) -> bool:
        if not self._redo:
            return False
        label, snap = self._redo.pop()
        self._undo.append((label, self._seg))
        self._seg = snap
        self.last_label = label
        return True

    def checks(self) -> CheckReport:
        if self.inputs is None:
            return CheckReport(validate_structure(self._seg))
        rep = check_segment(self._seg, self.inputs)
        if not any(i.code in ("TOPOLOGY", "ZONES") for i in rep.issues):
            from .detailing_lengths import length_issues  # (needs eurocodepy)
            rep.issues += length_issues(self._seg, self.inputs)
        return rep

    def _commit(self, label, mutate):
        snap = copy.deepcopy(self._seg)
        try:
            mutate(self._seg)
            issues = validate_structure(self._seg)
            if issues:
                raise DetailError(issues[0].message)
        except Exception:
            self._seg = snap                 # leave it as it was
            raise
        self._undo.append((label, snap))
        self._redo.clear()
        self.last_label = label

    # -- operations ----------------------------------------------------
    def _zone(self, seg, span_id, k):
        sp = seg.span(span_id)
        if not 0 <= k < len(sp.zones):
            raise DetailError(f"span {span_id} has no zone {k}")
        return sp, sp.zones[k]

    def set_layers(self, span_id, k, face, layers):
        """Replace the layers of one face of a zone (``[(n, Ø), …]``)."""
        new = _layers(layers)
        if face not in FACES:
            raise DetailError(f"face must be one of {FACES}")

        def do(seg):
            _sp, z = self._zone(seg, span_id, k)
            setattr(z, face, new)
            z.origin = "manual"
        self._commit(f"set {face} layers of {span_id}.{k + 1}", do)

    def set_zone(self, span_id, k, bottom=None, top=None):
        """Replace the layers of one or both faces of a zone in ONE undo step
        (``None`` leaves that face as it is)."""
        nb = None if bottom is None else _layers(bottom)
        nt = None if top is None else _layers(top)
        if nb is None and nt is None:
            raise DetailError("nothing to set")

        def do(seg):
            _sp, z = self._zone(seg, span_id, k)
            if nb is not None:
                z.bottom = nb
            if nt is not None:
                z.top = nt
            z.origin = "manual"
        self._commit(f"set layers of {span_id}.{k + 1}", do)

    def split_zone(self, span_id, k, x):
        """Split zone *k* at *x* [m from the span start]; both halves keep
        the layers and become manual."""
        def do(seg):
            sp, z = self._zone(seg, span_id, k)
            tol = _tol(sp.length)
            if not z.x0 + tol < x < z.x1 - tol:
                raise DetailError(
                    f"x={x:.3f} is not inside zone {k + 1} "
                    f"[{z.x0:.3f}, {z.x1:.3f}]")
            a = ZoneDetail(z.x0, x, "manual", list(z.bottom), list(z.top))
            b = ZoneDetail(x, z.x1, "manual", list(z.bottom), list(z.top))
            sp.zones[k:k + 1] = [a, b]
        self._commit(f"split {span_id}.{k + 1}", do)

    def merge_zones(self, span_id, k):
        """Merge zone *k* with the next one. Each face keeps the layers with
        the larger area (the conservative choice)."""
        def do(seg):
            sp, z = self._zone(seg, span_id, k)
            if k + 1 >= len(sp.zones):
                raise DetailError(f"zone {k + 1} is the last of {span_id}")
            n = sp.zones[k + 1]
            pick = lambda a, b: list(a if layers_area(a) >= layers_area(b) else b)
            sp.zones[k:k + 2] = [ZoneDetail(
                z.x0, n.x1, "manual", pick(z.bottom, n.bottom),
                pick(z.top, n.top))]
        self._commit(f"merge {span_id}.{k + 1}+{k + 2}", do)

    def move_cut(self, span_id, k, x):
        """Move the limit between zone *k* and zone *k + 1* to *x*."""
        def do(seg):
            sp, z = self._zone(seg, span_id, k)
            if k + 1 >= len(sp.zones):
                raise DetailError(f"zone {k + 1} has no cut after it")
            n = sp.zones[k + 1]
            tol = _tol(sp.length)
            if not z.x0 + tol < x < n.x1 - tol:
                raise DetailError(
                    f"the cut must stay inside [{z.x0:.3f}, {n.x1:.3f}]")
            z.x1 = n.x0 = x
            z.origin = n.origin = "manual"
        self._commit(f"move cut {span_id}.{k + 1}|{k + 2}", do)

    # -- stirrups (their own zones) ------------------------------------
    def _stzone(self, seg, span_id, k):
        sp = seg.span(span_id)
        if not 0 <= k < len(sp.stirrups):
            raise DetailError(f"span {span_id} has no stirrup zone {k}")
        return sp, sp.stirrups[k]

    def set_stirrups(self, span_id, k, legs=None, diameter=None, spacing=None):
        """Change legs / Ø [mm] / spacing [mm] of stirrup zone *k* (``None``
        keeps that value) in one undo step."""
        if legs is None and diameter is None and spacing is None:
            raise DetailError("nothing to set")
        if legs is not None and int(legs) < 1:
            raise DetailError("legs must be >= 1")
        if diameter is not None and float(diameter) <= 0:
            raise DetailError("the diameter must be > 0")
        if spacing is not None and float(spacing) <= 0:
            raise DetailError("the spacing must be > 0")

        def do(seg):
            _sp, z = self._stzone(seg, span_id, k)
            if legs is not None:
                z.legs = int(legs)
            if diameter is not None:
                z.diameter = float(diameter)
            if spacing is not None:
                z.spacing = float(spacing)
            z.origin = "manual"
        self._commit(f"set stirrups of {span_id}.s{k + 1}", do)

    def split_stirrup_zone(self, span_id, k, x):
        def do(seg):
            sp, z = self._stzone(seg, span_id, k)
            tol = _tol(sp.length)
            if not z.x0 + tol < x < z.x1 - tol:
                raise DetailError(
                    f"x={x:.3f} is not inside stirrup zone {k + 1} "
                    f"[{z.x0:.3f}, {z.x1:.3f}]")
            a = StirrupZone(z.x0, x, "manual", z.legs, z.diameter, z.spacing)
            b = StirrupZone(x, z.x1, "manual", z.legs, z.diameter, z.spacing)
            sp.stirrups[k:k + 1] = [a, b]
        self._commit(f"split {span_id}.s{k + 1}", do)

    def merge_stirrup_zones(self, span_id, k):
        """Merge stirrup zone *k* with the next; the heavier stirrups (larger
        Asw/s) are kept — the conservative choice."""
        def do(seg):
            sp, z = self._stzone(seg, span_id, k)
            if k + 1 >= len(sp.stirrups):
                raise DetailError(f"stirrup zone {k + 1} is the last of "
                                  f"{span_id}")
            n = sp.stirrups[k + 1]
            keep = z if z.asw_s() >= n.asw_s() else n
            sp.stirrups[k:k + 2] = [StirrupZone(
                z.x0, n.x1, "manual", keep.legs, keep.diameter, keep.spacing)]
        self._commit(f"merge {span_id}.s{k + 1}+{k + 2}", do)

    def move_stirrup_cut(self, span_id, k, x):
        """Move the limit between stirrup zone *k* and *k + 1* to *x*."""
        def do(seg):
            sp, z = self._stzone(seg, span_id, k)
            if k + 1 >= len(sp.stirrups):
                raise DetailError(f"stirrup zone {k + 1} has no cut after it")
            n = sp.stirrups[k + 1]
            tol = _tol(sp.length)
            if not z.x0 + tol < x < n.x1 - tol:
                raise DetailError(
                    f"the cut must stay inside [{z.x0:.3f}, {n.x1:.3f}]")
            z.x1 = n.x0 = x
            z.origin = n.origin = "manual"
        self._commit(f"move stirrup cut {span_id}.s{k + 1}|{k + 2}", do)

    def reset_stirrup_zone(self, span_id, k):
        """Restore stirrup zone *k* from the proposal (the proposal zone that
        contains its midpoint); ``auto`` only if the limits are the same."""
        if self.proposal is None:
            raise DetailError("there is no automatic proposal to restore")

        def do(seg):
            sp, z = self._stzone(seg, span_id, k)
            try:
                psp = self.proposal.span(span_id)
            except DetailError:
                raise DetailError(f"the proposal has no span {span_id}")
            mid = 0.5 * (z.x0 + z.x1)
            src = next((q for q in psp.stirrups if q.x0 <= mid <= q.x1), None)
            if src is None:
                raise DetailError("the proposal has no stirrup zone there")
            tol = _tol(sp.length)
            same = abs(src.x0 - z.x0) < tol and abs(src.x1 - z.x1) < tol
            z.legs, z.diameter, z.spacing = src.legs, src.diameter, src.spacing
            z.origin = "auto" if same else "manual"
        self._commit(f"reset {span_id}.s{k + 1}", do)

    def reset_zone(self, span_id, k):
        """Restore the layers of zone *k* from the proposal (the proposal zone
        that contains its midpoint). The zone becomes ``auto`` only if its
        limits equal that proposal zone's."""
        if self.proposal is None:
            raise DetailError("there is no automatic proposal to restore")

        def do(seg):
            sp, z = self._zone(seg, span_id, k)
            try:
                psp = self.proposal.span(span_id)
            except DetailError:
                raise DetailError(f"the proposal has no span {span_id}")
            mid = 0.5 * (z.x0 + z.x1)
            src = next((q for q in psp.zones if q.x0 <= mid <= q.x1), None)
            if src is None:
                raise DetailError("the proposal has no zone there")
            tol = _tol(sp.length)
            same = abs(src.x0 - z.x0) < tol and abs(src.x1 - z.x1) < tol
            z.bottom, z.top = list(src.bottom), list(src.top)
            z.origin = "auto" if same else "manual"
        self._commit(f"reset {span_id}.{k + 1}", do)

    def reset_all(self):
        """Back to the full automatic proposal."""
        if self.proposal is None:
            raise DetailError("there is no automatic proposal to restore")
        prop = copy.deepcopy(self.proposal)

        def do(seg):
            seg.spans = prop.spans
        self._commit("reset all", do)


# ======================================================================
# Reconciliation with a new proposal
# ======================================================================

@dataclass
class ReconcileReport:
    status: str                     # new | unchanged | merged | stale | orphan
    kept_manual: int = 0
    replaced_auto: int = 0
    issues: list = field(default_factory=list)


def _same_topology(a: SegmentDetail, b: SegmentDetail) -> bool:
    if len(a.spans) != len(b.spans):
        return False
    return all(x.id == y.id and abs(x.length - y.length) <= _tol(y.length)
               for x, y in zip(a.spans, b.spans))


def _fill_list(zones: list, pzones: list, length: float, clone) -> list:
    """*zones* with its manual zones kept and every auto zone re-filled from
    the proposal zones *pzones* (cut at the limits of the auto zone;
    ``clone(lo, hi, q)`` makes the piece). Pieces cut from the *same* proposal
    zone are joined back into one zone; pieces of different proposal zones
    stay apart even if they hold the same bars."""
    pieces = []                          # (zone, index of the proposal zone)
    for z in zones:
        if z.origin == "manual":
            pieces.append((copy.deepcopy(z), None))
            continue
        for qi, q in enumerate(pzones):
            lo, hi = max(z.x0, q.x0), min(z.x1, q.x1)
            if hi - lo > _tol(length):
                pieces.append((clone(lo, hi, q), qi))
    merged = []
    for z, qi in pieces:
        if (merged and qi is not None and merged[-1][1] == qi
                and abs(merged[-1][0].x1 - z.x0) <= _tol(length)):
            merged[-1][0].x1 = z.x1
        else:
            merged.append((z, qi))
    return [z for z, _ in merged]


def _fill_auto(span: SpanDetail, proposal: SpanDetail) -> SpanDetail:
    """The span with its manual zones (bars and stirrups) kept and every auto
    zone re-filled from the proposal. Stirrups: no stored ones → the
    proposal's; no proposal ones → the stored ones are left as they are."""
    zones = _fill_list(
        span.zones, proposal.zones, span.length,
        lambda lo, hi, q: ZoneDetail(lo, hi, "auto", list(q.bottom),
                                     list(q.top)))
    if not span.stirrups:
        stirrups = [copy.deepcopy(z) for z in proposal.stirrups]
    elif not proposal.stirrups:
        stirrups = [copy.deepcopy(z) for z in span.stirrups]
    else:
        stirrups = _fill_list(
            span.stirrups, proposal.stirrups, span.length,
            lambda lo, hi, q: StirrupZone(lo, hi, "auto", q.legs, q.diameter,
                                          q.spacing))
    return SpanDetail(span.id, span.length, zones, stirrups)


def reconcile_segment(existing: SegmentDetail | None,
                      proposal: SegmentDetail,
                      inputs: SegmentInputs | None = None):
    """Merge a new *proposal* into the *existing* segment.

    * no existing detail → the proposal (``new``);
    * the spans changed (another beam) → the existing detail is returned
      untouched and marked ``stale``: the caller decides to rebase on the
      proposal or to keep it;
    * otherwise manual zones are kept as they are (and reported if they no
      longer comply), auto zones follow the proposal (``merged`` /
      ``unchanged``). Nothing is erased silently.
    """
    if existing is None:
        out = copy.deepcopy(proposal)
        rep = ReconcileReport("new", replaced_auto=sum(
            len(s.zones) + len(s.stirrups) for s in out.spans))
        if inputs is not None:
            rep.issues = check_segment(out, inputs).issues
        return out, rep
    if not _same_topology(existing, proposal):
        rep = ReconcileReport("stale")
        if inputs is not None:
            rep.issues = check_segment(existing, inputs).issues
        return copy.deepcopy(existing), rep

    out = SegmentDetail(existing.name, [
        _fill_auto(e, proposal.span(e.id)) for e in existing.spans])
    allz = [z for s in out.spans for z in list(s.zones) + list(s.stirrups)]
    kept = sum(1 for z in allz if z.origin == "manual")
    auto = sum(1 for z in allz if z.origin == "auto")
    rep = ReconcileReport(
        "unchanged" if out.to_dict() == existing.to_dict() else "merged",
        kept, auto)
    if inputs is not None:
        rep.issues = check_segment(out, inputs).issues
    return out, rep


def reconcile_document(existing: DetailDocument | None, proposals: dict,
                       inputs: list):
    """Merge new proposals (``{key: BeamDetail}``) into *existing*.

    Returns ``(document, {key: {segment name: ReconcileReport}})``. Beams only
    in the proposals are added (``new``); beams only in the document are kept
    and reported ``orphan``. ``based_on`` of the merged beams is refreshed
    from *inputs* (a list of :class:`SegmentInputs`).
    """
    doc = DetailDocument(
        {k: copy.deepcopy(v) for k, v in (existing.beams if existing else {}).items()})
    reports: dict = {}
    for key, prop in proposals.items():
        old = doc.beams.get(key)
        # Segments are matched by POSITION in the beam, not by name: a name
        # follows the beam's display name (and its ``.1`` / ``.2`` suffixes),
        # which changes when the beam is renamed or promoted to a tag.
        ins = sorted((i for i in inputs if i.tag == key),
                     key=lambda i: _natural(i.name))
        segs, rep = [], {}
        for n, ps in enumerate(prop.segments):
            es = old.segments[n] if old is not None and n < len(old.segments) \
                else None
            s, r = reconcile_segment(es, ps, ins[n] if n < len(ins) else None)
            s.name = ps.name                        # follow the proposal's name
            segs.append(s)
            rep[ps.name] = r
        for es in (old.segments[len(prop.segments):] if old else []):
            segs.append(copy.deepcopy(es))            # segments that vanished
            rep[es.name] = ReconcileReport("orphan")
        nb = BeamDetail(key, segs)
        stamp(nb, [i for i in inputs if i.tag == key])
        doc.beams[key] = nb
        reports[key] = rep
    for key in set(doc.beams) - set(proposals):
        reports[key] = {s.name: ReconcileReport("orphan")
                        for s in doc.beams[key].segments}
    return doc, reports


def refresh_document(existing: DetailDocument | None, rows: list,
                     inputs: list):
    """After a design run: reconcile the *stored* decisions with the new
    proposal (built from the design *rows*) and return
    ``(document, {key: {segment: ReconcileReport}})``.

    Only beams already in the document are touched — a beam enters the
    document when the user first edits it — so a model with no manual detail
    stays exactly as it is. ``based_on`` is refreshed from *inputs*.
    """
    doc = existing if existing is not None else DetailDocument()
    keep = {k: v for k, v in details_from_rows(rows, inputs).items()
            if k in doc.beams}
    return reconcile_document(doc, keep, inputs)


def refresh_model_detail(struc, rows: list, inputs: list):
    """:func:`refresh_document` applied to ``struc.beam_detail`` in place.
    Returns ``(summary, changed)`` — the counts of :func:`summarize_reports`
    and whether the stored decisions changed (the model then needs saving)."""
    before = struc.beam_detail.to_dict()
    doc, reports = refresh_document(struc.beam_detail, rows, inputs)
    struc.beam_detail = doc
    return summarize_reports(reports), doc.to_dict() != before


def summary_text(summary: dict) -> str:
    """One sentence for the Beam bars header ('' when nothing is stored)."""
    if not summary or not summary.get("beams"):
        return ""
    bits = [f"manual detail kept on {summary['beams']} beam(s) "
            f"({summary['manual_zones']} zone(s))"]
    if summary["errors"]:
        bits.append(f"{summary['errors']} check error(s) to review")
    if summary["stale"]:
        bits.append(f"{summary['stale']} beam(s) changed since (detail "
                    "left as it was)")
    if summary["orphan"]:
        bits.append(f"{summary['orphan']} detail(s) without a beam now")
    return "; ".join(bits) + "."


def summarize_reports(reports: dict) -> dict:
    """Counts over the reports of :func:`refresh_document`:
    ``{"beams", "manual_zones", "stale", "orphan", "errors", "warnings"}``."""
    out = {"beams": len(reports), "manual_zones": 0, "stale": 0, "orphan": 0,
           "errors": 0, "warnings": 0}
    for segs in reports.values():
        for r in segs.values():
            out["manual_zones"] += r.kept_manual
            out["stale"] += r.status == "stale"
            out["orphan"] += r.status == "orphan"
            out["errors"] += sum(i.severity == "error" for i in r.issues)
            out["warnings"] += sum(i.severity == "warning" for i in r.issues)
    return out


# ======================================================================
# The automatic proposal as a function of the inputs
# ======================================================================

@dataclass
class ZoneProposal:
    """One bar zone of a proposal with its raw requirement [mm²] (the value
    the design reports, before the As,min floors)."""
    span: str
    k: int
    x0: float                       # span-local [m]
    x1: float
    req_bottom: float
    req_top: float


@dataclass
class Proposal:
    """What :func:`propose_segment` returns for one segment."""
    segment: SegmentDetail          # bars and stirrups, every zone "auto"
    zones: list                     # [ZoneProposal], same order as the zones
    through: dict                   # {"bottom": Ø | None, "top": Ø | None}
    notes: list                     # warnings of the bar selection
    error: str | None = None        # no layout fits (zones then hold no bars)


def propose_segment(inputs: SegmentInputs) -> Proposal:
    """The automatic bars and stirrups of one segment, from its inputs alone
    (no model): zones from the cutoff points of the shifted envelope — or from
    fixed fractions — requirement per zone, bar selection with symmetry,
    through bars and uniform diameters (:func:`beam_reinforcement`), and the
    stirrups (:func:`propose_stirrups`). The design uses exactly this function,
    so the detailing window can regenerate a proposal when a parameter
    changes."""
    p = inputs.params_obj
    beam = inputs.beam()
    stations = sorted(inputs.stations)
    xs = [t[0] for t in stations]
    off = inputs.offsets()
    a_l = inputs.a_l

    cut = []
    fr = []
    if p.zone_mode == "cutoff":
        sh_b, sh_t = [], []
        for x in xs:                       # envelopes shifted by a_l
            win = [t for t in stations if abs(t[0] - x) <= a_l + 1e-9]
            sh_b.append(max(t[1] for t in win))
            sh_t.append(max(t[2] for t in win))
        floor = beam.as_min
        cut = sorted(set(cutoff_points(xs, sh_b, p.cutoff_levels, floor)
                         + cutoff_points(xs, sh_t, p.cutoff_levels, floor)))
    else:
        tot = float(sum(p.zones))
        fr = [z / tot for z in p.zones]

    zones, spans = [], []
    for sp in inputs.spans:
        s0 = off[sp.id]
        s1 = s0 + sp.length
        if p.zone_mode == "cutoff":
            tol_x = 1e-6 * max(sp.length, 1.0)
            inner = [c for c in cut if s0 + tol_x < c < s1 - tol_x]
            edges = merge_short_zones([s0] + inner + [s1], p.min_zone_length)
        else:
            edges = [s0]
            for f in fr:
                edges.append(edges[-1] + f * sp.length)
        reqs = zone_requirements(stations, list(zip(edges[:-1], edges[1:])),
                                 a_l, p.support_bottom_ratio)
        for z, (rb, rt) in enumerate(reqs):
            zones.append(ZoneProposal(sp.id, z, edges[z] - s0,
                                      edges[z + 1] - s0, rb, rt))
            spans.append(Span(length=edges[z + 1] - edges[z], as_bottom=rb,
                              as_top=rt, name=f"{sp.id}.{z + 1}"))

    beam.spans = spans
    res, error, notes = None, None, []
    try:
        res = beam_reinforcement(beam, **p.reinforcement_kwargs())
        notes = list(res.warnings)
    except ValueError as err:
        error = str(err)
        notes = [error]

    by_span: dict = {}
    for i, zp in enumerate(zones):
        bottom = top = []
        if res is not None:
            sr = res.spans[i]
            bottom = [(g.n, float(g.diameter)) for g in sr.bottom.layers]
            top = [(g.n, float(g.diameter)) for g in sr.top.layers]
        by_span.setdefault(zp.span, []).append(
            ZoneDetail(zp.x0, zp.x1, "auto", bottom, top))
    stirrups = propose_stirrups(inputs)
    seg = SegmentDetail(inputs.name, [
        SpanDetail(sp.id, sp.length, by_span.get(sp.id, []),
                   [copy.deepcopy(z) for z in stirrups.get(sp.id, [])])
        for sp in inputs.spans])
    through = {"bottom": None, "top": None}
    if res is not None:
        through = {k: v for k, v in res.through.items()}
    return Proposal(seg, zones, through, notes, error)


def proposals_from_inputs(inputs: list) -> dict:
    """The automatic proposals of every beam in *inputs* —
    ``{key: BeamDetail}`` with the segments of a beam sorted by name — made
    from the inputs alone (what ``details_from_rows`` gives from the design
    rows)."""
    by_key: dict = {}
    for i in inputs:
        by_key.setdefault(i.tag, []).append(i)
    out = {}
    for key, ins in by_key.items():
        ins = sorted(ins, key=lambda i: _natural(i.name))
        out[key] = BeamDetail(
            key, [propose_segment(i).segment for i in ins])
    return out


# ======================================================================
# Parameters: layers and recalculation
# ======================================================================

def refit_inputs(inputs: SegmentInputs, params: BeamBarParams) -> SegmentInputs:
    """*inputs* with new effective *params*, recomputing what depends on them:
    the stirrup diameter assumed for the cover (hence the cover of the bars)
    and the envelope shift ``a_l``. The bar diameter assumed for ``d`` and the
    aggregate size come from the section / preferences, not from the
    parameters being changed, so they are kept."""
    sec = dict(inputs.section)
    d_est = inputs.params.get("d_bar_est", sec.get("d_bar_est", 16.0))
    dg = inputs.params.get("dg", sec.get("dg", 20.0))
    rc_cover = sec.get("rc_cover",
                       sec["cover"] + sec["stirrup"] + d_est / 2.0)
    new = replace(params, d_bar_est=d_est, dg=dg)
    sec["rc_cover"] = rc_cover
    sec["stirrup"] = new.stirrup_diameter_mm
    sec["cover"] = rc_cover - new.stirrup_diameter_mm - d_est / 2.0
    sec["d_bar_est"] = d_est
    sec["dg"] = dg
    a_l = new.shift_d * (sec["h"] - rc_cover) / 1000.0
    pd = asdict(new)
    pd = {k: (list(v) if isinstance(v, tuple) else v) for k, v in pd.items()}
    return replace(inputs, section=sec, params=pd, a_l=a_l)


def _check_override_keys(d: dict, allowed, what: str) -> dict:
    d = dict(d or {})
    bad = set(d) - set(allowed)
    if bad:
        raise DetailError(f"{what}: unknown/not allowed parameter(s) "
                          f"{sorted(bad)}")
    return d


@dataclass
class ParameterLayers:
    """The three layers a beam's parameters come from, most specific last:
    ``defaults`` (the model's design preferences: the ``beam_*`` keys and
    ``dmax``), ``sections`` (``{section name: overrides}``) and ``beams``
    (``{tag: overrides}``) — overrides are ``{parameter name: value}``. The
    effective parameters of a beam are exactly what
    :func:`~xdfem2d.beam_bars_params.resolve_beam_params` gives."""
    defaults: dict = field(default_factory=dict)
    sections: dict = field(default_factory=dict)
    beams: dict = field(default_factory=dict)

    def effective(self, tag: str, section_name: str,
                  base: BeamBarParams | None = None) -> BeamBarParams:
        from types import SimpleNamespace
        sec = SimpleNamespace(
            beam_overrides=self.sections.get(section_name) or None)
        p = resolve_beam_params(self.defaults, sec, self.beams.get(tag))
        if base is not None:
            p = replace(p, d_bar_est=base.d_bar_est, dg=base.dg)
        return p

    def inherited(self, scope: str, tag: str = "",
                  section_name: str = "") -> BeamBarParams:
        """What a field takes when it is not overridden at *scope* — the level
        above it: for ``"beam"`` the section's, for ``"section"`` the defaults,
        for ``"defaults"`` the built-in values."""
        from types import SimpleNamespace
        if scope == "defaults":
            return BeamBarParams()
        if scope == "section":
            return resolve_beam_params(self.defaults)
        if scope == "beam":
            sec = SimpleNamespace(
                beam_overrides=self.sections.get(section_name) or None)
            return resolve_beam_params(self.defaults, sec)
        raise DetailError(f"unknown scope {scope!r}")

    def to_dict(self) -> dict:
        def clean(d):
            return {k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in d.items()}
        return {"defaults": clean(self.defaults),
                "sections": {k: clean(v) for k, v in self.sections.items()},
                "beams": {k: clean(v) for k, v in self.beams.items()}}

    @classmethod
    def from_dict(cls, d: dict | None) -> "ParameterLayers":
        if not d:
            return cls()
        if not isinstance(d, dict):
            raise DetailError("parameters must be an object")
        pref_keys = set(PREF_KEYS)
        out = cls(
            _check_override_keys(d.get("defaults"), pref_keys, "defaults"),
            {str(k): _check_override_keys(v, SECTION_PARAMS, f"section {k!r}")
             for k, v in (d.get("sections") or {}).items()},
            {str(k): _check_override_keys(v, BEAM_PARAMS, f"beam {k!r}")
             for k, v in (d.get("beams") or {}).items()})
        try:                                    # every layer must resolve
            resolve_beam_params(out.defaults)
            for name, ov in out.sections.items():
                out.effective("", name)
            for tag in out.beams:
                out.effective(tag, "")
        except ValueError as e:
            raise DetailError(f"invalid parameters: {e}") from None
        return out


def refit_for_layers(inputs: list, layers: ParameterLayers) -> list:
    """Every segment of *inputs* with the effective parameters its beam and
    section get from *layers* (:func:`refit_inputs`)."""
    out = []
    for i in inputs:
        eff = layers.effective(i.tag, i.section_name, i.params_obj)
        out.append(refit_inputs(i, eff))
    return out


# ======================================================================
# The self-contained package
# ======================================================================

@dataclass
class BeamPackage:
    """Inputs + the automatic proposal + the user's detail in one JSON: all a
    standalone window needs (the proposal is what ``reset`` returns to and
    what a beam without decisions shows)."""
    inputs: list = field(default_factory=list)          # [SegmentInputs]
    detail: DetailDocument = field(default_factory=DetailDocument)
    proposal: DetailDocument = field(default_factory=DetailDocument)
    parameters: ParameterLayers = field(default_factory=ParameterLayers)
    version: int = SCHEMA_VERSION

    def to_dict(self) -> dict:
        return {"version": self.version,
                "inputs": [i.to_dict() for i in self.inputs],
                "proposal": self.proposal.to_dict(),
                "parameters": self.parameters.to_dict(),
                "detail": self.detail.to_dict()}

    @classmethod
    def from_dict(cls, d: dict) -> "BeamPackage":
        if not isinstance(d, dict):
            raise DetailError("package must be an object")
        ver = d.get("version", SCHEMA_VERSION)
        if not isinstance(ver, int) or ver > SCHEMA_VERSION:
            raise DetailError(
                f"package version {ver!r} is newer than this program "
                f"understands ({SCHEMA_VERSION})")
        return cls([SegmentInputs.from_dict(i) for i in d.get("inputs", [])],
                   DetailDocument.from_dict(d.get("detail")),
                   DetailDocument.from_dict(d.get("proposal")),
                   ParameterLayers.from_dict(d.get("parameters")), ver)

    def dumps(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def loads(cls, text: str) -> "BeamPackage":
        try:
            return cls.from_dict(json.loads(text))
        except json.JSONDecodeError as e:
            raise DetailError(f"not valid JSON: {e}") from None
