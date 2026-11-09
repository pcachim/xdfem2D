"""Anchorage and lap lengths of the bars of a beam detail.

The calculation belongs to **eurocodepy** (EN 1992-1-1 §8.4 anchorage and §8.7
laps: :mod:`eurocodepy.ec2.uls.anchorage`); this module only supplies the
geometry of a beam — the bond condition of each face, the cover distance of
each layer, which bars end where — and asks eurocodepy for the lengths.

Everything is derived from a :class:`~xdfem2d.detailing.SegmentDetail` and its
:class:`~xdfem2d.detailing.SegmentInputs` (no Qt, no model):

``bar_lengths``       l_b,rqd, l_bd and l_0 of a bar of diameter Ø on a face;
``segment_lengths``   the table for every diameter used on every face;
``curtailments``      where bars stop and how far they must extend (EC2 9.2.1.3:
                      beyond the point where they are no longer needed);
``end_anchorages``    the anchorage the bars of the end zones need;
``through_laps``      laps of the through bars for a commercial bar length;
``length_issues``     warnings for what does not fit.

Simplifications (safe side): the bar is taken fully stressed (``σ_sd = f_yd``)
where its anchorage starts; stirrups give no credit to ``α3``; the bond
condition is good for the bottom face and — beams deeper than 250 mm — poor
for the top face (:func:`eurocodepy.ec2.uls.bond_conditions_beam`).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from eurocodepy.ec2.uls import (
    beam_cover_distance, bond_conditions_beam, design_anchorage_length,
    lap_length,
)

from .detailing import FACES, Issue, SegmentDetail, SegmentInputs

GAMMA_C = 1.5
GAMMA_S = 1.15


@dataclass(frozen=True)
class BarLengths:
    """Lengths [mm] of a bar of diameter ``diameter`` on ``face`` (``n_bars``
    is the largest number of such bars in a layer, which fixes ``cd``)."""
    face: str
    diameter: float
    n_bars: int
    bond: str                # "good" | "poor"
    cd: float
    fbd: float
    lb_rqd: float
    lbd: float               # design anchorage length (shape per the params)
    lb_min: float
    l0: float                # lap length
    l0_min: float
    shape: str               # "straight" | "bent"


@dataclass(frozen=True)
class Curtailment:
    """Bars that start or stop at a zone limit and must extend ``lbd`` past
    it. ``direction`` is +1 when the bars lie to the right of the cut (they
    stop after it), -1 when they lie to the left (they start before it)."""
    face: str
    layer: int               # 0 = outer layer
    span: str                # span of the zone on the LEFT of the cut
    zone: int                # its index in the span
    x_cut: float             # [m along the segment]
    direction: int
    n_bars: int
    diameter: float
    lbd: float               # [mm]
    x_end: float             # [m along the segment]
    outside: bool            # the extension leaves the beam


def _gammas(inputs: SegmentInputs):
    s = inputs.section
    return s.get("gamma_c", GAMMA_C), s.get("gamma_s", GAMMA_S)


def bar_lengths(inputs: SegmentInputs, face: str, diameter: float,
                n_bars: int = 2) -> BarLengths:
    """Anchorage (``l_bd``) and lap (``l_0``) lengths of a bar of *diameter*
    on *face*, asking eurocodepy. ``n_bars`` bars of that diameter in the
    layer set the cover distance ``c_d`` (EC2 Fig. 8.3)."""
    if face not in FACES:
        raise ValueError(f"face must be one of {FACES}")
    sec = inputs.section
    p = inputs.params_obj
    gc, gs = _gammas(inputs)
    cd = beam_cover_distance(sec["b"], sec["cover"] + sec["stirrup"],
                             max(int(n_bars), 1), diameter)
    bond = bond_conditions_beam(sec["h"], face)
    good = bond == "good"
    a = design_anchorage_length(diameter, sec["fck"], sec["fyk"], cd,
                                good_bond=good, shape=p.anchorage_shape,
                                gamma_c=gc, gamma_s=gs)
    lap = lap_length(diameter, sec["fck"], sec["fyk"], cd,
                     rho1=p.lap_percentage, good_bond=good,
                     gamma_c=gc, gamma_s=gs)
    return BarLengths(face, float(diameter), int(n_bars), bond, cd, a["fbd"],
                      a["lb_rqd"], a["lbd"], a["lb_min"], lap["l0"],
                      lap["l0_min"], p.anchorage_shape)


def segment_lengths(segment: SegmentDetail, inputs: SegmentInputs) -> list:
    """One :class:`BarLengths` per (face, diameter) used in the segment, sorted
    by face (top first) and diameter. The layer with the most bars of that
    diameter fixes ``c_d`` (the least favourable)."""
    most: dict = {}
    for sp in segment.spans:
        for z in sp.zones:
            for face in FACES:
                for n, d in z.layers(face):
                    key = (face, float(d))
                    most[key] = max(most.get(key, 0), int(n))
    order = {"top": 0, "bottom": 1}
    return [bar_lengths(inputs, face, d, n)
            for (face, d), n in sorted(most.items(),
                                       key=lambda kv: (order[kv[0][0]],
                                                       kv[0][1]))]


def _flat_zones(segment: SegmentDetail, inputs: SegmentInputs):
    """``[(span id, index, x0, x1, zone)]`` along the segment."""
    off = inputs.offsets()
    out = []
    for sp in segment.spans:
        if sp.id not in off:
            continue
        for k, z in enumerate(sp.zones):
            out.append((sp.id, k, off[sp.id] + z.x0, off[sp.id] + z.x1, z))
    return out


def curtailments(segment: SegmentDetail, inputs: SegmentInputs) -> list:
    """Every place where bars start or stop, with how far they extend.

    For two neighbouring zones A (left) and B (right), layer by layer:
    bars of A that B does not have stop at the cut and extend ``l_bd`` into B
    (``direction = +1``); bars of B that A does not have start ``l_bd`` before
    the cut, inside A (``direction = -1``). The same diameter in the layer →
    only the difference in number; a different diameter → all of both.
    """
    zones = _flat_zones(segment, inputs)
    length = inputs.length()
    cache: dict = {}

    def lbd(face, d, n):
        key = (face, float(d), int(n))
        if key not in cache:
            cache[key] = bar_lengths(inputs, face, d, n).lbd
        return cache[key]

    out = []
    for (sa, ka, _x0a, x1a, za), (_sb, _kb, x0b, _x1b, zb) in zip(zones,
                                                                  zones[1:]):
        x_cut = 0.5 * (x1a + x0b)
        for face in FACES:
            la, lb = za.layers(face), zb.layers(face)
            for k in range(max(len(la), len(lb))):
                ga = la[k] if k < len(la) else None
                gb = lb[k] if k < len(lb) else None
                drops = []                            # (direction, n, d, n_layer)
                if ga and gb and abs(ga[1] - gb[1]) < 1e-9:
                    if ga[0] > gb[0]:
                        drops.append((+1, ga[0] - gb[0], ga[1], ga[0]))
                    elif gb[0] > ga[0]:
                        drops.append((-1, gb[0] - ga[0], gb[1], gb[0]))
                else:
                    if ga:
                        drops.append((+1, ga[0], ga[1], ga[0]))
                    if gb:
                        drops.append((-1, gb[0], gb[1], gb[0]))
                for direction, n, d, n_layer in drops:
                    ext = lbd(face, d, n_layer)
                    x_end = x_cut + direction * ext / 1000.0
                    out.append(Curtailment(
                        face, k, sa, ka, x_cut, direction, int(n), float(d),
                        ext, x_end, x_end < -1e-9 or x_end > length + 1e-9))
    return out


def end_anchorages(segment: SegmentDetail, inputs: SegmentInputs) -> list:
    """The anchorage the bars of the two end zones need: dicts ``{end
    ("start"|"end"), face, layer, n_bars, diameter, lbd}`` [mm] — measured from
    the point where the bar is fully stressed; the support width is not known
    here, so it is information, not a check."""
    zones = _flat_zones(segment, inputs)
    if not zones:
        return []
    out = []
    for end, z in (("start", zones[0][4]), ("end", zones[-1][4])):
        for face in FACES:
            for k, (n, d) in enumerate(z.layers(face)):
                out.append({"end": end, "face": face, "layer": k,
                            "n_bars": int(n), "diameter": float(d),
                            "lbd": bar_lengths(inputs, face, d, n).lbd})
    return out


def lap_count(length: float, bar_length: float, l0: float) -> int:
    """Laps along *length* [m] with bars of *bar_length* [m] and laps of *l0*
    [mm]: each extra bar adds ``bar_length − l0`` of coverage."""
    if bar_length <= 0:
        raise ValueError("bar_length must be > 0")
    net = bar_length - l0 / 1000.0
    if net <= 0:
        raise ValueError("the lap is longer than the bar")
    if length <= bar_length + 1e-9:
        return 0
    return max(0, math.ceil((length - bar_length) / net - 1e-9))


def through_laps(segment: SegmentDetail, inputs: SegmentInputs) -> list:
    """Laps of the through bars (the outer layer, when it has the same Ø in
    every zone): ``[{face, diameter, n_through, l0, length, n_laps}]``."""
    zones = _flat_zones(segment, inputs)
    p = inputs.params_obj
    out = []
    for face in FACES:
        firsts = [z.layers(face)[0] for _s, _k, _a, _b, z in zones
                  if z.layers(face)]
        if not firsts or len({round(d, 6) for _n, d in firsts}) != 1:
            continue
        d = firsts[0][1]
        n = min(n for n, _d in firsts)
        bl = bar_lengths(inputs, face, d, max(n for n, _d in firsts))
        out.append({"face": face, "diameter": float(d), "n_through": int(n),
                    "l0": bl.l0, "length": inputs.length(),
                    "n_laps": lap_count(inputs.length(), p.bar_length, bl.l0)})
    return out


def length_issues(segment: SegmentDetail, inputs: SegmentInputs) -> list:
    """Warnings about the lengths: a bar whose required extension leaves the
    beam (it has to be anchored in the support or the neighbouring member)."""
    out = []
    for c in curtailments(segment, inputs):
        if c.outside:
            where = "start" if c.x_end < 0 else "end"
            out.append(Issue(
                "CURTAIL_BEYOND", "warning",
                f"{c.face}: {c.n_bars}Ø{c.diameter:g} must extend "
                f"{c.lbd:.0f} mm past x = {c.x_cut:.2f} m, beyond the {where} "
                "of the beam (anchorage in the support)",
                c.span, c.zone, c.face))
    return out
