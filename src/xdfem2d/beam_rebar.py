"""Convert required reinforcement areas of a beam into bars.

A beam is a sequence of spans (tramos); each span has a required steel area
per face (bottom / top).  ``beam_reinforcement`` turns those areas into bar
layouts, applying:

  - EC2 §9.2.1.1 minimum area As,min and §9.2.1.1(3) maximum area 0.04·Ac
  - EC2 §8.2 minimum clear spacing (bars per layer limited by the width)
  - uniformisation of diameters between spans (per face, at most
    ``max_diameters_beam`` distinct diameters along the whole beam)

Units: mm, mm², MPa (the length of a span is in m and only used as a weight).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations, product

DIAMETERS_MM = (8, 10, 12, 16, 20, 25, 32, 40)


def bar_area(d_mm: float) -> float:
    """Area of one bar [mm²]."""
    return math.pi * d_mm ** 2 / 4.0


def fctm(fck_mpa: float) -> float:
    """Mean tensile strength fctm [MPa] (EC2 Table 3.1)."""
    if fck_mpa <= 50.0:
        return 0.30 * fck_mpa ** (2.0 / 3.0)
    return 2.12 * math.log(1.0 + (fck_mpa + 8.0) / 10.0)


@dataclass(frozen=True)
class BarGroup:
    """Bars of equal diameter in one layer."""
    n: int
    diameter: float  # mm

    @property
    def area(self) -> float:
        return self.n * bar_area(self.diameter)


@dataclass
class Layout:
    """Bars of one face of one span: one or more layers."""
    layers: list[BarGroup] = field(default_factory=list)

    @property
    def area(self) -> float:
        return sum(g.area for g in self.layers)

    @property
    def n_bars(self) -> int:
        return sum(g.n for g in self.layers)

    @property
    def diameters(self) -> set[float]:
        return {g.diameter for g in self.layers}

    def __str__(self) -> str:
        return " + ".join(f"{g.n}Ø{g.diameter:g}" for g in self.layers) or "-"


@dataclass
class Span:
    """Beam span with the required areas [mm²] per face."""
    length: float      # m
    as_bottom: float   # mm²
    as_top: float      # mm²
    name: str = ""


@dataclass
class Beam:
    width: float               # b [mm]
    height: float              # h [mm]
    cover: float               # cover to the stirrup face [mm]
    stirrup_diameter: float    # [mm]
    fck: float                 # [MPa]
    fyk: float                 # [MPa]
    spans: list[Span]
    dg: float = 20.0           # max aggregate size [mm]
    d_bar_est: float = 16.0    # bar diameter assumed to compute d [mm]

    @property
    def d_eff(self) -> float:
        return self.height - self.cover - self.stirrup_diameter - self.d_bar_est / 2.0

    @property
    def as_min(self) -> float:
        """EC2 §9.2.1.1(1): 0.26·fctm/fyk·b·d >= 0.0013·b·d  [mm²]."""
        bd = self.width * self.d_eff
        return max(0.26 * fctm(self.fck) / self.fyk * bd, 0.0013 * bd)

    @property
    def as_max(self) -> float:
        """EC2 §9.2.1.1(3): 0.04·Ac  [mm²]."""
        return 0.04 * self.width * self.height


@dataclass
class SpanReinforcement:
    span: Span
    bottom: Layout
    top: Layout
    as_bottom_design: float   # requirement after As,min [mm²]
    as_top_design: float


@dataclass
class BeamReinforcement:
    spans: list[SpanReinforcement]
    warnings: list[str] = field(default_factory=list)
    through: dict[str, float | None] = field(default_factory=dict)  # face -> Ø

    def diameters(self, face: str) -> set[float]:
        return {d for s in self.spans for d in getattr(s, face).diameters}


# ---------------------------------------------------------------------------
# Geometry
# ---------------------------------------------------------------------------

def min_clear_spacing(d: float, dg: float) -> float:
    """EC2 §8.2(2): max(φ, dg + 5, 20) [mm]."""
    return max(d, dg + 5.0, 20.0)


def max_bars_per_layer(beam: Beam, d: float) -> int:
    """Maximum number of bars of diameter *d* in one layer."""
    free = beam.width - 2.0 * (beam.cover + beam.stirrup_diameter)
    s = min_clear_spacing(d, beam.dg)
    return max(int((free + s) // (d + s)), 0)  # n·d + (n-1)·s <= free


# ---------------------------------------------------------------------------
# Single area -> bars
# ---------------------------------------------------------------------------

SYMMETRY_MODES = ("rule", "even", "none")


def _counts_ok(n: int, symmetry: str) -> bool:
    """Is *n* bars an admissible layer count?"""
    if symmetry == "even":
        return n % 2 == 0
    if symmetry == "rule":
        return n == 2 or n >= 3      # no single bar; odd counts need >= 3
    return n >= 1


def _pair_symmetric(na: int, nb: int) -> bool:
    """Can a layer of *na* bars become *nb* (or back) keeping the symmetry?

    The smaller set must be the larger one minus a symmetric removal: pairs
    (even difference, always fine) or the centre bar plus pairs (odd
    difference, which requires the larger set to be odd, i.e. to have a
    centre bar).
    """
    d = abs(na - nb)
    return d % 2 == 0 or max(na, nb) % 2 == 1


def layouts_compatible(a: Layout, b: Layout) -> bool:
    """Symmetry of the curtailment between two adjacent zones.

    Layers are compared by index; two layers of the same diameter must pass
    :func:`_pair_symmetric` (a missing layer counts as 0 bars). Layers of
    different diameters share no bars and are not constrained.
    """
    for k in range(max(len(a.layers), len(b.layers))):
        ga = a.layers[k] if k < len(a.layers) else None
        gb = b.layers[k] if k < len(b.layers) else None
        if ga and gb and ga.diameter != gb.diameter:
            continue
        na, nb = (ga.n if ga else 0), (gb.n if gb else 0)
        if not _pair_symmetric(na, nb):
            return False
    return True


def layout_candidates(
    as_req: float,
    beam: Beam,
    diameters=DIAMETERS_MM,
    min_bars: int = 2,
    max_layers: int = 2,
    max_diameters_mixed: int = 2,
    mix_penalty: float = 25.0,
    bar_penalty: float = 15.0,
    symmetry: str = "rule",
    through: tuple[int, float] | None = None,
    top_k: int | None = None,
) -> list[tuple[float, Layout]]:
    """Feasible bar combinations covering *as_req*, cheapest first.

    Each item is ``(cost, layout)`` with ``cost = excess area + mix_penalty·
    (n_diam-1) + bar_penalty·n_bars`` [mm²].  Each layer has a single
    diameter and never more bars than the previous one.  ``through=(n, d)``
    forces the outer layer to be of diameter *d* with at least *n* bars (the
    continuous bars).

    ``symmetry`` keeps the section symmetric about its vertical axis:

    * ``"rule"`` (default) — a layer has 2 bars or >= 3 (odd allowed, never a
      single bar); curtailment between adjacent zones must be symmetric, see
      :func:`layouts_compatible` (odd counts can drop an even or odd number,
      even counts drop only even numbers) — enforced by ``beam_reinforcement``;
    * ``"even"`` — only even counts per layer;
    * ``"none"`` — no constraint.
    """
    if symmetry not in SYMMETRY_MODES:
        raise ValueError(f"symmetry must be one of {SYMMETRY_MODES}")
    per_layer = [
        BarGroup(n, d)
        for d in diameters
        for n in range(1, max_bars_per_layer(beam, d) + 1)
        if _counts_ok(n, symmetry)
    ]
    out: list[tuple[float, int, Layout]] = []
    for k in range(1, max_layers + 1):
        for combo in product(per_layer, repeat=k):
            if any(combo[i].n < combo[i + 1].n for i in range(k - 1)):
                continue
            if through and (combo[0].diameter != through[1]
                            or combo[0].n < through[0]):
                continue
            n_tot = sum(g.n for g in combo)
            area = sum(g.area for g in combo)
            if n_tot < min_bars or area < as_req:
                continue
            n_diam = len({g.diameter for g in combo})
            if n_diam > max_diameters_mixed:
                continue
            cost = (area - as_req + mix_penalty * (n_diam - 1)
                    + bar_penalty * n_tot)
            out.append((cost, n_tot, Layout(list(combo))))
    out.sort(key=lambda t: (t[0], t[1]))
    if top_k is not None:
        out = out[:top_k]
    return [(c, lay) for c, _, lay in out]


def bars_for_area(as_req: float, beam: Beam, diameters=DIAMETERS_MM,
                  **kwargs) -> Layout:
    """Cheapest bar combination covering *as_req* (see ``layout_candidates``).

    Raises ``ValueError`` if no combination fits.
    """
    cands = layout_candidates(as_req, beam, diameters, top_k=1, **kwargs)
    if not cands:
        raise ValueError(
            f"Cannot cover As={as_req:.0f} mm² with b={beam.width:g} mm and "
            f"diameters {tuple(diameters)}; enlarge the section, allow more "
            "layers or larger diameters."
        )
    return cands[0][1]


# ---------------------------------------------------------------------------
# Beam: As,min + uniformisation between spans
# ---------------------------------------------------------------------------

def _chain_min_cost(cands: list, lengths: list[float], check: bool):
    """Viterbi over the zones: pick one candidate per zone minimising
    ``Σ L·cost`` with adjacent layouts symmetric-compatible. Returns
    ``(total cost, layouts)`` or ``None`` when no chain exists."""
    best = [lengths[0] * c for c, _ in cands[0]]
    back: list[list[int]] = []
    for z in range(1, len(cands)):
        new, bp = [], []
        for c, lay in cands[z]:
            opt = (math.inf, -1)
            for j, (_, prev) in enumerate(cands[z - 1]):
                if best[j] < opt[0] and (not check
                                         or layouts_compatible(prev, lay)):
                    opt = (best[j], j)
            new.append(opt[0] + lengths[z] * c)
            bp.append(opt[1])
        best, back = new, back + [bp]
    k = min(range(len(best)), key=best.__getitem__)
    if not math.isfinite(best[k]):
        return None
    total, idx = best[k], [k]
    for bp in reversed(back):
        idx.append(bp[idx[-1]])
    idx.reverse()
    return total, [cands[z][i][1] for z, i in enumerate(idx)]


def _face_layouts(
    reqs: list[float],
    lengths: list[float],
    beam: Beam,
    diameters,
    max_diameters_beam: int,
    diam_penalty: float,
    n_through: int = 2,
    bar_penalty: float = 15.0,
    symmetry: str = "rule",
    **kwargs,
) -> tuple[list[Layout], float | None]:
    """Layouts of one face for all spans/zones, with at most
    *max_diameters_beam* distinct diameters along the beam.

    Every zone carries at least *n_through* continuous bars of one common
    diameter (outer layer); the through diameter is chosen with the set.
    For every diameter subset the zones are solved jointly (dynamic
    programming) so that the curtailment between adjacent zones keeps the
    symmetry (``symmetry="rule"``); the subset minimising
    ``Σ L_i·cost_i + diam_penalty·ΣL_i·(n_diam-1)`` wins.
    """
    best_cost = math.inf
    best: list[Layout] | None = None
    best_through: float | None = None
    err: ValueError | None = None
    check = symmetry == "rule"

    for k in range(1, max(1, max_diameters_beam) + 1):
        for subset in combinations(diameters, k):
            for dt in (subset if n_through > 0 else (None,)):
                thr = (n_through, dt) if dt is not None else None
                sol = None
                for top_k in (60, 600, None):      # widen if no chain exists
                    cands = [layout_candidates(
                        r, beam, subset, through=thr, bar_penalty=bar_penalty,
                        symmetry=symmetry, top_k=top_k, **kwargs)
                        for r in reqs]
                    if any(not c for c in cands):
                        err = ValueError(
                            f"Cannot cover the required steel with "
                            f"b={beam.width:g} mm and diameters {subset}; "
                            "enlarge the section, allow more layers or "
                            "larger diameters.")
                        break
                    sol = _chain_min_cost(cands, lengths, check)
                    if sol is not None:
                        break
                if sol is None:
                    continue
                cost, layouts = sol
                used = set().union(*(lay.diameters for lay in layouts))
                cost += diam_penalty * sum(lengths) * (len(used) - 1)
                if cost < best_cost - 1e-9:
                    best_cost, best, best_through = cost, layouts, dt

    if best is None:
        raise err or ValueError(
            "No layout keeps the bar curtailment symmetric; relax "
            "symmetry or allow more diameters/layers.")
    return best, best_through


def beam_reinforcement(
    beam: Beam,
    diameters=DIAMETERS_MM,
    max_diameters_beam: int = 2,
    diam_penalty: float = 50.0,
    top_min_ratio: float = 0.0,
    n_through: int = 2,
    bar_penalty: float = 15.0,
    **kwargs,
) -> BeamReinforcement:
    """Convert the required areas of every span into bars.

    Parameters
    ----------
    max_diameters_beam : max distinct diameters per face along the beam
        (1 = same diameter in every span).  Uniformisation is done per face.
    diam_penalty : cost [mm² per m of beam] of each extra diameter.
    top_min_ratio : fraction of As,min imposed on the top face
        (0 = only the constructive minimum ``min_bars`` hanger bars;
        1 = full As,min, e.g. for hogging zones).
    n_through : continuous bars per face, of one common diameter, present in
        every span (0 disables).  Lap/anchorage lengths are not computed.
    bar_penalty : cost [mm² per bar] that favours fewer, larger bars over an
        exact-area match made of many thin bars.
    symmetry : (via kwargs) ``"rule"`` (default: odd counts >= 3 allowed,
        curtailment between adjacent zones symmetric), ``"even"`` or ``"none"``.
    kwargs : forwarded to ``bars_for_area`` (min_bars, max_layers, ...).
    """
    warnings: list[str] = []
    as_min = beam.as_min
    lengths = [max(s.length, 1e-6) for s in beam.spans]

    # As,min per face
    req_bot, req_top = [], []
    for s in beam.spans:
        rb = max(s.as_bottom, as_min)
        rt = max(s.as_top, top_min_ratio * as_min)
        if s.as_bottom < as_min:
            warnings.append(
                f"{s.name or 'span'}: bottom As={s.as_bottom:.0f} raised to "
                f"As,min={as_min:.0f} mm²")
        req_bot.append(rb)
        req_top.append(rt)

    args = dict(diameters=diameters, max_diameters_beam=max_diameters_beam,
                diam_penalty=diam_penalty, beam=beam, lengths=lengths,
                n_through=n_through, bar_penalty=bar_penalty, **kwargs)
    bottom, thr_bot = _face_layouts(req_bot, **args)
    top, thr_top = _face_layouts(req_top, **args)

    out = []
    for s, lb, lt, rb, rt in zip(beam.spans, bottom, top, req_bot, req_top):
        for face, lay in (("bottom", lb), ("top", lt)):
            if lay.area > beam.as_max:
                warnings.append(
                    f"{s.name or 'span'}: {face} As={lay.area:.0f} exceeds "
                    f"As,max=0.04·Ac={beam.as_max:.0f} mm²")
        out.append(SpanReinforcement(s, lb, lt, rb, rt))
    return BeamReinforcement(out, warnings, {"bottom": thr_bot, "top": thr_top})
