"""Parameters of the beam-bars design (areas -> bars) and their resolution.

``resolve_beam_params`` merges, from the most specific to the most general:

    beam override  >  section override  >  model preferences  >  built-in

``None`` always means "inherit" (same convention as ``is_column`` /
``sd_ky``). See dev/BEAM_BARS_DEFINITION.md §6.

Sources
-------
* *prefs*   — the model's design preferences (flat dict, ``design.prefs`` of the
  ``.x2d``) with the keys of :data:`PREF_KEYS`; the aggregate size reuses the
  existing ``dmax`` key.
* *section* — any object; its ``beam_overrides`` dict (``Section.beam_overrides``,
  keys limited to :data:`SECTION_PARAMS`) is read through ``getattr`` (absent
  or ``None`` = inherit). The main-bar diameter used to estimate the
  effective depth comes from the existing ``Section.rc_bar_phi``.
* *beam*    — a dict of overrides keyed by the parameter names below (the
  ``overrides`` of a beam entry); unknown keys are rejected so a typo does not
  silently do nothing.
"""
from __future__ import annotations

from dataclasses import dataclass, fields, replace

ZONE_MODES = ("cutoff", "fixed")
SYMMETRY_MODES = ("rule", "even", "none")


@dataclass(frozen=True)
class BeamBarParams:
    # --- exposed (preferences / section / beam) -------------------------
    diameters: tuple = (10, 12, 16, 20, 25, 32)   # available bar Ø [mm]
    stirrup_diameter_mm: float = 8.0
    max_layers: int = 2
    max_diameters_beam: int = 2          # distinct Ø per face along the beam
    n_through: int = 2                   # continuous bars per face (0 = off)
    symmetry: str = "rule"               # "rule" | "even" | "none"
    zone_mode: str = "cutoff"            # "cutoff" | "fixed"
    zones: tuple = (0.25, 0.5, 0.25)     # fractions, zone_mode == "fixed"
    cutoff_levels: int = 3
    min_zone_length: float = 0.5         # [m]
    support_bottom_ratio: float = 0.25   # EC2 §9.2.1.4(1)
    shift_d: float = 0.0                 # EC2 §9.2.1.3 shift, in units of d
    top_min_ratio: float = 0.0           # share of As,min imposed on top
    # stirrups (shear reinforcement); ``stirrup_diameter_mm`` above stays the
    # diameter ASSUMED for the cover/width of the longitudinal layers
    stirrup_legs: int = 2
    stirrup_diameters: tuple = (6, 8, 10, 12)               # [mm]
    stirrup_spacings: tuple = (75, 100, 125, 150, 175, 200, 250, 300)  # [mm]
    # anchorage and laps (EC2 §8.4, §8.7 — computed by eurocodepy)
    anchorage_shape: str = "straight"    # end anchorage: "straight" | "bent"
    lap_percentage: float = 50.0         # ρ1: % of bars lapped in a section
    bar_length: float = 12.0             # commercial bar length [m]
    # --- derived / not user parameters ----------------------------------
    dg: float = 20.0                     # aggregate size [mm] (prefs ``dmax``)
    d_bar_est: float = 16.0              # Ø assumed for d [mm] (rc_bar_phi)
    # --- internal tuning (not exposed) ----------------------------------
    min_bars: int = 2
    max_diameters_mixed: int = 2
    bar_penalty: float = 15.0
    diam_penalty: float = 50.0
    mix_penalty: float = 25.0

    def validate(self) -> "BeamBarParams":
        if not self.diameters or any(d <= 0 for d in self.diameters):
            raise ValueError("diameters must be a non-empty list of Ø > 0")
        if self.stirrup_diameter_mm <= 0:
            raise ValueError("stirrup_diameter_mm must be > 0")
        if self.max_layers < 1 or self.max_diameters_beam < 1:
            raise ValueError("max_layers and max_diameters_beam must be >= 1")
        if self.n_through < 0:
            raise ValueError("n_through must be >= 0")
        if self.symmetry not in SYMMETRY_MODES:
            raise ValueError(f"symmetry must be one of {SYMMETRY_MODES}")
        if self.zone_mode not in ZONE_MODES:
            raise ValueError(f"zone_mode must be one of {ZONE_MODES}")
        if not self.zones or any(z <= 0 for z in self.zones):
            raise ValueError("zones must be positive fractions")
        if self.stirrup_legs < 2:
            raise ValueError("stirrup_legs must be >= 2")
        for name in ("stirrup_diameters", "stirrup_spacings"):
            v = getattr(self, name)
            if not v or any(x <= 0 for x in v):
                raise ValueError(f"{name} must be a non-empty list of values > 0")
        if self.anchorage_shape not in ("straight", "bent"):
            raise ValueError("anchorage_shape must be 'straight' or 'bent'")
        if not 0.0 <= self.lap_percentage <= 100.0:
            raise ValueError("lap_percentage must be in [0, 100]")
        if self.bar_length <= 0:
            raise ValueError("bar_length must be > 0")
        if self.cutoff_levels < 1:
            raise ValueError("cutoff_levels must be >= 1")
        if self.min_zone_length < 0 or self.shift_d < 0:
            raise ValueError("min_zone_length and shift_d must be >= 0")
        if not 0.0 <= self.support_bottom_ratio <= 1.0:
            raise ValueError("support_bottom_ratio must be in [0, 1]")
        if not 0.0 <= self.top_min_ratio <= 1.0:
            raise ValueError("top_min_ratio must be in [0, 1]")
        return self

    # keyword arguments understood by ``beam_reinforcement`` (and below)
    def reinforcement_kwargs(self) -> dict:
        return dict(
            diameters=tuple(self.diameters),
            max_diameters_beam=self.max_diameters_beam,
            n_through=self.n_through, symmetry=self.symmetry,
            max_layers=self.max_layers, top_min_ratio=self.top_min_ratio,
            min_bars=self.min_bars,
            max_diameters_mixed=self.max_diameters_mixed,
            bar_penalty=self.bar_penalty, diam_penalty=self.diam_penalty,
            mix_penalty=self.mix_penalty)


_PARAM_NAMES = {f.name for f in fields(BeamBarParams)}

# ----------------------------------------------------------------------
# The registry of the parameters a user can change
# ----------------------------------------------------------------------
#
# ONE list drives every place that shows them: the "EC2 — beam bars" group of
# the Code preferences, the Parameters tab of the detailing window and the
# validation of overrides. Adding a parameter is one line here (plus the field
# of BeamBarParams).

@dataclass(frozen=True)
class ParamSpec:
    """How a user-editable parameter is presented.

    ``kind`` / ``extra``: ``int`` → ``(min, max, step)``; ``float`` →
    ``(min, max, step, decimals)``; ``list`` → ``(lo, hi)`` (each value of the
    comma-separated list in ``[lo, hi]``); ``choice`` → ``[(label, value), …]``.
    """
    name: str                 # BeamBarParams field
    pref_key: str             # key in the model's design preferences
    label: str                # full label (Code preferences)
    short: str                # short label (compact forms)
    units: str
    kind: str
    extra: tuple
    section_ok: bool = False  # may a Section override it?

    def default(self):
        v = getattr(BeamBarParams(), self.name)
        return list(v) if isinstance(v, tuple) else v


_BB = BeamBarParams()
PARAM_SPECS: tuple = (
    ParamSpec("diameters", "beam_diameters", "Available bar diameters",
              "Diameters [mm]", "mm", "list", (4.0, 50.0), True),
    ParamSpec("stirrup_diameter_mm", "beam_stirrup_mm",
              "Stirrup diameter (assumed for the cover)",
              "Cover stirrup Ø [mm]", "mm", "int", (4, 25, 1), True),
    ParamSpec("max_layers", "beam_max_layers", "Max. layers per face",
              "Max. layers", "—", "int", (1, 4, 1), True),
    ParamSpec("max_diameters_beam", "beam_max_diameters",
              "Max. distinct diameters per face (beam)", "Max. Ø / face",
              "—", "int", (1, 4, 1)),
    ParamSpec("n_through", "beam_n_through",
              "Continuous (through) bars per face", "Through bars", "—",
              "int", (0, 8, 1)),
    ParamSpec("symmetry", "beam_symmetry", "Bar symmetry", "Symmetry", "—",
              "choice", [("Odd ≥ 3 allowed, symmetric curtailment", "rule"),
                         ("Even counts only", "even"),
                         ("No constraint", "none")]),
    ParamSpec("zone_mode", "beam_zone_mode", "Zones along each span", "Zones",
              "—", "choice", [("Envelope cutoff points", "cutoff"),
                              ("Fixed fractions of the span", "fixed")]),
    ParamSpec("cutoff_levels", "beam_cutoff_levels",
              "Envelope steps (cutoff mode)", "Envelope steps", "—", "int",
              (1, 10, 1)),
    ParamSpec("min_zone_length", "beam_min_zone_length",
              "Minimum zone length", "Min. zone [m]", "m", "float",
              (0.0, 5.0, 0.1, 2)),
    ParamSpec("support_bottom_ratio", "beam_support_bottom_ratio",
              "Bottom steel reaching the supports (EC2 9.2.1.4)",
              "Bottom at supp.", "× max. span", "float", (0.0, 1.0, 0.05, 2)),
    ParamSpec("shift_d", "beam_shift_d", "Envelope shift a_l (EC2 9.2.1.3)",
              "Shift a_l [× d]", "× d", "float", (0.0, 2.0, 0.05, 2)),
    ParamSpec("top_min_ratio", "beam_top_min_ratio",
              "As,min imposed on the top face", "As,min on top", "× As,min",
              "float", (0.0, 1.0, 0.05, 2)),
    ParamSpec("stirrup_legs", "beam_stirrup_legs", "Stirrup legs",
              "Stirrup legs", "—", "int", (2, 8, 1)),
    ParamSpec("stirrup_diameters", "beam_stirrup_diameters",
              "Available stirrup diameters", "Stirrup Ø [mm]", "mm", "list",
              (4.0, 25.0)),
    ParamSpec("stirrup_spacings", "beam_stirrup_spacings",
              "Available stirrup spacings", "Stirrup s [mm]", "mm", "list",
              (25.0, 600.0)),
    ParamSpec("anchorage_shape", "beam_anchorage_shape",
              "Bar ends (anchorage)", "Bar ends", "—", "choice",
              [("Straight", "straight"), ("Hooks / bends", "bent")]),
    ParamSpec("lap_percentage", "beam_lap_percentage",
              "Lapped bars in a section (ρ1)", "Lapped bars ρ1 [%]", "%",
              "float", (0.0, 100.0, 5.0, 0)),
    ParamSpec("bar_length", "beam_bar_length", "Commercial bar length",
              "Bar length [m]", "m", "float", (1.0, 30.0, 0.5, 1)),
)
SPEC_BY_NAME = {sp.name: sp for sp in PARAM_SPECS}

# model-preference key -> parameter name (the specs, plus the aggregate size
# that is the existing punching preference ``dmax``)
PREF_KEYS = {sp.pref_key: sp.name for sp in PARAM_SPECS}
PREF_KEYS["dmax"] = "dg"

# parameters a Section may override (those that depend on the beam *type*)
SECTION_PARAMS = tuple(sp.name for sp in PARAM_SPECS if sp.section_ok)

# parameters a beam may override: everything exposed, plus the fixed zone
# fractions (file/script only — a list of fractions is no use in a form)
BEAM_PARAMS = tuple(sp.name for sp in PARAM_SPECS) + ("zones",)


def parse_number_list(text: str, lo: float, hi: float) -> list:
    """``"10, 12 ; 16"`` → ``[10, 12, 16]`` (ints when integral, sorted,
    unique). Raises ``ValueError`` for empty / non-numeric / out-of-range."""
    import re
    parts = [t for t in re.split(r"[,;\s]+", str(text).strip()) if t]
    vals = sorted({float(t.replace(",", ".")) for t in parts})
    if not vals:
        raise ValueError("empty list")
    if vals[0] < lo or vals[-1] > hi:
        raise ValueError(f"values must be in [{lo:g}, {hi:g}]")
    return [int(v) if float(v).is_integer() else v for v in vals]


_INT = {"max_layers", "max_diameters_beam", "n_through", "cutoff_levels",
        "min_bars", "max_diameters_mixed", "stirrup_legs"}
_TUPLE = {"diameters", "zones", "stirrup_diameters", "stirrup_spacings"}


def _coerce(name: str, value):
    if name in _TUPLE:
        return tuple(float(v) if name == "zones" else v for v in value)
    if name in _INT:
        return int(value)
    if name in ("symmetry", "zone_mode", "anchorage_shape"):
        return str(value)
    return float(value)


def resolve_beam_params(prefs: dict | None = None, section=None,
                        beam: dict | None = None) -> BeamBarParams:
    """Effective parameters: beam > section > preferences > built-in."""
    values: dict = {}

    for key, name in PREF_KEYS.items():                     # preferences
        v = (prefs or {}).get(key)
        if v is not None and not (name == "dg" and not v):
            values[name] = v

    if section is not None:                                 # section
        sec_ov = getattr(section, "beam_overrides", None) or {}
        bad = set(sec_ov) - set(SECTION_PARAMS)
        if bad:
            raise ValueError(
                f"section override(s) not allowed: {sorted(bad)} "
                f"(allowed: {list(SECTION_PARAMS)})")
        values.update({k: v for k, v in sec_ov.items() if v is not None})
        phi = getattr(section, "rc_bar_phi", None)
        if phi:
            values["d_bar_est"] = phi

    if beam:                                                # beam overrides
        unknown = set(beam) - set(BEAM_PARAMS)
        if unknown:
            raise ValueError(f"unknown beam parameter(s): {sorted(unknown)}")
        values.update({k: v for k, v in beam.items() if v is not None})

    params = replace(BeamBarParams(),
                     **{k: _coerce(k, v) for k, v in values.items()})
    return params.validate()
