"""Helpers for tests that need a ``design_beam_bars`` with non-default
parameters. ``design_beam_bars`` takes its parameters only through ``prefs`` or
one ``params`` object (no loose keyword arguments), so a test that wants
``zones=(0.25, 0.5, 0.25)`` or ``n_through=4`` builds a ``BeamBarParams``."""
from dataclasses import replace

from xdfem2d.beam_bars_params import PARAM_SPECS, BeamBarParams

_PREF_OF = {sp.name: sp.pref_key for sp in PARAM_SPECS}


def bbp(**kw) -> BeamBarParams:
    """``BeamBarParams`` with *kw* changed; ``zones=(…)`` means fixed zoning
    with those fractions."""
    kw = dict(kw)
    zones = kw.pop("zones", None)
    if zones is not None:
        kw.update(zone_mode="fixed", zones=tuple(zones))
    return replace(BeamBarParams(), **kw)


def loose(kw: dict) -> dict:
    """``{"params": …}`` for ``design_beam_bars(..., **loose(kw))`` (empty when
    *kw* is)."""
    return {"params": bbp(**kw)} if kw else {}


def as_prefs(kw: dict) -> dict:
    """The preference keys that carry *kw* (registry parameters only)."""
    return {_PREF_OF[k]: (list(v) if isinstance(v, tuple) else v)
            for k, v in kw.items()}
