"""
EN 1990 load-combination generation for xdfem2D.

Thin adapter over ``eurocodepy.ec1.combos`` — eurocodepy is a required
dependency and owns the EN 1990 combination logic, so xdfem2D delegates to it
rather than carrying a parallel implementation. GUI-free and testable.
"""
from __future__ import annotations

_COMBO_TOLERANCE = 0.001


def _generate_ec_combos(load_cases, factors_map, combo_types):
    """Generate EN 1990 load combinations via eurocodepy.

    Parameters
    ----------
    load_cases : list[LoadCase]
    factors_map : dict[lc_id -> (gamma_fav, gamma_unf, psi0, psi1, psi2)]
    combo_types : set of strings from {'ULS','SLS-K','SLS-FR','SLS-QP'}

    Returns
    -------
    list of (combo_name: str, coefficients: dict[lc_id -> float], combo_type_label: str)
    """
    from eurocodepy.ec1.combos import (
        Loads as EcLoads, Load as EcLoad, LoadType as EcLoadType,
    )

    from .models import ActionType

    # Keyed by the enum member, not by the string it happens to carry.
    #
    # It was keyed by the string — "G – Permanent", with an en dash — and when
    # that value became "G" this map stopped matching anything. Every load
    # case silently became OTHER, so a permanent action was combined as a
    # variable one and G, Q, W produced three ULS combinations instead of two.
    # Nothing failed: the wrong combinations were generated, and generated
    # confidently. The tests that would have caught it had been skipped
    # wherever eurocodepy was not installed, which is where the change was made;
    # eurocodepy is now a required dependency and they always run.
    #
    # A member cannot drift from itself. ActionType.coerce below accepts the
    # old spellings, so a file written before the change still lands here.
    _AT_TO_EC = {
        ActionType.G: EcLoadType.PERMANENT,
        ActionType.Q: EcLoadType.LIVE,
        ActionType.W: EcLoadType.WIND,
        ActionType.E: EcLoadType.EARTHQUAKE,
        ActionType.T: EcLoadType.TEMPERATURE,
        # New action types (S, C, A, F): looked up defensively, since an
        # older/different eurocodepy version may not define these LoadType
        # members yet. Falls back to OTHER exactly like an unmapped
        # ActionType falls back to OTHER below, rather than raising
        # AttributeError.
        ActionType.S: getattr(EcLoadType, 'SNOW', EcLoadType.OTHER),
        ActionType.C: getattr(EcLoadType, 'CONSTRUCTION', EcLoadType.OTHER),
        ActionType.A: getattr(EcLoadType, 'ACCIDENTAL', EcLoadType.OTHER),
        ActionType.F: getattr(EcLoadType, 'FIRE', EcLoadType.OTHER),
        # "Other" is deliberately unclassified -- it maps directly to OTHER,
        # not looked up defensively like the members above (there is no
        # more-specific EcLoadType it could ever resolve to).
        ActionType.O: EcLoadType.OTHER,
    }

    ec_loads = EcLoads()
    for lc in load_cases:
        ec_type = _AT_TO_EC.get(ActionType.coerce(lc.action_type),
                                EcLoadType.OTHER)
        gf, gu, p0, p1, p2 = factors_map[lc.id]
        ec_loads.add(EcLoad(lc.id, ec_type, gf, gu, p0, p1, p2))

    combos = []
    seen = set()

    def _extract(ec_combo_dict, label):
        for name, combo in ec_combo_dict.items():
            coeffs = {k: round(v, 6) for k, (_, v) in combo.factors.items()
                      if abs(v) > _COMBO_TOLERANCE}
            if name not in seen and coeffs:
                seen.add(name)
                combos.append((name, coeffs, label))

    if 'ULS' in combo_types:
        _extract(ec_loads.get_ULS_combos(), 'ULS')
    if any(s in combo_types for s in ('SLS-K', 'SLS-FR', 'SLS-QP')):
        for name, combo in ec_loads.get_SLS_combos().items():
            label = str(combo.type)          # 'SLS-K', 'SLS-FR', 'SLS-QP'
            if label not in combo_types:
                continue
            coeffs = {k: round(v, 6) for k, (_, v) in combo.factors.items()
                      if abs(v) > _COMBO_TOLERANCE}
            if name not in seen and coeffs:
                seen.add(name)
                combos.append((name, coeffs, label))
    return combos


# ── Default EN 1990 factors per load case ─────────────────────────────────
#
# The numbers are eurocodepy's (eurocodepy.ec1.action_factors); this module
# only maps xdfem2D's ActionType onto its kinds and applies a load case's own
# category and ψ overrides.

# Which national table the defaults are read from: "EU" (EN recommended
# values) or "PT" (Portuguese National Annex; only the live-load q_k/Q_k
# differ in eurocodepy today).
DEFAULT_LOCALE = "PT"

_KIND_OF_ACTION = {
    'G': 'permanent', 'Q': 'live', 'W': 'wind', 'E': 'earthquake',
    'T': 'temperature', 'S': 'snow', 'C': 'construction',
    'A': 'accidental', 'F': 'fire', 'O': 'other',
}


def default_factors(action_type, category="", locale=None):
    """(gamma_fav, gamma_unf, psi0, psi1, psi2) of an action, from eurocodepy.

    ``category`` is the live-load category or the snow variant; an
    unrecognised one is not an error here — the generic row of the action
    type is returned, which is what an empty category means anyway.
    """
    from eurocodepy.ec1 import action_factors
    from .models import ActionType

    locale = locale or DEFAULT_LOCALE

    kind = _KIND_OF_ACTION[ActionType.coerce(action_type).value]
    try:
        return tuple(action_factors(kind, category or None, locale))
    except KeyError:
        return tuple(action_factors(kind, None, locale))


def load_case_factors(lc, locale=None):
    """Factors a load case combines with: its category's defaults, each ψ
    replaced by the case's own value where it has set one."""
    base = list(default_factors(lc.action_type, getattr(lc, 'category', ''),
                                locale or DEFAULT_LOCALE))
    for i, name in enumerate(('psi0', 'psi1', 'psi2'), start=2):
        v = getattr(lc, name, None)
        if v is not None:
            base[i] = float(v)
    return tuple(base)
