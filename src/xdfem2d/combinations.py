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


# ── Combinations of a model's own cases ───────────────────────────────────
#
# What the GUI's Auto Combinations dialog does, without the dialog: which
# analysis cases are one Eurocode action, with which factors, and the
# combinations of them. Rows are ANALYSIS cases, not load cases:
# ``add_load_combination`` only accepts analysis-case or combination ids.

#: Prefixes of the ids ``_generate_ec_combos`` gives its combinations: the ones a
#: regeneration replaces, and nothing the user named himself.
AUTO_PREFIXES = ("ULS:", "ULS-E:", "SLS-K:", "SLS-FR:", "SLS-QP:")


class PureCase:
    """Minimal (id, action_type) stand-in for an analysis case that is
    unambiguously a single Eurocode action type. ``_generate_ec_combos`` only
    ever reads ``.id`` and ``.action_type`` of the cases it is given."""
    __slots__ = ("id", "action_type")

    def __init__(self, id, action_type):
        self.id = id
        self.action_type = action_type


def pure_cases(struc) -> list:
    """``[(PureCase, (gamma_fav, gamma_unf, psi0, psi1, psi2)), ...]`` for the analysis
    cases of *struc* that are unambiguously one action type: a Spectrum case
    (always seismic) or a Linear case whose load cases all share one type.
    A Linear case that mixes types, names an unknown load case or has no
    coefficients, and any other kind of case (NonLinear, Mass, Modal ...), is left
    out: not an error, just not a candidate.

    The factors are those of the load cases (category and ψ overrides); load
    cases of one analysis case that disagree are combined as one action, with the
    onerous ψ of the two, per component.
    """
    from .models import ActionType
    out = []
    for ac in struc.analysis_cases:
        factors = None
        if ac.analysis_type == 'Spectrum':
            inferred = ActionType.E
        elif ac.analysis_type == 'Linear':
            coeffs = ac.coefficients
            if not coeffs:
                continue
            action_types, case_factors = set(), []
            unresolved = False
            for lc_id in coeffs:
                lc = struc.load_cases_by_id.get(lc_id)
                if lc is None:
                    unresolved = True
                    break
                action_types.add(ActionType.coerce(lc.action_type))
                case_factors.append(load_case_factors(lc))
            if unresolved or len(action_types) != 1:
                continue
            inferred = next(iter(action_types))
            factors = tuple(max(col) if i >= 2 else col[0]
                            for i, col in enumerate(zip(*case_factors)))
        else:
            continue
        # No fallback tuple: a silent default for a permanent action would be a
        # wrong partial factor presented as a default.
        out.append((PureCase(id=ac.id, action_type=inferred),
                    factors if factors is not None else default_factors(inferred)))
    return out


def auto_combinations(struc, types=("ULS",), replace: bool = True) -> dict:
    """Add the EN 1990 combinations of the cases of *struc* to it.

    types : any of ``'ULS'``, ``'SLS-K'``, ``'SLS-FR'``, ``'SLS-QP'``.
    replace : first remove the combinations an earlier run generated (the ids that
        start with :data:`AUTO_PREFIXES`); the ones the user named are never touched.

    Returns ``{'added': [ids], 'skipped': [ids already there], 'removed': n,
    'errors': ['id: why'], 'cases': [ids of the analysis cases combined]}``.
    """
    cases = pure_cases(struc)
    factors = {pc.id: f for pc, f in cases}
    combos = _generate_ec_combos([pc for pc, _ in cases], factors, set(types))
    out = {'added': [], 'skipped': [], 'removed': 0, 'errors': [],
           'cases': [pc.id for pc, _ in cases]}
    if replace:
        kept = [lc for lc in struc.load_combinations
                if not any(lc.id.startswith(p) for p in AUTO_PREFIXES)]
        out['removed'] = len(struc.load_combinations) - len(kept)
        struc.load_combinations = kept
    existing = {lc.id for lc in struc.load_combinations}
    for name, coeffs, _label in combos:
        if name in existing:
            out['skipped'].append(name)
            continue
        try:
            struc.add_load_combination(name, coeffs)
            existing.add(name)
            out['added'].append(name)
        except Exception as e:                           # noqa: BLE001 -- reported, not hidden
            out['errors'].append(f"{name}: {e}")
    return out
