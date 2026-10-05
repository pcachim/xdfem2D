"""
Variants — parallel structural scenarios over a shared entity space (Phase 1).

A *variant* is an overlay on a base :class:`~xdfem2d.structure.Structure2D` that
keeps the same id namespace but may change the supports (and, for what-if only,
the active geometry). Each variant is solved independently with the normal
``calculate()`` pipeline; this module then combines the per-variant results with
the same arithmetic used by the solver's load combinations:

  LinearSum  — Σ factor_i · value_i            (exact only if same stiffness)
  Envelope  — component-wise max/min envelope
  AbsSum  — Σ |factor_i · value_i|
  SRSS    — √(Σ (factor_i · value_i)²)

Results keep the shape produced by ``compute_results``: every quantity is indexed
by string id (node_id / elem_id), so combining across variants is a field-by-field
operation over shared ids — no index translation is ever needed.

The solver is **not** modified: this module reads the result dicts it produces.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Containers
# ---------------------------------------------------------------------------

@dataclass
class VariantResult:
    """Solved result of one variant plus validity-guard signatures.

    stiffness_hash : full effective-K hash; equal ⇒ linear sum is exact.
    geometry_sig   : signature of the active geometry only (nodes + elements);
        equal ⇒ same id space, so envelopes/comparisons align without being a
        mere what-if. Differs from stiffness_hash, which also changes when only
        the supports change.
    """
    variant_id: str
    stiffness_hash: str
    results: dict                       # same shape as compute_results()
    geometry_sig: str = ""


@dataclass(frozen=True)
class CombTerm:
    """One input to a cross-variant combination: variant + case + factor."""
    variant_id: str
    case_id: str
    factor: float = 1.0


_OPS = ('LinearSum', 'Envelope', 'AbsSum', 'SRSS')


# ---------------------------------------------------------------------------
# Solve variants
# ---------------------------------------------------------------------------

def solve_variants(base, variants: list, support_sets: Optional[dict] = None
                   ) -> dict:
    """Solve each variant independently. Returns ``{variant_id: VariantResult}``."""
    out: dict = {}
    for v in variants:
        sub = base.derive_variant(v, support_sets)
        res = sub.calculate()
        out[v.id] = VariantResult(variant_id=v.id,
                                  stiffness_hash=sub.stiffness_hash(),
                                  results=res,
                                  geometry_sig=_geometry_sig(sub))
    return out


def _geometry_sig(struc) -> str:
    """Signature of the active geometry only (node ids + element connectivity)."""
    import hashlib
    parts = [f"N|{nid}" for nid in sorted(struc.nodes)]
    parts += [f"E|{e.id}|{e.node_i}|{e.node_j}"
              for e in sorted(struc.bar_elements, key=lambda e: e.id)]
    return hashlib.sha256("\n".join(parts).encode('utf-8')).hexdigest()


# ---------------------------------------------------------------------------
# Pure result arithmetic (reusable; mirrors the solver's combination rules)
# ---------------------------------------------------------------------------

def _case_block(results: dict, case_id: str) -> Optional[dict]:
    """Return a flat ``{quantity: {id: [..]}}`` view for *case_id* inside a
    single variant's results dict (load case, analysis case or flat combination).
    """
    if case_id in results.get('displacements', {}):
        return {
            'displacements':  results['displacements'].get(case_id, {}),
            'reactions':      results['reactions'].get(case_id, {}),
            'element_forces': results['element_forces'].get(case_id, {}),
            'spring_forces':  results['spring_forces'].get(case_id, {}),
            'tri_stress':     results.get('tri_stress', {}).get(case_id, {}),
        }
    ac = results.get('analysis_cases', {}).get(case_id)
    if ac and 'displacements' in ac and 'max' not in ac['displacements']:
        return {
            'displacements':  ac.get('displacements', {}),
            'reactions':      ac.get('reactions', {}),
            'element_forces': ac.get('element_forces', {}),
            'spring_forces':  ac.get('spring_forces', {}),
            'tri_stress':     ac.get('tri_stress', {}),
        }
    combo = results.get('combinations', {}).get(case_id)
    if combo and 'displacements' in combo and 'max' not in combo['displacements']:
        return {
            'displacements':  combo.get('displacements', {}),
            'reactions':      combo.get('reactions', {}),
            'element_forces': combo.get('element_forces', {}),
            'spring_forces':  combo.get('spring_forces', {}),
            'tri_stress':     combo.get('tri_stress', {}),
        }
    return None


def _iter_ids(blocks: list, quantity: str, missing: str):
    """Yield the ids to combine for *quantity* across input blocks."""
    sets = [set(b[quantity].keys()) for b in blocks]
    if not sets:
        return set()
    if missing == 'skip':
        ids = set.intersection(*sets) if sets else set()
    else:  # 'zero' / 'error' use the union
        ids = set.union(*sets)
    return ids


def _get_vec(block: dict, quantity: str, ident: str, end=None):
    m = block[quantity].get(ident)
    if m is None:
        return None
    if end is not None:
        m = m.get(end)
        if m is None:
            return None
    return list(m)


# ---------------------------------------------------------------------------
# Combine across variants
# ---------------------------------------------------------------------------

def combine_across_variants(vres: dict, terms: list, op: str,
                            missing: str = 'zero',
                            require_same_stiffness: Optional[bool] = None,
                            allow_geometry_mismatch: bool = False) -> dict:
    """Combine results of several variants into a single result dict.

    vres   : {variant_id: VariantResult} from :func:`solve_variants`.
    terms  : list of :class:`CombTerm` (variant_id, case_id, factor).
    op     : one of LinearSum / Envelope / AbsSum / SRSS.
    missing: 'zero' | 'skip' | 'error' — how to treat ids absent from a term.
    require_same_stiffness: defaults to True for LinearSum. When the involved
        variants do not share a stiffness hash a ValueError is raised (LinearSum
        would be physically invalid) unless explicitly disabled.
    allow_geometry_mismatch: variants with different stiffness may still be
        *compared* (envelope) as a what-if; set True to acknowledge this.

    Returns a result dict; LinearSum/AbsSum/SRSS yield the flat
    ``{quantity: {id: [..]}}`` shape, Envelope yields ``{quantity:
    {'max':..., 'min':...}}`` (same convention as the solver).
    """
    if op not in _OPS:
        raise ValueError(f"op must be one of {_OPS}, got {op!r}")
    if require_same_stiffness is None:
        require_same_stiffness = (op == 'LinearSum')

    # Resolve each term to (factor, block).
    resolved: list = []
    hashes: set = set()
    geoms: set = set()
    for t in terms:
        vr = vres.get(t.variant_id)
        if vr is None:
            raise ValueError(f"Variant '{t.variant_id}' not in results.")
        block = _case_block(vr.results, t.case_id)
        if block is None:
            raise ValueError(
                f"Case '{t.case_id}' not found in variant '{t.variant_id}'.")
        resolved.append((t.factor, block))
        hashes.add(vr.stiffness_hash)
        geoms.add(vr.geometry_sig)

    same_stiffness = (len(hashes) <= 1)
    same_geometry = (len(geoms) <= 1)

    # Note (deliberate): bar internal forces (N/V/M) are a per-bar quantity, so
    # combining them across variants is always computed — including a LinearSum
    # sum of variants with *different* stiffness. The only consequence of a
    # different stiffness is that the combined displacements/reactions are not
    # physically meaningful; callers use the ``_same_stiffness`` flag to restrict
    # those views. No combination is blocked here.

    blocks = [b for _, b in resolved]
    quantities = ('displacements', 'reactions', 'element_forces', 'spring_forces')

    if op == 'Envelope':
        result = _envelope(resolved, quantities, missing)
    else:
        result = _accumulate(resolved, quantities, op, missing)

    # Triangle (plate/membrane) results: combined separately from the bar
    # quantities above because their derived fields (principal stresses /
    # Wood-Armer design moments) are nonlinear functions of the base fields —
    # see _combine_tri_stress.
    result['tri_stress'] = _combine_tri_stress(resolved, op, missing)

    # Also combine the N/V/M diagrams so the result can be drawn on the canvas
    # exactly like an in-model combination (under 'combo_distribution').
    dist_terms = []
    for t in terms:
        vr = vres.get(t.variant_id)
        dist_terms.append((t.factor, _case_distribution(vr.results, t.case_id)))
    result['combo_distribution'] = _combine_distributions(dist_terms, op, missing)
    # Whether displacements/reactions are physically meaningful: only when the
    # variants share the same stiffness (same geometry + supports). Otherwise the
    # combination is an envelope of *different structural systems* and only the
    # per-bar internal forces (and their diagrams) are valid.
    result['_same_stiffness'] = same_stiffness
    return result


def _case_distribution(results: dict, case_id: str) -> dict:
    """Return ``{elem_id: {x, N, V, M[, *_min]}}`` for a case (load case,
    analysis case or combination) inside one variant's results dict."""
    d = results.get('element_distribution', {}).get(case_id)
    if d is not None:
        return d
    ac = results.get('analysis_cases', {}).get(case_id, {})
    if ac.get('element_distribution'):
        return ac['element_distribution']
    cd = results.get('combo_distribution', {}).get(case_id)
    if cd is not None:
        return cd
    return {}


def _combine_distributions(dist_terms, op, missing):
    """Combine per-element N/V/M arrays across terms (mirrors the solver's
    combination diagram logic). Envelope yields max plus ``*_min`` bands."""
    out: dict = {}
    if missing == 'skip':
        elem_sets = [set(d.keys()) for _, d in dist_terms if d]
        elems = set.intersection(*elem_sets) if elem_sets else set()
    else:
        elems = set()
        for _, d in dist_terms:
            elems |= set(d.keys())

    for eid in elems:
        x = None
        comps = ('N', 'V', 'M')
        if op == 'Envelope':
            hi = {c: None for c in comps}
            lo = {c: None for c in comps}
            for factor, d in dist_terms:
                ed = d.get(eid)
                if ed is None:
                    continue
                if x is None:
                    x = ed['x']
                for c in comps:
                    base = np.asarray(ed[c])
                    band_lo = np.asarray(ed.get(c + '_min', ed[c]))
                    a, b = factor * base, factor * band_lo
                    chi = np.maximum(a, b); clo = np.minimum(a, b)
                    hi[c] = chi if hi[c] is None else np.maximum(hi[c], chi)
                    lo[c] = clo if lo[c] is None else np.minimum(lo[c], clo)
            if x is None:
                continue
            out[eid] = {'x': x,
                        'N': hi['N'], 'V': hi['V'], 'M': hi['M'],
                        'N_min': lo['N'], 'V_min': lo['V'], 'M_min': lo['M'],
                        'is_envelope': True}
        else:
            acc = {c: None for c in comps}
            for factor, d in dist_terms:
                ed = d.get(eid)
                if ed is None:
                    continue
                if x is None:
                    x = ed['x']
                for c in comps:
                    v = factor * np.asarray(ed[c])
                    if op == 'AbsSum':
                        v = np.abs(v)
                    elif op == 'SRSS':
                        v = v * v
                    acc[c] = v if acc[c] is None else acc[c] + v
            if x is None:
                continue
            if op == 'SRSS':
                for c in comps:
                    acc[c] = np.sqrt(acc[c])
            out[eid] = {'x': x, 'N': acc['N'], 'V': acc['V'], 'M': acc['M']}
    return out


def _accumulate(resolved, quantities, op, missing):
    out: dict = {}
    for q in quantities:
        is_elem = (q == 'element_forces')
        ids = _iter_ids([b for _, b in resolved], q, missing)
        acc: dict = {}
        for ident in ids:
            ends = ('i', 'j') if is_elem else (None,)
            slot = {} if is_elem else None
            for end in ends:
                vec = [0.0, 0.0, 0.0]
                for factor, block in resolved:
                    v = _get_vec(block, q, ident, end)
                    if v is None:
                        if missing == 'error':
                            raise ValueError(
                                f"Missing '{ident}' in {q} for a term.")
                        continue
                    for k in range(3):
                        cv = factor * v[k]
                        if op == 'LinearSum':
                            vec[k] += cv
                        elif op == 'AbsSum':
                            vec[k] += abs(cv)
                        elif op == 'SRSS':
                            vec[k] += cv * cv
                if op == 'SRSS':
                    vec = [math.sqrt(x) for x in vec]
                if is_elem:
                    slot[end] = vec
                else:
                    acc[ident] = vec
            if is_elem:
                acc[ident] = slot
        out[q] = acc
    return out


def _envelope(resolved, quantities, missing):
    out: dict = {}
    NINF, PINF = float('-inf'), float('inf')
    for q in quantities:
        is_elem = (q == 'element_forces')
        ids = _iter_ids([b for _, b in resolved], q, missing)
        hi: dict = {}
        lo: dict = {}
        for ident in ids:
            ends = ('i', 'j') if is_elem else (None,)
            hi_slot = {} if is_elem else None
            lo_slot = {} if is_elem else None
            for end in ends:
                hv = [NINF, NINF, NINF]
                lv = [PINF, PINF, PINF]
                touched = False
                for factor, block in resolved:
                    v = _get_vec(block, q, ident, end)
                    if v is None:
                        if missing == 'error':
                            raise ValueError(
                                f"Missing '{ident}' in {q} for a term.")
                        if missing == 'zero':
                            v = [0.0, 0.0, 0.0]
                        else:
                            continue
                    touched = True
                    for k in range(3):
                        cv = factor * v[k]
                        hv[k] = max(hv[k], cv)
                        lv[k] = min(lv[k], cv)
                if not touched:
                    hv = [0.0, 0.0, 0.0]; lv = [0.0, 0.0, 0.0]
                else:
                    hv = [0.0 if x == NINF else x for x in hv]
                    lv = [0.0 if x == PINF else x for x in lv]
                if is_elem:
                    hi_slot[end] = hv; lo_slot[end] = lv
                else:
                    hi[ident] = hv; lo[ident] = lv
            if is_elem:
                hi[ident] = hi_slot; lo[ident] = lo_slot
        out[q] = {'max': hi, 'min': lo}
    return out


# ---------------------------------------------------------------------------
# Triangle (plate/membrane) results — combined separately from the bar
# quantities above (Phase: variants × triangles).
#
# A triangle's per-case result dict carries a handful of *linear* state fields
# (sx/sy/txy for a membrane, mx/my/mxy/vx/vy for a plate) plus several fields
# *derived* from them by a nonlinear formula: principal stresses/moments
# (s1/s2/vm, m1/m2) and the Wood-Armer design moments (mx_bot/my_bot/mx_top/
# my_top). Summing (or enveloping) the derived fields directly, the way the
# bar quantities above are combined field-by-field, would not match the
# derived quantity of the actually-combined state — e.g. the sum of two
# von Mises stresses is not the von Mises stress of the summed tensor. So each
# base field is combined with the same arithmetic used for bars, and the
# derived fields are then recomputed from the combined base state, exactly as
# the solver computes them for one case.
# ---------------------------------------------------------------------------

_TRI_BASE_FIELDS = {
    'plane': ('sx', 'sy', 'txy'),
    'plate': ('mx', 'my', 'mxy', 'vx', 'vy'),
}


def _tri_kind(d: dict) -> Optional[str]:
    """'plane' (membrane CST/Allman/ESFEM) or 'plate' (DKT/MITC3) from the
    fields actually present in one triangle's result dict."""
    if 'sx' in d:
        return 'plane'
    if 'mx' in d:
        return 'plate'
    return None


def _tri_derived(kind: str, base: dict, formulation: str) -> dict:
    """Recompute the fields derived from *base* (the combined state) — the
    same formulas the solver itself uses for a single case, so a combined
    result reads exactly like a solved one."""
    if kind == 'plane':
        from .tri_elements import principal_stresses
        s1, s2, theta, vm = principal_stresses(base['sx'], base['sy'], base['txy'])
        return {'s1': s1, 's2': s2, 'theta': theta, 'vm': vm,
                'formulation': formulation}
    if kind == 'plate':
        from .plate_common import plate_moment_result
        return plate_moment_result(base['mx'], base['my'], base['mxy'],
                                   base['vx'], base['vy'], formulation)
    return {}


def _combine_tri_stress(resolved, op: str, missing: str) -> dict:
    """Combine ``tri_stress`` across variant terms. Mirrors ``_accumulate`` /
    ``_envelope`` for the base fields, then layers the derived fields back on
    via :func:`_tri_derived`. Returns the flat ``{tri_id: {...}}`` shape for
    LinearSum/AbsSum/SRSS, or ``{'max': ..., 'min': ...}`` for
    Envelope — same convention as the bar quantities."""
    blocks_ts = [(factor, block.get('tri_stress', {}) or {})
                for factor, block in resolved]
    sets = [set(ts.keys()) for _, ts in blocks_ts]
    if missing == 'skip':
        ids = set.intersection(*sets) if sets else set()
    else:
        ids = set.union(*sets) if sets else set()

    def _first_kind(tid):
        for _, ts in blocks_ts:
            d = ts.get(tid)
            if d is not None:
                k = _tri_kind(d)
                if k is not None:
                    return k, d.get('formulation', k)
        return None, None

    if op == 'Envelope':
        NINF, PINF = float('-inf'), float('inf')
        hi_out, lo_out = {}, {}
        for tid in ids:
            kind, formulation = _first_kind(tid)
            if kind is None:
                continue
            fields = _TRI_BASE_FIELDS[kind]
            hv = {f: NINF for f in fields}; lv = {f: PINF for f in fields}
            touched = False
            for factor, ts in blocks_ts:
                d = ts.get(tid)
                if d is None:
                    if missing == 'error':
                        raise ValueError(
                            f"Missing '{tid}' in tri_stress for a term.")
                    if missing != 'zero':
                        continue
                    d = {}
                touched = True
                for f in fields:
                    cv = factor * d.get(f, 0.0)
                    hv[f] = max(hv[f], cv); lv[f] = min(lv[f], cv)
            if not touched:
                hv = {f: 0.0 for f in fields}; lv = {f: 0.0 for f in fields}
            hi_entry = dict(hv); hi_entry.update(_tri_derived(kind, hv, formulation))
            lo_entry = dict(lv); lo_entry.update(_tri_derived(kind, lv, formulation))
            hi_out[tid] = hi_entry; lo_out[tid] = lo_entry
        return {'max': hi_out, 'min': lo_out}

    out = {}
    for tid in ids:
        kind, formulation = _first_kind(tid)
        if kind is None:
            continue
        fields = _TRI_BASE_FIELDS[kind]
        acc = {f: 0.0 for f in fields}
        for factor, ts in blocks_ts:
            d = ts.get(tid)
            if d is None:
                if missing == 'error':
                    raise ValueError(
                        f"Missing '{tid}' in tri_stress for a term.")
                continue
            for f in fields:
                cv = factor * d.get(f, 0.0)
                if op == 'AbsSum':
                    cv = abs(cv)
                elif op == 'SRSS':
                    cv = cv * cv
                acc[f] += cv
        if op == 'SRSS':
            acc = {f: math.sqrt(v) for f, v in acc.items()}
        entry = dict(acc); entry.update(_tri_derived(kind, acc, formulation))
        out[tid] = entry
    return out
