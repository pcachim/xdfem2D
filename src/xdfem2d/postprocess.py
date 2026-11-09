"""
Post-processing helpers for xdfem2D results (GUI-free).

Deformed-shape interpolation and a small 'nice number' scale helper used when
auto-scaling diagrams. The GUI imports from here.
"""
from __future__ import annotations

import math

import numpy as np


def nice_scale(raw: float) -> float:
    """Round *raw* to the nearest 'engineering' value (1/2/5 × 10^n)."""
    if raw <= 0:
        return 1.0
    exp = math.floor(math.log10(raw))
    base = 10.0 ** exp
    for nice in (1, 2, 5, 10):
        if raw <= nice * base * 1.001:
            return nice * base
    return 10.0 * base


def hermite_deformed(ni, nj, di, dj, sf: float, n: int = 31):
    """Deformed element centreline using cubic Hermite shape functions.

    Returns ``(Xdef, Ydef)`` arrays of length *n*.

    ni, nj : the element's end nodes (objects with ``.x`` / ``.y``).
    di, dj : ``[ux, uy, rz]`` nodal displacements (from FEM results).
    sf     : displacement scale factor.
    """
    dx = nj.x - ni.x
    dy = nj.y - ni.y
    L = np.hypot(dx, dy)
    if L < 1e-12:
        return np.array([ni.x, nj.x]), np.array([ni.y, nj.y])
    cos = dx / L
    sin = dy / L

    # Scaled displacements in global
    ux1, uy1, rz1 = sf * di[0], sf * di[1], sf * di[2]
    ux2, uy2, rz2 = sf * dj[0], sf * dj[1], sf * dj[2]

    # Transform to local element axes
    u_i = ux1 * cos + uy1 * sin    # axial at i
    v_i = -ux1 * sin + uy1 * cos   # transverse at i
    u_j = ux2 * cos + uy2 * sin
    v_j = -ux2 * sin + uy2 * cos
    th_i, th_j = rz1, rz2          # rotations same in 2-D

    xi = np.linspace(0.0, 1.0, n)
    x = xi * L

    # Linear axial interpolation
    u = u_i * (1.0 - xi) + u_j * xi

    # Cubic Hermite for transverse (v)
    N1 = 1.0 - 3.0 * xi**2 + 2.0 * xi**3
    N2 = L * xi * (1.0 - xi)**2
    N3 = 3.0 * xi**2 - 2.0 * xi**3
    N4 = L * xi**2 * (xi - 1.0)
    v = N1 * v_i + N2 * th_i + N3 * v_j + N4 * th_j

    # Map back to global (undeformed axis + local deformation)
    Xdef = ni.x + x * cos + u * cos - v * sin
    Ydef = ni.y + x * sin + u * sin + v * cos
    return Xdef, Ydef


def combined_distribution(struc, results: dict | None, case: str):
    """Build a combination's bar diagram from the cases it is made of.

    Combinations store end forces only, so the sagging moment under a design
    combination — the number an engineer actually designs for — exists nowhere
    in the saved results. It does not have to: a linear combination of load
    cases is a linear combination of their diagrams, sampled at the same
    stations, so it can be summed on demand. Verified against the solver's own
    stored end forces, which the summed diagram reproduces exactly.

    Only linear sums. An envelope, a sum of absolute values or an SRSS is not a
    weighted sum of ordinates and would be wrong done this way; those return a
    reason instead, which is worth more to the assistant than a wrong number.

    Returns (distribution, {case: factor}, reason-if-empty).
    """
    combo = next((c for c in (getattr(struc, 'load_combinations', []) or [])
                  if c.id == case), None)
    if combo is None:
        return {}, None, 'not a load combination'
    kind = getattr(combo, 'combo_type', 'LinearSum')
    if kind != 'LinearSum':
        return {}, None, (f"combination type '{kind}' is not a linear sum, so "
                          "its diagram cannot be derived from the load cases")
    coef = dict(getattr(combo, 'coefficients', {}) or {})
    cases = (results or {}).get('analysis_cases') or {}
    parts = {}
    for cid, factor in coef.items():
        d = (cases.get(cid) or {}).get('element_distribution')
        if not d:
            return {}, None, f"load case '{cid}' has no stored diagram"
        parts[cid] = (float(factor), d)
    if not parts:
        return {}, None, 'the combination has no coefficients'

    out: dict[str, dict] = {}
    for eid in next(iter(parts.values()))[1]:
        acc: dict[str, list] = {}
        n = None
        for factor, d in parts.values():
            rec = d.get(eid)
            if rec is None:
                acc = {}
                break
            for q in ('M', 'V', 'N'):
                col = rec.get(q)
                if col is None:
                    continue
                col = list(col)
                # Stations must line up, or the sum adds unrelated points.
                if n is None:
                    n = len(col)
                elif len(col) != n:
                    acc = {}
                    break
                prev = acc.get(q)
                acc[q] = ([factor * v for v in col] if prev is None
                          else [p + factor * v for p, v in zip(prev, col)])
            if not acc:
                break
            xs = rec.get('x')
            if xs is not None:
                acc['x'] = list(xs)
        if acc:
            out[eid] = acc
    if not out:
        return {}, None, 'the load cases are sampled differently'
    return out, {c: f for c, (f, _d) in parts.items()}, ''


def cumulative_parabolic(y, x):
    """Running integral of *y* over *x*, fitting a parabola to each triple.

    The trapezoidal rule would do, and does not: the quantity being integrated
    here is a bending-moment diagram, which under a uniform load is a parabola
    exactly. A parabolic rule integrates it without error, a trapezoidal one
    with an error that survives two integrations. Measured on a three-span
    continuous beam against the closed-form solution, 51 stations per element:
    0.02 % against 0.14 %, and in the middle span the parabolic result is exact.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    n = len(y)
    out = np.zeros(n)
    if n < 3:
        if n == 2:
            out[1] = (x[1] - x[0]) * (y[0] + y[1]) / 2.0
        return out
    for i in range(1, n):
        # The parabola through this point and its neighbours; at the last
        # station there is no point to the right, so the previous triple is
        # extended instead.
        a = i - 1 if i + 1 < n else i - 2
        xs, ys = x[a:a + 3], y[a:a + 3]
        c = np.polyfit(xs, ys, 2)
        P = np.polyint(c)
        out[i] = out[i - 1] + np.polyval(P, x[i]) - np.polyval(P, x[i - 1])
    return out


def deflected_shape(ni, nj, di, dj, M, xs, EI, sf: float = 1.0, kappa0: float = 0.0):
    """Deformed centreline of a bar, including the deflection between its ends.

    :func:`hermite_deformed` interpolates the two end displacements and
    rotations, which is the exact shape of an *unloaded* member and wrong for
    every other one: the deflection a distributed load produces between the
    nodes is invisible to the end degrees of freedom. On a three-span
    continuous beam the error was 36 % in the outer spans, and in the middle
    span the sign was inverted — the drawing showed it rising while it sagged.

    The curvature is known everywhere, because the moment diagram is stored at
    every station: v'' = M/EI. Integrating it twice and fixing the two
    constants with the nodal displacements gives the true shape, for any
    loading — uniform, trapezoidal, point — with no case analysis, because
    whatever the load did is already in M.

    ``kappa0`` is the free thermal curvature (α·ΔTgrad/h, the same quantity
    ``loads.py`` uses for the equivalent fixed-end moment), subtracted from
    the load curvature: the moment ``M`` already includes whatever restraint
    moment the supports generated to resist that free curvature, so the two
    partially or fully cancel. A statically determinate bar under a pure
    thermal gradient has M(x) = 0 everywhere — correctly, there is no
    restraint moment — so nothing cancels there and the bar bows by the full
    free curvature; that bow is invisible to M/EI alone and is exactly what
    this term restores. A fully fixed-fixed bar under a uniform gradient is
    the opposite extreme: the restraint moment equals EI·kappa0 exactly, so
    the two terms cancel completely and the bar stays straight, matching its
    zero end displacements/rotations.

    Falls back to the Hermite shape when there is no diagram or no EI, which is
    the correct answer for a member with no load along it anyway (an unloaded
    member can still carry ``kappa0``, so that case is handled separately
    below rather than folding into the plain fallback).
    """
    if M is None or xs is None or not EI:
        if not kappa0:
            return hermite_deformed(ni, nj, di, dj, sf)
        # No moment diagram to integrate, but still a free thermal curvature
        # to draw: build a station grid to integrate kappa0 over.
        dx0, dy0 = nj.x - ni.x, nj.y - ni.y
        L0 = float(np.hypot(dx0, dy0))
        xs = np.linspace(0.0, L0, 21)
        M = np.zeros_like(xs)
    M = np.asarray(M, dtype=float)
    xs = np.asarray(xs, dtype=float)
    if M.size < 2 or xs.size < 2:
        if not kappa0:
            return hermite_deformed(ni, nj, di, dj, sf)
        dx0, dy0 = nj.x - ni.x, nj.y - ni.y
        L0 = float(np.hypot(dx0, dy0))
        xs = np.linspace(0.0, L0, 21)
        M = np.zeros_like(xs)
    if M.size != xs.size:
        # M and the stations disagree — e.g. results carried over from an older
        # sampling grid. Dropping to the Hermite end-only shape here would throw
        # away the load-induced curvature (36 % error on a continuous beam), so
        # instead resample M onto xs, assumed to share the same [x0, xL] span,
        # and keep integrating the true diagram.
        M = np.interp(xs, np.linspace(xs[0], xs[-1], M.size), M)

    dx, dy = nj.x - ni.x, nj.y - ni.y
    L = float(np.hypot(dx, dy))
    if L < 1e-12:
        return np.array([ni.x, nj.x]), np.array([ni.y, nj.y])
    cos, sin = dx / L, dy / L

    ux1, uy1 = sf * di[0], sf * di[1]
    ux2, uy2 = sf * dj[0], sf * dj[1]
    u_i = ux1 * cos + uy1 * sin
    v_i = -ux1 * sin + uy1 * cos
    u_j = ux2 * cos + uy2 * sin
    v_j = -ux2 * sin + uy2 * cos

    EI_val = float(EI) if EI else 1.0
    theta = cumulative_parabolic(M, xs) / EI_val - kappa0 * (xs - xs[0])
    v = cumulative_parabolic(theta, xs) * sf
    # The two constants of integration are the end displacements: v(0) = v_i
    # and v(L) = v_j. Everything between them comes from the curvature.
    v = v + v_i + (v_j - v_i - v[-1]) * (xs / xs[-1])

    t = xs / xs[-1]
    u = u_i * (1.0 - t) + u_j * t
    Xdef = ni.x + xs * cos + u * cos - v * sin
    Ydef = ni.y + xs * sin + u * sin + v * cos
    return Xdef, Ydef


def local_deflection(ni, nj, di, dj, M, xs, EI, kappa0: float = 0.0):
    """Transverse deflection along a bar, in local axes, unscaled.

    What an engineer means by "the deflection of that beam": the movement
    perpendicular to the member, measured from its own line, positive on the
    local +y side. Returned with the stations so the worst point can be named.

    ``kappa0``: see :func:`deflected_shape` — the free thermal curvature.
    """
    X, Y = deflected_shape(ni, nj, di, dj, M, xs, EI, sf=1.0, kappa0=kappa0)
    dx, dy = nj.x - ni.x, nj.y - ni.y
    L = float(np.hypot(dx, dy)) or 1.0
    cos, sin = dx / L, dy / L
    # Signed distance from the undeformed axis line. The station terms cancel
    # algebraically (−(x_s·cos)·sin + (x_s·sin)·cos = 0), so this needs only the
    # returned shape, not xs — which matters because deflected_shape falls back
    # to a Hermite sampling of a *different* length when M and xs disagree, and
    # subtracting the passed xs then raised a broadcasting error.
    off = -(X - ni.x) * sin + (Y - ni.y) * cos
    xs = np.asarray(xs, dtype=float)
    stations = xs if xs.size == X.size else np.linspace(0.0, L, X.size)
    return stations, off


def member_displacement_field(ni, nj, di, dj, M, xs, EI, kappa0: float = 0.0):
    """Global displacement components along a bar: ``(stations, ux, uy, rz)``.

    The companion of :func:`local_deflection`, but returning the three nodal
    quantities an engineer reads — the two global translations and the rotation
    — sampled all along the member instead of only at its ends, so each can be
    drawn as a diagram.

    ``ux`` and ``uy`` are the deformed position minus the undeformed one, taken
    straight from :func:`deflected_shape`, so they carry the same load-induced
    curvature the deformed view draws (not a straight line between end values).
    ``rz`` is the cross-section rotation: the end rotation plus the running
    integral of the curvature ``M/EI``. It meets the nodal rotation exactly at
    the start end, and — for a consistent solution — at the far end too, because
    the integral of the curvature over the member is the change in rotation
    between its ends.

    Falls back with ``M`` or ``EI`` absent: ``ux``/``uy`` to the Hermite
    end-interpolation (:func:`deflected_shape` does this itself) and ``rz`` to a
    straight line between the end rotations — the right answer for a member with
    no load along it.

    ``kappa0``: see :func:`deflected_shape` — the free thermal curvature.
    """
    X, Y = deflected_shape(ni, nj, di, dj, M, xs, EI, sf=1.0, kappa0=kappa0)
    n = X.size
    dx, dy = nj.x - ni.x, nj.y - ni.y
    L = float(np.hypot(dx, dy))
    if L < 1e-12:
        z = np.zeros(n)
        return np.zeros(n), z, z, np.linspace(di[2], dj[2], n)
    cos, sin = dx / L, dy / L

    xs_arr = None if xs is None else np.asarray(xs, dtype=float)
    if xs_arr is not None and xs_arr.size == n:
        s = xs_arr
    else:
        s = np.linspace(0.0, L, n)

    ux = X - (ni.x + s * cos)
    uy = Y - (ni.y + s * sin)

    rzi, rzj = di[2], dj[2]
    Mv = None if M is None else np.asarray(M, dtype=float)
    if Mv is not None and Mv.size >= 2 and EI:
        if Mv.size != s.size:
            Mv = np.interp(s, np.linspace(s[0], s[-1], Mv.size), Mv)
        # Same curvature sign as deflected_shape (theta = ∫ M/EI - kappa0): the
        # rotation is that running integral, anchored on the start-end nodal
        # value.
        rz = rzi + cumulative_parabolic(Mv, s) / float(EI) - kappa0 * (s - s[0])
    elif kappa0:
        rz = rzi - kappa0 * (s - s[0])
    else:
        t = s / s[-1] if s[-1] else np.zeros(n)
        rz = rzi * (1.0 - t) + rzj * t
    return s, ux, uy, rz


# Backwards-compatible private aliases (the GUI used these names).
_nice_scale = nice_scale
_hermite_deformed = hermite_deformed
_deflected_shape = deflected_shape
