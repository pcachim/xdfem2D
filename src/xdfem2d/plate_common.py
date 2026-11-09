"""Shared machinery for plate-bending triangles (thin and shear-deformable).

Everything the plate elements hold in common lives here, so a Mindlin-Reissner
element (MITC3, and later DST) reuses it rather than re-deriving it, and so the
*reported* result of a triangle can never drift between formulations:

* :func:`bending_D` — the plate bending constitutive D_b = (t³/12)·plane_D, the
  same one the DKT already used (re-exported there as ``plate_D``);
* :func:`shear_D` — the transverse-shear constitutive D_s = k·G·t·I, which a
  thin (Kirchhoff/DKT) element does not have and a shear-deformable one does;
* :func:`wood_armer` — the Wood–Armer design moments, independent of which
  element produced mx/my/mxy;
* :func:`plate_moment_result` — one triangle's reported dict (moments, principal
  values, transverse shears, Wood–Armer), assembled in one place so every
  formulation emits the same keys.

Sign convention (shared): sagging positive under a downward load, matching the
grillage bar's M. The caller passes mx/my/mxy already in that convention.
"""
from __future__ import annotations

import math

import numpy as np

from .tri_elements import plane_D

#: Transverse-shear correction factor for a section with a parabolic
#: through-thickness shear distribution — the standard 5/6 for a solid slab.
SHEAR_CORRECTION = 5.0 / 6.0


def bending_D(E: float, nu: float, t: float) -> np.ndarray:
    """3×3 plate bending constitutive D_b = (t³/12)·plane_D (m = D_b·κ).

    The plane-stress D the membrane elements already use, scaled by the plate
    rigidity. Shared by every plate-bending triangle."""
    return (t ** 3 / 12.0) * plane_D(E, nu, plane_strain=False)


def shear_D(E: float, nu: float, t: float,
            k: float = SHEAR_CORRECTION) -> np.ndarray:
    """2×2 transverse-shear constitutive D_s (q = D_s·γ) = k·G·t·I.

    Isotropic (equal in x and y, no coupling), with G = E / 2(1+ν) — the same
    shear modulus the grillage bar uses — and the shear correction factor *k*
    (5/6 by default). A thin-plate element has no such term; a Mindlin-Reissner
    element (MITC3, DST) does, and this is where it comes from."""
    G = E / (2.0 * (1.0 + nu))
    Gt = k * G * t
    return np.array([[Gt, 0.0], [0.0, Gt]])


def wood_armer(mx: float, my: float, mxy: float):
    """Wood–Armer (W-A) design moments for orthogonal X/Y reinforcement.

    Returns ``(mx_bot, my_bot, mx_top, my_top)``. The simple rule is
    ``m* = m ± |mxy|``; when that pushes a value across zero the wrong-sign term
    is not kept — that direction goes to zero and the other is corrected with the
    ``mxy²/m`` term, so no reinforcement is wasted and the twist is still carried.
    Bottom moments clamp at ≥ 0, top moments at ≤ 0.

    Independent of the element formulation: it takes the three moments and
    returns the four design moments, so every plate element shares it.
    """
    amxy = abs(mxy)
    mx_bot, my_bot = mx + amxy, my + amxy
    if mx_bot < 0.0:
        mx_bot = 0.0
        my_bot = my + abs(mxy * mxy / mx)
        if my_bot < 0.0:
            my_bot = 0.0
    elif my_bot < 0.0:
        my_bot = 0.0
        mx_bot = mx + abs(mxy * mxy / my)
        if mx_bot < 0.0:
            mx_bot = 0.0

    mx_top, my_top = mx - amxy, my - amxy
    if mx_top > 0.0:
        mx_top = 0.0
        my_top = my - abs(mxy * mxy / mx)
        if my_top > 0.0:
            my_top = 0.0
    elif my_top > 0.0:
        my_top = 0.0
        mx_top = mx - abs(mxy * mxy / my)
        if mx_top > 0.0:
            mx_top = 0.0

    return mx_bot, my_bot, mx_top, my_top


def plate_moment_result(mx: float, my: float, mxy: float,
                        vx: float, vy: float, formulation: str) -> dict:
    """One triangle's reported result dict from its stress resultants.

    *mx, my, mxy* are the centroid moments (sagging positive); *vx, vy* the
    transverse shears — however the element obtained them: a thin (DKT) element
    derives them from the moment field, a shear-deformable one (MITC3, DST)
    reads them from its own shear field D_s·γ. Either way the reported keys are
    the same, which is the point of doing this in one place: ``area_results``,
    ``report_io`` and the GUI never have to know which element answered.
    """
    # Principal moments and their direction (same algebra as plane stress).
    c = (mx + my) / 2.0
    r = math.hypot((mx - my) / 2.0, mxy)
    m1, m2 = c + r, c - r
    theta = 0.5 * math.degrees(math.atan2(2.0 * mxy, mx - my)) \
        if (mx - my or mxy) else 0.0

    mx_bot, my_bot, mx_top, my_top = wood_armer(mx, my, mxy)
    return {'mx': mx, 'my': my, 'mxy': mxy,
            'm1': m1, 'm2': m2, 'theta': theta,
            'vx': float(vx), 'vy': float(vy),
            'mx_bot': mx_bot, 'my_bot': my_bot,   # Wood–Armer (W-A), bottom
            'mx_top': mx_top, 'my_top': my_top,   # Wood–Armer (W-A), top
            'formulation': formulation}
