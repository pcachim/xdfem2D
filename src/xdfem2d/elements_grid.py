"""
Grillage bar element — the plate-domain counterpart of the plane frame bar.

A grillage bar lies in the model plane (X, Y) and carries out-of-plane
behaviour only: transverse deflection w along global Z, torsion about its own
axis and bending about its in-plane transverse axis. Nodal DOFs, in the
plate domain's storage order: (w, tx, ty) — deflection and rotations about
global X and Y, right-hand rule.

Local element DOF order (i-end then j-end):

    [w_i, φ_i, ψ_i, w_j, φ_j, ψ_j]

where φ is the rotation about the local x' axis (the bar axis — torsion) and
ψ the rotation about the local y' axis (bending). Both follow the right-hand
rule, which fixes the sign that trips this element up in every textbook: the
bending rotation relates to the slope as ψ = −dw/dx'.

The bending block is therefore the classic Euler-Bernoulli beam stiffness
written in (w, θ̃) variables with θ̃ = dw/dx' — i.e. *identical algebra* to
the plane beam in :mod:`xdfem2d.elements` — re-expressed in ψ = −θ̃, which
flips the sign of every w↔rotation coupling term. Torsion adds GJ/L on the
φ pair. End releases (``hinge_i``/``hinge_j``) release the *bending* rotation
ψ, which sits at local indices 2 and 5 — the same indices the plane beam's
releases use, so the static condensation helpers are shared unchanged.

Reference: e.g. R. C. Hibbeler, "Structural Analysis" (grid member stiffness),
or W. McGuire, R. Gallagher & R. Ziemian, "Matrix Structural Analysis".
"""
from __future__ import annotations

import numpy as np


def grid_stiffness_local(E: float, I: float, G: float, J: float,
                         L: float) -> tuple:
    """6×6 local stiffness for a grillage bar.

    DOF order: [w_i, φ_i, ψ_i, w_j, φ_j, ψ_j] (see module docstring).
    Returns ``(k, GJL, EIL, EIL2, EIL3)`` where GJL = G·J/L plays the role
    the axial EA/L coefficient plays for the plane bar, and EIL = 4EI/L,
    EIL2 = 6EI/L², EIL3 = 12EI/L³ are the same bending coefficients the
    plane beam caches (the force-recovery algebra is shared).
    """
    GJL  = G * J / L
    EIL  = 4.0 * E * I / L          # 4EI/L
    EIL2 = 1.5 * EIL / L            # 6EI/L²
    EIL3 = 2.0 * EIL2 / L           # 12EI/L³

    k = np.zeros((6, 6))
    # Transverse deflection / bending — beam stiffness in (w, θ̃), θ̃ = dw/dx',
    # converted to ψ = −θ̃ (every w↔rotation coupling changes sign).
    k[0, 0] =  EIL3; k[0, 2] = -EIL2; k[0, 3] = -EIL3; k[0, 5] = -EIL2
    k[2, 0] = -EIL2; k[2, 2] =  EIL;  k[2, 3] =  EIL2; k[2, 5] =  EIL / 2
    k[3, 0] = -EIL3; k[3, 2] =  EIL2; k[3, 3] =  EIL3; k[3, 5] =  EIL2
    k[5, 0] = -EIL2; k[5, 2] =  EIL / 2; k[5, 3] =  EIL2; k[5, 5] =  EIL
    # Torsion
    k[1, 1] =  GJL;  k[1, 4] = -GJL
    k[4, 1] = -GJL;  k[4, 4] =  GJL
    return k, GJL, EIL, EIL2, EIL3


def grid_transformation_matrix(cos: float, sin: float) -> np.ndarray:
    """6×6 transformation from global (w, tx, ty) to local (w, φ, ψ).

    w is invariant (global Z stays global Z); the two rotations rotate in the
    plane like a vector: φ (about x' = (c, s)) = c·tx + s·ty and
    ψ (about y' = (−s, c)) = −s·tx + c·ty. Orthogonal, so Tᵀ maps back.
    """
    T = np.zeros((6, 6))
    T[0, 0] = 1.0
    T[1, 1] = cos;  T[1, 2] = sin
    T[2, 1] = -sin; T[2, 2] = cos
    T[3, 3] = 1.0
    T[4, 4] = cos;  T[4, 5] = sin
    T[5, 4] = -sin; T[5, 5] = cos
    return T


def grid_stiffness_global(E: float, I: float, G: float, J: float, L: float,
                          cos: float, sin: float):
    """Return the global 6×6 stiffness and the cached coefficients
    ``(k_global, GJL, EIL, EIL2, EIL3)`` — the grid counterpart of
    :func:`xdfem2d.elements.bar_stiffness_global`."""
    k_local, GJL, EIL, EIL2, EIL3 = grid_stiffness_local(E, I, G, J, L)
    T = grid_transformation_matrix(cos, sin)
    return T.T @ k_local @ T, GJL, EIL, EIL2, EIL3
