"""
Element stiffness matrices for xdfem2D.
"""
from __future__ import annotations
import math
import numpy as np
from .models import BarElement, Section, Material, Node


def bar_element_geometry(node_i: Node, node_j: Node):
    """Return (length, cos, sin, angle_deg) for a bar element."""
    dx = node_j.x - node_i.x
    dy = node_j.y - node_i.y
    length = math.sqrt(dx * dx + dy * dy)
    if length <= 0.0:
        raise ValueError(f"Element has zero length (nodes {node_i.id} and {node_j.id})")
    angle = math.atan2(dy, dx)
    return length, math.cos(angle), math.sin(angle), math.degrees(angle)


def bar_stiffness_local(E: float, A: float, I: float, L: float) -> np.ndarray:
    """
    6×6 local stiffness matrix for an Euler-Bernoulli beam element.
    DOF order: [u_i, v_i, θ_i, u_j, v_j, θ_j]
    """
    EAL  = E * A / L
    EIL  = 4.0 * E * I / L          # 4EI/L
    EIL2 = 1.5 * EIL / L            # 6EI/L²
    EIL3 = 2.0 * EIL2 / L           # 12EI/L³

    k = np.zeros((6, 6))
    k[0, 0] =  EAL;  k[0, 3] = -EAL
    k[1, 1] =  EIL3; k[1, 2] =  EIL2; k[1, 4] = -EIL3; k[1, 5] =  EIL2
    k[2, 1] =  EIL2; k[2, 2] =  EIL;  k[2, 4] = -EIL2; k[2, 5] =  EIL / 2
    k[3, 0] = -EAL;  k[3, 3] =  EAL
    k[4, 1] = -EIL3; k[4, 2] = -EIL2; k[4, 4] =  EIL3; k[4, 5] = -EIL2
    k[5, 1] =  EIL2; k[5, 2] =  EIL / 2; k[5, 4] = -EIL2; k[5, 5] =  EIL
    return k, EAL, EIL, EIL2, EIL3


def transformation_matrix(cos: float, sin: float) -> np.ndarray:
    """6×6 transformation matrix from local to global coordinates."""
    T = np.zeros((6, 6))
    T[0, 0] = cos;  T[0, 1] = sin
    T[1, 0] = -sin; T[1, 1] = cos
    T[2, 2] = 1.0
    T[3, 3] = cos;  T[3, 4] = sin
    T[4, 3] = -sin; T[4, 4] = cos
    T[5, 5] = 1.0
    return T


def bar_stiffness_global(E: float, A: float, I: float, L: float,
                          cos: float, sin: float):
    """Return global 6×6 stiffness matrix and pre-computed stiffness coefficients."""
    k_local, EAL, EIL, EIL2, EIL3 = bar_stiffness_local(E, A, I, L)
    T = transformation_matrix(cos, sin)
    k_global = T.T @ k_local @ T
    return k_global, EAL, EIL, EIL2, EIL3


def bar_geometric_stiffness_global(N: float, L: float,
                                    cos: float, sin: float) -> np.ndarray:
    """
    6×6 geometric (initial stress) stiffness matrix for P-Delta analysis.
    Based on the standard linearised geometric stiffness for a beam-column.
    DOF order (global): [ux_i, uy_i, rz_i, ux_j, uy_j, rz_j]

    Reference: McGuire, Gallagher & Ziemian, "Matrix Structural Analysis", 2nd ed.
    """
    if L <= 0:
        return np.zeros((6, 6))

    NL = N / L
    kg_local = np.zeros((6, 6))
    # Standard Euler-Bernoulli beam-column geometric stiffness (transverse terms)
    c1 = NL * 6.0 / 5.0
    c2 = NL * L / 10.0
    c3 = NL * 2.0 * L * L / 15.0
    c4 = NL * L * L / 30.0

    kg_local[1, 1] =  c1;  kg_local[1, 2] =  c2
    kg_local[1, 4] = -c1;  kg_local[1, 5] =  c2
    kg_local[2, 1] =  c2;  kg_local[2, 2] =  c3
    kg_local[2, 4] = -c2;  kg_local[2, 5] = -c4
    kg_local[4, 1] = -c1;  kg_local[4, 2] = -c2
    kg_local[4, 4] =  c1;  kg_local[4, 5] = -c2
    kg_local[5, 1] =  c2;  kg_local[5, 2] = -c4
    kg_local[5, 4] = -c2;  kg_local[5, 5] =  c3

    T = transformation_matrix(cos, sin)
    return T.T @ kg_local @ T


def is_beam(cos: float, sin: float) -> bool:
    """Return True if the element is a beam (inclination < 45°, i.e. more horizontal)."""
    return abs(sin) < abs(cos)


# ---------------------------------------------------------------------------
# End releases (moment hinges) — static condensation
# ---------------------------------------------------------------------------

def released_local_dofs(hinge_i: bool, hinge_j: bool) -> list[int]:
    """
    Return the list of *local* DOF indices that are released (rotations).
    Local DOF order: [u_i, v_i, θ_i, u_j, v_j, θ_j] → θ_i = 2, θ_j = 5.
    """
    r: list[int] = []
    if hinge_i:
        r.append(2)
    if hinge_j:
        r.append(5)
    return r


def condense_local_stiffness(k: np.ndarray, released: list[int]) -> np.ndarray:
    """
    Static condensation of a local element stiffness matrix.

    The released DOFs are eliminated (Guyan/static condensation). The returned
    matrix has the same 6×6 shape with the released rows/columns set to zero,
    so it can be assembled into the global matrix exactly like the full matrix.
    """
    if not released:
        return k
    n = k.shape[0]
    keep = [i for i in range(n) if i not in released]
    r = released
    krr = k[np.ix_(r, r)]
    krr_inv = np.linalg.inv(krr)
    kcc = k[np.ix_(keep, keep)] - k[np.ix_(keep, r)] @ krr_inv @ k[np.ix_(r, keep)]
    out = np.zeros_like(k)
    out[np.ix_(keep, keep)] = kcc
    return out


def condense_local_load(k: np.ndarray, q: np.ndarray, released: list[int]) -> np.ndarray:
    """
    Condense a local equivalent nodal-load vector ``q`` for released DOFs,
    using the *full* (un-condensed) local stiffness ``k``.

    Returns a 6-vector with the released components set to zero and the retained
    components corrected by the redistribution of the released load.
    """
    if not released:
        return q
    n = len(q)
    keep = [i for i in range(n) if i not in released]
    r = released
    krr_inv = np.linalg.inv(k[np.ix_(r, r)])
    out = np.zeros_like(q)
    out[keep] = q[keep] - k[np.ix_(keep, r)] @ krr_inv @ q[r]
    return out
