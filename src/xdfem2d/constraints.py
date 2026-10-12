"""
Expansion of multi-point constraints into canonical linear equations.

A :class:`~xdfem2d.models.Constraint` is authored as a high-level kind
('rigid_link', 'equal_dof') or as a raw 'equation'. The solver, however, only
ever works with equations of the single canonical form::

    sum(coef_i * dof_i  for i in terms) = value

This module is the one place that turns the high-level kinds into that form, so
the assembly and (later) the master-slave reduction can never disagree on what a
constraint means. It is deliberately free of NumPy and of the solver, so it can
be unit-tested on its own.
"""
from __future__ import annotations

from .models import COMPONENT_OFFSET


def _dof(struc, node_id: str, comp: str) -> int:
    """Global DOF index of *comp* ('ux'|'uy'|'tz') at *node_id*."""
    base = struc.node_dof_index[node_id]
    return base + COMPONENT_OFFSET[comp]


def _rigid_link_equations(struc, c) -> list[tuple[list[tuple[int, float]], float]]:
    """Three equations per slave binding it to the master as a rigid body.

    In-plane rigid-body kinematics (master m, slave s):
        ux_s = ux_m - (y_s - y_m) * tz_m
        uy_s = uy_m + (x_s - x_m) * tz_m
        tz_s = tz_m
    written as ``... = 0``.
    """
    m = c.master
    nm = struc.nodes[m]
    eqs: list[tuple[list[tuple[int, float]], float]] = []
    for s in c.slaves:
        ns = struc.nodes[s]
        dx = ns.x - nm.x
        dy = ns.y - nm.y
        # ux_s - ux_m + dy * tz_m = 0
        eqs.append(([(_dof(struc, s, "ux"), 1.0),
                     (_dof(struc, m, "ux"), -1.0),
                     (_dof(struc, m, "tz"), dy)], 0.0))
        # uy_s - uy_m - dx * tz_m = 0
        eqs.append(([(_dof(struc, s, "uy"), 1.0),
                     (_dof(struc, m, "uy"), -1.0),
                     (_dof(struc, m, "tz"), -dx)], 0.0))
        # tz_s - tz_m = 0
        eqs.append(([(_dof(struc, s, "tz"), 1.0),
                     (_dof(struc, m, "tz"), -1.0)], 0.0))
    return eqs


def _equal_dof_equations(struc, c) -> list[tuple[list[tuple[int, float]], float]]:
    """For each component, tie every node to the first: dof_i - dof_0 = 0."""
    nodes = list(c.nodes)
    n0 = nodes[0]
    eqs: list[tuple[list[tuple[int, float]], float]] = []
    for comp in c.components:
        d0 = _dof(struc, n0, comp)
        for ni in nodes[1:]:
            eqs.append(([(_dof(struc, ni, comp), 1.0), (d0, -1.0)], 0.0))
    return eqs


def _equation_terms(struc, c) -> list[tuple[list[tuple[int, float]], float]]:
    """A raw 'equation' constraint, mapped from (node, comp, coef) to dof index."""
    terms = [(_dof(struc, node_id, comp), float(coef))
             for node_id, comp, coef in c.terms]
    return [(terms, float(c.value))]


def expand_constraints(struc) -> list[tuple[list[tuple[int, float]], float]]:
    """Return every enabled constraint as a list of canonical equations.

    Each equation is ``(terms, value)`` where *terms* is a list of
    ``(dof_index, coef)`` and the equation asserts ``sum(coef * u[dof]) = value``.
    Disabled constraints are skipped. Unknown kinds are ignored (forward-compat).
    """
    out: list[tuple[list[tuple[int, float]], float]] = []
    for c in struc.constraints.values():
        if not getattr(c, "enabled", True):
            continue
        if c.kind == "rigid_link":
            out.extend(_rigid_link_equations(struc, c))
        elif c.kind == "equal_dof":
            out.extend(_equal_dof_equations(struc, c))
        elif c.kind == "equation":
            out.extend(_equation_terms(struc, c))
    return out
