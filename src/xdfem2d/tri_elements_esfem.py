"""Edge-based Smoothed Finite Element Method (ES-FEM) for T3 triangles.

A member of the S-FEM family for the plane (membrane) triangle. It reuses the
standard 3-node linear triangle (2 DOF/node, ux/uy) and the constant CST
strain-displacement matrix :func:`xdfem2d.tri_elements.cst_B_area`, but assembles
the stiffness — and, for consistency, the thermal load and stress recovery —
over **edge-based smoothing domains** instead of per element.

For every edge the smoothing domain Ω_k is the union of the two sub-triangles
that join the edge to the centroids of the (one or two) elements sharing it. For
linear T3 elements the smoothed strain over Ω_k is the area-weighted average of
the constituent elements' constant strains, so the smoothed strain-displacement
matrix is

    B̄_k = (1 / A_k) · Σ_i (A_i / 3) · B_i ,   A_k = Σ_i (A_i / 3)

(the sum over the 1 or 2 elements sharing the edge, B_i their constant CST B),
and the domain stiffness / thermal load / stress are

    K_k = B̄_kᵀ · D · B̄_k · A_k · t
    f_k = B̄_kᵀ · D · ε̄₀_k · A_k · t          (ε̄₀_k the area-weighted mean ε₀)
    σ_k = D · (B̄_k · u_k − ε̄₀_k)

assembled onto the (ux, uy) DOFs of the union of the adjacent elements' nodes.
Summing the three edge domains of a boundary-only triangle recovers exactly its
CST stiffness and thermal load — a useful correctness check (see the tests).

Consistency note: the thermal load uses the SAME smoothed B̄_k as the stiffness,
so a freely expanding ES-FEM body reports zero stress, exactly as for CST.

Edge (Neumann boundary) loads need no ES-FEM variant — a consistent edge
traction distributes to the two edge nodes' translational DOFs independently of
the interior formulation, so :func:`xdfem2d.loads._apply_tri_edge_loads` already
handles ES-FEM meshes.

Scope: linear statics; smoothing happens within a single ES-FEM section (same
material and thickness); edges between different sections/formulations are not
smoothed across.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from .tri_elements import cst_B_area, plane_D, principal_stresses


def esfem_edge_domains(struc):
    """Yield ``(section_name, edge_key, [triangles])`` per edge, grouped so only
    triangles of the same ES-FEM section share a domain (1 triangle = boundary
    edge, 2 = interior edge)."""
    by_sec: dict = defaultdict(list)
    for tri in getattr(struc, "tri_elements", []):
        sec = struc.tri_sections.get(tri.section_name)
        if sec is not None and getattr(sec, "formulation", "CST") == "ES-FEM":
            by_sec[tri.section_name].append(tri)
    for sname, tris in by_sec.items():
        edge_map: dict = defaultdict(list)
        for tri in tris:
            ids = (tri.node_i, tri.node_j, tri.node_k)
            for a, b in ((ids[0], ids[1]), (ids[1], ids[2]), (ids[2], ids[0])):
                edge_map[tuple(sorted((a, b)))].append(tri)
        for key, adj in edge_map.items():
            yield sname, key, adj


def _smoothed_B(struc, adj):
    """Return ``(union_nodes, col_map, B̄, A_k)`` for the edge domain of the
    adjacent triangles *adj* (the area-weighted average CST B and domain area)."""
    union: list = []
    for tri in adj:
        for nid in (tri.node_i, tri.node_j, tri.node_k):
            if nid not in union:
                union.append(nid)
    col = {nid: 2 * i for i, nid in enumerate(union)}
    Bbar = np.zeros((3, 2 * len(union)))
    A_k = 0.0
    for tri in adj:
        ids = (tri.node_i, tri.node_j, tri.node_k)
        if any(n not in struc.nodes for n in ids):
            continue
        coords = [(struc.nodes[n].x, struc.nodes[n].y) for n in ids]
        B, area = cst_B_area(coords)               # 3×6, constant per element
        w = area / 3.0                              # portion of this element in Ω_k
        A_k += w
        for local, nid in enumerate(ids):
            c = col[nid]
            Bbar[:, c:c + 2] += w * B[:, 2 * local:2 * local + 2]
    if A_k > 0.0:
        Bbar /= A_k
    return union, col, Bbar, A_k


def esfem_domains(struc):
    """Yield a dict per edge smoothing domain with everything the stiffness,
    thermal-load and stress routines need: ``sname, edge, adj, union, col,
    Bbar, A_k, sec, mat, D, t, dofs``."""
    for sname, edge, adj in esfem_edge_domains(struc):
        sec = struc.tri_sections[sname]
        mat = struc.materials[sec.material_name]
        nu = getattr(mat, "poisson", 0.2)
        union, col, Bbar, A_k = _smoothed_B(struc, adj)
        if A_k <= 0.0:
            continue
        dofs: list = []
        for nid in union:
            base = struc.node_dof_index[nid]
            dofs += [base, base + 1]
        yield {"sname": sname, "edge": edge, "adj": adj, "union": union,
               "col": col, "Bbar": Bbar, "A_k": A_k, "sec": sec, "mat": mat,
               "D": plane_D(mat.elastic_modulus, nu, sec.plane_strain),
               "t": sec.thickness, "dofs": dofs}


def esfem_domain_matrices(struc):
    """Yield ``(k, dofs, cache_key, cache_entry)`` per edge smoothing domain."""
    for d in esfem_domains(struc):
        Bbar, D, A_k, t = d["Bbar"], d["D"], d["A_k"], d["t"]
        k = (Bbar.T @ D @ Bbar) * (A_k * t)
        entry = {"B": Bbar, "D": D, "area": A_k, "dofs": d["dofs"],
                 "formulation": "ES-FEM", "nodes": tuple(d["union"])}
        yield k, d["dofs"], ("esfem_edge", d["sname"], d["edge"]), entry


# ── Thermal loads ──────────────────────────────────────────────────────────

def esfem_thermal_loads(struc, F, case_index):
    """Add the ES-FEM consistent thermal loads to *F* (per-case columns).

    ``f_k = B̄_kᵀ · D · ε̄₀_k · A_k · t`` with ε̄₀_k the area-weighted mean of the
    adjacent triangles' free thermal strains for that load case — the same
    smoothing as the stiffness, so a free expansion produces zero stress."""
    from .loads import tri_thermal_strain
    temps = getattr(struc, "tri_temperature_loads", [])
    if not temps:
        return
    # {load_case: {tri_id: dt_mean}}
    by_case: dict = defaultdict(dict)
    for tl in temps:
        if tl.load_case_id in case_index:
            by_case[tl.load_case_id][tl.tri_id] = tl.dt_mean
    if not by_case:
        return
    for d in esfem_domains(struc):
        sec, mat, D, Bbar = d["sec"], d["mat"], d["D"], d["Bbar"]
        A_k, t, adj = d["A_k"], d["t"], d["adj"]
        for lc, dts in by_case.items():
            ic = case_index[lc]
            eps0 = np.zeros(3)
            for tri in adj:
                dt = dts.get(tri.id)
                if not dt:
                    continue
                ids = (tri.node_i, tri.node_j, tri.node_k)
                coords = [(struc.nodes[n].x, struc.nodes[n].y) for n in ids]
                _B, area = cst_B_area(coords)
                eps0 += (area / 3.0) * tri_thermal_strain(sec, mat, dt)
            if not eps0.any():
                continue
            eps0 /= A_k                              # area-weighted mean over Ω_k
            f = (Bbar.T @ D @ eps0) * (A_k * t)
            for dof, val in zip(d["dofs"], f):
                F[dof, ic] += val


# ── Stress recovery ────────────────────────────────────────────────────────

def _domain_stress(D, Bbar, u_dom, eps0):
    eps = Bbar @ u_dom
    if eps0 is not None:
        eps = eps - eps0
    sig = D @ eps
    return float(sig[0]), float(sig[1]), float(sig[2])


def _pack(acc):
    """Average each triangle's incident-edge stresses and pack the result."""
    out = {}
    for tid, (ssum, n) in acc.items():
        sx, sy, txy = ssum / n
        s1, s2, th, vm = principal_stresses(sx, sy, txy)
        out[tid] = {"sx": sx, "sy": sy, "txy": txy, "s1": s1, "s2": s2,
                    "theta": th, "vm": vm, "formulation": "ES-FEM"}
    return out


def _eps0_domain(struc, adj, A_k, thermal, sec, mat):
    """Area-weighted mean free thermal strain over the domain (or None)."""
    if not thermal:
        return None
    from .loads import tri_thermal_strain
    eps0 = np.zeros(3)
    hit = False
    for tri in adj:
        e = thermal.get(tri.id)
        if e is None:
            continue
        hit = True
        ids = (tri.node_i, tri.node_j, tri.node_k)
        coords = [(struc.nodes[n].x, struc.nodes[n].y) for n in ids]
        _B, area = cst_B_area(coords)
        eps0 += (area / 3.0) * np.asarray(e, float)
    return (eps0 / A_k) if hit else None


def esfem_stresses(struc, u_vec, thermal=None):
    """Per-triangle ES-FEM stress from a global displacement vector — each
    triangle takes the mean of its incident edge-domain smoothed stresses."""
    u = np.asarray(u_vec)
    acc: dict = {}
    for d in esfem_domains(struc):
        u_dom = u[d["dofs"]]
        eps0 = _eps0_domain(struc, d["adj"], d["A_k"], thermal, d["sec"], d["mat"])
        s = np.array(_domain_stress(d["D"], d["Bbar"], u_dom, eps0))
        for tri in d["adj"]:
            ssum, n = acc.get(tri.id, (np.zeros(3), 0))
            acc[tri.id] = (ssum + s, n + 1)
    return _pack(acc)


def esfem_stresses_from_disp(struc, disp, thermal=None):
    """Per-triangle ES-FEM stress from a displacement dict ``{node: [ux, uy,
    rz]}`` (analysis cases / combinations)."""
    acc: dict = {}
    for d in esfem_domains(struc):
        u_dom = np.array([v for nid in d["union"]
                          for v in disp.get(nid, (0.0, 0.0, 0.0))[:2]])
        eps0 = _eps0_domain(struc, d["adj"], d["A_k"], thermal, d["sec"], d["mat"])
        s = np.array(_domain_stress(d["D"], d["Bbar"], u_dom, eps0))
        for tri in d["adj"]:
            ssum, n = acc.get(tri.id, (np.zeros(3), 0))
            acc[tri.id] = (ssum + s, n + 1)
    return _pack(acc)
