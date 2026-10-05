"""Canonical validation cases — closed-form benchmarks for the FEM engine.

This is the single source of truth shared by:
  * ``build_x2d.py``           — writes one ``.x2d`` per case into ``models/``;
  * ``tests/tests_model/test_validation_x2d.py`` — loads those ``.x2d`` and checks them;
  * ``make_report.py``         — fills the validation document.

Each case mirrors one section of *Validacao_Programa_MEF.docx*. The analytical
value of every quantity is computed here from first principles (not the rounded
number in the document), so the tolerance can be tight.

Units are consistent: length m, force kN, stress kN/m² (÷1000 → MPa), so
E = 30×10⁶ kN/m². Displacements are in m (×1000 → mm).

Two cases deviate from the document, by design (see the project discussion):
  * 3.2 — the 17.26 mm target comes from beam theory, not plane elasticity, and
    a CST mesh converges to it slowly. It is therefore validated by *convergence*
    over three meshes, not by an exact value.
  * 3.3 — as written (edge x=0 fully fixed) the Poisson contraction is blocked
    near the support, so the stress is NOT constant. The support is relaxed to
    ux-only (one corner pinned in y) which makes it the intended uniaxial patch,
    reproduced exactly by any mesh.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

# ── Shared properties ────────────────────────────────────────────────────────
E = 30.0e6          # kN/m²
NU = 0.20
ALPHA = 1.0e-5      # °C⁻¹
B, H = 0.30, 0.50   # bar rectangular section
A_BAR = B * H                    # 0.15 m²
I_BAR = B * H ** 3 / 12.0        # 3.125e-3 m⁴

# Arc (quarter-circle cantilever) parameters.
ARC_R = 3.0                      # m
ARC_P = 10.0                     # kN, vertical load at the free tip


# ── Result-extraction helpers ────────────────────────────────────────────────
def node_at(struc, x, y, tol=1e-6):
    """Id of the node at (x, y)."""
    for nid, n in struc.nodes.items():
        if abs(n.x - x) <= tol and abs(n.y - y) <= tol:
            return nid
    raise KeyError(f"no node at ({x}, {y})")

def uy(res, case, nid):
    return res["displacements"][case][nid][1]

def ux(res, case, nid):
    return res["displacements"][case][nid][0]

def rot(res, case, nid):
    return res["displacements"][case][nid][2]

def max_abs_M(res, case, *elems):
    import numpy as np
    vals = [np.abs(res["element_distribution"][case][e]["M"]) for e in elems]
    return float(max(np.max(v) for v in vals))

def max_abs_V(res, case, *elems):
    import numpy as np
    vals = [np.abs(res["element_distribution"][case][e]["V"]) for e in elems]
    return float(max(np.max(v) for v in vals))

def max_abs_M_all(res, case):
    """Peak |M| over every element — for object beams whose element ids are
    generated at solve time and so are not known when the case is written."""
    import numpy as np
    ed = res["element_distribution"][case]
    return float(max(np.max(np.abs(ed[e]["M"])) for e in ed))

def max_abs_V_all(res, case):
    import numpy as np
    ed = res["element_distribution"][case]
    return float(max(np.max(np.abs(ed[e]["V"])) for e in ed))

def reaction_moment_sum(res, case):
    """Σ|reaction Mz| — the base moment for a single fixed support."""
    return sum(abs(rv[2]) for rv in res["reactions"][case].values())

def reaction_fy_sum(res, case):
    return sum(rv[1] for rv in res["reactions"][case].values())

def axial_N(res, case, elem):
    return res["element_forces"][case][elem]["i"][0]

def reaction(res, case, nid, comp):
    return res["reactions"][case][nid][comp]

def sx_range(res, case):
    ts = res["tri_stress"][case]
    v = [d["sx"] for d in ts.values()]
    return min(v), max(v)

def max_vm(res, case):
    ts = res["tri_stress"][case]
    return max(d["vm"] for d in ts.values())

def max_abs_sx(res, case):
    ts = res["tri_stress"][case]
    return max(abs(d["sx"]) for d in ts.values())

def max_abs_txy(res, case):
    ts = res["tri_stress"][case]
    return max(abs(d["txy"]) for d in ts.values())


# ── Global-equilibrium utilities (used by the universal check) ───────────────
def applied_resultant(struc, case):
    """(Fx, Fy) resultant of the applied mechanical loads of *case*.

    Settlements and temperature apply no external force, so a model driven only
    by them has a self-equilibrating reaction set (resultant ≈ 0)."""
    fx = fy = 0.0
    for p in struc.point_loads:
        if p.load_case_id == case:
            fx += p.fx
            fy += p.fy
    for d in struc.distributed_loads:
        if d.load_case_id == case:
            n_i = struc.nodes[struc.bar_elements_by_id[d.element_id].node_i]
            n_j = struc.nodes[struc.bar_elements_by_id[d.element_id].node_j]
            L = math.hypot(n_j.x - n_i.x, n_j.y - n_i.y)
            fx += 0.5 * (d.fxe + d.fxd) * L
            fy += 0.5 * (d.fye + d.fyd) * L
    # Surface edge tractions (used by the object-based plate 3.1a): (fx, fy) is a
    # force per unit length, so the resultant is that times the edge length.
    from xdfem2d.loads import _edge_load_fxy
    for el in getattr(struc, "surface_edge_loads", []):
        if el.load_case_id == case:
            na, nb = struc.nodes.get(el.node_a), struc.nodes.get(el.node_b)
            if na is None or nb is None:
                continue
            obj = struc.geometry_objects.get(el.object_id)
            sec = struc.tri_sections.get(getattr(obj, "tri_section_name", ""))
            thk = sec.thickness if sec is not None else 1.0
            efx, efy = _edge_load_fxy(el, na, nb, thk)
            L = math.hypot(nb.x - na.x, nb.y - na.y)
            fx += efx * L
            fy += efy * L
    return fx, fy

def reaction_resultant(res, case):
    R = res["reactions"][case]
    return sum(r[0] for r in R.values()), sum(r[1] for r in R.values())

def spring_resultant(res, struc, case):
    """(Fx, Fy) taken by the springs — Σ k·u over node and element springs.

    Springs are not supports, so their force never shows up in ``reactions``.
    Global equilibrium of a sprung model therefore reads

        Σ Reactions + Σ Applied loads − Σ Spring forces = 0

    The element-spring contribution uses the SAME lumped ``k·L/2`` block as the
    assembly (imported, not re-derived) so the check stays independent of the
    formulation while remaining consistent with it."""
    from xdfem2d.assembly import _elem_spring_global_block
    disp = res["displacements"][case]
    sx = sy = 0.0
    sf = res.get("spring_forces", {}).get(case, {})
    for nid, sp in struc.node_springs.items():
        v = sf.get(nid)
        if v is None:
            u = disp[nid]
            v = [u[0] * sp.kx, u[1] * sp.ky, u[2] * sp.kt]
        sx += v[0]; sy += v[1]
    for esp in struc.element_springs.values():
        el = struc.bar_elements_by_id[esp.element_id]
        ni, nj = struc.nodes[el.node_i], struc.nodes[el.node_j]
        L = math.hypot(nj.x - ni.x, nj.y - ni.y) or 1.0
        gxx, gxy, gyy = _elem_spring_global_block(esp, (nj.x - ni.x) / L,
                                                  (nj.y - ni.y) / L)
        for n in (el.node_i, el.node_j):
            u = disp[n]
            sx += (gxx * u[0] + gxy * u[1]) * L / 2.0
            sy += (gxy * u[0] + gyy * u[1]) * L / 2.0
    return sx, sy


# ── Model builders ───────────────────────────────────────────────────────────
def _new():
    from xdfem2d import Structure2D
    return Structure2D()

def _beam_material(s, gamma=0.0):
    s.add_material("C", E, gamma, alpha=ALPHA, poisson=NU)
    s.add_section("S", "C", b=B, h=H)

def _plate_material(s, thickness):
    s.add_material("C", E, 0.0, alpha=ALPHA, poisson=NU)
    s.add_tri_section("T", "C", thickness=thickness)

def _mesh_plate(s, L, h, nx, ny):
    """Structured triangle mesh over an L×h rectangle. Returns id grid."""
    ids = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            nid = f"n{i}_{j}"
            s.add_node(nid, L * i / nx, h * j / ny)
            ids[(i, j)] = nid
    e = 0
    for j in range(ny):
        for i in range(nx):
            a, b = ids[(i, j)], ids[(i + 1, j)]
            c, d = ids[(i + 1, j + 1)], ids[(i, j + 1)]
            s.add_tri_element(f"e{e}", a, b, c, "T"); e += 1
            s.add_tri_element(f"e{e}", a, c, d, "T"); e += 1
    return ids


def build_2_1():
    """Simply supported beam, central point load P=50, L=6."""
    s = _new(); _beam_material(s)
    s.add_node("A", 0, 0); s.add_node("C", 3, 0); s.add_node("B", 6, 0)
    s.add_bar_element("E1", "A", "C", "S"); s.add_bar_element("E2", "C", "B", "S")
    s.add_support("PIN", ux=True, uy=True); s.add_support("ROL", ux=False, uy=True)
    s.assign_support("A", "PIN"); s.assign_support("B", "ROL")
    s.add_load_case("LC"); s.add_point_load("C", "LC", fy=-50.0)
    return s

def build_2_2():
    """Simply supported beam, UDL q=20, L=6."""
    s = _new(); _beam_material(s)
    s.add_node("A", 0, 0); s.add_node("C", 3, 0); s.add_node("B", 6, 0)
    s.add_bar_element("E1", "A", "C", "S"); s.add_bar_element("E2", "C", "B", "S")
    s.add_support("PIN", ux=True, uy=True); s.add_support("ROL", ux=False, uy=True)
    s.assign_support("A", "PIN"); s.assign_support("B", "ROL")
    s.add_load_case("LC")
    s.add_distributed_load("E1", "LC", fye=-20.0, fyd=-20.0); s.add_distributed_load("E2", "LC", fye=-20.0, fyd=-20.0)
    return s

def build_cut_1():
    """Simply supported beam, UDL q=20, L=6 (same as 2.2) with two section
    cuts: C_MID at midspan (x=3) and C_Q at quarter-span (x=1.5). Validates
    ``xdfem2d.cuts.cut_result`` against the same closed-form V(x), M(x) used
    for case 2.2, at a point that falls on a node (C_MID) and one that falls
    inside an element (C_Q)."""
    s = build_2_2()
    s.add_cut("C_MID", 3.0, -1.0, 3.0, 1.0, name="Midspan")
    s.add_cut("C_Q", 1.5, -1.0, 1.5, 1.0, name="Quarter span")
    return s

def build_2_3():
    """Cantilever, tip point load P=30, L=3."""
    s = _new(); _beam_material(s)
    s.add_node("A", 0, 0); s.add_node("B", 3, 0)
    s.add_bar_element("E", "A", "B", "S")
    s.add_support("FIX", ux=True, uy=True, tz=True); s.assign_support("A", "FIX")
    s.add_load_case("LC"); s.add_point_load("B", "LC", fy=-30.0)
    return s

def build_2_4():
    """Fixed-fixed beam, imposed settlement δ=10 mm at B, L=5."""
    s = _new(); _beam_material(s)
    s.add_node("A", 0, 0); s.add_node("B", 5, 0)
    s.add_bar_element("E", "A", "B", "S")
    s.add_support("FIX", ux=True, uy=True, tz=True)
    s.assign_support("A", "FIX"); s.assign_support("B", "FIX")
    s.add_load_case("LC")
    s.create_support_settlement("B", "LC", uy=-0.010)
    return s

def build_2_5a():
    """Fully restrained bar, uniform ΔT=+20, L=6, A=0.15."""
    s = _new(); _beam_material(s)
    s.add_node("A", 0, 0); s.add_node("B", 6, 0)
    s.add_bar_element("E", "A", "B", "S")
    s.add_support("FIX", ux=True, uy=True, tz=True)
    s.assign_support("A", "FIX"); s.assign_support("B", "FIX")
    s.add_load_case("LC"); s.add_temperature_load("E", "LC", delta_t_uniform=20.0)
    return s

def build_2_5b():
    """Simply supported beam, thermal gradient ΔT_grad=+20, h=0.5, L=6."""
    s = _new(); _beam_material(s)
    s.add_node("A", 0, 0); s.add_node("C", 3, 0); s.add_node("B", 6, 0)
    s.add_bar_element("E1", "A", "C", "S"); s.add_bar_element("E2", "C", "B", "S")
    s.add_support("PIN", ux=True, uy=True); s.add_support("ROL", ux=False, uy=True)
    s.assign_support("A", "PIN"); s.assign_support("B", "ROL")
    s.add_load_case("LC")
    s.add_temperature_load("E1", "LC", delta_t_gradient=20.0)
    s.add_temperature_load("E2", "LC", delta_t_gradient=20.0)
    return s

def build_3_1():
    """Patch test — uniform tension. Plate L=2×h=1, t=0.1, F=100 in x."""
    s = _new(); _plate_material(s, 0.10)
    L, h, nx, ny = 2.0, 1.0, 2, 1
    ids = _mesh_plate(s, L, h, nx, ny)
    # x=0: ux fixed on the edge, corner (0,0) also pinned in y (remove rigid body).
    s.add_support("RX", ux=True, uy=False); s.add_support("PIN", ux=True, uy=True)
    for j in range(ny + 1):
        s.assign_support(ids[(0, j)], "PIN" if j == 0 else "RX")
    s.add_load_case("LC")
    # Consistent nodal forces for a uniform edge traction: ends get half.
    F = 100.0
    for j in range(ny + 1):
        w = 0.5 if (j == 0 or j == ny) else 1.0
        s.add_point_load(ids[(nx, j)], "LC", fx=F * w / ny)
    return s

def _build_3_2(nx, ny):
    """Cantilever wall-beam L=4×h=0.5, t=0.2, tip load P=50 at mid-height."""
    s = _new(); _plate_material(s, 0.20)
    L, h = 4.0, 0.5
    ids = _mesh_plate(s, L, h, nx, ny)
    s.add_support("FIX", ux=True, uy=True)
    for j in range(ny + 1):
        s.assign_support(ids[(0, j)], "FIX")
    s.add_load_case("LC")
    s.add_point_load(ids[(nx, ny // 2)], "LC", fy=-50.0)
    return s

def build_3_2_coarse():  return _build_3_2(16, 4)
def build_3_2_medium():  return _build_3_2(32, 8)
def build_3_2_fine():    return _build_3_2(64, 16)

def build_3_3():
    """Imposed-displacement patch — plate L=2×h=1, t=0.1, ux=1 mm at x=L.

    Support relaxed from the document (ux-only at x=0, one corner pinned in y)
    so the state is truly uniaxial and exact for any mesh."""
    s = _new(); _plate_material(s, 0.10)
    L, h, nx, ny = 2.0, 1.0, 2, 1
    ids = _mesh_plate(s, L, h, nx, ny)
    s.add_support("RX", ux=True, uy=False); s.add_support("PIN", ux=True, uy=True)
    for j in range(ny + 1):
        s.assign_support(ids[(0, j)], "PIN" if j == 0 else "RX")
    s.add_support("RXL", ux=True, uy=False)
    s.add_load_case("LC")
    for j in range(ny + 1):
        s.assign_support(ids[(nx, j)], "RXL")
        s.create_support_settlement(ids[(nx, j)], "LC", ux=0.001)
    return s

def build_3_4():
    """Free plate, uniform ΔT=+30. L=2×h=1, t=0.1, minimal restraint."""
    s = _new(); _plate_material(s, 0.10)
    L, h, nx, ny = 2.0, 1.0, 2, 1
    ids = _mesh_plate(s, L, h, nx, ny)
    s.add_support("PIN", ux=True, uy=True); s.add_support("RY", ux=False, uy=True)
    s.assign_support(ids[(0, 0)], "PIN"); s.assign_support(ids[(nx, 0)], "RY")
    s.add_load_case("LC")
    for t in s.tri_elements:
        s.add_tri_temperature_load(t.id, "LC", dt_i=30.0, dt_j=30.0, dt_k=30.0)
    return s


# ── Geometry-object versions (elements generated at solve time) ──────────────
def build_2_1a():
    """2.1 as a geometry OBJECT: a polyline (0,0)-(3,0)-(6,0) that meshes into
    bars, midspan being a defining node so the point load has a node to land on.
    Same simply-supported beam, P=50 at midspan — must match 2.1 exactly."""
    s = _new(); _beam_material(s)
    s.add_geo_polyline("L", [(0, 0), (3, 0), (6, 0)], section_name="S",
                       divisions=2)
    s.add_support("PIN", ux=True, uy=True); s.add_support("ROL", ux=False, uy=True)
    s.assign_support(node_at(s, 0, 0), "PIN"); s.assign_support(node_at(s, 6, 0), "ROL")
    s.add_load_case("LC")
    s.add_point_load(node_at(s, 3, 0), "LC", fy=-50.0)
    return s

def build_3_1a():
    """3.1 as a geometry OBJECT: a rectangle meshed into CST triangles, loaded
    by a surface edge traction. The left edge uses ONE support on BOTH corners
    (ux only) so it propagates along the whole edge — a mismatched pair does not
    propagate — plus a uy point at (L,0), which lies on y=0 where the exact
    field has uy=0. Must match 3.1 exactly for any mesh."""
    s = _new(); _plate_material(s, 0.10)
    L, h = 2.0, 1.0
    s.add_geo_rectangle("R", (0, 0), (L, h), section_name="T",
                        target_size=0.5)
    p00, p0h = node_at(s, 0, 0), node_at(s, 0, h)
    pL0, pLh = node_at(s, L, 0), node_at(s, L, h)
    s.add_support("RX", ux=True, uy=False)     # same support on both left corners
    s.add_support("RY", ux=False, uy=True)     # uy at (L,0): on y=0, so consistent
    s.assign_support(p00, "RX"); s.assign_support(p0h, "RX")
    s.assign_support(pL0, "RY")
    s.add_load_case("LC")
    s.add_surface_edge_load("EL", "R", pL0, pLh, "LC", fx=100.0)
    return s

def _build_arc(divisions):
    """Quarter-circle cantilever, radius R, fixed at (R,0), vertical load P at
    the free tip (0,R). Bending-dominated; the analytical tip deflection is
    δ_v = π P R³ / (4 E I) (Castigliano, thin curved beam). The FE uses straight
    chords + axial/shear, so it converges to this from a discretisation."""
    s = _new(); _beam_material(s)
    s.add_geo_arc("A", 0.0, 0.0, ARC_R, 0.0, 90.0, section_name="S",
                  divisions=divisions)
    s.add_support("FIX", ux=True, uy=True, tz=True)
    s.assign_support(node_at(s, ARC_R, 0.0), "FIX")
    s.add_load_case("LC")
    s.add_point_load(node_at(s, 0.0, ARC_R), "LC", fy=-ARC_P)
    return s

def build_arc():        return _build_arc(64)
def build_arc_coarse(): return _build_arc(16)
def build_arc_medium(): return _build_arc(32)


# ── Allman-triangle versions of the CST cases ────────────────────────────────
# The Allman element carries a drilling (tz) DOF at every node. It reproduces a
# constant-stress state exactly ONLY when that DOF is restrained (=0 for a pure
# stretch); left free, the patch test is not passed. So the membrane patches
# below fix tz on every node. Bending (3.2) is the opposite: the drilling DOF is
# what gives Allman its edge over CST, so it is left free there. Thermal loads
# are not implemented for Allman, so there is no Allman 3.4.
def _mesh_plate_allman(s, L, h, nx, ny, fix_tz_all):
    ids = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            nid = f"n{i}_{j}"
            s.add_node(nid, L * i / nx, h * j / ny)
            ids[(i, j)] = nid
    e = 0
    for j in range(ny):
        for i in range(nx):
            a, b = ids[(i, j)], ids[(i + 1, j)]
            c, d = ids[(i + 1, j + 1)], ids[(i, j + 1)]
            s.add_tri_element(f"e{e}", a, b, c, "T"); e += 1
            s.add_tri_element(f"e{e}", a, c, d, "T"); e += 1
    return ids

def build_3_1_allman():
    """3.1 tension patch with the Allman element, tz restrained on every node."""
    s = _new()
    s.add_material("C", E, 0.0, alpha=ALPHA, poisson=NU)
    s.add_tri_section("T", "C", thickness=0.10, formulation="Allman")
    L, h, nx, ny = 2.0, 1.0, 2, 1
    ids = _mesh_plate_allman(s, L, h, nx, ny, True)
    s.add_support("RXT", ux=True, uy=False, tz=True)
    s.add_support("PINT", ux=True, uy=True, tz=True)
    s.add_support("TZ", ux=False, uy=False, tz=True)
    for (i, j), nid in ids.items():
        if i == 0:
            s.assign_support(nid, "PINT" if j == 0 else "RXT")
        else:
            s.assign_support(nid, "TZ")
    s.add_load_case("LC")
    F = 100.0
    for j in range(ny + 1):
        w = 0.5 if (j == 0 or j == ny) else 1.0
        s.add_point_load(ids[(nx, j)], "LC", fx=F * w / ny)
    return s

def build_3_3_allman():
    """3.3 imposed-displacement patch with the Allman element, tz restrained."""
    s = _new()
    s.add_material("C", E, 0.0, alpha=ALPHA, poisson=NU)
    s.add_tri_section("T", "C", thickness=0.10, formulation="Allman")
    L, h, nx, ny = 2.0, 1.0, 2, 1
    ids = _mesh_plate_allman(s, L, h, nx, ny, True)
    s.add_support("RXT", ux=True, uy=False, tz=True)
    s.add_support("PINT", ux=True, uy=True, tz=True)
    s.add_support("RXLT", ux=True, uy=False, tz=True)
    s.add_support("TZ", ux=False, uy=False, tz=True)
    s.add_load_case("LC")
    for (i, j), nid in ids.items():
        if i == 0:
            s.assign_support(nid, "PINT" if j == 0 else "RXT")
        elif i == nx:
            s.assign_support(nid, "RXLT")
            s.create_support_settlement(nid, "LC", ux=0.001)
        else:
            s.assign_support(nid, "TZ")
    return s

def _build_3_2_allman(nx, ny):
    """3.2 cantilever wall-beam with the Allman element, drilling DOF FREE —
    which is where Allman beats CST. Converges to beam theory faster."""
    s = _new()
    s.add_material("C", E, 0.0, poisson=NU)
    s.add_tri_section("T", "C", thickness=0.20, formulation="Allman")
    L, h = 4.0, 0.5
    ids = _mesh_plate_allman(s, L, h, nx, ny, False)
    s.add_support("FIX", ux=True, uy=True)
    for j in range(ny + 1):
        s.assign_support(ids[(0, j)], "FIX")
    s.add_load_case("LC")
    s.add_point_load(ids[(nx, ny // 2)], "LC", fy=-50.0)
    return s

def build_3_2a_coarse(): return _build_3_2_allman(8, 2)
def build_3_2a_medium(): return _build_3_2_allman(16, 4)
def build_3_2a_fine():   return _build_3_2_allman(32, 8)

def _build_3_2_esfem(nx, ny):
    """3.2 cantilever wall-beam with the ES-FEM element (edge-smoothed CST, 2
    DOF/node). The smoothing relaxes the over-stiff CST, so it converges to beam
    theory faster on the same mesh — like Allman, but without a drilling DOF."""
    s = _new()
    s.add_material("C", E, 0.0, poisson=NU)
    s.add_tri_section("T", "C", thickness=0.20, formulation="ES-FEM")
    L, h = 4.0, 0.5
    ids = _mesh_plate(s, L, h, nx, ny)
    s.add_support("FIX", ux=True, uy=True)
    for j in range(ny + 1):
        s.assign_support(ids[(0, j)], "FIX")
    s.add_load_case("LC")
    s.add_point_load(ids[(nx, ny // 2)], "LC", fy=-50.0)
    return s

def build_3_2e_coarse(): return _build_3_2_esfem(8, 2)
def build_3_2e_medium(): return _build_3_2_esfem(16, 4)
def build_3_2e_fine():   return _build_3_2_esfem(32, 8)

def build_3_4_allman():
    """3.4 free thermal expansion with the Allman element. Now that the thermal
    load and stress recovery cover Allman, a free expansion gives σ=0 and the
    same δ as the CST — the drilling DOF stays free (no restraint to feel)."""
    s = _new()
    s.add_material("C", E, 0.0, alpha=ALPHA, poisson=NU)
    s.add_tri_section("T", "C", thickness=0.10, formulation="Allman")
    L, h, nx, ny = 2.0, 1.0, 2, 1
    ids = _mesh_plate_allman(s, L, h, nx, ny, False)
    s.add_support("PIN", ux=True, uy=True); s.add_support("RY", ux=False, uy=True)
    s.assign_support(ids[(0, 0)], "PIN"); s.assign_support(ids[(nx, 0)], "RY")
    s.add_load_case("LC")
    for t in s.tri_elements:
        s.add_tri_temperature_load(t.id, "LC", dt_i=30.0, dt_j=30.0, dt_k=30.0)
    return s


# ═══ SPRINGS ════════════════════════════════════════════════════════════════
# Node springs (kx, ky, kt on a node DOF) are added to the diagonal of K, so
# every closed-form "structure + elastic support" result is reproduced EXACTLY.
#
# Element (Winkler) springs are LUMPED — k·L/2 at each end node — not the
# consistent foundation matrix. Two consequences drive the design of the cases
# below:
#   * a uniform state (rigid block on a uniform foundation) is still exact,
#     because k·L/2 + k·L/2 = k·L reproduces the resultant exactly;
#   * a varying state (beam on elastic foundation) converges as O(h²), so those
#     cases are validated by convergence over three meshes, not by one value.
# A rigid block also picks up a spurious clamped-beam moment diagram of order
# q·h²/12 — an artefact of lumping the foundation reaction at the nodes only;
# case S-B1 measures it and checks that it dies out with refinement.

def _rigid_section(s, name="R"):
    """A practically rigid bar: used to isolate the spring behaviour from the
    flexibility of the member (the analytic values are then rigid-body ones)."""
    s.add_section(name, "C", b=1.0, h=1.0,
                  area_override=10.0, inertia_override=1.0e3)


# ── S-A1 — springs in series with an axial bar (exact) ──────────────────────
SA1_L = 6.0
SA1_KB = E * A_BAR / SA1_L               # bar axial stiffness, 7.5e5 kN/m
SA1_K = SA1_KB / 3.0                     # spring, 2.5e5 kN/m
SA1_P = 1000.0                           # kN
SA1_D = SA1_P / (SA1_KB + SA1_K)         # 1.0 mm
SA1_N = SA1_P * SA1_KB / (SA1_KB + SA1_K)

def build_s_a1():
    """Bar fixed at A, axial spring kx at the free end B, axial load P at B.
    δ = P/(k_bar + k), N = P·k_bar/(k_bar + k), F_mola = k·δ."""
    s = _new(); _beam_material(s)
    s.add_node("A", 0, 0); s.add_node("B", SA1_L, 0)
    s.add_bar_element("E", "A", "B", "S")
    s.add_support("FIX", ux=True, uy=True, tz=True)
    s.add_support("VY", ux=False, uy=True, tz=True)
    s.assign_support("A", "FIX"); s.assign_support("B", "VY")
    s.add_node_spring("B", kx=SA1_K)
    s.add_load_case("LC"); s.add_point_load("B", "LC", fx=SA1_P)
    return s


# ── S-A2 — cantilever on an elastic tip support (exact) ─────────────────────
SA2_L = 3.0
SA2_P = 30.0
SA2_KB = 3.0 * E * I_BAR / SA2_L ** 3    # tip stiffness of the cantilever
SA2_K = SA2_KB                           # spring chosen equal → δ = PL³/6EI
SA2_D = SA2_P / (SA2_KB + SA2_K)
SA2_M = (SA2_P - SA2_K * SA2_D) * SA2_L  # = P·L/2

def build_s_a2():
    """Cantilever L, tip load P, vertical spring ky at the tip.
    δ = P/(k + 3EI/L³); M_encastramento = (P − k·δ)·L. With k = 3EI/L³ the
    answer is exactly δ = PL³/(6EI) and M = PL/2."""
    s = _new(); _beam_material(s)
    s.add_node("A", 0, 0); s.add_node("B", SA2_L, 0)
    s.add_bar_element("E", "A", "B", "S")
    s.add_support("FIX", ux=True, uy=True, tz=True); s.assign_support("A", "FIX")
    s.add_node_spring("B", ky=SA2_K)
    s.add_load_case("LC"); s.add_point_load("B", "LC", fy=-SA2_P)
    return s


# ── S-A3 — beam with rotational springs at both ends (exact) ────────────────
SA3_L = 6.0
SA3_Q = 20.0
SA3_KT = 2.0 * E * I_BAR / SA3_L         # → exactly half of the clamped moment
SA3_M_END = SA3_Q * SA3_L ** 2 / 12.0 / (1.0 + 2.0 * E * I_BAR / (SA3_KT * SA3_L))
SA3_M_MID = SA3_Q * SA3_L ** 2 / 8.0 - SA3_M_END
SA3_THETA = SA3_M_END / SA3_KT
SA3_D = (5 * SA3_Q * SA3_L ** 4 / (384 * E * I_BAR)
         - SA3_M_END * SA3_L ** 2 / (8 * E * I_BAR))

def build_s_a3():
    """Simply supported beam, UDL q, rotational springs kt at both supports.
    M_apoio = qL²/12 · 1/(1 + 2EI/(kt·L)); M_meio = qL²/8 − M_apoio;
    θ_apoio = M_apoio/kt; δ_meio = 5qL⁴/384EI − M_apoio·L²/8EI.
    Limits: kt→∞ gives the clamped beam, kt→0 the simply supported one."""
    s = _new(); _beam_material(s)
    s.add_node("A", 0, 0); s.add_node("C", SA3_L / 2, 0); s.add_node("B", SA3_L, 0)
    s.add_bar_element("E1", "A", "C", "S"); s.add_bar_element("E2", "C", "B", "S")
    s.add_support("PIN", ux=True, uy=True); s.add_support("ROL", ux=False, uy=True)
    s.assign_support("A", "PIN"); s.assign_support("B", "ROL")
    s.add_node_spring("A", kt=SA3_KT); s.add_node_spring("B", kt=SA3_KT)
    s.add_load_case("LC")
    s.add_distributed_load("E1", "LC", fye=-SA3_Q, fyd=-SA3_Q); s.add_distributed_load("E2", "LC", fye=-SA3_Q, fyd=-SA3_Q)
    return s


# ── S-A4 — rigid body carried only by node springs (exact) ──────────────────
# No supports at all: the springs alone make K non-singular. A rigid bar on two
# vertical springs is statically determinate as a rigid body, so the spring
# forces follow from statics and the settlements from w = R/k. The horizontal
# load shares the node with the vertical one and is taken by kx alone — which
# also verifies that the node-spring block is diagonal (no x–y coupling).
SA4_L = 4.0
SA4_A = 1.0        # load abscissa
SA4_P = 100.0      # kN, downward
SA4_H = 40.0       # kN, horizontal
SA4_K = 5.0e4      # kN/m, each vertical spring
SA4_KX = 2.0e4     # kN/m, horizontal spring at A
SA4_RA = SA4_P * (SA4_L - SA4_A) / SA4_L
SA4_RB = SA4_P * SA4_A / SA4_L
SA4_WA = -SA4_RA / SA4_K
SA4_WB = -SA4_RB / SA4_K
SA4_UX = SA4_H / SA4_KX

def build_s_a4():
    """Rigid bar A(0)–C(1)–B(4) hanging on springs only: ky at A and B, kx at A.
    R_A = P(L−a)/L, R_B = P·a/L, w = −R/k, u_x = H/kx (uniform, rigid bar)."""
    s = _new(); _beam_material(s); _rigid_section(s)
    s.add_node("A", 0, 0); s.add_node("C", SA4_A, 0); s.add_node("B", SA4_L, 0)
    s.add_bar_element("E1", "A", "C", "R"); s.add_bar_element("E2", "C", "B", "R")
    s.add_node_spring("A", kx=SA4_KX, ky=SA4_K)
    s.add_node_spring("B", ky=SA4_K)
    s.add_load_case("LC")
    s.add_point_load("C", "LC", fx=SA4_H, fy=-SA4_P)
    return s


# ── S-B1 — rigid block on a Winkler foundation (uniform state, exact) ───────
# The "patch test" of the element spring: w = q/k for ANY mesh, because the
# lumped k·L/2 reproduces the resultant exactly. The residual bending moment is
# pure lumping artefact and must decay as O(h²).
SB1_L = 6.0
SB1_Q = 100.0      # kN/m, downward
SB1_K = 5.0e4      # kN/m per m of length
SB1_W = -SB1_Q / SB1_K

def _build_s_b1(n):
    s = _new(); _beam_material(s); _rigid_section(s)
    for i in range(n + 1):
        s.add_node(f"N{i}", SB1_L * i / n, 0.0)
    s.add_support("RX", ux=True, uy=False)
    s.assign_support("N0", "RX")                    # only removes the x rigid body
    s.add_load_case("LC")
    for i in range(n):
        eid = f"E{i}"
        s.add_bar_element(eid, f"N{i}", f"N{i+1}", "R")
        s.add_element_spring(eid, ky=SB1_K)
        s.add_distributed_load(eid, "LC", fye=-SB1_Q, fyd=-SB1_Q)
    return s

def build_s_b1():    return _build_s_b1(4)
def build_s_b1_c():  return _build_s_b1(2)
def build_s_b1_m():  return _build_s_b1(4)
def build_s_b1_f():  return _build_s_b1(8)

SB1_MESHES = [("s-b1-2", build_s_b1_c, "2"),
              ("s-b1-4", build_s_b1_m, "4"),
              ("s-b1-8", build_s_b1_f, "8")]


# ── S-B2 / S-B3 — beam on elastic foundation (convergence) ─────────────────
SBF_K = 4.0e4                                  # kN/m²  (kN/m per m)
SBF_LAMBDA = (SBF_K / (4.0 * E * I_BAR)) ** 0.25
SBF_L = 20.0                                   # λ·L/2 ≈ 5.7 → "long" beam
SBF_P = 100.0

# Hetényi, *Beams on Elastic Foundation*:
SB2_W0 = SBF_P * SBF_LAMBDA / (2.0 * SBF_K)         # infinite beam, central P
SB2_M0 = SBF_P / (4.0 * SBF_LAMBDA)
SB3_W0 = 2.0 * SBF_P * SBF_LAMBDA / SBF_K           # semi-infinite, end P
SB3_MMAX = 0.3224 * SBF_P / SBF_LAMBDA              # at λx = π/4

def _build_winkler(n, at_end):
    """Beam of length L on a Winkler foundation k, free ends, point load P at
    midspan (``at_end=False``) or at the left end (``at_end=True``)."""
    s = _new(); _beam_material(s)
    for i in range(n + 1):
        s.add_node(f"N{i}", SBF_L * i / n, 0.0)
    s.add_support("RX", ux=True, uy=False)
    s.assign_support("N0", "RX")
    s.add_load_case("LC")
    for i in range(n):
        eid = f"E{i}"
        s.add_bar_element(eid, f"N{i}", f"N{i+1}", "S")
        s.add_element_spring(eid, ky=SBF_K)
    s.add_point_load("N0" if at_end else f"N{n // 2}", "LC", fy=-SBF_P)
    return s

def build_s_b2_c(): return _build_winkler(20, False)
def build_s_b2_m(): return _build_winkler(40, False)
def build_s_b2_f(): return _build_winkler(80, False)
def build_s_b3_c(): return _build_winkler(20, True)
def build_s_b3_m(): return _build_winkler(40, True)
def build_s_b3_f(): return _build_winkler(80, True)

SB2_MESHES = [("s-b2-20", build_s_b2_c, "20"),
              ("s-b2-40", build_s_b2_m, "40"),
              ("s-b2-80", build_s_b2_f, "80")]
SB3_MESHES = [("s-b3-20", build_s_b3_c, "20"),
              ("s-b3-40", build_s_b3_m, "40"),
              ("s-b3-80", build_s_b3_f, "80")]

def winkler_w0(res, struc):
    """Deflection (m, magnitude) under the load — midspan for B2, x=0 for B3."""
    nid = node_at(struc, SBF_L / 2.0, 0.0)
    return abs(uy(res, CASE, nid))

def winkler_w_end(res, struc):
    return abs(uy(res, CASE, node_at(struc, 0.0, 0.0)))


# ── S-B5 — element spring in LOCAL axes on an inclined bar (exact) ──────────
# A rigid bar at 45° on a local (axial ka / transverse kt) foundation. As a
# rigid body it must satisfy [gxx gxy; gxy gyy]·(u,v) = (qx, qy) with the
# rotated block gxx = ka c² + kt s², gxy = (ka−kt) c s, gyy = ka s² + kt c².
# For ka = kt the block is isotropic and the model must equal the 'global' one.
SB5_KA = 8.0e4
SB5_KT = 2.0e4
SB5_Q = 100.0                    # kN/m, downward, per unit length of the bar
SB5_LEN = 4.0
_SB5_C = _SB5_S = math.sqrt(0.5)
SB5_GXX = SB5_KA * _SB5_C ** 2 + SB5_KT * _SB5_S ** 2
SB5_GXY = (SB5_KA - SB5_KT) * _SB5_C * _SB5_S
SB5_GYY = SB5_KA * _SB5_S ** 2 + SB5_KT * _SB5_C ** 2
_SB5_DET = SB5_GXX * SB5_GYY - SB5_GXY ** 2
SB5_U = (SB5_GXY * SB5_Q) / _SB5_DET            # solves G·(u,v) = (0, −q)
SB5_V = (-SB5_GXX * SB5_Q) / _SB5_DET

SB5_K_ISO = 5.0e4                # isotropic case: local ≡ global, w = q/k

def _build_s_b5(kx, ky, coord_sys):
    s = _new(); _beam_material(s); _rigid_section(s)
    d = SB5_LEN / math.sqrt(2.0)
    s.add_node("A", 0.0, 0.0); s.add_node("B", d, d)
    s.add_bar_element("E", "A", "B", "R")
    s.add_element_spring("E", kx=kx, ky=ky, coord_sys=coord_sys)
    s.add_load_case("LC")
    s.add_distributed_load("E", "LC", fye=-SB5_Q, fyd=-SB5_Q)
    return s

def build_s_b5():
    """Anisotropic local foundation (ka ≠ kt) → x–y coupling."""
    return _build_s_b5(SB5_KA, SB5_KT, "local")

def build_s_b5_iso_local():
    """ka = kt → the rotated block is k·I: must equal the 'global' twin and
    give w = q/k with no horizontal drift, whatever the bar's inclination."""
    return _build_s_b5(SB5_K_ISO, SB5_K_ISO, "local")

def build_s_b5_iso_global():
    return _build_s_b5(SB5_K_ISO, SB5_K_ISO, "global")


# ── S-C — unilateral springs (NonLinear analysis cases) ────────────────────
# Linear cases always treat a spring as bilateral; only a NonLinear case honours
# mode_x / mode_y. Every case below carries both an 'NL' and a 'LIN' case, so a
# single model gives the unilateral answer AND its bilateral reference.
SC1_L = SA2_L
SC1_P = SA2_P
SC1_K = SA2_K
SC1_D_ACTIVE = SC1_P / (SC1_K + SA2_KB)                 # = PL³/6EI
SC1_D_FREE = SC1_P * SC1_L ** 3 / (3.0 * E * I_BAR)     # spring inactive

def _build_s_c1(mode, fy):
    s = _new(); _beam_material(s)
    s.add_node("A", 0, 0); s.add_node("B", SC1_L, 0)
    s.add_bar_element("E", "A", "B", "S")
    s.add_support("FIX", ux=True, uy=True, tz=True); s.assign_support("A", "FIX")
    s.add_node_spring("B", ky=SC1_K, mode_y=mode)
    s.add_load_case("LC"); s.add_point_load("B", "LC", fy=fy)
    s.add_analysis_case("NL", "NonLinear", {"LC": 1.0})
    s.add_analysis_case("LIN", "Linear", {"LC": 1.0})
    return s

def build_s_c1a(): return _build_s_c1("compression", -SC1_P)  # active
def build_s_c1b(): return _build_s_c1("compression", +SC1_P)  # lifts off
def build_s_c2a(): return _build_s_c1("tension", +SC1_P)      # active (tie)
def build_s_c2b(): return _build_s_c1("tension", -SC1_P)      # slack

def nl_uy(res, nid, case="NL"):
    return res["analysis_cases"][case]["displacements"][nid][1]

def nl_spring_fy(res, nid, case="NL"):
    return res["analysis_cases"][case]["spring_forces"][nid][1]


# ── S-C3 — rigid footing with partial contact (compression-only Winkler) ────
# The classic closed form: a rigid strip footing with an eccentric load, e>L/6,
# separates from the soil. Contact length a = 3(L/2 − e) measured from the
# loaded edge, triangular pressure with p_max = 2N/a, so w_max = p_max/k.
SC3_L = 6.0
SC3_N = 300.0        # kN, downward
SC3_E = 1.5          # m from the centre (> L/6 = 1.0 → uplift)
SC3_K = 5.0e4
SC3_A = 3.0 * (SC3_L / 2.0 - SC3_E)          # contact length = 4.5 m
SC3_PMAX = 2.0 * SC3_N / SC3_A               # 133.33 kN/m
SC3_WMAX = -SC3_PMAX / SC3_K                 # −2.667 mm at the loaded edge

def _build_s_c3(n):
    """Rigid bar of length L on a compression-only Winkler foundation, vertical
    load N at x = L/2 + e. n must place a node on the load."""
    s = _new(); _beam_material(s); _rigid_section(s)
    for i in range(n + 1):
        s.add_node(f"N{i}", SC3_L * i / n, 0.0)
    s.add_support("RX", ux=True, uy=False); s.assign_support("N0", "RX")
    s.add_load_case("LC")
    for i in range(n):
        eid = f"E{i}"
        s.add_bar_element(eid, f"N{i}", f"N{i+1}", "R")
        s.add_element_spring(eid, ky=SC3_K, mode_y="compression")
    s.add_point_load(node_at(s, SC3_L / 2.0 + SC3_E, 0.0), "LC", fy=-SC3_N)
    s.add_analysis_case("NL", "NonLinear", {"LC": 1.0})
    s.add_analysis_case("LIN", "Linear", {"LC": 1.0})
    return s

def build_s_c3_c(): return _build_s_c3(12)
def build_s_c3_m(): return _build_s_c3(24)
def build_s_c3_f(): return _build_s_c3(48)

SC3_MESHES = [("s-c3-12", build_s_c3_c, "12"),
              ("s-c3-24", build_s_c3_m, "24"),
              ("s-c3-48", build_s_c3_f, "48")]

def c3_contact_length(res, struc, case="NL"):
    """Length of the compressed (w < 0) zone, from the nodal displacements of a
    rigid body: the exact contact length is bracketed to within one element."""
    d = res["analysis_cases"][case]["displacements"]
    xs = sorted((n.x, d[n.id][1]) for n in struc.nodes.values())
    down = [x for x, w in xs if w < -1e-12]
    return (max(down) - min(down)) if len(down) > 1 else 0.0

def c3_w_edge(res, struc, case="NL"):
    return res["analysis_cases"][case]["displacements"][
        node_at(struc, SC3_L, 0.0)][1]


def cut_total(res, struc, cut_id, key, case=None):
    """|key| (N/V/M) of the combined bars+areas resultant at a named cut.
    Magnitude only — a cut's sign depends on which way it was drawn (see
    ``dev/CUT_PLAN.md``), so validation checks it against |analytical|."""
    from xdfem2d.cuts import cut_result
    cut = next(c for c in struc.cuts if c.id == cut_id)
    r = cut_result(struc, res, cut, case or CASE)
    return abs(r["resultant"]["total"][key])


# ── Quantity + case specifications ───────────────────────────────────────────
@dataclass
class Quantity:
    label: str
    unit: str
    analytical: float                       # engineering value (in `unit`)
    extract: Callable                       # (res, struc) -> value in SI (m, kN, kN/m²)
    to_unit: float = 1.0                    # multiply SI value to get `unit`
    rel_tol: float = 1.0e-3
    abs_tol: float = 1.0e-9
    signed: bool = True                     # compare signed, else magnitudes

    def measured(self, res, struc):
        return self.extract(res, struc) * self.to_unit


@dataclass
class Case:
    id: str
    title: str
    element: str
    action: str
    build: Callable
    quantities: list
    doc_table: int = -1                     # index of the docx comparison table
    note: str = ""


CASE = "LC"
MM = 1000.0        # m -> mm
MPA = 1.0 / 1000.0  # kN/m² -> MPa

CASES: list[Case] = [
    Case("2.1", "Viga simplesmente apoiada — carga pontual a meio vão",
         "Barra", "Carga pontual", build_2_1, doc_table=1, quantities=[
            Quantity("M a meio vão", "kN·m", 50.0 * 6.0 / 4.0,
                     lambda r, s: max_abs_M(r, CASE, "E1", "E2"), signed=False),
            Quantity("V junto ao apoio A", "kN", 25.0,
                     lambda r, s: max_abs_V(r, CASE, "E1"), signed=False),
            Quantity("Flecha a meio vão", "mm",
                     50.0 * 6.0 ** 3 / (48 * E * I_BAR) * MM,
                     lambda r, s: uy(r, CASE, "C"), to_unit=MM, signed=False),
         ]),
    Case("2.2", "Viga simplesmente apoiada — carga uniformemente distribuída",
         "Barra", "Carga distribuída", build_2_2, doc_table=2, quantities=[
            Quantity("M a meio vão", "kN·m", 20.0 * 6.0 ** 2 / 8.0,
                     lambda r, s: max_abs_M(r, CASE, "E1", "E2"), signed=False),
            Quantity("V junto ao apoio", "kN", 60.0,
                     lambda r, s: max_abs_V(r, CASE, "E1"), signed=False),
            Quantity("Flecha a meio vão", "mm",
                     5 * 20.0 * 6.0 ** 4 / (384 * E * I_BAR) * MM,
                     lambda r, s: uy(r, CASE, "C"), to_unit=MM, signed=False),
         ]),
    Case("2.3", "Consola — carga pontual na extremidade livre",
         "Barra (consola)", "Carga pontual", build_2_3, doc_table=3, quantities=[
            Quantity("M no encastramento", "kN·m", 30.0 * 3.0,
                     lambda r, s: abs(reaction(r, CASE, "A", 2)), signed=False),
            Quantity("V", "kN", 30.0,
                     lambda r, s: max_abs_V(r, CASE, "E"), signed=False),
            Quantity("Flecha na extremidade livre", "mm",
                     30.0 * 3.0 ** 3 / (3 * E * I_BAR) * MM,
                     lambda r, s: uy(r, CASE, "B"), to_unit=MM, signed=False),
            Quantity("Rotação na extremidade livre", "rad",
                     30.0 * 3.0 ** 2 / (2 * E * I_BAR),
                     lambda r, s: rot(r, CASE, "B"), signed=False),
         ]),
    Case("2.4", "Viga biencastrada — assentamento diferencial de apoio",
         "Barra", "Assentamento de apoio", build_2_4, doc_table=4, quantities=[
            Quantity("M em A", "kN·m", 6 * E * I_BAR * 0.010 / 5.0 ** 2,
                     lambda r, s: abs(reaction(r, CASE, "A", 2)), signed=False),
            Quantity("M em B", "kN·m", 6 * E * I_BAR * 0.010 / 5.0 ** 2,
                     lambda r, s: abs(reaction(r, CASE, "B", 2)), signed=False),
            Quantity("V", "kN", 12 * E * I_BAR * 0.010 / 5.0 ** 3,
                     lambda r, s: abs(reaction(r, CASE, "A", 1)), signed=False),
         ]),
    Case("2.5a", "Barra biencastrada — dilatação uniforme impedida",
         "Barra", "Variação de temperatura", build_2_5a, doc_table=5, quantities=[
            Quantity("N na barra biencastrada", "kN",
                     -E * A_BAR * ALPHA * 20.0,
                     lambda r, s: axial_N(r, CASE, "E")),
            Quantity("σ na barra biencastrada", "MPa",
                     -E * ALPHA * 20.0 * MPA,
                     lambda r, s: axial_N(r, CASE, "E") / A_BAR, to_unit=MPA),
         ]),
    Case("2.5b", "Viga isostática — gradiente térmico",
         "Barra", "Variação de temperatura", build_2_5b, doc_table=5, quantities=[
            Quantity("M na viga isostática", "kN·m", 0.0,
                     lambda r, s: max_abs_M(r, CASE, "E1", "E2"),
                     signed=False, abs_tol=1e-6),
            Quantity("Flecha a meio vão", "mm",
                     (ALPHA * 20.0 / 0.50) * 6.0 ** 2 / 8.0 * MM,
                     lambda r, s: uy(r, CASE, "C"), to_unit=MM, signed=False),
         ]),
    Case("3.1", "Patch test — tração uniforme",
         "CST", "Carga distribuída (tração)", build_3_1, doc_table=7, quantities=[
            Quantity("σ_x em qualquer elemento", "MPa", 1.0,
                     lambda r, s: max_abs_sx(r, CASE), to_unit=MPA),
            Quantity("σ_y em qualquer elemento", "MPa", 0.0,
                     lambda r, s: max(abs(d["sy"])
                                      for d in r["tri_stress"][CASE].values()),
                     to_unit=MPA, abs_tol=1e-6),
            Quantity("δ_x no bordo carregado", "mm",
                     (1000.0 / E) * 2.0 * MM,
                     lambda r, s: ux(r, CASE, node_at(s, 2.0, 0.0)),
                     to_unit=MM, signed=False),
            Quantity("δ_y no bordo superior", "mm",
                     -NU * (1000.0 / E) * 1.0 * MM,
                     lambda r, s: uy(r, CASE, node_at(s, 0.0, 1.0)), to_unit=MM),
         ]),
    Case("3.3", "Placa — deslocamento imposto num bordo (assentamento)",
         "CST", "Assentamento de apoio", build_3_3, doc_table=9, quantities=[
            Quantity("σ_x em qualquer elemento", "MPa",
                     E * (0.001 / 2.0) * MPA,
                     lambda r, s: max_abs_sx(r, CASE), to_unit=MPA),
            Quantity("Reação total no bordo encastrado", "kN",
                     E * (0.001 / 2.0) * (1.0 * 0.10),
                     lambda r, s: abs(sum(
                         rv[0] for nid, rv in r["reactions"][CASE].items()
                         if abs(s.nodes[nid].x) < 1e-9)),
                     signed=False),
         ], note="Apoio relaxado para ux-only (ver módulo)."),
    Case("3.4", "Placa livre — variação uniforme de temperatura",
         "CST", "Variação de temperatura", build_3_4, doc_table=10, quantities=[
            Quantity("σ_x em qualquer elemento", "MPa", 0.0,
                     lambda r, s: max_abs_sx(r, CASE), to_unit=MPA, abs_tol=1e-6),
            Quantity("σ_y em qualquer elemento", "MPa", 0.0,
                     lambda r, s: max(abs(d["sy"])
                                      for d in r["tri_stress"][CASE].values()),
                     to_unit=MPA, abs_tol=1e-6),
            Quantity("δ_x no bordo x = L", "mm", ALPHA * 30.0 * 2.0 * MM,
                     lambda r, s: ux(r, CASE, node_at(s, 2.0, 0.0)),
                     to_unit=MM, signed=False),
            Quantity("δ_y no bordo superior", "mm", ALPHA * 30.0 * 1.0 * MM,
                     lambda r, s: uy(r, CASE, node_at(s, 0.0, 1.0)),
                     to_unit=MM, signed=False),
         ]),

    # ── Geometry-object versions (not in the document) ──────────────────────
    Case("2.1a", "Viga por objeto — polilinha com carga a meio vão",
         "Barra (objeto)", "Carga pontual", build_2_1a, quantities=[
            Quantity("M a meio vão", "kN·m", 75.0,
                     lambda r, s: max_abs_M_all(r, CASE), signed=False),
            Quantity("V junto ao apoio", "kN", 25.0,
                     lambda r, s: max_abs_V_all(r, CASE), signed=False),
            Quantity("Flecha a meio vão", "mm",
                     50.0 * 6.0 ** 3 / (48 * E * I_BAR) * MM,
                     lambda r, s: uy(r, CASE, node_at(s, 3.0, 0.0)),
                     to_unit=MM, signed=False),
         ], note="Mesmo caso que 2.1, construído como objeto-linha."),
    Case("3.1a", "Placa por objeto — retângulo com tração de bordo",
         "CST (objeto)", "Carga distribuída (tração)", build_3_1a, quantities=[
            Quantity("σ_x em qualquer elemento", "MPa", 1.0,
                     lambda r, s: max_abs_sx(r, CASE), to_unit=MPA),
            Quantity("σ_y em qualquer elemento", "MPa", 0.0,
                     lambda r, s: max(abs(d["sy"])
                                      for d in r["tri_stress"][CASE].values()),
                     to_unit=MPA, abs_tol=1e-6),
            Quantity("δ_x no bordo carregado", "mm",
                     (1000.0 / E) * 2.0 * MM,
                     lambda r, s: ux(r, CASE, node_at(s, 2.0, 0.0)),
                     to_unit=MM, signed=False),
         ], note="Mesmo caso que 3.1, construído como objeto-superfície."),
    Case("arc", "Consola em quarto de círculo — carga na ponta",
         "Barra (arco)", "Carga pontual", build_arc, quantities=[
            Quantity("Flecha vertical na ponta", "mm",
                     math.pi * ARC_P * ARC_R ** 3 / (4 * E * I_BAR) * MM,
                     lambda r, s: uy(r, CASE, node_at(s, 0.0, ARC_R)),
                     to_unit=MM, signed=False, rel_tol=0.015),
            Quantity("Momento no encastramento", "kN·m", ARC_P * ARC_R,
                     lambda r, s: reaction_moment_sum(r, CASE), signed=False),
            Quantity("Reação vertical total", "kN", ARC_P,
                     lambda r, s: reaction_fy_sum(r, CASE), signed=False),
         ], note="δ_v = πPR³/(4EI), flexão dominante; FE converge de baixo."),
    Case("3.1-allman", "Patch de tração — elemento Allman (tz fixo)",
         "Allman", "Carga distribuída (tração)", build_3_1_allman, quantities=[
            Quantity("σ_x em qualquer elemento", "MPa", 1.0,
                     lambda r, s: max_abs_sx(r, CASE), to_unit=MPA),
            Quantity("σ_y em qualquer elemento", "MPa", 0.0,
                     lambda r, s: max(abs(d["sy"])
                                      for d in r["tri_stress"][CASE].values()),
                     to_unit=MPA, abs_tol=1e-6),
         ], note="Allman exato só com o DOF de drilling restringido."),
    Case("3.3-allman", "Deslocamento imposto — elemento Allman (tz fixo)",
         "Allman", "Assentamento de apoio", build_3_3_allman, quantities=[
            Quantity("σ_x em qualquer elemento", "MPa",
                     E * (0.001 / 2.0) * MPA,
                     lambda r, s: max_abs_sx(r, CASE), to_unit=MPA),
            Quantity("Reação total no bordo encastrado", "kN",
                     E * (0.001 / 2.0) * (1.0 * 0.10),
                     lambda r, s: abs(sum(
                         rv[0] for nid, rv in r["reactions"][CASE].items()
                         if abs(s.nodes[nid].x) < 1e-9)),
                     signed=False),
         ], note="Allman exato só com o DOF de drilling restringido."),
    Case("3.4-allman", "Placa livre, ΔT uniforme — elemento Allman",
         "Allman", "Variação de temperatura", build_3_4_allman, quantities=[
            Quantity("σ_x em qualquer elemento", "MPa", 0.0,
                     lambda r, s: max_abs_sx(r, CASE), to_unit=MPA, abs_tol=1e-6),
            Quantity("σ_y em qualquer elemento", "MPa", 0.0,
                     lambda r, s: max(abs(d["sy"])
                                      for d in r["tri_stress"][CASE].values()),
                     to_unit=MPA, abs_tol=1e-6),
            Quantity("δ_x no bordo x = L", "mm", ALPHA * 30.0 * 2.0 * MM,
                     lambda r, s: ux(r, CASE, node_at(s, 2.0, 0.0)),
                     to_unit=MM, signed=False),
            Quantity("δ_y no bordo superior", "mm", ALPHA * 30.0 * 1.0 * MM,
                     lambda r, s: uy(r, CASE, node_at(s, 0.0, 1.0)),
                     to_unit=MM, signed=False),
         ], note="Térmico do Allman (agora suportado): expansão livre σ=0."),

    # ── Molas — casos lineares exatos ───────────────────────────────────────
    Case("s-a1", "Mola axial em paralelo com a barra",
         "Barra + mola nodal", "Carga pontual", build_s_a1, quantities=[
            Quantity("δ_x na extremidade com mola", "mm", SA1_D * MM,
                     lambda r, s: ux(r, CASE, "B"), to_unit=MM, rel_tol=1e-9),
            Quantity("N na barra", "kN", SA1_N,
                     lambda r, s: axial_N(r, CASE, "E"), signed=False,
                     rel_tol=1e-9),
            Quantity("Força na mola", "kN", SA1_K * SA1_D,
                     lambda r, s: r["spring_forces"][CASE]["B"][0],
                     signed=False, rel_tol=1e-9),
            Quantity("Reação no encastramento", "kN", -SA1_N,
                     lambda r, s: reaction(r, CASE, "A", 0), rel_tol=1e-9),
         ], note="δ = P/(EA/L + k); mola e barra em paralelo."),
    Case("s-a2", "Consola com apoio elástico na extremidade",
         "Barra + mola nodal", "Carga pontual", build_s_a2, quantities=[
            Quantity("Flecha na extremidade", "mm", SA2_D * MM,
                     lambda r, s: uy(r, CASE, "B"), to_unit=MM,
                     signed=False, rel_tol=1e-9),
            Quantity("M no encastramento", "kN·m", SA2_M,
                     lambda r, s: abs(reaction(r, CASE, "A", 2)),
                     signed=False, rel_tol=1e-9),
            Quantity("Força na mola", "kN", SA2_K * SA2_D,
                     lambda r, s: r["spring_forces"][CASE]["B"][1],
                     signed=False, rel_tol=1e-9),
            Quantity("Reação vertical no encastramento", "kN",
                     SA2_P - SA2_K * SA2_D,
                     lambda r, s: reaction(r, CASE, "A", 1),
                     signed=False, rel_tol=1e-9),
         ], note="δ = P/(k + 3EI/L³); com k = 3EI/L³ dá δ = PL³/6EI, M = PL/2."),
    Case("s-a3", "Viga com molas de rotação nos apoios",
         "Barra + mola nodal", "Carga distribuída", build_s_a3, quantities=[
            Quantity("M no apoio (mola)", "kN·m", SA3_M_END,
                     lambda r, s: abs(r["spring_forces"][CASE]["A"][2]),
                     signed=False, rel_tol=1e-9),
            Quantity("M a meio vão", "kN·m", SA3_M_MID,
                     lambda r, s: max_abs_M(r, CASE, "E1", "E2"),
                     signed=False, rel_tol=1e-9),
            Quantity("Rotação no apoio", "rad", SA3_THETA,
                     lambda r, s: rot(r, CASE, "A"), signed=False, rel_tol=1e-9),
            Quantity("Flecha a meio vão", "mm", SA3_D * MM,
                     lambda r, s: uy(r, CASE, "C"), to_unit=MM,
                     signed=False, rel_tol=1e-9),
         ], note="M_apoio = qL²/12 · 1/(1+2EI/ktL); kt = 2EI/L → metade do encastramento."),
    Case("s-a4", "Corpo rígido suspenso apenas em molas nodais",
         "Barra rígida + molas", "Carga pontual", build_s_a4, quantities=[
            Quantity("Assentamento em A", "mm", SA4_WA * MM,
                     lambda r, s: uy(r, CASE, "A"), to_unit=MM, rel_tol=1e-6),
            Quantity("Assentamento em B", "mm", SA4_WB * MM,
                     lambda r, s: uy(r, CASE, "B"), to_unit=MM, rel_tol=1e-6),
            Quantity("Força na mola de A", "kN", -SA4_RA,
                     lambda r, s: r["spring_forces"][CASE]["A"][1], rel_tol=1e-6),
            Quantity("Força na mola de B", "kN", -SA4_RB,
                     lambda r, s: r["spring_forces"][CASE]["B"][1], rel_tol=1e-6),
            Quantity("δ_x (mola horizontal, sem acoplamento)", "mm", SA4_UX * MM,
                     lambda r, s: ux(r, CASE, "A"), to_unit=MM, rel_tol=1e-6),
         ], note="Sem apoios: só as molas tornam K não-singular. R = estática do corpo rígido."),
    Case("s-b1", "Bloco rígido sobre fundação Winkler — estado uniforme",
         "Barra rígida + mola de elemento", "Carga distribuída", build_s_b1,
         quantities=[
            Quantity("Assentamento (uniforme)", "mm", SB1_W * MM,
                     lambda r, s: uy(r, CASE, "N0"), to_unit=MM, rel_tol=1e-5),
            Quantity("Assentamento na outra extremidade", "mm", SB1_W * MM,
                     lambda r, s: uy(r, CASE, f"N{len(s.nodes)-1}"),
                     to_unit=MM, rel_tol=1e-5),
         ], note="w = q/k, em qualquer malha (o lumping conserva a resultante); o "
                 "resíduo ~10⁻⁵ é a rigidez finita do bloco 'rígido'."),
    Case("s-b5", "Mola de elemento em eixos locais — barra a 45°",
         "Barra rígida + mola de elemento", "Carga distribuída", build_s_b5,
         quantities=[
            Quantity("δ_x do corpo rígido", "mm", SB5_U * MM,
                     lambda r, s: ux(r, CASE, "A"), to_unit=MM, rel_tol=1e-6),
            Quantity("δ_y do corpo rígido", "mm", SB5_V * MM,
                     lambda r, s: uy(r, CASE, "A"), to_unit=MM, rel_tol=1e-6),
         ], note="G = R(θ)·diag(ka,kt)·R(θ)ᵀ; ka≠kt acopla x e y."),
]


# ── Molas unilaterais — casos NonLinear (fora do documento) ─────────────────
SPRING_NL_CASES: list[Case] = [
    Case("s-c1a", "Mola só-compressão activa (carga descendente)",
         "Barra + mola nodal", "Carga pontual (NL)", build_s_c1a, quantities=[
            Quantity("Flecha na extremidade (NL)", "mm", SC1_D_ACTIVE * MM,
                     lambda r, s: nl_uy(r, "B"), to_unit=MM,
                     signed=False, rel_tol=1e-9),
            Quantity("Flecha do caso Linear (referência)", "mm",
                     SC1_D_ACTIVE * MM,
                     lambda r, s: nl_uy(r, "B", "LIN"), to_unit=MM,
                     signed=False, rel_tol=1e-9),
         ], note="Comprimida → activa: NL ≡ Linear bilateral."),
    Case("s-c1b", "Mola só-compressão inactiva (carga ascendente)",
         "Barra + mola nodal", "Carga pontual (NL)", build_s_c1b, quantities=[
            Quantity("Flecha na extremidade (NL)", "mm", SC1_D_FREE * MM,
                     lambda r, s: nl_uy(r, "B"), to_unit=MM,
                     signed=False, rel_tol=1e-6),
            Quantity("Força na mola (NL)", "kN", 0.0,
                     lambda r, s: nl_spring_fy(r, "B"), abs_tol=1e-6),
            Quantity("Flecha do caso Linear (bilateral)", "mm",
                     SC1_D_ACTIVE * MM,
                     lambda r, s: nl_uy(r, "B", "LIN"), to_unit=MM,
                     signed=False, rel_tol=1e-9),
         ], note="Levanta → inactiva: δ = PL³/3EI, exatamente a consola sem mola."),
    Case("s-c2a", "Mola só-tração activa (tirante)",
         "Barra + mola nodal", "Carga pontual (NL)", build_s_c2a, quantities=[
            Quantity("Flecha na extremidade (NL)", "mm", SC1_D_ACTIVE * MM,
                     lambda r, s: nl_uy(r, "B"), to_unit=MM,
                     signed=False, rel_tol=1e-9),
         ], note="Tracionada → activa: NL ≡ Linear bilateral."),
    Case("s-c2b", "Mola só-tração inactiva (folga)",
         "Barra + mola nodal", "Carga pontual (NL)", build_s_c2b, quantities=[
            Quantity("Flecha na extremidade (NL)", "mm", SC1_D_FREE * MM,
                     lambda r, s: nl_uy(r, "B"), to_unit=MM,
                     signed=False, rel_tol=1e-6),
            Quantity("Força na mola (NL)", "kN", 0.0,
                     lambda r, s: nl_spring_fy(r, "B"), abs_tol=1e-6),
         ], note="Comprimida → folgada: mola sem força."),
]


# ── Constraints (multi-point / linear DOF coupling) ─────────────────────────
# Two closed-form benchmarks for the constraint engine. The current engine
# imposes constraints by penalty, so these are reproduced to ~1e-6 relative
# (not to machine precision, and with a tiny spurious reaction that keeps them
# out of the strict global-equilibrium check used for the other cases — see
# tests/tests_model/test_constraints_validation.py).
EI_BAR = E * I_BAR                       # 93 750 kN·m²

# cn-1 — rigid offset (eccentric load through a rigid link).
CN1_H = 3.0                              # column height, m
CN1_E = 1.0                             # horizontal offset of the slave, m
CN1_P = 40.0                            # vertical load at the offset node, kN
CN1_UX = CN1_E * CN1_P * CN1_H ** 2 / (2 * EI_BAR)   # top lateral disp, m
CN1_ROT = CN1_E * CN1_P * CN1_H / EI_BAR             # top rotation, rad
CN1_MB = CN1_E * CN1_P                               # base moment, kN·m

# cn-2 — equal-DOF load sharing between two identical cantilevers.
CN2_L = 3.0
CN2_P1, CN2_P2 = 20.0, 60.0
CN2_K = 3.0 * EI_BAR / CN2_L ** 3        # cantilever tip stiffness, kN/m
CN2_D = (CN2_P1 + CN2_P2) / (2.0 * CN2_K)            # shared tip deflection, m
CN2_R = (CN2_P1 + CN2_P2) / 2.0                      # base reaction each, kN


def build_cn_1():
    """Rigid offset: a vertical cantilever column whose top node (master) is
    tied by a rigid link to an offset slave node carrying the load. The rigid
    link turns the eccentric vertical load into an axial force plus a moment at
    the column top, exercising the full 3-DOF (ux, uy, tz) rigid transfer."""
    s = _new(); _beam_material(s)
    s.add_node("N0", 0.0, 0.0)
    s.add_node("N1", 0.0, CN1_H)          # column top — master
    s.add_node("N2", CN1_E, CN1_H)        # offset — slave
    s.add_bar_element("COL", "N0", "N1", "S")
    s.add_support("FIX", ux=True, uy=True, tz=True); s.assign_support("N0", "FIX")
    s.add_rigid_link("N1", ["N2"], id="RL")
    s.add_load_case(CASE); s.add_point_load("N2", CASE, fy=-CN1_P)
    return s


def build_cn_2():
    """Equal-DOF load sharing: two identical cantilevers whose tips are tied to
    the same vertical displacement. The tie puts the two tip springs in
    parallel, so the total load is shared equally regardless of how it is
    applied between them."""
    s = _new(); _beam_material(s)
    s.add_node("A1", 0.0, 0.0);  s.add_node("B1", CN2_L, 0.0)
    s.add_node("A2", 0.0, -1.0); s.add_node("B2", CN2_L, -1.0)
    s.add_bar_element("E1", "A1", "B1", "S")
    s.add_bar_element("E2", "A2", "B2", "S")
    s.add_support("FIX", ux=True, uy=True, tz=True)
    s.assign_support("A1", "FIX"); s.assign_support("A2", "FIX")
    s.add_equal_dof(["B1", "B2"], ["uy"], id="EQ")
    s.add_load_case(CASE)
    s.add_point_load("B1", CASE, fy=-CN2_P1)
    s.add_point_load("B2", CASE, fy=-CN2_P2)
    return s


CONSTRAINT_CASES: list[Case] = [
    Case("cn-1", "Ligação rígida — carga excêntrica por braço rígido",
         "Barra + ligação rígida", "Carga pontual", build_cn_1, quantities=[
            Quantity("Deslocamento horizontal no topo (u_x em N1)", "mm",
                     CN1_UX * MM,
                     lambda r, s: ux(r, CASE, "N1"), to_unit=MM, signed=False),
            Quantity("Rotação no topo (θ em N1)", "rad", CN1_ROT,
                     lambda r, s: rot(r, CASE, "N1"), signed=False),
            Quantity("Momento na base", "kN·m", CN1_MB,
                     lambda r, s: reaction_moment_sum(r, CASE), signed=False),
            Quantity("Rotação do nó escravo (θ em N2)", "rad", CN1_ROT,
                     lambda r, s: rot(r, CASE, "N2"), signed=False),
         ], note="A ligação rígida transfere a carga excêntrica como força "
                 "axial + momento; o nó escravo segue o corpo rígido do topo."),
    Case("cn-2", "Igualdade de DOF — repartição de carga entre duas consolas",
         "Barra + igualdade de DOF", "Carga pontual", build_cn_2, quantities=[
            Quantity("Flecha na ponta ligada (u_y em B1)", "mm", CN2_D * MM,
                     lambda r, s: uy(r, CASE, "B1"), to_unit=MM, signed=False),
            Quantity("Flecha na ponta ligada (u_y em B2)", "mm", CN2_D * MM,
                     lambda r, s: uy(r, CASE, "B2"), to_unit=MM, signed=False),
            Quantity("Reação vertical no apoio A1", "kN", CN2_R,
                     lambda r, s: reaction(r, CASE, "A1", 1), signed=False),
         ], note="A ligação u_y = u_y põe as duas pontas em paralelo: a carga "
                 "total reparte-se em partes iguais, seja qual for a sua "
                 "distribuição original."),
]


CUT_CASES: list[Case] = [
    Case("cut-1", "Corte de secção — viga s.a. sob carga distribuída (= 2.2)",
         "Corte (bars)", "Carga distribuída", build_cut_1, quantities=[
            Quantity("|V| no corte a meio vão (C_MID)", "kN", 0.0,
                     lambda r, s: cut_total(r, s, "C_MID", "V"),
                     signed=False, abs_tol=1.0e-6),
            Quantity("|M| no corte a meio vão (C_MID)", "kN·m",
                     20.0 * 6.0 ** 2 / 8.0,
                     lambda r, s: cut_total(r, s, "C_MID", "M"), signed=False),
            Quantity("|V| no corte a 1/4 de vão (C_Q)", "kN",
                     20.0 * 6.0 / 2.0 - 20.0 * 1.5,
                     lambda r, s: cut_total(r, s, "C_Q", "V"), signed=False),
            Quantity("|M| no corte a 1/4 de vão (C_Q)", "kN·m",
                     20.0 * 6.0 / 2.0 * 1.5 - 20.0 * 1.5 ** 2 / 2.0,
                     lambda r, s: cut_total(r, s, "C_Q", "M"), signed=False),
         ], note="C_MID cai sobre o nó C (partilhado por E1/E2); C_Q cai "
                 "dentro de E1. Ambos reproduzem V(x), M(x) da viga "
                 "isostática sob carga distribuída (caso 2.2), confirmando "
                 "que cut_result agrega corretamente através de um nó e por "
                 "interpolação dentro de um elemento."),
]


# ── Case 3.2 — convergence over three meshes (special handling) ──────────────
CASE_3_2_TARGET = 17.26        # mm, beam theory (Timoshenko), approximate
CASE_3_2_MESHES = [
    ("3.2-16x4", build_3_2_coarse, "16×4"),
    ("3.2-32x8", build_3_2_medium, "32×8"),
    ("3.2-64x16", build_3_2_fine, "64×16"),
]

def case_3_2_tip_deflection(res, struc):
    """Downward tip deflection (mm) at the loaded mid-height free node."""
    nid = node_at(struc, 4.0, 0.25)
    return abs(uy(res, CASE, nid)) * MM

def case_3_2_max_sx(res, struc):
    return max_abs_sx(res, CASE) * MPA

def case_3_2_max_txy(res, struc):
    return max_abs_txy(res, CASE) * MPA


# ── Case 3.2 with the Allman element — convergence, faster than CST ──────────
CASE_3_2_ALLMAN_MESHES = [
    ("3.2a-8x2", build_3_2a_coarse, "8×2"),
    ("3.2a-16x4", build_3_2a_medium, "16×4"),
    ("3.2a-32x8", build_3_2a_fine, "32×8"),
]

# ── Case 3.2 with the ES-FEM element — convergence, faster than CST ──────────
CASE_3_2_ESFEM_MESHES = [
    ("3.2e-8x2", build_3_2e_coarse, "8×2"),
    ("3.2e-16x4", build_3_2e_medium, "16×4"),
    ("3.2e-32x8", build_3_2e_fine, "32×8"),
]

# ── Arc — convergence of the tip deflection to the analytical value ─────────
ARC_TARGET = math.pi * ARC_P * ARC_R ** 3 / (4 * E * I_BAR) * MM   # mm
ARC_MESHES = [
    ("arc-16", build_arc_coarse, "16"),
    ("arc-32", build_arc_medium, "32"),
    ("arc", build_arc, "64"),
]

def arc_tip_deflection(res, struc):
    return abs(uy(res, CASE, node_at(struc, 0.0, ARC_R))) * MM


# ── Plate bending (out-of-plane): thin slab vs Navier, thick slab vs Mindlin ──
# A square slab simply supported on all edges under a uniform pressure. Two
# regimes and two elements: the thin plate against the classical Kirchhoff
# (Navier) series with the DKT element, and the thick plate against the exact
# first-order-shear (Mindlin) series with the shear-deformable MITC3 element —
# the case the DKT cannot represent, since it has no transverse-shear deflection.
PLATE_A = 5.0            # square side [m]
PLATE_PZ = -10.0         # uniform pressure [kN/m²]
PLATE_N = 32             # mesh divisions per side
PLATE_THIN_T = 0.05      # thin slab (span/thickness = 100)
PLATE_THICK_T = 0.50     # thick slab (span/thickness = 10)


def _plate_navier_thin_w(q, a, t):
    """Centre deflection of a hard-SS square Kirchhoff plate, Navier series."""
    D = E * t ** 3 / (12.0 * (1.0 - NU ** 2))
    w = 0.0
    for m in range(1, 80, 2):
        for k in range(1, 80, 2):
            w += (math.sin(m * math.pi / 2) * math.sin(k * math.pi / 2)) \
                / (m * k * (m * m / a ** 2 + k * k / a ** 2) ** 2)
    return w * 16.0 * q / (math.pi ** 6 * D)


def _plate_mindlin_w(q, a, t, kappa=5.0 / 6.0):
    """Centre deflection of a hard-SS square Mindlin (first-order-shear) plate."""
    D = E * t ** 3 / (12.0 * (1.0 - NU ** 2))
    G = E / (2.0 * (1.0 + NU))
    w = 0.0
    for m in range(1, 80, 2):
        for k in range(1, 80, 2):
            lam = math.pi ** 2 * (m * m / a ** 2 + k * k / a ** 2)
            w += (16.0 * q / (math.pi ** 2 * m * k)) / (D * lam ** 2) \
                * math.sin(m * math.pi / 2) * math.sin(k * math.pi / 2) \
                * (1.0 + lam * D / (kappa * G * t))
    return w


PLATE_THIN_W = _plate_navier_thin_w(abs(PLATE_PZ), PLATE_A, PLATE_THIN_T)     # m
PLATE_THICK_W = _plate_mindlin_w(abs(PLATE_PZ), PLATE_A, PLATE_THICK_T)       # m
# The Kirchhoff (thin-theory) deflection at the thick thickness, for the prose:
# it is ~4.5% below the Mindlin value — the shear the MITC3 adds and the DKT
# misses.
PLATE_THICK_KIRCHHOFF_W = _plate_navier_thin_w(abs(PLATE_PZ), PLATE_A,
                                               PLATE_THICK_T)


def _build_plate_ss(t, formulation, hard):
    from xdfem2d import Structure2D
    s = Structure2D(domain="plate")
    s.add_material("C", E, 0.0, alpha=ALPHA, poisson=NU)
    s.add_tri_section("T", "C", thickness=t, formulation=formulation)
    a, n = PLATE_A, PLATE_N
    ids = {}
    for j in range(n + 1):
        for i in range(n + 1):
            nid = f"n{i}_{j}"
            ids[(i, j)] = nid
            s.add_node(nid, a * i / n, a * j / n)
    e = 0
    for j in range(n):
        for i in range(n):
            aa, bb = ids[(i, j)], ids[(i + 1, j)]
            cc, dd = ids[(i + 1, j + 1)], ids[(i, j + 1)]
            s.add_tri_element(f"e{e}", aa, bb, cc, "T"); e += 1
            s.add_tri_element(f"e{e}", aa, cc, dd, "T"); e += 1
    # Soft SS restrains only w; hard SS also holds the edge-tangential rotation
    # (θx on the vertical edges, θy on the horizontal ones) — the support the
    # Navier / Mindlin series assumes.
    s.add_support("W", w=True)
    s.add_support("WX", w=True, tx=True)
    s.add_support("WY", w=True, ty=True)
    s.add_support("WXY", w=True, tx=True, ty=True)
    for (i, j), nid in ids.items():
        onv, onh = i in (0, n), j in (0, n)
        if not (onv or onh):
            continue
        if not hard:
            s.assign_support(nid, "W")
        elif onv and onh:
            s.assign_support(nid, "WXY")
        elif onv:
            s.assign_support(nid, "WX")
        else:
            s.assign_support(nid, "WY")
    s.add_load_case("LC")
    for tid in list(s.tri_elements_by_id):
        s.add_area_load(tid, "LC", pz=PLATE_PZ)
    return s


def build_plate_thin():
    """Thin SS square slab (t=0.05), DKT element, hard support vs Navier."""
    return _build_plate_ss(PLATE_THIN_T, "DKT", hard=True)


def build_plate_thick():
    """Thick SS square slab (t=0.50), MITC3 element, hard support vs Mindlin."""
    return _build_plate_ss(PLATE_THICK_T, "MITC3", hard=True)


def plate_centre_w(res, case, struc):
    """Centre deflection w [m] (stored in slot 0 of the plate DOFs)."""
    nid = node_at(struc, PLATE_A / 2.0, PLATE_A / 2.0)
    return abs(ux(res, case, nid))


PLATE_CASES: list[Case] = [
    Case("P.1", "Laje fina simplesmente apoiada — DKT vs série de Navier",
         "Placa (DKT)", "Pressão pz", build_plate_thin, quantities=[
            Quantity("Flecha central", "mm", PLATE_THIN_W * MM,
                     lambda r, s: plate_centre_w(r, CASE, s), to_unit=MM,
                     signed=False, rel_tol=1.0e-2),
         ]),
    Case("P.2", "Laje espessa simplesmente apoiada — MITC3 vs teoria de Mindlin",
         "Placa (MITC3)", "Pressão pz", build_plate_thick, quantities=[
            Quantity("Flecha central", "mm", PLATE_THICK_W * MM,
                     lambda r, s: plate_centre_w(r, CASE, s), to_unit=MM,
                     signed=False, rel_tol=1.5e-2),
         ]),
]


# ── Grillage (plate-domain bars): out-of-plane bending, torsion, load split ──
# A grillage bar carries out-of-plane bending (EI), transverse shear and
# St-Venant torsion (GJ). Three closed-form checks: a cantilever under a tip
# transverse load (bending), a cantilever under a tip torque (torsion GJ), and
# the textbook crossed two-beam grid (the load splits between the beams by
# stiffness). All are exact.
GRID_B, GRID_H = 0.30, 0.50
GRID_I = GRID_B * GRID_H ** 3 / 12.0
GRID_G = E / (2.0 * (1.0 + NU))
GRID_L = 4.0                 # cantilever length [m]
GRID_F = -10.0               # tip transverse load [kN]
GRID_T = 5.0                 # tip torque [kN·m]
GRID_LG = 6.0                # crossed-grid span [m]
GRID_P = -100.0              # crossed-grid central load [kN]
GRID_NSEG = 8

# The torsion constant J the engine derives for the rectangular section, so the
# analytical twist uses exactly what the element uses.
from xdfem2d.models import section_torsion_constant   # noqa: E402
_GRID_J = section_torsion_constant("rectangular", GRID_B, GRID_H)

# Analytical targets.
GRID_TIP_W = GRID_F * GRID_L ** 3 / (3.0 * E * GRID_I)          # FL³/3EI [m]
GRID_TIP_ROT = GRID_F * GRID_L ** 2 / (2.0 * E * GRID_I)        # FL²/2EI [rad]
GRID_TWIST = GRID_T * GRID_L / (GRID_G * _GRID_J)               # TL/GJ [rad]
GRID_CROSS_W = (GRID_P / 2.0) * GRID_LG ** 3 / (48.0 * E * GRID_I)


def _grid_cantilever(load):
    """Cantilever grillage bar along x, fixed at N0. *load* is a dict of tip
    point-load components (fz / mx / my)."""
    from xdfem2d import Structure2D
    s = Structure2D(domain="plate")
    s.add_material("C", E, 0.0, alpha=ALPHA, poisson=NU)
    s.add_section("B", "C", b=GRID_B, h=GRID_H)
    n = GRID_NSEG
    for k in range(n + 1):
        s.add_node(f"N{k}", GRID_L * k / n, 0.0)
    for k in range(n):
        s.add_bar_element(f"E{k}", f"N{k}", f"N{k + 1}", "B")
    s.add_support("FIX", w=True, tx=True, ty=True)
    s.assign_support("N0", "FIX")
    s.add_load_case("LC")
    s.add_point_load(f"N{n}", "LC", **load)
    return s


def build_grid_bending():
    return _grid_cantilever({"fz": GRID_F})


def build_grid_torsion():
    return _grid_cantilever({"mx": GRID_T})


def build_grid_crossed():
    """Two identical simply supported beams crossing at the centre, a central
    transverse load at the crossing."""
    from xdfem2d import Structure2D
    s = Structure2D(domain="plate")
    s.add_material("C", E, 0.0, alpha=ALPHA, poisson=NU)
    s.add_section("B", "C", b=GRID_B, h=GRID_H)
    c = GRID_LG / 2.0
    n = GRID_NSEG

    def line(prefix, p0, p1):
        ids = []
        for k in range(n + 1):
            t = k / n
            x, y = p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1])
            nid = "C" if (abs(x - c) < 1e-9 and abs(y - c) < 1e-9) else f"{prefix}{k}"
            if nid not in s.nodes:
                s.add_node(nid, x, y)
            ids.append(nid)
        for k in range(n):
            s.add_bar_element(f"{prefix}e{k}", ids[k], ids[k + 1], "B")
        return ids

    nx = line("X", (0.0, c), (GRID_LG, c))
    ny = line("Y", (c, 0.0), (c, GRID_LG))
    s.add_support("SS", w=True)
    for nid in (nx[0], nx[-1], ny[0], ny[-1]):
        s.assign_support(nid, "SS")
    s.add_load_case("LC")
    s.add_point_load("C", "LC", fz=GRID_P)
    return s


GRILLAGE_CASES: list[Case] = [
    Case("G.1", "Consola de grelha — carga transversal na ponta (flexão)",
         "Grelha (barra)", "Carga transversal Fz", build_grid_bending,
         quantities=[
            Quantity("Flecha na ponta", "mm", GRID_TIP_W * MM,
                     lambda r, s: ux(r, CASE, f"N{GRID_NSEG}"), to_unit=MM,
                     signed=False),
            Quantity("Rotação de flexão na ponta", "mrad", GRID_TIP_ROT * MM,
                     lambda r, s: rot(r, CASE, f"N{GRID_NSEG}"), to_unit=MM,
                     signed=False),
         ]),
    Case("G.2", "Consola de grelha — momento torsor na ponta (torção GJ)",
         "Grelha (barra)", "Momento torsor Mx", build_grid_torsion,
         quantities=[
            Quantity("Rotação de torção na ponta", "mrad", GRID_TWIST * MM,
                     lambda r, s: uy(r, CASE, f"N{GRID_NSEG}"), to_unit=MM,
                     signed=False),
         ]),
    Case("G.3", "Grelha de duas vigas cruzadas — repartição da carga central",
         "Grelha (barra)", "Carga central Fz", build_grid_crossed,
         quantities=[
            Quantity("Flecha no cruzamento", "mm", GRID_CROSS_W * MM,
                     lambda r, s: ux(r, CASE, "C"), to_unit=MM, signed=False),
         ]),
]


ALL_MODELS = (
    [(c.id, c.build) for c in CASES]
    + [(c.id, c.build) for c in PLATE_CASES]
    + [(c.id, c.build) for c in GRILLAGE_CASES]
    + [(c.id, c.build) for c in SPRING_NL_CASES]
    + [(c.id, c.build) for c in CONSTRAINT_CASES]
    + [(c.id, c.build) for c in CUT_CASES]
    + [(mid, b) for (mid, b, _lbl) in CASE_3_2_MESHES]
    + [(mid, b) for (mid, b, _lbl) in CASE_3_2_ALLMAN_MESHES]
    + [(mid, b) for (mid, b, _lbl) in CASE_3_2_ESFEM_MESHES]
    + [(mid, b) for (mid, b, _lbl) in ARC_MESHES if mid != "arc"]
    + [(mid, b) for (mid, b, _lbl) in SB1_MESHES if mid != "s-b1-4"]
    + [(mid, b) for (mid, b, _lbl) in SB2_MESHES]
    + [(mid, b) for (mid, b, _lbl) in SB3_MESHES]
    + [(mid, b) for (mid, b, _lbl) in SC3_MESHES]
    + [("s-b5-iso-local", build_s_b5_iso_local),
       ("s-b5-iso-global", build_s_b5_iso_global)]
)
