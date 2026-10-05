"""
Example: triangles vs. quads on the same simply supported square slab.

This is the "why quads" example. The same square slab, simply supported on all
four edges under a uniform pressure, is meshed twice on the *same* n×n grid:

  * with MITC3 **triangles** (each grid cell split by a diagonal), and
  * with MITC4 **quads** (one element per cell).

Both centre deflections are checked against the Kirchhoff Navier series. Then
the script measures the **directional (diagonal) bias** the triangle split
introduces: inside every interior cell the two triangles report *different*
moments even though the cell is a single square of slab — a checkerboard that
biases the moment field near supports and point loads. The quad mesh has one
element per cell, so that intra-cell spread is exactly zero.

Run:  python examples/example-slab-quad.py

Geometry:   L x L square, n x n grid
Support:    w = 0 along the whole boundary (simply supported)
Load:       pz = -10 kN/m^2 (downward)
Section:    thickness t = 0.20 m,  E = 33e6 kN/m^2,  nu = 0.2
"""
import math
import os
import statistics
import sys

_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(_ROOT, 'src'))   # run from a source checkout
sys.path.insert(0, _ROOT)

from xdfem2d import Structure2D

E, NU, T = 33e6, 0.2, 0.20
L, N, Q = 6.0, 12, -10.0
D = E * T ** 3 / (12.0 * (1.0 - NU * NU))


def _navier_centre(terms=39):
    """Exact centre deflection of a simply supported square plate (Navier)."""
    w = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            f = 16.0 * (-Q) / (math.pi ** 6 * m * n
                               * (m ** 2 / L ** 2 + n ** 2 / L ** 2) ** 2)
            w += f / D * math.sin(m * math.pi / 2) * math.sin(n * math.pi / 2)
    return -w


def _grid_nodes(s):
    ids = {}
    for i in range(N + 1):
        for j in range(N + 1):
            nid = f'N{i}_{j}'
            ids[(i, j)] = nid
            s.add_node(nid, L * i / N, L * j / N)
    return ids


def _simply_support(s, ids):
    s.add_support('SS', w=True)
    for (i, j), nid in ids.items():
        if i in (0, N) or j in (0, N):
            s.assign_support(nid, 'SS')


def _tri_slab():
    s = Structure2D(domain='plate')
    s.add_material('C30/37', elastic_modulus=E, unit_weight=25.0, poisson=NU)
    s.add_plate_section('SLAB', 'C30/37', thickness=T, formulation='MITC3')
    s.add_load_case('LC')
    ids = _grid_nodes(s)
    cells = {}                       # (i, j) -> [element ids in that cell]
    k = 0
    for i in range(N):
        for j in range(N):
            a, b, c, d = (ids[(i, j)], ids[(i + 1, j)],
                          ids[(i + 1, j + 1)], ids[(i, j + 1)])
            e1 = f'T{k}'; s.add_tri_element(e1, a, b, c, 'SLAB'); k += 1
            e2 = f'T{k}'; s.add_tri_element(e2, a, c, d, 'SLAB'); k += 1
            cells[(i, j)] = [e1, e2]
    _simply_support(s, ids)
    for tid in list(s.tri_elements_by_id):
        s.add_area_load(tid, 'LC', pz=Q)
    return s, ids, cells


def _quad_slab():
    s = Structure2D(domain='plate')
    s.add_material('C30/37', elastic_modulus=E, unit_weight=25.0, poisson=NU)
    s.add_quad_section('SLAB', 'C30/37', thickness=T, formulation='MITC4')
    s.add_load_case('LC')
    ids = _grid_nodes(s)
    cells = {}
    k = 0
    for i in range(N):
        for j in range(N):
            a, b, c, d = (ids[(i, j)], ids[(i + 1, j)],
                          ids[(i + 1, j + 1)], ids[(i, j + 1)])
            q = f'Q{k}'; s.add_quad_element(q, a, b, c, d, 'SLAB'); k += 1
            cells[(i, j)] = [q]
    _simply_support(s, ids)
    for qid in list(s.quad_elements_by_id):
        s.add_area_load(qid, 'LC', pz=Q)
    return s, ids, cells


def _intra_cell_moment_spread(res, cells):
    """Mean |mx_max - mx_min| over interior cells — the diagonal bias.

    Zero for quads (one element per cell). Non-zero for triangles: the two
    diagonally-split triangles of the same square report different moments.
    """
    ts = res['tri_stress']['LC']
    spreads = []
    for (i, j), elems in cells.items():
        if i in (0, N - 1) or j in (0, N - 1):      # skip boundary cells
            continue
        mx = [ts[e]['mx'] for e in elems if e in ts]
        if len(mx) >= 1:
            spreads.append(max(mx) - min(mx))
    return statistics.mean(spreads) if spreads else 0.0


def run():
    w_exact = _navier_centre()
    print("=== Simply supported square slab: triangles vs. quads ===")
    print(f"  mesh: {N}x{N} grid,  Navier centre w = {w_exact:.6e} m\n")

    for label, builder in (("MITC3 triangles", _tri_slab),
                           ("MITC4 quads", _quad_slab)):
        s, ids, cells = builder()
        r = s.calculate()
        centre = ids[(N // 2, N // 2)]
        w_c = r['displacements']['LC'][centre][0]
        n_el = len(getattr(s, 'quad_elements', []) or []) + len(s.tri_elements)
        spread = _intra_cell_moment_spread(r, cells)
        print(f"  {label}:  {n_el} elements")
        print(f"    centre w   : {w_c:.6e} m   "
              f"(error {abs(w_c - w_exact) / abs(w_exact):.2%})")
        print(f"    diagonal bias (mean intra-cell mx spread): "
              f"{spread:.4f} kNm/m")
    print("\n  The quad mesh's intra-cell spread is 0 by construction — no "
          "diagonal to bias the moment field.")


if __name__ == '__main__':
    run()
