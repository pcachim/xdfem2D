"""
Example: simply supported square slab under uniform pressure (plate domain).

A DKT thin-plate model of a square slab, simply supported on all four edges,
under a uniform downward pressure. The result is checked against the Navier
double-series solution for a Kirchhoff plate.

Geometry:   L x L square, meshed n x n into DKT triangles
Support:    w = 0 along the whole boundary (simply supported)
Load:       pz = -10 kN/m^2 (downward)
Section:    thickness t = 0.20 m,  E = 33e6 kN/m^2,  nu = 0.2

The model's domain is 'plate', so the three nodal DOFs mean (w, theta_x,
theta_y) instead of (ux, uy, theta_z).
"""
import math
import os
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


def run():
    s = Structure2D(domain='plate')
    s.add_material('C30/37', elastic_modulus=E, unit_weight=25.0, poisson=NU)
    # Explicit DKT: this example checks against the *thin-plate* Navier series,
    # so it pins the Kirchhoff element (the default is now the shear-deformable
    # MITC3, which would add a small thick-plate correction).
    s.add_plate_section('SLAB', 'C30/37', thickness=T, formulation='DKT')
    s.add_load_case('LC')

    # n x n structured mesh of the square [0, L]^2.
    ids = {}
    for i in range(N + 1):
        for j in range(N + 1):
            nid = f'N{i}_{j}'
            ids[(i, j)] = nid
            s.add_node(nid, L * i / N, L * j / N)
    k = 0
    for i in range(N):
        for j in range(N):
            a, b, c, d = (ids[(i, j)], ids[(i + 1, j)],
                          ids[(i + 1, j + 1)], ids[(i, j + 1)])
            s.add_tri_element(f'T{k}', a, b, c, 'SLAB'); k += 1
            s.add_tri_element(f'T{k}', a, c, d, 'SLAB'); k += 1

    # Simply supported: restrain the transverse deflection w along the boundary.
    s.add_support('SS', w=True)
    for (i, j), nid in ids.items():
        if i in (0, N) or j in (0, N):
            s.assign_support(nid, 'SS')

    # Uniform pressure on every triangle.
    for tid in list(s.tri_elements_by_id):
        s.add_area_load(tid, 'LC', pz=Q)

    r = s.calculate()

    centre = ids[(N // 2, N // 2)]
    w_c = r['displacements']['LC'][centre][0]         # component 0 is w
    w_exact = _navier_centre()

    print("=== Simply supported square slab (DKT) ===")
    print(f"  mesh:            {N} x {N}  ({len(s.tri_elements)} triangles)")
    print(f"  centre w (MEF):  {w_c:.6e} m")
    print(f"  centre w (Navier): {w_exact:.6e} m")
    print(f"  error:           {abs(w_c - w_exact) / abs(w_exact):.2%}")

    # Peak sagging moment at the centre (mean of the triangles meeting there).
    tris = s.tri_elements_by_id
    mxs = [v['mx'] for tid, v in r['tri_stress']['LC'].items()
           if centre in (tris[tid].node_i, tris[tid].node_j, tris[tid].node_k)]
    print(f"  centre mx:       {sum(mxs) / len(mxs):+.3f} kNm/m")

    total_reaction = sum(v[0] for v in r['reactions']['LC'].values())
    print(f"  Sum Rz:          {total_reaction:+.3f} kN "
          f"(applied {-Q * L * L:+.3f} kN)")
    return r


if __name__ == '__main__':
    run()
