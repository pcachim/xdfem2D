"""
Example: thick simply supported square slab — the MITC3 shear-deformable plate.

The companion to example-slab.py. That one is a *thin* slab checked against the
Kirchhoff (Navier) series with the DKT element. This one is a *thick* slab
(span/thickness = 10) solved with the shear-deformable **MITC3** element and
checked against the exact first-order-shear (Mindlin) series — the deflection a
thin-plate element (DKT) cannot reproduce, because it has no transverse-shear
deformation.

Geometry:   L x L square, meshed n x n into MITC3 triangles
Support:    hard simple support (w = 0 and the edge-tangential rotation = 0),
            the condition the Navier/Mindlin series assumes
Load:       pz = -10 kN/m^2 (downward)
Section:    thickness t = 0.60 m (span/thickness = 10),  E = 33e6,  nu = 0.2

Run:  python examples/example-slab-mitc3.py
"""
import math
import os
import sys

_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(_ROOT, 'src'))   # run from a source checkout
sys.path.insert(0, _ROOT)

from xdfem2d import Structure2D

E, NU, T = 33e6, 0.2, 0.60
L, N, Q = 6.0, 24, -10.0
KAPPA = 5.0 / 6.0
D = E * T ** 3 / (12.0 * (1.0 - NU * NU))
G = E / (2.0 * (1.0 + NU))


def _navier_kirchhoff(terms=79):
    """Thin-plate (Kirchhoff) centre deflection — no shear."""
    w = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            w += (16.0 * (-Q) / (math.pi ** 6 * m * n
                  * (m ** 2 / L ** 2 + n ** 2 / L ** 2) ** 2)) / D \
                * math.sin(m * math.pi / 2) * math.sin(n * math.pi / 2)
    return -w


def _mindlin_centre(terms=79):
    """First-order-shear (Mindlin) centre deflection — bending + shear."""
    w = 0.0
    for m in range(1, terms + 1, 2):
        for n in range(1, terms + 1, 2):
            lam = math.pi ** 2 * (m ** 2 / L ** 2 + n ** 2 / L ** 2)
            w += (16.0 * (-Q) / (math.pi ** 2 * m * n)) / (D * lam ** 2) \
                * math.sin(m * math.pi / 2) * math.sin(n * math.pi / 2) \
                * (1.0 + lam * D / (KAPPA * G * T))
    return -w


def run():
    s = Structure2D(domain='plate')
    s.add_material('C30/37', elastic_modulus=E, unit_weight=25.0, poisson=NU)
    # MITC3 is the default; named here for clarity.
    s.add_plate_section('SLAB', 'C30/37', thickness=T, formulation='MITC3')
    s.add_load_case('LC')

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

    # Hard simple support: w = 0 everywhere on the boundary, plus the
    # edge-tangential rotation (theta_x on the vertical edges, theta_y on the
    # horizontal ones) — the boundary condition the analytical series assumes.
    s.add_support('W', w=True)
    s.add_support('WX', w=True, tx=True)
    s.add_support('WY', w=True, ty=True)
    s.add_support('WXY', w=True, tx=True, ty=True)
    for (i, j), nid in ids.items():
        onv, onh = i in (0, N), j in (0, N)
        if not (onv or onh):
            continue
        if onv and onh:
            s.assign_support(nid, 'WXY')
        elif onv:
            s.assign_support(nid, 'WX')
        else:
            s.assign_support(nid, 'WY')

    for tid in list(s.tri_elements_by_id):
        s.add_area_load(tid, 'LC', pz=Q)

    r = s.calculate()

    centre = ids[(N // 2, N // 2)]
    w_c = r['displacements']['LC'][centre][0]         # component 0 is w
    w_mindlin = _mindlin_centre()
    w_kirchhoff = _navier_kirchhoff()
    shear_pct = (w_mindlin / w_kirchhoff - 1.0) * 100.0

    print("=== Thick simply supported square slab (MITC3) ===")
    print(f"  mesh:              {N} x {N}  ({len(s.tri_elements)} triangles)")
    print(f"  span/thickness:    {L / T:.0f}")
    print(f"  centre w (MEF):    {w_c:.6e} m")
    print(f"  centre w (Mindlin):{w_mindlin:.6e} m   (exact, with shear)")
    print(f"  error vs Mindlin:  {abs(w_c - w_mindlin) / abs(w_mindlin):.2%}")
    print(f"  centre w (thin):   {w_kirchhoff:.6e} m   (Kirchhoff, no shear)")
    print(f"  shear contribution:+{shear_pct:.1f}%  "
          "(what MITC3 captures and DKT misses)")

    total_reaction = sum(v[0] for v in r['reactions']['LC'].values())
    applied = -Q * L * L
    print(f"  Sum Rz:            {total_reaction:+.3f} kN "
          f"(applied {applied:+.3f} kN)")


if __name__ == '__main__':
    run()
