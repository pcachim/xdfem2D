"""Quad-element validation — standalone, runnable regression.

Closed-form benchmarks for the 4-node quad elements, the counterpart of the
triangle validation cases. Each check prints its FE value, the analytical value
and PASS/FAIL. The same checks run as engine unit tests
(tests/tests_engine/test_quad_elements*.py); this script gathers them as a
single documented, human-readable validation pass.

Run:  python validation/validate_quads.py

Units: m, kN, kN/m² (so E = 30e6 kN/m²).
"""
import math
import os
import sys

_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(_ROOT, 'src'))
sys.path.insert(0, _ROOT)

from xdfem2d import Structure2D

_results = []


def _check(name, fe, exact, tol, unit=""):
    err = abs(fe - exact) / abs(exact) if exact else abs(fe)
    ok = err <= tol
    _results.append(ok)
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    print(f"         FE = {fe:.6g} {unit}   exact = {exact:.6g} {unit}   "
          f"error = {err:.3%}  (tol {tol:.1%})")


# ── 1. QM6 membrane — constant uniaxial stress patch (distorted quad) ─────────
def patch_qm6():
    """A QM6 rectangle in uniaxial tension recovers the exact constant stress
    σx = fx / T (a uniform edge traction fx [kN/m] over the thickness T). This
    is the constant-stress patch a valid incompatible-mode element must pass;
    the harder distorted-quad patch is checked at the kernel level in
    tests/tests_engine/test_quad_elements.py."""
    E, NU, T, S0 = 30e6, 0.2, 0.5, 1000.0    # target σx = S0 kN/m²
    fx = S0 * T                              # edge traction [kN/m]
    s = Structure2D(domain='plane')
    s.add_material('M', elastic_modulus=E, unit_weight=0.0, poisson=NU)
    s.add_quad_section('Q', 'M', thickness=T, formulation='QM6')
    # 2 x 1 rectangle, one element; left edge held in x (free to contract in y).
    s.add_node('n1', 0.0, 0.0); s.add_node('n2', 2.0, 0.0)
    s.add_node('n3', 2.0, 1.0); s.add_node('n4', 0.0, 1.0)
    s.add_quad_element('Q1', 'n1', 'n2', 'n3', 'n4', 'Q')
    s.add_support('ROLX', ux=True, uy=False)      # left edge: ux = 0
    s.assign_support('n1', 'ROLX'); s.assign_support('n4', 'ROLX')
    s.add_support('PIN', ux=True, uy=True)         # one corner pins rigid body
    s.assign_support('n1', 'PIN')
    s.add_load_case('LC')
    s.add_quad_edge_load('EL', 'Q1', 'n2', 'n3', 'LC', fx=fx)   # right edge
    r = s.calculate()
    sx = r['tri_stress']['LC']['Q1']['sx']
    _check("QM6 constant-stress patch (uniaxial), σx", sx, S0, 0.01, "kN/m²")


# ── 2. MITC4 slab — SS square plate vs Timoshenko table ───────────────────────
def ss_plate_mitc4():
    """Simply supported square plate under uniform load: centre deflection vs.
    the Timoshenko/Woinowsky-Krieger table value w = α q a⁴ / D, α = 0.00406
    (ν = 0.3)."""
    E, NU, T, Q, A, N = 30e6, 0.3, 0.1, -10.0, 4.0, 16
    D = E * T ** 3 / (12.0 * (1.0 - NU * NU))
    w_exact = -0.00406 * abs(Q) * A ** 4 / D
    s = Structure2D(domain='plate')
    s.add_material('M', elastic_modulus=E, unit_weight=0.0, poisson=NU)
    s.add_quad_section('S', 'M', thickness=T, formulation='MITC4')
    s.add_load_case('LC')
    ids = {}
    for i in range(N + 1):
        for j in range(N + 1):
            nid = f'N{i}_{j}'; ids[(i, j)] = nid
            s.add_node(nid, A * i / N, A * j / N)
    k = 0
    for i in range(N):
        for j in range(N):
            a, b, c, d = (ids[(i, j)], ids[(i + 1, j)],
                          ids[(i + 1, j + 1)], ids[(i, j + 1)])
            s.add_quad_element(f'Q{k}', a, b, c, d, 'S'); k += 1
    s.add_support('SS', w=True)
    for (i, j), nid in ids.items():
        if i in (0, N) or j in (0, N):
            s.assign_support(nid, 'SS')
    for qid in list(s.quad_elements_by_id):
        s.add_area_load(qid, 'LC', pz=Q)
    r = s.calculate()
    w_c = r['displacements']['LC'][ids[(N // 2, N // 2)]][0]
    _check("MITC4 SS square plate, centre w vs Timoshenko", w_c, w_exact, 0.03,
           "m")


# ── 3. QM6 cantilever — pure bending vs Euler beam ────────────────────────────
def cantilever_qm6():
    """A slender cantilever meshed one QM6 element deep under a pure tip moment:
    tip deflection vs. Euler-Bernoulli δ = M L² / (2 E I). QM6's incompatible
    modes remove the bending lock plain Q4 suffers."""
    E, NU, T = 30e6, 0.0, 0.1     # ν=0 so plane stress ≡ beam
    Lx, Hy, N = 10.0, 1.0, 10
    M = 100.0                     # tip moment [kNm] as a force couple
    I = T * Hy ** 3 / 12.0
    d_exact = M * Lx ** 2 / (2.0 * E * I)
    s = Structure2D(domain='plane')
    s.add_material('M', elastic_modulus=E, unit_weight=0.0, poisson=NU)
    s.add_quad_section('Q', 'M', thickness=T, formulation='QM6')
    s.add_load_case('LC')
    # One element through the depth, N along the length.
    top = {}; bot = {}
    for i in range(N + 1):
        x = Lx * i / N
        bot[i] = f'B{i}'; s.add_node(bot[i], x, 0.0)
        top[i] = f'T{i}'; s.add_node(top[i], x, Hy)
    for i in range(N):
        s.add_quad_element(f'Q{i}', bot[i], bot[i + 1], top[i + 1], top[i], 'Q')
    s.add_support('FIX', ux=True, uy=True)
    s.assign_support(bot[0], 'FIX'); s.assign_support(top[0], 'FIX')
    # Tip moment as a couple: +F on top node, -F on bottom node, F = M / Hy.
    F = M / Hy
    s.add_point_load(top[N], 'LC', fx=F)
    s.add_point_load(bot[N], 'LC', fx=-F)
    r = s.calculate()
    d_fe = r['displacements']['LC'][top[N]][1]   # transverse (uy) tip defl.
    # Compare magnitudes (sign depends on couple orientation).
    _check("QM6 cantilever pure bending, tip δ vs Euler", abs(d_fe), d_exact,
           0.02, "m")


def run():
    print("=== Quad element validation ===\n")
    patch_qm6()
    ss_plate_mitc4()
    cantilever_qm6()
    n_ok = sum(_results)
    print(f"\n  {n_ok}/{len(_results)} checks passed.")
    return n_ok == len(_results)


if __name__ == '__main__':
    sys.exit(0 if run() else 1)
