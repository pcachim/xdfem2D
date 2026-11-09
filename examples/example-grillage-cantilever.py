"""
Example: cantilever grillage bar — out-of-plane bending and St-Venant torsion.

The companion to example-grillage.py (the crossed two-beam grid). This one is a
single cantilever grillage bar, checked against two closed-form beam solutions
that between them exercise everything a grillage member adds over a plane beam:

  * a transverse tip load Fz  -> tip deflection F·L^3/3EI and rotation F·L^2/2EI
  * a tip torque Mx           -> tip twist T·L/GJ  (St-Venant torsion GJ)

The model's domain is 'plate', so the bar bends out of plane and the three
nodal DOFs mean (w, theta_x, theta_y). A point load's components are the
transverse force fz, the torque mx (about the bar axis) and the bending moment
my.

Run:  python examples/example-grillage-cantilever.py
"""
import os
import sys

_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(_ROOT, 'src'))   # run from a source checkout
sys.path.insert(0, _ROOT)

from xdfem2d import Structure2D
from xdfem2d.models import section_torsion_constant

E, NU = 30e6, 0.2
B, H = 0.30, 0.50
L, NSEG = 4.0, 8
F, T = -10.0, 5.0          # tip transverse load [kN]; tip torque [kN·m]

I = B * H ** 3 / 12.0
G = E / (2.0 * (1.0 + NU))
J = section_torsion_constant("rectangular", B, H)   # the engine's own J


def _cantilever(load):
    """A cantilever grillage bar along x, fixed at N0, with a tip point load
    (a dict of fz / mx / my components)."""
    s = Structure2D(domain='plate')
    s.add_material('C30/37', elastic_modulus=E, unit_weight=0.0, poisson=NU)
    s.add_section('B', 'C30/37', b=B, h=H)          # J derived automatically
    for k in range(NSEG + 1):
        s.add_node(f'N{k}', L * k / NSEG, 0.0)
    for k in range(NSEG):
        s.add_bar_element(f'E{k}', f'N{k}', f'N{k + 1}', 'B')
    s.add_support('FIX', w=True, tx=True, ty=True)
    s.assign_support('N0', 'FIX')
    s.add_load_case('LC')
    s.add_point_load(f'N{NSEG}', 'LC', **load)
    s.add_analysis_case('LC', 'Linear', {'LC': 1.0})
    return s


def run():
    tip = f'N{NSEG}'

    # ── Bending: transverse tip load ────────────────────────────────────
    rb = _cantilever({'fz': F}).calculate()['analysis_cases']['LC']
    w = rb['displacements'][tip][0]                 # slot 0 is w
    rot_y = rb['displacements'][tip][2]             # slot 2 is theta_y
    w_exact = F * L ** 3 / (3.0 * E * I)
    rot_exact = F * L ** 2 / (2.0 * E * I)

    print("=== Cantilever grillage bar ===")
    print(f"  length L = {L:.0f} m, section {B:.2f} x {H:.2f} m, {NSEG} elements")
    print("  -- bending (tip load Fz = %.0f kN) --" % F)
    print(f"     tip w      (MEF): {w:+.6e} m")
    print(f"     tip w    (F L3/3EI): {w_exact:+.6e} m   "
          f"err {abs(w - w_exact) / abs(w_exact):.2%}")
    print(f"     tip rot_y  (MEF): {abs(rot_y):.6e} rad")
    print(f"     tip rot (F L2/2EI): {abs(rot_exact):.6e} rad   "
          f"err {abs(abs(rot_y) - abs(rot_exact)) / abs(rot_exact):.2%}")

    # ── Torsion: tip torque about the bar axis ──────────────────────────
    rt = _cantilever({'mx': T}).calculate()['analysis_cases']['LC']
    twist = rt['displacements'][tip][1]             # slot 1 is theta_x (twist)
    twist_exact = T * L / (G * J)
    print("  -- torsion (tip torque Mx = %.0f kN.m) --" % T)
    print(f"     J (St-Venant): {J:.6e} m^4,  G: {G:.4e} kN/m^2")
    print(f"     tip twist  (MEF): {abs(twist):.6e} rad")
    print(f"     tip twist (T L/GJ): {twist_exact:.6e} rad   "
          f"err {abs(abs(twist) - twist_exact) / twist_exact:.2%}")


if __name__ == '__main__':
    run()
