"""
Example: section cuts on a simply supported beam under a uniform load.

Geometry:   N1 ----6m---- N2
Supports:   N1 pinned (UX+UY fixed), N2 roller (UY fixed)
Load:       10 kN/m uniform downward on element E1
Material:   E = 30e6 kN/m2, unit_weight = 25 kN/m3
Section:    b = 0.3 m, h = 0.6 m

Two cuts are drawn across the beam, perpendicular to it:
  - C_MID at x=3 (midspan)   -- analytical: V=0,        M=45.0 kNm (wL^2/8)
  - C_SUP at x=0.5 (near N1) -- analytical: V=w*L/2-w*x, M=w*L/2*x-w*x^2/2

See dev/CUT_PLAN.md for the cut API and sign conventions (a cut reports the
internal forces the "positive side" -- the side its normal points to --
exerts on the negative side, which is why the printed sign here depends on
which way each cut was drawn, not just on the beam's own diagram).
"""
import os
import sys

_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(_ROOT, 'src'))   # run from a source checkout
sys.path.insert(0, _ROOT)

from xdfem2d import Structure2D
from xdfem2d.cuts import cut_result


def run():
    struc = Structure2D()

    struc.add_node('N1', 0.0, 0.0)
    struc.add_node('N2', 6.0, 0.0)

    struc.add_material('C30', elastic_modulus=30e6, unit_weight=25.0)
    struc.add_section('S1', material_name='C30', b=0.3, h=0.6)
    struc.add_bar_element('E1', node_i='N1', node_j='N2', section_name='S1')

    struc.add_support('PIN', ux=True, uy=True, tz=False)
    struc.add_support('ROLLER', ux=False, uy=True, tz=False)
    struc.assign_support('N1', 'PIN')
    struc.assign_support('N2', 'ROLLER')

    struc.add_load_case('LC1', self_weight_factor=0.0)
    struc.add_distributed_load('E1', 'LC1', fye=-10.0, fyd=-10.0)

    # A cut is a plain segment crossing the model -- here, two short vertical
    # segments straddling the beam at the stations we want to report.
    struc.add_cut('C_MID', 3.0, -1.0, 3.0, 1.0, name='Midspan')
    struc.add_cut('C_SUP', 0.5, -1.0, 0.5, 1.0, name='Near support')

    results = struc.calculate()

    # Every load case has a twin analysis case sharing its id -- that twin is
    # what carries the diagrams cut_result reads.
    case = 'LC1'

    print("=== Cut results ===")
    for cut in struc.cuts:
        r = cut_result(struc, results, cut, case)
        total = r['resultant']['total']
        print(f"\n{cut.name} ({cut.id}) -- domain={r['domain']}")
        if r['reason']:
            print(f"  (reason: {r['reason']})")
        for group in ('bars', 'areas', 'total'):
            vals = r['resultant'][group]
            line = ", ".join(f"{k}={v:.3f}" for k, v in vals.items())
            print(f"  {group.capitalize():6s}: {line}")

    w, L = 10.0, 6.0
    print("\n=== Analytical cross-check (magnitudes only -- the printed sign "
          "above depends on how each cut was drawn, see the module docstring) ===")
    print(f"  Midspan:      |V|=0.000, |M|={w * L ** 2 / 8:.3f} kNm")
    x = 0.5
    v_sup = abs(w * L / 2 - w * x)
    m_sup = abs(w * L / 2 * x - w * x ** 2 / 2)
    print(f"  Near support: |V|={v_sup:.3f} kN, |M|={m_sup:.3f} kNm")


if __name__ == '__main__':
    run()
