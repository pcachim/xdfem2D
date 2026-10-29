"""
Example: crossed-beam grillage (plate domain).

Two identical, simply supported beams crossing at the centre, sharing the middle
node — the textbook grillage. A downward point load at the crossing splits
between the two beams by their relative stiffness; for equal beams and equal
spans it is a clean 50 / 50, and each beam then behaves as a simply supported
beam carrying half the load, with centre deflection (P/2)L^3 / 48EI.

The model's domain is 'plate', so a bar element is a grillage beam (bending EI
plus St-Venant torsion GJ/L) and the three nodal DOFs mean (w, theta_x,
theta_y). A point load's first component is the transverse force fz.
"""
import os
import sys

_ROOT = os.path.join(os.path.dirname(__file__), '..')
sys.path.insert(0, os.path.join(_ROOT, 'src'))   # run from a source checkout
sys.path.insert(0, _ROOT)

from xdfem2d import Structure2D

E, NU = 33e6, 0.2
L, P = 6.0, -100.0        # span of each beam [m]; central load [kN]
NSEG = 8                  # elements per half-beam


def run():
    s = Structure2D(domain='plate')
    s.add_material('C30/37', elastic_modulus=E, unit_weight=25.0, poisson=NU)
    s.add_section('B', 'C30/37', b=0.30, h=0.50)      # J derived automatically
    s.add_load_case('LC')

    # Beam X runs along y = L/2, beam Y along x = L/2; they share the centre.
    c = L / 2.0

    def line(prefix, p0, p1):
        (x0, y0), (x1, y1) = p0, p1
        node_ids = []
        for k in range(NSEG + 1):
            t = k / NSEG
            x, y = x0 + t * (x1 - x0), y0 + t * (y1 - y0)
            nid = f'{prefix}{k}'
            if abs(x - c) < 1e-9 and abs(y - c) < 1e-9:
                nid = 'C'                       # the shared crossing node
            if nid not in s.nodes:
                s.add_node(nid, x, y)
            node_ids.append(nid)
        for k in range(NSEG):
            s.add_bar_element(f'{prefix}e{k}', node_ids[k], node_ids[k + 1], 'B')
        return node_ids

    nx = line('X', (0.0, c), (L, c))
    ny = line('Y', (c, 0.0), (c, L))

    # Simply support the four beam ends (restrain w; leave rotations free).
    s.add_support('SS', w=True)
    for nid in (nx[0], nx[-1], ny[0], ny[-1]):
        s.assign_support(nid, 'SS')

    s.add_point_load('C', 'LC', fz=P)
    r = s.calculate()

    w_c = r['displacements']['LC']['C'][0]
    I = 0.30 * 0.50 ** 3 / 12.0
    w_beam = (abs(P) / 2.0) * L ** 3 / (48.0 * E * I)   # half load, one beam

    print("=== Crossed-beam grillage ===")
    print(f"  centre w (MEF):        {w_c:+.6e} m")
    print(f"  centre w (half-load beam): {-w_beam:+.6e} m")
    print(f"  error:                 {abs(abs(w_c) - w_beam) / w_beam:.2%}")

    total_reaction = sum(v[0] for v in r['reactions']['LC'].values())
    print(f"  Sum Rz:                {total_reaction:+.3f} kN "
          f"(applied {P:+.3f} kN)")
    return r


if __name__ == '__main__':
    run()
