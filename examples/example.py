"""
Example: simply supported beam with uniform transverse load.

Geometry:   N1 ----5m---- N2
Supports:   N1 pinned (UX+UY fixed), N2 roller (UY fixed)
Load:       10 kN/m uniform downward on element E1
Material:   E = 30e6 kN/m², unit_weight = 25 kN/m³
Section:    b = 0.3m, h = 0.6m

Analytical mid-span deflection: 5wL^4/(384EI)
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from xdfem2d import Structure2D

def run():
    struc = Structure2D()

    # Nodes
    struc.add_node('N1', 0.0, 0.0)
    struc.add_node('N2', 5.0, 0.0)

    # Material: concrete-like
    struc.add_material('C30', elastic_modulus=30e6, unit_weight=25.0)

    # Section: 0.3 x 0.6 m rectangle
    struc.add_section('S1', material_name='C30', b=0.3, h=0.6)

    # Bar element
    struc.add_bar_element('E1', node_i='N1', node_j='N2', section_name='S1')

    # Supports
    struc.add_support('PIN',    ux=True, uy=True, tz=False)
    struc.add_support('ROLLER', ux=False, uy=True, tz=False)
    struc.assign_support('N1', 'PIN')
    struc.assign_support('N2', 'ROLLER')

    # Load case: uniform downward load 10 kN/m
    struc.add_load_case('LC1', self_weight_factor=0.0)
    struc.add_distributed_load('E1', 'LC1', fye=-10.0, fyd=-10.0)

    # Solve
    results = struc.calculate()

    # --- Print results ---
    print("=== Displacements (m, rad) ===")
    for node_id, d in results['displacements']['LC1'].items():
        print(f"  Node {node_id}: ux={d[0]:.6f}  uy={d[1]:.6f}  rz={d[2]:.6f}")

    print("\n=== Reactions (kN, kNm) ===")
    for node_id, r in results['reactions']['LC1'].items():
        print(f"  Node {node_id}: Rx={r[0]:.3f}  Ry={r[1]:.3f}  Mz={r[2]:.3f}")

    print("\n=== Element forces (kN, kNm) ===")
    for elem_id, ef in results['element_forces']['LC1'].items():
        ni, nj = ef['i'], ef['j']
        print(f"  {elem_id}  i-end: N={ni[0]:.3f}  V={ni[1]:.3f}  M={ni[2]:.3f}")
        print(f"  {elem_id}  j-end: N={nj[0]:.3f}  V={nj[1]:.3f}  M={nj[2]:.3f}")

    # Analytical check
    import math
    E = 30e6
    b, h = 0.3, 0.6
    I = b * h**3 / 12.0
    w, L = 10.0, 5.0
    delta_analytical = 5 * w * L**4 / (384 * E * I)
    delta_fem = abs(results['displacements']['LC1']['N1'][1])  # mid span approximated at node
    print(f"\nAnalytical mid-span deflection (two-node beam approx): {delta_analytical*1000:.4f} mm")
    print("(Note: a single 2-node Euler-Bernoulli element gives exact end deflections for uniform load)")
    print(f"Node N1 uy = {results['displacements']['LC1']['N1'][1]*1000:.4f} mm  (expected 0.0 — pinned)")
    print(f"Node N2 uy = {results['displacements']['LC1']['N2'][1]*1000:.4f} mm  (expected 0.0 — roller)")


if __name__ == '__main__':
    run()
