"""Build examples/example-modal_frame.x2d.

A two-storey, one-bay reinforced-concrete frame with the dynamic chain the
other examples do not have: a Mass case fed by the gravity load cases, a Modal
case, and a response-spectrum case on an EC8 type-1 ground-A shape.

Kept small on purpose. It is opened to look at periods, mode shapes and
spectral results, and a model with three hundred elements teaches nothing extra
about those while making the file slow to open and the mode shapes hard to
read.
"""
from pathlib import Path

from xdfem2d import Structure2D
from xdfem2d.file_io import save_x2d

E = 33.0e6          # kN/m2  — C30/37
OUT = Path(__file__).resolve().parent / 'example-modal_frame.x2d'

# EC8 type 1, ground A, ag = 2.0 m/s2, q = 1.0, 5% damping. Written as the
# elastic acceleration in m/s2 against period, which is what the solver reads.
SPECTRUM = [
    [0.00, 5.00], [0.05, 7.25], [0.10, 5.00], [0.15, 5.00],
    [0.20, 5.00], [0.30, 5.00], [0.40, 5.00], [0.50, 4.00],
    [0.60, 3.33], [0.80, 2.50], [1.00, 2.00], [1.50, 1.33],
    [2.00, 1.00], [3.00, 0.44], [4.00, 0.25],
]


def build() -> Structure2D:
    s = Structure2D()
    s.add_material('C30/37', elastic_modulus=E, unit_weight=25.0)
    s.add_section('COL 0.30x0.30', 'C30/37', b=0.30, h=0.30)
    s.add_section('BEAM 0.25x0.50', 'C30/37', b=0.25, h=0.50)

    # ── Geometry: one 5 m bay, two 3 m storeys ────────────────────────
    L, H = 5.0, 3.0
    for i, (x, y) in enumerate([(0.0, 0.0), (L, 0.0),
                                (0.0, H), (L, H),
                                (0.0, 2 * H), (L, 2 * H)], start=1):
        s.add_node(f'N{i}', x, y)

    for eid, a, b in [('C1', 'N1', 'N3'), ('C2', 'N2', 'N4'),
                      ('C3', 'N3', 'N5'), ('C4', 'N4', 'N6')]:
        s.add_bar_element(eid, a, b, 'COL 0.30x0.30')
    for eid, a, b in [('B1', 'N3', 'N4'), ('B2', 'N5', 'N6')]:
        s.add_bar_element(eid, a, b, 'BEAM 0.25x0.50')

    s.add_support('FIXED', ux=True, uy=True, tz=True)
    s.assign_support('N1', 'FIXED')
    s.assign_support('N2', 'FIXED')

    # ── Load cases ────────────────────────────────────────────────────
    # Self weight plus the floor it carries; the live load is separate so the
    # mass case can take the quasi-permanent fraction of it.
    s.add_load_case('G', self_weight_factor=1.0)
    s.add_load_case('Q')
    for eid in ('B1', 'B2'):
        s.add_distributed_load(eid, 'G', fye=-20.0, fyd=-20.0)
        s.add_distributed_load(eid, 'Q', fye=-9.0, fyd=-9.0)

    # ── Dynamics ──────────────────────────────────────────────────────
    # psi2 = 0.3 on the live load, the quasi-permanent value EC8 asks for in
    # the seismic mass. The mass follows from the vertical load, so the two
    # cases above are also the mass definition.
    s.add_analysis_case('MASS', 'Mass', {'G': 1.0, 'Q': 0.3})
    s.add_analysis_case('MODAL', 'Modal', {}, modal_case_id='MASS',
                        num_modes=6)
    s.add_spectral_function('EC8-A', description='EC8 type 1, ground A, '
                            'ag=2.0 m/s2, q=1.0', damping=0.05,
                            points=SPECTRUM)
    s.add_analysis_case('SEISMIC-X', 'Spectrum', {}, modal_case_id='MODAL',
                        spectrum_id='EC8-A', combination_rule='CQC',
                        direction='X', damping=0.05)

    # ── Combinations ──────────────────────────────────────────────────
    s.add_load_combination('ULS', {'G': 1.35, 'Q': 1.50})
    s.add_load_combination('SLS', {'G': 1.00, 'Q': 1.00})

    s.project_info['Project name'] = (
        'Two-storey frame — modal and response spectrum')
    s.project_info['Designer'] = 'xdfem2D example'
    return s


if __name__ == '__main__':
    model = build()
    results = model.calculate()

    modal = results['analysis_cases']['MODAL']['modal_info']
    print('total mass X:', round(
        results['analysis_cases']['MODAL']['total_mass_x'], 3), 't')
    for m in modal:
        print(f"  mode {m['mode']}: T = {m['period']:.4f} s, "
              f"meff_x = {m['meff_x_pct']:.2f} %")
    print('sum meff_x:', round(sum(m['meff_x_pct'] for m in modal), 2))

    save_x2d(model, results, OUT)
    print('saved:', OUT)
