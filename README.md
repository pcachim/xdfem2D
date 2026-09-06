<p align="center">
  <img src="src/xdfem2d_app/icons/xdfem2d_512.png" alt="xdfem2D logo" width="140">
</p>

<h1 align="center">xdfem2D</h1>

<p align="center">
  2D plane-frame structural analysis: a Python FEM library (<code>xdfem2d</code>)
  paired with a PySide6 desktop application.<br>
  Units throughout: <b>kN · m · kNm</b>.
</p>

<p align="center">
  <a href="https://github.com/pcachim/xdfem2D/releases"><img alt="App release" src="https://img.shields.io/github/v/release/pcachim/xdfem2D?label=app"></a>
  <a href="https://pypi.org/project/xdfem2d/"><img alt="Engine on PyPI" src="https://img.shields.io/pypi/v/xdfem2d?label=engine"></a>
</p>

<p align="center">
  The app and the engine are versioned and released independently — the
  badges above always show the current number of each. To see exactly which
  engine version a given app build was compiled against, open
  <b>Help → About</b> in the app: it lists both.
</p>

---

## Features

### Modelling & elements
- Euler–Bernoulli beam–column elements (full 6-DOF, 3 per node)
- Constant-strain triangle (CST) plane-stress/plane-strain surface elements,
  with automatic Delaunay meshing of polygon surfaces (built-in mesher, or the
  higher-quality [Triangle](https://github.com/drufat/triangle) library as an
  optional extra)
- Pinned, roller, fixed, guided, and rotational supports
- Node springs (Kx/Ky/Kt) and element foundation (Winkler) springs, including
  tension-only / compression-only (unilateral) springs
- Parametric geometry objects (arcs, polylines) that auto-resolve T-junctions
  and crossings when the model is analysed
- Automatic topology checks: isolated nodes, coincident nodes, unconnected
  junctions/crossings, duplicate bars

### Loads & analysis
- Point loads, trapezoidal distributed loads (global or local axes),
  thermal loads, support settlements, nodal masses, and self-weight by
  load-case factor
- A three-tier model — **load cases** (define loads) → **analysis cases**
  (solve them) → **combinations** (combine analysis cases) — mirroring the
  convention used by mainstream structural software
- Analysis types: Linear, NonLinear (unilateral springs, active-set
  iteration), Mass, Modal (eigenvalue), Response Spectrum (EC8), and
  Geometric Nonlinear (P-Delta)
- Load combinations: linear, envelope (max/min), SRSS, and automatic
  Eurocode (EC0) combination generation

### Structural design
- Simplified **Eurocode 2** reinforced-concrete design for rectangular
  sections: flexural + shear reinforcement, with EC2 crack-width (SLS)
  verification
- **EC8** seismic response spectra (elastic and design)

### Variants, phasing & interoperability
- **Variants**: parallel what-if scenarios over a shared model, combined with
  the same arithmetic as load combinations
- **Construction phasing / sequencing**: incremental analysis carrying
  forward locked-in internal forces from earlier phases
- Import a second model into a shared entity space, welding coincident nodes
  automatically (the shared foundation behind variants and phasing)
- Import/export **DXF** drawings, import/export **SAP2000** (`.s2k`), and
  export to **Autodesk Robot** (`.str`)

### Interface & results
- Interactive canvas: cursor-anchored zoom, pan, window selection, hover
  tooltips, and named **Scenes** (saved working subsets of the model)
- Parametric **templates** (portal frame, continuous beam, frame, Warren/
  Howe/Pratt trusses, parabolic arch) to start a model in seconds
- Results: displacements, reactions, element end-forces, N/V/M diagrams,
  deformed and modal shapes, CST stress fields, and reaction sums
- Exports: drawing (PNG/SVG/PDF), a formatted PDF results report, an Excel
  workbook (model or results), and Word reports
- Save/load structures as `.x2d` (default, includes results) or `.json`
  (structure only)

---

## Download the desktop app

Prebuilt, no-install-needed apps for **macOS, Windows and Linux** are published
on every release at [github.com/pcachim/xdfem2D/releases](https://github.com/pcachim/xdfem2D/releases).
Opening a `.x2d` file from your file browser launches the app directly.

Full documentation (also built from this repo) is published at
[pcachim.github.io/xdfem2D](https://pcachim.github.io/xdfem2D/).

---

## Graphical Interface

```bash
uv run xdfem2d   # launch the GUI
```

### Layout

| Area | Contents |
|---|---|
| Left panel — **Properties** | Project · Materials · Bar sections · CST sections |
| Left panel — **Geometry** | Nodes · Bars · Triangles · Supports · Springs |
| Left panel — **Loads** | Loads · Edge loads · Analysis Cases · Combinations |
| Left panel — **Results** / **Design** | Numerical results (displacements, reactions, forces) / RC design table |
| Right | Structure canvas |

### Toolbar

| Control | Description |
|---|---|
| ▶ Run (F5) | Run FEM analysis and save results |
| 🔒 Lock / 🔓 Unlock (Ctrl+L) | Lock model after analysis / unlock to edit |
| View | Switch canvas display: Structure · Deformed · M · V · N · Reactions |
| Case | Select load case or combination to display |
| Scene | Switch to a named working subset of the model, or the full model |
| Scale | Deformation scale multiplier (e.g. 100 × real), with an Auto option |

### Keyboard shortcuts

| Key | Action |
|---|---|
| Ctrl+N | New model |
| Ctrl+T | New from template |
| Ctrl+O | Open file |
| Ctrl+S | Save |
| Ctrl+Shift+S | Save As |
| Ctrl+E | Export drawing |
| F5 | Run analysis |
| Ctrl+L | Toggle lock |
| Ctrl+1 … Ctrl+8 | Switch view mode (Structure … Modal) |
| F9 | Show/hide tables panel |
| Ctrl+F | Find object |
| F1 | Open documentation |

Full list (incl. Edit/View/Canvas shortcuts): **Help → Keyboard Shortcuts** in
the app, or [Keyboard Shortcuts](https://pcachim.github.io/xdfem2D/shortcuts.html) online.

### Canvas interaction

- **Hover** over a node or element to see its properties as a tooltip
- **Click** a node to edit geometry, support, and node spring
- **Click** an element to edit connectivity, section, and foundation spring

### File formats

| Extension | Description |
|---|---|
| `.x2d` | ZIP containing `structure.json` + `results.json` — default format |
| `.json` | Structure definition only (no results) |

Analysis results can additionally be exported to `.json`, `.xlsx`, or a PDF
report; the model alone can be exported to `.xlsx`, DXF, SAP2000 (`.s2k`), or
Robot (`.str`) — see [Exporting](https://pcachim.github.io/xdfem2D/exporting.html).

After every successful analysis the `.x2d` file is saved automatically.
Opening a `.x2d` that contains results restores them and locks the model.

---

## Python API

```python
from xdfem2d import Structure2D

struc = Structure2D()

# Nodes
struc.add_node('N1', 0.0, 0.0)
struc.add_node('N2', 0.0, 4.0)
struc.add_node('N3', 6.0, 4.0)
struc.add_node('N4', 6.0, 0.0)

# Material and cross-section
struc.add_material('C25', elastic_modulus=30e6, unit_weight=25.0)
struc.add_section('COL', material_name='C25', b=0.3, h=0.3)
struc.add_section('BM',  material_name='C25', b=0.25, h=0.5)

# Elements
struc.add_bar_element('E1', 'N1', 'N2', 'COL')   # left column
struc.add_bar_element('E2', 'N2', 'N3', 'BM')    # beam
struc.add_bar_element('E3', 'N4', 'N3', 'COL')   # right column

# Supports
struc.add_support('FIXED', ux=True, uy=True, tz=True)
struc.assign_support('N1', 'FIXED')
struc.assign_support('N4', 'FIXED')

# Loads
struc.add_load_case('SW', self_weight_factor=1.0)
struc.add_load_case('LL')
struc.add_distributed_load('E2', 'LL', fye=-15.0, fyd=-15.0)   # 15 kN/m down
struc.add_point_load('N2', 'LL', fx=10.0)                       # 10 kN horizontal

# Combination
struc.add_load_combination('ULS', coefficients={'SW': 1.35, 'LL': 1.5})

# Solve
results = struc.calculate()

# Displacements
for nid, d in results['displacements']['LL'].items():
    print(f"{nid}: ux={d[0]:.6f} m  uy={d[1]:.6f} m  rz={d[2]:.6f} rad")

# Reactions
for nid, r in results['reactions']['LL'].items():
    print(f"{nid}: Rx={r[0]:.3f} kN  Ry={r[1]:.3f} kN  Mz={r[2]:.3f} kNm")

# Save / load
struc.save('portal.json')
struc2 = Structure2D.load('portal.json')
```

### Export helpers

```python
from xdfem2d import save_structure_json, load_structure_json
from xdfem2d.results_io import save_json, save_excel

save_json(results,  'results.json')
save_excel(results, 'results.xlsx')
```

### RC design (EC2)

```python
from xdfem2d import design_reinforcement

struc.bar_elements_by_id['E2'].rc_design = True
struc.bar_elements_by_id['E2'].rc_cover   = 0.03   # m

struc.add_concrete_material(
    'C25', concrete_class='C25/30', steel_class='A500',
    gamma_c=1.5, gamma_s=1.15, alpha_cc=1.0)

results = struc.calculate()
rc = design_reinforcement(struc, results)

for elem_id, cases in rc.items():
    for case_id, r in cases.items():
        print(f"{elem_id} / {case_id}: "
              f"As_bot={r.As_bot*1e4:.2f} cm²  "
              f"As_top={r.As_top*1e4:.2f} cm²  "
              f"Asw/s={r.Asw_s*1e4:.2f} cm²/m")
```

### Variants and construction phasing

```python
from xdfem2d import solve_variants, combine_across_variants
from xdfem2d.workflows import run_sequence

# Variants: independent what-if overlays on the same base model
variant_results = solve_variants(struc, variants)
combined = combine_across_variants(variant_results, terms)

# Construction phasing: incremental analysis carrying forward locked-in forces
phase_results = run_sequence(struc, sequence)
```

### Interoperability

```python
from xdfem2d.dxf_io import load_dxf, save_dxf
from xdfem2d.sap2000_io import save_s2k, load_s2k
from xdfem2d.robot_io import save_str
```

## Sample file

`examples/example_2level_2bay_full.x2d` — a two-level, two-bay frame ready to
open in the GUI, with a solved model and results included.

---

## License

xdfem2D, the desktop application, is **freeware**: free to download and use, but distributed only as a compiled application,
without redistribution or modification rights. See [LICENSE.md](LICENSE.md)
for the full terms, including the "as is" / no-warranty disclaimer.

The app is built on two open-source libraries, each published and licensed
separately:

| Component | What it does | License |
|---|---|---|
| [xdfem2d](https://github.com/pcachim/xdfem2D) | The 2D finite-element calculation engine at the core of the app — frames, plates/slabs and grillages, load cases and combinations, modal and response-spectrum analysis. Also on PyPI (`pip install xdfem2d`). | LGPLv3 — [LICENSE-engine.md](LICENSE-engine.md) |
| [eurocodepy](https://github.com/pcachim/eurocodepy) | Eurocode material databases and design checks (concrete, steel, timber) used by the engine for reinforced-concrete and timber design. | LGPLv3 |

The LGPLv3 allows both libraries to be embedded in xdfem2D's freeware
distribution while keeping them free software: you may obtain, modify and
relink either one, independently of the app, under its own LGPLv3 terms.

See [NOTICE.md](NOTICE.md) for the full list of third-party components
bundled in the app, including Qt/PySide6 (also LGPLv3).
