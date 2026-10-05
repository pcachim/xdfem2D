# xdfem2D

[![Engine on PyPI](https://img.shields.io/pypi/v/xdfem2d?label=engine)](https://pypi.org/project/xdfem2d/)
[![Python versions](https://img.shields.io/badge/python-3.12%2B-blue)](https://pypi.org/project/xdfem2d/)
[![Licence: LGPLv3](https://img.shields.io/badge/licence-LGPLv3-blue)](https://github.com/pcachim/xdfem2D/blob/main/LICENSE.md)
[![App release](https://img.shields.io/github/v/release/pcachim/xdfem2D?label=app)](https://github.com/pcachim/xdfem2D/releases)

This repository holds two things, under two licences.

**The calculation engine** — the source in this tree, `xdfem2d`, is a Python
library for two-dimensional finite element analysis of frames, shells, slabs
and grillages, under the GNU Lesser General Public License v3 (LGPLv3). No
interface, no dependency on one: it builds a model, solves it, and gives back
numbers.

**The desktop application** — a graphical interface built on that engine, with
an assistant that can answer questions about the model you have open. It is
free to use, distributed as a compiled application under its own terms, and its
source is not published. Download it from
[Releases](https://github.com/pcachim/xdfem2D/releases); the documentation is
at [pcachim.github.io/xdfem2D](https://pcachim.github.io/xdfem2D).

The rest of this page is about the library. Releases tagged `engine-v*` are
the engine; those tagged `v*` are the application.

## Install

Python **3.12 or later**.

```bash
pip install xdfem2d
```

The core install is NumPy, SciPy and
[eurocodepy](https://github.com/pcachim/eurocodepy) — what solving and design
need. Reporting and file exchange are separate extras, because a solver running
in a script or a service should not have to install a Word library:

```bash
pip install "xdfem2d[report]"   # matplotlib, openpyxl, python-docx
pip install "xdfem2d[dxf]"      # ezdxf
pip install "xdfem2d[ifc]"      # ifcopenshell — IFC (openBIM) import/export
pip install "xdfem2d[all]"      # all of the above
```

Everything they cover is reached lazily, so the engine imports, assembles,
solves and designs without them.

## Versioning

The engine and the desktop app are versioned and released independently, on
separate cycles — an engine release does not imply a matching app release,
and vice versa. Don't expect the two numbers to line up. The app's **Help →
About** dialog lists both, so a bug report can name the right one.

---

## A first model

```python
from xdfem2d import Structure2D

s = Structure2D()
s.add_material('C30/37', elastic_modulus=33e6, unit_weight=25.0)
s.add_section('S', 'C30/37', b=0.3, h=0.5)
s.add_node('N1', 0.0, 0.0)
s.add_node('N2', 6.0, 0.0)
s.add_bar_element('E1', 'N1', 'N2', 'S')
s.add_support('PIN', ux=True, uy=True)
s.add_support('ROLL', uy=True)
s.assign_support('N1', 'PIN')
s.assign_support('N2', 'ROLL')
s.add_load_case('LC')
s.add_distributed_load('E1', 'LC', fye=-20.0, fyd=-20.0)
s.add_analysis_case('LIN', 'Linear', {'LC': 1.0})

results = s.calculate()
print(results['reactions']['LC']['N1'])   # [Rx, Ry, Mz] — 60 kN up at N1
```

Units are metres, kN, kN·m and kN/m². Angles are in degrees where an angle is
named. The full API is in the
[Engine Reference](https://pcachim.github.io/xdfem2D/engine-api.html).

## What it does

- **Two analysis domains** — a model is either *plane* (frames and walls in
  their plane: ux, uy, θz) or *plate* (slabs and grillages out of plane: w, θx,
  θy). The DOF machinery is shared; only the meaning of the three components
  changes, so cases, combinations, envelopes and phasing work identically in
  both.
- **Bar elements** — plane frames and trusses (releases, offsets, springs) and,
  in the plate domain, grillage beams (bending + St-Venant torsion GJ/L).
- **Triangular and quadrilateral elements** — plane stress / plane strain
  membranes (CST, Allman, ES-FEM; QM6, Q4) and, in the plate domain, plate
  bending: MITC3 (shear-deformable Mindlin–Reissner, the default — thin and
  thick slabs) and DKT (thin-plate Kirchhoff), plus MITC4 and DKT4 quads. They
  report moments mx/my/mxy, principal moments, shears and Wood–Armer design
  moments.
- **Slab loads** — transverse pressure pz, self-weight, a through-thickness
  thermal gradient, and Winkler area springs (slab on grade). A mesher expands
  geometric regions into elements.
- **Analysis** — linear static, non-linear (unilateral springs), P-Delta,
  modal, and response spectrum (EC8); in the plate domain the modes are the
  out-of-plane (vertical) vibration.
- **Load cases and combinations**, including nested combinations, envelopes and
  Eurocode (EN 1990) generation.
- **Design to Eurocodes**, via [eurocodepy](https://github.com/pcachim/eurocodepy)
  (LGPLv3): reinforced concrete (EC2), steel members (EC3) and timber members
  (EC5).
- **Construction phasing** and model variants.
- **File formats** — its own `.x2d`, plus SAP2000 `.s2k`, Robot, DXF, IFC and
  Excel.

## Validation and known limits

[`validation/`](https://github.com/pcachim/xdfem2D/tree/main/validation) holds
cases with a closed-form solution, built by the same code that the test suite
runs, together with the generated report (`Validacao_Programa_MEF_resultados.docx`,
in Portuguese). They cover bars (N/V/M, settlement, temperature, curved arch),
membrane triangles, DKT and MITC3 plates against Navier and Mindlin series,
grillages, node and Winkler springs, multi-point constraints, modal analysis
(closed-form frequencies, effective masses) and the response-spectrum chain
(SRSS, CQC, interpolation), and the plate quads.

What that does and does not establish, so the result is not taken for more:

- **Validated against a formula** is not **validated for your structure**. A
  case here checks that an element reproduces a textbook solution, not that it
  suits a given geometry, mesh or load path.
- The response-spectrum cases check the spectral chain (Sd = Sa/ω², SRSS,
  CQC); they do not certify the EC8 parameters you choose.
- **Design** (EC2/EC3/EC5) is checked on worked examples in `validation/`; it is
  an aid to calculation, not a code-compliance certificate.
- P-Delta, construction phasing and variants have regression tests but no
  closed-form case in `validation/`.
- **Mass is lumped and diagonal**: concentrated masses give exact frequencies,
  distributed mass converges as O(h²) from below. Rotary inertia exists only if
  `mtz` is given. Mass is taken from the absolute value of Fy, and the
  percentages of effective mass refer to the mass on the free DOFs.
- **Multi-point constraints are imposed by penalty** (relative error ~10⁻⁶,
  with a tiny spurious reaction).
- **Element springs are lumped** (k·L/2 at each end): a uniform state is
  exact, a varying one converges as O(h²). Unilateral springs are honoured
  only by `NonLinear` analysis cases.
- The CST triangle converges slowly, from below, in bending; the Allman
  triangle converges faster.

## A note on results

xdfem2d computes analysis and design results. It does not verify them, and
producing a number is not the same as the number being right for your problem.
It is an aid to calculation, not a substitute for engineering judgement:
anyone using its output in the design or construction of a real structure is
responsible for verifying it independently and remains the engineer of record.

## Citing

If xdfem2d contributes to published work, please cite the version you used:

> Cachim, P. *xdfem2D — 2D finite element analysis of frames, walls, slabs and
> grillages* (engine version X.Y.Z). https://github.com/pcachim/xdfem2D

The version is in `xdfem2d.__version__` and on
[PyPI](https://pypi.org/project/xdfem2d/).

## Support and contributing

Bugs and questions, for the library or the application, go to
[Issues](https://github.com/pcachim/xdfem2D/issues). A report with a model
that reproduces the problem is the most useful thing you can send. For
anything else, write to [eurocodepy@gmail.com](mailto:eurocodepy@gmail.com).
The documentation is at [pcachim.github.io/xdfem2D](https://pcachim.github.io/xdfem2D).

This tree is a mirror. Development happens in a private repository and each
engine release replaces it wholesale, so a pull request here has nowhere to
land — it would be overwritten by the next one.

## Licence

The engine — everything in this tree — is under the **LGPLv3**; see
[LICENSE.md](https://github.com/pcachim/xdfem2D/blob/main/LICENSE.md). It
depends on [eurocodepy](https://github.com/pcachim/eurocodepy), also LGPLv3,
for its Eurocode material databases and design checks.

The desktop application is a separate work, distributed as freeware in compiled
form under its own terms (see the
[Licenses page](https://pcachim.github.io/xdfem2D/license.html), and
`LICENSE.md` shipped with the app). Nothing in those terms restricts your
rights to the engine: you may use, modify and redistribute it under the LGPLv3,
from here or from PyPI.
