# xdfem2D — test suite

The tests run with `pytest` and are split into three folders under `tests/`:

| Folder | Scope |
|--------|-------|
| `tests_engine/` | The calculation core `xdfem2d`: API, persistence, IO, meshing, combinations, variants, phasing — everything except pure result quality. |
| `tests_model/` | **Calculation-result quality**: closed-form analytical benchmarks and physics checks (validation, CST patch tests, thermal, springs, distributed/point loads, post-processing, incremental-solve equivalence). |
| `tests_app/` | The desktop application (Qt GUI + assistant). Not published with the engine, so absent from the public mirror; the licence boundary is enforced by `tests_app/test_engine_boundary.py`. |

`conftest.py` puts `src/` and `tests/` on `sys.path`; `context.py` holds the
shared helpers (`assert_close`, section properties). No install is required.

## Running

From the project root:

```bash
pytest                       # everything (testpaths = tests/ in pyproject.toml)
pytest tests/tests_model     # just the calculation-quality suite
pytest tests/tests_engine    # just the engine
pytest tests/tests_app       # just the application
```

VS Code discovers all three folders once `python.testing.pytestEnabled` is on
(already set in `.vscode/settings.json`).

`tests_engine` and `tests_model` are headless (only `numpy`; the concrete-design
tests additionally need `eurocodepy`). `tests_app` needs the app dependencies
(PySide6 …).

## Result-quality benchmarks (`tests_model`)

`test_validation.py` and `test_validation_x2d.py` compare the engine against
closed-form solutions; the latter is driven by the `.x2d` models in
`../validation/` (see `validation/README.md`). Formulas below.

## Analytical cases and expected results

| Case | Quantity | Formula |
|------|----------|---------|
| Simply supported, UDL `w`, span `L` | mid-span moment | `wL²/8` |
| | end reactions | `wL/2` |
| | end rotation | `wL³/24EI` |
| Simply supported, central `P` | mid moment / deflection | `PL/4` · `PL³/48EI` |
| Cantilever, tip `P` | tip deflection / base moment | `PL³/3EI` · `PL` |
| Cantilever, UDL `w` | tip deflection / base moment | `wL⁴/8EI` · `wL²/2` |
| Clamped-clamped, UDL `w` | end moment | `wL²/12` |
| Axial bar, load `P` | elongation / force | `PL/EA` · `P` |
| Propped cantilever (hinge), UDL | clamp moment / reactions | `wL²/8` · `3wL/8`, `5wL/8` |
| Restrained bar, uniform `ΔT` | axial force (compression) | `−EA·α·ΔT` |
| Free bar, uniform `ΔT` | elongation / force | `α·ΔT·L` / `0` |
| Simply-supported, gradient `ΔT` | bending moment | `0` |
| Clamped-clamped, gradient `ΔT` | restraint moment | `EI·α·ΔT/h` |

## Bugs found and fixed during test development

**1 — Distributed loads on non-horizontal members.** `loads._apply_distributed_loads`
split each distributed load into *global* X/Y components and applied the axial
(linear) shape functions to the X part and the bending (Hermitian) ones to the Y
part, regardless of the element's orientation. This is only valid for horizontal
members: a horizontal wind load on a vertical column produced **zero base
moment** and a drift that vanished under mesh refinement. Fixed by resolving the
load into the element's *local* axes first, forming the consistent nodal loads
there, and rotating them to global with the element transformation matrix.
Horizontal members are bit-for-bit unchanged. Covered by
`test_advanced.py::TestLocalAxisLoad` (base moment `qL²/2`, tip drift `qL⁴/8EI`).

**2 — Non-linear combination rules silently linearised.** `AbsSum` (Σ|·|)
and `SRSS` (√Σ(·)²) combinations built directly on load cases were converted
to plain linear sums, because `migrate_combinations_to_analysis_cases` moved
every combo's coefficients into linearly-added analysis cases regardless of the
combination type (and their internal-force diagrams came out zero). Fixed by
migrating only `LinearSum` combinations and leaving the non-linear rules and
`Envelope` with their direct load-case coefficients, which their dedicated
branches already handle. Covered by `test_advanced.py::TestCombinationRules`.

**3 — Thermal element-force recovery.** It double-counted the free thermal pre-strain:
an axially unrestrained heated bar reported `N = 2·EA·α·ΔT` instead of `0`
(global displacements were already correct). The fixed-end-force terms for
thermal loads in `loads._apply_temperature_loads` used the wrong sign at the
i-end, inconsistent with the mechanical-load convention used elsewhere. Fixed by
aligning the thermal `fef` signs with the mechanical ones; the tests above now
pin the corrected behaviour (`N = 0` free, `N = −EA·α·ΔT` restrained, `M = 0`
for a simply-supported gradient).
