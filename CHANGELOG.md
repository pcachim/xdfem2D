# Changelog — xdfem2D engine

Engine releases are tagged `engine-v<version>`. Versions up to 2026.10.4 were not released
separately; their changes are grouped below by date.

## 2026.10.12 — 2026-10-12
- `.x2d` files can carry named snapshots of the model in an optional `snapshots.json` entry
  (`save_x2d(..., snapshots=...)`); `load_x2d` returns them on `struc._loaded_snapshots`. Files
  without the entry open as before.
- Circular slab sector templates (`_tpl_slab_sector`, `_tpl_slab_sector_obj`) accept a
  `formulation`. The object variant meshes with triangles (`prefer_quad=False`) instead of asking
  for a quad section it could never produce.
- `create_edge_load`: `p` is now the load per unit length [kN/m] as documented. It was stored
  as a pressure and multiplied by the thickness, so a surface edge carried `p × thickness`
  instead of `p`. Saved models and `add_surface_edge_load` are unchanged.

## 2026.10.7 — 2026-10-05
- README badges.

## 2026.10.6 — 2026-10-05
- README badges; a GitHub Release is created for each engine release.
- Publishing now requires an `engine-v*` tag.

## 2026.10.5 — 2026-10-05
- **Modal analysis:** massless degrees of freedom are condensed exactly instead of being given
  a token mass.
- Packaging: PyPI metadata, logo; no `dist/` in the published tree; documentation reworked
  (fuller README; the engine is licensed under the LGPL v3).

## 2026.9.30 – 2026.10.4 — 2026-09-30 to 2026-10-04
- **Beam bars:** required areas converted into bars per zone of continuous beams; bar and
  stirrup detailing with anchorage and lap lengths; one parameters object / registry.
- Steel design: choice of interaction method (Annex A / Annex B) for EN 1993-1-1 §6.3.3;
  hollow sections are not LTB-susceptible; EN shear area for RHS/SHS.
- Timber design: corrected `V_Ed`; EC5 report fixes.
- Load cases carry a category and ψ factors, with defaults from eurocodepy.
- Design summary by material; area tools return the maxima; reference inventory and a
  results/lock invariant; plate labels; templates keep the material type.

## 2026.9.5 – 2026.9.23 — 2026-09-06 to 2026-09-28
- `Structure2D.create_from_template(kind, **params)` with 15 template kinds
  (`template_api.py`).
- `create_*` loads create the load case when it does not exist; load-case, analysis-case and
  combination names are case-insensitive; Portuguese combination names supported.
- Mesh generation: conformity by finest-first ordering; fuller model check; merge of models
  (triangles, quads and geometry objects).
- Design report: material-aware summaries, per-report utilization, General/Detailed tabs.
- Uniform naming of `utilization` (punching used `utilisation`).

## 2026.7.19 – 2026.9.5 — 2026-07-19 to 2026-09-06
### Elements and analysis
- Plate/grillage domain with the DKT, MITC3 and ES-FEM triangles; **quad elements**
  (DKT4 / MITC4); constraints; springs; rigid/dummy elements; modal analysis.
- Temperature loads on bars, triangles and quads now accumulate instead of replacing; fixed a
  uniform temperature load producing spurious shear/moment at hinges and the thermal
  curvature sign.
- New action types S, C, A, F (snow, construction, accident, fire); fixes in load
  combinations; `create_load_combination` alias.
### Design
- Torsion and punching design; steel and timber design; buckling length definitions;
  RC columns (rectangular and circular) with interaction diagrams.
- Design report with detailed equations in HTML, Markdown, DOCX and PDF.
### Modelling
- **Variants and construction sequences**; section cuts; edge supports; fields and field
  templates; object-level loads (line/area, pressure, edge).
- Python scripting API (`create_*` functions, `create_polygon`, `create_uniform_load`),
  export of the model as a Python script, JSON validation when opening models.
- Templates: portal frames, continuous beams, trusses, arches, plate templates.
