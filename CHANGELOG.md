# Changelog — xdfem2D engine

Engine releases are tagged `engine-v<version>`. Versions up to 2026.10.4 were not released
separately; their changes are grouped below by date.

## Unreleased

## 2026.10.21 — 2026-10-21
- **EN 1992-1-1:2023 concrete design.** `ec2_edition="2023"` (preference; `edition=` on the functions,
  `rc_design.edition_kwargs(prefs)`) now selects the second-generation Eurocode 2 for:
  - bars: `RCSection(edition="2023")` — `fcd = ηcc·ktc·fck/γc` (Eq. 5.3; `alpha_cc` is not used),
    shear of 8.2 (`γv`, `ddg`, `τRdc,min`, compression field with ν = 0,5, minimum stirrups of Eq. 12.4; with the
    moment `Med` the axial force and shear span of 8.2.2(3)-(4) enter VRd,c), torsion of 8.3 (ν from Annex G)
    and the combined shear + torsion of Eq. 8.86. `design_concrete_sections`, `reinforcement_envelope`, `design_and_store`
    and the beam bars follow the edition; rows have `edition`;
  - membranes (`design_concrete_planes`): Annex G.3 (Table G.1) with `σcd ≤ ν·fcd`, ν = 0,4 or from the strains
    (`nu_mode` / preference `ec2_nu_mode`);
  - slabs (`design_concrete_slabs`): Annex G.4, the element moments `mx, my, mxy` give the capacities of the bottom and
    top layers with Tables G.2/G.3 instead of the Wood-Armer moments; where `|mxy| ≥ 0,07·d²·fcd` the **sandwich model**
    (G.4(1)) is used — two layers of thickness 2·cover, each a membrane, strut checked against ν·fcd (rows have `method`
    and `crushing`; preferences `ec2_slab_method` auto/optimum/sandwich and `ec2_nu_mode`);
  - punching (`design_punching`): `gamma_v`, `dmax` and `eta_sys` may be `None` (from the edition data, Table 4.3 and
    `punch_params`);
  - columns: `fcd` of 2023 (`ηcc·ktc`); the slenderness, second-order and N-M rules are still those of 2004 (eurocodepy's
    `uls2023.column` re-exports them).
  New preferences: `ec2_k_tc` (ktc, 1,0 or 0,85), `ec2_nu_mode`, and `ec2_annex` (country code of a national annex, none by
  default). Crack width (SLS) and the anchorage/lap lengths are not part of this and keep EN 1992-1-1:2004.
- Slabs in 2004: Wood-Armer has no strut check and EN 1992-1-1:2004 gives no validity limit for it. Where the strut of
  the sandwich layers would exceed ν·fcd (ν = 0,6·(1 − fck/250); `σcd = 2|mxy|/(t·z)`) the row's `note` says so and
  `strut_warning` is true — informative only: the reinforcement does not change (`crushing` is for the 2023 check). An
  element that is not cracked (EN 1992-2 Annex LL (107), with the transverse shears `vx`, `vy`) has no such strut and gets no note.
- Slabs, out-of-plane shear (it was not verified): rows have `v_ed` (principal shear from `vx`, `vy`), `shear_mode`
  ('none', 'no_shear_reinf', 'stirrups', 'strut_crushing'), `asw_s` (shear reinforcement [m²/m²]) and a crushed shear strut
  sets `crushing`. 2023: 8.2 (G.4(4)) with the chord force `vEd·cot θ` of the truss added to the sandwich layers; 2004: the inner
  layer of EN 1992-2 Annex LL (109)-(111) from the Wood-Armer bars (`ρl = ρx·cos²φ0 + ρy·sin²φ0`), the chord force is only noted
  in the row.
- **With eurocodepy 2027.0.12** (needed): the crack width is computed for all the stations of a section at once with
  `ec2.sls.crack_opening_rect` / `is_cracked_rect` (earlier releases do not have them and the crack-width envelope failed
  with an `ImportError`).
- **With eurocodepy 2027.0.10** (needed): the 2004 membrane design compares σc with ν·fcd, ν = 0,6·(1 − fck/250), as
  EN 1992-1-1:2004 Annex F.1(4) asks (fcd without reinforcement; it compared with fcd, unsafe); the preference
  `ec2_strut_model` = `6.109` uses instead the strength of EN 1992-2 §6.109 (Eqs. 6.110/6.112, with the angle between the
  strut and the elastic principal direction), also for the informative strut note of the Wood-Armer slabs; rows have `nu` and `f_lim` (= ν·fcd, what σc is compared with — the strut tables, the
  canvas and the report show it) and `theta` is `None` where cot θ is undefined (no reinforcement required). Slab rows have
  `note`: the warnings of the check (strip over-reinforced, `x/d > 0,45`; no solution — `As` is `nan`; 2023: `|mxy| ≥ 0,07·d²·fcd`).
- README: the documentation link has its own paragraph (with the Engine Reference), and *Citing* gives the Zenodo DOI
  (all versions, always the latest release) instead of a template.

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
