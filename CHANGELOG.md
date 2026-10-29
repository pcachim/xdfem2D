# Changelog — xdfem2D engine

Engine releases are tagged `engine-v<version>`. Versions up to 2026.10.4 were not released
separately; their changes are grouped below by date.

## Unreleased
- **`.x2d` format versions.** A file now carries `manifest.json` with `format_version` (currently 1), the writing program's version and its
  entries. `load_x2d` reads its own version and older ones, converting through a chain of `@migration(n)` functions on the parsed JSON
  (`xdfem2d.x2d_format`, which also applies them to the models inside `variants.json` and `snapshots.json`); a file with no manifest is
  version 1; a newer version raises `X2DFormatError` with a message. A conversion that changes the model can drop the stored results.
  Saved files are now byte-for-byte reproducible (entries carry a fixed date). A frozen version-1 file is in `tests/data/formats/v1/`.
- **Fix:** opening a `.x2d` no longer drops the triangles, quads and objects a variant chose (the load-time repair of its active set compared
  only bars).
- **Variants choose triangles, quads and area objects** like bars: `Variant.active_elements` names them in the same set, an area is kept
  only if its id is in the set (`area_selection`; a set of bar ids alone leaves the variant without areas), and the base/variant sync merges `tris`, `quads` and `objects` field by field (an edit on the
  variant stays; objects are no longer overwritten by the base's copy at every sync).
- **Ids are unique across the base and every variant model**: `create_node`, `create_bar_element`, triangles, quads and objects created without
  an id take one no model of the family uses. `variant_id_collisions` / `variant_base_diff()['id_collisions']` report an id that names two
  different things; `Structure2D.variant_nesting_problems()` reports a variant whose model carries sequences or variants.
- **`SupportSet.edge_supports`** (`{object_id: [(ux, uy, tz) | None, ...]}`, one entry per edge) sets the edges of area objects when
  `apply_support_set` runs; an unlisted object keeps its own, and a wrong object or edge count raises `ValueError`. It is saved in
  `support_sets`, follows an object that is renamed and is dropped when the object is removed. `solve_sequence` refuses a phase that
  takes edge supports away while `release_supports` is on, since the reaction of mesh nodes cannot be released yet.
- **Faster meshing and model check.** `mesh_polygon` does its edge flips, point-in-polygon tests and boundary filters in numpy, and
  `model_check` finds the nodes on an element edge by bounding box instead of testing every node against every edge (the 9 000-quad
  panel: 76 s to 0.75 s). The meshes and the issues found are the same. `expand_geometry` no longer searches a list for every node.
- **Baking an object (`xdfem2d.geo_expand`).** `bake_object_loads` also bakes the area springs of a surface, the line element springs
  and the uniform `loads` of a line object. New `bake_object_restraints(struc, compiled, trace, ids, live_node)` copies the supports
  and node springs that `expand_geometry` spread along the edges of the baked surfaces (`trace[id]['edge_supports']`,
  `['edge_springs']`) onto the live nodes.
- **A support set changes the supports of a phase (breaking).** `Structure2D.apply_support_set(support_set)` (used by `derive_variant`
  and so by `solve_phase`/`solve_sequence`) applies a set on top of the structure's supports: a listed node takes the set's restraint
  (all `False`: no support), the others keep theirs; listed node and element springs replace those, the others stay. It used to replace
  every support and spring, so an empty set removed them all. `phasing._effective_restraints` (the supports compared between phases for
  the release of a removed support) uses the same rule, and `import_model` builds the imported variant's supports from the other model's
  alone.
- **A phase holds a list of support sets (breaking).** `ConstructionPhase.support_set_id` is now `support_set_ids`, the sets in force in the
  order added. `resolve_phase_state(prev_active, prev_support_set_ids, ...)` returns the list: an `ADD` supports operation appends the
  ids it names, a `REMOVE` takes out those ids and only those. `derive_variant(variant, support_set)` takes a set or a list of them, applied in
  order, and `phasing._effective_restraints` follows the same order. `references` gains `Site.where` and covers the phases' lists and the
  `ids` of their supports operations, so deleting a set strips it from both (`remove_support_set` also drops an operation left
  with no ids) and `reference_problems()` reports an operation that names a missing set. New `Structure2D.rename_support_set(old, new)`. `references` keeps the type of a list it renames or strips (an operation's `ids` stay a tuple).
- **Variants are models of their own, kept in step by a three-way merge (breaking).** `Variant.support_set_id` is gone: a variant's
  supports, springs, bars and actions are those of its `model`. `Variant.base_state` records the base at the last sync — supports and
  springs, nodes, bars, sections and materials (one entry per field, `"E1|hinge_j"`), load cases, analysis cases and combinations
  (`"<id>|__exists__"` for the case itself) and loads (grouped by list, what they act on and case) — and `sync_variant_with_base`
  takes into the variant what the base changed where the variant has not changed the same thing (`merge_supports`), never overwriting
  what the user changed; `apply_variant_config` no longer copies coordinates, hinges or sections over the variant's. `Variant.
  inherit_actions=False` (set by `new_variant(inherit=False)`) opts a clean variant out of the actions. `AnalysisCase.solved` says
  what has been run, so it is not compared. New: `new_variant`, `base_state`, `reset_variant_supports`, `copy_variant_supports`;
  `variant_base_diff` reports `overrides`, `follow` and `conflicts`. `solve_variants` and `sync_variant_with_base` no longer take
  `support_sets`; `import_model` returns `(unified, overlay, report)` with the imported supports in the variant's model.
- **`solve_all_variants` marks the cases of each variant's own model solved** (a variant is solved on a copy, so they stayed pending).
- **Fix: `variant_combinations` were lost on opening a `.x2d`.** `load_variant_overlays` is run for `variants.json` and then for
  `sequences.json`, and the second, without the key, reset them to `{}`.
- **Smaller `.x2d`.** `save_x2d` writes `results.json`, and `structure.json` for a model of more than 3 000 items, without
  indentation (the indented encoder is pure Python and about ten times slower). Reading is unchanged.

## 2026.10.27 — 2026-10-27
- **Analysis and design without a window (`xdfem2d.workflows`).** The checks and the calculation that the application's Run and
  Design menu did inside the window, now as functions that need no Qt, so a script can do the same:
  - `validate_for_analysis(struc)`: what stops a model from being analysed, as `[{'code', 'msg'}, ...]` (empty when it can run);
  - `run_analysis(struc, scope="active" | "all", results=None)` -> `AnalysisOutcome`: solves the model, and with
    `scope="all"` also its variants and the combinations across variants (`solve_all_variants` -> `VariantRun`) and the
    construction sequences (`run_sequences`); a solved model gets only its pending analysis cases solved;
  - `run_design(struc, results, kind, prefs=None, combinations=None, sls_combinations=None)`: designs the members of a
    material and stores the outcome in `results` where `get design` and the design views read it. `kind` is one of
    `DESIGN_KINDS`: `"concrete"` (EC2 beams and column checks), `"bars"` (the bars of each beam by zone, a proposal),
    `"areas"` (slabs and walls), `"crack"` (EC2 7.3.4), `"punching"` (EC2 6.4, plate domain), `"steel"` (EC3), `"timber"`
    (EC5). It returns `{'kind', 'designed', 'failing', 'combinations'}`, and raises `ValueError` for an unknown kind, missing
    results or no combination to design for;
  - `available_combinations(struc, results)` and `vertical_support_nodes(struc)` (the nodes a punching column can sit on).
  The application's window calls the same functions, so the two cannot differ.
- **Automatic EN 1990 combinations (`xdfem2d.combinations`).** `auto_combinations(struc, types=("ULS",), replace=True)` adds the
  combinations of the analysis cases whose action type is unambiguous (`pure_cases(struc)`) for any of `ULS`, `SLS-K`,
  `SLS-FR` and `SLS-QP`, and returns `{'added', 'skipped', 'removed', 'errors', 'cases'}`. With `replace=True` it first removes
  the ones an earlier run generated (the ids that start with `AUTO_PREFIXES`: `ULS:`, `ULS-E:`, `SLS-K:`, `SLS-FR:`, `SLS-QP:`),
  and never a combination you named.
- **Unique ids.** `add_node` and `rename_node` refuse an id that a bar, a triangle, a quad or a geometry object has, and the elements
  refuse the id of a node: a node and an element can no longer share a name. The `rename_*` methods check the other kinds as well.
- **Edge loads in the global axes.** `create_edge_load(object_id, edge, load_case, p, direction)` takes `'x'` and `'y'` as well as
  `'perp'` and `'tang'`: `p` is signed, the load is stored as `fx` / `fy` in the global axes (`coord_sys = 'global'`).
- **Parameters of the EC8 spectrum (`xdfem2d.seismic`).** `ec8_parameters(code, soil, zone, importance="ii")` gives
  `{'ag', 'S', 'TB', 'TC', 'TD'}` for `EC8_CODES` (`CEN-1`, `CEN-2`, `PT-1`, `PT-2`) from `eurocodepy.ec8.get_spectrum_parameters`:
  `ag` multiplied by the importance factor and, in the Portuguese annex, `S` reduced as `ag` grows. `EC8_IMPORTANCE` is
  `('i', 'ii', 'iii', 'iv')`. `ec8_spectrum` and `ec8_design_spectrum` sample the spectra as before.

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
