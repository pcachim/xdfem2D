"""
Structure2D — main container class for xdfem2D analysis.
"""
from __future__ import annotations
import math
from typing import Optional


def _opt_float(v):
    """A float, or None for "not set" (None and the empty string)."""
    if v is None or v == "":
        return None
    return float(v)
from . import references as _refs
from .models import (
    Node, Material, Section, BarElement, TriElement, TriSection,
    QuadElement, QuadSection, QuadAreaLoad,
    Support, SupportAssignment,
    NodeSpring,
    ElementSpring,
    LoadCase, PointLoad, DistributedLoad, ElementPointLoad, LoadCombination,
    SupportSettlement, TemperatureLoad, TriTemperatureLoad,
    QuadTemperatureLoad,
    AreaTemperatureLoad, LineTemperatureLoad, LineDistributedLoad,
    LineElementSpring,
    AnalysisCase, NodalMass, SpectralFunction,
    ConcreteMaterial,
    GeoArc, GeoMultisegment, GeoSegment, GeoRectangle, GeoPolygon,
    Constraint, CONSTRAINT_COMPONENTS,
    Cut,
)


class Structure2D:
    """
    Holds all structural data and drives the analysis pipeline.

    Usage
    -----
    struc = Structure2D()
    struc.add_node(...)
    struc.add_material(...)
    ...
    results = struc.calculate()
    """

    def __init__(self, domain: str = "plane"):
        # Analysis domain — "plane" (frames + membranes: ux, uy, tz) or
        # "plate" (grillages + plate bending: w, tx, ty). One per model,
        # chosen at creation and saved with the file; absent in a file means
        # "plane", so every model saved before the field existed keeps its
        # meaning. Both domains carry 3 DOFs/node, so DOF numbering and the
        # shapes of K/F/U are identical — only the component semantics change.
        from .models import DOMAINS
        if domain not in DOMAINS:
            raise ValueError(
                f"Unknown domain '{domain}' (expected one of {DOMAINS}).")
        self.domain:                  str                           = domain
        self.nodes:                   dict[str, Node]               = {}
        self.materials:               dict[str, Material]           = {}
        self.sections:                dict[str, Section]            = {}
        self.bar_elements:            list[BarElement]              = []
        self.bar_elements_by_id:      dict[str, BarElement]         = {}
        self.tri_sections:            dict[str, TriSection]         = {}
        self.tri_elements:            list[TriElement]              = []
        self.tri_elements_by_id:      dict[str, TriElement]         = {}
        self.quad_sections:           dict[str, QuadSection]        = {}
        self.quad_elements:           list[QuadElement]             = []
        self.quad_elements_by_id:     dict[str, QuadElement]        = {}
        self.quad_area_loads:         list                          = []  # plate: pz per quad
        self.quad_edge_loads:         list                          = []
        self.tri_edge_loads:          list                          = []
        self.surface_edge_loads:      list                          = []
        self.tri_area_loads:          list                          = []  # plate: pz per triangle
        self.surface_area_loads:      list                          = []  # plate: pz per surface object
        self.tri_area_springs:        list                          = []  # plate: Winkler kz per triangle
        self.quad_area_springs:       list                          = []  # plate: Winkler kz per quad
        self.surface_area_springs:    list                          = []  # plate: Winkler kz per surface object
        self.punch_columns:           list                          = []  # plate: EC2 §6.4 punching columns
        self.fields:                  dict                          = {}   # name → Field
        self.supports:                dict[str, Support]            = {}
        self.support_assignments:     list[SupportAssignment]       = []
        self.node_springs:            dict[str, NodeSpring]         = {}  # keyed by node_id
        self.element_springs:         dict[str, ElementSpring]      = {}  # keyed by element_id
        self.constraints:             dict[str, Constraint]         = {}  # keyed by constraint id
        self.load_cases:              list[LoadCase]                = []
        self.load_cases_by_id:        dict[str, LoadCase]           = {}
        self.point_loads:             list[PointLoad]               = []
        self.distributed_loads:       list[DistributedLoad]         = []
        self.element_point_loads:     list[ElementPointLoad]        = []
        self.load_combinations:       list[LoadCombination]         = []
        self.cuts:                    list[Cut]                     = []  # user-defined section cuts
        self.support_settlements:     list[SupportSettlement]       = []
        self.temperature_loads:       list[TemperatureLoad]         = []
        self.tri_temperature_loads:   list[TriTemperatureLoad]      = []
        self.quad_temperature_loads:  list[QuadTemperatureLoad]     = []
        self.area_temperature_loads:  list[AreaTemperatureLoad]     = []
        self.line_temperature_loads:  list[LineTemperatureLoad]     = []
        self.line_distributed_loads:  list[LineDistributedLoad]     = []
        self.line_element_springs:    list[LineElementSpring]       = []
        self.analysis_cases:          list[AnalysisCase]            = []
        self.analysis_cases_by_id:    dict[str, AnalysisCase]       = {}
        self.nodal_masses:            list[NodalMass]               = []
        self.spectral_functions:      dict[str, SpectralFunction]   = {}
        # Objects (parametric geometry, discretised at solve time).
        self.geometry_objects:          dict                          = {}  # {id: GeoArc|GeoMultisegment}
        # Beam registry for the beam-bars design: {tag: {"name": str,
        # "overrides": {BeamBarParams field: value}}}. Membership is the
        # ``beam`` tag on each bar; this holds only what the tag cannot
        # (display name, parameter overrides). See dev/BEAM_BARS_DEFINITION.md.
        self.beams:                     dict                          = {}
        # The user's bar decisions per beam (dev/BEAM_BARS_DEFINITION.md §21):
        # a detailing.DetailDocument. Only beams with a manual zone are kept;
        # automatic proposals are regenerated by every design run. Travels in
        # the model dict (so undo/redo keeps it) and, in a .x2d, in its own
        # ``beam_detail.json`` entry.
        from .detailing import DetailDocument
        self.beam_detail:               DetailDocument                = DetailDocument()
        # why a stored detail could not be read on load ("" = fine)
        self.beam_detail_load_error:    str                           = ""
        self.concrete_materials:      dict[str, ConcreteMaterial]   = {}
        # Variants / phasing (Phases 1–4). Optional overlays on this entity
        # space; empty by default so existing models are unaffected.
        self.support_sets:            dict                          = {}  # {id: SupportSet}
        self.variants:                dict                          = {}  # {id: Variant}
        self.construction_sequences:  dict                          = {}  # {id: ConstructionSequence}
        # Saved variant combinations: {id: {'op': str, 'terms': [(variant, case, factor), ...]}}
        self.variant_combinations:    dict                          = {}
        # Free-form project metadata (ordered): fixed fields first, then any
        # user-added key/value pairs.
        self.project_info:            dict[str, str]                = {
            "Project name": "", "Location": "", "GPS": "",
            "Owner": "", "Designer": "",
        }
        # When a surface object is meshed, restrain the mesh nodes that fall
        # between two consecutive defining nodes carrying the same support.
        #
        # Without it a meshed wall could only ever be held at its corners: the
        # nodes along an edge do not exist until the mesh is built, so "fixed
        # along the base" was not expressible at all. A language model asked
        # for exactly that supported the two base corners, which was not a
        # mistake on its part — it was the whole of what the format allowed.
        #
        # Kept on the structure rather than in the application's preferences so
        # that a file gives the same answer wherever it is opened.
        self.propagate_edge_supports: bool                          = True
        # The same idea for node springs: two consecutive corners of a surface
        # carrying the same spring propagate it to the mesh nodes along that
        # edge — an elastic edge, the way propagate_edge_supports gives a
        # restrained one.
        self.propagate_edge_springs:  bool                          = True

        # Internal caches (populated during analysis)
        self._elem_cache:       dict          = {}
        self._fixed_end_forces: dict          = {}
        self._elem_eqload:      dict          = {}
        self._node_dof_index:   Optional[dict] = None
        self._num_dofs:         int           = 0

    # ------------------------------------------------------------------
    # Add methods
    # ------------------------------------------------------------------

    def add_node(self, id: str, x: float, y: float) -> Node:
        if not isinstance(id, str) or not id.strip():
            # A node with id None used to be stored under the key None: nothing
            # complained until the GUI tried to list it. create_node numbers a
            # node itself.
            raise ValueError(
                f"Node id must be a text such as 'N1', not {id!r}: use "
                "create_node(x, y) to have it numbered automatically.")
        if id in self.nodes:
            raise ValueError(f"Node '{id}' already exists.")
        n = Node(id=id, x=x, y=y)
        self.nodes[id] = n
        self._node_dof_index = None
        return n

    def add_material(self, name: str, elastic_modulus: float, unit_weight: float,
                     alpha: float = 1.0e-5, unit_mass=None,
                     material_type=None, design=None, poisson: float = 0.2) -> Material:
        from .models import MaterialType
        mt = MaterialType(material_type) if material_type else MaterialType.CONCRETE
        # Validated against the CALLER'S OWN design dict, before defaults are
        # merged in, and only when there is one: default_design_for's own
        # constants ('C30/37', 'A500NR', 'S275', 'C24') are always valid by
        # construction, and every add_material call without an explicit
        # design -- the overwhelming majority, template/default_structure
        # included -- would otherwise import xdfem2d.databases (and so need
        # eurocodepy) just to re-confirm a hardcoded literal is itself valid.
        if design:
            self._validate_material_class(mt, design)
        return self._add_material_unchecked(name, elastic_modulus, unit_weight,
                                            alpha=alpha, unit_mass=unit_mass,
                                            material_type=mt, design=design,
                                            poisson=poisson)

    def _add_material_unchecked(self, name: str, elastic_modulus: float, unit_weight: float,
                                alpha: float = 1.0e-5, unit_mass=None,
                                material_type=None, design=None,
                                poisson: float = 0.2) -> Material:
        """add_material's construction, without validating design's class
        label(s) -- used by structure_io's plain loader (26/09/2026), which
        has to stay permissive: a saved file with a stale or hand-edited
        class is the CHECKED loader's job to flag (see
        test_structure_io_checked.py), not the plain open's job to refuse.
        Leading underscore also keeps it off api_reference_tool (see
        _api_entries: any name starting with '_' is skipped), so a script
        can never reach for this to bypass add_material's own check.
        """
        from .models import MaterialType, default_design_for
        mt = MaterialType(material_type) if material_type else MaterialType.CONCRETE
        # A concrete (or steel) material must carry the design strengths the RC
        # design needs; fill any missing key with the type's defaults so the
        # dimensioning always has fck/fyk (etc.) to work with, whatever the
        # caller passed (templates, GUI, import).
        des = dict(design) if design else {}
        for k, v in default_design_for(mt, des).items():
            des.setdefault(k, v)
        m = Material(name=name, elastic_modulus=elastic_modulus, unit_weight=unit_weight,
                     alpha=alpha, unit_mass=unit_mass, material_type=mt,
                     poisson=poisson, design=des)
        self.materials[name] = m
        return m

    @staticmethod
    def _validate_material_class(mt, des: dict) -> None:
        """The Eurocode class label(s) in *des* (the caller's OWN design
        dict, not yet merged with the type's defaults), if given, must name
        a real grade -- not the material's own name (26/09/2026, Matias:
        "o que é restrito é a classe do material definidas nas propriedades
        do material", not the name).

        create_rc_material/create_steel_material/create_timber_material
        already only ever put a class here that they first validated
        against these same tables (raising "unknown concrete/steel/timber
        ..." otherwise) -- add_material is the one path that takes a
        hand-written design dict straight from the caller, so without this
        a material can carry a class_conc/class/class_reinf that names
        nothing real (or names the wrong grade for its own fck/fy/fck
        numbers), and every design check or report that reads it (see
        rc_design._section_strengths) reads a value that looks right and
        is not.
        """
        from .models import MaterialType
        # Each databases import stays INSIDE its own "there is actually a
        # class to check" branch -- not just inside the CONCRETE/STEEL/
        # TIMBER branch -- because a design dict without a class key (b/h
        # overrides, fck/fu given numerically, ...) is the common case for
        # an explicit design= (see _tpl_wall_obj and friends), and importing
        # xdfem2d.databases at all needs eurocodepy: confirmed live, moving
        # the import up to the type-branch alone still broke every template
        # test that passes its own design dict without a class in it.
        if mt == MaterialType.CONCRETE:
            cls = des.get('class_conc')
            if cls is not None:
                from .databases import concrete_design_grades
                if cls not in concrete_design_grades():
                    raise ValueError(
                        f"unknown concrete class '{cls}' -- valid classes: "
                        f"{', '.join(sorted(concrete_design_grades()))}")
            reinf = des.get('class_reinf')
            if reinf is not None:
                from .databases import reinforcement_design_grades
                if reinf not in reinforcement_design_grades():
                    raise ValueError(
                        f"unknown reinforcement grade '{reinf}' -- valid "
                        f"grades: {', '.join(sorted(reinforcement_design_grades()))}")
        elif mt == MaterialType.STEEL:
            cls = des.get('class')
            if cls is not None:
                from .databases import steel_design_grades
                if cls not in steel_design_grades():
                    raise ValueError(
                        f"unknown steel grade '{cls}' -- valid grades: "
                        f"{', '.join(sorted(steel_design_grades()))}")
        elif mt == MaterialType.TIMBER:
            cls = des.get('class')
            if cls is not None:
                from .databases import timber_design_grades
                if cls not in timber_design_grades():
                    raise ValueError(f"unknown timber grade '{cls}'")

    def add_section(self, name: str, material_name: str, b: float, h: float,
                    area_override=None, inertia_override=None,
                    profile_name=None, section_type=None,
                    rc_cover=0.045, rc_alpha_s=90.0, rc_bar_phi=16.0,
                    shape=None, tw=0.0, tf=0.0, rc_shear_min=True,
                    torsion_override=None, angle=0.0,
                    inertia_minor_override=None,
                    wel_y_override=None, wpl_y_override=None,
                    wel_z_override=None, wpl_z_override=None,
                    av_y_override=None, av_z_override=None,
                    warping_override=None, wt_override=None,
                    curve_y_override=None, curve_z_override=None,
                    timber_service_class="SC1",
                    is_column: bool = False, rc_phi_ef: float = 2.0,
                    rc_n_bars: int = 4,
                    rc_n_bars_y: int = 2, rc_n_bars_z: int = 2,
                    rc_second_order_method: str = "nominal_curvature",
                    rc_torsion_distribution: str = "top_bottom",
                    beam_overrides: Optional[dict] = None,
                    ) -> Section:
        # ``section_type`` is accepted for backward compatibility but ignored —
        # the design type is now derived from the material (see section_type_of).
        # ``torsion_override`` sets the St-Venant torsion constant J [m⁴] used
        # by grillage bars (plate domain); by default J is derived from the
        # shape (see models.section_torsion_constant).
        from .models import SectionShape
        if material_name not in self.materials:
            raise ValueError(f"Material '{material_name}' not found.")
        sh = SectionShape(shape) if shape else SectionShape.GENERIC
        s = Section(name=name, material_name=material_name, b=b, h=h,
                    area_override=area_override, inertia_override=inertia_override,
                    inertia_minor_override=inertia_minor_override,
                    torsion_override=torsion_override,
                    wel_y_override=wel_y_override, wpl_y_override=wpl_y_override,
                    wel_z_override=wel_z_override, wpl_z_override=wpl_z_override,
                    av_y_override=av_y_override, av_z_override=av_z_override,
                    warping_override=warping_override,
                    wt_override=wt_override,
                    curve_y_override=curve_y_override,
                    curve_z_override=curve_z_override,
                    profile_name=profile_name, angle=angle,
                    shape=sh, tw=tw, tf=tf,
                    rc_cover=rc_cover, rc_alpha_s=rc_alpha_s,
                    rc_bar_phi=rc_bar_phi, rc_shear_min=rc_shear_min,
                    timber_service_class=timber_service_class,
                    is_column=is_column, rc_phi_ef=rc_phi_ef,
                    rc_n_bars=rc_n_bars,
                    rc_n_bars_y=rc_n_bars_y, rc_n_bars_z=rc_n_bars_z,
                    rc_second_order_method=rc_second_order_method,
                    rc_torsion_distribution=rc_torsion_distribution,
                    beam_overrides=(dict(beam_overrides)
                                    if beam_overrides else None))
        self.sections[name] = s
        return s

    def _check_id_not_used_elsewhere(self, id: str, kind_label: str, exclude: str) -> None:
        """Cross-check *id* against the OTHER element/geometry-object id
        namespaces (bar_elements_by_id, tri_elements_by_id,
        quad_elements_by_id, geometry_objects). ``exclude`` names the
        namespace the caller already self-checked (e.g. "bar") so it is
        skipped here. Raises ValueError naming which existing kind of thing
        already holds the id, matching the wording already used by the
        bar/tri/quad cross-checks (dev/refactor_area_path.md item F /
        IMPLEMENT_QUAD.md Phase 5) — now extended to geometry objects too,
        since those were not previously cross-checked against any of the
        three element kinds.
        """
        checks = [("bar", self.bar_elements_by_id, "a bar"),
                  ("tri", self.tri_elements_by_id, "a triangle"),
                  ("quad", self.quad_elements_by_id, "a quad"),
                  ("geo", self.geometry_objects, "a geometry object")]
        for key, d, desc in checks:
            if key == exclude:
                continue
            if id in d:
                raise ValueError(
                    f"{kind_label} '{id}' rejected: {desc} already uses this id.")


    def add_bar_element(self, id: str, node_i: str, node_j: str,
                        section_name: str,
                        hinge_i=False, hinge_j=False,
                        rc_design=False, rc_cover=0.025,
                        rc_cotg_theta=1.0, rc_alpha_s=90.0,
                        sd_ky=None, sd_kz=None, sd_klt=None,
                        sd_ltb=True, stage: int = 1,
                        is_column: Optional[bool] = None,
                        beam: Optional[str] = None,
                        ) -> BarElement:
        if id in self.bar_elements_by_id:
            raise ValueError(f"Element '{id}' already exists.")
        # Reject an id already used by an area element too (dev/refactor_area_
        # path.md item F / implement_quad.md Phase 13): any code that keys a
        # dict by "element id" across kinds (reports, GUI selection, cuts)
        # would otherwise inherit a silent bar/area collision.
        self._check_id_not_used_elsewhere(id, "Element", exclude="bar")
        # ``is_column``: None → follow the section's ``is_column`` default;
        # True/False overrides it for this bar only. ``rc_phi_ef``/
        # ``rc_n_bars``/``rc_second_order_method`` no longer take a per-bar
        # value — they live on the Section only (dev/
        # BUCKLING_COLUMN_PERSISTENCE.md).
        e = BarElement(id=id, node_i=node_i, node_j=node_j,
                       section_name=section_name,
                       hinge_i=hinge_i, hinge_j=hinge_j,
                       rc_design=rc_design, rc_cover=rc_cover,
                       rc_cotg_theta=rc_cotg_theta, rc_alpha_s=rc_alpha_s,
                       sd_ky=sd_ky, sd_kz=sd_kz, sd_klt=sd_klt, sd_ltb=sd_ltb,
                       is_column=is_column, beam=beam)
        e.stage = stage
        self.bar_elements.append(e)
        self.bar_elements_by_id[id] = e
        return e

    def add_tri_section(self, name: str, material_name: str,
                        thickness: float = 0.1, plane_strain: bool = False,
                        formulation: str = "CST",
                        rc_cover: float = 0.045, rc_alpha_s: float = 90.0,
                        rc_bar_phi: float = 16.0,
                        rc_shear_min: bool = True,
                        rc_cover_top_x: float = None,
                        rc_cover_top_y: float = None,
                        rc_cover_bot_x: float = None,
                        rc_cover_bot_y: float = None) -> TriSection:
        """Define a triangle section. ``rc_cover`` is the single mechanical
        cover; a slab may instead give per-face, per-direction covers
        (rc_cover_top_x/top_y/bot_x/bot_y), each falling back to ``rc_cover``
        when left None (see :meth:`TriSection.resolved_covers`)."""
        if material_name not in self.materials:
            raise ValueError(f"Material '{material_name}' not found.")
        if formulation not in ("CST", "Allman", "ES-FEM", "DKT", "MITC3"):
            raise ValueError(
                f"Unknown tri section formulation '{formulation}' "
                "(expected 'CST', 'Allman', 'ES-FEM', 'DKT' or 'MITC3').")
        ts = TriSection(name=name, material_name=material_name,
                        thickness=thickness, plane_strain=plane_strain,
                        formulation=formulation,
                        rc_cover=rc_cover, rc_alpha_s=rc_alpha_s,
                        rc_bar_phi=rc_bar_phi, rc_shear_min=rc_shear_min,
                        rc_cover_top_x=rc_cover_top_x,
                        rc_cover_top_y=rc_cover_top_y,
                        rc_cover_bot_x=rc_cover_bot_x,
                        rc_cover_bot_y=rc_cover_bot_y)
        self.tri_sections[name] = ts
        return ts

    def add_plate_section(self, name: str, material_name: str,
                          thickness: float = 0.2, formulation: str = "MITC3",
                          **kw) -> TriSection:
        """Sugar for a plate-bending triangle section — plate domain.

        Defaults to the shear-deformable **MITC3** element (valid for thin and
        thick slabs); pass ``formulation='DKT'`` for the thin-plate (Kirchhoff)
        element instead. thickness in m.
        """
        return self.add_tri_section(name, material_name, thickness=thickness,
                                    formulation=formulation, **kw)

    def remove_tri_section(self, name: str):
        self.tri_sections.pop(name, None)

    def add_quad_section(self, name: str, material_name: str,
                         thickness: float = 0.1, plane_strain: bool = False,
                         formulation: str = "MITC4",
                         rc_cover: float = 0.045, rc_alpha_s: float = 90.0,
                         rc_bar_phi: float = 16.0,
                         rc_shear_min: bool = True,
                         rc_cover_top_x: float = None,
                         rc_cover_top_y: float = None,
                         rc_cover_bot_x: float = None,
                         rc_cover_bot_y: float = None) -> QuadSection:
        """Define a quadrilateral section (dev/IMPLEMENT_QUAD.md Phase 3) —
        the 4-node analogue of :meth:`add_tri_section`. ``rc_cover`` is the
        single mechanical cover; a slab may instead give per-face,
        per-direction covers, each falling back to ``rc_cover`` when left
        None (see :meth:`QuadSection.resolved_covers`)."""
        if material_name not in self.materials:
            raise ValueError(f"Material '{material_name}' not found.")
        if formulation not in ("Q4", "QM6", "DKT4", "MITC4"):
            raise ValueError(
                f"Unknown quad section formulation '{formulation}' "
                "(expected 'Q4', 'QM6', 'DKT4' or 'MITC4').")
        qs = QuadSection(name=name, material_name=material_name,
                         thickness=thickness, plane_strain=plane_strain,
                         formulation=formulation,
                         rc_cover=rc_cover, rc_alpha_s=rc_alpha_s,
                         rc_bar_phi=rc_bar_phi, rc_shear_min=rc_shear_min,
                         rc_cover_top_x=rc_cover_top_x,
                         rc_cover_top_y=rc_cover_top_y,
                         rc_cover_bot_x=rc_cover_bot_x,
                         rc_cover_bot_y=rc_cover_bot_y)
        self.quad_sections[name] = qs
        return qs

    def remove_quad_section(self, name: str):
        self.quad_sections.pop(name, None)

    def rename_quad_section(self, old: str, new: str):
        """The quad counterpart of :meth:`rename_tri_section` (added
        dev/IMPLEMENT_QUAD.md Phase 8, GUI item 2 — the QuadSection manager
        panel mirrors TriSectionPanel's rename-on-name-change edit flow, and
        that flow needs this to exist)."""
        if old not in self.quad_sections or new in self.quad_sections:
            return
        qs = self.quad_sections.pop(old); qs.name = new
        self.quad_sections[new] = qs
        for q in self.quad_elements:
            if q.section_name == old:
                q.section_name = new

    # add_tri_section/add_quad_section/add_plate_section stay exactly as
    # they are — every loader, importer and template in this codebase
    # already knows which shape it is building and calls the right one
    # directly, with no ambiguity to resolve. add_area_section is a NEW,
    # additional entry point for the one caller that does not know in
    # advance which shape a name will end up serving: the create_* layer
    # (and the GUI's "Panel" section manager, which already built this
    # same pair by hand). Creating both halves together, under one name,
    # is what makes create_area_element safe regardless of whether it
    # picks add_tri_element or add_quad_element — the section it passes
    # through always resolves either way.
    def add_area_section(self, name: str, material_name: str,
                         thickness: float = 0.1, plane_strain: bool = False,
                         formulation: str = "CST", quad_formulation: str = "QM6",
                         rc_cover: float = 0.045, rc_alpha_s: float = 90.0,
                         rc_bar_phi: float = 16.0, rc_shear_min: bool = True,
                         rc_cover_top_x: float = None, rc_cover_top_y: float = None,
                         rc_cover_bot_x: float = None, rc_cover_bot_y: float = None
                         ) -> TriSection:
        """Creates a same-named TriSection + QuadSection pair. Returns the
        TriSection; the QuadSection is ``self.quad_sections[name]``. If the
        quad half fails to validate (e.g. a bad ``quad_formulation``), the
        tri half already created is rolled back — never leaves an unpaired
        section behind from a call that raised."""
        ts = self.add_tri_section(name, material_name, thickness=thickness,
                                  plane_strain=plane_strain, formulation=formulation,
                                  rc_cover=rc_cover, rc_alpha_s=rc_alpha_s,
                                  rc_bar_phi=rc_bar_phi, rc_shear_min=rc_shear_min,
                                  rc_cover_top_x=rc_cover_top_x, rc_cover_top_y=rc_cover_top_y,
                                  rc_cover_bot_x=rc_cover_bot_x, rc_cover_bot_y=rc_cover_bot_y)
        try:
            self.add_quad_section(name, material_name, thickness=thickness,
                                  plane_strain=plane_strain, formulation=quad_formulation,
                                  rc_cover=rc_cover, rc_alpha_s=rc_alpha_s,
                                  rc_bar_phi=rc_bar_phi, rc_shear_min=rc_shear_min,
                                  rc_cover_top_x=rc_cover_top_x, rc_cover_top_y=rc_cover_top_y,
                                  rc_cover_bot_x=rc_cover_bot_x, rc_cover_bot_y=rc_cover_bot_y)
        except ValueError:
            self.tri_sections.pop(name, None)
            raise
        return ts

    def remove_area_section(self, name: str):
        """Removes both halves of a paired area section. Tolerant of
        either half already being missing (a model saved before sections
        were always paired may have an unpaired one)."""
        _refs.remove(self, 'area_section', name)   # refused while in use
        self.remove_tri_section(name)
        self.remove_quad_section(name)

    def rename_area_section(self, old: str, new: str):
        """Renames both halves of a paired area section together.
        rename_tri_section/rename_quad_section are each already a no-op
        when ``old`` is not in their own registry, so calling both here is
        safe for a legacy unpaired section too."""
        self.rename_tri_section(old, new)
        self.rename_quad_section(old, new)
        # surface objects name their section too
        _refs.rename(self, 'area_section', old, new)

    # The non-raising counterpart of check_references, and the wider one.
    # Material and section references were always caught, because assembly
    # indexes them directly and a bad name is a KeyError in the stiffness loop.
    # The references between *cases* are worse: an analysis case or a
    # combination whose coefficients name something that is not there is not an
    # error to the solver, it is a sum with a missing term, and the result is
    # quietly zero. A four-span beam whose ULS case combined a load case 'G'
    # that had been named 'ULS' drew nothing, ran without complaint, and gave
    # no hint why. So every reference the solve depends on is checked here, in
    # one pass. Element/node existence is only decidable on the editable mesh —
    # object models generate both at solve time — so those checks are skipped
    # when geometry objects are present; the top-level case/load references are
    # still checked.
    def reference_problems(self) -> list[str]:
        """Return a list of dangling cross-references (case, load, material,
        section, support, spring), each as a readable ``'A → B'`` line. Empty
        when the model is consistent. Non-raising; :func:`check_references`
        raises the same findings for the solve path."""
        P: list[str] = []
        mats, secs, tsecs = self.materials, self.sections, self.tri_sections
        lcs = set(self.load_cases_by_id)
        acs = set(self.analysis_cases_by_id)
        combos = {c.id for c in self.load_combinations}
        specs = set(self.spectral_functions)

        # ── Material / section (the original coverage) ────────────
        for sc in secs.values():
            if sc.material_name not in mats:
                P.append(f"section '{sc.name}' → material '{sc.material_name}'")
        for ts in tsecs.values():
            if ts.material_name not in mats:
                P.append(f"triangle section '{ts.name}' → material "
                         f"'{ts.material_name}'")
        for e in self.bar_elements:
            if e.section_name not in secs:
                P.append(f"bar '{e.id}' → section '{e.section_name}'")
        for t in self.tri_elements:
            if t.section_name not in tsecs:
                P.append(f"triangle '{t.id}' → triangle section "
                         f"'{t.section_name}'")
        qsecs = self.quad_sections
        for qs in qsecs.values():
            if qs.material_name not in mats:
                P.append(f"quad section '{qs.name}' → material "
                         f"'{qs.material_name}'")
        for q in self.quad_elements:
            if q.section_name not in qsecs:
                P.append(f"quad '{q.id}' → quad section "
                         f"'{q.section_name}'")

        # ── Analysis-case coefficients → load case (or nested case) ──
        for ac in self.analysis_cases:
            kind = getattr(ac, 'analysis_type', 'Linear')
            if kind in ('Linear', 'NonLinear', 'Mass', 'GeometricNonlinear'):
                for key in (getattr(ac, 'coefficients', {}) or {}):
                    if key not in lcs and key not in acs:
                        P.append(f"analysis case '{ac.id}' → load case '{key}'")
            if kind == 'Modal':
                mc = getattr(ac, 'modal_case_id', '')
                if mc and mc not in acs:
                    P.append(f"modal case '{ac.id}' → mass case '{mc}'")
            if kind == 'Spectrum':
                mc = getattr(ac, 'modal_case_id', '')
                sp = getattr(ac, 'spectrum_id', '')
                if mc and mc not in acs:
                    P.append(f"spectrum case '{ac.id}' → modal case '{mc}'")
                if sp and sp not in specs:
                    P.append(f"spectrum case '{ac.id}' → spectral function '{sp}'")
            ss = getattr(ac, 'stored_stiffness_id', '')
            if ss and ss not in acs:
                P.append(f"analysis case '{ac.id}' → stored stiffness '{ss}'")

        # ── Combination coefficients → analysis case (or nested) ──
        for c in self.load_combinations:
            for key in (getattr(c, 'coefficients', {}) or {}):
                if key not in acs and key not in combos and key not in lcs:
                    P.append(f"combination '{c.id}' → analysis case '{key}'")

        # ── Loads and masses → load / mass case ───────────────────
        for p in self.point_loads:
            if p.load_case_id not in lcs:
                P.append(f"point load on node '{p.node_id}' → load case "
                         f"'{p.load_case_id}'")
        for d in self.distributed_loads:
            if d.load_case_id not in lcs:
                P.append(f"distributed load on '{d.element_id}' → load case "
                         f"'{d.load_case_id}'")
        for ep in self.element_point_loads:
            if ep.load_case_id not in lcs:
                P.append(f"element point load on '{ep.element_id}' → load case "
                         f"'{ep.load_case_id}'")
        for nm in self.nodal_masses:
            if nm.mass_case_id not in acs:
                P.append(f"nodal mass on '{nm.node_id}' → mass case "
                         f"'{nm.mass_case_id}'")

        # ── Element / node existence (editable mesh only) ─────────
        if not self.geometry_objects:
            elem_ids = {e.id for e in self.bar_elements}
            node_ids = set(self.nodes)
            for d in self.distributed_loads:
                if d.element_id not in elem_ids:
                    P.append(f"distributed load → element '{d.element_id}'")
            for ep in self.element_point_loads:
                if ep.element_id not in elem_ids:
                    P.append(f"element point load → element '{ep.element_id}'")
            for p in self.point_loads:
                if p.node_id not in node_ids:
                    P.append(f"point load → node '{p.node_id}'")
            for a in self.support_assignments:
                if a.node_id not in node_ids:
                    P.append(f"support assignment → node '{a.node_id}'")
                if a.support_name not in self.supports:
                    P.append(f"support on '{a.node_id}' → definition "
                             f"'{a.support_name}'")
            for nid in self.node_springs:
                if nid not in node_ids:
                    P.append(f"node spring → node '{nid}'")
            for eid in self.element_springs:
                if eid not in elem_ids:
                    P.append(f"element spring → element '{eid}'")
        # Every other reference in the inventory (references.SITES): loads on a
        # load case that is gone, elements on a missing node, ...
        P.extend(_refs.dangling(self))
        return P

    # ── Element / area convenience accessors ──────────────────────────
    # A single source of truth for "does the model have elements" and "iterate
    # every area element", so the GUI (analysis guard, exporters, load pickers,
    # load managers) never again counts triangles while forgetting quads.

    def element_count(self) -> int:
        """Total finite elements: bars + triangles + quads."""
        return (len(self.bar_elements) + len(self.tri_elements)
                + len(self.quad_elements))

    def has_elements(self) -> bool:
        """True when the model has at least one bar, triangle or quad."""
        return bool(self.bar_elements or self.tri_elements
                    or self.quad_elements)

    def iter_area_elements(self):
        """Yield (id, kind, element) for every area element — triangles then
        quads — with *kind* in {'tri', 'quad'}. Bars are not area elements."""
        for t in self.tri_elements:
            yield t.id, 'tri', t
        for q in self.quad_elements:
            yield q.id, 'quad', q

    def area_element_ids(self) -> list[str]:
        """Ids of every area element (triangles then quads)."""
        return list(self.tri_elements_by_id) + list(self.quad_elements_by_id)

    def iter_area_loads(self):
        """Yield (kind, load) for every transverse area pressure — per-triangle,
        per-quad and per-surface-object — with *kind* in
        {'tri', 'quad', 'area'}."""
        for a in getattr(self, 'tri_area_loads', []) or []:
            yield 'tri', a
        for a in getattr(self, 'quad_area_loads', []) or []:
            yield 'quad', a
        for a in getattr(self, 'surface_area_loads', []) or []:
            yield 'area', a

    def iter_area_temperature_loads(self):
        """Yield (kind, load) for every area temperature load — per-triangle,
        per-quad and per-object — with *kind* in {'tri', 'quad', 'area'}."""
        for tl in getattr(self, 'tri_temperature_loads', []) or []:
            yield 'tri', tl
        for tl in getattr(self, 'quad_temperature_loads', []) or []:
            yield 'quad', tl
        for tl in getattr(self, 'area_temperature_loads', []) or []:
            yield 'area', tl

    def iter_area_edge_loads(self):
        """Yield (kind, load) for every edge load — per-triangle, per-quad and
        per-surface-object — with *kind* in {'tri', 'quad', 'surface'}."""
        for e in getattr(self, 'tri_edge_loads', []) or []:
            yield 'tri', e
        for e in getattr(self, 'quad_edge_loads', []) or []:
            yield 'quad', e
        for e in getattr(self, 'surface_edge_loads', []) or []:
            yield 'surface', e

    # The domain fixes what every DOF, support, load and result means, so it
    # cannot change under existing nodes, bars, areas or objects: with any of
    # those this raises ValueError (build the other domain's model apart).
    # Without geometry nothing has a meaning to lose. What the model does hold
    # is kept when the domain does not matter to it (materials, bar sections,
    # load cases and combinations, analysis cases) and dropped when it does:
    # support definitions (a PIN is ux, uy in a plane model and w in a plate
    # one) and area sections (their formulations belong to one domain). None of
    # it is attached to a node, since there are no nodes, so nothing dangles.
    def set_domain(self, domain: str) -> dict:
        """Change the domain of a model with no geometry yet.

        Raises ValueError once there are nodes, bars or areas. Drops the
        support definitions and area sections, which depend on the domain, and
        returns their names as {'supports': [...], 'area_sections': [...]}.

        Args:
            domain: ``"plane"`` or ``"plate"``.
        """
        from .models import DOMAINS
        from .template_api import has_geometry
        if domain not in DOMAINS:
            raise ValueError(
                f"Unknown domain '{domain}' (expected one of {DOMAINS}).")
        dropped: dict = {'supports': [], 'area_sections': []}
        if domain == self.domain:
            return dropped
        if has_geometry(self):
            raise ValueError(
                f"This model already has geometry, so its domain ('{self.domain}') "
                f"cannot change to '{domain}': nodes, bars and areas mean "
                f"something else in each. Start a new {domain} model instead.")
        dropped['supports'] = list(self.supports)
        self.supports.clear()
        names = list(dict.fromkeys(
            [*self.tri_sections, *self.quad_sections]))
        dropped['area_sections'] = names
        self.tri_sections.clear()
        self.quad_sections.clear()
        self.domain = domain
        return dropped

    # For plate it names, rather than mis-solves, what is not implemented
    # yet: modal/spectrum/P-Delta arrive with the plate mass model (phase 3);
    # unilateral springs need an out-of-plane projection of their own. Each
    # domain also refuses the other's triangle formulations — a membrane in a
    # plate model (or a DKT in a plane one) would assemble into DOFs whose
    # meaning it does not share. check_references raises these; this lists
    # them.
    def domain_problems(self) -> list[str]:
        """Features the model uses that its analysis domain ('plane' or
        'plate') does not support. Empty when the model is consistent;
        non-raising counterpart of the domain gate in check_references()."""
        P: list[str] = []
        used_secs = {t.section_name for t in self.tri_elements}
        used_qsecs = {q.section_name for q in self.quad_elements}
        # A surface object's tri_section_name resolves against EITHER section
        # table (dev/IMPLEMENT_QUAD.md Phase 6 widened the field to also
        # accept a QuadSection name — see GeoRectangle/GeoPolygon's
        # docstring): sort each referenced name into whichever set matches
        # what it actually resolves to, so a plate-formulation QUAD section
        # referenced by a surface object in a plane-domain model is still
        # flagged below (checking it only against tri_sections, as before
        # Phase 6, would silently miss it). A name that resolves to neither
        # (unset/typo) falls into used_secs, unchanged from before — this
        # method has never validated tri_section_name's existence, that is
        # reference_problems()'s job, and it does not check it either
        # (documented gap, dev/IMPLEMENT_QUAD.md Phase 6 investigation).
        for o in getattr(self, 'geometry_objects', {}).values():
            if type(o).__name__ not in ('GeoRectangle', 'GeoPolygon'):
                continue
            name = getattr(o, 'tri_section_name', '')
            if name in self.quad_sections and name not in self.tri_sections:
                used_qsecs.add(name)
            else:
                used_secs.add(name)
        if getattr(self, 'domain', 'plane') != 'plate':
            for name in sorted(used_secs):
                ts = self.tri_sections.get(name)
                if ts is not None and getattr(ts, 'formulation', 'CST') in (
                        'DKT', 'MITC3'):
                    P.append(f"triangle section '{name}' "
                             f"({getattr(ts, 'formulation', 'CST')}) — plate "
                             "bending needs a plate-domain model")
            for name in sorted(used_qsecs):
                qs = self.quad_sections.get(name)
                if qs is not None and getattr(qs, 'formulation', 'MITC4') in (
                        'DKT4', 'MITC4'):
                    P.append(f"quad section '{name}' "
                             f"({getattr(qs, 'formulation', 'MITC4')}) — plate "
                             "bending needs a plate-domain model")
            if self.tri_area_loads or self.surface_area_loads or self.quad_area_loads:
                P.append("area loads (pz) — transverse pressure only exists "
                         "in the plate domain")
            if (self.tri_area_springs or self.surface_area_springs
                    or self.quad_area_springs):
                P.append("area springs (Winkler kz) — a slab-on-grade "
                         "foundation only exists in the plate domain")
            # P-Delta safety guard (dev/refactor_area_path.md item D-Part 1 /
            # implement_quad.md §5 point 6): the geometric stiffness is
            # assembled ONLY from bar elements today. A plane model with no
            # bars — e.g. a quad/triangle-meshed wall — would otherwise get a
            # zero stiffness matrix that the solver's zero-stiffness auto-pin
            # silently resolves to an all-zero displacement field: a plausible,
            # fully-populated, completely wrong result. Refuse it up front with
            # a clear message instead (this does not add P-Delta capability to
            # area elements — that is the separately-scoped D-Part 2).
            if not self.bar_elements:
                for ac in self.analysis_cases:
                    if ac.analysis_type == 'GeometricNonlinear':
                        P.append(
                            f"analysis case '{ac.id}' (P-Delta) — geometric "
                            "nonlinearity is assembled only from bar elements, "
                            "and this model has none. The result would be a "
                            "meaningless all-zero field; add bar elements, or "
                            "use a Linear/NonLinear analysis case for a "
                            "membrane-only (wall/panel) model")
            return P
        for name in sorted(used_secs):
            ts = self.tri_sections.get(name)
            if ts is not None and getattr(ts, 'formulation', 'CST') not in (
                    'DKT', 'MITC3'):
                P.append(f"triangle section '{name}' "
                         f"({getattr(ts, 'formulation', 'CST')}) — membrane "
                         "formulations belong to the plane domain; a plate "
                         "model's triangles must use 'DKT' or 'MITC3'")
        for name in sorted(used_qsecs):
            qs = self.quad_sections.get(name)
            if qs is not None and getattr(qs, 'formulation', 'MITC4') not in (
                    'DKT4', 'MITC4'):
                P.append(f"quad section '{name}' "
                         f"({getattr(qs, 'formulation', 'MITC4')}) — membrane "
                         "formulations belong to the plane domain; a plate "
                         "model's quads must use 'DKT4' or 'MITC4'")
        for ac in self.analysis_cases:
            # Dynamics (Mass / Modal / Spectrum) is available in the plate
            # domain: the mass sits on w and the modes are the out-of-plane
            # (vertical) vibration. Only the geometric-nonlinear (P-Delta) case
            # stays blocked — its geometric stiffness is built from in-plane
            # axial force, which the plate domain does not carry.
            if ac.analysis_type == 'GeometricNonlinear':
                P.append(f"analysis case '{ac.id}' (P-Delta) — geometric "
                         "nonlinearity uses in-plane axial force and is not "
                         "available in the plate domain")
        for sp in self.node_springs.values():
            if sp.mode_x != 'both' or sp.mode_y != 'both':
                P.append(f"node spring at '{sp.node_id}' — unilateral "
                         "(tension/compression-only) modes are not available "
                         "in the plate domain yet")
        for es in self.element_springs.values():
            if es.mode_x != 'both' or es.mode_y != 'both':
                P.append(f"element spring on '{es.element_id}' — unilateral "
                         "modes are not available in the plate domain yet")
        for ls in getattr(self, 'line_element_springs', []):
            if ls.mode_x != 'both' or ls.mode_y != 'both':
                P.append(f"line spring on '{ls.object_id}' — unilateral "
                         "modes are not available in the plate domain yet")
        return P

    def check_references(self):
        """Raise ``ValueError`` if the model carries any dangling reference.

        The gate the solve passes through: see :func:`reference_problems` for
        what is checked and why the case references matter as much as the
        material ones. Runs on every solve, so a model that would compute a
        silent zero stops with a message naming the reference instead.

        Also raises when the model uses a feature its analysis domain does not
        support (see :func:`domain_problems`) — refusing beats mis-solving.
        """
        dp = self.domain_problems()
        if dp:
            raise ValueError(
                f"This model's domain is '{self.domain}' and it uses features "
                "that domain does not support yet:\n  - " + "\n  - ".join(dp))
        problems = self.reference_problems()
        if problems:
            defined = (f"Load cases: {', '.join(sorted(self.load_cases_by_id)) or '(none)'}."
                       f"  Analysis cases: {', '.join(sorted(self.analysis_cases_by_id)) or '(none)'}.")
            raise ValueError(
                "The model references definitions that do not exist:\n  - "
                + "\n  - ".join(problems)
                + f"\n\n{defined}"
                + "\nFix the reference (a case, load case, material or section "
                  "was probably renamed or deleted).")

    def rename_tri_section(self, old: str, new: str):
        if old not in self.tri_sections or new in self.tri_sections:
            return
        ts = self.tri_sections.pop(old); ts.name = new
        self.tri_sections[new] = ts
        for t in self.tri_elements:
            if t.section_name == old:
                t.section_name = new

    def add_tri_element(self, id: str, node_i: str, node_j: str, node_k: str,
                        section_name: str) -> TriElement:
        # A quad with this id is rejected too (not just another triangle):
        # dev/IMPLEMENT_QUAD.md Phase 5 merges tri and quad stress/moment
        # results into the same dict, keyed by element id
        # (tri_elements.tri_stresses_dispatch) — an id shared between a
        # triangle and a quad would silently drop one of the two from that
        # merged dict instead of raising, which is worse than rejecting it
        # here at creation time.
        if id in self.tri_elements_by_id:
            raise ValueError(f"Triangle '{id}' already exists.")
        self._check_id_not_used_elsewhere(id, "Triangle", exclude="tri")
        t = TriElement(id=id, node_i=node_i, node_j=node_j, node_k=node_k,
                       section_name=section_name)
        self.tri_elements.append(t)
        self.tri_elements_by_id[id] = t
        self._node_dof_index = None
        return t

    def tri_section_of(self, tri: TriElement) -> TriSection | None:
        """The TriSection referenced by *tri* (or None if missing)."""
        return self.tri_sections.get(tri.section_name)

    # Geometry (distinct nodes, CCW winding, convex, non-degenerate) is
    # checked eagerly via model_check.quad_geometry_problems and *rejected*,
    # not auto-fixed, so a bad quad never reaches assembly — dev/
    # IMPLEMENT_QUAD.md Phase 3's "err toward rejecting, not silently
    # reordering nodes" decision. If a referenced node does not exist yet the
    # check is skipped (mirrors add_tri_element, which has the same
    # leniency); reference_problems()/check_references() catch a dangling
    # node reference later, at solve time.
    def add_quad_element(self, id: str, node_i: str, node_j: str,
                         node_k: str, node_l: str,
                         section_name: str) -> QuadElement:
        """Define a 4-node quadrilateral element, the analogue of
        :meth:`add_tri_element` (dev/IMPLEMENT_QUAD.md Phase 3)."""
        if id in self.quad_elements_by_id:
            raise ValueError(f"Quad '{id}' already exists.")
        # See the matching check in add_tri_element: a triangle already using
        # this id is rejected too, so the two element kinds can never
        # collide in the merged tri_stress dict (dev/IMPLEMENT_QUAD.md
        # Phase 5).
        self._check_id_not_used_elsewhere(id, "Quad", exclude="quad")
        node_ids = (node_i, node_j, node_k, node_l)
        if len(set(node_ids)) != 4:
            raise ValueError(
                f"Quad '{id}' must reference 4 distinct nodes, got {node_ids}.")
        if all(nid in self.nodes for nid in node_ids):
            coords = [(self.nodes[nid].x, self.nodes[nid].y) for nid in node_ids]
            from .model_check import quad_geometry_problems
            problems = quad_geometry_problems(coords)
            if problems:
                raise ValueError(
                    f"Quad '{id}' rejected: " + "; ".join(problems))
        q = QuadElement(id=id, node_i=node_i, node_j=node_j, node_k=node_k,
                        node_l=node_l, section_name=section_name)
        self.quad_elements.append(q)
        self.quad_elements_by_id[id] = q
        self._node_dof_index = None
        return q

    def quad_section_of(self, quad: QuadElement) -> QuadSection | None:
        """The QuadSection referenced by *quad* (or None if missing)."""
        return self.quad_sections.get(quad.section_name)

    # Same reasoning as add_area_section: add_tri_element/add_quad_element
    # stay untouched for every caller that already knows its shape
    # (loaders, importers, templates). add_area_element is the additional
    # entry point for the one caller that decides shape by argument count
    # rather than by construction — the create_* layer.
    def add_area_element(self, id: str, node_i: str, node_j: str, node_k: str,
                         node_l: str | None = None, *,
                         section_name: str) -> "TriElement | QuadElement":
        """3 nodes -> add_tri_element, 4 (node_l given) -> add_quad_element.
        section_name is keyword-only: with node_l optional, a positional
        section_name would sit in a different slot depending on which
        branch runs, which is exactly the kind of position-dependent
        surprise this method exists to avoid."""
        if node_l is None:
            return self.add_tri_element(id, node_i, node_j, node_k, section_name)
        return self.add_quad_element(id, node_i, node_j, node_k, node_l, section_name)

    def remove_quad_element(self, id: str):
        self.quad_elements = [q for q in self.quad_elements if q.id != id]
        self.quad_elements_by_id.pop(id, None)
        _refs.remove(self, 'quad', id)
        self._node_dof_index = None

    def remove_tri_element(self, id: str):
        self.tri_elements = [t for t in self.tri_elements if t.id != id]
        self.tri_elements_by_id.pop(id, None)
        _refs.remove(self, 'tri', id)

    # ── Neutral "area element" accessors (dev/refactor_area_path.md Phase 1) ──
    # A triangle and a quad are the same modelling role (a Panel/Slab). These
    # thin, derived helpers let collection/selection/verb code treat both as one
    # "area element" and only bifurcate to the tri/quad kernel where the math
    # actually differs. Nothing is stored; element ids are unique across tri and
    # quad (enforced in add_tri_element/add_quad_element), so an id alone
    # identifies the element unambiguously.
    def area_elements(self):
        """Every area element (triangles then quads), in a single iterable."""
        return list(self.tri_elements) + list(self.quad_elements)

    def area_element_by_id(self, eid: str):
        """The tri or quad element with id *eid*, or ``None``."""
        return (self.tri_elements_by_id.get(eid)
                or self.quad_elements_by_id.get(eid))

    def area_kind_of(self, eid: str):
        """``'tri'``, ``'quad'`` or ``None`` for element id *eid*."""
        if eid in self.tri_elements_by_id:
            return 'tri'
        if eid in self.quad_elements_by_id:
            return 'quad'
        return None

    def remove_area_element(self, eid: str) -> bool:
        """Remove the area element *eid* whatever its kind. Returns True if one
        was removed (dispatches to remove_tri_element/remove_quad_element, which
        also purge that element's attached loads/springs/temperatures)."""
        kind = self.area_kind_of(eid)
        if kind == 'tri':
            self.remove_tri_element(eid)
            return True
        if kind == 'quad':
            self.remove_quad_element(eid)
            return True
        return False

    def add_tri_edge_load(self, id: str, tri_id: str, node_a: str, node_b: str,
                          load_case_id: str, fx: float = 0.0, fy: float = 0.0,
                          coord_sys: str = "global", pn: float = 0.0,
                          pt: float = 0.0) -> "TriEdgeLoad":
        from .models import TriEdgeLoad
        self._require_exists(tri_id, self.tri_elements_by_id, "triangle",
                             "add_tri_edge_load")
        e = TriEdgeLoad(id=id, tri_id=tri_id, node_a=node_a, node_b=node_b,
                        load_case_id=load_case_id, fx=fx, fy=fy,
                        coord_sys=coord_sys, pn=pn, pt=pt)
        self.tri_edge_loads.append(e)
        return e

    def remove_tri_edge_load(self, id: str):
        self.tri_edge_loads = [e for e in self.tri_edge_loads if e.id != id]

    def add_quad_edge_load(self, id: str, quad_id: str, node_a: str,
                           node_b: str, load_case_id: str, fx: float = 0.0,
                           fy: float = 0.0, coord_sys: str = "global",
                           pn: float = 0.0, pt: float = 0.0) -> "QuadEdgeLoad":
        """Uniform load on one edge of a quad element — the 4-node analogue of
        :meth:`add_tri_edge_load`."""
        from .models import QuadEdgeLoad
        self._require_exists(quad_id, self.quad_elements_by_id, "quad",
                             "add_quad_edge_load")
        e = QuadEdgeLoad(id=id, quad_id=quad_id, node_a=node_a, node_b=node_b,
                         load_case_id=load_case_id, fx=fx, fy=fy,
                         coord_sys=coord_sys, pn=pn, pt=pt)
        self.quad_edge_loads.append(e)
        return e

    def remove_quad_edge_load(self, id: str):
        self.quad_edge_loads = [e for e in self.quad_edge_loads if e.id != id]

    def add_surface_edge_load(self, id: str, object_id: str, node_a: str,
                              node_b: str, load_case_id: str, fx: float = 0.0,
                              fy: float = 0.0, coord_sys: str = "global",
                              pn: float = 0.0, pt: float = 0.0):
        """Edge load on a boundary side (node_a→node_b) of a surface object."""
        from .models import SurfaceEdgeLoad
        self._require_area_object(object_id, "add_surface_edge_load")
        e = SurfaceEdgeLoad(id=id, object_id=object_id, node_a=node_a,
                            node_b=node_b, load_case_id=load_case_id, fx=fx,
                            fy=fy, coord_sys=coord_sys, pn=pn, pt=pt)
        self.surface_edge_loads.append(e)
        return e

    def remove_surface_edge_load(self, id: str):
        self.surface_edge_loads = [e for e in self.surface_edge_loads
                                   if e.id != id]

    # The area equivalent of a bar's add_distributed_load/create_uniform_load:
    # one value spread over the whole target, not per element.
    def add_area_load(self, target_id: str, load_case_id: str,
                      pz: float = 0.0):
        """Uniform transverse pressure pz [kN/m²] on a slab — plate domain.

        target_id is a surface object (applied to every triangle it meshes
        into), a single triangle element id, or a single quad element id
        (dev/IMPLEMENT_QUAD.md Phase 4). Negative pz is downward. Replaces
        any existing area load on the same target and case.
        """
        from .models import TriAreaLoad, SurfaceAreaLoad, QuadAreaLoad
        if target_id in self.geometry_objects:
            self.surface_area_loads = [
                a for a in self.surface_area_loads
                if not (a.object_id == target_id
                        and a.load_case_id == load_case_id)]
            a = SurfaceAreaLoad(object_id=target_id,
                                load_case_id=load_case_id, pz=pz)
            self.surface_area_loads.append(a)
            return a
        if target_id in self.tri_elements_by_id:
            self.tri_area_loads = [
                a for a in self.tri_area_loads
                if not (a.tri_id == target_id
                        and a.load_case_id == load_case_id)]
            a = TriAreaLoad(tri_id=target_id, load_case_id=load_case_id,
                            pz=pz)
            self.tri_area_loads.append(a)
            return a
        if target_id in self.quad_elements_by_id:
            self.quad_area_loads = [
                a for a in self.quad_area_loads
                if not (a.quad_id == target_id
                        and a.load_case_id == load_case_id)]
            a = QuadAreaLoad(quad_id=target_id, load_case_id=load_case_id,
                             pz=pz)
            self.quad_area_loads.append(a)
            return a
        raise ValueError(
            f"'{target_id}' is neither a surface object, a triangle "
            "element, nor a quad element — an area load needs one of the "
            "three.")

    def remove_area_load(self, target_id: str, load_case_id: str = ""):
        """Remove area loads on *target_id* (all cases, or one case)."""
        def _keep(a, tid):
            return not (tid == target_id and
                        (not load_case_id or a.load_case_id == load_case_id))
        self.surface_area_loads = [a for a in self.surface_area_loads
                                   if _keep(a, a.object_id)]
        self.tri_area_loads = [a for a in self.tri_area_loads
                               if _keep(a, a.tri_id)]
        self.quad_area_loads = [a for a in self.quad_area_loads
                                if _keep(a, a.quad_id)]

    # kz [kN/m³] is the modulus of subgrade reaction resisting the slab's
    # transverse deflection w ("slab on grade"); it is lumped kz·A/3 onto the
    # w DOF of each vertex of every triangle in the region, or kz·A/4 for a
    # quad (dev/IMPLEMENT_QUAD.md Phase 6). target_id may be a surface object
    # (applied to every triangle/quad it meshes into), a single triangle
    # element id, or a single quad element id.
    def add_area_spring(self, target_id: str, kz: float = 0.0):
        """Attach a Winkler (elastic-foundation) area spring — plate domain.

        ``kz`` [kN/m³] is the subgrade modulus ("slab on grade"). *target_id*
        is a surface object, a triangle element id, or a quad element id.
        Replaces any existing area spring on the target.
        """
        from .models import TriAreaSpring, QuadAreaSpring, SurfaceAreaSpring
        if target_id in self.geometry_objects:
            self.surface_area_springs = [
                a for a in self.surface_area_springs
                if a.object_id != target_id]
            a = SurfaceAreaSpring(object_id=target_id, kz=kz)
            self.surface_area_springs.append(a)
            return a
        if target_id in self.tri_elements_by_id:
            self.tri_area_springs = [
                a for a in self.tri_area_springs if a.tri_id != target_id]
            a = TriAreaSpring(tri_id=target_id, kz=kz)
            self.tri_area_springs.append(a)
            return a
        if target_id in self.quad_elements_by_id:
            self.quad_area_springs = [
                a for a in self.quad_area_springs if a.quad_id != target_id]
            a = QuadAreaSpring(quad_id=target_id, kz=kz)
            self.quad_area_springs.append(a)
            return a
        raise ValueError(
            f"'{target_id}' is neither a surface object, a triangle "
            "element, nor a quad element — an area spring needs one of "
            "the three.")

    def add_punch_column(self, id: str, node_id: str,
                         shape: str = "rectangular",
                         bx: float = 0.30, by: float = 0.30,
                         position: str = "center", force: float = None,
                         dx: float = 0.0, dy: float = 0.0,
                         beta_min: float = 1.0):
        """Define a column / concentrated support for the EC2 §6.4 punching
        check (plate domain). See :class:`~xdfem2d.models.PunchColumn`.

        The column sits under *node_id*; the punching force is *force* [kN] or,
        when None, the support reaction at that node in the designed combination.
        Replaces any existing column with the same id."""
        from .models import PunchColumn
        if shape not in ("rectangular", "circular"):
            raise ValueError("shape must be 'rectangular' or 'circular'.")
        if position not in ("center", "edgex", "edgey", "corner"):
            raise ValueError("position must be 'center', 'edgex', 'edgey' or "
                             "'corner'.")
        self.punch_columns = [c for c in self.punch_columns if c.id != id]
        col = PunchColumn(id=id, node_id=node_id, shape=shape, bx=bx, by=by,
                          position=position, force=force, dx=dx, dy=dy,
                          beta_min=beta_min)
        self.punch_columns.append(col)
        return col

    def remove_punch_column(self, id: str):
        """Remove the punching column with the given id."""
        self.punch_columns = [c for c in self.punch_columns if c.id != id]

    def remove_area_spring(self, target_id: str):
        """Remove any Winkler area spring on *target_id* (surface, triangle,
        or quad)."""
        self.surface_area_springs = [a for a in self.surface_area_springs
                                     if a.object_id != target_id]
        self.tri_area_springs = [a for a in self.tri_area_springs
                                 if a.tri_id != target_id]
        self.quad_area_springs = [a for a in self.quad_area_springs
                                  if a.quad_id != target_id]

    # ── Scalar fields  value = f(x, y) ─────────────────────────────────
    def add_field(self, name: str, expression: str = "0.0"):
        from .models import Field
        f = Field(name=name, expression=expression)
        self.fields[name] = f
        return f

    def remove_field(self, name: str):
        self.fields.pop(name, None)

    # These are the fields whose *name* is stored and re-evaluated at solve
    # time — line/area temperatures, line distributed loads, line element
    # springs. Editing one changes a solved model's input, so the application
    # locks exactly these when results exist. A field sampled to a value at
    # edit time (a point load's field) stores the number, not the name, and so
    # is not here.
    def used_field_names(self) -> set:
        """Names of fields referenced by object loads and springs."""
        used: set = set()
        for tl in getattr(self, 'line_temperature_loads', []):
            used |= {tl.field_name, getattr(tl, 'grad_field_name', '')}
        for dl in getattr(self, 'line_distributed_loads', []):
            used |= {dl.fx_field, dl.fy_field}
        for ls in getattr(self, 'line_element_springs', []):
            used |= {ls.kx_field, ls.ky_field}
        for at in getattr(self, 'area_temperature_loads', []):
            used |= {at.field_name}
        used.discard('')
        return used

    def used_spectral_function_names(self) -> set:
        """Names of spectral functions referenced by a Spectrum analysis case.

        The spectral analogue of :meth:`used_field_names`."""
        return {ac.spectrum_id for ac in getattr(self, 'analysis_cases', [])
                if getattr(ac, 'spectrum_id', '')}

    def field_node_values(self, name: str) -> dict:
        """{node_id: f(x, y)} for the named field, evaluated at each node."""
        from .models import evaluate_field
        fld = self.fields.get(name)
        if fld is None:
            return {}
        return {nid: evaluate_field(fld.expression, nd.x, nd.y)
                for nid, nd in self.nodes.items()}

    # ── Objects (parametric geometry) ──────────────────────────
    @staticmethod
    def _arc_3pts_from_center(cx, cy, radius, start_angle, end_angle):
        """Three points (start, mid, end) on an arc given centre/radius/angles."""
        import math
        a0 = math.radians(start_angle); a1 = math.radians(end_angle)
        sweep = a1 - a0
        am = a0 + sweep / 2.0
        return [(cx + radius * math.cos(a0), cy + radius * math.sin(a0)),
                (cx + radius * math.cos(am), cy + radius * math.sin(am)),
                (cx + radius * math.cos(a1), cy + radius * math.sin(a1))]

    @staticmethod
    def _collinear(p1, p2, p3, tol: float = 1.0e-9) -> bool:
        area2 = abs((p2[0] - p1[0]) * (p3[1] - p1[1])
                    - (p3[0] - p1[0]) * (p2[1] - p1[1]))
        return area2 <= tol

    def add_geo_arc(self, id: str, cx: float, cy: float, radius: float,
                      start_angle: float, end_angle: float,
                      section_name: str = "", divisions: int = 16,
                      max_chord: float = 0.0) -> GeoArc:
        """Add an arc given centre/radius/angles (converted to 3 points)."""
        p1, pm, p2 = self._arc_3pts_from_center(cx, cy, radius,
                                                start_angle, end_angle)
        return self.add_geo_arc_3pts(id, p1, pm, p2, section_name=section_name,
                                     divisions=divisions, max_chord=max_chord)

    def add_geo_arc_3pts(self, id: str, p1, pm, p2, section_name: str = "",
                         divisions: int = 16, max_chord: float = 0.0) -> GeoArc:
        """Add an arc through three points (start, mid, end). Raises ValueError
        when the points are collinear (no unique circle)."""
        if id in self.geometry_objects:
            raise ValueError(f"Object '{id}' already exists.")
        self._check_id_not_used_elsewhere(id, "Object", exclude="geo")
        if self._collinear(p1, pm, p2):
            raise ValueError("Arc points are collinear — no unique arc.")
        m = GeoArc(id=id, section_name=section_name, divisions=divisions,
                   max_chord=max_chord)
        self.geometry_objects[id] = m
        self.materialize_object_nodes(m, [tuple(p1), tuple(pm), tuple(p2)])
        return m

    def add_geo_multisegment(self, id: str, vertices, closed: bool = False,
                           section_name: str = "", divisions: int = 1,
                           max_chord: float = 0.0,
                           span_divisions: list | None = None) -> GeoMultisegment:
        """``span_divisions`` (optional): one division count per span instead
        of sharing ``divisions`` across the whole curve — see
        :class:`~xdfem2d.models.GeoMultisegment`'s docstring."""
        if id in self.geometry_objects:
            raise ValueError(f"Object '{id}' already exists.")
        self._check_id_not_used_elsewhere(id, "Object", exclude="geo")
        m = GeoMultisegment(id=id, closed=closed, section_name=section_name,
                        divisions=divisions, max_chord=max_chord,
                        span_divisions=list(span_divisions) if span_divisions else None)
        self.geometry_objects[id] = m
        self.materialize_object_nodes(m, [(float(x), float(y)) for x, y in vertices])
        return m

    def add_geo_segment(self, id: str, x1: float, y1: float, x2: float, y2: float,
                     section_name: str = "", divisions: int = 1,
                     max_chord: float = 0.0) -> GeoSegment:
        if id in self.geometry_objects:
            raise ValueError(f"Object '{id}' already exists.")
        self._check_id_not_used_elsewhere(id, "Object", exclude="geo")
        m = GeoSegment(id=id, section_name=section_name, divisions=divisions,
                    max_chord=max_chord)
        self.geometry_objects[id] = m
        self.materialize_object_nodes(m, [(x1, y1), (x2, y2)])
        return m

    # ``section_name`` names the TriSection OR QuadSection the surface meshes
    # into (the same meaning GeoSegment/GeoArc give the name). The stored
    # attribute and the .x2d field stay ``tri_section_name`` — the persisted
    # field is deliberately not renamed (see models.GeoRectangle) — so the old
    # ``tri_section_name=`` keyword is still accepted for scripts and exported
    # files written against it.
    def add_geo_rectangle(self, id: str, corner_a, corner_c,
                          section_name: str = "",
                          target_size: float = 0.5,
                          prefer_quad: bool = True,
                          *, tri_section_name: str | None = None) -> GeoRectangle:
        """Axis-aligned rectangle from two opposite corner points (meshed into
        CST triangles at solve time, or quads — see GeoRectangle.prefer_quad's
        docstring, dev/IMPLEMENT_QUAD.md Phase 8). ``section_name`` names the
        section; the old ``tri_section_name=`` keyword still works."""
        if id in self.geometry_objects:
            raise ValueError(f"Object '{id}' already exists.")
        self._check_id_not_used_elsewhere(id, "Object", exclude="geo")
        if tri_section_name is not None:      # deprecated keyword alias
            section_name = tri_section_name
        m = GeoRectangle(id=id, tri_section_name=section_name,
                         target_size=target_size, prefer_quad=prefer_quad)
        self.geometry_objects[id] = m
        # Materialise all FOUR corners (CCW) so every vertex is an editable node.
        ax, ay = float(corner_a[0]), float(corner_a[1])
        cx, cy = float(corner_c[0]), float(corner_c[1])
        self.materialize_object_nodes(
            m, [(ax, ay), (cx, ay), (cx, cy), (ax, cy)])
        return m

    # ``section_name`` names the TriSection OR QuadSection the surface meshes
    # into. The stored attribute and the .x2d field stay ``tri_section_name``
    # (see models.GeoPolygon) — the persisted field is deliberately not
    # renamed — so the old ``tri_section_name=`` keyword is still accepted for
    # scripts and exported files written against it.
    def add_geo_polygon(self, id: str, vertices,
                        section_name: str = "",
                        target_size: float = 0.5,
                        prefer_quad: bool = True,
                        *, tri_section_name: str | None = None) -> GeoPolygon:
        """Closed-polygon region through *vertices* (meshed into DKT/CST
        triangles, or quads — see GeoRectangle.prefer_quad's docstring,
        dev/IMPLEMENT_QUAD.md Phase 8). ``section_name`` names the section;
        the old ``tri_section_name=`` keyword still works."""
        if id in self.geometry_objects:
            raise ValueError(f"Object '{id}' already exists.")
        self._check_id_not_used_elsewhere(id, "Object", exclude="geo")
        if tri_section_name is not None:      # deprecated keyword alias
            section_name = tri_section_name
        m = GeoPolygon(id=id, tri_section_name=section_name,
                       target_size=target_size, prefer_quad=prefer_quad)
        self.geometry_objects[id] = m
        self.materialize_object_nodes(
            m, [(float(x), float(y)) for x, y in vertices])
        return m

    # Deprecated method names kept as aliases (objects were renamed
    # line→segment, polyline→multisegment, surface→polygon). Old scripts and
    # the assistant keep working; new code should use the new names.
    add_geo_line = add_geo_segment
    add_geo_polyline = add_geo_multisegment
    add_geo_surface = add_geo_polygon

    # ------------------------------------------------------------------
    # `create_*` convenience layer (see dev/XDFEM2D_ENGINE.md).
    # Every method here forwards to an existing `add_*` after resolving
    # permissive arguments (auto ids, Node/Section objects, unresolved
    # Eurocode class names) — no new state, no new JSON keys.
    # ------------------------------------------------------------------

    def create_node(self, x: float, y: float, id: str | None = None) -> Node:
        """Like add_node: id optional, auto ("N1", "N2", ...)."""
        from ._compat import auto_name
        if id is None:
            # Set only while an assistant increment is applied (see
            # script_check.apply_increment): a node asked for on top of an
            # existing one is that node, so the new bars connect to the model.
            tol = getattr(self, "_script_weld_tol", None)
            if tol is not None:
                for n in self.nodes.values():
                    if math.hypot(n.x - x, n.y - y) <= tol:
                        self._script_weld_log.append(n.id)
                        return n
            id = auto_name("N", self.nodes)
        return self.add_node(id, x, y)

    def create_node_list(self, outline, id_prefix: str | None = None) -> list[Node]:
        """One node per point in outline (flexible point-list forms — see
        ``_compat.as_points``: pairs, flat list, dicts, or two columns).
        Ids auto-assigned; pass id_prefix (e.g. "P") for "P1", "P2", ...
        instead of the "N1", "N2", ... ids create_node uses by default."""
        from ._compat import as_points, auto_name
        nodes = []
        for x, y in as_points(outline):
            nid = auto_name(id_prefix, self.nodes) if id_prefix else None
            nodes.append(self.create_node(x, y, id=nid))
        return nodes

    # Pure rename — same signature/behaviour as add_material.
    create_material = add_material

    def create_rc_material(self, concrete: str = "C30/37", steel: str = "B500B",
                           name: str | None = None) -> Material:
        """Get-or-create a Concrete Material carrying fck/fyk from the
        Eurocode tables. name=None reuses create_rc_section's own
        get-or-create/dedup naming (`_rc_material`), so a material made
        here and one made implicitly by create_rc_section for the same
        classes are the same object."""
        if name is None:
            return self.materials[self._rc_material(concrete, steel)]
        if name in self.materials:
            return self.materials[name]
        from .databases import material_from_grade
        return self.add_material(name, **material_from_grade(
            "Concrete", concrete, reinforcement=steel))

    def create_steel_material(self, steel: str = "S275",
                              name: str | None = None) -> Material:
        """Get-or-create a Steel Material carrying fy/fu from the Eurocode
        tables (same naming scheme create_steel_section uses: "S_<grade>")."""
        mat = name or f"S_{steel}"
        if mat in self.materials:
            return self.materials[mat]
        from .databases import material_from_grade
        return self.add_material(mat, **material_from_grade("Steel", steel))

    def create_timber_material(self, timber: str = "C24",
                               name: str | None = None) -> Material:
        """Get-or-create a Timber Material carrying fmk/fvk/fc0k/ft0k from
        the Eurocode (EN 338) tables. New family — no add_timber_material
        equivalent existed before."""
        mat = name or f"T_{timber}"
        if mat in self.materials:
            return self.materials[mat]
        from .databases import material_from_grade
        return self.add_material(mat, **material_from_grade("Timber", timber))

    def _auto_material_from_class(self, class_name: str) -> str | None:
        """If class_name is a recognised Eurocode concrete/steel/timber
        grade, get-or-create the matching Material and return its name;
        None if not recognised — callers fall back to the normal 'material
        not found' error, no silent guessing (see dev/XDFEM2D_ENGINE.md §4.2)."""
        from .databases import (concrete_design_grades, steel_design_grades,
                                timber_design_grades)
        if class_name in concrete_design_grades():
            return self.create_rc_material(concrete=class_name).name
        if class_name in steel_design_grades():
            return self.create_steel_material(steel=class_name).name
        if class_name in timber_design_grades():
            return self.create_timber_material(timber=class_name).name
        return None

    # Not domain-aware: add_section already serves both bar and area domains
    # through the same call. Defaults shape to RECTANGULAR here (add_section
    # itself defaults to GENERIC when shape is not given) because a plain
    # b x h bar made through this call is a rectangle, not "unknown fibre
    # geometry" -- GENERIC earlier left it displaying as "Generic" in the
    # section dialog for no reason (see dev/BUCKLING_COLUMN_PERSISTENCE.md
    # and rc_design._is_generic_concrete_section for why the design code
    # never relied on shape=='RECTANGULAR' to detect this, but nothing else
    # should have to be as careful). area_override/inertia_override, if
    # given, still win over shape either way (Section.area/inertia check the
    # override first), so this changes nothing for a section meant to be
    # generic on purpose -- only the actual default for a plain rectangular
    # bar. (The API-reference tool's summary is this docstring verbatim, so
    # the reasoning lives here in a comment, not in the string a small model
    # has to read.)
    def create_bar_section(self, name: str, material_name: "str | Material",
                           b: float, h: float, **kw) -> Section:
        """Like add_section, but material_name may be an unresolved Eurocode
        class (e.g. "C25/30", "S275", "C24") -- auto-created via
        create_rc_material/create_steel_material/create_timber_material if
        recognised -- or a Material object. Defaults shape to RECTANGULAR."""
        from ._compat import as_material_name
        from .models import SectionShape
        mat = as_material_name(material_name)
        if mat not in self.materials:
            resolved = self._auto_material_from_class(mat)
            if resolved is not None:
                mat = resolved
        kw.setdefault('shape', SectionShape.RECTANGULAR)
        return self.add_section(name, mat, b=b, h=h, **kw)

    def create_area_section(self, name: str, material_name: "str | Material",
                            thickness: float = 0.2, **kw) -> TriSection:
        """Domain-aware area section pair: plane -> CST/QM6 defaults, plate
        -> MITC3/MITC4 defaults (add_area_section — a TriSection AND a
        QuadSection under this name, so create_area_element resolves the
        section whichever shape it ends up building). Same permissive
        material resolution as create_bar_section."""
        from ._compat import as_material_name
        mat = as_material_name(material_name)
        if mat not in self.materials:
            resolved = self._auto_material_from_class(mat)
            if resolved is not None:
                mat = resolved
        if self.domain == "plate":
            kw.setdefault("formulation", "MITC3")
            kw.setdefault("quad_formulation", "MITC4")
        else:
            kw.setdefault("formulation", "CST")
            kw.setdefault("quad_formulation", "QM6")
        return self.add_area_section(name, mat, thickness=thickness, **kw)

    # Section first, then the nodes it applies to — matches
    # create_area_element below, one order across the whole create_*
    # element layer instead of one order per function (the API-reference
    # tool's summary is this docstring verbatim, so the reasoning lives
    # here in a comment, not in the string a small model has to read).
    def create_bar_element(self, section: "str | Section",
                           node_i: "str | Node", node_j: "str | Node",
                           id: str | None = None, **kw) -> BarElement:
        """Like add_bar_element: id optional; nodes/section take id or
        object. Section first, then the nodes. To give an existing bar
        (found by id, not just returned here) a different section: FIRST
        create it (create_rc_section, create_steel_section or
        create_timber_bar_section) -- there is no set_bar_section(). THEN
        point section_name at it:
        model.bar_elements_by_id['B1'].section_name = 'S2'."""
        from ._compat import as_node_id, as_section_name, auto_name
        if id is None:
            id = auto_name("B", self.bar_elements_by_id)
        return self.add_bar_element(id, as_node_id(node_i), as_node_id(node_j),
                                    as_section_name(section), **kw)

    # node_l is the only thing that decides triangle vs quad, so one call
    # covers both instead of the caller having to pick add_tri_element vs
    # add_quad_element up front. Section first, same reasoning as
    # create_bar_element above.
    def create_area_element(self, section: "str | TriSection | QuadSection",
                            node_i: "str | Node", node_j: "str | Node",
                            node_k: "str | Node", node_l: "str | Node | None" = None,
                            id: str | None = None, **kw) -> "TriElement | QuadElement":
        """Like add_area_element: id optional; nodes/section take id or
        object. 3 nodes -> triangle, 4 -> quad; section first."""
        from ._compat import as_node_id, as_section_name, auto_name
        section_name = as_section_name(section)
        node_l_id = as_node_id(node_l) if node_l is not None else None
        if id is None:
            id = (auto_name("T", self.tri_elements_by_id) if node_l is None
                  else auto_name("Q", self.quad_elements_by_id))
        return self.add_area_element(id, as_node_id(node_i), as_node_id(node_j),
                                     as_node_id(node_k), node_l_id,
                                     section_name=section_name, **kw)


    def _find_load_case_id_ci(self, name: str) -> str | None:
        """The existing load case id matching *name* case-insensitively, or
        None. The id actually stored keeps whatever case it was first given
        (see create_load_case) -- this is how every later call finds that
        same case again regardless of the case it is asked for in."""
        from ._compat import as_node_id
        name = as_node_id(name)
        if name in self.load_cases_by_id:
            return name
        low = name.lower()
        for existing_id in self.load_cases_by_id:
            if existing_id.lower() == low:
                return existing_id
        return None

    # A given *id* is matched against existing load cases case-insensitively
    # first: 'sw' finds an already-created 'SW' and returns it unchanged
    # (self_weight_factor/action_type are then ignored, same as any other
    # get-or-create) instead of creating a second, near-identical case that
    # only differs by case -- a small model naming the same case with a
    # different capitalisation in a later turn is a real, observed failure
    # mode (qwen3.5:4b, 25/09/2026 calibration), and a silent duplicate load
    # case is far harder to notice than one that just got reused. The
    # grafia of the FIRST call to create a name wins; every later call,
    # whatever case it uses, folds into that one. add_load_case itself is
    # untouched -- it stays the raw, exact-match layer this sits on top of,
    # same relationship as create_support/add_support.
    def create_load_case(self, id: str | None = None,
                         self_weight_factor: float = 0.0, action_type=None,
                         create_analysis_case: bool = True) -> LoadCase:
        """Like add_load_case: id optional, auto ("LC1", "LC2", ...). A name
        matching an existing case (any capitalisation) is returned unchanged
        instead of creating a duplicate."""
        from ._compat import auto_name
        if id is None:
            id = auto_name("LC", self.load_cases_by_id)
        else:
            existing = self._find_load_case_id_ci(id)
            if existing is not None:
                return self.load_cases_by_id[existing]
        return self.add_load_case(id, self_weight_factor=self_weight_factor,
                                  action_type=action_type,
                                  create_analysis_case=create_analysis_case)

    def _find_analysis_case_id_ci(self, name: str) -> str | None:
        """The existing analysis case id matching *name* case-insensitively,
        or None -- the analysis-case twin of _find_load_case_id_ci, same
        reasoning (see create_load_case)."""
        from ._compat import as_node_id
        name = as_node_id(name)
        if name in self.analysis_cases_by_id:
            return name
        low = name.lower()
        for existing_id in self.analysis_cases_by_id:
            if existing_id.lower() == low:
                return existing_id
        return None

    # Same case-insensitive get-or-create treatment as create_load_case,
    # and the same reasoning: 'ac1' finds an already-created 'AC1' and
    # returns it unchanged rather than creating a near-duplicate that only
    # differs by case. add_analysis_case stays the raw, exact-refuse layer
    # this sits on top of.
    # The analysis_type enum is repeated in this docstring, not just in
    # add_analysis_case's (28/09/2026): this is the call the assistant is
    # told to use instead, and in Create mode a name= lookup for the
    # low-level call redirects straight to THIS docstring, never
    # add_analysis_case's -- so the enum has to live here too, or the
    # assistant never sees it (get_api_reference/_low_level_redirect).
    def create_analysis_case(self, id: str | None = None,
                             analysis_type: str = 'Linear',
                             coefficients: dict | None = None,
                             **kwargs) -> AnalysisCase:
        """Like add_analysis_case; id optional (auto AC1, AC2, ...); an
        existing name (any case) is returned unchanged, not duplicated.
        analysis_type: 'Linear'|'NonLinear'|'Mass'|'Modal'|'Spectrum'|
        'GeometricNonlinear'|'Sequence'. coefficients={load_case_id:factor}
        for Linear/NonLinear/Mass. Modal: num_modes, modal_case_id (its
        Mass case id). Spectrum: modal_case_id (its Modal case id),
        spectrum_id, direction ('X'/'Y'/'XY')."""
        from ._compat import as_case_coefficients, auto_name
        coefficients = as_case_coefficients(coefficients)
        if id is None:
            id = auto_name("AC", self.analysis_cases_by_id)
        else:
            existing = self._find_analysis_case_id_ci(id)
            if existing is not None:
                return self.analysis_cases_by_id[existing]
        return self.add_analysis_case(id, analysis_type=analysis_type,
                                      coefficients=coefficients, **kwargs)

    # ── Real nodes = the object geometry (node-driven) ────────────────
    def object_defining_points(self, obj) -> list:
        """Current (x, y) of the object's defining nodes (read from the nodes,
        since objects are node-driven)."""
        return [(self.nodes[nid].x, self.nodes[nid].y)
                for nid in getattr(obj, "node_ids", []) if nid in self.nodes]

    def _unique_node_id(self, base: str) -> str:
        nid = base; k = 2
        while nid in self.nodes:
            nid = f"{base}_{k}"; k += 1
        return nid

    def materialize_object_nodes(self, obj, pts, tol: float = 1.0e-6):
        """Create/reposition the real nodes at the target defining points *pts*,
        storing their ids on ``obj.node_ids``. Reuses an owned id when it exists
        (so supports/connections survive an edit) or a coincident node (so the
        object ties into the frame); otherwise creates a new node. Drops surplus
        owned nodes (if unused)."""
        ids = list(getattr(obj, "node_ids", []))

        def _find_coincident(x, y):
            for nid, n in self.nodes.items():
                if abs(n.x - x) <= tol and abs(n.y - y) <= tol:
                    return nid
            return None

        for k, (x, y) in enumerate(pts):
            cur = ids[k] if (k < len(ids) and ids[k] in self.nodes) else None
            if cur is not None:
                self.nodes[cur].x = x
                self.nodes[cur].y = y
                nid = cur
            else:
                nid = _find_coincident(x, y)
                if nid is None:
                    nid = self._unique_node_id(f"{obj.id}.p{k}")
                    self.add_node(nid, x, y)
            if k < len(ids):
                ids[k] = nid
            else:
                ids.append(nid)
        # Drop surplus owned nodes (e.g. a polyline lost vertices) if unused.
        for extra in ids[len(pts):]:
            self._remove_node_if_unused(extra, exclude_obj=obj)
        obj.node_ids = ids[:len(pts)]

    def _node_in_use(self, nid: str, exclude_obj=None) -> bool:
        """True if a node is referenced by a bar or by another object's nodes."""
        if any(e.node_i == nid or e.node_j == nid for e in self.bar_elements):
            return True
        for o in self.geometry_objects.values():
            if o is exclude_obj:
                continue
            if nid in getattr(o, "node_ids", []):
                return True
        return False

    def _remove_node_if_unused(self, nid: str, exclude_obj=None):
        if nid in self.nodes and not self._node_in_use(nid, exclude_obj):
            self.remove_node(nid)

    def remove_geo_object(self, id: str):
        from ._compat import as_object_id
        id = as_object_id(id)
        obj = self.geometry_objects.pop(id, None)
        if obj is not None:
            _refs.remove(self, 'object', id)      # its loads and springs
            # Remove the object's owned endpoint/vertex nodes, but only those not
            # still connected to a bar or another object.
            for nid in list(getattr(obj, "node_ids", [])):
                self._remove_node_if_unused(nid, exclude_obj=obj)

    def add_support(self, name: str, ux=False, uy=False, tz=False,
                    w=None, tx=None, ty=None) -> Support:
        """Define a named set of restraints. Does NOT support anything yet.

        Creates the description — "a pin", "a roller". Putting it on a node is
        a second call, assign_support(node_id, name), and both are needed:

            model.add_support('PIN', ux=True, uy=True)
            model.assign_support('N1', 'PIN')

        True restrains, False leaves free. Plate domain: use the aliases
        w/tx/ty, e.g. add_support('ENC', w=True, tx=True, ty=True).
        """
        # Spelled out because the name reads like it does the whole job: four
        # of six language models asked to build a beam stopped after the first
        # call, including ones that got everything else right, and what they
        # produced was a mechanism.
        if w is not None:
            ux = w
        if tx is not None:
            uy = tx
        if ty is not None:
            tz = ty
        s = Support(name=name, ux=ux, uy=uy, tz=tz)
        self.supports[name] = s
        return s

    def assign_support(self, node_id: str, support_name: str) -> SupportAssignment:
        """Put a support defined by :meth:`add_support` on a node.

        This is the step that actually restrains the structure. A model with
        supports defined and none assigned is a mechanism.
        """
        a = SupportAssignment(node_id=node_id, support_name=support_name)
        self.support_assignments.append(a)
        return a

    # Touches one node, not every node sharing a canonical name the way
    # remove_support(name) does — the fine-grained primitive whose absence used
    # to force callers to the coarse one, which is how one edit wiped another
    # region's edge supports. The facade over this is delete_support.
    def remove_support_assignment(self, node_id: str, support_name: str = ""):
        """Remove the support on a node — the by-node inverse of
        :meth:`assign_support`. ``support_name=""`` clears whatever the node
        carries; a name clears only that one. The support *definition* is left
        for reuse (an orphan is tidied by :meth:`prune_unused_supports`)."""
        self.support_assignments = [
            a for a in self.support_assignments
            if not (a.node_id == node_id
                    and (not support_name or a.support_name == support_name))]

    def add_node_spring(self, node_id: str, kx=0.0, ky=0.0, kt=0.0,
                        mode_x='both', mode_y='both',
                        kz=None, ktx=None, kty=None) -> NodeSpring:
        """Attach a spring directly to a node (replaces any existing spring on that node).

        mode_x / mode_y: 'both' | 'tension' | 'compression' — unilateral
        behaviour for the translational components, honoured only by NonLinear
        analysis cases (Linear cases treat the spring as bilateral).

        Plate domain: use ``kz`` [kN/m] (transverse), ``ktx``/``kty`` [kNm/rad]
        (rotational about X and Y) — aliases of the kx/ky/kt storage slots.
        """
        if kz is not None:
            kx = kz
        if ktx is not None:
            ky = ktx
        if kty is not None:
            kt = kty
        self._require_exists(node_id, self.nodes, "node", "add_node_spring")
        valid = ('both', 'tension', 'compression')
        if mode_x not in valid or mode_y not in valid:
            raise ValueError(f"spring mode must be one of {valid}")
        sp = NodeSpring(node_id=node_id, kx=kx, ky=ky, kt=kt,
                        mode_x=mode_x, mode_y=mode_y)
        self.node_springs[node_id] = sp
        return sp

    def add_element_spring(self, element_id: str, kx=0.0, ky=0.0,
                           coord_sys="global",
                           mode_x='both', mode_y='both') -> ElementSpring:
        """Attach a foundation spring directly to an element (replaces any existing one).

        coord_sys: 'global' (X, Y) or 'local' (axial, transverse to the element).
        mode_x / mode_y: 'both' | 'tension' | 'compression' — unilateral
        behaviour for the X/axial and Y/transverse components, honoured only by
        NonLinear analysis cases (Linear cases treat the spring as bilateral).
        """
        self._require_exists(element_id, self.bar_elements_by_id, "bar",
                             "add_element_spring")
        valid = ('both', 'tension', 'compression')
        if mode_x not in valid or mode_y not in valid:
            raise ValueError(f"spring mode must be one of {valid}")
        es = ElementSpring(element_id=element_id, kx=kx, ky=ky,
                           coord_sys=coord_sys, mode_x=mode_x, mode_y=mode_y)
        self.element_springs[element_id] = es
        return es

    # Applied to the bars the object subdivides into at solve time. kx and ky
    # are per unit length, each a value or a field (the field at the bar
    # midpoint). Replaces any existing line spring on the same object.
    def add_line_element_spring(self, object_id: str, kx=0.0, ky=0.0,
                                kx_field="", ky_field="", coord_sys="global",
                                mode_x="both", mode_y="both"
                                ) -> LineElementSpring:
        """Attach a foundation spring to a line object (line / arc / polyline)."""
        self._require_line_object(object_id, "add_line_element_spring")
        valid = ('both', 'tension', 'compression')
        if mode_x not in valid or mode_y not in valid:
            raise ValueError(f"spring mode must be one of {valid}")
        ls = LineElementSpring(object_id=object_id, kx=kx, ky=ky,
                               kx_field=kx_field, ky_field=ky_field,
                               coord_sys=coord_sys, mode_x=mode_x,
                               mode_y=mode_y)
        self.line_element_springs = [
            s for s in self.line_element_springs if s.object_id != object_id]
        self.line_element_springs.append(ls)
        return ls

    def remove_element_spring(self, element_id: str):
        """Remove the foundation spring on *element_id* (the inverse of
        :meth:`add_element_spring`). No error if there is none — removing what is
        not there is the intended end state."""
        self.element_springs.pop(element_id, None)

    def remove_line_element_spring(self, object_id: str):
        """Remove the line spring on *object_id* (the inverse of
        :meth:`add_line_element_spring`)."""
        from ._compat import as_object_id
        object_id = as_object_id(object_id)
        self.line_element_springs = [
            s for s in self.line_element_springs if s.object_id != object_id]

    # -- Constraints (multi-point / linear DOF coupling) ------------------
    #
    # A constraint invalidates the DOF map like a topology change does: Phase 2
    # (master-slave reduction) changes the count of free DOFs, so the cache must
    # be rebuilt. Phase 1 (penalty) does not, but invalidating here keeps the two
    # phases interchangeable without touching call sites.

    def _next_constraint_id(self, prefix: str) -> str:
        n = 1
        while f"{prefix}{n}" in self.constraints:
            n += 1
        return f"{prefix}{n}"

    def add_constraint_equation(self, terms, value: float = 0.0,
                                id: str = "") -> Constraint:
        """Add a raw linear constraint ``sum(coef * dof) = value``.

        *terms* is an iterable of ``(node_id, component, coef)`` with component
        one of ('ux', 'uy', 'tz'). This is the advanced / machine-written form;
        the shortcuts below build on it.
        """
        norm = []
        for node_id, comp, coef in terms:
            if comp not in CONSTRAINT_COMPONENTS:
                raise ValueError(
                    f"constraint component must be one of {CONSTRAINT_COMPONENTS}")
            norm.append((node_id, comp, float(coef)))
        cid = id or self._next_constraint_id("EQ")
        c = Constraint(id=cid, kind="equation", terms=norm, value=float(value))
        self.constraints[cid] = c
        self._node_dof_index = None
        return c

    def add_equal_dof(self, nodes, components, id: str = "") -> Constraint:
        """Constrain every node in *nodes* to share the same value of each
        component in *components* (e.g. ``components=['ux']`` ties horizontal
        translation across a set of nodes)."""
        nodes = list(nodes)
        comps = [components] if isinstance(components, str) else list(components)
        for comp in comps:
            if comp not in CONSTRAINT_COMPONENTS:
                raise ValueError(
                    f"constraint component must be one of {CONSTRAINT_COMPONENTS}")
        if len(nodes) < 2:
            raise ValueError("equal_dof needs at least two nodes")
        cid = id or self._next_constraint_id("EQUAL")
        c = Constraint(id=cid, kind="equal_dof",
                       nodes=nodes, components=comps)
        self.constraints[cid] = c
        self._node_dof_index = None
        return c

    def add_rigid_link(self, master: str, slaves, id: str = "") -> Constraint:
        """Tie *slaves* to *master* as a single rigid body (in-plane): each slave
        follows the master's two translations and rotation through the rigid-body
        kinematics evaluated from the nodal coordinates."""
        slaves = [slaves] if isinstance(slaves, str) else list(slaves)
        if not slaves:
            raise ValueError("rigid_link needs at least one slave node")
        cid = id or self._next_constraint_id("RIGID")
        c = Constraint(id=cid, kind="rigid_link",
                       master=master, slaves=slaves)
        self.constraints[cid] = c
        self._node_dof_index = None
        return c

    def remove_constraint(self, id: str) -> None:
        if self.constraints.pop(id, None) is not None:
            self._node_dof_index = None

    def add_load_case(self, id: str, self_weight_factor=0.0, action_type=None,
                      create_analysis_case: bool = True,
                      load_duration=None, category: str = "",
                      psi0=None, psi1=None, psi2=None) -> LoadCase:
        from .models import ActionType, coerce_load_duration
        # Until 26/09/2026 this had no existence check at all -- not even
        # exact-match: calling it twice with the same id appended a second
        # LoadCase to self.load_cases while load_cases_by_id[id] silently
        # moved to point at the newer one, orphaning the first in the list
        # (still iterated by the load-case table, exports, ...) but
        # unreachable by id. Reached from the GUI too: the plain "+ Add"
        # button (LoadCasesPanel._add_lc) called this with no pre-check of
        # its own. Matched case-insensitively, like create_load_case's own
        # fold (see _find_load_case_id_ci) -- 'sw' is refused just as loudly
        # as 'SW' when 'SW' already exists, rather than only the exact
        # spelling. Raises instead of folding: add_* stays the raw,
        # exact-refuse layer (same as add_node); create_load_case is the one
        # that resolves a case-insensitive match for the caller.
        existing = self._find_load_case_id_ci(id)
        if existing is not None:
            raise ValueError(
                f"Load case '{id}' already exists"
                + (f" (as '{existing}')" if existing != id else "") + ".")
        # A very common assistant slip is add_load_case('Q', 'Live') — the
        # action type passed positionally, where self_weight_factor sits. Left
        # alone it stores a string in a numeric field and every later reader
        # (the load-case table, combinations, the solver) breaks on it. So:
        # a non-numeric self_weight_factor with no explicit action_type is
        # taken as the action type it plainly is, and self_weight_factor falls
        # back to 0.0. This is the same "coerce, don't refuse" stance as below.
        try:
            self_weight_factor = float(self_weight_factor)
        except (TypeError, ValueError):
            if action_type is None:
                action_type = self_weight_factor
            self_weight_factor = 0.0
        # coerce, not ActionType(): the letter, the word, and the old dashed
        # label all mean the same thing, and refusing two of the three only
        # taught callers to guess.
        at = ActionType.coerce(action_type) if action_type else ActionType.G
        lc = LoadCase(id=id, self_weight_factor=self_weight_factor, action_type=at,
                      load_duration=coerce_load_duration(load_duration),
                      category=str(category or ""),
                      psi0=_opt_float(psi0), psi1=_opt_float(psi1),
                      psi2=_opt_float(psi2))
        self.load_cases.append(lc)
        self.load_cases_by_id[id] = lc
        # Every load case gets a twin Linear analysis case (factor 1) so that
        # combinations — which always combine analysis cases — can reference it
        # straight away. Skipped when reloading a saved model (the analysis
        # cases are restored separately).
        if create_analysis_case:
            self.twin_analysis_case_id(id)
        return lc

    def twin_analysis_case_id(self, load_case_id: str) -> str:
        """Return the id of the Linear analysis case that applies *load_case_id*
        with a unit factor, creating the twin (same id as the load case) if
        needed."""
        for ac in self.analysis_cases:
            if (ac.analysis_type == 'Linear'
                    and ac.coefficients == {load_case_id: 1.0}
                    and not getattr(ac, 'stored_stiffness_id', None)):
                return ac.id
        # ac_id = f"AC_{load_case_id}"
        ac_id = f"{load_case_id}"
        existing = self._find_analysis_case_id_ci(ac_id)
        if existing is not None:
            return existing
        self.add_analysis_case(ac_id, 'Linear', {load_case_id: 1.0})
        return ac_id

    # Used when loading a saved model: maps any legacy load-case references in
    # a combination to that load case's twin analysis case (creating the twin
    # on demand if the file predates analysis cases), and folds any stored
    # ``analysis_coefficients`` into ``coefficients``.
    #
    # Deliberately does NOT create a twin for every load case up front — only
    # load cases actually referenced by a legacy combination get one here. A
    # load case's twin is otherwise created exactly once, at add_load_case()
    # time; if the user later removes that twin, it must stay removed across
    # save/reload and undo/redo.
    def normalize_combinations_to_analysis_cases(self):
        """Ensure every combination references analysis cases only."""
        for combo in self.load_combinations:
            merged = dict(getattr(combo, 'analysis_coefficients', {}))
            for key, coeff in combo.coefficients.items():
                if key in self.analysis_cases_by_id:
                    target = key
                elif key in self.load_cases_by_id:
                    target = self.twin_analysis_case_id(key)
                else:
                    target = key
                merged[target] = merged.get(target, 0.0) + coeff
            combo.coefficients = merged
            combo.analysis_coefficients = {}

    def add_point_load(self, node_id: str, load_case_id: str,
                       fx=0.0, fy=0.0, mz=0.0,
                       fz=None, mx=None, my=None) -> PointLoad:
        """Nodal concentrated load.

        Plane domain: ``fx``/``fy`` are forces along X/Y [kN], ``mz`` a moment
        about Z [kNm]. Plate domain: use ``fz`` (transverse force, negative
        downward), ``mx`` and ``my`` (moments about X and Y) — they alias the
        same three storage slots (component 0, 1, 2) as fx/fy/mz.
        """
        if fz is not None:
            fx = fz
        if mx is not None:
            fy = mx
        if my is not None:
            mz = my
        self._require_exists(node_id, self.nodes, "node", "add_point_load")
        pl = PointLoad(node_id=node_id, load_case_id=load_case_id, fx=fx, fy=fy, mz=mz)
        self.point_loads.append(pl)
        return pl

    def add_distributed_load(self, element_id: str, load_case_id: str,
                              fxe=0.0, fxd=0.0, fye=0.0, fyd=0.0,
                              coord_sys='global',
                              fze=None, fzd=None) -> DistributedLoad:
        """Trapezoidal distributed load, given at BOTH ends.

        fye/fyd are the transverse load at each end, fxe/fxd the axial one;
        negative is downward. For a uniform load use create_uniform_load, or set
        both ends equal — setting one leaves the other at zero, which is a
        triangular load carrying half the total. coord_sys: 'global' or
        'local'. Plate domain: use fze/fzd [kN/m] (aliases of fye/fyd);
        fxe/fxd are ignored there.
        """
        if fze is not None:
            fye = fze
        if fzd is not None:
            fyd = fzd
        # Why the docstring spells that out: two language models asked for "a
        # uniform load of 10 kN/m" wrote `fyd=-10.0` alone, and one explained
        # in a comment that fyd was "the transverse component" — confident, and
        # off by a factor of two. The model builds, solves and reports numbers,
        # so nothing downstream can tell.
        #
        # Kept as a comment and not in the docstring: split_doc ships
        # everything before Args: to the assistant, and an anecdote in there is
        # several hundred characters of context spent on a story.
        csys = coord_sys.lower()
        if csys not in ('global', 'local'):
            raise ValueError(f"coord_sys must be 'global' or 'local', got '{coord_sys}'")
        self._require_exists(element_id, self.bar_elements_by_id, "bar",
                             "add_distributed_load")
        if csys == 'local':
            # Transform to global using element geometry (computed lazily at solve time)
            dl = DistributedLoad(element_id=element_id, load_case_id=load_case_id,
                                 fxe=fxe, fxd=fxd, fye=fye, fyd=fyd, coord_sys='local')
        else:
            dl = DistributedLoad(element_id=element_id, load_case_id=load_case_id,
                                 fxe=fxe, fxd=fxd, fye=fye, fyd=fyd, coord_sys='global')
        self.distributed_loads.append(dl)
        return dl

    # Applied to the bars the object subdivides into at solve time. Each
    # direction is a value or a field, independently: a value is uniform (both
    # ends equal); a field sets the end values from the field at the bar's two
    # nodes, so the trapezoid follows the field. Replaces any existing line
    # distributed load on the same object and case.
    def add_line_distributed_load(self, object_id: str, load_case_id: str,
                                  fx=0.0, fy=0.0, fx_field="", fy_field="",
                                  coord_sys="global") -> LineDistributedLoad:
        """Add a distributed load to a line object (line / arc / polyline)."""
        self._require_line_object(object_id, "add_line_distributed_load")
        dl = LineDistributedLoad(object_id=object_id, load_case_id=load_case_id,
                                 fx=fx, fy=fy, fx_field=fx_field,
                                 fy_field=fy_field, coord_sys=coord_sys)
        self.line_distributed_loads = [
            d for d in self.line_distributed_loads
            if not (d.object_id == object_id
                    and d.load_case_id == load_case_id)]
        self.line_distributed_loads.append(dl)
        return dl

    def add_element_point_load(self, element_id: str, load_case_id: str,
                               a: float, fx=0.0, fy=0.0, mz=0.0,
                               coord_sys='global', fz=None) -> ElementPointLoad:
        """Add a concentrated load on an element at distance ``a`` from node i.

        a : distance from the i-end along the element axis [m] (0 < a < L).
        fx, fy : force components (global axes, or local x'/y' if
                 ``coord_sys='local'``) [kN].
        mz : concentrated moment about Z [kNm]. Plate domain: use fz [kN]
        (transverse force, alias of fy); mz is the concentrated bending
        moment there and fx is ignored.
        """
        if fz is not None:
            fy = fz
        csys = coord_sys.lower()
        if csys not in ('global', 'local'):
            raise ValueError(f"coord_sys must be 'global' or 'local', got '{coord_sys}'")
        self._require_exists(element_id, self.bar_elements_by_id, "bar",
                             "add_element_point_load")
        epl = ElementPointLoad(element_id=element_id, load_case_id=load_case_id,
                               a=a, fx=fx, fy=fy, mz=mz, coord_sys=csys)
        self.element_point_loads.append(epl)
        return epl

    # This is what stops the two ways of writing a model from drifting apart.
    # Drawing a support used to create a definition per node, named after the
    # node, so one boundary condition appeared under forty names — and anything
    # matching supports by name, such as propagating one along the edge of a
    # meshed region, saw forty different things where there was one.
    #
    # An existing definition wins whatever it is called, so a file holding a
    # 'BASE' with these restraints keeps using BASE rather than gaining an
    # identical twin beside it. And a name already taken by *different*
    # restraints is not stolen: renaming what someone else defined would be the
    # worse trespass, so the new one becomes PIN-2.
    def support_for(self, ux=False, uy=False, tz=False) -> str | None:
        """The name of a support with these restraints, reusing or creating one.

        Returns None when nothing is restrained — that is not a support.
        Assigning it to a node is still a second call, assign_support().
        """
        from .models import canonical_support_name
        want = (bool(ux), bool(uy), bool(tz))
        if want == (False, False, False):
            return None
        for name, s in self.supports.items():
            if (bool(s.ux), bool(s.uy), bool(s.tz)) == want:
                return name
        base = canonical_support_name(*want, domain=getattr(self, 'domain', 'plane'))
        name, n = base, 1
        while name in self.supports:        # taken, by different restraints
            n += 1
            name = f"{base}-{n}"
        self.add_support(name, ux=want[0], uy=want[1], tz=want[2])
        return name

    def prune_unused_supports(self, only: set | None = None) -> list:
        """Drop support definitions no node uses. Returns the names removed.

        ``only`` limits it to a set of names — the caller passes the ones it
        created itself. Nothing a user defined is removed by tidying up: an
        unused definition in a file they wrote may be about to be used again,
        and deleting it is not ours to do.
        """
        used = {a.support_name for a in self.support_assignments}
        # A support named in an object's per-edge ``edge_supports`` list is in
        # use too, even though it is not written into support_assignments until
        # the mesh is baked. Counting only the assignments made "unused" any
        # canonical support (SIMPLE, CLAMPED, ...) that a *region's* edge relies
        # on — so editing a node support on one object pruned the definition and
        # silently stripped the pinned/clamped edges of every other object that
        # shared it (their names then resolving to nothing on the next expand).
        for obj in self.geometry_objects.values():
            for nm in getattr(obj, 'edge_supports', None) or []:
                if nm and nm != 'free':
                    used.add(nm)
        gone = [n for n in list(self.supports)
                if n not in used and (only is None or n in only)]
        for name in gone:
            self.supports.pop(name, None)
        return gone

    # dt_uniform/dt_gradient are the canonical names — matching the rest of
    # the temperature-load family (tri/quad/area/line all use dt_uniform/
    # dt_gradient or dt_i/dt_j/dt_k/dt_gradient). bar_id likewise matches
    # tri_id/quad_id (add_tri_temperature_load/add_quad_temperature_load) —
    # a per-kind name is more informative than the generic element_id this
    # used to share with nothing else in the family. The stored
    # TemperatureLoad attributes and the .x2d field stay element_id/
    # delta_t_uniform/delta_t_gradient — deliberately not renamed, same as
    # tri_section_name below — so the old element_id=/delta_t_uniform=/
    # delta_t_gradient= keywords are still accepted for scripts and exported
    # files written against them.
    # bar_id/load_case_id default to None only so a caller using the old
    # element_id= alias (with no positional args at all) does not trip
    # Python's own "non-default argument follows default argument" rule —
    # both are still effectively required, checked explicitly below rather
    # than silently accepting a missing one as "" (the two are the model's
    # own foreign keys: a blank of either would misfile the load instead of
    # raising).
    def add_temperature_load(self, bar_id: str | None = None,
                              load_case_id: str | None = None,
                              dt_uniform=0.0, dt_gradient=0.0, alpha=None,
                              *, element_id: str | None = None,
                              delta_t_uniform: float | None = None,
                              delta_t_gradient: float | None = None
                              ) -> TemperatureLoad:
        """Add a thermal load to a bar. ``bar_id``/``dt_uniform``/
        ``dt_gradient`` name the canonical parameters; the old ``element_id=``/
        ``delta_t_uniform=``/``delta_t_gradient=`` keywords still work.

        *alpha* is accepted but ignored — the thermal expansion coefficient is
        taken from the element's material (Material.alpha) at analysis time.
        The parameter is kept only for backward-compatibility with old call sites.
        """
        if element_id is not None:             # deprecated keyword alias
            bar_id = element_id
        if bar_id is None:
            raise TypeError("add_temperature_load() missing required "
                            "argument: 'bar_id'")
        if load_case_id is None:
            raise TypeError("add_temperature_load() missing required "
                            "argument: 'load_case_id'")
        if delta_t_uniform is not None:        # deprecated keyword alias
            dt_uniform = delta_t_uniform
        if delta_t_gradient is not None:       # deprecated keyword alias
            dt_gradient = delta_t_gradient
        tl = TemperatureLoad(element_id=bar_id, load_case_id=load_case_id,
                             delta_t_uniform=dt_uniform,
                             delta_t_gradient=dt_gradient)
        # Accumulates rather than replaces — like tri/quad temperature loads
        # (and point loads), multiple calls for the same bar/case are
        # meant to stack, not overwrite each other. The solver
        # (_apply_temperature_loads) and canvas._elem_kappa0() already sum
        # unconditionally over all entries, so this only changes storage.
        self.temperature_loads.append(tl)
        return tl

    # Plane domain: dt_i/dt_j/dt_k are an in-plane ΔT per node (a uniform rise
    # is dt_i = dt_j = dt_k, and only their mean drives a CST). Plate domain:
    # ``dt_gradient`` = T_top − T_bottom is the through-thickness gradient that
    # bends the DKT; the in-plane values are stress-free there. Accumulates,
    # like the bar version, so repeated calls stack rather than replace.
    def add_tri_temperature_load(self, tri_id: str, load_case_id: str,
                                 dt_i=0.0, dt_j=0.0, dt_k=0.0, dt_gradient=0.0
                                 ) -> TriTemperatureLoad:
        """Add a thermal load to a triangle: an in-plane ΔT per node (plane
        domain) or a through-thickness ``dt_gradient`` T_top − T_bottom [°C]
        that bends the DKT (plate domain). Multiple calls for the same
        triangle and case accumulate rather than replace."""
        tl = TriTemperatureLoad(tri_id=tri_id, load_case_id=load_case_id,
                                dt_i=dt_i, dt_j=dt_j, dt_k=dt_k,
                                dt_gradient=dt_gradient)
        # Accumulates rather than replaces — like bar and quad temperature
        # loads (and point loads), multiple calls for the same triangle/case
        # are meant to stack, not overwrite each other. The solver and
        # canvas drawing already sum unconditionally over all entries, so
        # this only changes storage.
        self.tri_temperature_loads.append(tl)
        return tl

    # The 4-node analogue of add_tri_temperature_load (dev/IMPLEMENT_QUAD.md
    # Phase 6): dt_i/dt_j/dt_k/dt_l are an in-plane ΔT per node (plane
    # domain, Q4/QM6), dt_gradient bends DKT4/MITC4 (plate domain) — same
    # split, same "in-plane values ignored in the plate domain" rule as the
    # triangle version.
    def add_quad_temperature_load(self, quad_id: str, load_case_id: str,
                                  dt_i=0.0, dt_j=0.0, dt_k=0.0, dt_l=0.0,
                                  dt_gradient=0.0) -> QuadTemperatureLoad:
        """Add a thermal load to a quad: an in-plane ΔT per node (plane
        domain) or a through-thickness ``dt_gradient`` T_top − T_bottom [°C]
        that bends DKT4/MITC4 (plate domain). Multiple calls for the same
        quad and case accumulate rather than replace."""
        tl = QuadTemperatureLoad(quad_id=quad_id, load_case_id=load_case_id,
                                 dt_i=dt_i, dt_j=dt_j, dt_k=dt_k, dt_l=dt_l,
                                 dt_gradient=dt_gradient)
        # Accumulates rather than replaces — see add_tri_temperature_load.
        self.quad_temperature_loads.append(tl)
        return tl

    # Shared by the object_id-taking add_*_load/spring family below: object_id
    # is ambiguous on its own (a GeoRectangle and a GeoArc are both just
    # strings to the caller), and unlike add_area_load/add_area_spring —
    # which resolve target_id against tri/quad elements too and so fall
    # through to their own final ValueError — these only ever accept a
    # geometry *object*. Passing the wrong kind (e.g. a line object where an
    # area one belongs) used to be accepted silently: the load was stored,
    # never consumed at mesh-expansion time, and nothing downstream — not the
    # solver, not the GUI, not model_check — ever said so.
    def _require_area_object(self, object_id: str, action: str):
        from ._compat import as_object_id
        object_id = as_object_id(object_id)
        obj = self.geometry_objects.get(object_id)
        if isinstance(obj, (GeoRectangle, GeoPolygon)):
            return obj
        if isinstance(obj, (GeoSegment, GeoMultisegment, GeoArc)):
            raise ValueError(
                f"{action}: '{object_id}' is a line object (rectangle/"
                "polygon needed) — did you mean add_line_* instead?")
        raise ValueError(
            f"{action}: '{object_id}' is not a surface object (rectangle "
            "or polygon).")

    def _require_line_object(self, object_id: str, action: str):
        from ._compat import as_object_id
        object_id = as_object_id(object_id)
        obj = self.geometry_objects.get(object_id)
        if isinstance(obj, (GeoSegment, GeoMultisegment, GeoArc)):
            return obj
        if isinstance(obj, (GeoRectangle, GeoPolygon)):
            raise ValueError(
                f"{action}: '{object_id}' is a surface object (line/arc/"
                "polyline needed) — did you mean add_area_* instead?")
        raise ValueError(
            f"{action}: '{object_id}' is not a line object (line, arc, "
            "or polyline).")

    def _require_exists(self, id_value: str, collection, kind_label: str,
                        action: str):
        """Guard for the add_* methods below that store a record keyed to an
        id from another collection (a node, a bar/tri/quad element) instead
        of validating a geometry object like ``_require_area_object`` /
        ``_require_line_object`` above. Without this, a typo'd or since-
        removed id was accepted silently: the record was stored, never
        matched at solve/mesh time, and nothing said so."""
        if id_value not in collection:
            raise ValueError(
                f"{action}: '{id_value}' is not a known {kind_label} id.")
        return id_value

    def add_area_temperature_load(self, object_id: str, load_case_id: str,
                                  dt_uniform=0.0, dt_gradient=0.0,
                                  field_name="", grad_field_name=""
                                  ) -> AreaTemperatureLoad:
        """Add a temperature to an area object (rectangle / surface), applied to
        the triangles it meshes into. ``field_name`` varies the in-plane part
        across the mesh; plate ``dt_gradient`` = T_top − T_bottom [°C] (or
        ``grad_field_name``) is the gradient that bends the DKT. Replaces any
        existing area temperature on the same object and case."""
        self._require_area_object(object_id, "add_area_temperature_load")
        tl = AreaTemperatureLoad(object_id=object_id, load_case_id=load_case_id,
                                 dt_uniform=dt_uniform, field_name=field_name,
                                 dt_gradient=dt_gradient,
                                 grad_field_name=grad_field_name)
        self.area_temperature_loads = [
            t for t in self.area_temperature_loads
            if not (t.object_id == object_id
                    and t.load_case_id == load_case_id)]
        self.area_temperature_loads.append(tl)
        return tl

    # Applied to the bars the object subdivides into at solve time. Either
    # component may come from a field, sampled at each generated bar's midpoint:
    # field_name for the uniform ΔT, grad_field_name for the gradient — the same
    # field or different ones; an empty name uses the numeric dt_*. Replaces any
    # existing line temperature on the same object and case.
    def add_line_temperature_load(self, object_id: str, load_case_id: str,
                                  dt_uniform=0.0, dt_gradient=0.0,
                                  field_name="", grad_field_name=""
                                  ) -> LineTemperatureLoad:
        """Add a temperature to a line object (line / arc / polyline)."""
        self._require_line_object(object_id, "add_line_temperature_load")
        tl = LineTemperatureLoad(object_id=object_id, load_case_id=load_case_id,
                                 dt_uniform=dt_uniform, dt_gradient=dt_gradient,
                                 field_name=field_name,
                                 grad_field_name=grad_field_name)
        self.line_temperature_loads = [
            t for t in self.line_temperature_loads
            if not (t.object_id == object_id
                    and t.load_case_id == load_case_id)]
        self.line_temperature_loads.append(tl)
        return tl

    # Temperature-load removals — the inverses of the add_*_temperature_load
    # family, missing until now (no way to clear a thermal load once set, in
    # the API or the GUI). load_case_id="" clears every case on the target,
    # matching remove_area_load's convention; a given case clears only that one.
    def remove_temperature_load(self, element_id: str, load_case_id: str = ""):
        """Remove bar temperature loads on *element_id* (all cases, or one)."""
        self.temperature_loads = [
            t for t in self.temperature_loads
            if not (t.element_id == element_id
                    and (not load_case_id or t.load_case_id == load_case_id))]

    def remove_tri_temperature_load(self, tri_id: str, load_case_id: str = ""):
        """Remove triangle temperature loads on *tri_id* (all cases, or one)."""
        self.tri_temperature_loads = [
            t for t in self.tri_temperature_loads
            if not (t.tri_id == tri_id
                    and (not load_case_id or t.load_case_id == load_case_id))]

    def remove_quad_temperature_load(self, quad_id: str, load_case_id: str = ""):
        """Remove quad temperature loads on *quad_id* (all cases, or one)."""
        self.quad_temperature_loads = [
            t for t in self.quad_temperature_loads
            if not (t.quad_id == quad_id
                    and (not load_case_id or t.load_case_id == load_case_id))]

    def remove_area_temperature_load(self, object_id: str,
                                     load_case_id: str = ""):
        """Remove area temperature loads on *object_id* (all cases, or one)."""
        from ._compat import as_object_id
        object_id = as_object_id(object_id)
        self.area_temperature_loads = [
            t for t in self.area_temperature_loads
            if not (t.object_id == object_id
                    and (not load_case_id or t.load_case_id == load_case_id))]

    def remove_line_temperature_load(self, object_id: str,
                                     load_case_id: str = ""):
        """Remove line temperature loads on *object_id* (all cases, or one)."""
        from ._compat import as_object_id
        object_id = as_object_id(object_id)
        self.line_temperature_loads = [
            t for t in self.line_temperature_loads
            if not (t.object_id == object_id
                    and (not load_case_id or t.load_case_id == load_case_id))]

    # ``coefficients`` is ``{case_id: factor}`` where each ``case_id`` is an
    # *analysis case* id (every load case has a twin Linear analysis case
    # sharing its id, so referencing that twin is how a load case enters a
    # combination) **or** the id of another *combination* — combinations may
    # take other combinations as inputs. A referenced combination is expanded
    # into its underlying analysis cases (see
    # expand_combination_coefficients); nested combinations are only allowed
    # between linear (LinearSum) combinations.
    def _find_load_combination_id_ci(self, name: str) -> str | None:
        """The existing combination id matching *name* case-insensitively, or
        None. No dict backs load_combinations (it is a plain list, looked up
        by scanning for .id), so this scans it directly -- otherwise the same
        reasoning as _find_load_case_id_ci/_find_analysis_case_id_ci."""
        from ._compat import as_node_id
        name = as_node_id(name)
        for combo in self.load_combinations:
            if combo.id == name or combo.id.lower() == name.lower():
                return combo.id
        return None

    def add_load_combination(self, id: str, coefficients: dict[str, float],
                             combo_type='LinearSum') -> LoadCombination:
        """Create a load combination.

        combo_type options:
          LinearSum      — linear superposition: Σ coeff_i * case_i
          Envelope       — envelope: max and min of each component across cases
          AbsSum         — sum of absolutes: Σ |coeff_i * case_i|
          SRSS           — SRSS: √(Σ (coeff_i * case_i)²)
          NonLinearCombo — wraps a single non-linear analysis case (NonLinear or
                           GeometricNonlinear) with factor 1.0; no superposition.

        Renamed from Portuguese to English (26/09/2026); an old (Portuguese)
        value is still accepted and translated -- see coerce_combo_type.
        """
        valid = ('LinearSum', 'Envelope', 'AbsSum', 'SRSS',
                 'NonLinearCombo')
        # Accept an old (pre-26/09/2026) Portuguese value too -- coerce_combo_type
        # translates SomaLinear/Envolvente/SomaModulo/RaizSoma (and the old
        # 'SomaSodulo' misspelling) to their English equivalent; anything else,
        # including a genuinely invalid value, passes through unchanged for the
        # check below to reject.
        from .models import coerce_combo_type
        combo_type = coerce_combo_type(combo_type)
        if combo_type not in valid:
            raise ValueError(f"combo_type must be one of {valid}")
        # Same bug, same fix as add_load_case/add_analysis_case (26/09/2026):
        # no existence check at all, not even exact-match. With no dict behind
        # load_combinations, the corruption reads a little differently -- a
        # second entry with the same id becomes the invisible one, since
        # every lookup elsewhere is next(c for c in load_combinations if
        # c.id == ...), which returns the FIRST match -- but the underlying
        # bug is identical. Matched case-insensitively, like the other two;
        # raises rather than folding, same raw-layer reasoning.
        existing = self._find_load_combination_id_ci(id)
        if existing is not None:
            raise ValueError(
                f"Combination '{id}' already exists"
                + (f" (as '{existing}')" if existing != id else "") + ".")
        lc = LoadCombination(id=id, coefficients=dict(coefficients),
                             combo_type=combo_type)
        # Validate nesting/cycles and (via the expanded cases) the NonLinear
        # rules. Expansion needs the combo registered, so append first and roll
        # back if validation fails.
        self.load_combinations.append(lc)
        try:
            self.validate_combination_coefficients(coefficients, combo=lc)
        except Exception:
            self.load_combinations.pop()
            raise
        return lc

    # Same case-insensitive get-or-create treatment as create_load_case and
    # create_analysis_case, for consistency across the create_* family
    # (26/09/2026) -- until now this was a bare alias for
    # add_load_combination.
    def create_load_combination(self, id: str, coefficients: dict[str, float],
                                combo_type='LinearSum') -> LoadCombination:
        """Like add_load_combination, but a name matching an existing
        combination (any capitalisation) is returned unchanged instead of
        raising."""
        from ._compat import as_case_coefficients
        existing = self._find_load_combination_id_ci(id)
        if existing is not None:
            return next(c for c in self.load_combinations if c.id == existing)
        return self.add_load_combination(
            id, as_case_coefficients(coefficients), combo_type=combo_type)

    def expand_combination_coefficients(self, combo) -> dict[str, float]:
        """Resolve *combo* to a flat ``{analysis_case_id: factor}`` map,
        expanding any referenced sub-combinations recursively.

        A coefficient key that matches another combination's id is replaced by
        that combination's own (recursively expanded) coefficients, scaled by
        the referencing factor. Any *referenced* combination must be of type
        ``LinearSum`` (linear superposition), which makes the flattening exact;
        the referencing combination itself may be of any type.

        Note: fully flattening is only mathematically equivalent for a
        ``LinearSum`` parent. For an ``Envelope`` / ``SRSS`` /
        ``AbsSum`` parent the solver treats each referenced ``LinearSum``
        combination as a single input unit instead (see the solver); this method
        is still used there to enumerate the involved analysis cases and to
        detect cycles / illegal nesting.

        Raises ``ValueError`` on circular references or when a referenced
        combination is not of type ``LinearSum``.
        """
        combos_by_id = {c.id: c for c in self.load_combinations}
        flat: dict[str, float] = {}

        def _walk(c, scale, stack):
            for key, coeff in c.coefficients.items():
                sub = combos_by_id.get(key)
                if sub is None:
                    flat[key] = flat.get(key, 0.0) + scale * coeff
                    continue
                if sub.combo_type != 'LinearSum':
                    raise ValueError(
                        f"Combination '{c.id}' references combination '{key}', "
                        f"which is of type '{sub.combo_type}': only linear "
                        "(LinearSum) combinations may be used as inputs to "
                        "another combination.")
                if key in stack:
                    raise ValueError(
                        f"Circular combination reference involving '{key}'.")
                _walk(sub, scale * coeff, stack | {key})

        _walk(combo, 1.0, {combo.id})
        return flat

    def _validate_combination_nesting(self, combo):
        """Validate that *combo*'s combination references are legal.

        Edge rule (parent → referenced sub-combination): a referenced
        combination must be of type ``LinearSum`` **unless** the referencing
        combination is of type ``Envelope`` — an envelope is a component-wise
        max/min union and so may take inputs of any type. Also detects circular
        references.
        """
        combos_by_id = {c.id: c for c in self.load_combinations}

        def _walk(c, stack):
            for key in c.coefficients:
                sub = combos_by_id.get(key)
                if sub is None:
                    continue
                if sub.combo_type != 'LinearSum' and c.combo_type != 'Envelope':
                    raise ValueError(
                        f"Combination '{c.id}' (type '{c.combo_type}') cannot "
                        f"use combination '{key}' (type '{sub.combo_type}') as "
                        "an input: only 'Envelope' combinations may take "
                        "non-linear combinations as inputs; other types accept "
                        "only 'LinearSum' combinations.")
                if key in stack:
                    raise ValueError(
                        f"Circular combination reference involving '{key}'.")
                _walk(sub, stack | {key})

        _walk(combo, {combo.id})

    def validate_combination_coefficients(self, coefficients: dict[str, float],
                                          combo=None):
        """Validate a combination's coefficients.

        Checks combination-reference nesting/cycles (see
        :meth:`_validate_combination_nesting`) and the non-linear rule: a
        combination involving a non-linear analysis case (``NonLinear`` or
        ``GeometricNonlinear``) must be of type ``NonLinearCombo`` and hold only
        that single case, referenced directly, with factor 1.0. Conversely a
        ``NonLinearCombo`` must wrap exactly one such case.

        Raises ``ValueError`` if the rules are violated.
        """
        nonlinear_types = ('NonLinear', 'GeometricNonlinear')
        if combo is not None:
            self._validate_combination_nesting(combo)
            direct = combo.coefficients
            ctype = combo.combo_type
        else:
            direct = coefficients
            ctype = None

        # Every direct reference must be a real analysis case or another
        # combination — never a load case. A load case's results only ever
        # feed its own analysis case (the "twin" sharing its id, created
        # alongside it); referencing the load case id directly is only valid
        # because that twin exists under the same id, never as a fallback for
        # a *missing* twin. Silently accepting an unresolved id here is what
        # let load cases leak into combinations and into the case selectors
        # that list them — this must fail at definition time instead.
        combo_ids = {c.id for c in self.load_combinations}
        for cid in direct:
            if cid in self.analysis_cases_by_id or cid in combo_ids:
                continue
            hint = (" (there is a load case with this id, but load cases "
                    "are never a valid combination input — reference its "
                    "analysis case instead)"
                    if cid in self.load_cases_by_id else "")
            raise ValueError(
                f"Combination '{combo.id if combo is not None else '?'}' "
                f"references '{cid}', which is not an analysis case or "
                f"combination.{hint}")

        # Only the combination's *direct* analysis-case references matter here:
        # referenced sub-combinations were validated when they were created (a
        # non-linear case can only live inside a NonLinearCombo), so a parent may
        # reference such a wrapper without re-triggering the rule.
        nl_ids = [cid for cid in direct
                  if (ac := self.analysis_cases_by_id.get(cid)) is not None
                  and ac.analysis_type in nonlinear_types]

        if ctype == 'NonLinearCombo':
            if len(direct) != 1 or len(nl_ids) != 1 or set(direct) != set(nl_ids):
                raise ValueError(
                    "A 'NonLinearCombo' must contain exactly one non-linear "
                    "analysis case (NonLinear or GeometricNonlinear) and "
                    "nothing else.")
            ac_id = next(iter(direct))
            if abs(direct[ac_id] - 1.0) > 1e-9:
                raise ValueError(
                    "A 'NonLinearCombo' must use a combination factor of 1.0.")
            return

        if nl_ids:
            raise ValueError(
                "A combination involving a non-linear analysis case (NonLinear "
                "or GeometricNonlinear) must be of type 'NonLinearCombo' and "
                "contain only that single case with factor 1.0.")

    def add_spectral_function(self, id: str, description: str = "",
                              damping: float = 0.05,
                              points: list | None = None) -> SpectralFunction:
        sf = SpectralFunction(id=id, description=description,
                              damping=damping, points=list(points or []))
        self.spectral_functions[id] = sf
        return sf

    def remove_spectral_function(self, id: str):
        self.spectral_functions.pop(id, None)

    def add_analysis_case(self, id: str, analysis_type: str = 'Linear',
                          coefficients: dict | None = None, **kwargs) -> AnalysisCase:
        """
        Add an analysis case.
        analysis_type: 'Linear' | 'NonLinear' | 'Mass' | 'Modal' | 'Spectrum'
                       | 'GeometricNonlinear' | 'Sequence'.
        coefficients: {load_case_id: factor} for Linear, NonLinear and Mass types.
        """
        valid = ('Linear', 'NonLinear', 'Mass', 'Modal', 'Spectrum',
                 'GeometricNonlinear', 'Sequence')
        # Matched case-insensitively, then normalised to the canonical
        # spelling before storage -- same treatment as a case id (see
        # _find_load_case_id_ci): 'mass'/'MASS' should not raise just
        # because it is not spelled exactly like the tuple above.
        canonical = next((v for v in valid
                          if v.lower() == str(analysis_type).lower()), None)
        if canonical is None:
            raise ValueError(f"analysis_type must be one of {valid}")
        analysis_type = canonical
        # Same bug, same fix as add_load_case (26/09/2026): this had no
        # existence check at all -- not even exact-match -- so a second call
        # with the same id silently orphaned the first AnalysisCase in
        # self.analysis_cases while analysis_cases_by_id moved on to the
        # newer one. Matched case-insensitively, like add_load_case; raises
        # instead of folding, for the same reason: this is the raw layer,
        # create_analysis_case is the one that resolves a match for the
        # caller.
        existing = self._find_analysis_case_id_ci(id)
        if existing is not None:
            raise ValueError(
                f"Analysis case '{id}' already exists"
                + (f" (as '{existing}')" if existing != id else "") + ".")
        # Pull damping out of kwargs to avoid duplicate keyword if already in AnalysisCase defaults
        ac = AnalysisCase(id=id, analysis_type=analysis_type,
                          coefficients=dict(coefficients or {}), **kwargs)
        self.analysis_cases.append(ac)
        self.analysis_cases_by_id[id] = ac
        return ac

    def add_nodal_mass(self, node_id: str, mass_case_id: str,
                       mx: float = 0.0, my: float = 0.0, mtz: float = 0.0) -> NodalMass:
        """Add (or replace) a concentrated nodal mass for a specific Mass analysis case."""
        self.nodal_masses = [nm for nm in self.nodal_masses
                             if not (nm.node_id == node_id and nm.mass_case_id == mass_case_id)]
        nm = NodalMass(node_id=node_id, mass_case_id=mass_case_id, mx=mx, my=my, mtz=mtz)
        self.nodal_masses.append(nm)
        return nm

    def remove_nodal_mass(self, node_id: str, mass_case_id: str):
        self.nodal_masses = [nm for nm in self.nodal_masses
                             if not (nm.node_id == node_id and nm.mass_case_id == mass_case_id)]

    def remove_analysis_case(self, case_id: str):
        self.analysis_cases = [ac for ac in self.analysis_cases if ac.id != case_id]
        self.analysis_cases_by_id.pop(case_id, None)
        _refs.remove(self, 'analysis_case', case_id)

    def add_concrete_material(self, material_name: str, concrete_class: str,
                               steel_class: str, gamma_c=1.5,
                               gamma_s=1.15, alpha_cc=1.0) -> ConcreteMaterial:
        """Link EC2 concrete/steel properties to an existing material for RC design."""
        cm = ConcreteMaterial(
            material_name=material_name, concrete_class=concrete_class,
            steel_class=steel_class, gamma_c=gamma_c,
            gamma_s=gamma_s, alpha_cc=alpha_cc)
        self.concrete_materials[material_name] = cm
        return cm

    # ==================================================================
    # Simplified convenience API ("simple" facade)
    #
    # Thin, lossless wrappers over the generic methods above, for the common
    # cases: a magnitude + a named direction instead of signed components and a
    # reference frame, and a support created-and-assigned in one call. They read
    # ``self.domain`` and adapt (plane vs plate), and raise a clear error when a
    # direction/support does not apply to the current domain. Nothing new is
    # stored — each forwards to the generic method, so a model built this way is
    # identical to one built the long way.
    # ==================================================================

    def _default_load_case_id(self) -> str:
        """Get-or-create the simple facade's default load case ("LC1",
        self_weight_factor=0), reused across every facade call that omits
        load_case_id. Matched case-insensitively like every other named load
        case (see create_load_case/add_load_case) -- an existing 'lc1' is
        reused as itself, not shadowed by (or clashing with) a second,
        differently-cased 'LC1'."""
        existing = self._find_load_case_id_ci("LC1")
        if existing is not None:
            return existing
        self.add_load_case("LC1", self_weight_factor=0.0)
        return "LC1"

    def _resolve_load_case_id(self, load_case_id) -> str:
        """None -> the default case (get-or-create); otherwise resolve a
        LoadCase object or a raw id to its id string."""
        if load_case_id is None:
            return self._default_load_case_id()
        from ._compat import as_node_id
        return as_node_id(load_case_id)

    def _resolve_or_create_load_case_id(self, load_case) -> str:
        """None -> the default case (get-or-create); a LoadCase object or an
        existing name -> its id; a name with no existing match (case
        -insensitively -- see create_load_case) -> a new case created with
        that exact spelling. The get-or-create counterpart of
        _resolve_load_case_id, used by every create_* load method (never by
        add_*, which stays exact-match and creates nothing) so that a load
        case named in passing does not first need its own create_load_case
        call to exist."""
        if load_case is None:
            return self._default_load_case_id()
        from ._compat import as_node_id
        name = as_node_id(load_case)
        existing = self._find_load_case_id_ci(name)
        if existing is not None:
            return existing
        return self.create_load_case(name).id

    def create_bar_distributed_load(self, elements, load_case=None,
                                    q_i: float = None, q_j: float = None,
                                    direction: str = "down"):
        """Trapezoidal bar load — ``q_i``/``q_j`` at each end [kN/m].
        direction: ``"down"``/``"up"`` only, bars only."""
        if q_i is None or q_j is None:
            raise TypeError("create_bar_distributed_load() missing required "
                            "argument: 'q_i' and 'q_j'")
        from ._compat import as_node_id_list
        lc = self._resolve_or_create_load_case_id(load_case)
        d = direction.lower()
        if d not in ("down", "up"):
            raise ValueError("direction must be 'down' or 'up'")
        sign = -1.0 if d == "down" else 1.0
        ids, single = as_node_id_list(elements)
        out = [self._create_bar_distributed_load_one(eid, lc, sign * abs(q_i),
                                                      sign * abs(q_j))
               for eid in ids]
        return out[0] if single else out

    def _create_bar_distributed_load_one(self, element_id, load_case_id: str,
                                         q_i: float, q_j: float):
        if element_id in self.bar_elements_by_id:
            if self.domain == "plate":
                return self.add_distributed_load(element_id, load_case_id,
                                                  fze=q_i, fzd=q_j)
            return self.add_distributed_load(element_id, load_case_id,
                                              fye=q_i, fyd=q_j)
        obj = self.geometry_objects.get(element_id)
        if isinstance(obj, (GeoSegment, GeoMultisegment, GeoArc)):
            # add_line_distributed_load has no two-end-value form of its own
            # (its varying case is a named field, via add_field — a heavier
            # mechanism a facade should not reach for on your behalf); a
            # trapezoidal load on a not-yet-meshed line is the fundamental's
            # job, same as a field-varying one would be.
            raise ValueError(
                "a trapezoidal load on a not-yet-meshed geometry line object "
                "is not supported here — use create_uniform_load if the "
                "two ends are equal, or add_line_distributed_load's "
                "fy_field/fx_field (with add_field) for a real gradient")
        raise ValueError(
            f"'{element_id}' is not a bar element or a line geometry object.")

    def create_bar_point_load(self, elements, load_case=None, p: float = None,
                              at: float = 0.5, direction: str = "down"):
        """Concentrated load on a bar at a fraction of its span, bars only.

        Args:
            elements: a bar element (id, object, or a list of either —
                applied to each; a list returns a list).
            load_case: the load case (id, ``LoadCase`` object, or
                omitted/``None`` for the default case).
            p: load magnitude [kN] (a positive number).
            at: position as a fraction of the span from the i-end (0..1);
                0.5 is mid-span.
            direction: ``"down"``/``"up"`` (gravity axis), or ``"perp"``/
                ``"axial"`` (local, plane only) — as in
                :meth:`create_uniform_load`.
        """
        if p is None:
            raise TypeError(
                "create_bar_point_load() missing required argument: 'p'")
        if not 0.0 <= at <= 1.0:
            raise ValueError("at must be a fraction 0..1 of the span")
        from ._compat import as_node_id_list
        lc = self._resolve_or_create_load_case_id(load_case)
        d = direction.lower()
        ids, single = as_node_id_list(elements)
        out = [self._add_bar_point_load_one(eid, lc, p, at, d) for eid in ids]
        return out[0] if single else out

    def _add_bar_point_load_one(self, element_id, load_case_id: str, p: float,
                                at: float, d: str):
        bar = self.bar_elements_by_id.get(element_id)
        if bar is None:
            raise KeyError(f"Element '{element_id}' not found.")
        ni, nj = self.nodes.get(bar.node_i), self.nodes.get(bar.node_j)
        if ni is None or nj is None:
            raise KeyError(f"Element '{element_id}' has a dangling node.")
        import math
        a = at * math.hypot(nj.x - ni.x, nj.y - ni.y)
        if self.domain == "plate":
            if d == "down":
                return self.add_element_point_load(element_id, load_case_id, a,
                                                   fz=-abs(p))
            if d == "up":
                return self.add_element_point_load(element_id, load_case_id, a,
                                                   fz=abs(p))
            raise ValueError(
                "in the plate domain a bar point load is transverse — use "
                "direction 'down' or 'up'")
        if d == "down":
            return self.add_element_point_load(element_id, load_case_id, a,
                                               fy=-abs(p))
        if d == "up":
            return self.add_element_point_load(element_id, load_case_id, a,
                                               fy=abs(p))
        if d == "perp":
            return self.add_element_point_load(element_id, load_case_id, a,
                                               fy=p, coord_sys="local")
        if d == "axial":
            return self.add_element_point_load(element_id, load_case_id, a,
                                               fx=p, coord_sys="local")
        raise ValueError("direction must be 'down', 'up', 'perp' or 'axial'")

    def create_node_load(self, nodes, load_case=None, p: float = None,
                         direction: str = "down"):
        """Concentrated load at a node, magnitude ``p`` [kN]. direction:
        ``"down"``/``"up"`` (both domains), ``"left"``/``"right"`` (plane
        only). Else (plate in-plane, a moment): :meth:`add_point_load`.

        Args:
            nodes: a node (id, ``Node`` object, or a list of either —
                applied to each; a list returns a list).
            load_case: the load case (id, ``LoadCase`` object, or
                omitted/``None`` for the default case).
            p: load magnitude [kN] (a positive number).
            direction: ``"down"`` (global −Y in plane, −Z in plate),
                ``"up"`` (opposite), ``"left"`` (global −X, plane only) or
                ``"right"`` (global +X, plane only).
        """
        if p is None:
            raise TypeError("create_node_load() missing required argument: 'p'")
        d = direction.lower()
        if d not in ("down", "up", "left", "right"):
            raise ValueError(
                "direction must be 'down', 'up', 'left' or 'right' (for "
                "anything else use add_point_load)")
        from ._compat import as_node_id_list
        lc = self._resolve_or_create_load_case_id(load_case)
        ids, single = as_node_id_list(nodes)
        if self.domain == "plate":
            if d in ("left", "right"):
                raise ValueError(
                    "'left'/'right' are plane-domain only — a plate node "
                    "load is transverse only; use add_point_load for an "
                    "in-plane force on a plate node")
            val = -abs(p) if d == "down" else abs(p)
            out = [self.add_point_load(nid, lc, fz=val) for nid in ids]
        else:
            if d in ("down", "up"):
                val = -abs(p) if d == "down" else abs(p)
                out = [self.add_point_load(nid, lc, fy=val) for nid in ids]
            else:
                val = -abs(p) if d == "left" else abs(p)
                out = [self.add_point_load(nid, lc, fx=val) for nid in ids]
        return out[0] if single else out

    # Unified uniform-load facade: dispatches per target's actual kind, same
    # lookup precedent as add_area_load's own dispatch (bar_elements_by_id /
    # tri_elements_by_id / quad_elements_by_id / geometry_objects). Replaces
    # the old create_bar_uniform_load / create_area_uniform_load / (bar-only)
    # add_uniform_load split: one method, targets may mix bar-like and
    # area-like ids in the same call.
    def create_uniform_load(self, targets, load_case, value,
                            direction: str = "down", coord_sys: str = "global"):
        """Uniform load of ``value``, on bar-like targets [kN/m] or
        area-like targets [kN/m²] (pressure — dispatched per target's
        actual kind, may mix both in one call). direction:
        ``"down"``/``"up"`` (both kinds), ``"perp"``/``"axial"`` (bar-like
        only, local axes). Preferred over add_distributed_load for a bar
        when constant: one value, no end left at zero by accident.

        Args:
            targets: a single id/object, or a list/tuple of ids/objects —
                each item is either a raw id string or the actual element/
                object instance (its ``.id`` is used). Bar-like and
                area-like targets may be mixed in one call; each is routed
                independently.
            load_case: the load case (id or ``LoadCase`` object).
            value: load magnitude — force/length for bar-like targets,
                pressure for area-like targets (sign follows ``direction``).
            direction: ``"down"``/``"up"`` (both kinds), or ``"perp"``/
                ``"axial"`` (bar-like targets only, local axes).
            coord_sys: ``"global"``/``"local"`` — only meaningful for
                bar-like targets; silently has no effect on area-like ones.

        Validates every target up front (kind, direction, existence) before
        applying anything, so a call either fully applies or raises without
        side effects.

        Returns:
            dict mapping each resolved target id to ``'bar'`` or ``'area'``,
            indicating how it was classified and applied.
        """
        from ._compat import as_node_id_list
        lc = self._resolve_or_create_load_case_id(load_case)
        d = direction.lower()
        if d not in ("down", "up", "perp", "axial"):
            raise ValueError(
                "create_uniform_load: direction must be 'down', 'up', "
                "'perp' or 'axial'.")
        ids, _single = as_node_id_list(targets)

        # Pass 1: classify and validate every target before applying anything.
        kinds = {}
        for tid in ids:
            if tid in self.bar_elements_by_id:
                kinds[tid] = "bar"
            elif (tid in self.geometry_objects
                  and isinstance(self.geometry_objects[tid],
                                 (GeoSegment, GeoMultisegment, GeoArc))):
                kinds[tid] = "bar"
            elif tid in self.tri_elements_by_id or tid in self.quad_elements_by_id:
                kinds[tid] = "area"
            elif (tid in self.geometry_objects
                  and isinstance(self.geometry_objects[tid],
                                 (GeoRectangle, GeoPolygon))):
                kinds[tid] = "area"
            elif tid in self.load_cases_by_id:
                raise ValueError(
                    f"create_uniform_load: '{tid}' is not a bar element, a "
                    "tri/quad element, or a geometry object — cannot "
                    "resolve this target (it is a load case id — did you "
                    "mean to pass a different target?)")
            else:
                raise ValueError(
                    f"create_uniform_load: '{tid}' is not a bar element, a "
                    "tri/quad element, or a geometry object — cannot "
                    "resolve this target.")
            if kinds[tid] == "area" and d in ("perp", "axial"):
                raise ValueError(
                    f"create_uniform_load: direction '{direction}' is not "
                    f"valid for area target '{tid}' (tri/quad element or "
                    "surface object) — area targets only support "
                    "'down'/'up'.")

        # Pass 2: apply.
        out = {}
        for tid in ids:
            if kinds[tid] == "bar":
                self._create_uniform_bar_load_one(tid, lc, value, d, coord_sys)
            else:
                self._create_uniform_area_load_one(tid, lc, value, d)
            out[tid] = kinds[tid]
        return out

    def _create_uniform_bar_load_one(self, element_id, load_case_id: str,
                                     q: float, d: str, coord_sys: str):
        if element_id in self.bar_elements_by_id:
            if self.domain == "plate":
                if d == "down":
                    return self.add_distributed_load(element_id, load_case_id,
                                                      fze=-abs(q), fzd=-abs(q))
                if d == "up":
                    return self.add_distributed_load(element_id, load_case_id,
                                                      fze=abs(q), fzd=abs(q))
                raise ValueError(
                    "in the plate domain a bar load is transverse — use "
                    "direction 'down' or 'up' (not 'perp'/'axial')")
            if d == "down":
                return self.add_distributed_load(element_id, load_case_id,
                                                  fye=-abs(q), fyd=-abs(q),
                                                  coord_sys=coord_sys)
            if d == "up":
                return self.add_distributed_load(element_id, load_case_id,
                                                  fye=abs(q), fyd=abs(q),
                                                  coord_sys=coord_sys)
            if d == "perp":
                return self.add_distributed_load(element_id, load_case_id,
                                                  fye=q, fyd=q,
                                                  coord_sys="local")
            if d == "axial":
                return self.add_distributed_load(element_id, load_case_id,
                                                  fxe=q, fxd=q,
                                                  coord_sys="local")
            raise ValueError("direction must be 'down', 'up', 'perp' or 'axial'")

        obj = self.geometry_objects.get(element_id)
        if isinstance(obj, (GeoSegment, GeoMultisegment, GeoArc)):
            if self.domain != "plane":
                raise ValueError(
                    "a line-object bar load is plane-domain only for now "
                    "(add_line_distributed_load has no transverse/plate slot)")
            if d == "down":
                return self.add_line_distributed_load(element_id, load_case_id, fy=-abs(q))
            if d == "up":
                return self.add_line_distributed_load(element_id, load_case_id, fy=abs(q))
            raise ValueError(
                "a line-object bar load has no single local axis before "
                "meshing — use direction 'down' or 'up' (not 'perp'/'axial')")

        raise ValueError(
            f"'{element_id}' is not a bar element or a line geometry object.")

    def _create_uniform_area_load_one(self, target_id, load_case_id: str,
                                      p: float, d: str):
        if self.domain != "plate":
            raise ValueError("area pressure is a plate-domain load")
        val = -abs(p) if d == "down" else abs(p)
        return self.add_area_load(target_id, load_case_id, pz=val)

    def create_self_weight(self, factor: float = 1.0, load_case=None) -> LoadCase:
        """Set/create self-weight factor.

        Args:
            factor: the self-weight factor (1.0 = full self-weight).
            load_case: the load case (id, ``LoadCase`` object, or
                omitted/``None`` for the default case, auto-created if new).
        """
        lc_id = self._resolve_or_create_load_case_id(load_case)
        lc = self.load_cases_by_id[lc_id]
        lc.self_weight_factor = factor
        return lc

    def create_temperature(self, target, uniform: float = 0.0,
                           gradient: float = 0.0, load_case=None):
        """Temperature action, any target — dispatches by type.

        Args:
            target: a bar element, triangle element, quad element, or
                geometry object (id or object) -- or a list/tuple of any
                mix of these, applied to each in turn (list in, list out,
                same convention as :meth:`pin`).
            uniform: uniform temperature change.
            gradient: through-thickness/depth temperature gradient.
            load_case: the load case (id, ``LoadCase`` object, or
                omitted/``None`` for the default case).
        """
        from ._compat import as_object_id_list
        lc = self._resolve_or_create_load_case_id(load_case)
        tids, single = as_object_id_list(target)
        out = [self._create_temperature_one(tid, lc, uniform, gradient)
              for tid in tids]
        return out[0] if single else out

    def _create_temperature_one(self, tid, lc, uniform, gradient):
        """create_temperature's per-id dispatch, split out so the method
        itself only has to handle the single-vs-list normalisation."""
        if tid in self.bar_elements_by_id:
            return self.add_temperature_load(tid, lc, dt_uniform=uniform,
                                             dt_gradient=gradient)
        if tid in self.tri_elements_by_id:
            return self.add_tri_temperature_load(tid, lc, dt_i=uniform,
                                                 dt_j=uniform, dt_k=uniform,
                                                 dt_gradient=gradient)
        if tid in self.quad_elements_by_id:
            return self.add_quad_temperature_load(tid, lc, dt_i=uniform,
                                                   dt_j=uniform, dt_k=uniform,
                                                   dt_l=uniform,
                                                   dt_gradient=gradient)
        obj = self.geometry_objects.get(tid)
        if isinstance(obj, (GeoSegment, GeoMultisegment, GeoArc)):
            return self.add_line_temperature_load(tid, lc, dt_uniform=uniform,
                                                  dt_gradient=gradient)
        if isinstance(obj, (GeoRectangle, GeoPolygon)):
            return self.add_area_temperature_load(tid, lc, dt_uniform=uniform,
                                                  dt_gradient=gradient)
        if tid in self.load_cases_by_id:
            raise ValueError(
                f"'{tid}' is not a bar, triangle, quad, or geometry object "
                "(it is a load case id — did you mean to pass a different "
                "target?)")
        raise ValueError(
            f"'{tid}' is not a bar, triangle, quad, or geometry object.")

    def create_edge_load(self, object_id: str, edge, load_case=None,
                         p: float = None, direction: str = "perp"):
        """Distributed load along one edge — ``p`` [kN/m]. direction:
        ``'perp'`` (normal) or ``'tang'`` (follows edge winding, not global —
        rectangle ``'top'`` points left). edge: same as :meth:`support_edge`.
        For a global direction, prefer :meth:`create_node_load`."""
        if p is None:
            raise TypeError("create_edge_load() missing required argument: 'p'")
        d = direction.lower()
        if d not in ("perp", "tang"):
            raise ValueError("direction must be 'perp' or 'tang'")
        from ._compat import as_object_id
        object_id = as_object_id(object_id)
        obj = self.geometry_objects.get(object_id)
        if obj is None:
            raise KeyError(f"Object '{object_id}' not found.")
        nids = [n for n in getattr(obj, "node_ids", []) if n in self.nodes]
        if len(nids) < 2:
            raise ValueError(f"Object '{object_id}' has no edge nodes.")
        segments = self._edge_segments(obj, nids, edge)
        from ._compat import auto_name
        lc = self._resolve_or_create_load_case_id(load_case)
        kw = {"pn": abs(p), "coord_sys": "local"} if d == "perp" \
            else {"pt": abs(p), "coord_sys": "local"}
        out = []
        for na, nb in segments:
            existing = {e.id for e in self.surface_edge_loads}
            eid = auto_name("EL", existing)
            out.append(self.add_surface_edge_load(eid, object_id, na, nb, lc,
                                                   **kw))
        return out[0] if len(out) == 1 else out

    # A fundamental after all, not a facade — script_export.py serialises
    # `support_settlements` by writing a literal `model.add_support_settlement(
    # ...)` call for each one, so the name has to stay a real, single-node
    # method for File > Export as Python to keep producing runnable scripts.
    # create_support_settlement (below) is the permissive wrapper on top,
    # same relationship as add_load_case/create_load_case.
    def add_support_settlement(self, node_id: str, load_case_id: str,
                               ux=0.0, uy=0.0, tz=0.0,
                               w=None, tx=None, ty=None) -> SupportSettlement:
        """Prescribe a non-zero displacement at a supported node.

        Plate domain: use ``w`` [m], ``tx``/``ty`` [rad] — aliases of the same
        three storage slots as ux/uy/tz.
        """
        if w is not None:
            ux = w
        if tx is not None:
            uy = tx
        if ty is not None:
            tz = ty
        ss = SupportSettlement(node_id=node_id, load_case_id=load_case_id,
                               ux=ux, uy=uy, tz=tz)
        # Replace any existing settlement for same node+case
        self.support_settlements = [s for s in self.support_settlements
                                    if not (s.node_id == node_id and s.load_case_id == load_case_id)]
        self.support_settlements.append(ss)
        return ss

    def create_support_settlement(self, nodes: "str | Node | list",
                                  load_case=None, ux=0.0, uy=0.0, tz=0.0,
                                  w=None, tx=None, ty=None):
        """Like add_support_settlement: load_case optional, nodes
        single ref or list."""
        from ._compat import as_node_id_list
        lc = self._resolve_or_create_load_case_id(load_case)
        ids, single = as_node_id_list(nodes)
        out = [self.add_support_settlement(nid, lc, ux=ux, uy=uy, tz=tz,
                                           w=w, tx=tx, ty=ty)
               for nid in ids]
        return out[0] if single else out

    def _simple_support(self, node_id: str, name: str, **restraints):
        """Get-or-create a named support with the given restraints and assign it
        to *node_id* (deduping the definition by name).

        The node's earlier support is REPLACED, not added to. pin/fix/roller/
        symm/create_support all come through here, and "make N1 a roller" is a
        change of support: two assignments on one node restrain the union of
        both, so the old pin stayed in force and the change did nothing. (The
        low-level assign_support still appends; loaders and the edge-support
        expansion rely on that.)"""
        if name not in self.supports:
            self.add_support(name, **restraints)
        self.remove_support_assignment(node_id)
        return self.assign_support(node_id, name)

    # add_support alone only *defines* a restraint; nothing is actually
    # supported until a separate assign_support call, and the shape reads as
    # though it already did the whole job — the single most common mistake
    # this API has measured (see add_support's own docstring).
    # create_support always does both, and — like pin/fix — dedupes the same
    # restraint combination under its canonical name (canonical_support_name
    # in models.py: "PIN", "ROLLER-X", "SIMPLE", "CLAMP-X", ...) instead of a
    # fresh name every call; a combination with no canonical name falls back
    # to an auto id ("S1", "S2", ...).
    def create_support(self, nodes: "str | Node | list", ux=False, uy=False,
                       tz=False, w=None, tx=None, ty=None,
                       name: str | None = None):
        """Create+assign a support. True fixes that DOF, False frees it —
        all default False, so create_support(nodes) restrains nothing.

        Args:
            nodes: a single node id/Node, or a list of either.
            ux/uy/tz/w/tx/ty: restraint flags (w/tx/ty are the plate-domain
                aliases for the same three slots).
            name: override the support name; auto-resolved when omitted.
        """
        from ._compat import as_node_id_list, auto_name
        from .models import canonical_support_name
        rux, ruy, rtz = ux, uy, tz
        if w is not None:
            rux = w
        if tx is not None:
            ruy = tx
        if ty is not None:
            rtz = ty
        if name is None:
            name = (canonical_support_name(rux, ruy, rtz, domain=self.domain)
                    or auto_name("S", self.supports))
        ids, single = as_node_id_list(nodes)
        out = [self._simple_support(nid, name, ux=ux, uy=uy, tz=tz,
                                    w=w, tx=tx, ty=ty) for nid in ids]
        return out[0] if single else out

    def pin(self, nodes: "str | Node | list"):
        """Pinned support (create+assign in one call): translation fixed,
        rotation free. Plane: ``ux, uy``. Plate: ``w``. nodes: single
        id/``Node`` or list — list in, list out."""
        from ._compat import as_node_id_list
        ids, single = as_node_id_list(nodes)
        if self.domain == "plate":
            out = [self._simple_support(nid, "PIN", w=True) for nid in ids]
        else:
            out = [self._simple_support(nid, "PIN", ux=True, uy=True) for nid in ids]
        return out[0] if single else out

    def fix(self, nodes: "str | Node | list"):
        """Fixed (encastré): restrains everything. Plane: ``ux,uy,θz``.
        Plate: ``w,θx,θy``. nodes: single ref or list, see :meth:`pin`."""
        from ._compat import as_node_id_list
        from .models import canonical_support_name
        ids, single = as_node_id_list(nodes)
        if self.domain == "plate":
            name = canonical_support_name(True, True, True, domain="plate")
            out = [self._simple_support(nid, name, w=True, tx=True, ty=True)
                   for nid in ids]
        else:
            name = canonical_support_name(True, True, True, domain="plane")
            out = [self._simple_support(nid, name, ux=True, uy=True, tz=True)
                   for nid in ids]
        return out[0] if single else out

    def roller(self, nodes: "str | Node | list", free: str = "x"):
        """Roller (plane only). free='x' restrains uy; free='y' restrains
        ux. Plate: use :meth:`symm`. nodes: single ref or list, see
        :meth:`pin`."""
        if self.domain != "plane":
            raise ValueError(
                "roller is a plane-domain support; in the plate domain use symm")
        from ._compat import as_node_id_list
        from .models import canonical_support_name
        ids, single = as_node_id_list(nodes)
        f = free.lower()
        if f == "x":
            name = canonical_support_name(False, True, False, domain="plane")
            out = [self._simple_support(nid, name, uy=True) for nid in ids]
        elif f == "y":
            name = canonical_support_name(True, False, False, domain="plane")
            out = [self._simple_support(nid, name, ux=True) for nid in ids]
        else:
            raise ValueError("free must be 'x' or 'y'")
        return out[0] if single else out

    def symm(self, nodes: "str | Node | list", axis: str = "x"):
        """Symmetry (plate only): ``w`` free, rotation restrained
        (``axis="x"``→θx, ``"y"``→θy). Plane domain: use :meth:`roller`.
        nodes: single ref or list, see :meth:`pin`."""
        if self.domain != "plate":
            raise ValueError(
                "symm is a plate-domain support; in the plane domain use roller")
        from ._compat import as_node_id_list
        from .models import canonical_support_name
        ids, single = as_node_id_list(nodes)
        a = axis.lower()
        if a == "x":
            name = canonical_support_name(False, True, False, domain="plate")
            out = [self._simple_support(nid, name, tx=True) for nid in ids]
        elif a == "y":
            name = canonical_support_name(False, False, True, domain="plate")
            out = [self._simple_support(nid, name, ty=True) for nid in ids]
        else:
            raise ValueError("axis must be 'x' or 'y'")
        return out[0] if single else out

    def create_rc_section(self, name: str, b: float, h: float,
                       concrete: str = "C30/37", steel: str = "B500B",
                       cover: float = 0.045) -> Section:
        # Looks the strengths up from the Eurocode databases and builds a
        # Concrete material carrying fck/fyk (so the RC design works) plus a
        # b×h rectangular section on it. The material is shared/deduped
        # across sections of the same concrete+steel grade.
        """RC rect. section + material in one call.

        Args:
            name: the section name.
            b: width [m].
            h: height [m].
            concrete: concrete class (e.g. ``"C30/37"``).
            steel: reinforcement grade (e.g. ``"B500B"``).
            cover: mechanical cover to the bars [m].
        """
        mat = self.create_rc_material(concrete=concrete, steel=steel)
        return self.create_bar_section(name, mat, b=b, h=h,
                                       shape="Rectangular", rc_cover=cover)

    def _rc_material(self, concrete: str, steel: str) -> str:
        """Get-or-create a Concrete material for the given grades (deduped by
        name), carrying the ``fck``/``fyk`` the RC design needs. Returns its
        name. Shared by the RC section and panel shortcuts."""
        from .databases import material_from_grade
        mat = f"C_{concrete}_{steel}".replace("/", "_")
        if mat not in self.materials:
            self.add_material(mat, **material_from_grade(
                "Concrete", concrete, reinforcement=steel))
        return mat

    def create_steel_section(self, name: str, profile: str,
                          grade: str = "S275") -> Section:
        # Looks the profile up by name (IPE/HEA/HEB/HEM, CHS, RHS, SHS) and
        # fills every calculation property from the catalogue, and builds a
        # Steel material carrying fy from the grade. The material is
        # shared/deduped across sections of the same grade.
        """Steel section from a catalogue profile + material.

        Args:
            name: the section name.
            profile: catalogue profile name (e.g. ``"IPE300"``).
            grade: structural-steel grade (e.g. ``"S275"``).
        """
        from .databases import steel_profile
        found = steel_profile(profile)
        if found is None:
            from .databases import _STEEL_PROFILES
            families = ", ".join(
                f"{fam} ({len(entries)})"
                for fam, entries in _STEEL_PROFILES.items())
            raise ValueError(
                f"unknown steel profile '{profile}' -- no profile with that "
                f"exact name in the catalogue; available families: {families}")
        shape, e = found
        from .databases import profile_shear_areas
        av_y, av_z = profile_shear_areas(shape, e)
        mat = self.create_steel_material(steel=grade)
        return self.create_bar_section(
            name, mat, b=e["b_m"], h=e["h_m"], tw=e.get("tw_m", 0.0),
            tf=e.get("tf_m", 0.0), shape=shape, profile_name=profile,
            area_override=e.get("A_m2"), inertia_override=e.get("Iy_m4"),
            inertia_minor_override=e.get("Iz_m4"),
            wel_y_override=e.get("Wel_y_m3"), wpl_y_override=e.get("Wpl_y_m3"),
            wel_z_override=e.get("Wel_z_m3"), wpl_z_override=e.get("Wpl_z_m3"),
            av_y_override=av_y, av_z_override=av_z,
            warping_override=e.get("Iw_m6"), wt_override=e.get("Wt_m3"),
            curve_y_override=e.get("curve_y"), curve_z_override=e.get("curve_z"),
            torsion_override=e.get("J_m4"))

    # The timber counterpart of create_rc_section/create_steel_section — no
    # add_* equivalent existed before this (new, see dev/XDFEM2D_ENGINE.md §7).
    def create_timber_bar_section(self, name: str, b: float, h: float,
                                  timber: str = "C24",
                                  service_class: str = "SC1") -> Section:
        """Create a timber rectangular section (and its material) in one
        call. Looks the strengths up from the Eurocode (EN 338) databases
        and builds a Timber material carrying fmk/fvk/fc0k/ft0k, plus a
        ``b × h`` rectangular section on it. The material is shared/deduped
        across sections of the same timber grade.

        Args:
            name: the section name.
            b: width [m].
            h: height [m].
            timber: timber grade (e.g. ``"C24"``, ``"GL24h"``).
            service_class: EN 1995 service class (``"SC1"``/``"SC2"``/``"SC3"``),
                used by the timber design checks (duration/moisture factors).
        """
        mat = self.create_timber_material(timber=timber)
        return self.create_bar_section(name, mat, b=b, h=h, shape="Rectangular",
                                       timber_service_class=service_class)

    # Renamed into the `create_*` vocabulary (see dev/XDFEM2D_ENGINE.md).
    # Not deprecated — no warning; old scripts and the assistant keep working.
    add_rc_section = create_rc_section
    add_steel_section = create_steel_section

    def _coerce_points(self, outline) -> list:
        """Parse a flexible outline into ``[(x, y), …]``.

        Accepts, interchangeably:

        * a sequence of ``(x, y)`` pairs (tuples or lists);
        * a **flat** sequence ``[x1, y1, x2, y2, …]`` — coordinates only, see
          below;
        * a sequence of ``{"x": …, "y": …}`` dicts;
        * two **columns** ``[[x1, x2, …], [y1, y2, …]]`` (an all-x list and an
          all-y list) — used when there are **3 or more** points; with exactly
          two points write the pairs form, since ``[[a, b], [c, d]]`` is read as
          the two points ``(a, b)`` and ``(c, d)``.
        * an existing node's id (a string) or a ``Node`` object, in place of
          any entry above (except inside the flat-coordinate form, which has
          no room for anything but numbers) — resolved to that node's
          *current* ``(x, y)``, so the polygon ties into the model's real
          geometry there instead of a new coincident node.

        The form is detected from the shape, so the caller (and the assistant)
        need not remember one convention. A list may freely mix ids, ``Node``
        objects and coordinates entry by entry (e.g. ``["N0", "N4", (3.0,
        4.0)]``)."""
        pts = list(outline or [])
        if not pts:
            raise ValueError("outline is empty")
        if isinstance(pts[0], (int, float)):                 # flat [x1,y1,x2,y2,…]
            if len(pts) % 2 != 0:
                raise ValueError("a flat coordinate list must have an even count")
            return [(float(pts[i]), float(pts[i + 1]))
                    for i in range(0, len(pts), 2)]
        # Two columns [[x…], [y…]]: exactly two equal-length sequences of 3+.
        if (len(pts) == 2 and all(isinstance(c, (list, tuple)) for c in pts)
                and len(pts[0]) == len(pts[1]) >= 3):
            xs, ys = pts
            return [(float(x), float(y)) for x, y in zip(xs, ys)]
        out = []
        for p in pts:
            if isinstance(p, Node):
                out.append((p.x, p.y))
            elif isinstance(p, str):
                node = self.nodes.get(p)
                if node is None:
                    raise ValueError(f"outline: no node '{p}' in this model")
                out.append((node.x, node.y))
            elif isinstance(p, dict):
                out.append((float(p["x"]), float(p["y"])))
            else:
                out.append((float(p[0]), float(p[1])))
        return out

    def create_from_template(self, kind: str, **params) -> "Structure2D":
        """A standard structure in one call, on an EMPTY model."""
        from .template_api import (build_template, check_params,
                                   domain_mismatch, fill_in_place,
                                   has_geometry, normalize_kind)
        kind = normalize_kind(kind)
        check_params(kind, params)
        if has_geometry(self):
            raise ValueError(
                "create_from_template only fills an empty model, and this one "
                "already has geometry: build on it with create_node, "
                "create_bar_element, create_polygon and the other create_* calls.")
        mismatch = domain_mismatch(kind, self)
        if mismatch:
            raise ValueError(mismatch)
        fill_in_place(self, build_template(kind, **params))
        return self

    def create_polygon(self, outline, thickness: float,
                       material: str = "C30/37", target_size: float = 0.5,
                       id: str | None = None):
        # The domain decides the kind: a plate model builds a slab (plate
        # bending), a plane model a wall/membrane. outline is flexible (see
        # _coerce_points): two points are read as the opposite corners of a
        # rectangle, three or more as a polygon. The RC material (with
        # fck/fyk) is created/shared like create_rc_section.
        #
        # Reinforcement grade and cover are fixed internal defaults here
        # (not caller-settable through this call, unlike the old
        # create_area) — adjust them on the created section afterward (e.g.
        # via the GUI's area-section editor) if a job needs something else.
        """RC polygon (region + section) in one call — keep the return:
        ``.node_ids`` are real corner nodes (rectangle CCW: bottom-left,
        bottom-right, top-right, top-left), for e.g.
        :meth:`create_node_load` on one corner.

        Args:
            outline: exactly 2 points → opposite corners of a rectangle; 3+
                → a general polygon; pairs, flat list or ``{"x","y"}`` dicts —
                any entry (except in the flat-list form) may instead be an
                existing node's id or ``Node`` object, mixed freely with
                coordinates.
            thickness: panel thickness [m].
            target_size: target mesh size [m].
            id: the geometry-object id (also names its section ``<id>_sec``);
                optional, auto ("Panel1", "Panel2", ...).
            material: an existing material name, a ``Material`` object, or
                an unresolved Eurocode concrete class (e.g. ``"C30/37"``,
                auto-created via create_rc_material with steel fixed at
                "B500B" if not already a known material).
        """
        from ._compat import auto_name, as_material_name
        if id is None:
            id = auto_name("Panel", self.geometry_objects)
        cover = 0.045
        pts = self._coerce_points(outline)
        mat = as_material_name(material)
        if mat not in self.materials:
            resolved = self._auto_material_from_class(mat)
            if resolved is not None:
                mat = resolved
        sec = f"{id}_sec"
        # Pair a same-named QuadSection alongside the tri/plate one (26/09/2026,
        # Matias: "'quad' e o novo default do programa" -- every add_geo_*
        # already prefers quads by default (see add_geo_rectangle/
        # add_geo_polygon's prefer_quad=True), the same way
        # templates._default_new_structure() pairs 'CST'/'Slab' for a brand
        # new model -- but geo_expand._expand_surface only meshes into quads
        # when struc.quad_sections has an entry under THIS SAME name, so
        # without this pairing prefer_quad=True has nothing to prefer and
        # every create_polygon call fell back to triangles regardless.
        # Formulation mirrors the tri/plate pick: QM6 (membrane) alongside
        # CST in the plane domain, MITC4 (plate bending) alongside MITC3 in
        # the plate domain -- the same pairs _default_new_structure() uses.
        if self.domain == "plate":
            self.add_plate_section(sec, mat, thickness=thickness, rc_cover=cover)
            self.add_quad_section(sec, mat, thickness=thickness, rc_cover=cover,
                                  formulation="MITC4")
        else:
            self.add_tri_section(sec, mat, thickness=thickness, rc_cover=cover)
            self.add_quad_section(sec, mat, thickness=thickness, rc_cover=cover,
                                  formulation="QM6")
        if len(pts) == 2:
            return self.add_geo_rectangle(id, pts[0], pts[1],
                                          section_name=sec,
                                          target_size=target_size)
        return self.add_geo_polygon(id, pts, section_name=sec,
                                    target_size=target_size)

    # Renamed into the `create_*` vocabulary (see dev/XDFEM2D_ENGINE.md).
    # Not deprecated — no warning; old scripts and the assistant keep working.
    add_panel = create_polygon
    # Convenience alias — "panel" is the term used throughout the AI-facing
    # docs and _KIND_SYNONYMS for this shape; kept alongside add_panel (the
    # old add_*-vocabulary name) rather than replacing it, since old scripts
    # already call add_panel and must keep working unchanged.
    create_panel = create_polygon

    def _edge_support_want(self, kind: str):
        """The (ux/uy/tz)-slot restraints a ``support_edge`` *kind* stands for,
        domain-aware — the single mapping that pin_edge/fix_edge/roller_edge/
        symm_edge and support_edge all resolve through, so the six ``kind``
        strings and the four named conveniences never drift apart."""
        k = kind.lower()
        plate = getattr(self, 'domain', 'plane') == 'plate'
        if k == "pin":
            return (True, False, False) if plate else (True, True, False)
        if k == "fix":
            return (True, True, True)
        if k in ("roller-x", "roller-y"):
            if plate:
                raise ValueError("roller-x/roller-y are plane-domain; in the "
                                 "plate domain use symm-x/symm-y")
            return (False, True, False) if k == "roller-x" else (True, False, False)
        if k in ("symm-x", "symm-y"):
            if not plate:
                raise ValueError("symm-x/symm-y are plate-domain; in the plane "
                                 "domain use roller-x/roller-y")
            return (False, True, False) if k == "symm-x" else (False, False, True)
        raise ValueError(
            "kind must be 'pin', 'fix', 'symm-x', 'symm-y', 'roller-x' or "
            "'roller-y'")

    def _edge_segments(self, obj, nids, edge) -> list:
        """Consecutive ``(node_a, node_b)`` pairs along the requested edge of
        *obj* (``node_ids`` = *nids*), in perimeter order.

        ``edge='all'`` → every segment around the whole perimeter; a named
        side (``bottom/right/top/left``) → a rectangle's edge (one segment);
        an integer ``i`` → the edge from vertex ``i`` to ``i+1`` (wrapping,
        one segment). The ordered counterpart of :meth:`_edge_nodes` — needed
        wherever a segment's own direction matters, not just which nodes
        participate (as for an edge load's ``node_a``/``node_b``)."""
        n = len(nids)
        if isinstance(edge, str) and edge.lower() == "all":
            return [(nids[i], nids[(i + 1) % n]) for i in range(n)]
        if isinstance(edge, str):
            named = {"bottom": (0, 1), "right": (1, 2),
                     "top": (2, 3), "left": (3, 0)}
            if n != 4:
                raise ValueError(
                    "named edges (bottom/right/top/left) need exactly 4 "
                    "corners (this object has "
                    f"{n}) — use an edge index or 'all'. A rectangle built "
                    "from a 4-point outline works the same as one built "
                    "from 2 opposite corners, as long as the 4 points are "
                    "given in order: bottom-left, bottom-right, top-right, "
                    "top-left")
            e = edge.lower()
            if e not in named:
                raise ValueError(
                    "edge must be 'bottom'/'right'/'top'/'left', an index, or "
                    "'all'")
            i, j = named[e]
            return [(nids[i], nids[j])]
        i = int(edge) % n                                    # edge index
        return [(nids[i], nids[(i + 1) % n])]

    def _edge_nodes(self, obj, nids, edge) -> set:
        """Node ids of the requested edge of *obj* — see
        :meth:`_edge_segments` for what ``edge`` accepts. Flattens the
        ordered segments since a support does not care which node came
        first, only which nodes are on the edge."""
        segs = self._edge_segments(obj, nids, edge)
        return {n for pair in segs for n in pair}

    # Writes the resolved support onto the object's per-edge edge_supports list
    # (via support_object_edge), so every mesh node the edge generates is
    # restrained. This is the reliable route: the earlier version restrained
    # only the two corner nodes and left edge_support_mode to propagate the
    # rest — exactly the approach the per-edge list was added to replace. The
    # canonical-name path exists because an assistant asked for "apoio simples"
    # reaches for 'SIMPLE', so that is accepted rather than refused for a
    # vocabulary it never saw.
    def support_edge(self, object_id: str, edge, kind: str = "pin"):
        """Support an edge of an area object — the general edge-support call.

        edge: ``'all'``, a side name (bottom/right/top/left), or an index.
        kind: ``'pin'``/``'fix'``/``'roller-x'``/``'roller-y'``/``'symm-x'``/
        ``'symm-y'``, a canonical support name, a defined support, or
        ``'free'``."""
        # Three ways to name the restraint, in order: a support already defined
        # (or 'free') is used as-is; a canonical name (SIMPLE/CLAMPED/… for this
        # domain) is resolved to its restraints and created if need be; only the
        # pin/fix/roller/symm vocabulary goes through _edge_support_want.
        k = str(kind)
        if k == 'free' or k in self.supports:
            return self.support_object_edge(object_id, edge, k)
        canon = self._canonical_support_want(k)
        want = canon if canon is not None else self._edge_support_want(k)
        return self.support_object_edge(object_id, edge, self.support_for(*want))

    def _canonical_support_want(self, name: str):
        """The (ux/uy/tz)-slot restraints of a canonical support *name* in this
        domain (SIMPLE, CLAMPED, ROLLER-X, SYM-X, …), or None if *name* is not a
        canonical name here — so "apoio simples" → SIMPLE resolves like pin."""
        try:
            from .models import canonical_support_name
        except Exception:                                # noqa: BLE001
            return None
        from itertools import product
        dom = getattr(self, 'domain', 'plane')
        for combo in product((False, True), repeat=3):
            if any(combo) and canonical_support_name(*combo, domain=dom) == name:
                return combo
        return None

    def _edge_indices(self, obj, nids, edge) -> list:
        """Edge indices (0-based, perimeter order) selected by *edge* — the
        index counterpart of :meth:`_edge_segments` (``'all'`` → every edge; a
        named side bottom/right/top/left, rectangles only → its one index; an
        integer → that edge, wrapping)."""
        n = len(nids)
        if isinstance(edge, str) and edge.lower() == "all":
            return list(range(n))
        if isinstance(edge, str):
            named = {"bottom": 0, "right": 1, "top": 2, "left": 3}
            if n != 4:
                raise ValueError(
                    "named edges (bottom/right/top/left) need exactly 4 "
                    f"corners (this object has {n}) — use an edge index or "
                    "'all'.")
            e = edge.lower()
            if e not in named:
                raise ValueError(
                    "edge must be 'bottom'/'right'/'top'/'left', an index, or "
                    "'all'")
            return [named[e]]
        return [int(edge) % n]

    # Unlike support_edge (which restrains the two corner NODES and lets
    # edge_support_mode='common' propagate their shared DOFs along the meshed
    # edge), this writes the object's own per-edge edge_supports list, so every
    # node an edge generates is restrained by exactly that support — the only
    # way to express a mixed per-edge pattern (clamped/simply/clamped/simply
    # around a rectangle), which corner propagation cannot.
    def support_object_edge(self, object_id: str, edge, support_name: str):
        """Restrain one edge (or ``'all'``) of a surface object with a named
        support, written onto the object's per-edge ``edge_supports`` list.

        Args:
            object_id: the surface object (a GeoRectangle or GeoPolygon), or
                its id -- create_polygon's own return value works here too.
            edge: ``'all'`` | ``'bottom'/'right'/'top'/'left'`` (rectangles) |
                an integer edge index (vertex i -> i+1, wrapping) — the same
                vocabulary as support_edge.
            support_name: a support already defined (add_support, or a canonical
                name such as 'SIMPLE'/'CLAMPED' in a plate model), or ``'free'``
                to clear that edge's restraint.
        """
        from ._compat import as_object_id
        object_id = as_object_id(object_id)
        obj = self.geometry_objects.get(object_id)
        if obj is None:
            raise KeyError(f"Object '{object_id}' not found.")
        if not isinstance(obj, (GeoRectangle, GeoPolygon)):
            raise ValueError(
                f"Object '{object_id}' is not a surface (rectangle/polygon).")
        nids = [n for n in getattr(obj, "node_ids", []) if n in self.nodes]
        n = len(nids)
        if n < 3:
            raise ValueError(f"Object '{object_id}' has no edges yet.")
        if support_name != 'free' and support_name not in self.supports:
            raise ValueError(
                f"Unknown support '{support_name}'. Define it with add_support "
                "first, or use a canonical name (e.g. 'SIMPLE').")
        es = list(getattr(obj, "edge_supports", None) or [])
        if len(es) != n:
            es = ['free'] * n
        for i in self._edge_indices(obj, nids, edge):
            es[i] = support_name
        obj.edge_supports = es
        return obj

    # Per-edge support conveniences — the edge counterparts of pin/fix/roller/
    # symm, each just :meth:`support_edge` with a fixed ``kind``. support_edge
    # is the general case; these name the common ones and validate their own
    # free/axis argument for a clearer message than the generic kind check.
    def pin_edge(self, object_id: str, edge):
        """Simply support an edge of a surface object (the edge counterpart of
        :meth:`pin`). Plate: SIMPLE (w held, free to rotate). Plane: PIN
        (ux, uy). ``support_edge(object_id, edge, 'pin')``. ``edge``: same
        vocabulary as :meth:`support_edge`."""
        return self.support_edge(object_id, edge, "pin")

    def fix_edge(self, object_id: str, edge):
        """Fully restrain an edge — encastré (the edge counterpart of
        :meth:`fix`). Plate: CLAMPED (w, θx, θy). Plane: FIXED (ux, uy, θz).
        ``support_edge(object_id, edge, 'fix')``."""
        return self.support_edge(object_id, edge, "fix")

    def roller_edge(self, object_id: str, edge, free: str = "x"):
        """Roller along an edge (plane only; the edge counterpart of
        :meth:`roller`). ``free='x'`` restrains uy, ``free='y'`` restrains ux.
        Plate: use :meth:`symm_edge`."""
        f = free.lower()
        if f not in ("x", "y"):
            raise ValueError("free must be 'x' or 'y'")
        return self.support_edge(object_id, edge, f"roller-{f}")

    def symm_edge(self, object_id: str, edge, axis: str = "x"):
        """Symmetry line along an edge (plate only; the edge counterpart of
        :meth:`symm`): w free, a rotation restrained (``axis='x'``->θx,
        ``'y'``->θy). Plane: use :meth:`roller_edge`."""
        a = axis.lower()
        if a not in ("x", "y"):
            raise ValueError("axis must be 'x' or 'y'")
        return self.support_edge(object_id, edge, f"symm-{a}")

    # Removal facade — the intent-level inverses of the create_*/pin/support
    # family, the removal twin of the create_* layer. Named delete_*/free_edge
    # (never remove_*) so they read as "the facade for undoing an edit", and so
    # they stay visible to the assistant while the low-level remove_* stay
    # hidden from browsing. Each is a thin, forgiving wrapper: removing what is
    # not there is a no-op, so one call clears a node/bar/region without the
    # caller first classifying the target.
    def delete_support(self, nodes):
        """Remove the support from a node or nodes — the inverse of
        :meth:`pin`/:meth:`fix`/:meth:`roller`/:meth:`symm`/
        :meth:`create_support`. ``nodes``: a single id/``Node`` or a list. The
        support *definition* is left for reuse; an orphan is pruned by
        :meth:`prune_unused_supports`. For an object's edge use
        :meth:`free_edge`."""
        from ._compat import as_node_id_list
        ids, _ = as_node_id_list(nodes)
        for nid in ids:
            self.remove_support_assignment(nid)

    # Clears both routes an edge can be restrained by — the corner-node supports
    # support_edge assigns, and the object's own per-edge edge_supports list —
    # so it undoes support_edge and pin_edge/support_object_edge alike.
    def free_edge(self, object_id: str, edge):
        """Free an edge of an area object — the inverse of :meth:`support_edge`/
        :meth:`pin_edge`/:meth:`fix_edge`. ``edge``: same vocabulary as
        :meth:`support_edge` (``'all'``/``'bottom'``/``'right'``/``'top'``/
        ``'left'``/index)."""
        from ._compat import as_object_id
        object_id = as_object_id(object_id)
        obj = self.geometry_objects.get(object_id)
        if obj is None:
            raise KeyError(f"Object '{object_id}' not found.")
        nids = [n for n in getattr(obj, "node_ids", []) if n in self.nodes]
        if len(nids) >= 2:
            for n in self._edge_nodes(obj, nids, edge):
                self.remove_support_assignment(n)
            es = getattr(obj, "edge_supports", None)
            if es:
                for i in self._edge_indices(obj, nids, edge):
                    if 0 <= i < len(es):
                        es[i] = 'free'
                obj.edge_supports = es

    def delete_area(self, object_id: str):
        """Delete a region/line object and the nodes it owns that nothing else
        uses — the inverse of :meth:`create_polygon`/``add_geo_*``. The facade name
        for :meth:`remove_geo_object`, so removal reads in the same vocabulary
        as creation."""
        self.remove_geo_object(object_id)

    def delete_load(self, target: str, load_case_id: str = ""):
        """Remove loads on *target* — the inverse of the ``create_*_load``
        family. Clears node point loads, bar distributed loads and area loads on
        the target (whichever it has), for one case (id) or every case (``""``).
        One call covers a node, a bar or a region without saying which."""
        self.remove_point_loads(target, load_case_id)
        self.remove_distributed_loads(target, load_case_id)
        self.remove_area_load(target, load_case_id)

    def delete_temperature(self, target: str, load_case_id: str = ""):
        """Remove temperature loads on *target* — the inverse of
        :meth:`create_temperature` (and the ``add_*_temperature_load`` family).
        Covers a bar, triangle, quad, or an area/line object, for one case or
        every case (``""``)."""
        self.remove_temperature_load(target, load_case_id)
        self.remove_tri_temperature_load(target, load_case_id)
        self.remove_quad_temperature_load(target, load_case_id)
        self.remove_area_temperature_load(target, load_case_id)
        self.remove_line_temperature_load(target, load_case_id)

    def delete_spring(self, target: str):
        """Remove the spring on *target* — the inverse of the ``add_*_spring``
        family. Covers a node, an element, an area object or a line object."""
        self.remove_node_spring(target)
        self.remove_element_spring(target)
        self.remove_area_spring(target)
        self.remove_line_element_spring(target)

    # ------------------------------------------------------------------
    # Remove/modify methods
    # ------------------------------------------------------------------

    def remove_node(self, node_id: str):
        """Remove a node with everything that hangs on it: its supports,
        springs, loads, settlements, masses, constraints and the elements that
        use it (see references.SITES)."""
        if node_id not in self.nodes:
            raise KeyError(f"Node '{node_id}' not found.")
        self.nodes.pop(node_id)
        _refs.remove(self, 'node', node_id)
        self._node_dof_index = None

    def remove_element(self, element_id: str):
        if element_id not in self.bar_elements_by_id:
            raise KeyError(f"Element '{element_id}' not found.")
        self.bar_elements = [e for e in self.bar_elements if e.id != element_id]
        self.bar_elements_by_id.pop(element_id)
        _refs.remove(self, 'bar', element_id)

    # -- Beams (beam-bars design) ----------------------------------------

    def next_beam_tag(self) -> str:
        """A free ``V<n>`` tag."""
        used = set(self.beams) | self._used_beam_tags()
        n = 1
        while f"V{n}" in used:
            n += 1
        return f"V{n}"

    def _used_beam_tags(self) -> set:
        """Tags carried by bars or by line objects (segment/arc/multisegment)."""
        return ({e.beam for e in self.bar_elements if e.beam}
                | {o.beam for o in self.geometry_objects.values()
                   if getattr(o, 'beam', None)})

    def assign_beam(self, bar_ids=(), tag: Optional[str] = None,
                    name: Optional[str] = None, object_ids=()) -> str:
        """Tag *bar_ids* (and line objects *object_ids*) as one beam (a new
        free ``V<n>`` when *tag* is None) and register it. Returns the tag.
        Unknown ids raise ``KeyError``; the model is not touched in that
        case. Every bar an object generates inherits the object's tag."""
        bars = []
        for bid in bar_ids:
            if bid not in self.bar_elements_by_id:
                raise KeyError(f"Element '{bid}' not found.")
            bars.append(self.bar_elements_by_id[bid])
        objs = []
        for oid in object_ids:
            o = self.geometry_objects.get(oid)
            if o is None or not hasattr(o, 'beam'):
                raise KeyError(f"Line object '{oid}' not found.")
            objs.append(o)
        if not bars and not objs:
            raise ValueError("No bars to assign to a beam.")
        tag = tag or self.next_beam_tag()
        for b in bars + objs:
            b.beam = tag
        entry = self.beams.setdefault(tag, {"name": "", "overrides": {}})
        if name is not None:
            entry["name"] = name
        self.prune_beams()
        return tag

    def clear_beam(self, bar_ids=(), object_ids=()):
        """Remove the beam tag from *bar_ids* / line objects *object_ids*
        (they become auto-detected)."""
        for bid in bar_ids:
            b = self.bar_elements_by_id.get(bid)
            if b is not None:
                b.beam = None
        for oid in object_ids:
            o = self.geometry_objects.get(oid)
            if o is not None and hasattr(o, 'beam'):
                o.beam = None
        self.prune_beams()

    def prune_beams(self):
        """Drop registry entries with no bar left, unless they carry a name
        or overrides the user set (those survive an empty beam)."""
        used = self._used_beam_tags()
        for tag in list(self.beams):
            entry = self.beams[tag]
            if tag not in used and not entry.get("name") \
                    and not entry.get("overrides"):
                del self.beams[tag]

    def remove_material(self, name: str):
        """Remove a material. Refused (``ReferenceInUse``) while a section
        still uses it."""
        _refs.remove(self, 'material', name)
        self.materials.pop(name, None)

    def remove_section(self, name: str):
        """Remove a bar section. Refused (``ReferenceInUse``) while a bar or
        line object still uses it."""
        _refs.remove(self, 'section', name)
        self.sections.pop(name, None)

    def rename_material(self, old: str, new: str):
        """Rename a material, and every section / concrete link that names it."""
        if old == new:
            return
        if new in self.materials:
            raise ValueError(f"Material '{new}' already exists.")
        m = self.materials.get(old)
        if m is None:
            raise ValueError(f"Material '{old}' not found.")
        m.name = new
        self.materials[new] = self.materials.pop(old)
        _refs.rename(self, 'material', old, new)

    def rename_section(self, old: str, new: str):
        """Rename a bar section, and every bar / line object that uses it."""
        if old == new:
            return
        if new in self.sections:
            raise ValueError(f"Section '{new}' already exists.")
        sc = self.sections.get(old)
        if sc is None:
            raise ValueError(f"Section '{old}' not found.")
        sc.name = new
        self.sections[new] = self.sections.pop(old)
        _refs.rename(self, 'section', old, new)

    def remove_support(self, name: str):
        self.supports.pop(name, None)
        _refs.remove(self, 'support', name)

    def remove_node_spring(self, node_id: str):
        self.node_springs.pop(node_id, None)

    def remove_load_case(self, case_id: str):
        # Remove the twin analysis case (same id as the load case) along with the
        # load case, plus any other Linear analysis case that referenced only
        # this load case.
        twins = [ac.id for ac in self.analysis_cases
                 if ac.analysis_type == 'Linear' and ac.coefficients == {case_id: 1.0}]
        for ac_id in twins:
            self.remove_analysis_case(ac_id)
        self.load_cases = [lc for lc in self.load_cases if lc.id != case_id]
        self.load_cases_by_id.pop(case_id, None)
        # the loads of every kind on this case, and its analysis-case terms
        _refs.remove(self, 'load_case', case_id)

    def remove_load_combination(self, combo_id: str):
        self.load_combinations = [lc for lc in self.load_combinations if lc.id != combo_id]
        # Drop any references to the removed combination from other combinations.
        _refs.remove(self, 'combination', combo_id)

    def add_cut(self, id: str, x1: float, y1: float, x2: float, y2: float,
                name: str = "") -> Cut:
        """Create a user-defined section cut, a segment ``(x1,y1)->(x2,y2)``
        used to report resultants of internal forces crossing it (see
        dev/CUT_PLAN.md). Cuts carry no references to other entities."""
        if any(c.id == id for c in self.cuts):
            raise ValueError(f"Cut '{id}' already exists.")
        cut = Cut(id=id, name=name, x1=x1, y1=y1, x2=x2, y2=y2)
        self.cuts.append(cut)
        return cut

    def remove_cut(self, cut_id: str):
        self.cuts = [c for c in self.cuts if c.id != cut_id]

    # ------------------------------------------------------------------
    # Rename methods (update the entity id and every reference to it)
    # ------------------------------------------------------------------

    def rename_node(self, old_id: str, new_id: str):
        if old_id == new_id:
            return
        if new_id in self.nodes:
            raise ValueError(f"Node '{new_id}' already exists.")
        node = self.nodes.get(old_id)
        if node is None:
            raise ValueError(f"Node '{old_id}' not found.")
        node.id = new_id
        self.nodes[new_id] = self.nodes.pop(old_id)
        _refs.rename(self, 'node', old_id, new_id)
        self._node_dof_index = None

    def rename_bar_element(self, old_id: str, new_id: str):
        if old_id == new_id:
            return
        if new_id in self.bar_elements_by_id:
            raise ValueError(f"Element '{new_id}' already exists.")
        elem = self.bar_elements_by_id.get(old_id)
        if elem is None:
            raise ValueError(f"Element '{old_id}' not found.")
        elem.id = new_id
        self.bar_elements_by_id[new_id] = self.bar_elements_by_id.pop(old_id)
        _refs.rename(self, 'bar', old_id, new_id)

    def rename_tri_element(self, old_id: str, new_id: str):
        """Rename a CST triangle, carrying its temperature/edge loads along —
        the triangle counterpart of :meth:`rename_bar_element`."""
        if old_id == new_id:
            return
        if new_id in self.tri_elements_by_id:
            raise ValueError(f"Triangle '{new_id}' already exists.")
        tri = self.tri_elements_by_id.get(old_id)
        if tri is None:
            raise ValueError(f"Triangle '{old_id}' not found.")
        tri.id = new_id
        self.tri_elements_by_id[new_id] = self.tri_elements_by_id.pop(old_id)
        _refs.rename(self, 'tri', old_id, new_id)

    # No edge-load loop here (unlike rename_tri_element): quads have no
    # edge-load feature at all yet (context_menu.QUAD_ENTRIES's "Add edge
    # load" note, Phase 8 item 1) — nothing to carry. Like
    # rename_tri_element, this does not carry area loads/springs targeting
    # the quad id (quad_area_loads/quad_area_springs) — that parity gap
    # already exists for triangles too, not introduced here.
    def rename_quad_element(self, old_id: str, new_id: str):
        """Rename a quad, carrying its temperature loads along — the quad
        counterpart of :meth:`rename_tri_element`."""
        if old_id == new_id:
            return
        if new_id in self.quad_elements_by_id:
            raise ValueError(f"Quad '{new_id}' already exists.")
        quad = self.quad_elements_by_id.get(old_id)
        if quad is None:
            raise ValueError(f"Quad '{old_id}' not found.")
        quad.id = new_id
        self.quad_elements_by_id[new_id] = self.quad_elements_by_id.pop(old_id)
        _refs.rename(self, 'quad', old_id, new_id)

    def rename_load_case(self, old_id: str, new_id: str):
        if old_id == new_id:
            return
        if new_id in self.load_cases_by_id:
            raise ValueError(f"Load case '{new_id}' already exists.")
        lc = self.load_cases_by_id.get(old_id)
        if lc is None:
            raise ValueError(f"Load case '{old_id}' not found.")
        lc.id = new_id
        self.load_cases_by_id.pop(old_id, None)
        self.load_cases_by_id[new_id] = lc
        # Everything that names the load case: the loads of every kind and the
        # analysis-case coefficients (see references.SITES).
        _refs.rename(self, 'load_case', old_id, new_id)
        # Keep the twin analysis case name in step with the load case
        # (<old> -> <new>), which also updates combinations that use it.
        if f"{old_id}" in self.analysis_cases_by_id and \
                f"{new_id}" not in self.analysis_cases_by_id:
            self.rename_analysis_case(f"{old_id}", f"{new_id}")

    def rename_analysis_case(self, old_id: str, new_id: str):
        if old_id == new_id:
            return
        if new_id in self.analysis_cases_by_id:
            raise ValueError(f"Analysis case '{new_id}' already exists.")
        ac = self.analysis_cases_by_id.get(old_id)
        if ac is None:
            raise ValueError(f"Analysis case '{old_id}' not found.")
        ac.id = new_id
        self.analysis_cases_by_id.pop(old_id, None)
        self.analysis_cases_by_id[new_id] = ac
        # Combinations, other cases (coefficients, modal / stored stiffness)
        # and nodal masses name an analysis case.
        _refs.rename(self, 'analysis_case', old_id, new_id)

    def rename_load_combination(self, old_id: str, new_id: str):
        if old_id == new_id:
            return
        if any(c.id == new_id for c in self.load_combinations):
            raise ValueError(f"Combination '{new_id}' already exists.")
        combo = next((c for c in self.load_combinations if c.id == old_id), None)
        if combo is None:
            raise ValueError(f"Combination '{old_id}' not found.")
        combo.id = new_id
        # Other combinations may reference this one by id.
        _refs.rename(self, 'combination', old_id, new_id)

    def rename_cut(self, old_id: str, new_id: str):
        if old_id == new_id:
            return
        if any(c.id == new_id for c in self.cuts):
            raise ValueError(f"Cut '{new_id}' already exists.")
        cut = next((c for c in self.cuts if c.id == old_id), None)
        if cut is None:
            raise ValueError(f"Cut '{old_id}' not found.")
        cut.id = new_id

    def remove_point_loads(self, node_id: str, load_case_id: str = ""):
        """Remove point loads on *node_id* (all cases, or one). ``""`` = all,
        matching remove_area_load's convention."""
        self.point_loads = [
            pl for pl in self.point_loads
            if not (pl.node_id == node_id
                    and (not load_case_id or pl.load_case_id == load_case_id))]

    def remove_distributed_loads(self, element_id: str, load_case_id: str = ""):
        """Remove distributed loads on *element_id* (all cases, or one)."""
        self.distributed_loads = [
            dl for dl in self.distributed_loads
            if not (dl.element_id == element_id
                    and (not load_case_id or dl.load_case_id == load_case_id))]

    # ------------------------------------------------------------------
    # Variants (Phase 1 — parallel scenarios over a shared entity space)
    # ------------------------------------------------------------------

    def add_support_set(self, support_set) -> object:
        self.support_sets[support_set.id] = support_set
        return support_set

    def remove_support_set(self, set_id: str):
        self.support_sets.pop(set_id, None)
        _refs.remove(self, 'support_set', set_id)   # variants and phases using it

    def purge_dangling_references(self) -> list[str]:
        """Remove the loads, springs and other entries that refer to something
        that no longer exists (see :func:`xdfem2d.references.purge`). Returns a
        readable summary, empty when there was nothing to remove."""
        return _refs.purge(self)

    def add_variant(self, variant) -> object:
        self.variants[variant.id] = variant
        return variant

    def remove_variant(self, variant_id: str):
        self.variants.pop(variant_id, None)

    def add_construction_sequence(self, sequence) -> object:
        self.construction_sequences[sequence.id] = sequence
        # Keep a matching AnalysisCase(type='Sequence') in sync.
        if sequence.id not in self.analysis_cases_by_id:
            ac = AnalysisCase(id=sequence.id, analysis_type='Sequence',
                              sequence_id=sequence.id)
            self.analysis_cases.append(ac)
            self.analysis_cases_by_id[sequence.id] = ac
        return sequence

    def remove_construction_sequence(self, seq_id: str):
        self.construction_sequences.pop(seq_id, None)
        # Remove the matching AnalysisCase (if it is a Sequence type).
        ac = self.analysis_cases_by_id.get(seq_id)
        if ac is not None and getattr(ac, 'analysis_type', '') == 'Sequence':
            self.analysis_cases = [a for a in self.analysis_cases if a.id != seq_id]
            self.analysis_cases_by_id.pop(seq_id, None)

    def derive_variant(self, variant, support_sets: Optional[dict] = None,
                       filter_tri: bool = False) -> 'Structure2D':
        """Return a NEW Structure2D representing *variant* — a pure transform.

        Does not mutate ``self``. Steps:
          1. deep-copy this structure (same ids preserved);
          2. if ``variant.active_elements`` is not None, remove inactive bars
             (and, when ``filter_tri``, triangles and geometry objects too)
             and any node left orphan (a node with no active bar/triangle/
             geometry-object corner and no support);
          3. if ``variant.support_set_id`` is set, replace the support
             assignments and springs by those of the referenced SupportSet.

        ``filter_tri``: the *variants* feature has never offered a way to pick
        individual triangles (its UI only lists bars), so every existing
        variant's ``active_elements`` is bar-ids-only — filtering triangles by
        the same set would silently delete every triangle in a hybrid model.
        Construction phasing, on the other hand, means ``active_elements`` as
        "every element built so far, of any kind" — pass ``filter_tri=True``
        there (see ``phasing.py``) so a plate/wall region can be staged in.

        Materials, sections, load cases, loads and analysis cases are kept. The
        caller runs ``.calculate()`` on the result.
        """
        import copy
        # A variant with its own model is fully authoritative: its model already
        # reflects the active geometry, supports and its loads/cases/combos. Use
        # it as-is (no further filtering).
        if getattr(variant, 'model', None) is not None:
            sub = copy.deepcopy(variant.model)
            sub._node_dof_index = None
            return sub

        sub = copy.deepcopy(self)

        # Fall back to the structure's own registered support sets.
        if support_sets is None:
            support_sets = self.support_sets

        active = getattr(variant, 'active_elements', None)
        sset = None
        if getattr(variant, 'support_set_id', None) is not None:
            sset = (support_sets or {}).get(variant.support_set_id)
            if sset is None:
                raise ValueError(
                    f"SupportSet '{variant.support_set_id}' not found.")

        # 1) Apply the support set first so orphan detection sees the right
        #    supported nodes.
        if sset is not None:
            restraints = getattr(sset, 'restraints', None)
            if restraints:
                # Per-node restraints (the GUI model). Rebuild supports/
                # assignments from scratch, via the same canonical/shared
                # definitions (PIN/FIXED/ROLLER-X/... or the plate domain's
                # SIMPLE/CLAMPED/...) the canvas "Add support" flow uses,
                # instead of one bespoke Support named after the node itself
                # — see workflows.apply_variant_config for the twin of this
                # block (used when a variant already has a cached model).
                from .models import CANONICAL_SUPPORTS, CANONICAL_SUPPORTS_PLATE
                sub.support_assignments = []
                for nid, r in restraints.items():
                    ux, uy, tz = (bool(r[0]), bool(r[1]), bool(r[2]))
                    name = sub.support_for(ux, uy, tz)
                    if name is not None:
                        sub.assign_support(nid, name)
                canonical = set(CANONICAL_SUPPORTS.values()) | \
                    set(CANONICAL_SUPPORTS_PLATE.values())
                ours = {n for n in sub.supports
                        if n in canonical
                        or (n.rsplit('-', 1)[0] in canonical
                            and n.rsplit('-', 1)[-1].isdigit())}
                sub.prune_unused_supports(only=ours)
            else:
                # Named-support assignments (programmatic / backward compatible).
                sub.support_assignments = [copy.deepcopy(a)
                                           for a in sset.assignments]
            sub.node_springs = {k: copy.deepcopy(v)
                                for k, v in sset.node_springs.items()}
            sub.element_springs = {k: copy.deepcopy(v)
                                   for k, v in sset.element_springs.items()}

        # 2) Filter elements / drop orphan nodes. An EMPTY active set is
        #    treated as "all" (same coercion as make_variant_model): a variant
        #    with zero bars cannot be viewed or solved, so an empty set is
        #    always a data accident, never an intent.
        if active:
            active = set(active)
            for eid in [e.id for e in sub.bar_elements]:
                if eid not in active:
                    sub.remove_element(eid)
            if filter_tri:
                # Already-meshed (explicitly authored) triangles are filtered
                # individually. A GeoRectangle/GeoPolygon/GeoArc/etc. has no
                # per-triangle ids of its own before expansion, so it is
                # staged in or out as a WHOLE unit: its own id must appear in
                # ``active`` (like a bar's or triangle's id would).
                for tid in [t.id for t in sub.tri_elements]:
                    if tid not in active:
                        sub.remove_tri_element(tid)
                # Quads are staged individually too (dev/refactor_area_path.md
                # Phase 2) — the 4-node twin of the triangle filter above, so a
                # quad-meshed region actually appears/disappears with its stage
                # instead of being present in every phase.
                for qid in [q.id for q in getattr(sub, 'quad_elements', [])]:
                    if qid not in active:
                        sub.remove_quad_element(qid)
                for oid in [oid for oid in sub.geometry_objects]:
                    if oid not in active:
                        del sub.geometry_objects[oid]
            supported = {a.node_id for a in sub.support_assignments}
            used = set()
            for e in sub.bar_elements:
                used.add(e.node_i); used.add(e.node_j)
            for e in sub.tri_elements:
                used.add(e.node_i); used.add(e.node_j); used.add(e.node_k)
            # Quad nodes must survive the orphan prune too, or a node used only
            # by a hand-added quad is silently removed (dev/refactor_area_path.md
            # Phase 2).
            for e in getattr(sub, 'quad_elements', []):
                used.add(e.node_i); used.add(e.node_j)
                used.add(e.node_k); used.add(e.node_l)
            # A plate/wall region is often still a GeoRectangle/GeoPolygon
            # (meshed into tri_elements only at solve time) rather than actual
            # elements yet — its corner/vertex nodes must survive the prune too,
            # or the object silently loses corners and fails to mesh later.
            for obj in sub.geometry_objects.values():
                used.update(getattr(obj, 'node_ids', []) or [])
            for nid in [n for n in sub.nodes]:
                if nid not in used and nid not in supported:
                    sub.remove_node(nid)

        sub._node_dof_index = None
        return sub

    def with_supports(self, support_set) -> 'Structure2D':
        """Sugar: derive a variant that only swaps the supports."""
        from .models import Variant
        v = Variant(id=getattr(support_set, 'id', 'variant'),
                    support_set_id=getattr(support_set, 'id', None))
        return self.derive_variant(v, {v.support_set_id: support_set})

    def with_active(self, active_elements) -> 'Structure2D':
        """Sugar: derive a variant that only restricts the active geometry."""
        from .models import Variant
        v = Variant(id='variant', active_elements=set(active_elements))
        return self.derive_variant(v)

    def stiffness_hash(self) -> str:
        """Deterministic hash of the *effective stiffness* of this structure.

        Includes everything that defines K: active node coordinates, bar
        connectivity, section/material properties, end releases (hinges),
        supports and springs. EXCLUDES loads and analysis/load cases.

        Two structures with the same hash have the same K, so a linear sum of
        their results is exact. Used to guard combinations across variants.
        """
        import hashlib
        parts: list[str] = []

        for nid in sorted(self.nodes):
            n = self.nodes[nid]
            parts.append(f"N|{nid}|{n.x!r}|{n.y!r}")

        for e in sorted(self.bar_elements, key=lambda e: e.id):
            sec = self.sections.get(e.section_name)
            mat = self.materials.get(sec.material_name) if sec else None
            E = getattr(mat, 'elastic_modulus', None)
            A = getattr(sec, 'area_override', None)
            I = getattr(sec, 'inertia_override', None)
            b = getattr(sec, 'b', None); h = getattr(sec, 'h', None)
            parts.append(
                f"E|{e.id}|{e.node_i}|{e.node_j}|{e.section_name}|"
                f"{E!r}|{A!r}|{I!r}|{b!r}|{h!r}|"
                f"hi={int(bool(e.hinge_i))}|hj={int(bool(e.hinge_j))}")

        for a in sorted(self.support_assignments,
                        key=lambda a: (a.node_id, a.support_name)):
            sp = self.supports.get(a.support_name)
            parts.append(
                f"S|{a.node_id}|{a.support_name}|"
                f"{int(bool(getattr(sp,'ux',False)))}"
                f"{int(bool(getattr(sp,'uy',False)))}"
                f"{int(bool(getattr(sp,'tz',False)))}")

        for nid in sorted(self.node_springs):
            s = self.node_springs[nid]
            parts.append(
                f"NS|{nid}|{s.kx!r}|{s.ky!r}|{s.kt!r}|{s.mode_x}|{s.mode_y}")

        for eid in sorted(self.element_springs):
            s = self.element_springs[eid]
            parts.append(
                f"ES|{eid}|{s.kx!r}|{s.ky!r}|"
                f"{getattr(s,'coord_sys','global')}|{s.mode_x}|{s.mode_y}")

        blob = "\n".join(parts).encode('utf-8')
        return hashlib.sha256(blob).hexdigest()

    # ------------------------------------------------------------------
    # DOF numbering
    # ------------------------------------------------------------------

    @property
    def node_dof_index(self) -> dict[str, int]:
        if self._node_dof_index is None:
            self._node_dof_index = {nid: i*3 for i, nid in enumerate(self.nodes)}
            self._num_dofs = len(self.nodes) * 3
        return self._node_dof_index

    @property
    def num_dofs(self) -> int:
        _ = self.node_dof_index
        return self._num_dofs

    @property
    def dof_labels(self) -> tuple[str, str, str]:
        """Names of the three nodal DOF components in storage order — ('ux',
        'uy', 'tz') in the plane domain, ('w', 'tx', 'ty') in the plate domain.
        The one place results, reports and the GUI should take labels from."""
        from .models import DOF_LABELS
        return DOF_LABELS[getattr(self, 'domain', 'plane')]

    # ------------------------------------------------------------------
    # Analysis
    # ------------------------------------------------------------------

    def calculate(self) -> dict:
        """Run the full analysis and return a results dictionary.

        Objects (parametric geometry) are discretised into concrete nodes
        and bar elements first (see :mod:`xdfem2d.geo_expand`); the solver then
        runs on that compiled model. The editable model is left untouched. When
        the model has no objects, the compiled model is the model itself (no copy)
        so the classic path is unchanged.
        """
        import time
        from .assembly import assemble_stiffness
        from .loads    import assemble_loads
        from .solver   import solve, compute_results, _build_fixed_dofs

        if not self.load_cases:
            raise ValueError("No load cases defined.")

        t0 = time.perf_counter()

        # Compile objects → concrete geometry (only when present).
        object_trace: dict = {}
        expansion_version = 0
        if getattr(self, "geometry_objects", None):
            from .geo_expand import expand_geometry, EXPANSION_VERSION
            compiled, object_trace = expand_geometry(self)
            expansion_version = EXPANSION_VERSION
        else:
            compiled = self

        if (not compiled.bar_elements and not compiled.tri_elements
                and not compiled.quad_elements):
            raise ValueError("No elements defined.")

        compiled.check_references()
        t1 = time.perf_counter()

        K = assemble_stiffness(compiled)
        t2 = time.perf_counter()
        F = assemble_loads(compiled)
        t3 = time.perf_counter()
        U = solve(compiled, K, F)
        t4 = time.perf_counter()
        results = compute_results(compiled, K, F, U)
        t5 = time.perf_counter()
        # Carry the obj trace so results on the compiled mesh can be mapped
        # back to each obj (and to detect a stale cached mesh — Option B).
        if object_trace:
            results['object_trace'] = object_trace
            results['expansion_version'] = expansion_version
        # Fingerprint of the effective stiffness at solve time (see
        # stiffness_hash()). Design tools compare this against the current
        # model's live hash to warn when the structure has changed since
        # these results were computed -- geometry/sections/supports, not
        # loads, so a load-only edit never triggers it.
        results['stiffness_hash'] = self.stiffness_hash()
        # System size and per-phase timing — read by the GUI's post-Run
        # statistics box (nodes/elements, DOF count, analysis cases, wall
        # time), computed here once rather than re-measured by the caller.
        fixed = _build_fixed_dofs(compiled)
        results['system_size'] = {
            'nodes':        len(compiled.nodes),
            'bar_elements': len(compiled.bar_elements),
            'tri_elements': len(getattr(compiled, 'tri_elements', [])),
            'dofs_total':   int(compiled.num_dofs),
            'dofs_free':    int((~fixed).sum()),
            'dofs_fixed':   int(fixed.sum()),
        }
        results['timing'] = {
            'expand_geometry':    t1 - t0,
            'assemble_stiffness': t2 - t1,
            'assemble_loads':     t3 - t2,
            'solve':              t4 - t3,
            'post_process':       t5 - t4,
            'total':              t5 - t0,
        }
        # A full run leaves every analysis case with current results. The flag
        # is on the editable cases (compiled is a throwaway copy when objects
        # are present), so it is set here on self.
        for ac in self.analysis_cases:
            ac.solved = True
        return results

    # 'solve' is the generic verb every other FEM/scipy-flavoured library
    # uses (and the name the internal solver function itself has, in
    # .solver — never exposed as a method, but close enough in the source
    # to reinforce the guess); a script written from memory of another
    # library reaches for it before 'calculate'. Pure alias, not a
    # different call.
    solve = calculate

    # ------------------------------------------------------------------
    # Persistence shortcuts
    # ------------------------------------------------------------------

    def save(self, path: str):
        """Save the model as JSON. NOT the .x2d format.

        .x2d is a zip holding the model, the results and the view, written
        with xdfem2d.file_io.save_x2d. A .x2d path is refused here rather than
        written as JSON under a name the loader will reject.
        """
        # Refused because three of six language models finished with
        # `struc.save('model.x2d')` — the method is called save, it takes any
        # path, and nothing said otherwise. What came out was rejected by
        # load_x2d with "File is not a zip file", which points at the reader
        # rather than at the line that wrote it. A person guesses the same way.
        from .structure_io import save_structure_json
        if str(path).lower().endswith('.x2d'):
            raise ValueError(
                f"save() writes JSON, not .x2d, and would leave "
                f"'{path}' unreadable by load_x2d. For .x2d (model + results "
                f"+ view):\n"
                f"    from xdfem2d.file_io import save_x2d\n"
                f"    save_x2d(struc, results, {path!r})\n"
                f"For JSON, give it a .json name.")
        save_structure_json(self, path)

    @staticmethod
    def load(path: str) -> 'Structure2D':
        """Load structure from JSON (.json)."""
        from .structure_io import load_structure_json
        return load_structure_json(path)
