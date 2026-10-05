"""IFC (openBIM) export for xdfem2D — Phase A: structural-analysis skeleton.

Writes the model as an ``IfcStructuralAnalysisModel`` (IFC4 structural
analysis view): nodes → ``IfcStructuralPointConnection``, bars →
``IfcStructuralCurveMember``, triangles/quads → ``IfcStructuralSurfaceMember``,
supports → ``IfcBoundaryNodeCondition``. Geometry + topology only; materials,
sections and loads are Phase B (see ``dev/IMPORT_IFC.md``).

``ifcopenshell`` is an optional dependency, imported lazily like ``ezdxf`` in
``dxf_io``: if it is missing, an actionable ``ImportError`` is raised.

Axis convention mirrors the SAP2000 exporter (``dev/sap2000_domain_io.md``):
  * ``plane`` model → global **X–Z** plane, node (x, y) → (x, 0, y);
  * ``plate`` model → global **X–Y** plane, node (x, y) → (x, y, 0).
"""
from __future__ import annotations

from pathlib import Path


def _require_ifcopenshell():
    """Return the ifcopenshell module or raise an actionable ImportError."""
    try:
        import ifcopenshell  # noqa: F401
        return ifcopenshell
    except ImportError as exc:                          # pragma: no cover
        raise ImportError(
            "IFC support requires the 'ifcopenshell' package. "
            "Install it with:  pip install ifcopenshell"
        ) from exc


def _xyz(node, plate: bool) -> tuple[float, float, float]:
    """3D coordinate of *node* for the given domain (plate ⇒ X–Y, else X–Z)."""
    if plate:
        return (float(node.x), float(node.y), 0.0)
    return (float(node.x), 0.0, float(node.y))


def to_ifc(struc):
    """Build and return an ``ifcopenshell.file`` for *struc* (Structure2D)."""
    ios = _require_ifcopenshell()
    guid = ios.guid.new

    # Expand geometry objects to a plain mesh, as the SAP exporter does.
    if getattr(struc, "geometry_objects", None):
        from .geo_expand import expand_geometry
        struc = expand_geometry(struc)[0]

    plate = getattr(struc, "domain", "plane") == "plate"
    f = ios.file(schema="IFC4")

    # ── Units: kN · m · t · °C (SAP2000 convention) ───────────────────
    # xdfem2D works internally in kN, m, t, °C, so values are written natively
    # (no scaling) and every measure gets a declared unit — matching how
    # SAP2000 itself exports IFC (force = kN, mass = tonne, temperature = °C).
    # Without the derived units (modulus of elasticity, mass density, …) an
    # importer reports "units are not defined" and can fail while resolving
    # them, so they are all declared here.
    metre = f.create_entity("IfcSIUnit", UnitType="LENGTHUNIT", Name="METRE")
    kilonewton = f.create_entity(
        "IfcSIUnit", UnitType="FORCEUNIT", Name="NEWTON", Prefix="KILO")
    tonne = f.create_entity(
        "IfcSIUnit", UnitType="MASSUNIT", Name="GRAM", Prefix="MEGA")
    celsius = f.create_entity(
        "IfcSIUnit", UnitType="THERMODYNAMICTEMPERATUREUNIT",
        Name="DEGREE_CELSIUS")

    def _du(unit_type, elems):
        return f.create_entity(
            "IfcDerivedUnit", UnitType=unit_type, Elements=[
                f.create_entity("IfcDerivedUnitElement", Unit=u, Exponent=e)
                for u, e in elems])

    unit_modulus = _du("MODULUSOFELASTICITYUNIT", [(kilonewton, 1), (metre, -2)])
    unit_density = _du("MASSDENSITYUNIT", [(tonne, 1), (metre, -3)])
    unit_thermal = _du("THERMALEXPANSIONCOEFFICIENTUNIT", [(celsius, -1)])
    units = f.create_entity("IfcUnitAssignment", Units=[
        metre, kilonewton, tonne, celsius,
        unit_modulus, unit_density, unit_thermal,
        _du("LINEARFORCEUNIT", [(kilonewton, 1), (metre, -1)]),
        _du("PLANARFORCEUNIT", [(kilonewton, 1), (metre, -2)]),
        _du("TORQUEUNIT", [(kilonewton, 1), (metre, 1)]),
    ])

    # ── Geometric context, project ────────────────────────────────────
    origin = f.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0))
    axis_z = f.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
    axis_x = f.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
    world = f.create_entity(
        "IfcAxis2Placement3D", Location=origin, Axis=axis_z, RefDirection=axis_x)
    ctx = f.create_entity(
        "IfcGeometricRepresentationContext", ContextType="Model",
        CoordinateSpaceDimension=3, Precision=1e-6,
        WorldCoordinateSystem=world)
    project = f.create_entity(
        "IfcProject", GlobalId=guid(), Name="xdfem2D export",
        UnitsInContext=units, RepresentationContexts=[ctx])

    def _topology(items):
        return f.create_entity(
            "IfcProductDefinitionShape", Representations=[
                f.create_entity(
                    "IfcTopologyRepresentation", ContextOfItems=ctx,
                    RepresentationIdentifier="Reference",
                    RepresentationType="Vertex" if len(items) == 1 else "Edge",
                    Items=items)])

    # ── Nodes → point connections (shared vertices) ───────────────────
    vertex_of: dict[str, object] = {}
    conn_of: dict[str, object] = {}
    members: list = []            # every structural item, for the group
    connections: list = []
    member_by_id: dict[str, object] = {}   # element id → structural member
    # Section name → list of member entities, split by kind, so a single
    # material/profile association can cover every member that shares a section.
    bar_members_by_section: dict[str, list] = {}
    area_members_by_section: dict[str, list] = {}

    # Precompute which nodes carry a support, mapped by node id.
    supp_by_node = {}
    for a in getattr(struc, "support_assignments", []) or []:
        sup = struc.supports.get(a.support_name)
        if sup is not None:
            supp_by_node[a.node_id] = sup

    # Only nodes referenced by a member become point connections: IFC requires
    # every IfcStructuralPointConnection to connect to at least one member
    # (ConnectsStructuralMembers SET [1:?]). An orphan node is both schema-
    # invalid and a known trigger for importers that iterate a node's (empty)
    # member list, so isolated nodes — and any support on them — are dropped.
    used_ids = set()
    for e in getattr(struc, "bar_elements", []) or []:
        used_ids.update((e.node_i, e.node_j))
    for t in getattr(struc, "tri_elements", []) or []:
        used_ids.update((t.node_i, t.node_j, t.node_k))
    for q in getattr(struc, "quad_elements", []) or []:
        used_ids.update((q.node_i, q.node_j, q.node_k, q.node_l))

    for n in struc.nodes.values():
        if n.id not in used_ids:
            continue
        pt = f.create_entity("IfcCartesianPoint", Coordinates=_xyz(n, plate))
        vertex = f.create_entity("IfcVertexPoint", VertexGeometry=pt)
        vertex_of[n.id] = vertex
        cond = _boundary_condition(f, supp_by_node.get(n.id), plate)
        conn = f.create_entity(
            "IfcStructuralPointConnection", GlobalId=guid(), Name=str(n.id),
            Representation=_topology([vertex]), AppliedCondition=cond)
        conn_of[n.id] = conn
        connections.append(conn)

    rel_seq = [0]

    def _connect(member, node_ids):
        """Relate *member* to each of its point connections."""
        for nid in node_ids:
            conn = conn_of.get(nid)
            if conn is None:
                continue
            rel_seq[0] += 1
            f.create_entity(
                "IfcRelConnectsStructuralMember", GlobalId=guid(),
                RelatingStructuralMember=member, RelatedStructuralConnection=conn)

    # ── Bars → curve members ──────────────────────────────────────────
    default_axis = f.create_entity(
        "IfcDirection", DirectionRatios=(0.0, 1.0, 0.0) if not plate
        else (0.0, 0.0, 1.0))
    for e in getattr(struc, "bar_elements", []) or []:
        vi, vj = vertex_of.get(e.node_i), vertex_of.get(e.node_j)
        if vi is None or vj is None:
            continue
        edge = f.create_entity("IfcEdge", EdgeStart=vi, EdgeEnd=vj)
        member = f.create_entity(
            "IfcStructuralCurveMember", GlobalId=guid(), Name=str(e.id),
            Representation=_topology([edge]),
            PredefinedType="RIGID_JOINED_MEMBER", Axis=default_axis)
        members.append(member)
        member_by_id[e.id] = member
        bar_members_by_section.setdefault(e.section_name, []).append(member)
        _connect(member, (e.node_i, e.node_j))

    # ── Triangles / quads → surface members ───────────────────────────
    surf_type = "BENDING_ELEMENT" if plate else "MEMBRANE_ELEMENT"
    tsecs = getattr(struc, "tri_sections", {}) or {}
    qsecs = getattr(struc, "quad_sections", {}) or {}

    def _thickness(section_name):
        sec = tsecs.get(section_name) or qsecs.get(section_name)
        t = getattr(sec, "thickness", None)
        try:
            t = float(t)
            return t if t > 0 else None
        except (TypeError, ValueError):
            return None

    def _emit_surface(aid, node_ids, section_name):
        loop_pts = []
        for nid in node_ids:
            v = vertex_of.get(nid)
            if v is None:
                return
            loop_pts.append(v.VertexGeometry)
        loop = f.create_entity("IfcPolyLoop", Polygon=loop_pts)
        bound = f.create_entity("IfcFaceOuterBound", Bound=loop, Orientation=True)
        plane = f.create_entity(
            "IfcPlane", Position=f.create_entity(
                "IfcAxis2Placement3D",
                Location=loop_pts[0], Axis=axis_z, RefDirection=axis_x))
        face = f.create_entity(
            "IfcFaceSurface", Bounds=[bound], FaceSurface=plane, SameSense=True)
        rep = f.create_entity(
            "IfcProductDefinitionShape", Representations=[
                f.create_entity(
                    "IfcTopologyRepresentation", ContextOfItems=ctx,
                    RepresentationIdentifier="Reference",
                    RepresentationType="Face", Items=[face])])
        thick = _thickness(section_name)
        member = f.create_entity(
            "IfcStructuralSurfaceMember", GlobalId=guid(), Name=str(aid),
            Representation=rep, PredefinedType=surf_type, Thickness=thick)
        members.append(member)
        member_by_id[aid] = member
        area_members_by_section.setdefault(section_name, []).append(member)
        _connect(member, node_ids)

    for t in getattr(struc, "tri_elements", []) or []:
        _emit_surface(t.id, (t.node_i, t.node_j, t.node_k), t.section_name)
    for q in getattr(struc, "quad_elements", []) or []:
        _emit_surface(q.id, (q.node_i, q.node_j, q.node_k, q.node_l),
                      q.section_name)

    # ── Phase B: materials + profiles ─────────────────────────────────
    mat_cache: dict[str, object] = {}

    def _ifc_material(name):
        """IfcMaterial for *name* with a Pset_MaterialMechanical, cached."""
        if name in mat_cache:
            return mat_cache[name]
        m = struc.materials.get(name)
        mat = f.create_entity("IfcMaterial", Name=str(name))
        mat_cache[name] = mat
        if m is not None:
            props = []

            def _sv(pname, val, typ, unit=None):
                if val is None:
                    return
                props.append(f.create_entity(
                    "IfcPropertySingleValue", Name=pname,
                    NominalValue=f.create_entity(typ, float(val)), Unit=unit))

            rho = getattr(m, "unit_mass", None)          # t/m³
            if rho is None and getattr(m, "unit_weight", None):
                rho = m.unit_weight / 9.81
            # Values are native xdfem2D units (kN/m², t/m³, 1/°C).
            _sv("YoungModulus", getattr(m, "elastic_modulus", None),
                "IfcModulusOfElasticityMeasure", unit_modulus)
            _sv("PoissonRatio", getattr(m, "poisson", None), "IfcRatioMeasure")
            _sv("ThermalExpansionCoefficient", getattr(m, "alpha", None),
                "IfcThermalExpansionCoefficientMeasure", unit_thermal)
            _sv("MassDensity", rho, "IfcMassDensityMeasure", unit_density)
            if props:
                f.create_entity(
                    "IfcMaterialProperties", Name="Pset_MaterialMechanical",
                    Material=mat, Properties=props)
        return mat

    def _profile(section_name):
        """IfcProfileDef for a bar section, mapped from its shape."""
        s = struc.sections.get(section_name)
        if s is None:
            return None
        from .models import SectionShape
        shape = getattr(s, "shape", None)
        b, h = float(getattr(s, "b", 0) or 0), float(getattr(s, "h", 0) or 0)
        pos = f.create_entity(
            "IfcAxis2Placement2D",
            Location=f.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0)))
        if shape == SectionShape.CIRCULAR and h > 0:
            return f.create_entity(
                "IfcCircleProfileDef", ProfileType="AREA",
                ProfileName=str(section_name), Position=pos, Radius=h / 2.0)
        if shape == SectionShape.I and b > 0 and h > 0:
            return f.create_entity(
                "IfcIShapeProfileDef", ProfileType="AREA",
                ProfileName=str(section_name), Position=pos,
                OverallWidth=b, OverallDepth=h,
                WebThickness=float(getattr(s, "tw", 0) or 0.01) or 0.01,
                FlangeThickness=float(getattr(s, "tf", 0) or 0.01) or 0.01)
        if b > 0 and h > 0:
            return f.create_entity(
                "IfcRectangleProfileDef", ProfileType="AREA",
                ProfileName=str(section_name), Position=pos, XDim=b, YDim=h)
        return None

    # Bars: associate a material-profile set (material + section profile).
    for sec_name, mems in bar_members_by_section.items():
        s = struc.sections.get(sec_name)
        mat_name = getattr(s, "material_name", None) if s else None
        mat = _ifc_material(mat_name) if mat_name else None
        prof = _profile(sec_name)
        if mat is None and prof is None:
            continue
        mprof = f.create_entity(
            "IfcMaterialProfile", Name=str(sec_name), Material=mat, Profile=prof)
        mset = f.create_entity(
            "IfcMaterialProfileSet", Name=str(sec_name), MaterialProfiles=[mprof])
        f.create_entity(
            "IfcRelAssociatesMaterial", GlobalId=guid(),
            RelatedObjects=list(mems), RelatingMaterial=mset)

    # Surfaces: associate the section's material directly.
    for sec_name, mems in area_members_by_section.items():
        sec = tsecs.get(sec_name) or qsecs.get(sec_name)
        mat_name = getattr(sec, "material_name", None) if sec else None
        if not mat_name:
            continue
        f.create_entity(
            "IfcRelAssociatesMaterial", GlobalId=guid(),
            RelatedObjects=list(mems), RelatingMaterial=_ifc_material(mat_name))

    # ── Phase B: load cases + actions ─────────────────────────────────
    load_case_of: dict[str, object] = {}
    actions_of: dict[str, list] = {}

    def _load_case(lc_id):
        if lc_id in load_case_of:
            return load_case_of[lc_id]
        lc = f.create_entity(
            "IfcStructuralLoadCase", GlobalId=guid(), Name=str(lc_id),
            PredefinedType="LOAD_GROUP", ActionType="NOTDEFINED",
            ActionSource="NOTDEFINED")
        load_case_of[lc_id] = lc
        actions_of[lc_id] = []
        return lc

    def _add_action(lc_id, action, target):
        """Register *action* under load case *lc_id*, connected to *target*."""
        _load_case(lc_id)
        actions_of[lc_id].append(action)
        if target is not None:
            f.create_entity(
                "IfcRelConnectsStructuralActivity", GlobalId=guid(),
                RelatingElement=target, RelatedStructuralActivity=action)

    # Nodal point loads. fx/fy/mz slots: (Fx,Fy,Mz) plane / (Fz,Mx,My) plate.
    for pl in getattr(struc, "point_loads", []) or []:
        conn = conn_of.get(pl.node_id)
        if conn is None:
            continue
        if plate:
            fz, mx, my = pl.fx, pl.fy, pl.mz
            fx = None
        else:
            fx, fz, my = pl.fx, pl.fy, -pl.mz
            mx = None
        force = f.create_entity(
            "IfcStructuralLoadSingleForce", Name=str(pl.load_case_id),
            ForceX=fx, ForceY=None, ForceZ=fz,
            MomentX=mx, MomentY=my, MomentZ=None)
        action = f.create_entity(
            "IfcStructuralPointAction", GlobalId=guid(),
            Name=f"PL-{pl.node_id}", AppliedLoad=force, GlobalOrLocal="GLOBAL_COORDS")
        _add_action(pl.load_case_id, action, conn)

    # Bar distributed loads (plane): axial → X, transverse → Z (global).
    # A trapezoidal load is exported as its mean intensity (Phase B limitation).
    for dl in getattr(struc, "distributed_loads", []) or []:
        member = member_by_id.get(dl.element_id)
        if member is None:
            continue
        fx = 0.5 * (dl.fxe + dl.fxd)
        ftrans = 0.5 * (dl.fye + dl.fyd)
        if not fx and not ftrans:
            continue
        if plate:
            lfx = None
            lfz = ftrans or None
        else:
            lfx = fx or None
            lfz = ftrans or None
        load = f.create_entity(
            "IfcStructuralLoadLinearForce", Name=str(dl.load_case_id),
            LinearForceX=lfx, LinearForceY=None, LinearForceZ=lfz)
        action = f.create_entity(
            "IfcStructuralLinearAction", GlobalId=guid(),
            Name=f"DL-{dl.element_id}", AppliedLoad=load,
            GlobalOrLocal="GLOBAL_COORDS", PredefinedType="CONST")
        _add_action(dl.load_case_id, action, member)

    # Area loads (plate): uniform pressure pz along global Z.
    def _emit_area_load(elem_id, lc_id, pz):
        member = member_by_id.get(elem_id)
        if member is None or not pz:
            return
        load = f.create_entity(
            "IfcStructuralLoadPlanarForce", Name=str(lc_id),
            PlanarForceX=None, PlanarForceY=None, PlanarForceZ=float(pz))
        action = f.create_entity(
            "IfcStructuralPlanarAction", GlobalId=guid(),
            Name=f"AL-{elem_id}", AppliedLoad=load,
            GlobalOrLocal="GLOBAL_COORDS", PredefinedType="CONST")
        _add_action(lc_id, action, member)

    for al in getattr(struc, "tri_area_loads", []) or []:
        _emit_area_load(al.tri_id, al.load_case_id, al.pz)
    for al in getattr(struc, "quad_area_loads", []) or []:
        _emit_area_load(al.quad_id, al.load_case_id, al.pz)

    # Group each load case's actions and mark it as loading the model.
    load_groups = []
    for lc_id, lc in load_case_of.items():
        acts = actions_of.get(lc_id) or []
        if acts:
            f.create_entity(
                "IfcRelAssignsToGroup", GlobalId=guid(),
                RelatedObjects=acts, RelatingGroup=lc)
        load_groups.append(lc)

    # ── Analysis model grouping ───────────────────────────────────────
    plane_dir = f.create_entity(
        "IfcAxis2Placement3D", Location=origin, Axis=axis_z, RefDirection=axis_x)
    model = f.create_entity(
        "IfcStructuralAnalysisModel", GlobalId=guid(),
        Name="xdfem2D analysis model",
        PredefinedType="NOTDEFINED", OrientationOf2DPlane=plane_dir,
        LoadedBy=load_groups or None)
    all_items = connections + members
    if all_items:
        f.create_entity(
            "IfcRelAssignsToGroup", GlobalId=guid(),
            RelatedObjects=all_items, RelatingGroup=model)
    f.create_entity(
        "IfcRelDeclares", GlobalId=guid(),
        RelatingContext=project, RelatedDefinitions=[model])
    return f


def _boundary_condition(f, sup, plate: bool):
    """Return an IfcBoundaryNodeCondition for *sup*, or None if no support.

    The three stored flags (ux/uy/tz slots) mean (ux, uy, θz) in the plane
    domain and (w, θx, θy) in the plate domain, mapped to global DOFs:
      plane: ux→transl X, uy→transl Z, θz→rot Y;
      plate: w→transl Z, θx→rot X, θy→rot Y.
    A fixed DOF is exported as a rigid (IfcBoolean True) stiffness.
    """
    if sup is None:
        return None

    def b(flag):
        return f.create_entity("IfcBoolean", bool(flag)) if flag else None

    if plate:
        tx = ty = None
        tz = b(sup.ux)          # w
        rx = b(sup.uy)          # θx
        ry = b(sup.tz)          # θy
        rz = None
    else:
        tx = b(sup.ux)          # ux
        ty = None
        tz = b(sup.uy)          # uy → global Z
        rx = None
        ry = b(sup.tz)          # θz (in-plane) → rotation about global Y
        rz = None
    return f.create_entity(
        "IfcBoundaryNodeCondition",
        TranslationalStiffnessX=tx,
        TranslationalStiffnessY=ty,
        TranslationalStiffnessZ=tz,
        RotationalStiffnessX=rx,
        RotationalStiffnessY=ry,
        RotationalStiffnessZ=rz)


def save_ifc(struc, path: str | Path) -> None:
    """Write *struc* to an IFC4 file at *path*."""
    f = to_ifc(struc)
    f.write(str(path))


# ══════════════════════════════════════════════════════════════════════
#  Import (Phase C) — IFC structural-analysis view → Structure2D
# ══════════════════════════════════════════════════════════════════════

class AmbiguousDomainError(ValueError):
    """Raised when the analysis plane does not identify a 2D domain.

    The caller can retry ``from_ifc``/``load_ifc`` with an explicit
    ``domain=`` ("plane" or "plate"), mirroring the SAP2000 importer.
    """


class NonPlanarModelError(ValueError):
    """Raised when the IFC model is genuinely 3D and cannot map to 2D."""


_TOL = 1e-6


def _infer_domain(coords):
    """Infer ("plane"|"plate") from the spread of 3D node coordinates.

    A model lying in the global X–Z plane (Y≈const) is a ``plane`` model; one
    in the X–Y plane (Z≈const) is a ``plate`` model. Genuinely 3D geometry
    raises NonPlanarModelError; a degenerate set (a line/point) that fits both
    planes raises AmbiguousDomainError.
    """
    if not coords:
        raise AmbiguousDomainError("The IFC model has no structural nodes.")
    ys = [c[1] for c in coords]
    zs = [c[2] for c in coords]
    y_flat = (max(ys) - min(ys)) <= _TOL
    z_flat = (max(zs) - min(zs)) <= _TOL
    if y_flat and z_flat:
        raise AmbiguousDomainError(
            "The IFC geometry is degenerate (all nodes share Y and Z); "
            "the analysis domain cannot be inferred.")
    if y_flat:
        return "plane"          # X–Z plane
    if z_flat:
        return "plate"          # X–Y plane
    raise NonPlanarModelError(
        "The IFC model is genuinely three-dimensional (nodes vary in both "
        "Y and Z) and cannot be imported as a 2D model.")


def _material_from_ifc(f, ifc_mat):
    """Return (name, E, nu, unit_weight) read from an IfcMaterial, with defaults."""
    name = getattr(ifc_mat, "Name", None) or "IFC_material"
    # Values are in xdfem2D / SAP2000 units (kN/m², t/m³, 1/°C). Defaults too.
    E, nu, rho = 30e6, 0.2, None
    for mp in f.by_type("IfcMaterialProperties"):
        if mp.Material != ifc_mat:
            continue
        for p in mp.Properties or []:
            if not p.is_a("IfcPropertySingleValue") or p.NominalValue is None:
                continue
            val = p.NominalValue.wrappedValue
            if p.Name in ("YoungModulus", "ModulusOfElasticity"):
                E = float(val)
            elif p.Name == "PoissonRatio":
                nu = float(val)
            elif p.Name == "MassDensity":
                rho = float(val)                 # t/m³
    uw = rho * 9.81 if rho else 25.0
    return name, E, nu, uw


def _member_material_map(f):
    """Map each structural member entity → (IfcMaterial, IfcProfileDef|None)."""
    out = {}
    for rel in f.by_type("IfcRelAssociatesMaterial"):
        rm = rel.RelatingMaterial
        mat = prof = None
        if rm.is_a("IfcMaterial"):
            mat = rm
        elif rm.is_a("IfcMaterialProfileSet") and rm.MaterialProfiles:
            mp = rm.MaterialProfiles[0]
            mat, prof = mp.Material, mp.Profile
        elif rm.is_a("IfcMaterialProfile"):
            mat, prof = rm.Material, rm.Profile
        for obj in rel.RelatedObjects:
            out[obj] = (mat, prof)
    return out


def _section_from_profile(prof):
    """Return (b, h, shape) for a bar section from an IfcProfileDef."""
    from .models import SectionShape
    if prof is None:
        return 0.3, 0.3, SectionShape.RECTANGULAR
    if prof.is_a("IfcRectangleProfileDef"):
        return float(prof.XDim), float(prof.YDim), SectionShape.RECTANGULAR
    if prof.is_a("IfcCircleProfileDef"):
        d = 2.0 * float(prof.Radius)
        return d, d, SectionShape.CIRCULAR
    if prof.is_a("IfcIShapeProfileDef"):
        return (float(prof.OverallWidth), float(prof.OverallDepth),
                SectionShape.I)
    return 0.3, 0.3, SectionShape.RECTANGULAR


def from_ifc(model, domain: str | None = None):
    """Build a Structure2D from an ``ifcopenshell.file`` (analysis view).

    ``domain`` forces the 2D domain; if None it is inferred from the analysis
    plane (X–Z → plane, X–Y → plate). Ambiguous or 3D geometry raises
    AmbiguousDomainError / NonPlanarModelError.
    """
    from .models import DOMAINS, SectionShape
    from .structure import Structure2D
    f = model

    conns = f.by_type("IfcStructuralPointConnection")
    if not conns:
        raise ValueError("No IfcStructuralPointConnection found — the file has "
                         "no structural-analysis model to import.")

    # node id + 3D coordinate + the geometry entities that identify it.
    node3d = {}          # node_id → (x, y, z)
    vertex_to_node = {}
    cp_to_node = {}
    cond_of = {}         # node_id → IfcBoundaryNodeCondition
    for i, c in enumerate(conns):
        nid = c.Name or f"N{i}"
        vertex = c.Representation.Representations[0].Items[0]
        cp = vertex.VertexGeometry
        xyz = tuple(float(v) for v in cp.Coordinates)
        node3d[nid] = xyz
        vertex_to_node[vertex] = nid
        cp_to_node[cp] = nid
        if getattr(c, "AppliedCondition", None) is not None:
            cond_of[nid] = c.AppliedCondition

    if domain is not None and domain not in DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'.")
    dom = domain or _infer_domain(list(node3d.values()))
    plate = dom == "plate"

    def _xy(xyz):
        return (xyz[0], xyz[1]) if plate else (xyz[0], xyz[2])

    struc = Structure2D(domain=dom)
    for nid, xyz in node3d.items():
        x, y = _xy(xyz)
        struc.add_node(nid, x, y)

    # Materials + per-member associations.
    mat_map = _member_material_map(f)
    added_mats = {}

    def _ensure_material(ifc_mat):
        if ifc_mat is None:
            key = "_IFC_DEFAULT"
            if key not in added_mats:
                struc.add_material(key, elastic_modulus=30e6, unit_weight=25.0,
                                   poisson=0.2)
                added_mats[key] = key
            return key
        name, E, nu, uw = _material_from_ifc(f, ifc_mat)
        if name not in added_mats:
            struc.add_material(name, elastic_modulus=E, unit_weight=uw,
                               poisson=nu)
            added_mats[name] = name
        return name

    # Bars.
    bar_sections = {}
    for m in f.by_type("IfcStructuralCurveMember"):
        edge = m.Representation.Representations[0].Items[0]
        ni = vertex_to_node.get(edge.EdgeStart)
        nj = vertex_to_node.get(edge.EdgeEnd)
        if ni is None or nj is None:
            continue
        ifc_mat, prof = mat_map.get(m, (None, None))
        mat_name = _ensure_material(ifc_mat)
        b, h, shape = _section_from_profile(prof)
        sec_name = (getattr(prof, "ProfileName", None) or m.Name
                    or f"S_{m.id()}")
        if sec_name not in bar_sections:
            struc.add_section(sec_name, mat_name, b=b, h=h,
                              shape=shape.value if hasattr(shape, "value")
                              else shape)
            bar_sections[sec_name] = True
        struc.add_bar_element(m.Name or f"B{m.id()}", ni, nj, sec_name)

    # Surface members → triangles / quads.
    area_sections = {}
    tri_form = "MITC3" if plate else "CST"
    quad_form = "MITC4" if plate else "QM6"

    def _area_section(mat_name, thickness, nsides):
        key = (mat_name, round(thickness, 6), nsides)
        if key in area_sections:
            return area_sections[key]
        name = f"A{nsides}_{mat_name}_{thickness:g}"
        if nsides == 3:
            struc.add_tri_section(name, mat_name, thickness=thickness,
                                  formulation=tri_form)
        else:
            struc.add_quad_section(name, mat_name, thickness=thickness,
                                   formulation=quad_form)
        area_sections[key] = name
        return name

    for m in f.by_type("IfcStructuralSurfaceMember"):
        face = m.Representation.Representations[0].Items[0]
        loop = face.Bounds[0].Bound
        pts = loop.Polygon
        nids = [cp_to_node.get(cp) for cp in pts]
        if any(n is None for n in nids):
            continue
        ifc_mat, _ = mat_map.get(m, (None, None))
        mat_name = _ensure_material(ifc_mat)
        thick = float(m.Thickness) if m.Thickness else 0.2
        aid = m.Name or f"A{m.id()}"
        if len(nids) == 3:
            sec = _area_section(mat_name, thick, 3)
            struc.add_tri_element(aid, nids[0], nids[1], nids[2], sec)
        elif len(nids) == 4:
            sec = _area_section(mat_name, thick, 4)
            struc.add_quad_element(aid, nids[0], nids[1], nids[2], nids[3], sec)

    # Supports from boundary conditions.
    _import_supports(struc, cond_of, plate)

    # Loads.
    _import_loads(f, struc, plate, vertex_to_node)
    return struc


def _fixed(stiff):
    """True when an IfcBoundaryNodeCondition stiffness select means 'restrained'."""
    if stiff is None:
        return False
    wrapped = getattr(stiff, "wrappedValue", stiff)
    if isinstance(wrapped, bool):
        return wrapped
    try:
        return float(wrapped) != 0.0
    except (TypeError, ValueError):
        return True


def _import_supports(struc, cond_of, plate):
    """Recreate supports from per-node IfcBoundaryNodeCondition entities."""
    made = {}
    for nid, cond in cond_of.items():
        if plate:
            ux = _fixed(cond.TranslationalStiffnessZ)   # w
            uy = _fixed(cond.RotationalStiffnessX)       # θx
            tz = _fixed(cond.RotationalStiffnessY)       # θy
        else:
            ux = _fixed(cond.TranslationalStiffnessX)    # ux
            uy = _fixed(cond.TranslationalStiffnessZ)    # uy
            tz = _fixed(cond.RotationalStiffnessY)       # θz
        if not (ux or uy or tz):
            continue
        key = (ux, uy, tz)
        name = made.get(key)
        if name is None:
            name = f"S_{'1' if ux else '0'}{'1' if uy else '0'}{'1' if tz else '0'}"
            if name not in struc.supports:
                struc.add_support(name, ux=ux, uy=uy, tz=tz)
            made[key] = name
        struc.assign_support(nid, name)


def _action_load_case(f):
    """Map each structural action entity → its load-case id (Name)."""
    out = {}
    for rel in f.by_type("IfcRelAssignsToGroup"):
        grp = rel.RelatingGroup
        if not grp.is_a("IfcStructuralLoadGroup"):
            continue
        for obj in rel.RelatedObjects:
            out[obj] = grp.Name
    return out


def _action_target(f):
    """Map each structural action entity → the structural item it loads."""
    out = {}
    for rel in f.by_type("IfcRelConnectsStructuralActivity"):
        out[rel.RelatedStructuralActivity] = rel.RelatingElement
    return out


def _import_loads(f, struc, plate, vertex_to_node):
    """Recreate load cases, point / linear / planar actions."""
    lc_of = _action_load_case(f)
    tgt_of = _action_target(f)
    seen_lc = set()

    def _lc(name):
        name = name or "LC"
        if name not in seen_lc and name not in struc.load_cases_by_id:
            struc.add_load_case(name)
            seen_lc.add(name)
        return name

    # Values are in xdfem2D / SAP2000 units (kN, kN/m, kN/m²).
    def _v(x):
        return x or 0.0

    for a in f.by_type("IfcStructuralPointAction"):
        tgt = tgt_of.get(a)
        if tgt is None or not tgt.is_a("IfcStructuralPointConnection"):
            continue
        nid = tgt.Name
        load = a.AppliedLoad
        lc = _lc(lc_of.get(a))
        if plate:
            struc.add_point_load(nid, lc, fz=_v(load.ForceZ),
                                 mx=_v(load.MomentX), my=_v(load.MomentY))
        else:
            struc.add_point_load(nid, lc, fx=_v(load.ForceX),
                                 fy=_v(load.ForceZ), mz=-_v(load.MomentY))

    for a in f.by_type("IfcStructuralLinearAction"):
        tgt = tgt_of.get(a)
        if tgt is None:
            continue
        load = a.AppliedLoad
        lc = _lc(lc_of.get(a))
        fx = _v(load.LinearForceX)
        fy = _v(load.LinearForceZ)
        struc.add_distributed_load(tgt.Name, lc, fxe=fx, fxd=fx, fye=fy, fyd=fy)

    for a in f.by_type("IfcStructuralPlanarAction"):
        tgt = tgt_of.get(a)
        if tgt is None:
            continue
        load = a.AppliedLoad
        lc = _lc(lc_of.get(a))
        struc.add_area_load(tgt.Name, lc, pz=_v(load.PlanarForceZ))


def load_ifc(path: str | Path, domain: str | None = None):
    """Open the IFC file at *path* and import it as a Structure2D."""
    ios = _require_ifcopenshell()
    model = ios.open(str(path))
    return from_ifc(model, domain=domain)


# ══════════════════════════════════════════════════════════════════════
#  Phase E — physical-model geometry import (lossy, drawing layer only)
# ══════════════════════════════════════════════════════════════════════
#
# Many BIM tools export only the *physical* model (IfcBeam, IfcColumn, IfcSlab,
# …) — geometry, no nodes/restraints/loads. This path pulls the beam/column
# centre-lines and slab/wall footprints in as xdfem2D *geometry objects* (the
# drawing layer): open polylines for members, closed polygons for slabs. It is
# explicitly lossy — no supports, loads, materials or sections are inferred —
# and is meant as a drafting aid, not an analysis import.

_CURVE_ELEMENTS = ("IfcBeam", "IfcColumn", "IfcMember")
_SURFACE_ELEMENTS = ("IfcSlab", "IfcWall", "IfcWallStandardCase", "IfcPlate")


def _local_xyz(cartesian_point):
    c = list(cartesian_point.Coordinates)
    while len(c) < 3:
        c.append(0.0)
    return c[:3]


def _rep_polylines(rep):
    """Local-coordinate polylines (list of [x,y,z]) from a shape representation."""
    out = []
    if rep is None:
        return out
    for item in rep.Items:
        if item.is_a("IfcMappedItem"):
            out.extend(_rep_polylines(item.MappingSource.MappedRepresentation))
        elif item.is_a("IfcPolyline"):
            out.append([_local_xyz(p) for p in item.Points])
        elif item.is_a("IfcIndexedPolyCurve"):
            pl = item.Points
            if pl.is_a() in ("IfcCartesianPointList2D", "IfcCartesianPointList3D"):
                pts = []
                for c in pl.CoordList:
                    c = list(c)
                    while len(c) < 3:
                        c.append(0.0)
                    pts.append(c[:3])
                out.append(pts)
        elif item.is_a("IfcTrimmedCurve"):
            trims = [t for t in list(item.Trim1) + list(item.Trim2)
                     if hasattr(t, "is_a") and t.is_a("IfcCartesianPoint")]
            if len(trims) >= 2:
                out.append([_local_xyz(trims[0]), _local_xyz(trims[1])])
        elif item.is_a("IfcGeometricCurveSet") or item.is_a("IfcGeometricSet"):
            for e in item.Elements or []:
                if e.is_a("IfcPolyline"):
                    out.append([_local_xyz(p) for p in e.Points])
    return out


def _get_rep(product, identifier):
    if not getattr(product, "Representation", None):
        return None
    for r in product.Representation.Representations:
        if r.RepresentationIdentifier == identifier:
            return r
    return None


def from_ifc_physical(model, domain: str | None = None):
    """Import IfcBeam/Column/Member axes and IfcSlab/Wall footprints as
    xdfem2D geometry objects (drawing layer only — see the module note)."""
    import numpy as np
    import ifcopenshell.util.placement as placement
    from .models import DOMAINS
    from .templates import _default_new_structure
    f = model

    def _world(product, local_pts):
        if getattr(product, "ObjectPlacement", None) is not None:
            M = placement.get_local_placement(product.ObjectPlacement)
        else:
            M = np.eye(4)
        out = []
        for c in local_pts:
            w = M @ np.array([c[0], c[1], c[2], 1.0])
            out.append((float(w[0]), float(w[1]), float(w[2])))
        return out

    members, areas = [], []      # world-coordinate polylines
    for t in _CURVE_ELEMENTS:
        for prod in f.by_type(t):
            rep = _get_rep(prod, "Axis")
            for pl in _rep_polylines(rep):
                if len(pl) >= 2:
                    members.append(_world(prod, pl))
    for t in _SURFACE_ELEMENTS:
        for prod in f.by_type(t):
            rep = _get_rep(prod, "FootPrint") or _get_rep(prod, "Footprint")
            for pl in _rep_polylines(rep):
                if len(pl) >= 3:
                    areas.append(_world(prod, pl))

    if not members and not areas:
        raise ValueError(
            "No importable physical geometry found (no IfcBeam/IfcColumn axis "
            "curves or IfcSlab/IfcWall footprints).")

    all_pts = [p for pl in members + areas for p in pl]
    if domain is not None and domain not in DOMAINS:
        raise ValueError(f"Unknown domain '{domain}'.")
    dom = domain or _infer_domain(all_pts)
    plate = dom == "plate"

    def _xy(p):
        return (p[0], p[1]) if plate else (p[0], p[2])

    s = _default_new_structure(domain=dom)
    bar_sec = "Sec"
    area_sec = "Slab" if plate else "CST"
    oid = 0
    for pl in members:
        verts = [_xy(p) for p in pl]
        # Drop consecutive duplicates that would create zero-length segments.
        clean = [verts[0]]
        for v in verts[1:]:
            if abs(v[0] - clean[-1][0]) > _TOL or abs(v[1] - clean[-1][1]) > _TOL:
                clean.append(v)
        if len(clean) < 2:
            continue
        oid += 1
        s.add_geo_polyline(f"O{oid}", clean, closed=False, section_name=bar_sec)
    for pl in areas:
        verts = [_xy(p) for p in pl]
        if len(verts) > 1 and abs(verts[0][0] - verts[-1][0]) <= _TOL \
                and abs(verts[0][1] - verts[-1][1]) <= _TOL:
            verts = verts[:-1]
        if len(verts) < 3:
            continue
        oid += 1
        s.add_geo_polygon(f"O{oid}", verts, section_name=area_sec)
    return s


def load_ifc_physical(path: str | Path, domain: str | None = None):
    """Open the IFC file at *path* and import its physical geometry (Phase E)."""
    ios = _require_ifcopenshell()
    model = ios.open(str(path))
    return from_ifc_physical(model, domain=domain)
