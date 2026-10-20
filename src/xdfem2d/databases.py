"""
Eurocode material and steel-profile databases for xdfem2D.

Sources concrete/steel/timber properties from the ``eurocodepy`` package (a
required dependency) via its loaded material database, instead of carrying
embedded copies of the values/formulae. GUI-free so it can be used from scripts
and tests; the GUI material/section dialogs call these.
"""
from __future__ import annotations


# ---------------------------------------------------------------------------
# Reference materials catalog (generic E / γ / α library)
# ---------------------------------------------------------------------------
# Built-in reference values for common structural materials, baked from
# ``materiais_estruturais.xlsx`` (EN 1992–1995 / EN 338 typical values). Each
# entry carries (min, typical, max) tuples already in xdfem2D units:
#   E     — kN/m²   (source GPa × 1e6)
#   gamma — kN/m³
#   alpha — 1/°C    (source ×10⁻⁶/°C)
# These are generic reference materials (no design class); the material dialog
# applies the *typical* value and sets the material type to "Other".
_MATERIALS_CATALOG = [
    {"name": "Structural steel (S275/S355)",        "category": "Metal",         "E": (200000000, 205000000, 210000000), "gamma": (77, 77, 77), "alpha": (1.2e-05, 1.2e-05, 1.2e-05)},
    {"name": "Stainless steel (austenitic)",        "category": "Metal",         "E": (193000000, 196500000, 200000000), "gamma": (78.5, 78.5, 78.5), "alpha": (1.6e-05, 1.65e-05, 1.7e-05)},
    {"name": "Cast iron",                           "category": "Metal",         "E": (100000000, 135000000, 170000000), "gamma": (71, 71, 71), "alpha": (1e-05, 1.05e-05, 1.1e-05)},
    {"name": "Aluminum (structural alloys)",        "category": "Metal",         "E": (68000000, 70000000, 72000000), "gamma": (27, 27, 27), "alpha": (2.3e-05, 2.3e-05, 2.3e-05)},
    {"name": "Copper",                              "category": "Metal",         "E": (110000000, 120000000, 130000000), "gamma": (87, 87, 87), "alpha": (1.7e-05, 1.7e-05, 1.7e-05)},
    {"name": "Titanium (Ti-6Al-4V alloy)",          "category": "Metal",         "E": (110000000, 112500000, 115000000), "gamma": (44, 44, 44), "alpha": (8e-06, 8.5e-06, 9e-06)},
    {"name": "Plain concrete",                      "category": "Concrete",      "E": (20000000, 25000000, 30000000), "gamma": (24, 24, 24), "alpha": (1e-05, 1e-05, 1e-05)},
    {"name": "Reinforced concrete (C25/30)",        "category": "Concrete",      "E": (30000000, 30500000, 31000000), "gamma": (25, 25, 25), "alpha": (1e-05, 1e-05, 1e-05)},
    {"name": "High-strength concrete (C70+)",       "category": "Concrete",      "E": (40000000, 42500000, 45000000), "gamma": (25, 25, 25), "alpha": (1e-05, 1e-05, 1e-05)},
    {"name": "Lightweight concrete",                "category": "Concrete",      "E": (10000000, 15000000, 20000000), "gamma": (8, 13, 18), "alpha": (8e-06, 9e-06, 1e-05)},
    {"name": "Prestressed concrete",                "category": "Concrete",      "E": (30000000, 32500000, 35000000), "gamma": (25, 25, 25), "alpha": (1e-05, 1e-05, 1e-05)},
    {"name": "Sawn timber (C24 – EN 338)",          "category": "Timber",        "E": (11000000, 11000000, 11000000), "gamma": (4.2, 4.6, 5), "alpha": (3e-06, 4e-06, 5e-06)},
    {"name": "Glued-laminated timber (GL24h)",      "category": "Timber",        "E": (11500000, 11500000, 11500000), "gamma": (4.2, 4.6, 5), "alpha": (3e-06, 4e-06, 5e-06)},
    {"name": "LVL (Kerto)",                         "category": "Timber",        "E": (13000000, 13500000, 14000000), "gamma": (5, 5, 5), "alpha": (3e-06, 4e-06, 5e-06)},
    {"name": "CLT (Cross-Laminated Timber)",        "category": "Timber",        "E": (9000000, 10500000, 12000000), "gamma": (4.5, 4.75, 5), "alpha": (3e-06, 4e-06, 5e-06)},
    {"name": "Granite",                             "category": "Stone/Masonry", "E": (40000000, 55000000, 70000000), "gamma": (27, 27, 27), "alpha": (8e-06, 9e-06, 1e-05)},
    {"name": "Limestone",                           "category": "Stone/Masonry", "E": (20000000, 45000000, 70000000), "gamma": (22, 24.5, 27), "alpha": (8e-06, 8.5e-06, 9e-06)},
    {"name": "Marble",                              "category": "Stone/Masonry", "E": (50000000, 70000000, 90000000), "gamma": (26, 27, 28), "alpha": (5e-06, 6e-06, 7e-06)},
    {"name": "Slate",                                "category": "Stone/Masonry", "E": (20000000, 25000000, 30000000), "gamma": (28, 28, 28), "alpha": (9e-06, 1e-05, 1.1e-05)},
    {"name": "Brick masonry",                       "category": "Stone/Masonry", "E": (1000000, 5500000, 10000000), "gamma": (18, 19, 20), "alpha": (5e-06, 6e-06, 7e-06)},
    {"name": "Concrete block masonry",               "category": "Stone/Masonry", "E": (5000000, 10000000, 15000000), "gamma": (20, 21, 22), "alpha": (6e-06, 8e-06, 1e-05)},
    {"name": "Rigid PVC",                           "category": "Polymer",       "E": (2400000, 3250000, 4100000), "gamma": (13, 13.5, 14), "alpha": (5e-05, 6.5e-05, 8e-05)},
    {"name": "Polycarbonate (PC)",                  "category": "Polymer",       "E": (2000000, 2200000, 2400000), "gamma": (12, 12, 12), "alpha": (6.5e-05, 6.75e-05, 7e-05)},
    {"name": "HDPE",                                "category": "Polymer",       "E": (600000, 1000000, 1400000), "gamma": (9.3, 9.5, 9.7), "alpha": (0.0001, 0.00015, 0.0002)},
    {"name": "Nylon (PA6/PA66)",                    "category": "Polymer",       "E": (2500000, 3000000, 3500000), "gamma": (11, 11, 11), "alpha": (8e-05, 8e-05, 8e-05)},
    {"name": "GFRP (glass fiber / epoxy)",          "category": "Composite",     "E": (15000000, 27500000, 40000000), "gamma": (18, 20, 22), "alpha": (6e-06, 7e-06, 8e-06)},
    {"name": "CFRP (carbon fiber / epoxy)",         "category": "Composite",     "E": (70000000, 135000000, 200000000), "gamma": (15, 16, 17), "alpha": (0, 1e-06, 2e-06)},
    {"name": "AFRP (aramid fiber / epoxy)",         "category": "Composite",     "E": (40000000, 85000000, 130000000), "gamma": (12, 13, 14), "alpha": (-2e-06, 1e-06, 4e-06)},
    {"name": "Pultruded FRP profiles",               "category": "Composite",     "E": (17000000, 22500000, 28000000), "gamma": (18, 19, 20), "alpha": (6e-06, 8e-06, 1e-05)},
]


def materials_catalog() -> list[dict]:
    """Return the built-in reference materials catalog (see _MATERIALS_CATALOG).

    Each entry: ``{name, category, E:(min,typ,max), gamma:(...), alpha:(...)}``
    with E in kN/m², gamma in kN/m³, alpha in 1/°C.
    """
    return _MATERIALS_CATALOG


def _load_eurocodepy_json():
    """Return the eurocodepy material database (``dbase.Materials``).

    This is the dict eurocodepy itself loads from ``eurocodes.json``: a mapping
    of material category → {"Grade": {...}, "Parameters": {...}}. xdfem2D reads
    eurocodepy's loaded variables, never the JSON file directly.
    """
    from eurocodepy import dbase
    return dbase.Materials


def _build_eco_db():
    """Return {type_label: {grade: (E_kN_m2, gamma_kN_m3, alpha)}}.

    Built from eurocodepy's exposed database variables (``dbase.ConcreteGrades``
    etc.), not by reading ``eurocodes.json`` directly; no embedded fallback.
    """
    from eurocodepy import dbase
    db = {}

    # ── Concrete EC2 ──────────────────────────────────────────
    uw_c = dbase.ConcreteParams["weigh"]
    alpha_c = dbase.ConcreteParams["alpha"]
    db["Concrete (EC2)"] = {
        key.replace("_", "/"): (vals["Ecm"] * 1e3, uw_c, alpha_c)  # MPa → kN/m²
        for key, vals in dbase.ConcreteGrades.items()
    }

    # ── Structural Steel EC3 ───────────────────────────────────
    uw_s = dbase.SteelParams["weigh"]
    alpha_s = dbase.SteelParams.get("alpha", 1.2e-5)
    db["Structural Steel (EC3)"] = {
        g: (vals["Es"] * 1e3, uw_s, alpha_s)
        for g, vals in dbase.SteelGrades.items()
    }

    # ── Reinforcing Steel ─────────────────────────────────────
    uw_r = dbase.ReinforcementParams.get("weigh", 77.0)
    db["Reinforcing Steel"] = {
        g: (vals["Es"] * 1e3, uw_r, 1.2e-5)
        for g, vals in dbase.ReinforcementGrades.items()
    }

    # ── Timber EC5 ────────────────────────────────────────────
    timber_dict = {}
    for g, vals in dbase.TimberGrades.items():
        E_kN = vals["E0mean"] * 1e3                      # MPa → kN/m²
        rho = vals.get("rhom", vals.get("rhok", 400))   # kg/m³
        uw = rho * 9.81 / 1000                          # kN/m³
        timber_dict[g] = (E_kN, uw, 5e-6)
    db["Timber (EC5)"] = timber_dict

    return db


_ECO_DB = _build_eco_db()


# ---------------------------------------------------------------------------
# Design-grade catalogues (for the material editor dropdowns)
# ---------------------------------------------------------------------------
# Each returns {grade_name: {property: value}} with strengths in MPa, ρ in
# kg/m³ and E in kN/m² (consistent with the Material.elastic_modulus units).

def concrete_design_grades() -> dict:
    """Concrete grades → {fck [MPa], E [kN/m²], unit_weight [kN/m³], alpha}."""
    from eurocodepy import dbase
    uw = dbase.ConcreteParams.get("weigh", 25.0)
    al = dbase.ConcreteParams.get("alpha", 1e-5)
    return {g.replace("_", "/"): {
                "fck": v["fck"], "E": v["Ecm"] * 1e3,
                "unit_weight": uw, "alpha": al}
            for g, v in dbase.ConcreteGrades.items()}


def reinforcement_design_grades() -> dict:
    """Reinforcement grades → {fyk [MPa], E [kN/m²]}."""
    from eurocodepy import dbase
    return {g: {"fyk": v["fyk"], "E": v["Es"] * 1e3}
            for g, v in dbase.ReinforcementGrades.items()}


def steel_design_grades() -> dict:
    """Structural-steel grades → {fy, fu [MPa], E [kN/m²], unit_weight}."""
    from eurocodepy import dbase
    uw = dbase.SteelParams.get("weigh", 77.0)
    return {g: {"fy": v["fyk"], "fu": v["fuk"],
                "E": v["Es"] * 1e3, "unit_weight": uw}
            for g, v in dbase.SteelGrades.items()}


def timber_design_grades() -> dict:
    """Timber grades → {fmk, fvk, fc0k, ft0k [MPa], rhok [kg/m³], E, unit_weight}."""
    from eurocodepy import dbase
    out = {}
    for g, v in dbase.TimberGrades.items():
        rhok = v["rhok"]
        out[g] = {
            "fmk": v["fmk"], "fvk": v["fvk"],
            "fc0k": v["fc0k"], "ft0k": v["ft0k"],
            "rhok": rhok, "E": v["E0mean"] * 1e3,
            "unit_weight": rhok * 9.81 / 1000.0,
        }
    return out


# Coefficient of thermal expansion [1/°C] and Poisson's ratio ν per material
# type. eurocodepy has no value for these, so they live here, once, for every
# path that builds a material from a class (the Material dialog, the template
# dialog and Structure2D.create_*_material). Timber's ν is 0: its elastic
# behaviour is orthotropic and G is not E/2(1+ν), so ν carries no information.
_ALPHA = {"Concrete": None, "Steel": 1.2e-5, "Timber": 5e-6}   # concrete: eurocodepy
_POISSON = {"Concrete": 0.2, "Steel": 0.3, "Timber": 0.0}


def material_from_grade(material_type, grade: str,
                        reinforcement: str | None = None) -> dict:
    """Everything a Material takes, from a Eurocode class in the eurocodepy
    database: ``add_material(name, **material_from_grade("Steel", "S355"))``.

    This is what selecting a class in the Material dialog does, as one function
    so every path that makes a material from a class gives the same material.

    *material_type* is "Concrete" (class "C30/37", with *reinforcement* the
    reinforcement class, default "A500NR"), "Steel" ("S355") or "Timber"
    ("GL24h", "C24"). Returns ``elastic_modulus`` [kN/m²], ``unit_weight``
    [kN/m³], ``alpha``, ``poisson``, ``material_type`` and the complete
    ``design`` dict (class label and every strength the design reads).
    Raises ValueError, listing the valid classes, for an unknown one.
    """
    mt = getattr(material_type, "value", material_type)

    def _grade(table, name, what, plural="grades"):
        if name not in table:
            raise ValueError(f"unknown {what} '{name}' -- valid {plural}: "
                             f"{', '.join(sorted(table))}")
        return table[name]

    if mt == "Concrete":
        c = _grade(concrete_design_grades(), grade, "concrete class", "classes")
        reinf = reinforcement or "A500NR"
        r = _grade(reinforcement_design_grades(), reinf, "reinforcement grade")
        return {"elastic_modulus": c["E"], "unit_weight": c["unit_weight"],
                "alpha": c["alpha"], "poisson": _POISSON[mt],
                "material_type": mt,
                "design": {"class_conc": grade, "fck": c["fck"],
                           "class_reinf": reinf, "fyk": r["fyk"]}}
    if mt == "Steel":
        g = _grade(steel_design_grades(), grade, "steel grade")
        return {"elastic_modulus": g["E"], "unit_weight": g["unit_weight"],
                "alpha": _ALPHA[mt], "poisson": _POISSON[mt],
                "material_type": mt,
                "design": {"class": grade, "fy": g["fy"], "fu": g["fu"]}}
    if mt == "Timber":
        t = _grade(timber_design_grades(), grade, "timber grade")
        return {"elastic_modulus": t["E"], "unit_weight": t["unit_weight"],
                "alpha": _ALPHA[mt], "poisson": _POISSON[mt],
                "material_type": mt,
                "design": {"class": grade, "fmk": t["fmk"], "fvk": t["fvk"],
                           "fc0k": t["fc0k"], "ft0k": t["ft0k"],
                           "rhok": t["rhok"]}}
    raise ValueError(f"no class database for material type '{mt}'")


# ─────────────────────────────────────────────────────────────
# Material panel
# ─────────────────────────────────────────────────────────────


def _load_steel_profiles():
    """Return {family: [{name, h_m, b_m, A_m2, Iy_m4, m_kg_m}, ...]}.

    Dimensions in m, A in m², Iy in m⁴. Sourced from eurocodepy's loaded steel
    profile variables (``dbase.SteelIProfiles`` etc.), not by locating and
    reading the profile JSON files directly.
    """
    from eurocodepy import dbase

    families = {
        "I-profiles (HEA/HEB/HEM/IPE)": dbase.SteelIProfiles,
        "CHS (circular hollow)":         dbase.SteelCHSProfiles,
        "RHS (rectangular hollow)":      dbase.SteelRHSProfiles,
        "SHS (square hollow)":           dbase.SteelSHSProfiles,
    }
    profiles = {}
    for family, raw in families.items():
        entries = []
        for item in raw:
            # dimensions in cm → m; areas in cm² → m²; inertia in cm⁴ → m⁴
            def _cm(*keys):
                for k in keys:
                    v = item.get(k)
                    if v not in (None, ""):
                        try:
                            return float(v) / 100.0
                        except (TypeError, ValueError):
                            pass
                return 0.0
            def _num(key, factor):
                v = item.get(key)
                if v in (None, ""):
                    return None
                try:
                    return float(v) / factor
                except (TypeError, ValueError):
                    return None

            entries.append({
                "name":   item["Section"],
                "h_m":    item["h"]  / 100,
                "b_m":    item["b"]  / 100,
                "A_m2":   item["A"]  / 1e4,
                "Iy_m4":  item["Iy"] / 1e8,
                "m_kg_m": item.get("m", 0.0),
                # Web / flange thickness (I) or wall thickness (hollow). Mapped to
                # the section's tw (web/wall) and tf (flange) inputs.
                "tw_m":   _cm("tw", "t"),
                "tf_m":   _cm("tf", "t"),
                # Full design properties (SI): minor inertia, St-Venant torsion
                # constant, elastic/plastic section moduli and shear areas. Used
                # to fill a section completely when a catalogue profile is
                # applied (cm⁴→m⁴ /1e8, cm³→m³ /1e6, cm²→m² /1e4).
                "Iz_m4":    _num("Iz", 1e8),
                "J_m4":     _num("IT", 1e8),
                "Wel_y_m3": _num("Wel_y", 1e6),
                "Wpl_y_m3": _num("Wpl_y", 1e6),
                "Wel_z_m3": _num("Wel_z", 1e6),
                "Wpl_z_m3": _num("Wpl_z", 1e6),
                "Av_y_m2":  _num("Av_y", 1e4),
                "Av_z_m2":  _num("Av_z", 1e4),
                # Warping constant Iw (cm⁶→m⁶ /1e12) — needed for LTB (M_cr).
                "Iw_m6":    _num("Iw", 1e12),
                # Torsional section modulus Wt (cm³→m³) — for the torsion check.
                "Wt_m3":    _num("WT", 1e6),
                # Buckling curves for the y-y and z-z axes (EN 1993-1-1 Table
                # 6.2), tabulated in the catalogue as CurveA / CurveB.
                "curve_y":  item.get("CurveA"),
                "curve_z":  item.get("CurveB"),
            })
        profiles[family] = entries
    return profiles


_STEEL_PROFILES = _load_steel_profiles()

# Catalogue-family → section shape (mirrors the section dialog's mapping), so a
# profile looked up by name knows which SectionShape to build.
_FAMILY_SHAPE = {
    "I-profiles (HEA/HEB/HEM/IPE)": "I",
    "CHS (circular hollow)":        "Circular hollow",
    "RHS (rectangular hollow)":     "Rectangular hollow",
    "SHS (square hollow)":          "Rectangular hollow",
}


def steel_profile(name: str):
    """Look up a catalogue steel profile by name (e.g. ``"IPE300"``).

    Returns ``(shape_value, entry)`` where ``shape_value`` is the
    :class:`~xdfem2d.models.SectionShape` value string and ``entry`` the
    dimensions/properties dict (``b_m``, ``h_m``, ``tw_m``, ``tf_m``, ``A_m2``,
    ``Iy_m4`` …). ``None`` when no profile of that name exists.
    """
    for family, entries in _STEEL_PROFILES.items():
        for e in entries:
            if e.get("name") == name:
                return _FAMILY_SHAPE.get(family, "Generic"), e
    return None


def profile_shear_areas(shape: str, entry: dict) -> tuple:
    """``(Av_y, Av_z)`` [m²] of a catalogue steel profile.

    Rolled RHS/SHS take the EN 1993-1-1 §6.2.6(3)(e) values ``A·b/(b+h)`` (load
    parallel to the width) and ``A·h/(b+h)`` (parallel to the depth); the
    catalogue's webs-only figure is up to 8 % higher in the depth direction.
    Everything else keeps the catalogue value.
    """
    av_y, av_z = entry.get("Av_y_m2"), entry.get("Av_z_m2")
    if shape == "Rectangular hollow" and entry.get("A_m2"):
        a, b, h = entry["A_m2"], entry["b_m"], entry["h_m"]
        av_y, av_z = a * b / (b + h), a * h / (b + h)
    return av_y, av_z


# ─────────────────────────────────────────────────────────────
# Section panel
# ─────────────────────────────────────────────────────────────
