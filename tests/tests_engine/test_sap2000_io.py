"""Tests for the SAP2000 .s2k exporter (xdfem2d.sap2000_io)."""
import os
import tempfile
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)
from xdfem2d import save_s2k
from xdfem2d.sap2000_io import to_s2k


def _model():
    s = Structure2D()
    s.add_material("C30", elastic_modulus=30e6, unit_weight=25.0)
    s.add_section("S1", "C30", b=0.3, h=0.6)
    s.add_node("N1", 0.0, 0.0)
    s.add_node("N2", 5.0, 0.0)
    s.add_bar_element("E1", "N1", "N2", "S1", hinge_j=True)
    s.add_support("PIN", ux=True, uy=True)
    s.add_support("ROLLER", ux=False, uy=True)
    s.assign_support("N1", "PIN")
    s.assign_support("N2", "ROLLER")
    s.add_load_case("G", self_weight_factor=1.0)
    s.add_load_case("Q")
    s.add_point_load("N2", "Q", fy=-8.0, mz=2.0)
    s.add_distributed_load("E1", "G", fye=-10.0, fyd=-10.0)
    s.add_analysis_case("ULS", "Linear", {"G": 1.35, "Q": 1.5})
    s.add_analysis_case("PD", "GeometricNonlinear", {"G": 1.0, "Q": 1.0})
    s.add_load_combination("CO", {"G": 1.35, "Q": 1.5}, combo_type="LinearSum")
    return s


class TestS2kText(unittest.TestCase):
    def setUp(self):
        self.txt = to_s2k(_model())

    def test_required_tables_present(self):
        for table in (
            "PROGRAM CONTROL",
            "MATERIAL PROPERTIES 01 - GENERAL",
            "MATERIAL PROPERTIES 02 - BASIC MECHANICAL PROPERTIES",
            "FRAME SECTION PROPERTIES 01 - GENERAL",
            "JOINT COORDINATES",
            "CONNECTIVITY - FRAME",
            "FRAME SECTION ASSIGNMENTS",
            "JOINT RESTRAINT ASSIGNMENTS",
            "LOAD PATTERN DEFINITIONS",
            "LOAD CASE DEFINITIONS",
            "CASE - STATIC 1 - LOAD ASSIGNMENTS",
            "JOINT LOADS - FORCE",
            "FRAME LOADS - DISTRIBUTED",
            "COMBINATION DEFINITIONS",
        ):
            self.assertIn(f'TABLE:  "{table}"', self.txt)
        self.assertTrue(self.txt.rstrip().endswith("END TABLE DATA"))

    def test_program_control_uses_recognised_codes(self):
        self.assertIn("Version=26.3.0", self.txt)
        self.assertIn('SteelCode="AISC 360-16"', self.txt)
        # A concrete material carries EC2 design strengths (fck/fyk) by default,
        # so the export runs RC design under Eurocode 2, not the ACI fallback.
        self.assertIn('ConcCode="Eurocode 2-2004"', self.txt)

    def test_material_has_unitmass_and_tempdepend(self):
        self.assertIn("TempDepend=No", self.txt)
        self.assertIn("UnitMass=", self.txt)  # was a SAP import warning when blank

    def test_xy_mapped_to_xz(self):
        self.assertIn("XorR=5   Y=0   Z=0", self.txt)

    def test_active_dof_planar(self):
        # Out-of-plane DOFs deactivated globally; only UX, UZ, RY active.
        self.assertIn("UX=Yes   UY=No   UZ=Yes   RX=No   RY=Yes   RZ=No", self.txt)

    def test_restraints(self):
        # Only in-plane DOFs restrained (out-of-plane are inactive globally);
        # only supported joints get a restraint row.
        self.assertIn('Joint="N1"   U1=Yes   U2=No   U3=Yes   R1=No   R2=No   R3=No',
                      self.txt)
        self.assertIn('Joint="N2"   U1=No   U2=No   U3=Yes', self.txt)

    def test_point_load_mapping(self):
        # fy=-8 -> F3=-8 ; mz=2 -> M2=-2 (out-of-plane axis reversed by mapping).
        self.assertIn("F3=-8", self.txt)
        self.assertIn("M2=-2", self.txt)

    def test_distributed_load_dir_z(self):
        self.assertIn("Dir=Z", self.txt)
        self.assertIn("FOverLA=-10   FOverLB=-10", self.txt)

    def test_hinge_release(self):
        self.assertIn('Frame="E1"   M3I=No   M3J=Yes', self.txt)

    def test_load_patterns_defined(self):
        # Load cases become load patterns, and their twin Linear analysis cases
        # ARE emitted explicitly as LOAD CASE DEFINITIONS: on .s2k import SAP does
        # not auto-create a load case per pattern, so combinations referencing a
        # pattern name would otherwise fail to resolve.
        self.assertIn('LoadPat="G"', self.txt)
        self.assertIn('LoadPat="Q"', self.txt)
        self.assertIn('Case="G"   Type=LinStatic', self.txt)

    def test_linear_analysis_case(self):
        self.assertIn('Case="ULS"   Type=LinStatic', self.txt)
        self.assertIn('Case="ULS"   LoadType="Load pattern"   LoadName="G"   LoadSF=1.35',
                      self.txt)
        self.assertIn('Case="ULS"   LoadType="Load pattern"   LoadName="Q"   LoadSF=1.5',
                      self.txt)

    def test_pdelta_case(self):
        self.assertIn('Case="PD"   Type=NonStatic', self.txt)
        self.assertIn("GeoNonLin=P-Delta", self.txt)

    def test_combination_references_cases(self):
        self.assertIn('ComboName="CO"   ComboType="Linear Add"', self.txt)
        self.assertIn('CaseName="G"   ScaleFactor=1.35', self.txt)
        self.assertIn('CaseName="Q"   ScaleFactor=1.5', self.txt)


class TestS2kExtras(unittest.TestCase):
    def test_settlement_and_temperature(self):
        s = Structure2D()
        s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
        s.add_section("S", "M", b=0.3, h=0.6)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", 5.0, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("N1", "FIX")
        s.assign_support("N2", "FIX")
        s.add_load_case("LC")
        s.create_support_settlement("N2", "LC", uy=-0.01)
        s.add_temperature_load("E1", "LC", delta_t_uniform=20.0)
        txt = to_s2k(s)
        self.assertIn('TABLE:  "JOINT LOADS - GROUND DISPLACEMENT"', txt)
        self.assertIn("U3=-0.01", txt)
        self.assertIn('TABLE:  "FRAME LOADS - TEMPERATURE"', txt)
        self.assertIn("Temp=20", txt)


class TestCombosAfterCalculate(unittest.TestCase):
    """Combinations are stable across calculate() (no migration anymore) and
    still export."""

    def test_combo_exported_post_calculate(self):
        s = _model()
        before = dict(s.load_combinations[0].coefficients)
        s.calculate()
        self.assertEqual(s.load_combinations[0].coefficients, before)  # unchanged
        txt = to_s2k(s)
        self.assertIn('ComboName="CO"   ComboType="Linear Add"', txt)
        self.assertIn("ScaleFactor=1.35", txt)
        self.assertIn("ScaleFactor=1.5", txt)

    def test_load_case_twin_analysis_cases_emitted(self):
        # Each load case has a twin <id> Linear analysis case sharing the load
        # case name. These twins ARE emitted as LOAD CASE DEFINITIONS so that
        # combinations referencing a pattern name resolve on .s2k import (SAP
        # does not auto-create a load case per pattern).
        s = _model()
        txt = to_s2k(s)
        self.assertIn('Case="G"   Type=LinStatic', txt)
        self.assertIn('Case="Q"   Type=LinStatic', txt)
        self.assertIn('CaseName="G"', txt)
        self.assertIn('CaseName="Q"', txt)


class TestResponseSpectrum(unittest.TestCase):
    def setUp(self):
        s = Structure2D()
        s.add_material("C30", elastic_modulus=30e6, unit_weight=25.0)
        s.add_section("S1", "C30", b=0.3, h=0.6)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", 0.0, 3.0)
        s.add_bar_element("E1", "N1", "N2", "S1")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("N1", "FIX")
        s.add_load_case("G", self_weight_factor=1.0)
        s.add_analysis_case("MASS", "Mass", {"G": 1.0})
        s.add_analysis_case("MOD", "Modal", {}, modal_case_id="MASS", num_modes=6)
        s.add_spectral_function("EC8", damping=0.05,
                                points=[[0.0, 2.5], [0.5, 2.5], [2.0, 0.6]])
        s.add_analysis_case("SPEC", "Spectrum", {}, modal_case_id="MOD",
                            spectrum_id="EC8", combination_rule="CQC",
                            direction="X", damping=0.05)
        self.txt = to_s2k(s)

    def test_spectrum_case_defined(self):
        self.assertIn('Case="SPEC"   Type=LinRespSpec   ModalCase="MOD"', self.txt)

    def test_spectrum_general(self):
        self.assertIn('TABLE:  "CASE - RESPONSE SPECTRUM 1 - GENERAL"', self.txt)
        self.assertIn("ModalCombo=CQC", self.txt)
        self.assertIn("ConstDamp=0.05", self.txt)

    def test_spectrum_direction_assignment(self):
        self.assertIn('LoadName=U1   CoordSys=GLOBAL   Function="EC8"', self.txt)

    def test_spectrum_function_points(self):
        self.assertIn('TABLE:  "FUNCTION - RESPONSE SPECTRUM - USER"', self.txt)
        self.assertIn('Name="EC8"   Period=0   Accel=2.5   FuncDamp=0.05', self.txt)
        self.assertIn('Name="EC8"   Period=2   Accel=0.6', self.txt)

    def test_mass_source_from_mass_case(self):
        self.assertIn('TABLE:  "MASS SOURCE"', self.txt)
        self.assertIn('LoadPat="G"   Multiplier=1', self.txt)


class TestSprings(unittest.TestCase):
    def setUp(self):
        s = Structure2D()
        s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
        s.add_section("S", "M", b=0.3, h=0.6)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", 5.0, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_support("PIN", ux=True, uy=True)
        s.assign_support("N1", "PIN")
        s.add_node_spring("N2", kx=1000.0, ky=2000.0, kt=500.0)
        s.add_element_spring("E1", ky=5000.0)
        s.add_load_case("G")
        self.txt = to_s2k(s)

    def test_joint_spring(self):
        # kx -> U1, ky -> U3, kt -> R2.
        self.assertIn('TABLE:  "JOINT SPRING ASSIGNMENTS 1 - UNCOUPLED"', self.txt)
        self.assertIn('Joint="N2"   CoordSys=GLOBAL   U1=1000   U2=0   U3=2000'
                      '   R1=0   R2=500   R3=0', self.txt)

    def test_frame_line_spring(self):
        # ky on a horizontal element -> local axis 2 (in-plane transverse).
        self.assertIn('TABLE:  "FRAME SPRING ASSIGNMENTS"', self.txt)
        self.assertIn('Frame="E1"   Type=Simple   Stiffness=5000   '
                      'SimpleType="Tension and Compression"   '
                      'Dir1Type="Object Axes"   Dir=2', self.txt)


class TestNonLinearAndSpringModes(unittest.TestCase):
    """NonLinear cases and unilateral element springs in export + round-trip."""

    def _model(self):
        s = Structure2D()
        s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
        s.add_section("S", "M", b=0.3, h=0.6)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", 5.0, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_support("PIN", ux=True, uy=True)
        s.assign_support("N1", "PIN")
        s.add_element_spring("E1", ky=5000.0, coord_sys="local",
                             mode_y="compression")
        s.add_load_case("G", self_weight_factor=1.0)
        s.add_analysis_case("NL", "NonLinear", {"G": 1.0})
        s.add_analysis_case("PD", "GeometricNonlinear", {"G": 1.0})
        return s

    def test_nonlinear_case_is_nonstatic_without_pdelta(self):
        txt = to_s2k(self._model())
        self.assertIn('Case="NL"   Type=NonStatic', txt)
        # NonLinear → GeoNonLin None; P-Delta case → GeoNonLin P-Delta
        self.assertIn('Case="NL"   GeoNonLin=None', txt)
        self.assertIn('Case="PD"   GeoNonLin=P-Delta', txt)

    def test_element_spring_simpletype_from_mode(self):
        txt = to_s2k(self._model())
        self.assertIn('SimpleType="Compression Only"', txt)

    def test_roundtrip_case_types_and_spring_mode(self):
        from xdfem2d.sap2000_io import from_s2k
        s2 = from_s2k(to_s2k(self._model()))
        self.assertEqual(s2.analysis_cases_by_id["NL"].analysis_type, "NonLinear")
        self.assertEqual(s2.analysis_cases_by_id["PD"].analysis_type,
                         "GeometricNonlinear")
        self.assertEqual(s2.analysis_cases_by_id["NL"].coefficients, {"G": 1.0})
        es = s2.element_springs["E1"]
        self.assertEqual(es.mode_y, "compression")
        self.assertAlmostEqual(es.ky, 5000.0)


class TestConcreteEC2Design(unittest.TestCase):
    def _model(self):
        s = Structure2D()
        s.add_material("C30", elastic_modulus=33e6, unit_weight=25.0)
        s.add_section("B", "C30", b=0.3, h=0.6, section_type="Concrete",
                      rc_cover=0.04)
        s.add_node("N1", 0.0, 0.0); s.add_node("N2", 6.0, 0.0)
        s.add_bar_element("E1", "N1", "N2", "B", rc_design=True)
        s.add_support("PIN", ux=True, uy=True); s.assign_support("N1", "PIN")
        s.add_support("ROL", uy=True); s.assign_support("N2", "ROL")
        s.add_concrete_material("C30", "C30/37", "A500",
                                gamma_c=1.5, gamma_s=1.15, alpha_cc=0.85)
        s.add_load_case("G", self_weight_factor=1.0)
        return s

    def test_ec2_code_and_tables(self):
        txt = to_s2k(self._model())
        self.assertIn('ConcCode="Eurocode 2-2004"', txt)
        self.assertIn('TABLE:  "MATERIAL PROPERTIES 03B - CONCRETE DATA"', txt)
        self.assertIn("Fc=30000", txt)                 # 30 MPa → 30000 kN/m²
        self.assertIn('TABLE:  "MATERIAL PROPERTIES 03E - REBAR DATA"', txt)
        self.assertIn("Fy=500000", txt)                # 500 MPa
        self.assertIn('TABLE:  "FRAME DESIGN PROCEDURES"', txt)
        self.assertIn('DesignProc="Concrete"', txt)
        # Concrete section rebar/cover data (current table name and fields).
        self.assertIn('TABLE:  "FRAME SECTION PROPERTIES 03 - CONCRETE BEAM"', txt)
        self.assertIn("TopCover=0.04", txt)
        self.assertIn('PREFERENCES - CONCRETE DESIGN - EUROCODE 2-2004', txt)
        self.assertIn("AlphaCC=0.85", txt)

    def test_roundtrip_concrete_material(self):
        from xdfem2d.sap2000_io import from_s2k
        s2 = from_s2k(to_s2k(self._model()))
        self.assertIn("C30", s2.concrete_materials)
        cm = s2.concrete_materials["C30"]
        self.assertEqual(cm.concrete_class, "C30/37")
        self.assertEqual(cm.steel_class, "A500")
        self.assertAlmostEqual(cm.alpha_cc, 0.85)
        self.assertTrue(s2.bar_elements_by_id["E1"].rc_design)


def _plate_model():
    """A minimal slab: two DKT-domain triangles, a boundary support and a
    transverse point load — for the plate (X–Y) export orientation."""
    s = Structure2D(domain='plate')
    s.add_material("C30", elastic_modulus=30e6, unit_weight=25.0, poisson=0.2)
    s.add_plate_section("SLAB", "C30", thickness=0.20)
    for nid, (x, y) in {"n1": (0, 0), "n2": (4, 0), "n3": (4, 3),
                        "n4": (0, 3)}.items():
        s.add_node(nid, x, y)
    s.add_tri_element("T1", "n1", "n2", "n3", "SLAB")
    s.add_tri_element("T2", "n1", "n3", "n4", "SLAB")
    s.add_support("SS", w=True); s.assign_support("n1", "SS")
    s.assign_support("n2", "SS")
    s.add_node_spring("n3", kz=500.0)          # Winkler
    s.add_load_case("LC")
    s.add_point_load("n3", "LC", fz=-30.0)
    return s


class TestPlateDomainExport(unittest.TestCase):
    """dev/sap2000_domain_io.md Phase 1: a plate model is written in SAP's
    horizontal X–Y plane with the out-of-plane DOFs (UZ/RX/RY) active, so its
    domain is self-identifying on re-import."""

    def setUp(self):
        self.txt = to_s2k(_plate_model())

    def test_active_dof_is_plate(self):
        line = next(l for l in self.txt.splitlines() if "UX=" in l)
        self.assertIn("UX=No", line)
        self.assertIn("UZ=Yes", line)
        self.assertIn("RX=Yes", line)
        self.assertIn("RY=Yes", line)

    def test_joints_in_the_xy_plane(self):
        # A plate joint keeps y in Y and zeroes Z (X–Y plane).
        line = next(l for l in self.txt.splitlines()
                    if 'Joint="n3"' in l and "XorR" in l)
        self.assertIn("Y=3", line)
        self.assertIn("Z=0", line)

    def test_transverse_load_is_F3(self):
        line = next(l for l in self.txt.splitlines()
                    if "LoadPat=" in l and "F3=" in l)
        self.assertIn("F3=-30", line)

    def test_support_restrains_U3(self):
        line = next(l for l in self.txt.splitlines()
                    if 'Joint="n1"' in l and "U3=" in l and "R1=" in l)
        self.assertIn("U3=Yes", line)     # w restrained

    def test_mitc_section_is_thick_shell(self):
        # MITC3/MITC4 are thick-plate (transverse-shear) elements.
        line = next(l for l in self.txt.splitlines()
                    if 'Section="SLAB"' in l and "AreaType=" in l)
        self.assertIn("Type=Shell-Thick", line)

    def test_dkt_section_is_thin_shell(self):
        s = Structure2D(domain='plate')
        s.add_material("C30", elastic_modulus=30e6, unit_weight=25.0, poisson=0.2)
        s.add_plate_section("SLAB", "C30", thickness=0.20, formulation="DKT")
        for nid, (x, y) in {"n1": (0, 0), "n2": (4, 0), "n3": (4, 3)}.items():
            s.add_node(nid, x, y)
        s.add_tri_element("T1", "n1", "n2", "n3", "SLAB")
        txt = to_s2k(s)
        line = next(l for l in txt.splitlines()
                    if 'Section="SLAB"' in l and "AreaType=" in l)
        self.assertIn("Type=Shell-Thin", line)

    def test_shell_type_roundtrips_formulation(self):
        from xdfem2d.sap2000_io import from_s2k

        def _slab(form):
            s = Structure2D(domain='plate')
            s.add_material("C30", elastic_modulus=30e6, unit_weight=25.0,
                           poisson=0.2)
            s.add_plate_section("SLAB", "C30", thickness=0.2, formulation=form)
            for nid, (x, y) in {"n1": (0, 0), "n2": (4, 0),
                                "n3": (4, 3)}.items():
                s.add_node(nid, x, y)
            s.add_tri_element("T1", "n1", "n2", "n3", "SLAB")
            return s

        back_mitc = from_s2k(to_s2k(_slab("MITC3")))
        self.assertEqual(back_mitc.tri_sections["SLAB"].formulation, "MITC3")
        back_dkt = from_s2k(to_s2k(_slab("DKT")))
        self.assertEqual(back_dkt.tri_sections["SLAB"].formulation, "DKT")

    def test_plane_export_is_unchanged(self):
        # The plane path must stay byte-for-byte as before (guard against the
        # domain refactor moving it). A plane model still writes X–Z + UX/UZ/RY.
        plane = to_s2k(_model())
        dof = next(l for l in plane.splitlines() if "UX=" in l)
        self.assertIn("UX=Yes", dof)
        self.assertIn("RX=No", dof)
        joint = next(l for l in plane.splitlines()
                     if "Joint=" in l and "XorR" in l)
        self.assertIn("Y=0", joint)


class TestQuadAreaExport(unittest.TestCase):
    """dev/sap2000_domain_io.md Phase 2: quads export as 4-joint SAP area
    objects alongside the 3-joint triangles, with paired 'Panel' sections
    written once."""

    def setUp(self):
        s = Structure2D(domain='plate')
        s.add_material("C30", elastic_modulus=30e6, unit_weight=25.0,
                       poisson=0.2)
        s.add_plate_section("SLAB", "C30", thickness=0.20)             # tri
        s.add_quad_section("SLAB", "C30", thickness=0.20,
                           formulation="MITC4")                        # paired
        for nid, (x, y) in {"n1": (0, 0), "n2": (1, 0), "n3": (1, 1),
                            "n4": (0, 1), "n5": (2, 0), "n6": (2, 1)}.items():
            s.add_node(nid, x, y)
        s.add_quad_element("Q1", "n1", "n2", "n3", "n4", "SLAB")
        s.add_tri_element("T1", "n2", "n5", "n6", "SLAB")
        s.add_load_case("LC")
        s.add_quad_temperature_load("Q1", "LC", dt_i=5, dt_j=5, dt_k=5, dt_l=5)
        self.txt = to_s2k(s)

    def test_quad_is_a_four_joint_area(self):
        line = next(l for l in self.txt.splitlines()
                    if 'Area="Q1"' in l and "NumJoints" in l)
        self.assertIn("NumJoints=4", line)
        self.assertIn('Joint4="n4"', line)

    def test_triangle_is_still_three_joint(self):
        line = next(l for l in self.txt.splitlines()
                    if 'Area="T1"' in l and "NumJoints" in l)
        self.assertIn("NumJoints=3", line)

    def test_paired_section_emitted_once(self):
        rows = [l for l in self.txt.splitlines()
                if 'Section="SLAB"' in l and "Thickness" in l]
        self.assertEqual(len(rows), 1)

    def test_quad_temperature_exported(self):
        line = next(l for l in self.txt.splitlines()
                    if 'Area="Q1"' in l and "Temperature" in l)
        self.assertIn("Temp=5", line)


class TestDomainAwareImport(unittest.TestCase):
    """dev/sap2000_domain_io.md Phase 3: from_s2k infers plane vs plate from the
    active-DOF table (or takes an explicit domain=), imports 4-joint areas as
    real quads, and maps geometry/DOFs/loads to the resolved domain."""

    def _plate_slab(self):
        s = Structure2D(domain='plate')
        s.add_material("C30", elastic_modulus=33e6, unit_weight=25.0,
                       poisson=0.2)
        s.add_plate_section("SLAB", "C30", thickness=0.2)
        s.add_quad_section("SLAB", "C30", thickness=0.2, formulation="MITC4")
        for nid, (x, y) in {"n1": (0, 0), "n2": (1, 0), "n3": (1, 1),
                            "n4": (0, 1), "n5": (2, 0), "n6": (2, 1)}.items():
            s.add_node(nid, x, y)
        s.add_quad_element("Q1", "n1", "n2", "n3", "n4", "SLAB")
        s.add_tri_element("T1", "n2", "n5", "n6", "SLAB")
        s.add_support("SS", w=True); s.assign_support("n1", "SS")
        s.add_load_case("LC"); s.add_point_load("n3", "LC", fz=-30.0)
        return s

    def test_plate_roundtrip_infers_plate(self):
        from xdfem2d.sap2000_io import from_s2k
        back = from_s2k(to_s2k(self._plate_slab()))
        self.assertEqual(back.domain, "plate")

    def test_quad_imported_as_a_real_quad(self):
        from xdfem2d.sap2000_io import from_s2k
        back = from_s2k(to_s2k(self._plate_slab()))
        self.assertIn("Q1", back.quad_elements_by_id)
        q = back.quad_elements_by_id["Q1"]
        self.assertEqual([q.node_i, q.node_j, q.node_k, q.node_l],
                         ["n1", "n2", "n3", "n4"])
        self.assertEqual(back.quad_sections["SLAB"].formulation, "MITC4")
        self.assertIn("T1", back.tri_elements_by_id)   # 3-joint stays a triangle

    def test_plate_geometry_and_load_mapping(self):
        from xdfem2d.sap2000_io import from_s2k
        back = from_s2k(to_s2k(self._plate_slab()))
        self.assertAlmostEqual(back.nodes["n3"].y, 1.0)      # X–Y plane
        self.assertAlmostEqual(back.point_loads[0].fx, -30.0)  # fz slot

    def test_plane_roundtrip_infers_plane_and_qm6(self):
        from xdfem2d.sap2000_io import from_s2k
        s = Structure2D(domain='plane')
        s.add_material("M", elastic_modulus=30e6, unit_weight=0.0, poisson=0.2)
        s.add_quad_section("W", "M", thickness=0.5, formulation="QM6")
        for nid, (x, y) in {"a": (0, 0), "b": (2, 0), "c": (2, 1),
                            "d": (0, 1)}.items():
            s.add_node(nid, x, y)
        s.add_quad_element("Q1", "a", "b", "c", "d", "W")
        back = from_s2k(to_s2k(s))
        self.assertEqual(back.domain, "plane")
        self.assertEqual(back.quad_sections["W"].formulation, "QM6")
        self.assertAlmostEqual(back.nodes["c"].y, 1.0)       # X–Z plane

    def test_ambiguous_raises_and_explicit_domain_works(self):
        from xdfem2d.sap2000_io import from_s2k, AmbiguousDomainError
        txt = to_s2k(self._plate_slab())
        amb = "\n".join(l for l in txt.splitlines()
                        if "ACTIVE DEGREES" not in l and "UX=" not in l)
        with self.assertRaises(AmbiguousDomainError):
            from_s2k(amb)
        self.assertEqual(from_s2k(amb, domain="plate").domain, "plate")


class TestSaveS2k(unittest.TestCase):
    def test_writes_file(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "model.s2k")
            save_s2k(_model(), path)
            self.assertTrue(os.path.isfile(path))
            with open(path, encoding="utf-8") as f:
                content = f.read()
        self.assertIn('TABLE:  "JOINT COORDINATES"', content)


if __name__ == "__main__":
    unittest.main(verbosity=2)
