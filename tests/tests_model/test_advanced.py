"""Advanced-feature tests: support settlements, self-weight, local-axis loads,
inclined elements, modal/mass analysis, and load-combination envelopes.

Each case is checked against a closed-form result or an exact physical
invariant, extending the coverage of the core validation suite.
"""
import math
import unittest

from context import Structure2D, rect_inertia, rect_area, assert_close, midspan_moment

E = 30e6  # kN/m²


def _mat_sec(s, b=0.3, h=0.6, gamma=0.0):
    s.add_material("M", elastic_modulus=E, unit_weight=gamma)
    s.add_section("S", "M", b=b, h=h)
    return rect_inertia(b, h), rect_area(b, h)


class TestSupportSettlement(unittest.TestCase):
    """Clamped-clamped beam, one end forced down by Δ (no external load)."""

    def setUp(self):
        self.L, self.D = 4.0, 0.01
        s = Structure2D()
        self.I, _ = _mat_sec(s)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", self.L, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("N1", "FIX")
        s.assign_support("N2", "FIX")
        s.add_load_case("LC")
        s.create_support_settlement("N2", "LC", uy=self.D)
        self.r = s.calculate()

    def test_induced_shear(self):
        V = 12.0 * E * self.I * self.D / self.L ** 3  # 12EIΔ/L³
        assert_close(self, abs(self.r["reactions"]["LC"]["N1"][1]), V,
                     msg="settlement-induced shear")

    def test_induced_moment(self):
        M = 6.0 * E * self.I * self.D / self.L ** 2   # 6EIΔ/L²
        assert_close(self, abs(self.r["reactions"]["LC"]["N1"][2]), M,
                     msg="settlement-induced moment")

    def test_prescribed_displacement_applied(self):
        assert_close(self, self.r["displacements"]["LC"]["N2"][1], self.D,
                     msg="prescribed settlement at N2")


class TestSelfWeight(unittest.TestCase):
    """A load case with self_weight_factor applies w = γ·A per metre."""

    def test_self_weight_reactions(self):
        gamma, L = 25.0, 5.0
        s = Structure2D()
        _, A = _mat_sec(s, gamma=gamma)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", L, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_support("PIN", ux=True, uy=True)
        s.add_support("ROLLER", ux=False, uy=True)
        s.assign_support("N1", "PIN")
        s.assign_support("N2", "ROLLER")
        s.add_load_case("SW", self_weight_factor=1.0)
        r = s.calculate()
        w = gamma * A                       # kN/m
        assert_close(self, r["reactions"]["SW"]["N1"][1], w * L / 2.0,
                     msg="self-weight reaction")
        assert_close(self, midspan_moment(r, "SW", "E1"), w * L ** 2 / 8.0,
                     msg="self-weight mid-span moment")


class TestLocalAxisLoad(unittest.TestCase):
    """A load given in element-local axes must transform correctly to global.

    A vertical cantilever column with a *local* transverse uniform load behaves
    like a cantilever with a horizontal load: tip drift qL⁴/8EI, base moment
    qL²/2. This exercises the local→global load transformation.
    """

    def setUp(self):
        self.q, self.L = 6.0, 4.0
        s = Structure2D()
        self.I, _ = _mat_sec(s)
        s.add_node("N1", 0.0, 0.0)       # base
        s.add_node("N2", 0.0, self.L)    # top (column points +Y)
        s.add_bar_element("C1", "N1", "N2", "S")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("N1", "FIX")
        s.add_load_case("LC")
        # Local transverse load on the column (acts horizontally in global X).
        s.add_distributed_load("C1", "LC", fye=self.q, fyd=self.q, coord_sys="local")
        self.r = s.calculate()

    def test_tip_drift(self):
        drift = self.q * self.L ** 4 / (8.0 * E * self.I)  # qL⁴/8EI
        assert_close(self, abs(self.r["displacements"]["LC"]["N2"][0]), drift,
                     msg="tip horizontal drift")

    def test_base_moment(self):
        M = self.q * self.L ** 2 / 2.0
        assert_close(self, abs(self.r["reactions"]["LC"]["N1"][2]), M,
                     msg="base moment")

    def test_matches_global_equivalent(self):
        """For this column (axis +Y), the local +y axis points to global −X, so
        a local fy=+q load equals a global fx=−q load — same drift, same sign."""
        s = Structure2D()
        _mat_sec(s)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", 0.0, self.L)
        s.add_bar_element("C1", "N1", "N2", "S")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("N1", "FIX")
        s.add_load_case("LC")
        s.add_distributed_load("C1", "LC", fxe=-self.q, fxd=-self.q, coord_sys="global")
        r2 = s.calculate()
        assert_close(self, r2["displacements"]["LC"]["N2"][0],
                     self.r["displacements"]["LC"]["N2"][0],
                     msg="local vs global equivalence")


class TestInclinedAxialBar(unittest.TestCase):
    """Axial force in a 45° bar loaded along its own axis."""

    def test_axial_force_projection(self):
        P, Lproj = 100.0, 3.0          # P along the bar; geometry at 45°
        s = Structure2D()
        _, A = _mat_sec(s)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", Lproj, Lproj, )   # 45° diagonal
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("N1", "FIX")
        # Apply load along the bar axis: components (P/√2, P/√2).
        comp = P / math.sqrt(2.0)
        # Restrain the transverse and rotational DOFs at the free node so only
        # the axial DOF responds (pure bar behaviour).
        s.add_load_case("LC")
        s.add_point_load("N2", "LC", fx=comp, fy=comp)
        # Pin rotation + transverse via a roller aligned... simplest: fix tz only
        s.add_support("NOROT", ux=False, uy=False, tz=True)
        s.assign_support("N2", "NOROT")
        r = s.calculate()
        # Axial force magnitude should equal the applied axial component resultant P.
        assert_close(self, abs(r["element_forces"]["LC"]["E1"]["i"][0]), P,
                     rel=1e-6, msg="axial force in inclined bar")


class TestModalAnalysis(unittest.TestCase):
    """Single-DOF cantilever oscillator: ω² = k/m with k = 3EI/L³."""

    def setUp(self):
        self.L, self.m = 4.0, 10.0
        s = Structure2D()
        self.I, _ = _mat_sec(s)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", self.L, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("N1", "FIX")
        s.add_load_case("LC")
        s.add_analysis_case("MASS", "Mass", {})
        s.add_nodal_mass("N2", "MASS", mx=self.m, my=self.m)
        s.add_analysis_case("MODAL", "Modal", {}, modal_case_id="MASS", num_modes=2)
        self.r = s.calculate()

    def test_total_mass(self):
        assert_close(self, self.r["analysis_cases"]["MASS"]["total_mass"], self.m,
                     msg="total mass")

    def test_fundamental_frequency(self):
        k = 3.0 * E * self.I / self.L ** 3      # tip transverse stiffness
        f = math.sqrt(k / self.m) / (2.0 * math.pi)
        mi = self.r["analysis_cases"]["MODAL"]["modal_info"]
        self.assertGreaterEqual(len(mi), 1)
        assert_close(self, mi[0]["frequency"], f, rel=1e-3,
                     msg="fundamental frequency")

    def test_period_frequency_consistency(self):
        mi = self.r["analysis_cases"]["MODAL"]["modal_info"][0]
        assert_close(self, mi["period"], 1.0 / mi["frequency"], rel=1e-9,
                     msg="T = 1/f")


class TestThreeDOFChain(unittest.TestCase):
    """Three-DOF axial mass-spring chain (fixed-free), an exact modal benchmark.

    Three equal masses m are linked by three identical axial bars (stiffness
    k = EA/L) and clamped at one end; the transverse and rotational DOFs of the
    free nodes are restrained so only the three axial DOFs are dynamic. The
    natural frequencies of such a uniform fixed-free chain are known in closed
    form:  ω²_j = 4(k/m)·sin²((2j-1)π/(2·(2N+1))) with N = 3  ⇒  denominator 14.
    """

    N = 3

    def setUp(self):
        self.E, self.b, self.h = E, 0.3, 0.6
        self.A = rect_area(self.b, self.h)
        self.L, self.m = 5.0, 100.0
        s = Structure2D()
        _mat_sec(s, b=self.b, h=self.h)
        for i in range(self.N + 1):
            s.add_node(f"N{i}", i * self.L, 0.0)
        for i in range(self.N):
            s.add_bar_element(f"E{i}", f"N{i}", f"N{i+1}", "S")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("N0", "FIX")
        s.add_support("AX", ux=False, uy=True, tz=True)   # axial DOF only
        for i in range(1, self.N + 1):
            s.assign_support(f"N{i}", "AX")
        s.add_load_case("LC")
        s.add_analysis_case("MASS", "Mass", {})
        for i in range(1, self.N + 1):
            s.add_nodal_mass(f"N{i}", "MASS", mx=self.m, my=self.m)
        s.add_analysis_case("MODAL", "Modal", {}, modal_case_id="MASS", num_modes=3)
        self.r = s.calculate()
        self.k = self.E * self.A / self.L
        self.modal = self.r["analysis_cases"]["MODAL"]["modal_info"]

    def _analytic_freqs(self):
        km = self.k / self.m
        w2 = [4.0 * km * math.sin((2*j - 1) * math.pi / (2 * (2*self.N + 1))) ** 2
              for j in range(1, self.N + 1)]
        return sorted(math.sqrt(w) / (2.0 * math.pi) for w in w2)

    def test_three_modes_found(self):
        self.assertEqual(len(self.modal), 3)

    def test_total_mass(self):
        assert_close(self, self.r["analysis_cases"]["MASS"]["total_mass"],
                     self.N * self.m, msg="total mass (per DOF direction)")

    def test_natural_frequencies(self):
        expected = self._analytic_freqs()
        got = sorted(mi["frequency"] for mi in self.modal)
        for j, (g, e) in enumerate(zip(got, expected), start=1):
            assert_close(self, g, e, rel=1e-6, msg=f"natural frequency mode {j}")

    def test_frequencies_ascending(self):
        freqs = [mi["frequency"] for mi in self.modal]
        self.assertEqual(freqs, sorted(freqs), "modes should be ordered by frequency")


class TestResponseSpectrum(unittest.TestCase):
    """Single-DOF cantilever under a response spectrum.

    For a one-mode system the peak response is exact: with a flat spectrum
    Sa = const, the tip displacement equals the spectral displacement
    Sd = Sa/ω², and the base shear equals m·Sa (= k·Sd).
    """

    def setUp(self):
        self.L, self.m, self.Sa = 4.0, 10.0, 5.0
        s = Structure2D()
        self.I, _ = _mat_sec(s)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", self.L, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("N1", "FIX")
        s.add_load_case("LC")
        s.add_analysis_case("MASS", "Mass", {})
        s.add_nodal_mass("N2", "MASS", mx=self.m, my=self.m)
        s.add_analysis_case("MODAL", "Modal", {}, modal_case_id="MASS", num_modes=1)
        # Flat (period-independent) design spectrum.
        s.add_spectral_function("SP", points=[[0.0, self.Sa], [10.0, self.Sa]])
        s.add_analysis_case("SPEC", "Spectrum", {}, modal_case_id="MODAL",
                            spectrum_id="SP", combination_rule="SRSS",
                            direction="Y", damping=0.05)
        self.r = s.calculate()
        self.k = 3.0 * E * self.I / self.L ** 3
        self.Sd = self.Sa / (self.k / self.m)   # Sa/ω²

    def test_is_spectrum_result(self):
        self.assertTrue(self.r["analysis_cases"]["SPEC"].get("is_spectrum"))

    def test_peak_tip_displacement(self):
        uy = self.r["analysis_cases"]["SPEC"]["displacements"]["N2"][1]
        assert_close(self, uy, self.Sd, rel=1e-6, msg="peak tip displacement = Sd")

    def test_base_shear(self):
        ry = self.r["analysis_cases"]["SPEC"]["reactions"]["N1"][1]
        assert_close(self, ry, self.m * self.Sa, rel=1e-6, msg="base shear = m·Sa")

    def test_spectrum_results_are_nonnegative(self):
        """Spectrum combination returns peak (absolute) values → never negative."""
        ef = self.r["analysis_cases"]["SPEC"]["element_forces"]["E1"]
        self.assertGreaterEqual(ef["i"][1], 0.0)


class TestSpectrumInterpolation(unittest.TestCase):
    """Scaling the spectral ordinate scales the linear-elastic response."""

    def _run(self, sa):
        L, m = 4.0, 10.0
        s = Structure2D()
        _mat_sec(s)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", L, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("N1", "FIX")
        s.add_load_case("LC")
        s.add_analysis_case("MASS", "Mass", {})
        s.add_nodal_mass("N2", "MASS", mx=m, my=m)
        s.add_analysis_case("MODAL", "Modal", {}, modal_case_id="MASS", num_modes=1)
        s.add_spectral_function("SP", points=[[0.0, sa], [10.0, sa]])
        s.add_analysis_case("SPEC", "Spectrum", {}, modal_case_id="MODAL",
                            spectrum_id="SP", combination_rule="SRSS",
                            direction="Y")
        return s.calculate()["analysis_cases"]["SPEC"]["reactions"]["N1"][1]

    def test_response_scales_linearly_with_sa(self):
        v1 = self._run(5.0)
        v2 = self._run(10.0)
        assert_close(self, v2, 2.0 * v1, rel=1e-6, msg="double Sa → double response")


class TestCQCRho(unittest.TestCase):
    """Unit tests for the CQC cross-correlation coefficient ρ_ij."""

    def setUp(self):
        from xdfem2d.solver import _cqc_rho
        self.rho = _cqc_rho

    def test_self_correlation_is_one(self):
        assert_close(self, self.rho(12.3, 12.3, 0.05), 1.0, msg="ρ(ω,ω)=1")

    def test_symmetry(self):
        assert_close(self, self.rho(8.0, 13.0, 0.05), self.rho(13.0, 8.0, 0.05),
                     msg="ρ symmetric")

    def test_bounded_unit_interval(self):
        for wi, wj in [(1, 3), (1, 1.05), (5, 5.2), (2, 20)]:
            v = self.rho(wi, wj, 0.05)
            self.assertGreaterEqual(v, 0.0)
            self.assertLessEqual(v, 1.0)

    def test_decreases_with_separation(self):
        close = self.rho(10.0, 10.5, 0.05)   # closely spaced
        far = self.rho(10.0, 30.0, 0.05)     # well separated
        self.assertGreater(close, far, "ρ should fall as modes separate")


class TestSpectrumCQC(unittest.TestCase):
    """CQC modal-combination rule on the 3-DOF axial chain."""

    def _chain(self, rule):
        E_, b, h, L, m, Sa = E, 0.3, 0.6, 5.0, 100.0, 5.0
        s = Structure2D()
        _mat_sec(s, b=b, h=h)
        for i in range(4):
            s.add_node(f"N{i}", i * L, 0.0)
        for i in range(3):
            s.add_bar_element(f"E{i}", f"N{i}", f"N{i+1}", "S")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("N0", "FIX")
        s.add_support("AX", ux=False, uy=True, tz=True)
        for i in (1, 2, 3):
            s.assign_support(f"N{i}", "AX")
        s.add_load_case("LC")
        s.add_analysis_case("MASS", "Mass", {})
        for i in (1, 2, 3):
            s.add_nodal_mass(f"N{i}", "MASS", mx=m, my=m)
        s.add_analysis_case("MODAL", "Modal", {}, modal_case_id="MASS", num_modes=3)
        s.add_spectral_function("SP", points=[[0.0, Sa], [10.0, Sa]])
        s.add_analysis_case("SPEC", "Spectrum", {}, modal_case_id="MODAL",
                            spectrum_id="SP", combination_rule=rule,
                            direction="X", damping=0.05)
        return s.calculate()["analysis_cases"]["SPEC"]["reactions"]["N0"][0]

    def test_cqc_runs_and_is_positive(self):
        self.assertGreater(self._chain("CQC"), 0.0)

    def test_cqc_close_to_srss_for_separated_modes(self):
        """With well-separated modes the off-diagonal CQC terms are tiny, so
        CQC must agree with SRSS to within a few percent (and never be smaller)."""
        srss = self._chain("SRSS")
        cqc = self._chain("CQC")
        self.assertGreaterEqual(cqc, srss * (1.0 - 1e-6))
        assert_close(self, cqc, srss, rel=0.05, msg="CQC ≈ SRSS (separated modes)")


class TestSpectrumSdofCQCequalsSRSS(unittest.TestCase):
    """For a single mode, CQC collapses to SRSS (ρ_11 = 1)."""

    def _sdof(self, rule):
        L, m, Sa = 4.0, 10.0, 5.0
        s = Structure2D()
        _mat_sec(s)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", L, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("N1", "FIX")
        s.add_load_case("LC")
        s.add_analysis_case("MASS", "Mass", {})
        s.add_nodal_mass("N2", "MASS", mx=m, my=m)
        s.add_analysis_case("MODAL", "Modal", {}, modal_case_id="MASS", num_modes=1)
        s.add_spectral_function("SP", points=[[0.0, Sa], [10.0, Sa]])
        s.add_analysis_case("SPEC", "Spectrum", {}, modal_case_id="MODAL",
                            spectrum_id="SP", combination_rule=rule, direction="Y")
        return s.calculate()["analysis_cases"]["SPEC"]["reactions"]["N1"][1]

    def test_equal(self):
        assert_close(self, self._sdof("CQC"), self._sdof("SRSS"), rel=1e-9,
                     msg="single-mode CQC equals SRSS")


class TestSRSSDecomposition(unittest.TestCase):
    """SRSS must equal √(Σ Rₖ²) of the individual modal responses.

    The flat-spectrum SRSS base reaction of the 3-DOF chain is compared against
    the root-sum-square of three single-mode responses, each isolated with a
    narrow triangular 'spike' spectrum centred on that mode's period.
    """

    Sa = 5.0

    def _chain(self, points):
        E_, b, h, L, m = E, 0.3, 0.6, 5.0, 100.0
        s = Structure2D()
        _mat_sec(s, b=b, h=h)
        for i in range(4):
            s.add_node(f"N{i}", i * L, 0.0)
        for i in range(3):
            s.add_bar_element(f"E{i}", f"N{i}", f"N{i+1}", "S")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("N0", "FIX")
        s.add_support("AX", ux=False, uy=True, tz=True)
        for i in (1, 2, 3):
            s.assign_support(f"N{i}", "AX")
        s.add_load_case("LC")
        s.add_analysis_case("MASS", "Mass", {})
        for i in (1, 2, 3):
            s.add_nodal_mass(f"N{i}", "MASS", mx=m, my=m)
        s.add_analysis_case("MODAL", "Modal", {}, modal_case_id="MASS", num_modes=3)
        s.add_spectral_function("SP", points=points)
        s.add_analysis_case("SPEC", "Spectrum", {}, modal_case_id="MODAL",
                            spectrum_id="SP", combination_rule="SRSS", direction="X")
        r = s.calculate()
        R = r["analysis_cases"]["SPEC"]["reactions"]["N0"][0]
        periods = [mi["period"] for mi in r["analysis_cases"]["MODAL"]["modal_info"]]
        return R, sorted(periods)

    def test_srss_equals_root_sum_square_of_modes(self):
        flat, periods = self._chain([[0.0, self.Sa], [10.0, self.Sa]])
        eps = min(periods[i+1] - periods[i] for i in range(len(periods)-1)) * 0.25
        per_mode = []
        for T in periods:
            spike = [[0.0, 0.0], [T - eps, 0.0], [T, self.Sa], [T + eps, 0.0], [10.0, 0.0]]
            per_mode.append(self._chain(spike)[0])
        rss = math.sqrt(sum(r * r for r in per_mode))
        assert_close(self, flat, rss, rel=1e-6,
                     msg="SRSS = √(ΣRₖ²) over isolated modes")


def _pdelta_column(P, w, nel=6, L=8.0, b=0.3, h=0.6, nonlinear=True):
    """Pinned-pinned column (axis +Y), axial load P (compression > 0) at the top
    and a lateral uniform load w; return the mid-height transverse displacement
    from either the linear or the P-Delta (GeometricNonlinear) analysis."""
    s = Structure2D()
    _mat_sec(s, b=b, h=h)
    for i in range(nel + 1):
        s.add_node(f"N{i}", 0.0, i * L / nel)
    for i in range(nel):
        s.add_bar_element(f"E{i}", f"N{i}", f"N{i+1}", "S")
    s.add_support("BOT", ux=True, uy=True)
    s.assign_support("N0", "BOT")
    s.add_support("TOP", ux=True, uy=False)        # transversely held, axially free
    s.assign_support(f"N{nel}", "TOP")
    s.add_load_case("LC")
    s.add_point_load(f"N{nel}", "LC", fy=-P)        # +P → downward → compression
    for i in range(nel):
        s.add_distributed_load(f"E{i}", "LC", fxe=w, fxd=w)
    mid = f"N{nel // 2}"
    if not nonlinear:
        return s.calculate()["displacements"]["LC"][mid][0]
    s.add_analysis_case("NL", "GeometricNonlinear", {"LC": 1.0},
                        max_iterations=200, tolerance=1e-9)
    return s.calculate()["analysis_cases"]["NL"]["displacements"][mid][0]


class TestGeometricNonlinear(unittest.TestCase):
    """P-Delta (ANLG / GeometricNonlinear) second-order analysis."""

    L, b, h = 8.0, 0.3, 0.6

    @property
    def Pcr(self):
        I = rect_inertia(self.b, self.h)
        return math.pi ** 2 * E * I / self.L ** 2     # Euler load, pinned-pinned

    def test_zero_axial_recovers_linear(self):
        nl = _pdelta_column(0.0, 1.0, L=self.L, b=self.b, h=self.h, nonlinear=True)
        lin = _pdelta_column(0.0, 1.0, L=self.L, b=self.b, h=self.h, nonlinear=False)
        assert_close(self, nl, lin, rel=1e-6, msg="ANLG with no axial = linear")

    def test_amplification_factor(self):
        """Mid-height drift amplified by ≈ 1/(1−P/Pcr) at P = 0.5·Pcr."""
        P = 0.5 * self.Pcr
        lin = _pdelta_column(P, 1.0, L=self.L, b=self.b, h=self.h, nonlinear=False)
        nl = _pdelta_column(P, 1.0, L=self.L, b=self.b, h=self.h, nonlinear=True)
        assert_close(self, nl / lin, 1.0 / (1.0 - 0.5), rel=5e-3,
                     msg="P-Delta amplification 1/(1−P/Pcr)")

    def test_compression_amplifies_tension_stiffens(self):
        P = 0.5 * self.Pcr
        comp = _pdelta_column(P, 1.0, L=self.L, b=self.b, h=self.h, nonlinear=True)
        tens = _pdelta_column(-P, 1.0, L=self.L, b=self.b, h=self.h, nonlinear=True)
        linear = _pdelta_column(0.0, 1.0, L=self.L, b=self.b, h=self.h, nonlinear=False)
        self.assertGreater(comp, linear, "compression should amplify drift")
        self.assertLess(tens, linear, "tension should stiffen (reduce drift)")


class TestCombinationRules(unittest.TestCase):
    """AbsSum (Σ|·|) and SRSS (√Σ(·)²) combination rules.

    Regression guard: these used to be silently linearised because direct
    load-case coefficients were migrated into linearly-added analysis cases.
    """

    def _beam(self, combo_type, q_sign=-1.0):
        s = Structure2D()
        _mat_sec(s)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", 5.0, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_support("PIN", ux=True, uy=True)
        s.add_support("ROLLER", ux=False, uy=True)
        s.assign_support("N1", "PIN")
        s.assign_support("N2", "ROLLER")
        s.add_load_case("G")
        s.add_load_case("Q")
        s.add_distributed_load("E1", "G", fye=-10.0, fyd=-10.0)         # ⇒ Ry(N2)=25
        s.add_point_load("N2", "Q", fy=8.0 * q_sign)    # ⇒ Ry(N2)=∓8
        s.add_load_combination("C", {"G": 1.35, "Q": 1.5}, combo_type=combo_type)
        return s.calculate()["combinations"]["C"]["reactions"]["N2"][1]

    def test_raizsoma_is_srss(self):
        got = self._beam("SRSS")
        expected = math.sqrt((1.35 * 25.0) ** 2 + (1.5 * 8.0) ** 2)
        assert_close(self, got, expected, msg="SRSS = SRSS of factored cases")

    def test_somasodulo_is_abs_sum(self):
        # Uplift Q (opposite sign) so |Σ| differs from the linear sum.
        got = self._beam("AbsSum", q_sign=+1.0)
        expected = abs(1.35 * 25.0) + abs(1.5 * 8.0)
        assert_close(self, got, expected, msg="AbsSum = Σ|factored cases|")

    def test_somasodulo_differs_from_linear(self):
        absum = self._beam("AbsSum", q_sign=+1.0)
        linear = self._beam("LinearSum", q_sign=+1.0)
        self.assertGreater(absum, linear,
                           "abs-sum must exceed the (partly cancelling) linear sum")


class TestSpectrumDirectionXY(unittest.TestCase):
    """Direction 'XY' combines the X and Y spectra by SRSS, component-wise."""

    def _spec(self, direction):
        L, m, Sa = 4.0, 10.0, 5.0
        s = Structure2D()
        _mat_sec(s)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", L, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("N1", "FIX")
        s.add_load_case("LC")
        s.add_analysis_case("MASS", "Mass", {})
        s.add_nodal_mass("N2", "MASS", mx=m, my=m)
        s.add_analysis_case("MODAL", "Modal", {}, modal_case_id="MASS", num_modes=2)
        s.add_spectral_function("SP", points=[[0.0, Sa], [10.0, Sa]])
        s.add_analysis_case("SPEC", "Spectrum", {}, modal_case_id="MODAL",
                            spectrum_id="SP", combination_rule="SRSS",
                            direction=direction)
        return s.calculate()["analysis_cases"]["SPEC"]["reactions"]["N1"]

    def test_xy_is_srss_of_x_and_y(self):
        rx = self._spec("X")
        ry = self._spec("Y")
        rxy = self._spec("XY")
        for k in range(3):
            assert_close(self, rxy[k], math.sqrt(rx[k] ** 2 + ry[k] ** 2),
                         rel=1e-6, abs_tol=1e-6, msg=f"XY component {k}")


class TestCombinationEnvelope(unittest.TestCase):
    """Envelope (Envelope) combination yields consistent max ≥ min bounds."""

    def setUp(self):
        s = Structure2D()
        _mat_sec(s)
        s.add_node("N1", 0.0, 0.0)
        s.add_node("N2", 5.0, 0.0)
        s.add_bar_element("E1", "N1", "N2", "S")
        s.add_support("PIN", ux=True, uy=True)
        s.add_support("ROLLER", ux=False, uy=True)
        s.assign_support("N1", "PIN")
        s.assign_support("N2", "ROLLER")
        s.add_load_case("G")
        s.add_load_case("Q")
        s.add_distributed_load("E1", "G", fye=-10.0, fyd=-10.0)
        s.add_point_load("N2", "Q", fy=-8.0)
        s.add_load_combination("ENV", {"G": 1.35, "Q": 1.5}, combo_type="Envelope")
        self.r = s.calculate()

    def test_envelope_max_ge_min(self):
        env = self.r["combinations"]["ENV"]["reactions"]
        mx = env["max"]["N1"][1]
        mn = env["min"]["N1"][1]
        self.assertGreaterEqual(mx, mn, "envelope max must be ≥ min")

    def test_envelope_brackets_each_case(self):
        # The factored G reaction must lie within [min, max] of the envelope.
        rg = self.r["reactions"]["G"]["N1"][1] * 1.35
        env = self.r["combinations"]["ENV"]["reactions"]
        self.assertLessEqual(env["min"]["N1"][1] - 1e-6, rg)
        self.assertGreaterEqual(env["max"]["N1"][1] + 1e-6, rg)


if __name__ == "__main__":
    unittest.main(verbosity=2)
