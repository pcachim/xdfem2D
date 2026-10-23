"""Validation of MODAL and RESPONSE-SPECTRUM analysis against closed forms.

Mirrors ``test_springs_validation.py``: the exact cases live in
``validation/modal_cases.py`` as ``Case``/``Quantity`` objects, and everything
that does not fit that mould — convergence families, effective-mass identities,
the deliberate quirks of the mass definition, and the spectrum chain — is
checked here.

What can be exact and what cannot is decided by the engine's lumped, diagonal
mass matrix (see the module docstring of ``modal_cases``):

  * mass concentrated at nodes  → frequencies exact, mesh-independent;
  * distributed mass            → O(h²), converging FROM BELOW;
  * DOFs without mass           → condensed out exactly (no spurious modes).

Run:  python -m pytest tests/tests_model/test_modal_validation.py
"""
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import context  # noqa: F401  (puts src/ on the path)

ROOT = Path(__file__).resolve().parent.parent.parent
_VAL = ROOT / "validation"
if str(_VAL) not in sys.path:
    sys.path.insert(0, str(_VAL))

import modal_cases as mc                   # noqa: E402
from xdfem2d import Structure2D            # noqa: E402


def _solve(build):
    s = build()
    return s, s.calculate()


def _rel_errors(freqs, targets):
    return [abs(f - t) / t for f, t in zip(freqs, targets)]


# ═══ A — exact cases ═══════════════════════════════════════════════════════
class TestExactModalCases(unittest.TestCase):
    """m-a1…m-a4: every quantity against its closed form."""

    def test_quantities(self):
        for c in mc.MODAL_CASES:
            struc, res = _solve(c.build)
            for q in c.quantities:
                with self.subTest(case=c.id, quantity=q.label):
                    a, m = q.analytical, q.measured(res, struc)
                    if not q.signed:
                        a, m = abs(a), abs(m)
                    tol = max(q.abs_tol, q.rel_tol * abs(a))
                    self.assertLessEqual(
                        abs(m - a), tol,
                        f"{c.id} — {q.label}: teórico {a:.8g} {q.unit}, "
                        f"MEF {m:.8g} {q.unit} (|Δ|={abs(m - a):.3e} > {tol:.3e})")

    def test_nodal_mass_makes_frequency_mesh_independent(self):
        """The whole mass being nodal, refining the member must NOT move f."""
        _s1, r1 = _solve(mc.build_m_a1)
        _s2, r2 = _solve(mc.build_m_a1_refined)
        self.assertAlmostEqual(mc.freq(r1, 1), mc.freq(r2, 1),
                               delta=1e-6 * mc.freq(r1, 1))

    def test_modes_are_ordered_and_orthogonal(self):
        """φᵢᵀMφⱼ = 0 for i ≠ j, on the 2-DOF cantilever where M is known."""
        struc, res = _solve(mc.build_m_a3)
        f = [m["frequency"] for m in mc.modes(res)]
        self.assertEqual(f, sorted(f), "as frequências devem vir por ordem")
        p1 = [mc.shape_uy(res, 1, n) for n in ("N1", "N2")]
        p2 = [mc.shape_uy(res, 2, n) for n in ("N1", "N2")]
        prod = sum(a * b * m for a, b, m in zip(p1, p2, mc.MA3_M))
        scale = sum(a * a * m for a, m in zip(p1, mc.MA3_M))
        self.assertLess(abs(prod) / scale, 1e-6, "modos não ortogonais em M")

    def test_rigid_block_rocking_is_exact_on_every_mesh(self):
        """With the rotary inertia m_i·h²/12 given, the rocking frequency of the
        DISCRETE block has a closed form and must be hit on every mesh."""
        for mid, build, lbl in mc.MA4_MESHES:
            _s, res = _solve(build)
            n = int(lbl)
            with self.subTest(mesh=lbl):
                self.assertAlmostEqual(mc.freq(res, 1), mc.ma4_f_rock(n),
                                       delta=1e-3 * mc.ma4_f_rock(n))
                self.assertAlmostEqual(mc.freq(res, 2), mc.MA4_F_HEAVE,
                                       delta=1e-3 * mc.MA4_F_HEAVE)

    def test_rocking_converges_to_the_degenerate_continuum_value(self):
        """As h → 0 the segment spin M·h²/12 vanishes and rocking merges with
        heave — the double eigenvalue ω² = kL/M of the continuum."""
        errs = []
        for mid, build, lbl in mc.MA4_MESHES:
            _s, res = _solve(build)
            errs.append(abs(mc.freq(res, 1) - mc.MA4_F_HEAVE) / mc.MA4_F_HEAVE)
        for a, b in zip(errs, errs[1:]):
            self.assertLess(b, a, f"convergência não monótona: {errs}")
        self.assertLess(errs[-1], 0.01)

    def test_rocking_without_rotary_inertia_is_unreliable(self):
        """Without ``mtz`` the rotation DOFs are massless.

        They used to be given a token mass (m_max·1e-8), which degraded the
        conditioning of K̃ by ~10⁸ and made the frequency drift with the
        block's rigidity and with the LAPACK build (19,62 Hz on one machine,
        19,69 Hz on another). They are now condensed out exactly, so the
        result is reproducible; the assertion stays loose because without
        rotary inertia the rocking mode no longer exists as such — it merges
        with heave, the degenerate continuum value."""
        _s, res = _solve(mc.build_m_a4_no_rotary)
        f = mc.freq(res, 2)
        self.assertLess(abs(f - mc.MA4_F_HEAVE) / mc.MA4_F_HEAVE, 0.10)
        # …and with mtz the same quantity is reproducible to 10⁻⁴:
        _s2, res2 = _solve(mc.build_m_a4)
        self.assertAlmostEqual(mc.freq(res2, 2), mc.MA4_F_HEAVE,
                               delta=1e-4 * mc.MA4_F_HEAVE)


# ═══ B — convergence families ══════════════════════════════════════════════
class _ConvergenceMixin:
    """Monotone convergence from below toward the Euler–Bernoulli frequencies."""

    def _check_family(self, meshes, targets, kind="y", skip=0, tol_finest=0.01,
                      strictly_below=True, mono_floor=0.0):
        errs_by_mesh = []
        for mid, build, lbl in meshes:
            _s, res = _solve(build)
            f = mc.freqs_of_kind(res, kind)[skip:skip + len(targets)]
            self.assertEqual(len(f), len(targets),
                             f"{mid}: modos {kind} insuficientes ({len(f)})")
            if strictly_below:
                for fi, t in zip(f, targets):
                    self.assertLess(fi, t * (1 + 1e-9),
                                    f"{mid}: massa concentrada não pode "
                                    f"sobrestimar ({fi:.5g} > {t:.5g})")
            errs_by_mesh.append(_rel_errors(f, targets))
        for a, b in zip(errs_by_mesh, errs_by_mesh[1:]):
            for ea, eb, i in zip(a, b, range(len(a))):
                if ea <= mono_floor:
                    continue        # already at the noise floor: nothing to see
                self.assertLess(eb, ea + 1e-12,
                                f"modo {i+1}: erro cresceu {ea:.3e} → {eb:.3e}")
        for i, e in enumerate(errs_by_mesh[-1]):
            self.assertLess(e, tol_finest, f"modo {i+1}: {e:.3%} na malha fina")
        return errs_by_mesh

    def _check_second_order(self, errs_by_mesh, mode=0, expected=4.0, band=2.0):
        """Halving h must divide the error by ≈ 4 (O(h²)).

        Only the finest pair is asserted: the coarse meshes are not yet in the
        asymptotic regime (the 4-element simply supported beam is already within
        0,03 % of the exact f₁, so its 'error ratio' is meaningless)."""
        ratios = [a[mode] / b[mode] for a, b in zip(errs_by_mesh,
                                                    errs_by_mesh[1:])]
        r = ratios[-1]
        self.assertGreater(r, expected / band, f"razão de erros {ratios}")
        self.assertLess(r, expected * band, f"razão de erros {ratios}")


class TestBeamFrequencies(_ConvergenceMixin, unittest.TestCase):

    def test_simply_supported(self):
        """f_n = (nπ/L)²·√(EI/m̄)/2π."""
        errs = self._check_family(mc.MB1_MESHES, mc.MB1_F, tol_finest=1e-3)
        # The rotations are massless and condensed out exactly, which leaves a
        # compact (Hermite) scheme: the error falls as h⁴ here (ratio ≈ 16-18
        # on every pair), not h². The cantilever and the axial bar below stay
        # at h² because of their free/lumped end.
        self._check_second_order(errs, mode=0, expected=16.0)

    def test_cantilever(self):
        """βL = 1,8751 / 4,6941 / 7,8548."""
        errs = self._check_family(mc.MB2_MESHES, mc.MB2_F, tol_finest=0.011)
        self._check_second_order(errs, mode=0)

    def test_axial_bar(self):
        """ω_n = (2n−1)π/2L·√(E/ρ) — the modes dominated by ux."""
        errs = self._check_family(mc.MB4_MESHES, mc.MB4_F, kind="x",
                                  tol_finest=1e-3)
        self._check_second_order(errs, mode=0)

    def test_beam_on_elastic_foundation(self):
        """ω_n² = [EI(nπ/L)⁴ + k]/m̄ — modal × molas de elemento.

        Aqui a convergência não é monotonamente por baixo: a rigidez de flexão
        (que a malha subestima) e a fundação lumped (que não o faz da mesma
        forma) contribuem em sentidos opostos, e o resultado fica a ~10⁻⁷ do
        valor exato pelos dois lados. Verifica-se o erro absoluto, não o sinal."""
        self._check_family(mc.MB5_MESHES, mc.MB5_F, tol_finest=1e-3,
                           strictly_below=False, mono_floor=1e-5)


class TestFreeFreeBeam(unittest.TestCase):
    """A model with no supports cannot be solved statically (K singular), so the
    free-free beam is modelled with springs soft enough that the rigid-body
    frequencies fall three orders below the first elastic one. Those rigid-body
    modes must then be clearly separated, and the elastic ones must converge to
    βL = 4,7300 / 7,8532."""

    def _split(self, res):
        f = [m["frequency"] for m in mc.modes(res)]
        elastic = [x for x in f if x > 1.0]
        rigid = [x for x in f if x <= 1.0]
        return rigid, elastic

    def test_rigid_body_modes_are_separated(self):
        for mid, build, lbl in mc.MB3_MESHES:
            _s, res = _solve(build)
            rigid, elastic = self._split(res)
            with self.subTest(mesh=lbl):
                self.assertTrue(elastic, "sem modos elásticos")
                self.assertTrue(all(r < 0.01 * elastic[0] for r in rigid),
                                f"modos de corpo rígido {rigid} vs "
                                f"f₁ = {elastic[0]:.4g}")

    def test_elastic_frequencies_converge(self):
        errs = []
        for mid, build, lbl in mc.MB3_MESHES:
            _s, res = _solve(build)
            _rigid, elastic = self._split(res)
            errs.append(_rel_errors(elastic[:len(mc.MB3_F)], mc.MB3_F))
        for a, b in zip(errs, errs[1:]):
            self.assertLess(b[0], a[0])
        self.assertLess(errs[-1][0], 0.01)


# ═══ C — effective mass and participation ══════════════════════════════════
class TestEffectiveMass(unittest.TestCase):

    def _y_modes(self, res):
        return [(m["frequency"], m["meff_y"])
                for k, m in enumerate(mc.modes(res), start=1)
                if mc.mode_kind(res, k) == "y"]

    def test_completeness(self):
        """Σ m_eff over ALL modes = the mass on the free DOFs, in both
        directions. The strongest single check of Γ and of the normalisation."""
        _s, res = _solve(mc.build_mb2_f)          # cantilever, 16 elem, 40 modes
        info = mc.modes(res)
        for key, total in (("meff_x", "total_mass_x"), ("meff_y", "total_mass_y")):
            with self.subTest(direction=key):
                self.assertAlmostEqual(sum(m[key] for m in info),
                                       mc.modal(res)[total],
                                       delta=1e-6 * mc.modal(res)[total])

    def test_cantilever_truncation_fractions(self):
        """0,6131 / 0,1883 / 0,0647 da massa total m̄·L."""
        _s, res = _solve(mc.build_mb2_f)
        meff = [m for _f, m in self._y_modes(res)][:3]
        for i, (m, frac) in enumerate(zip(meff, mc.MC_CANT_FRACTIONS)):
            with self.subTest(mode=i + 1):
                self.assertAlmostEqual(m, frac * mc.MC_TOTAL,
                                       delta=0.01 * frac * mc.MC_TOTAL)

    def test_simply_supported_odd_and_even_modes(self):
        """8/(n²π²) nos modos ímpares; exatamente 0 nos pares (antissimétricos)."""
        _s, res = _solve(mc.build_mb1_f)
        meff = [m for _f, m in self._y_modes(res)]
        self.assertAlmostEqual(meff[0], mc.MC_SS_FRACTIONS[0] * mc.MC_TOTAL,
                               delta=0.02 * mc.MC_TOTAL)
        self.assertAlmostEqual(meff[1], 0.0, delta=1e-6 * mc.MC_TOTAL)
        self.assertAlmostEqual(meff[2], mc.MC_SS_FRACTIONS[1] * mc.MC_TOTAL,
                               delta=0.02 * mc.MC_TOTAL)

    def test_percentages_are_relative_to_the_free_mass(self):
        """Documented subtlety: meff_%% uses the mass on the FREE DOFs, so on a
        simply supported beam the supported nodes' mass is excluded and the
        percentage drifts with the mesh even though m_eff converges."""
        _s, res = _solve(mc.build_mb1_f)
        total_free = mc.total_mass_y(res)
        self.assertLess(total_free, mc.MC_TOTAL)
        self.assertAlmostEqual(total_free,
                               mc.MC_TOTAL * (1 - 1.0 / 16), delta=1e-9)
        first = [m for m in mc.modes(res) if m["meff_y"] > 0][0]
        self.assertAlmostEqual(first["meff_y_pct"],
                               100 * first["meff_y"] / total_free, places=6)

    def test_model_mass_is_reported_alongside_the_free_mass(self):
        """The excluded mass must be visible, not merely absent: the results
        carry the model's total and the restrained part, so the EC8 ratio
        (90 % of the TOTAL mass) can be computed from the output."""
        _s, res = _solve(mc.build_mb1_f)
        m = mc.modal(res)
        self.assertAlmostEqual(m["model_mass_y"], mc.MC_TOTAL, delta=1e-9)
        # simply supported, 16 elements → the supports hold exactly 1/16
        self.assertAlmostEqual(m["restrained_mass_y"], mc.MC_TOTAL / 16,
                               delta=1e-9)
        self.assertAlmostEqual(m["total_mass_y"] + m["restrained_mass_y"],
                               m["model_mass_y"], delta=1e-9)
        # …and the same numbers are available straight from the Mass case
        mass = mc.modal(res, mc.MASS_CASE)
        self.assertAlmostEqual(mass["restrained_mass_y"],
                               m["restrained_mass_y"], delta=1e-9)

    def test_restrained_mass_raises_a_warning(self):
        """Above 2 % of the model mass the percentages stop being a usable
        stand-in for the EC8 criterion, and the results say so."""
        _s, res = _solve(mc.build_mb1_f)                 # 6,25 % apoiada
        self.assertTrue(any("massa livre" in w
                            for w in mc.modal(res)["warnings"]))

    def test_no_warning_when_all_the_mass_is_free(self):
        _s, res = _solve(mc.build_m_a1)
        m = mc.modal(res)
        self.assertEqual(m["warnings"], [])
        self.assertAlmostEqual(m["restrained_mass_y"], 0.0, places=12)
        self.assertAlmostEqual(m["model_mass_y"], m["total_mass_y"], places=9)


class TestMassOnSupportCheck(unittest.TestCase):
    """``model_check`` flags a concentrated mass typed onto a restrained DOF —
    it is silently excluded from the modal analysis, and is nearly always a
    modelling slip."""

    def _issues(self, struc):
        from xdfem2d.model_check import model_check
        return [i for i in model_check(struc) if i["type"] == "mass_on_support"]

    def test_clean_model_has_no_issue(self):
        self.assertEqual(self._issues(mc.build_m_a1()), [])

    def test_mass_on_a_fixed_node_is_flagged(self):
        s = mc.build_m_a1()
        s.add_nodal_mass("A", mc.MASS_CASE, mx=5.0, my=5.0)
        issues = self._issues(s)
        self.assertEqual(len(issues), 1)
        self.assertIn("A", issues[0]["items"])
        self.assertIn("x, y", issues[0]["msg"])

    def test_mass_on_a_free_direction_is_not_flagged(self):
        """A roller leaves ux free: a mass in x there is legitimate."""
        s = mc.build_m_a1()
        s.add_support("ROL", ux=False, uy=True)
        s.add_node("C", 2 * mc.MA1_L, 0)
        s.add_bar_element("E2", "B", "C", "S")
        s.assign_support("C", "ROL")
        s.add_nodal_mass("C", mc.MASS_CASE, mx=5.0)
        self.assertEqual(self._issues(s), [])


# ═══ D — how the mass is defined ═══════════════════════════════════════════
class TestMassDefinition(unittest.TestCase):
    """Deliberate behaviours of ``|Fy|/g``, pinned so they cannot drift."""

    def _f1(self, build):
        _s, res = _solve(build)
        return mc.freq(res, 1)

    def test_mass_from_load_case(self):
        """m = |Fy|/g, aplicada a X e a Y."""
        _s, res = _solve(mc.build_m_d_down)
        m = mc.modal(res, mc.MASS_CASE)["nodal_masses"]["B"]
        self.assertAlmostEqual(m["my"], mc.MD_P / mc.G, places=9)
        self.assertAlmostEqual(m["mx"], m["my"], places=12)

    def test_upward_load_gives_the_same_mass(self):
        self.assertAlmostEqual(self._f1(mc.build_m_d_down),
                               self._f1(mc.build_m_d_up), places=9)

    def test_horizontal_load_gives_no_mass(self):
        """Só Fy alimenta a massa — uma carga horizontal não conta."""
        _s = mc.build_m_d_horiz()
        res = _s.calculate()
        self.assertEqual(mc.modal(res, mc.MASS_CASE)["nodal_masses"], {})
        self.assertIn("error", mc.modal(res))          # massa total nula

    def test_mass_scaling_halves_the_frequency(self):
        """m ×4 → f ÷2 (a rigidez não muda)."""
        f1 = self._f1(mc.build_m_d_down)
        s = mc.build_m_d_down()
        s.analysis_cases_by_id[mc.MASS_CASE].coefficients["LC"] = 4.0
        f4 = mc.freq(s.calculate(), 1)
        self.assertAlmostEqual(f4, f1 / 2.0, delta=1e-9 * f1)

    def test_stiffness_scaling_doubles_the_frequency(self):
        """E ×4 → f ×2."""
        base = mc.freq(mc.build_m_a1().calculate(), 1)
        s = mc.build_m_a1()
        s.materials["C"].elastic_modulus = 4 * mc.E
        self.assertAlmostEqual(mc.freq(s.calculate(), 1), 2 * base,
                               delta=1e-9 * base)


class TestModalRobustness(unittest.TestCase):

    def test_spurious_modes_stay_far_above(self):
        """Massless rotational DOFs are condensed out, so no spurious mode at
        ~10⁴ × f₁ appears; the check holds for the regularised fallback too.
        Guard that nothing spurious sneaks into the useful range."""
        _s, res = _solve(mc.build_m_a1)
        f = [m["frequency"] for m in mc.modes(res)]
        physical = [x for x in f if x < 1000 * f[0]]
        spurious = [x for x in f if x >= 1000 * f[0]]
        self.assertEqual(len(physical), 2, "esperados 2 modos físicos (1 GL×2)")
        self.assertTrue(all(x > 1e4 * f[0] for x in spurious), f"{spurious}")

    def test_more_modes_requested_than_available(self):
        s = mc.build_m_a1()
        s.analysis_cases_by_id[mc.MODAL_CASE].num_modes = 99
        res = s.calculate()
        self.assertLessEqual(len(mc.modes(res)), 99)
        self.assertGreaterEqual(len(mc.modes(res)), 2)

    def test_zero_mass_is_reported(self):
        s = mc.build_m_a1()
        s.nodal_masses = []
        self.assertIn("error", mc.modal(s.calculate()))

    def test_frequencies_survive_the_x2d_roundtrip(self):
        import tempfile
        from xdfem2d import load_x2d, save_x2d
        s, res = _solve(mc.build_m_a3)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "m.x2d"
            save_x2d(s, None, p)
            s2, _, _ = load_x2d(p)
            res2 = s2.calculate()
        for k in (1, 2):
            self.assertAlmostEqual(mc.freq(res2, k), mc.freq(res, k),
                                   delta=1e-9 * mc.freq(res, k))


# ═══ E — response spectrum ═════════════════════════════════════════════════
class TestResponseSpectrum(unittest.TestCase):

    def test_flat_spectrum_equals_the_static_response(self):
        """Com Sa constante, Sd = Sa/ω² e Γ·Sd·φ colapsa na resposta estática
        sob F = m·Sa. É a verificação mais apertada da cadeia Mass→Modal→Espectro."""
        _s, res = _solve(mc.build_m_e1)
        ac = res["analysis_cases"]
        lin = ac["LIN"]["displacements"]["B"][1]
        spy = ac["SPY"]["displacements"]["B"][1]
        self.assertAlmostEqual(spy, abs(lin), delta=1e-9 * abs(lin))
        self.assertAlmostEqual(spy, mc.ME1_UY, delta=1e-9 * mc.ME1_UY)
        for comp in (1, 2):
            self.assertAlmostEqual(ac["SPY"]["reactions"]["A"][comp],
                                   abs(ac["LIN"]["reactions"]["A"][comp]),
                                   delta=1e-6 * abs(ac["LIN"]["reactions"]["A"][comp]))

    def test_direction_x_uses_the_axial_mode(self):
        _s, res = _solve(mc.build_m_e1)
        ux = res["analysis_cases"]["SPX"]["displacements"]["B"][0]
        self.assertAlmostEqual(ux, mc.ME1_UX, delta=1e-6 * mc.ME1_UX)

    def test_direction_xy_is_the_srss_of_x_and_y(self):
        _s, res = _solve(mc.build_m_e1)
        ac = res["analysis_cases"]
        for comp in (0, 1):
            x = ac["SPX"]["displacements"]["B"][comp]
            y = ac["SPY"]["displacements"]["B"][comp]
            xy = ac["SPXY"]["displacements"]["B"][comp]
            self.assertAlmostEqual(xy, math.hypot(x, y),
                                   delta=1e-6 * max(1e-12, abs(xy)))

    def test_cqc_matches_srss_for_well_separated_modes(self):
        """f₂/f₁ = 5,15 e ξ = 5 % → correlação cruzada desprezável."""
        _s, res = _solve(mc.build_m_e2)
        ac = res["analysis_cases"]
        for nid in ("N1", "N2"):
            srss = ac["SRSS"]["displacements"][nid][1]
            cqc = ac["CQC"]["displacements"][nid][1]
            self.assertAlmostEqual(cqc, srss, delta=0.01 * abs(srss))

    def test_two_mode_srss_beats_each_mode_alone(self):
        """SRSS = √(ΣR_k²) ≥ qualquer contribuição isolada, e a resposta do 1.º
        modo domina (contém ~toda a massa efetiva)."""
        _s, res = _solve(mc.build_m_e2)
        srss = res["analysis_cases"]["SRSS"]["displacements"]["N2"][1]
        _s1, res1 = _solve(mc.build_m_e2)
        s1 = mc.build_m_e2()
        s1.analysis_cases_by_id[mc.MODAL_CASE].num_modes = 1
        one = s1.calculate()["analysis_cases"]["SRSS"]["displacements"]["N2"][1]
        self.assertGreater(srss, one)
        self.assertLess(srss - one, 0.05 * srss)

    def test_spectrum_is_linearly_interpolated(self):
        """Sa entre dois pontos do espectro = interpolação linear em T."""
        _s, res = _solve(mc.build_m_e3)
        T = mc.period(res, 1)
        self.assertGreater(T, 0.10); self.assertLess(T, 0.30)
        sa = mc.me3_expected_sa(T)
        expected = mc.ME_M * sa / mc.MA1_KB
        uy = res["analysis_cases"]["SPY"]["displacements"]["B"][1]
        self.assertAlmostEqual(uy, expected, delta=1e-6 * expected)


if __name__ == "__main__":
    unittest.main(verbosity=2)
