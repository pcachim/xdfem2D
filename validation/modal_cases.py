"""Canonical validation cases for MODAL (free-vibration) and RESPONSE-SPECTRUM
analysis — the dynamic counterpart of ``validation_cases.py``.

Shared by:
  * ``build_x2d.py``                              — writes one ``.x2d`` per case;
  * ``tests/tests_model/test_modal_validation.py`` — checks them.

Three properties of the engine shape every case here, and are worth stating up
front because they decide what can be checked exactly and what can only be
checked by convergence:

1. **The mass matrix is diagonal (lumped).** It is built from ``|Fy|/g`` of the
   load cases referenced by a Mass analysis case, plus the concentrated
   ``add_nodal_mass`` entries. A structure whose whole mass is concentrated at
   nodes is therefore reproduced EXACTLY; distributed mass converges as O(h²),
   and (lumped mass being softer in inertia terms) from BELOW.
2. **There is no rotary inertia** unless ``mtz`` is given explicitly. DOFs left
   without mass are condensed out exactly (static condensation, K_zz⁻¹), so
   they add no spurious modes and no conditioning loss. Only if K_zz is
   singular (massless DOFs forming a mechanism) does the solver fall back to
   a token mass ``eps = m_max·1e-8``, with spurious modes at ~10⁴–10⁵ × f₁.
3. **Mass comes from the ABSOLUTE value of Fy** and is applied equally to the X
   and Y DOFs. So an upward load contributes the same mass as a downward one,
   and a purely horizontal load contributes none. Deliberate; pinned by a test.

Units follow ``validation_cases``: m, kN, kN/m², plus t for mass and s for time
(1 t·m/s² = 1 kN, so a spectral acceleration in m/s² times a mass in t gives kN
directly).
"""
from __future__ import annotations

import math

import numpy as np

from validation_cases import (              # shared properties and machinery
    E, NU, B, H, A_BAR, I_BAR, Case, Quantity, MM, _new, _beam_material,
    _rigid_section, node_at,
)

G = 9.81                     # m/s², the solver's default
GAMMA_C = 25.0               # kN/m³ — gives the distributed mass of the members
MBAR = GAMMA_C * A_BAR / G   # t/m, distributed mass of the standard section

MASS_CASE = "MASS"
MODAL_CASE = "MOD"


# ── Result-extraction helpers ────────────────────────────────────────────────
def modal(res, ac=MODAL_CASE):
    return res["analysis_cases"][ac]

def modes(res, ac=MODAL_CASE):
    return modal(res, ac)["modal_info"]

def freq(res, k, ac=MODAL_CASE):
    """Frequency [Hz] of mode k (1-based, in the engine's own ordering)."""
    return modes(res, ac)[k - 1]["frequency"]

def period(res, k, ac=MODAL_CASE):
    return modes(res, ac)[k - 1]["period"]

def mode_kind(res, k, ac=MODAL_CASE):
    """'x' or 'y' — which translation dominates mode k's shape.

    Needed because the engine orders modes by frequency alone, so the axial
    modes of a horizontal member are interleaved with the bending ones (on the
    standard section the 3rd mode of a 6 m beam is axial, not flexural)."""
    shape = modal(res, ac)["mode_shapes"][k - 1]
    mx = max(abs(v[0]) for v in shape.values())
    my = max(abs(v[1]) for v in shape.values())
    return "x" if mx > my else "y"

def freqs_of_kind(res, kind, ac=MODAL_CASE):
    """Ascending frequencies of the modes dominated by ``kind`` ('x' or 'y')."""
    return [m["frequency"] for k, m in enumerate(modes(res, ac), start=1)
            if mode_kind(res, k, ac) == kind]

def meff_y(res, k, ac=MODAL_CASE):
    return modes(res, ac)[k - 1]["meff_y"]

def total_mass_y(res, ac=MODAL_CASE):
    """Mass on the FREE y DOFs — the reference the engine uses for meff_y_pct.

    Note this is NOT the total mass of the structure: the mass lumped on a
    restrained node is not free to move and is excluded. On a simply supported
    beam that is m̄·L·(1/n) of the total, which is why the percentages drift
    with the mesh while the absolute effective masses converge."""
    return modal(res, ac)["total_mass_y"]

def shape_uy(res, k, nid, ac=MODAL_CASE):
    return modal(res, ac)["mode_shapes"][k - 1][nid][1]


# ── Model-building helpers ───────────────────────────────────────────────────
def _mass_and_modal(s, num_modes=10, from_self_weight=False):
    """Attach the Mass + Modal analysis cases in the canonical way."""
    if from_self_weight:
        s.add_load_case("PP", self_weight_factor=1.0)
        s.add_analysis_case(MASS_CASE, "Mass", {"PP": 1.0})
    else:
        s.add_load_case("LC")
        s.add_analysis_case(MASS_CASE, "Mass", {})
    s.add_analysis_case(MODAL_CASE, "Modal", {},
                        modal_case_id=MASS_CASE, num_modes=num_modes)
    return s

def _heavy_material(s):
    """Standard section with a real unit weight, so the self-weight load case
    produces the distributed mass m̄ = γ·A/g."""
    s.add_material("C", E, GAMMA_C, poisson=NU)
    s.add_section("S", "C", b=B, h=H)

def _chain(s, n, L, section="S", ky=0.0, y=0.0):
    """n bar elements N0…Nn along y = const, optionally on a Winkler foundation."""
    for i in range(n + 1):
        s.add_node(f"N{i}", L * i / n, y)
    for i in range(n):
        eid = f"E{i}"
        s.add_bar_element(eid, f"N{i}", f"N{i+1}", section)
        if ky:
            s.add_element_spring(eid, ky=ky)
    return s


# ═══ A — exact cases (mass fully concentrated at nodes) ═════════════════════

# ── M-A1 — single degree of freedom ─────────────────────────────────────────
MA1_L = 3.0
MA1_M = 10.0                                  # t
MA1_KB = 3.0 * E * I_BAR / MA1_L ** 3         # bending tip stiffness
MA1_KA = E * A_BAR / MA1_L                    # axial stiffness
MA1_F_BEND = math.sqrt(MA1_KB / MA1_M) / (2 * math.pi)
MA1_F_AXIAL = math.sqrt(MA1_KA / MA1_M) / (2 * math.pi)

def build_m_a1():
    """Massless cantilever with a lumped mass at the tip — the reference case.
    f = √(3EI/mL³)/2π in bending and √(EA/mL)/2π in the axial direction, both
    exact and INDEPENDENT OF THE MESH (the only mass is nodal)."""
    s = _new(); _beam_material(s)
    s.add_node("A", 0, 0); s.add_node("B", MA1_L, 0)
    s.add_bar_element("E", "A", "B", "S")
    s.add_support("FIX", ux=True, uy=True, tz=True); s.assign_support("A", "FIX")
    _mass_and_modal(s, num_modes=4)
    s.add_nodal_mass("B", MASS_CASE, mx=MA1_M, my=MA1_M)
    return s

def build_m_a1_refined():
    """Same system, 8 elements — must give the SAME frequencies to 10⁻¹²."""
    s = _new(); _beam_material(s)
    _chain(s, 8, MA1_L)
    s.add_support("FIX", ux=True, uy=True, tz=True); s.assign_support("N0", "FIX")
    _mass_and_modal(s, num_modes=6)
    s.add_nodal_mass("N8", MASS_CASE, mx=MA1_M, my=MA1_M)
    return s


# ── M-A2 — the same mass with a node spring in parallel with the bar ───────
MA2_KS = MA1_KB                                     # spring = bar → k_eq = 2k
MA2_KEQ = MA2_KS + MA1_KB
MA2_F = math.sqrt(MA2_KEQ / MA1_M) / (2 * math.pi)

def build_m_a2():
    """Cantilever + tip spring to ground: the spring acts IN PARALLEL with the
    member, so k_eq = k_mola + 3EI/L³ and f = √(k_eq/m)/2π. Cross-checks that
    the node springs enter the eigenproblem's K, not just the static solve."""
    s = _new(); _beam_material(s)
    s.add_node("A", 0, 0); s.add_node("B", MA1_L, 0)
    s.add_bar_element("E", "A", "B", "S")
    s.add_support("FIX", ux=True, uy=True, tz=True); s.assign_support("A", "FIX")
    s.add_node_spring("B", ky=MA2_KS)
    _mass_and_modal(s, num_modes=4)
    s.add_nodal_mass("B", MASS_CASE, my=MA1_M)
    return s


# ── M-A3 — two masses on a cantilever (exact 2-DOF eigenproblem) ───────────
MA3_L = 6.0
MA3_X = (3.0, 6.0)          # abscissas of the masses
MA3_M = (8.0, 4.0)          # t

def _cantilever_flexibility(xs):
    """f_ij = x²(3y − x)/6EI with x = min(xi,xj), y = max(xi,xj)."""
    F = np.zeros((len(xs), len(xs)))
    for i, xi in enumerate(xs):
        for j, xj in enumerate(xs):
            x, y = min(xi, xj), max(xi, xj)
            F[i, j] = x * x * (3 * y - x) / (6.0 * E * I_BAR)
    return F

_MA3_W2 = np.sort(1.0 / np.linalg.eigvals(
    _cantilever_flexibility(MA3_X) @ np.diag(MA3_M)).real)
MA3_F = [float(math.sqrt(w) / (2 * math.pi)) for w in _MA3_W2]

def build_m_a3():
    """Massless cantilever carrying two lumped masses. The reference is the
    exact 2-DOF eigenproblem built from the cantilever flexibility matrix —
    derived here independently of the FE, so it is a real cross-check."""
    s = _new(); _beam_material(s)
    s.add_node("N0", 0, 0)
    s.add_node("N1", MA3_X[0], 0); s.add_node("N2", MA3_X[1], 0)
    s.add_bar_element("E1", "N0", "N1", "S")
    s.add_bar_element("E2", "N1", "N2", "S")
    s.add_support("FIX", ux=True, uy=True, tz=True); s.assign_support("N0", "FIX")
    _mass_and_modal(s, num_modes=4)
    s.add_nodal_mass("N1", MASS_CASE, my=MA3_M[0])
    s.add_nodal_mass("N2", MASS_CASE, my=MA3_M[1])
    return s


# ── M-A4 — rigid block on a Winkler foundation: heave and rocking ─────────
# Continuum: heave ω² = kL/M; rocking ω² = (∫k x²dx)/(∫m̄ x²dx) =
# (kL³/12)/(ML²/12) = kL/M — the SAME. A uniform rigid block on a uniform
# foundation has a DOUBLE eigenvalue.
#
# The DISCRETE model does not, and the reason is instructive. Rocking needs
# rotary inertia; with a lumped mass matrix that inertia has two parts: the
# translations of the nodes (Σ m_i x_i², which the trapezoidal lumping gets
# only to O(h²)) and the spin of each tributary segment about its own centroid
# (Σ m_i h²/12), which is NOT automatic — it has to be given as ``mtz``.
# Providing it gives an exact closed form for the discrete system,
#
#     ω_rock² = (k/m̄) · S/(S + M·h²/12),   S = Σ m_i x_i²,
#
# which tends to the continuum kL/M as h → 0. So M-A4 is exact on the discrete
# model and convergent on the continuum one, both checkable.
#
# WITHOUT ``mtz`` the nodal ROTATION DOFs have no mass. They used to be given
# a token mass (m_max·1e-8), which degraded K̃ by ~10⁸ and made the rocking
# frequency drift with the block's rigidity and with the LAPACK build (19,62 Hz
# here, 19,69 Hz elsewhere, 19,385 Hz instead of 19,346 Hz on Linux). They are
# now condensed out exactly — see ``test_rocking_without_rotary_inertia_is_unreliable``.
MA4_L = 6.0
MA4_K = 5.0e4
MA4_M = 20.0
MA4_F_HEAVE = math.sqrt(MA4_K * MA4_L / MA4_M) / (2 * math.pi)

def _ma4_tributary(n):
    """(x_i, m_i) of the lumped block with n elements."""
    h = MA4_L / n
    out = []
    for i in range(n + 1):
        w = 0.5 if i in (0, n) else 1.0
        out.append((MA4_L * i / n - MA4_L / 2.0, MA4_M * w / n))
    return h, out

def ma4_f_rock(n):
    """Exact rocking frequency OF THE DISCRETE MODEL with n elements."""
    h, trib = _ma4_tributary(n)
    S = sum(m * x * x for x, m in trib)
    spin = MA4_M * h * h / 12.0
    return MA4_F_HEAVE * math.sqrt(S / (S + spin))

def _build_m_a4(n):
    """Rigid block on a Winkler foundation, with the consistent rotary inertia
    m_i·h²/12 of each tributary segment given explicitly as ``mtz``.

    The section is rigid but not absurdly so (I = 10² m⁴ → λL = 0,27, deep in
    the rigid regime): pushing it further degrades the conditioning of the
    eigenproblem without making the block any more rigid in practice."""
    s = _new(); _beam_material(s)
    s.add_section("R", "C", b=1.0, h=1.0,
                  area_override=10.0, inertia_override=1.0e2)
    _chain(s, n, MA4_L, section="R", ky=MA4_K)
    s.add_support("RX", ux=True, uy=False); s.assign_support("N0", "RX")
    _mass_and_modal(s, num_modes=4)
    h, trib = _ma4_tributary(n)
    for i, (_x, m) in enumerate(trib):
        s.add_nodal_mass(f"N{i}", MASS_CASE, my=m, mtz=m * h * h / 12.0)
    return s

def build_m_a4():    return _build_m_a4(4)
def build_m_a4_c():  return _build_m_a4(2)
def build_m_a4_m():  return _build_m_a4(4)
def build_m_a4_f():  return _build_m_a4(8)

MA4_MESHES = [("m-a4-2", build_m_a4_c, "2"),
              ("m-a4-4", build_m_a4_m, "4"),
              ("m-a4-8", build_m_a4_f, "8")]

def build_m_a4_no_rotary():
    """The same block WITHOUT rotary inertia — the unreliable configuration."""
    s = _build_m_a4(4)
    for nm in s.nodal_masses:
        nm.mtz = 0.0
    return s


# ═══ B — convergence families (distributed mass, O(h²)) ════════════════════
MB_L = 6.0

def _beam_freq(beta_L, L=MB_L):
    """f = (βL)²·√(EI/m̄L⁴)/2π — the Euler–Bernoulli family."""
    return beta_L ** 2 * math.sqrt(E * I_BAR / (MBAR * L ** 4)) / (2 * math.pi)

# Simply supported: βL = nπ.
MB1_F = [_beam_freq(n * math.pi) for n in (1, 2, 3)]
# Cantilever: the roots of cos·cosh + 1 = 0.
MB2_BETA = (1.8751041, 4.6940911, 7.8547574)
MB2_F = [_beam_freq(b) for b in MB2_BETA]
# Free-free: the roots of cos·cosh − 1 = 0.
MB3_BETA = (4.7300408, 7.8532046)
MB3_F = [_beam_freq(b) for b in MB3_BETA]
# Axial (fixed-free bar): ω_n = (2n−1)π/2L·√(E/ρ).
MB4_RHO = GAMMA_C / G
MB4_F = [(2 * n - 1) * math.pi / (2 * MB_L) * math.sqrt(E / MB4_RHO)
         / (2 * math.pi) for n in (1, 2)]
# Simply supported beam on an elastic foundation: ω² = [EI(nπ/L)⁴ + k]/m̄.
MB5_K = 4.0e4
MB5_F = [math.sqrt((E * I_BAR * (n * math.pi / MB_L) ** 4 + MB5_K) / MBAR)
         / (2 * math.pi) for n in (1, 2)]

MB3_KSOFT = 1.0e-3      # "free-free" via springs soft enough to be invisible


def _build_mb(n, kind, num_modes=12, ky=0.0):
    s = _new(); _heavy_material(s)
    _chain(s, n, MB_L, ky=ky)
    if kind == "ss":
        s.add_support("PIN", ux=True, uy=True); s.add_support("ROL", ux=False, uy=True)
        s.assign_support("N0", "PIN"); s.assign_support(f"N{n}", "ROL")
    elif kind == "cant":
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("N0", "FIX")
    elif kind == "free":
        # A truly unsupported model cannot even be solved statically (K is
        # singular), so the classic soft-spring trick is used: springs whose
        # rigid-body frequencies are ~3 orders below f₁ and therefore do not
        # perturb the elastic modes.
        s.add_node_spring("N0", kx=MB3_KSOFT, ky=MB3_KSOFT, kt=MB3_KSOFT)
        s.add_node_spring(f"N{n}", ky=MB3_KSOFT)
    elif kind == "winkler":
        s.add_support("PIN", ux=True, uy=True); s.add_support("ROL", ux=False, uy=True)
        s.assign_support("N0", "PIN"); s.assign_support(f"N{n}", "ROL")
    else:
        raise ValueError(kind)
    return _mass_and_modal(s, num_modes=num_modes, from_self_weight=True)

def build_mb1_c():  return _build_mb(4, "ss")
def build_mb1_m():  return _build_mb(8, "ss")
def build_mb1_f():  return _build_mb(16, "ss")
def build_mb2_c():  return _build_mb(4, "cant")
def build_mb2_m():  return _build_mb(8, "cant")
def build_mb2_f():  return _build_mb(16, "cant", num_modes=40)
def build_mb3_c():  return _build_mb(8, "free")
def build_mb3_m():  return _build_mb(16, "free")
def build_mb3_f():  return _build_mb(32, "free")
def build_mb4_c():  return _build_mb(8, "cant", num_modes=40)
def build_mb4_m():  return _build_mb(16, "cant", num_modes=60)
def build_mb4_f():  return _build_mb(32, "cant", num_modes=90)
def build_mb5_c():  return _build_mb(8, "winkler", ky=MB5_K)
def build_mb5_m():  return _build_mb(16, "winkler", ky=MB5_K)
def build_mb5_f():  return _build_mb(32, "winkler", ky=MB5_K)

MB1_MESHES = [("m-b1-4", build_mb1_c, "4"), ("m-b1-8", build_mb1_m, "8"),
              ("m-b1-16", build_mb1_f, "16")]
MB2_MESHES = [("m-b2-4", build_mb2_c, "4"), ("m-b2-8", build_mb2_m, "8"),
              ("m-b2-16", build_mb2_f, "16")]
MB3_MESHES = [("m-b3-8", build_mb3_c, "8"), ("m-b3-16", build_mb3_m, "16"),
              ("m-b3-32", build_mb3_f, "32")]
MB4_MESHES = [("m-b4-8", build_mb4_c, "8"), ("m-b4-16", build_mb4_m, "16"),
              ("m-b4-32", build_mb4_f, "32")]
MB5_MESHES = [("m-b5-8", build_mb5_c, "8"), ("m-b5-16", build_mb5_m, "16"),
              ("m-b5-32", build_mb5_f, "32")]


# ═══ C — effective masses and participation ════════════════════════════════
# Classic truncation figures, as a FRACTION OF THE TOTAL m̄·L:
#   simply supported: 8/(n²π²) for odd n, exactly 0 for even n;
#   cantilever:       0.6131, 0.1883, 0.0647 for the first three modes.
MC_TOTAL = MBAR * MB_L
MC_SS_FRACTIONS = [8.0 / (n ** 2 * math.pi ** 2) for n in (1, 3)]
MC_CANT_FRACTIONS = [0.61310, 0.18811, 0.06474]


# ═══ D — mass definition (deliberate behaviours worth pinning) ═════════════
MD_P = 100.0        # kN

def _build_md(fx=0.0, fy=0.0, factor=1.0):
    """Cantilever with one nodal load feeding the Mass case."""
    s = _new(); _beam_material(s)
    s.add_node("A", 0, 0); s.add_node("B", MA1_L, 0)
    s.add_bar_element("E", "A", "B", "S")
    s.add_support("FIX", ux=True, uy=True, tz=True); s.assign_support("A", "FIX")
    s.add_load_case("LC")
    s.add_point_load("B", "LC", fx=fx, fy=fy)
    s.add_analysis_case(MASS_CASE, "Mass", {"LC": factor})
    s.add_analysis_case(MODAL_CASE, "Modal", {},
                        modal_case_id=MASS_CASE, num_modes=4)
    return s

def build_m_d_down(): return _build_md(fy=-MD_P)
def build_m_d_up():   return _build_md(fy=+MD_P)
def build_m_d_horiz(): return _build_md(fx=MD_P)


# ═══ E — response spectrum ══════════════════════════════════════════════════
ME_A0 = 2.0          # m/s², flat spectrum
ME_M = MA1_M

def build_m_e1():
    """SDOF with a FLAT spectrum: Sd = Sa/ω² and Γ·Sd·φ collapses to the static
    response under F = m·Sa. The Linear case in the same model carries exactly
    that static load, so the two must agree to machine precision — the tightest
    possible check of the Mass → Modal → Spectrum chain."""
    s = _new(); _beam_material(s)
    s.add_node("A", 0, 0); s.add_node("B", MA1_L, 0)
    s.add_bar_element("E", "A", "B", "S")
    s.add_support("FIX", ux=True, uy=True, tz=True); s.assign_support("A", "FIX")
    s.add_load_case("LC")
    s.add_point_load("B", "LC", fy=-ME_M * ME_A0)
    s.add_analysis_case("LIN", "Linear", {"LC": 1.0})
    s.add_analysis_case(MASS_CASE, "Mass", {})
    s.add_nodal_mass("B", MASS_CASE, mx=ME_M, my=ME_M)
    s.add_analysis_case(MODAL_CASE, "Modal", {},
                        modal_case_id=MASS_CASE, num_modes=2)
    s.add_spectral_function("SP", points=[[0.0, ME_A0], [10.0, ME_A0]])
    s.add_analysis_case("SPY", "Spectrum", {}, modal_case_id=MODAL_CASE,
                        spectrum_id="SP", direction="Y", combination_rule="SRSS")
    s.add_analysis_case("SPX", "Spectrum", {}, modal_case_id=MODAL_CASE,
                        spectrum_id="SP", direction="X", combination_rule="SRSS")
    s.add_analysis_case("SPXY", "Spectrum", {}, modal_case_id=MODAL_CASE,
                        spectrum_id="SP", direction="XY", combination_rule="SRSS")
    return s

ME1_UY = ME_M * ME_A0 / MA1_KB      # static tip deflection, = spectral response
ME1_UX = ME_M * ME_A0 / MA1_KA

def build_m_e2():
    """Two-mass cantilever (M-A3) driven by the same flat spectrum, with SRSS
    and with CQC. The modes are 5× apart in frequency, so at ξ = 5 % the CQC
    cross terms vanish and CQC must reproduce SRSS."""
    s = build_m_a3()
    s.add_spectral_function("SP", points=[[0.0, ME_A0], [10.0, ME_A0]])
    s.add_analysis_case("SRSS", "Spectrum", {}, modal_case_id=MODAL_CASE,
                        spectrum_id="SP", direction="Y",
                        combination_rule="SRSS")
    s.add_analysis_case("CQC", "Spectrum", {}, modal_case_id=MODAL_CASE,
                        spectrum_id="SP", direction="Y",
                        combination_rule="CQC", damping=0.05)
    return s

def build_m_e3():
    """Same SDOF, but with a spectrum that must be INTERPOLATED: the structure's
    period falls between two spectrum points, so Sa = linear interpolation."""
    s = build_m_e1()
    s.remove_spectral_function("SP")
    s.add_spectral_function("SP", points=[[0.10, 1.0], [0.30, 3.0]])
    return s

def me3_expected_sa(T):
    return 1.0 + (T - 0.10) / 0.20 * 2.0


# ═══ Case + quantity specifications ════════════════════════════════════════
HZ = 1.0

MODAL_CASES: list[Case] = [
    Case("m-a1", "Oscilador de 1 GL — massa concentrada na ponta da consola",
         "Barra + massa nodal", "Modal", build_m_a1, quantities=[
            Quantity("f₁ (flexão)", "Hz", MA1_F_BEND,
                     lambda r, s: freq(r, 1), rel_tol=1e-8),
            Quantity("T₁", "s", 1.0 / MA1_F_BEND,
                     lambda r, s: period(r, 1), rel_tol=1e-8),
            Quantity("f₂ (axial)", "Hz", MA1_F_AXIAL,
                     lambda r, s: freq(r, 2), rel_tol=1e-8),
            Quantity("Massa efetiva do modo 1", "t", MA1_M,
                     lambda r, s: meff_y(r, 1), rel_tol=1e-8),
            Quantity("Massa efetiva do modo 1", "%", 100.0,
                     lambda r, s: modes(r)[0]["meff_y_pct"], rel_tol=1e-8),
         ], note="f = √(3EI/mL³)/2π; exato e independente da malha."),
    Case("m-a2", "1 GL com mola nodal em paralelo",
         "Barra + mola + massa", "Modal", build_m_a2, quantities=[
            Quantity("f₁", "Hz", MA2_F, lambda r, s: freq(r, 1), rel_tol=1e-8),
            Quantity("k_eq implícito", "kN/m", MA2_KEQ,
                     lambda r, s: (2 * math.pi * freq(r, 1)) ** 2 * MA1_M,
                     rel_tol=1e-8),
         ], note="k_eq = k_mola + 3EI/L³ (molas em paralelo) — a mola entra no "
                 "K do eigenproblema."),
    Case("m-a3", "Duas massas concentradas numa consola (2 GL)",
         "Barra + massas nodais", "Modal", build_m_a3, quantities=[
            Quantity("f₁", "Hz", MA3_F[0], lambda r, s: freq(r, 1), rel_tol=1e-5),
            Quantity("f₂", "Hz", MA3_F[1], lambda r, s: freq(r, 2), rel_tol=1e-5),
            Quantity("f₂/f₁", "—", MA3_F[1] / MA3_F[0],
                     lambda r, s: freq(r, 2) / freq(r, 1), rel_tol=1e-5),
         ], note="Referência: eigenproblema 2×2 da matriz de flexibilidade da consola."),
    Case("m-a4", "Bloco rígido sobre fundação Winkler — translação e rotação",
         "Barra rígida + fundação", "Modal", build_m_a4, quantities=[
            Quantity("f (rotação, modelo discreto)", "Hz", ma4_f_rock(4),
                     lambda r, s: freq(r, 1), rel_tol=1e-3),
            Quantity("f (translação)", "Hz", MA4_F_HEAVE,
                     lambda r, s: freq(r, 2), rel_tol=1e-3),
         ], note="No contínuo os dois modos são degenerados (ω² = kL/M). No "
                 "modelo discreto a rotação vem ω² = (k/m̄)·S/(S+Mh²/12) com "
                 "S = Σm_i x_i² — exato, e tende para kL/M com h→0. Requer a "
                 "inércia de rotação mtz = m_i h²/12 explícita."),
]


# ═══ C — membrane (triangle) modal: CST vs ES-FEM on the same mesh ══════════
# A slender cantilever wall-beam (in-plane bending dominated), the modal
# counterpart of the static case 3.2. The mass is the distributed self-weight
# (lumped), so it is the same on every mesh; only the stiffness formulation
# changes. The CST is over-stiff and overestimates the frequencies; ES-FEM
# smooths that excess, giving lower frequencies closer to the converged value —
# and, being edge-based, it is temporally stable (no spurious modes), unlike
# the node-based NS-FEM. A fine ES-FEM mesh is the reference.
MC1_L, MC1_H, MC1_T = 4.0, 0.5, 0.20
# Euler–Bernoulli reference for the fundamental (in-plane bending) frequency of
# the cantilever wall-beam: f₁ = (β₁²/2π)·√(EI/(m̄L⁴)), β₁L = 1.875104,
# I = t·H³/12, m̄ = γ·(t·H)/g. A deep beam (L/H = 8) is softened by shear, so
# the true value is a little below this; CST overestimates it, ES-FEM sits
# closer to (and just below) the converged value.
_MC1_I = MC1_T * MC1_H ** 3 / 12.0
_MC1_MBAR = GAMMA_C * (MC1_T * MC1_H) / G
MC1_F1_BEAM = (1.875104 ** 2 / (2 * math.pi)) * \
    math.sqrt(E * _MC1_I / (_MC1_MBAR * MC1_L ** 4))

def _build_mc1(nx, ny, formulation):
    s = _new()
    s.add_material("C", E, GAMMA_C, poisson=NU)
    s.add_tri_section("W", "C", thickness=MC1_T, formulation=formulation)
    ids = {}
    for j in range(ny + 1):
        for i in range(nx + 1):
            nid = f"n{i}_{j}"
            s.add_node(nid, MC1_L * i / nx, MC1_H * j / ny)
            ids[(i, j)] = nid
    e = 0
    for j in range(ny):
        for i in range(nx):
            a, b = ids[(i, j)], ids[(i + 1, j)]
            c, d = ids[(i + 1, j + 1)], ids[(i, j + 1)]
            s.add_tri_element(f"e{e}", a, b, c, "W"); e += 1
            s.add_tri_element(f"e{e}", a, c, d, "W"); e += 1
    s.add_support("FIX", ux=True, uy=True)
    for j in range(ny + 1):
        s.assign_support(ids[(0, j)], "FIX")
    _mass_and_modal(s, num_modes=4, from_self_weight=True)
    return s

def build_mc1_cst_c():  return _build_mc1(8, 2, "CST")
def build_mc1_cst_m():  return _build_mc1(16, 4, "CST")
def build_mc1_es_c():   return _build_mc1(8, 2, "ES-FEM")
def build_mc1_es_m():   return _build_mc1(16, 4, "ES-FEM")
def build_mc1_es_ref(): return _build_mc1(48, 12, "ES-FEM")

MC1_MESHES = [
    ("m-c1-cst-8x2", build_mc1_cst_c, "CST 8×2"),
    ("m-c1-cst-16x4", build_mc1_cst_m, "CST 16×4"),
    ("m-c1-es-8x2", build_mc1_es_c, "ES-FEM 8×2"),
    ("m-c1-es-16x4", build_mc1_es_m, "ES-FEM 16×4"),
    ("m-c1-es-48x12", build_mc1_es_ref, "ES-FEM 48×12 (ref)"),
]


MODAL_MODELS = (
    [(c.id, c.build) for c in MODAL_CASES]
    + [(mid, b) for (mid, b, _lbl) in MC1_MESHES]
    + [("m-a1-refinado", build_m_a1_refined),
       ("m-a4-sem-inercia-rotacao", build_m_a4_no_rotary)]
    + [(mid, b) for (mid, b, _lbl) in MA4_MESHES if mid != "m-a4-4"]
    + [(mid, b) for fam in (MB1_MESHES, MB2_MESHES, MB3_MESHES,
                            MB4_MESHES, MB5_MESHES)
       for (mid, b, _lbl) in fam]
    + [("m-d-baixo", build_m_d_down), ("m-d-cima", build_m_d_up),
       ("m-d-horizontal", build_m_d_horiz)]
    + [("m-e1", build_m_e1), ("m-e2", build_m_e2), ("m-e3", build_m_e3)]
)
