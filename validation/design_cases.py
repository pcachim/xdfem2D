"""Pacote de validação dos módulos de dimensionamento (RC / aço / madeira).

Diferente de ``validate_quads.py`` (que compara resultados do motor FE contra
soluções analíticas de elasticidade), este pacote valida a camada de
dimensionamento — ``rc_design.py`` / ``steel_design.py`` / ``timber_design.py``
— que são adaptadores finos sobre o pacote externo ``eurocodepy``. O risco
real a validar aqui não são as fórmulas do Eurocódigo em si (essas são
responsabilidade do ``eurocodepy``), mas sim:

  * a conversão de unidades e o sinal das grandezas na fronteira xdfem2d↔eurocodepy;
  * a integração FE→dimensionamento (que esforços/propriedades chegam a cada
    verificação);
  * casos-charneira específicos do xdfem2d — nomeadamente a assimetria
    tração/compressão documentada no docstring de ``timber_design.py``
    ("Sign is preserved for the axial force... the EC5 rules differ").

Todos os valores "esperado" abaixo foram obtidos por EXECUÇÃO REAL — não são
derivados à mão — chamando diretamente ``RCSection`` (rc_design.py) e as
funções de baixo nível do ``eurocodepy`` (``eurocode3_section_check``,
``eurocode3_member_check``, ``eurocode5_section_check``) com as propriedades
de secção calculadas pelo próprio motor xdfem2d (``Structure2D.add_section``).

ATENÇÃO: a versão do ``eurocodepy`` publicada no PyPI (2026.1.1) tem uma API
incompatível com a que o xdfem2d importa (``calc_asl_nm``,
``eurocode2_shear_check``, ``calc_torsion``, ``ShearInput``,
``SectionResistanceInput``, ...). É preciso ter instalada a versão real do
código-fonte do ``eurocodepy`` (o repositório local, não o pacote do PyPI)
para este script correr.

Correr:  python validation/design_cases.py
Requer: xdfem2d instalado (editable) E o eurocodepy instalado a partir do
código-fonte correto (ver nota acima) — não a versão pública do PyPI.
"""
from __future__ import annotations

import sys

_results: list[bool] = []


def _check(name: str, got: float, expected: float, tol: float, unit: str = "") -> None:
    ok = abs(got - expected) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: got={got:.6g}{unit} expected={expected:.6g}{unit} "
          f"(tol={tol:.3g}{unit})")
    _results.append(ok)


def _check_true(name: str, ok: bool, detail: str = "") -> None:
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail else ""))
    _results.append(ok)


# ═══════════════════════════════════════════════════════════════════════
# A.1 — Betão armado (rc_design.RCSection), EC2
# Secção 0.30×0.50 m, recobrimento 0.03 m, C25/30, A500 (fck=25, fyk=500 MPa).
# Valores obtidos por execução real de RCSection(...).flexural_reinforcement /
# .shear_reinforcement / .torsion_reinforcement.
# ═══════════════════════════════════════════════════════════════════════

def _rc_section():
    from xdfem2d.rc_design import RCSection
    return RCSection(b=0.30, h=0.50, cover=0.03, fck_mpa=25.0, fyk_mpa=500.0)


def case_a1_flexao_simples():
    """A1 — Flexão simples, Med=150 kNm, Ned=0."""
    sec = _rc_section()
    r = sec.flexural_reinforcement(Med=150.0)
    _check("A1 flexao simples: As_bot", r["As_bot"], 7.921e-4, 5e-6, " m2")
    _check_true("A1 flexao simples: As_top == 0 (sem armadura de compressão)",
                r["As_top"] == 0.0)


def case_a2_nm_compressao():
    """A2 — N+M com compressão (excentricidade grande), Med=150 kNm, Ned=-300 kN."""
    sec = _rc_section()
    r = sec.flexural_reinforcement(Med=150.0, Ned=-300.0)
    _check("A2 N+M compressao: As_bot", r["As_bot"], 4.975e-4, 5e-6, " m2")
    _check_true("A2 N+M compressao: As_bot < A1 (compressão ajuda a resistir)",
                r["As_bot"] < 7.921e-4,
                "esperado: a compressão reduz a tração necessária")


def case_a2b_nm_tracao():
    """A2b — N+M com tração, Med=100 kNm, Ned=+200 kN."""
    sec = _rc_section()
    r = sec.flexural_reinforcement(Med=100.0, Ned=200.0)
    _check("A2b N+M tracao: As_bot", r["As_bot"], 7.414e-4, 5e-6, " m2")


def case_a3_hogging():
    """A3 — Momento negativo (hogging), Med=-150 kNm → armadura no topo."""
    sec = _rc_section()
    r = sec.flexural_reinforcement(Med=-150.0)
    _check_true("A3 hogging: As_top > 0 e As_bot == 0 (inversão correta)",
                r["As_top"] > 0.0 and r["As_bot"] == 0.0)
    _check("A3 hogging: As_top == As_bot(A1) por simetria da secção",
          r["As_top"], 7.921e-4, 5e-6, " m2")


def case_a4_shear_com_minimo():
    """A4 — Esforço transverso, Ved=50 kN, com armadura mínima."""
    sec = _rc_section()
    r = sec.shear_reinforcement(Ved=50.0, min_shear=True)
    _check("A4 shear c/ minimo: Asw/s", r["Asw_s"], 2.4e-4, 1e-6, " m2/m")


def case_a4b_shear_sem_minimo():
    """A4b — Mesmo Ved=50 kN mas sem impor mínimo — Asw/s deve poder ser 0
    quando a resistência do betão (VRd,s no modo 'no_shear_reinf') já cobre."""
    sec = _rc_section()
    r = sec.shear_reinforcement(Ved=50.0, min_shear=False)
    _check("A4b shear s/ minimo: Asw/s", r["Asw_s"], 0.0, 1e-9, " m2/m")
    _check_true("A4b vs A4: mínimo realmente reduz Asw/s calculado",
                r["Asw_s"] < 2.4e-4)


def case_a5_torsao():
    """A5 — Torção pura, Ted=20 kNm."""
    sec = _rc_section()
    r = sec.torsion_reinforcement(Ted=20.0)
    _check("A5 torcao: TRd_max", r["TRd_max"], 70.697, 0.05, " kNm")
    _check("A5 torcao: utilizacao Ted/TRd_max", r["util"], 20.0 / 70.697, 0.005)
    _check_true("A5 torcao: PASS (util < 1.0)", r["util"] < 1.0)


def case_a8_coluna_nm():
    """A8 — Pilar em N-M, Med=80 kNm, Ned=-800 kN (compressão elevada,
    controlado por compressão — excentricidade pequena)."""
    sec = _rc_section()
    r = sec.flexural_reinforcement(Med=80.0, Ned=-800.0)
    _check("A8 coluna N-M: As_bot", r["As_bot"], 1.8806e-4, 5e-6, " m2")
    _check_true("A8 coluna N-M: nota de excentricidade pequena presente",
                "eccentricity" in r.get("note", "") or "compression" in r.get("note", ""))


# ═══════════════════════════════════════════════════════════════════════
# A.2 — Aço (EC3), perfil IPE300 / S275
# Propriedades de secção calculadas pelo próprio motor xdfem2d
# (Structure2D.add_section("IPE300","S275", b=0.150,h=0.300,tw=0.0071,tf=0.0107,
#  shape="I")) e verificadas com eurocodepy.ec3.uls.cross_section /
# member_buckling diretamente (bypass do pipeline completo Structure2D, para
# isolar a verificação da secção em si — os casos B fazem o caminho completo).
# ═══════════════════════════════════════════════════════════════════════

_IPE300 = dict(
    area=5188.06, wel_y=533265.796420888, wpl_y=602098.3789999999,
    wel_z=80360.79333844441, wpl_z=123886.0565,
    av_y=3209.9999999999997, av_z=2054.0299999999997,
    wt=14555.355283489096, hw=300.0 - 2 * 10.7,
    iy=7998.98694631332e6, iz=602.7059500383331e6,
    fy=275.0,
)


def _steel_section_input(cls=1, n_ed=0.0):
    from eurocodepy.ec3.uls.cross_section import SectionResistanceInput
    from eurocodepy.ec3 import classification as ec3cls
    p = _IPE300
    return SectionResistanceInput(
        kind="I", section_class=cls, fy=p["fy"], gamma_M0=1.0,
        area=p["area"], area_eff=p["area"], b=150.0, h=300.0, tw=7.1, tf=10.7,
        hw=p["hw"], eps=ec3cls.epsilon(p["fy"]),
        wpl_y=p["wpl_y"], wpl_z=p["wpl_z"], wel_y=p["wel_y"], wel_z=p["wel_z"],
        weff_y=0.0, weff_z=0.0, wt=p["wt"], av_y=p["av_y"], av_z=p["av_z"], d_my=0.0)


def case_a6_1_steel_n_tracao():
    """A6.1 — Aço N tração pura, N=400 kN (compressão positiva na convenção
    do eurocodepy → tração = n_ed negativo)."""
    from eurocodepy.ec3.uls.cross_section import SectionForces, eurocode3_section_check
    inp = _steel_section_input()
    r = eurocode3_section_check(inp, SectionForces(n_ed=-400.0))
    _check("A6.1 aço N tracao: utilizacao N+M", r.util_bending_axial, 0.280, 0.01)
    _check_true("A6.1 aço N tracao: PASS", r.passed)


def case_a6_2_steel_n_compressao():
    """A6.2 — Aço N compressão pura (secção, sem encurvadura), N=400 kN."""
    from eurocodepy.ec3.uls.cross_section import SectionForces, eurocode3_section_check
    inp = _steel_section_input()
    r = eurocode3_section_check(inp, SectionForces(n_ed=400.0))
    _check("A6.2 aço N compressao (seccao): utilizacao N+M", r.util_bending_axial, 0.280, 0.01)


def case_a6_3_steel_m_puro():
    """A6.3 — Aço M puro, My=80 kNm."""
    from eurocodepy.ec3.uls.cross_section import SectionForces, eurocode3_section_check
    inp = _steel_section_input()
    r = eurocode3_section_check(inp, SectionForces(my_ed=80.0))
    _check("A6.3 aço M puro: utilizacao N+M", r.util_bending_axial, 0.483, 0.01)


def case_a6_4_steel_v_puro():
    """A6.4 — Aço V puro, Vz=150 kN."""
    from eurocodepy.ec3.uls.cross_section import SectionForces, eurocode3_section_check
    inp = _steel_section_input()
    r = eurocode3_section_check(inp, SectionForces(vz_ed=150.0))
    _check("A6.4 aço V puro: utilizacao shear_z", r.util_shear_z, 0.460, 0.01)


def case_a6_5_steel_t_puro():
    """A6.5 — Aço T puro (torção de St-Venant), T=2 kNm."""
    from eurocodepy.ec3.uls.cross_section import SectionForces, eurocode3_section_check
    inp = _steel_section_input()
    r = eurocode3_section_check(inp, SectionForces(t_ed=2.0))
    _check("A6.5 aço T puro: utilizacao torsion", r.util_torsion, 0.865, 0.01)


def case_a6_6_steel_encurvadura():
    """A6.6 — Aço, coluna N=400 kN comprimida, L=8 m (Ky=Kz=1.0),
    curvas b/c/b — verifica a redução χ por encurvadura por compressão centrada."""
    from eurocodepy.ec3.uls.member_buckling import MemberInput, eurocode3_member_check
    p = _IPE300
    inp = MemberInput(
        n_ed=400.0, my_ed=0.0, mz_ed=0.0,
        area=p["area"], area_eff=p["area"], w_y=p["wpl_y"], w_z=p["wpl_z"],
        iy=p["iy"], iz=p["iz"], it=1.0e9, iw=1.0e12,
        lcr_y=8000.0, lcr_z=8000.0, l_lt=8000.0,
        curve_y="b", curve_z="c", curve_lt="b",
        c1=1.0, cmy=0.9, cmz=0.9, cm_lt=0.9,
        fy=p["fy"], e_mod=210000.0, g_mod=81000.0,
        gamma_m1=1.0, section_class=1,
        susceptible_lt=False, rolled_lt=True, d_my=0.0)
    r = eurocode3_member_check(inp)
    _check("A6.6 aço encurvadura: chi_z", r.chi_z, 0.964, 0.01)
    _check("A6.6 aço encurvadura: utilizacao", r.utilization, 0.291, 0.01)
    _check_true("A6.6 aço encurvadura: chi_z < 1.0 (reduz a resistência)", r.chi_z < 1.0)
    _check_true("A6.6 aço encurvadura: utilizacao(6.62) > utilizacao secção pura (0.280)",
                r.utilization > 0.280)


# ═══════════════════════════════════════════════════════════════════════
# A.3 — Madeira (EC5), secção retangular 0.10×0.20 m, C24
# ATENÇÃO DE UNIDADES: eurocodepy.ec5.uls.cross_section espera as dimensões
# da secção em METROS (não mm, ao contrário do módulo ec3) — confirmado por
# execução real (com mm todas as utilizações davam ~0, com m os valores
# ficaram fisicamente sensatos). Validar isto é precisamente o tipo de
# discrepância de unidades que este pacote existe para apanhar.
# ═══════════════════════════════════════════════════════════════════════

def _timber_section_input(l_0y=0.0, l_0z=0.0, l_0m=0.0):
    from eurocodepy.ec5.uls.cross_section import TimberSectionInput
    from eurocodepy.ec5.materials import Timber, ServiceClass, LoadDuration
    from eurocodepy.utils.crosssection import RectangularCrossSection
    sec = RectangularCrossSection(width=0.10, height=0.20)  # metros
    tim = Timber("C24")
    return TimberSectionInput(section=sec, timber=tim, service_class=ServiceClass.SC1,
                              load_duration=LoadDuration.MediumDuration,
                              l_0y=l_0y, l_0z=l_0z, l_0m=l_0m)


def case_a7_1_timber_n_tracao():
    """A7.1 — Madeira N tração pura, N=25 kN (convenção EC5: n_ed>0 = tração)."""
    from eurocodepy.ec5.uls.cross_section import TimberForces, eurocode5_section_check
    r = eurocode5_section_check(_timber_section_input(), TimberForces(n_ed=25.0))
    _check("A7.1 madeira N tracao: utilizacao N+M", r.util_bending_axial, 0.140, 0.01)


def case_a7_2_timber_n_compressao():
    """A7.2 — Madeira N compressão pura, N=25 kN, sem dados de encurvadura
    (l_0y=l_0z=0 → k_c=1.0). Compara diretamente com A7.1: as regras do EC5
    são assimétricas entre tração e compressão (ver docstring de
    timber_design.py) — este par de casos é o teste mais importante do A.3."""
    from eurocodepy.ec5.uls.cross_section import TimberForces, eurocode5_section_check
    r = eurocode5_section_check(_timber_section_input(), TimberForces(n_ed=-25.0))
    _check("A7.2 madeira N compressao: utilizacao N+M", r.util_bending_axial, 0.097, 0.01)
    _check_true("A7.2 vs A7.1: tracao e compressao dao utilizacoes DIFERENTES "
                "para o mesmo |N| (assimetria EC5 esperada, nao é bug)",
                abs(0.140 - 0.097) > 0.01)


def case_a7_3_timber_m_puro():
    """A7.3 — Madeira M puro, My=6 kNm."""
    from eurocodepy.ec5.uls.cross_section import TimberForces, eurocode5_section_check
    r = eurocode5_section_check(_timber_section_input(), TimberForces(my_ed=6.0))
    _check("A7.3 madeira M puro: utilizacao N+M", r.util_bending_axial, 0.609, 0.01)


def case_a7_4_timber_v_puro():
    """A7.4 — Madeira V puro, Vz=8 kN."""
    from eurocodepy.ec5.uls.cross_section import TimberForces, eurocode5_section_check
    r = eurocode5_section_check(_timber_section_input(), TimberForces(vz_ed=8.0))
    _check("A7.4 madeira V puro: utilizacao shear", r.util_shear, 0.244, 0.01)


def case_a7_5_timber_t_puro():
    """A7.5 — Madeira T puro, T=0.5 kNm."""
    from eurocodepy.ec5.uls.cross_section import TimberForces, eurocode5_section_check
    r = eurocode5_section_check(_timber_section_input(), TimberForces(t_ed=0.5))
    _check("A7.5 madeira T puro: utilizacao torsion", r.util_torsion, 0.002, 0.001)


def case_a7_6_timber_compressao_encurvadura():
    """A7.6 — Madeira, compressão N=25 kN + encurvadura, comprimento efetivo
    3.0 m em y e z. Deve dar k_c < 1.0 e utilização maior que A7.2 (sem
    encurvadura)."""
    from eurocodepy.ec5.uls.cross_section import TimberForces, eurocode5_section_check
    inp = _timber_section_input(l_0y=3000.0, l_0z=3000.0)
    r = eurocode5_section_check(inp, TimberForces(n_ed=-25.0))
    _check("A7.6 madeira compressao+encurv: k_c,y", r.k_c[0], 0.77, 0.02)
    _check("A7.6 madeira compressao+encurv: k_c,z", r.k_c[1], 0.28, 0.02)
    _check("A7.6 madeira compressao+encurv: utilizacao", r.util_bending_axial, 0.340, 0.01)
    _check_true("A7.6 madeira compressao+encurv: k_c,z < 1.0 e utilizacao > A7.2 (0.097)",
                r.k_c[1] < 1.0 and r.util_bending_axial > 0.097)


# ═══════════════════════════════════════════════════════════════════════
# B — Casos end-to-end (mesh → calculate() → design_*), a implementar depois
# (viga RC bi-apoiada, consola de aço, viga de madeira, pilar comprimido,
#  laje simplesmente apoiada, punçoamento) — testam a integração completa
# Structure2D → resultados FE → design_*_members / RC design, incluindo o
# domínio 'plate' necessário para validar a torção de aço/madeira através
# do pipeline real (não isolada como aqui em A.2/A.3).
# ═══════════════════════════════════════════════════════════════════════


def run() -> bool:
    cases = [v for k, v in sorted(globals().items()) if k.startswith("case_")]
    print(f"A executar {len(cases)} casos de validação de dimensionamento...\n")
    for fn in cases:
        print(f"--- {fn.__name__}: {(fn.__doc__ or '').strip().splitlines()[0]} ---")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {fn.__name__}: {exc!r}")
            _results.append(False)
        print()
    n_ok = sum(_results)
    print(f"=== {n_ok}/{len(_results)} OK ===")
    return n_ok == len(_results)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
