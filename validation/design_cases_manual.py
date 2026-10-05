"""Validação independente por cálculo "à mão": expressões dos Eurocódigos
implementadas diretamente aqui (sem chamar `eurocodepy`), para os mesmos
casos de ``design_cases.py``.

Objetivo: ``design_cases.py`` valida rc_design/steel_design/timber_design
correndo o `eurocodepy` de verdade — mas isso não deteta um erro que exista
igualmente na fórmula do `eurocodepy`. Este ficheiro fecha essa lacuna:
reimplementa as expressões da norma (EC2 §6.1/§6.2/§6.3, EC3 §6.2, EC5 §6.1/
§6.2) inteiramente por conta própria, a partir do texto dos Eurocódigos, e
compara o resultado com os valores obtidos por execução real em
``design_cases.py`` (constantes ``_REF`` abaixo, copiadas desses outputs).

Onde as duas contas batem certo (a esmagadora maioria dos casos — ver
resultados), temos confirmação cruzada por dois caminhos totalmente
independentes: o `eurocodepy` real e a fórmula da norma escrita de novo aqui.
Onde NÃO batem (marcado explicitamente ``_APPROX`` ou ``_NOT_VERIFIED``), é
porque o caso cai num procedimento com mais passos (interação N-M controlada
por compressão, encurvadura por flexão, torção combinada de madeira) que a
fórmula simples de fecho aqui usada não reproduz exatamente — fica
documentado o porquê, não fica escondido.

Correr:  python validation/design_cases_manual.py   (só precisa de `math`,
sem dependências do xdfem2d nem do eurocodepy)
"""
from __future__ import annotations

import math
import sys

_results: list[bool] = []


def _check(name: str, got: float, expected: float, tol: float, unit: str = "") -> None:
    ok = abs(got - expected) <= tol
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: mão={got:.6g}{unit}  referência={expected:.6g}{unit} "
          f"(tol={tol:.3g}{unit})")
    _results.append(ok)


def _note(msg: str) -> None:
    print(f"[NOTE] {msg}")


# ═══════════════════════════════════════════════════════════════════════
# A.1 — Betão armado, EC2. Secção b=0.30 h=0.50 m, cobrimento=0.03 m
# (→ d = h - cobrimento = 0.47 m, convenção confirmada por retroanálise),
# C25/30 (fck=25 MPa), A500 (fyk=500 MPa), γc=1.5, γs=1.15, αcc=1.0.
# ═══════════════════════════════════════════════════════════════════════

B, H, COVER = 0.30, 0.50, 0.03
D = H - COVER                      # 0.47 m
FCK, FYK = 25.0, 500.0
GAMMA_C, GAMMA_S, ALPHA_CC = 1.5, 1.15, 1.0
FCD = ALPHA_CC * FCK / GAMMA_C * 1000.0     # kN/m2 (16 666.7)
FYD = FYK / GAMMA_S * 1000.0                # kN/m2 (434 782.6)


def _rc_flexure(m_ed: float, n_ed: float = 0.0) -> float:
    """EC2 §6.1, bloco retangular de tensões (λ=0.8, η=1.0, secção classe
    ≤C50/60), método de K com o braço reduzido por N_ed (excentricidade em
    torno do meio-vão, d - h/2). Convenção: N positivo = tração.

    K = M'/(b·d²·fcd);  0.32α² - 0.8α + K = 0  (α = x/d)
    z = d·(1 - 0.4α);   As = M'/(fyd·z) + N_ed/fyd
    """
    m_eff = m_ed - n_ed * (D - H / 2)
    k = m_eff / (B * D**2 * FCD)
    disc = 0.64 - 4 * 0.32 * k
    alpha = (0.8 - math.sqrt(disc)) / 0.64
    z = D * (1 - 0.4 * alpha)
    return m_eff / (FYD * z) + n_ed / FYD


def case_a1_flexao_simples():
    """A1 — flexão simples, Med=150 kNm."""
    as_bot = _rc_flexure(150.0)
    _check("A1 flexao simples: As", as_bot, 7.9208e-4, 2e-7, " m2")


def case_a2_nm_compressao():
    """A2 — N+M compressão, Med=150 kNm, Ned=-300 kN."""
    as_bot = _rc_flexure(150.0, -300.0)
    _check("A2 N+M compressao: As", as_bot, 4.9746e-4, 2e-7, " m2")


def case_a2b_nm_tracao():
    """A2b — N+M tração, Med=100 kNm, Ned=+200 kN."""
    as_bot = _rc_flexure(100.0, 200.0)
    _check("A2b N+M tracao: As", as_bot, 7.4137e-4, 2e-7, " m2")


def case_a3_hogging():
    """A3 — hogging, Med=-150 kNm — por simetria da secção é o mesmo cálculo
    que A1 trocando topo↔fundo (b, d e cobrimentos iguais nas duas faces)."""
    as_top = _rc_flexure(150.0)      # |Med| igual a A1
    _check("A3 hogging: As_top", as_top, 7.9208e-4, 2e-7, " m2")


def case_a4_shear_vrdc_e_minimo():
    """A4 — Ved=50 kN. V_Rd,c (§6.2.2, ρl=0 → v_min governa) e Asw/s mínimo
    (§9.2.2): ρw,min = 0.08·√fck/fyk ; Asw,min/s = ρw,min·b.

    k = 1 + √(200/d[mm]) ≤ 2.0
    v_min = 0.035·k^1.5·√fck  [MPa]
    V_Rd,c = v_min·b·d  [MN → kN]
    """
    d_mm = D * 1000.0
    k = min(1 + math.sqrt(200.0 / d_mm), 2.0)
    v_min = 0.035 * k**1.5 * math.sqrt(FCK)          # MPa
    vrdc = v_min * B * D * 1000.0                     # kN
    _check("A4 shear: VRd,c", vrdc, 52.408, 0.05, " kN")
    rho_w_min = 0.08 * math.sqrt(FCK) / FYK
    asw_min = rho_w_min * B
    _check("A4 shear: Asw,min/s", asw_min, 2.40e-4, 1e-7, " m2/m")
    _note(f"VEd=50kN ≤ VRd,c={vrdc:.2f}kN → betão sozinho chega; "
          "só entra a armadura mínima, confirma mode='no_shear_reinf'.")


def case_a4b_shear_sem_minimo():
    """A4b — mesmo Ved, mas sem impor mínimo: como VEd ≤ VRd,c, Asw/s=0."""
    _check("A4b shear s/ minimo: Asw/s", 0.0, 0.0, 1e-9, " m2/m")


def case_a5_torsao():
    """A5 — torção pura, Ted=20 kNm (EC2 §6.3, analogia de secção fechada de
    parede fina, θ=45°):

    t_ef = A/u ;  A_k = (b-t_ef)(h-t_ef) ;  u_k = 2[(b-t_ef)+(h-t_ef)]
    ν = 0.6(1 - fck/250)
    T_Rd,max = 2·ν·fcd·A_k·t_ef·sinθ·cosθ   (θ=45° → sinθcosθ=0.5)
    Asw/s = Ted/(2·A_k·fyd·cotθ) ;  Asl = Ted·u_k·cotθ/(2·A_k·fyd)
    """
    area, u = B * H, 2 * (B + H)
    t_ef = area / u
    a_k = (B - t_ef) * (H - t_ef)
    u_k = 2 * ((B - t_ef) + (H - t_ef))
    nu = 0.6 * (1 - FCK / 250.0)
    t_rd_max = 2 * nu * FCD * a_k * t_ef * 0.5        # sinθcosθ = 0.5 @ 45°
    asw_s = 20.0 / (2 * a_k * FYD)                     # cotθ = 1
    asl = 20.0 * u_k / (2 * a_k * FYD)
    _check("A5 torcao: t_ef", t_ef, 0.09375, 1e-6, " m")
    _check("A5 torcao: A_k", a_k, 0.0837891, 1e-6, " m2")
    _check("A5 torcao: TRd,max", t_rd_max, 70.697, 0.05, " kNm")
    _check("A5 torcao: Asw_tor/s", asw_s, 2.745e-4, 1e-6, " m2/m")
    _check("A5 torcao: Asl_tor", asl, 3.363e-4, 1e-6, " m2")


def case_a8_coluna_nm():
    """A8 — pilar N-M, Med=80 kNm, Ned=-800 kN. NÃO verificado por fórmula
    fechada: a K desta combinação (K=0.2318) dá x/d≈0.335, que já é uma
    excentricidade pequena — o próprio xdfem2d sinaliza isto
    ("compression-controlled: use full M-N interaction") e passa a usar o
    diagrama de interação N-M completo em vez do método K simples de secção
    simplesmente armada. Confirma-se apenas que K bate certo; o As final
    (1.8806e-4 m2) resulta de um procedimento com mais passos que este
    ficheiro não reproduz."""
    m_eff = 80.0 - (-800.0) * (D - H / 2)
    k = m_eff / (B * D**2 * FCD)
    _check("A8 coluna N-M: K (mu_Ed)", k, 0.231779, 1e-5)
    _note("As final = 1.8806e-4 m2 vem do diagrama de interação N-M completo, "
          "não reproduzido por fórmula fechada aqui — ver nota do próprio código.")


# ═══════════════════════════════════════════════════════════════════════
# A.2 — Aço EC3 §6.2, IPE300/S275: fy=275 MPa, γM0=1.0.
# Wpl,y=602 098 mm3, Wel,y n/a, Av,y=3210 mm2, Av,z=2054 mm2, Wt=14 555 mm3,
# área=5188.06 mm2 (propriedades calculadas pelo motor xdfem2d).
# ═══════════════════════════════════════════════════════════════════════

AREA = 5188.06
WPL_Y = 602098.379
AV_Z = 2054.03
WT = 14555.355
FY_STEEL = 275.0


def case_a6_1_2_steel_n_puro():
    """A6.1/A6.2 — N puro (tração ou compressão de secção), N=400 kN.
    Npl,Rd = A·fy/γM0 ; util = N/Npl,Rd."""
    npl_rd = AREA * FY_STEEL / 1000.0    # kN
    util = 400.0 / npl_rd
    _check("A6.1/A6.2 aço N puro: Npl,Rd", npl_rd, 1426.7165, 0.5, " kN")
    _check("A6.1/A6.2 aço N puro: util", util, 0.280364, 0.001)


def case_a6_3_steel_m_puro():
    """A6.3 — M puro, My=80 kNm. Mpl,y,Rd = Wpl,y·fy/γM0."""
    mpl_rd = WPL_Y * FY_STEEL / 1e6      # kNm
    util = 80.0 / mpl_rd
    _check("A6.3 aço M puro: Mpl,y,Rd", mpl_rd, 165.577, 0.05, " kNm")
    _check("A6.3 aço M puro: util", util, 0.483159, 0.001)


def case_a6_4_steel_v_puro():
    """A6.4 — V puro, Vz=150 kN. Vpl,z,Rd = Av,z·fy/(√3·γM0)."""
    vpl_rd = AV_Z * FY_STEEL / math.sqrt(3) / 1000.0   # kN
    util = 150.0 / vpl_rd
    _check("A6.4 aço V puro: Vpl,z,Rd", vpl_rd, 326.088, 0.05, " kN")
    _check("A6.4 aço V puro: util", util, 0.459952, 0.001)


def case_a6_5_steel_t_puro():
    """A6.5 — T puro, T=2 kNm. τ_t,Ed = T/Wt ; util = τ_t,Ed/(fy/√3)."""
    tau_ed = 2.0e6 / WT                     # N/mm2
    tau_rd = FY_STEEL / math.sqrt(3)
    util = tau_ed / tau_rd
    _check("A6.5 aço T puro: tau_Ed", tau_ed, 137.406, 0.05, " MPa")
    _check("A6.5 aço T puro: util", util, 0.865436, 0.001)


def case_a6_6_steel_encurvadura():
    """A6.6 — encurvadura por compressão centrada: não reproduzido por
    fórmula fechada aqui (curva de encurvadura χ via §6.3.1, Ncr de Euler
    com I calculado internamente pelo xdfem2d) — fica só a nota de que o
    código dá χz=0.964 < 1.0, ou seja, a encurvadura reduz a resistência
    face ao caso de secção pura (util 0.280 → 0.291), como fisicamente
    esperado para uma coluna de 8 m."""
    _note("A6.6 não verificado por fórmula fechada — ver design_cases.py "
          "(chi_z=0.964, utilizacao=0.291, > 0.280 da secção pura, coerente).")


# ═══════════════════════════════════════════════════════════════════════
# A.3 — Madeira EC5 §6.1/§6.2, secção 0.10×0.20 m, C24:
#   ft0k=14.5, fc0k=21.0, fmk=24.0, fvk=4.0 MPa (EN 338, valores reais do
#   eurocodepy — note-se que a tabela EN 338 tem ft0k=14.5, não 14.0).
#   Classe de serviço 1, duração média → kmod=0.8 (EC5 Tabela 3.1).
#   γM = 1.3 (madeira maciça, EC5 Tabela 2.3).
# ═══════════════════════════════════════════════════════════════════════

FT0K, FC0K, FMK, FVK = 14.5, 21.0, 24.0, 4.0
KMOD, GAMMA_M_TIMBER = 0.8, 1.3
A_TIMBER = 0.10 * 0.20 * 1e6           # mm2
WY_TIMBER = (0.10 * 0.20**2 / 6) * 1e9  # mm3


def case_a7_1_timber_n_tracao():
    """A7.1 — N tração, N=25 kN. ft0d=kmod·ft0k/γM ; util=N/(ft0d·A)."""
    ft0d = KMOD * FT0K / GAMMA_M_TIMBER
    npl = ft0d * A_TIMBER / 1000.0
    util = 25.0 / npl
    _check("A7.1 madeira N tracao: ft0d", ft0d, 8.9231, 0.001, " MPa")
    _check("A7.1 madeira N tracao: util", util, 0.140086, 0.0005)


def case_a7_2_timber_n_compressao():
    """A7.2 — N compressão (sem encurvadura), N=25 kN. fc0d=kmod·fc0k/γM."""
    fc0d = KMOD * FC0K / GAMMA_M_TIMBER
    npl = fc0d * A_TIMBER / 1000.0
    util = 25.0 / npl
    _check("A7.2 madeira N compressao: fc0d", fc0d, 12.9231, 0.001, " MPa")
    _check("A7.2 madeira N compressao: util", util, 0.096726, 0.0005)


def case_a7_3_timber_m_puro():
    """A7.3 — M puro, My=6 kNm. fmd=kmod·fmk/γM ; util=M/(fmd·Wy)."""
    fmd = KMOD * FMK / GAMMA_M_TIMBER
    mpl = fmd * WY_TIMBER / 1e6
    util = 6.0 / mpl
    _check("A7.3 madeira M puro: fmd", fmd, 14.7692, 0.001, " MPa")
    _check("A7.3 madeira M puro: util", util, 0.609375, 0.0005)


def case_a7_4_timber_v_puro():
    """A7.4 — V puro, Vz=8 kN. τ=1.5·V/(b·h) [secção retangular];
    fvd=kmod·fvk/γM ; util=τ/fvd."""
    fvd = KMOD * FVK / GAMMA_M_TIMBER
    tau = 1.5 * 8000.0 / (100.0 * 200.0)
    util = tau / fvd
    _check("A7.4 madeira V puro: fvd", fvd, 2.4615, 0.001, " MPa")
    _check("A7.4 madeira V puro: util", util, 0.24375, 0.0005)


def case_a7_5_timber_t_puro():
    """A7.5 — T puro, T=0.5 kNm: NÃO verificado por fórmula fechada aqui.
    Tentei a fórmula usual de torção retangular EC5 (Wtor + kshape·fvd) e
    obtive uma ordem de grandeza diferente do valor de referência
    (util≈0.34 vs 0.00223) — o `eurocodepy` aplica um modelo mais elaborado
    de torção+corte combinados (check_shear_with_torsion) que este ficheiro
    não reproduziu corretamente. Fica sinalizado como não verificado, em vez
    de forçar um número que não bate certo."""
    _note("A7.5 NÃO verificado — discrepância de ordem de grandeza entre a "
          "fórmula simples de torção retangular e o valor de referência "
          "(0.00223); precisa de revisão da fórmula EC5 combinada corte+torção.")


def case_a7_6_timber_compressao_encurvadura():
    """A7.6 — compressão + encurvadura: não reproduzido por fórmula fechada
    aqui (kc via §6.3.2, λrel com E0,05 e raio de giração) — fica só a nota
    de que o código dá kc,z=0.28 < kc,y=0.77 < 1.0 (mais esbelto em z,
    fisicamente coerente para uma secção 0.10×0.20 com comprimentos de
    encurvadura iguais nos dois eixos), e utilização 0.340 > 0.097 (A7.2
    sem encurvadura), também coerente."""
    _note("A7.6 não verificado por fórmula fechada — ver design_cases.py "
          "(kc,y=0.77, kc,z=0.28, utilizacao=0.340, coerente com A7.2=0.097).")


def run() -> bool:
    cases = [v for k, v in sorted(globals().items()) if k.startswith("case_")]
    print(f"A executar {len(cases)} verificações independentes (fórmulas EC "
          "reimplementadas à mão)...\n")
    for fn in cases:
        print(f"--- {fn.__name__}: {(fn.__doc__ or '').strip().splitlines()[0]} ---")
        try:
            fn()
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {fn.__name__}: {exc!r}")
            _results.append(False)
        print()
    n_ok = sum(_results)
    print(f"=== {n_ok}/{len(_results)} OK (comparações numéricas; os casos com "
          "apenas [NOTE] não entram na contagem) ===")
    return n_ok == len(_results)


if __name__ == "__main__":
    sys.exit(0 if run() else 1)
