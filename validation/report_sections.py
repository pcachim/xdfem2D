"""Narrative for the validation document's later chapters — springs (Part III)
and dynamics (Part IV).

The chapters for cases 2.1–3.4 are hand-written in the ``.docx`` template and
``make_report.py`` only fills their result columns. The cases added afterwards
(springs, modal, response spectrum) are not in the template: this module holds
their text, their analytical values and their result extractors, and
``make_report.py`` appends them — heading, description, closed-form solution,
comparison table and notes — to the generated document.

Everything is derived from the same constants the tests use, so a change in a
case propagates to the document instead of drifting away from it. The numbers
quoted inside the prose are formatted from those constants for the same reason.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import math

import validation_cases as vc
import modal_cases as mc
from validation_cases import Quantity


# Chapter numbers in the generated document, for cross-references in the prose
# (the chapters themselves are numbered in ``make_report.py``):
#   2 barras · 3 arcos · 4 CST · 5 Allman · 6 molas · 7 dinâmica · 8 conclusões


# ── Number formatting (Portuguese, matching the document) ───────────────────
def n(v: float, dec: int = 2) -> str:
    return f"{v:,.{dec}f}".replace(",", "§").replace(".", ",").replace("§", " ")

def sci(v: float, dec: int = 3) -> str:
    return f"{v:.{dec}g}".replace(".", ",")


@dataclass
class Section:
    """One case in the document: heading, prose, table, notes."""
    model_id: str                     # the .x2d to solve
    titulo: str
    descricao: str
    analitica: list                   # lines under "Solução analítica"
    quantities: list                  # list[Quantity]
    notas: list = field(default_factory=list)

    @property
    def resumo(self):
        """(elemento, ação, grandezas-chave) for the summary table."""
        return RESUMO.get(self.model_id, ("—", "—", "—"))


def _q(label, unit, analytical, extract, to_unit=1.0, signed=False):
    return Quantity(label, unit, analytical, extract,
                    to_unit=to_unit, signed=signed)


def _from_case(case, model_id=None):
    return case.quantities, (model_id or case.id)


# ═══ PART III — springs ════════════════════════════════════════════════════
def _case(cid, cases):
    return next(c for c in cases if c.id == cid)


def spring_sections() -> list[Section]:
    S = []
    CASE = vc.CASE

    # ── s-a1 ────────────────────────────────────────────────────────────
    c = _case("s-a1", vc.CASES)
    S.append(Section(
        "s-a1", "Mola axial num nó, em paralelo com a barra",
        f"Barra de comprimento L = {n(vc.SA1_L)} m encastrada em A, com uma mola "
        f"de translação k = {n(vc.SA1_K, 0)} kN/m no nó livre B e uma carga axial "
        f"P = {n(vc.SA1_P, 0)} kN aplicada em B. A mola liga o nó ao exterior, "
        f"pelo que actua em paralelo com a barra: as duas rigidezes somam-se.",
        [f"Rigidez axial da barra: k_barra = E·A/L = {n(vc.SA1_KB, 0)} kN/m",
         f"Deslocamento: δ = P/(k_barra + k) = {sci(vc.SA1_D * 1000)} mm",
         f"Esforço na barra: N = P·k_barra/(k_barra + k) = {n(vc.SA1_N)} kN",
         f"Força na mola: F = k·δ = {n(vc.SA1_K * vc.SA1_D)} kN",
         "Verificação: N + F = P (equilíbrio do nó B)."],
        c.quantities,
        ["A mola não é um apoio: a sua força não aparece nas reações. O "
         "equilíbrio global do modelo lê-se ΣReações + ΣCargas − ΣForças nas "
         "molas = 0, e é assim que a suite de testes o verifica."]))

    # ── s-a2 ────────────────────────────────────────────────────────────
    c = _case("s-a2", vc.CASES)
    S.append(Section(
        "s-a2", "Consola com apoio elástico na extremidade",
        f"Consola de vão L = {n(vc.SA2_L)} m com carga P = {n(vc.SA2_P, 0)} kN na "
        f"extremidade livre e uma mola vertical k = {n(vc.SA2_K, 0)} kN/m nesse "
        f"mesmo ponto. A rigidez da mola foi escolhida igual à rigidez de ponta "
        f"da consola (3EI/L³), o que torna a solução particularmente legível.",
        [f"Rigidez de ponta da consola: 3EI/L³ = {n(vc.SA2_KB, 0)} kN/m",
         f"Flecha: δ = P/(k + 3EI/L³) = P·L³/(6EI) = {n(vc.SA2_D * 1000)} mm",
         f"Força na mola: F = k·δ = {n(vc.SA2_K * vc.SA2_D)} kN",
         f"Momento no encastramento: M = (P − k·δ)·L = P·L/2 = {n(vc.SA2_M)} kN·m",
         f"Reação vertical: R = P − k·δ = {n(vc.SA2_P - vc.SA2_K * vc.SA2_D)} kN"],
        c.quantities,
        ["Casos-limite úteis para inspeção: k → 0 devolve δ = PL³/3EI e "
         "M = P·L (consola livre); k → ∞ devolve a consola escorada, com "
         "M = 3PL/16 quando a carga está a meio vão. Ambos são verificados "
         "nos testes automáticos."]))

    # ── s-a3 ────────────────────────────────────────────────────────────
    c = _case("s-a3", vc.CASES)
    S.append(Section(
        "s-a3", "Viga com molas de rotação nos apoios",
        f"Viga simplesmente apoiada de vão L = {n(vc.SA3_L)} m com carga "
        f"uniformemente distribuída q = {n(vc.SA3_Q, 0)} kN/m e molas de rotação "
        f"kt = {n(vc.SA3_KT, 0)} kN·m/rad nos dois apoios. É o caso de "
        f"encastramento parcial — a situação real de uma viga ligada a pilares.",
        ["Compatibilidade no apoio: θ = qL³/(24EI) − M·L/(2EI) e M = kt·θ, donde",
         f"M_apoio = qL²/12 · 1/(1 + 2EI/(kt·L)) = {n(vc.SA3_M_END)} kN·m",
         f"M_meio = qL²/8 − M_apoio = {n(vc.SA3_M_MID)} kN·m",
         f"Rotação no apoio: θ = M_apoio/kt = {sci(vc.SA3_THETA)} rad",
         f"Flecha a meio vão: δ = 5qL⁴/(384EI) − M_apoio·L²/(8EI) = "
         f"{n(vc.SA3_D * 1000)} mm"],
        c.quantities,
        [f"Com kt = 2EI/L o encastramento é exactamente parcial a 50 %: "
         f"M_apoio = {n(vc.SA3_M_END)} kN·m contra os {n(vc.SA3_Q * vc.SA3_L ** 2 / 12)} "
         f"kN·m do encastramento perfeito. Os limites kt → 0 (apoio simples, "
         f"M_apoio = 0) e kt → ∞ (qL²/12) são verificados à parte."]))

    # ── s-a4 ────────────────────────────────────────────────────────────
    c = _case("s-a4", vc.CASES)
    S.append(Section(
        "s-a4", "Corpo rígido suspenso apenas em molas (sem apoios)",
        f"Barra praticamente rígida de comprimento L = {n(vc.SA4_L)} m, sem "
        f"qualquer apoio: a estrutura é equilibrada exclusivamente por molas "
        f"nodais — ky = {n(vc.SA4_K, 0)} kN/m nos dois extremos e "
        f"kx = {n(vc.SA4_KX, 0)} kN/m num deles. Cargas P = {n(vc.SA4_P, 0)} kN "
        f"(vertical) e H = {n(vc.SA4_H, 0)} kN (horizontal) aplicadas a "
        f"a = {n(vc.SA4_A)} m do extremo A.",
        ["Sendo o corpo rígido, as forças nas molas saem da estática:",
         f"R_A = P·(L−a)/L = {n(vc.SA4_RA)} kN;  R_B = P·a/L = {n(vc.SA4_RB)} kN",
         f"Assentamentos: w = R/k → w_A = {n(abs(vc.SA4_WA) * 1000)} mm, "
         f"w_B = {n(abs(vc.SA4_WB) * 1000)} mm",
         f"Deslocamento horizontal: u = H/kx = {n(vc.SA4_UX * 1000)} mm"],
        c.quantities,
        ["Este caso verifica duas coisas de uma vez: que um modelo sem apoios é "
         "resolúvel desde que as molas tornem K não-singular, e que o bloco da "
         "mola nodal é diagonal — a carga horizontal não produz deslocamento "
         "vertical nem vice-versa."]))

    # ── s-b1 ────────────────────────────────────────────────────────────
    c = _case("s-b1", vc.CASES)
    S.append(Section(
        "s-b1", "Bloco rígido sobre fundação de Winkler — estado uniforme",
        f"Bloco praticamente rígido de comprimento L = {n(vc.SB1_L)} m assente "
        f"numa fundação de Winkler k = {sci(vc.SB1_K)} kN/m² (kN/m por metro de "
        f"desenvolvimento), com carga uniforme q = {n(vc.SB1_Q, 0)} kN/m. É o "
        f"«patch test» da mola de elemento.",
        [f"Assentamento uniforme: w = q/k = {n(abs(vc.SB1_W) * 1000)} mm",
         "Momento flector: M = 0 e esforço transverso V = 0 (a reação do solo "
         "equilibra a carga ponto a ponto)."],
        c.quantities,
        ["A mola de elemento é agregada de forma *lumped* — k·L/2 em cada nó "
         "extremo — e não como matriz de fundação consistente. Num estado "
         "uniforme isso é irrelevante, porque k·L/2 + k·L/2 = k·L conserva "
         "exactamente a resultante: o assentamento é exacto em qualquer malha.",
         "O momento, porém, não é nulo no modelo discreto: como a reação do "
         "solo é concentrada nos nós, cada elemento «vê» a carga distribuída "
         "desequilibrada e aparece um momento espúrio de q·h²/8 (h = "
         "comprimento do elemento). É artefacto de discretização e não erro de "
         "formulação — verifica-se que se divide exactamente por 4 a cada "
         "refinamento, isto é, O(h²): com 2, 4 e 8 elementos vale "
         f"{n(vc.SB1_Q * (vc.SB1_L / 2) ** 2 / 8)}, "
         f"{n(vc.SB1_Q * (vc.SB1_L / 4) ** 2 / 8)} e "
         f"{n(vc.SB1_Q * (vc.SB1_L / 8) ** 2 / 8)} kN·m."]))

    # ── s-b2 ────────────────────────────────────────────────────────────
    S.append(Section(
        "s-b2-80", "Viga longa sobre fundação elástica — carga pontual",
        f"Viga de comprimento L = {n(vc.SBF_L)} m sobre fundação de Winkler "
        f"k = {sci(vc.SBF_K)} kN/m², com carga pontual P = {n(vc.SBF_P, 0)} kN a "
        f"meio vão e extremidades livres. Solução de referência: Hetényi, "
        f"*Beams on Elastic Foundation*, para a viga infinita.",
        [f"Comprimento elástico: λ = (k/4EI)^(1/4) = {sci(vc.SBF_LAMBDA)} m⁻¹, "
         f"donde λL/2 = {n(vc.SBF_LAMBDA * vc.SBF_L / 2)} > 5 (viga «longa»)",
         f"Flecha sob a carga: w₀ = Pλ/(2k) = {n(vc.SB2_W0 * 1000, 4)} mm",
         f"Momento sob a carga: M₀ = P/(4λ) = {n(vc.SB2_M0)} kN·m",
         "Campo: w(x) = w₀·e^(−λx)·(cos λx + sin λx)"],
        [_q("Flecha sob a carga", "mm", vc.SB2_W0 * 1000,
            lambda r, s: vc.winkler_w0(r, s), to_unit=1000.0),
         _q("Momento sob a carga", "kN·m", vc.SB2_M0,
            lambda r, s: vc.max_abs_M_all(r, vc.CASE))],
        ["Aqui o estado é variável e o *lumping* da fundação faz-se sentir: a "
         "convergência é O(h²). Com 20, 40 e 80 divisões o erro no momento é "
         "−5,6 %, −1,4 % e −0,34 %; a flecha é muito menos sensível (−0,08 %, "
         "+0,002 %, +0,007 %). Regra prática de malha: h ≤ π/(4λ) = "
         f"{n(math.pi / (4 * vc.SBF_LAMBDA))} m.",
         "Os valores da tabela são os da malha mais fina (80 divisões)."]))

    # ── s-b3 ────────────────────────────────────────────────────────────
    S.append(Section(
        "s-b3-80", "Viga semi-infinita sobre fundação elástica — carga no bordo",
        f"A mesma viga e a mesma fundação, com a carga P = {n(vc.SBF_P, 0)} kN "
        f"aplicada agora na extremidade livre. É o caso mais exigente da série, "
        f"porque a solução decai a partir do bordo e a malha tem de resolver o "
        f"comprimento elástico 1/λ = {n(1 / vc.SBF_LAMBDA)} m.",
        [f"Flecha no bordo: w₀ = 2Pλ/k = {n(vc.SB3_W0 * 1000, 4)} mm",
         f"Momento máximo: M_máx = 0,3224·P/λ = {n(vc.SB3_MMAX)} kN·m, em λx = π/4",
         "Campo: w(x) = w₀·e^(−λx)·cos λx"],
        [_q("Flecha no bordo carregado", "mm", vc.SB3_W0 * 1000,
            lambda r, s: vc.winkler_w_end(r, s), to_unit=1000.0),
         _q("Momento máximo", "kN·m", vc.SB3_MMAX,
            lambda r, s: vc.max_abs_M_all(r, vc.CASE))],
        ["Convergência monótona por baixo em ambas as grandezas: com 20, 40 e "
         "80 divisões a flecha vem −9,6 %, −2,6 % e −0,68 %. Os valores da "
         "tabela são os da malha mais fina."]))

    # ── s-b5 ────────────────────────────────────────────────────────────
    c = _case("s-b5", vc.CASES)
    S.append(Section(
        "s-b5", "Mola de elemento em eixos locais — barra inclinada a 45°",
        f"Barra rígida a 45°, de comprimento {n(vc.SB5_LEN)} m, sobre uma "
        f"fundação definida em eixos **locais**: ka = {sci(vc.SB5_KA)} kN/m² na "
        f"direção axial e kt = {sci(vc.SB5_KT)} kN/m² na transversal. Carga "
        f"uniforme q = {n(vc.SB5_Q, 0)} kN/m, vertical.",
        ["O bloco de rigidez da fundação é o bloco local rodado para os eixos "
         "globais, G = R(θ)·diag(ka, kt)·R(θ)ᵀ, isto é",
         "g_xx = ka·cos²θ + kt·sin²θ;  g_xy = (ka − kt)·cosθ·sinθ;  "
         "g_yy = ka·sin²θ + kt·cos²θ",
         f"A 45°: g_xx = g_yy = {sci(vc.SB5_GXX)} e g_xy = {sci(vc.SB5_GXY)} kN/m²",
         "Como corpo rígido, o deslocamento resolve G·(u, v) = (0, −q):",
         f"u = {n(vc.SB5_U * 1000, 3)} mm;  v = {n(vc.SB5_V * 1000, 3)} mm"],
        c.quantities,
        ["O termo g_xy é o essencial deste caso: com ka ≠ kt, uma carga "
         "exclusivamente vertical produz também deslocamento horizontal. Uma "
         "mola definida em eixos globais nunca reproduz este acoplamento.",
         "Quando ka = kt o bloco reduz-se a k·I e as versões «local» e "
         "«global» coincidem até 10⁻¹² — verificado à parte."]))

    # ── s-c1 / s-c2 (unilaterais) ───────────────────────────────────────
    for cid, titulo, desc in (
        ("s-c1a", "Mola só-compressão, activa",
         "carga descendente comprime a mola, que resiste"),
        ("s-c1b", "Mola só-compressão, inactiva",
         "carga ascendente levanta o nó e a mola liberta-se"),
        ("s-c2a", "Mola só-tração, activa",
         "carga ascendente tracciona o tirante, que resiste"),
        ("s-c2b", "Mola só-tração, inactiva",
         "carga descendente deixa o tirante em folga"),
    ):
        c = _case(cid, vc.SPRING_NL_CASES)
        activa = cid.endswith("a")
        S.append(Section(
            cid, f"Molas unilaterais — {titulo.lower()}",
            f"Consola de vão L = {n(vc.SC1_L)} m com carga "
            f"P = {n(vc.SC1_P, 0)} kN na extremidade livre e uma mola "
            f"k = {n(vc.SC1_K, 0)} kN/m nesse ponto, declarada unilateral "
            f"({'só-compressão' if 'c1' in cid else 'só-tração'}). Neste caso a "
            f"{desc}. O modelo traz dois casos de análise — NonLinear e "
            f"Linear — pelo que a referência bilateral vem do mesmo ficheiro.",
            ([f"Mola activa: δ = P/(k + 3EI/L³) = {n(vc.SC1_D_ACTIVE * 1000)} mm, "
              f"igual ao resultado bilateral"]
             if activa else
             [f"Mola inactiva: a estrutura responde como a consola sem mola, "
              f"δ = P·L³/(3EI) = {n(vc.SC1_D_FREE * 1000)} mm",
              "Força na mola: F = 0"]),
            c.quantities,
            ["Só os casos de análise **NonLinear** respeitam os modos "
             "unilaterais; um caso Linear trata sempre a mola como bilateral. "
             "A solução não-linear é obtida por iteração de conjunto activo — "
             "a mola entra na rigidez apenas enquanto está activa."] if activa
            else []))

    # ── s-c3 ────────────────────────────────────────────────────────────
    S.append(Section(
        "s-c3-48", "Sapata rígida com contacto parcial (fundação só-compressão)",
        f"Sapata rígida de comprimento L = {n(vc.SC3_L)} m sobre fundação de "
        f"Winkler k = {sci(vc.SC3_K)} kN/m² declarada **só-compressão**, com "
        f"carga vertical N = {n(vc.SC3_N, 0)} kN aplicada com excentricidade "
        f"e = {n(vc.SC3_E)} m. Como e > L/6 = {n(vc.SC3_L / 6)} m, a resultante "
        f"sai do núcleo central e a sapata descola do solo numa parte da base.",
        [f"Comprimento de contacto: a = 3·(L/2 − e) = {n(vc.SC3_A)} m, medido a "
         f"partir do bordo carregado",
         f"Diagrama de pressões triangular, com p_máx = 2N/a = "
         f"{n(vc.SC3_PMAX)} kN/m",
         f"Assentamento no bordo carregado: w_máx = p_máx/k = "
         f"{n(abs(vc.SC3_WMAX) * 1000, 3)} mm"],
        [_q("Assentamento no bordo carregado", "mm", abs(vc.SC3_WMAX) * 1000,
            lambda r, s: vc.c3_w_edge(r, s), to_unit=1000.0),
         _q("Comprimento de contacto", "m", vc.SC3_A,
            lambda r, s: vc.c3_contact_length(r, s))],
        ["É o caso que justifica a análise não-linear: o modelo linear do mesmo "
         "ficheiro mantém toda a base em contacto e **tracciona** o solo no "
         "bordo oposto, o que é fisicamente impossível.",
         "O comprimento de contacto discreto acerta o teórico dentro de um "
         "elemento; o assentamento converge por baixo (12, 24 e 48 divisões: "
         "−1,1 %, −0,29 %, −0,075 %). A tabela mostra a malha mais fina."]))

    return S


# ═══ PART IV — dynamics ════════════════════════════════════════════════════
def _freq_q(label, target, index, kind="y", skip_rigid=False):
    def extract(r, s, index=index, kind=kind, skip_rigid=skip_rigid):
        f = mc.freqs_of_kind(r, kind)
        if skip_rigid:
            f = [x for x in f if x > 1.0]
        return f[index]
    return _q(label, "Hz", target, extract)


def modal_sections() -> list[Section]:
    S = []

    # ── m-a1 ────────────────────────────────────────────────────────────
    c = _case("m-a1", mc.MODAL_CASES)
    S.append(Section(
        "m-a1", "Oscilador de um grau de liberdade",
        f"Consola sem peso próprio, de vão L = {n(mc.MA1_L)} m, com uma massa "
        f"concentrada m = {n(mc.MA1_M)} t na extremidade livre. É o caso de "
        f"referência da análise modal: sendo toda a massa nodal, o resultado é "
        f"exacto e independente da malha.",
        [f"Rigidez de ponta (flexão): k = 3EI/L³ = {n(mc.MA1_KB, 0)} kN/m",
         f"f₁ = √(k/m)/2π = {n(mc.MA1_F_BEND, 4)} Hz;  "
         f"T₁ = {n(1 / mc.MA1_F_BEND, 4)} s",
         f"Rigidez axial: EA/L = {n(mc.MA1_KA, 0)} kN/m → "
         f"f₂ = {n(mc.MA1_F_AXIAL, 3)} Hz (modo axial)",
         "Massa efectiva do 1.º modo: m_ef = m (100 % da massa)"],
        c.quantities,
        ["A matriz de massa do programa é diagonal (*lumped*), construída a "
         "partir de |Fy|/g dos casos de carga referidos pelo caso Mass mais as "
         "massas nodais concentradas. Toda a massa aqui é nodal, pelo que "
         "refinar a barra de 1 para 8 elementos não altera a frequência — "
         "verificado nos testes.",
         "Os modos de ordem superior que o programa devolve para este modelo "
         "correspondem aos graus de liberdade de rotação, que não têm massa e "
         "são regularizados numericamente: aparecem a ~10⁴–10⁵ vezes f₁ e não "
         "têm significado físico."]))

    # ── m-a2 ────────────────────────────────────────────────────────────
    c = _case("m-a2", mc.MODAL_CASES)
    S.append(Section(
        "m-a2", "Um grau de liberdade com mola nodal",
        f"O mesmo oscilador, agora com uma mola k = {n(mc.MA2_KS, 0)} kN/m a "
        f"ligar a extremidade livre ao exterior. Verifica que as molas entram "
        f"na matriz de rigidez do problema de valores próprios, e não apenas na "
        f"análise estática.",
        [f"Rigidez equivalente (molas em paralelo): k_eq = k_mola + 3EI/L³ = "
         f"{n(mc.MA2_KEQ, 0)} kN/m",
         f"f₁ = √(k_eq/m)/2π = {n(mc.MA2_F, 4)} Hz"],
        c.quantities))

    # ── m-a3 ────────────────────────────────────────────────────────────
    c = _case("m-a3", mc.MODAL_CASES)
    S.append(Section(
        "m-a3", "Duas massas concentradas numa consola (2 GL)",
        f"Consola sem peso próprio de vão L = {n(mc.MA3_L)} m com massas "
        f"m₁ = {n(mc.MA3_M[0])} t e m₂ = {n(mc.MA3_M[1])} t em "
        f"x = {n(mc.MA3_X[0])} m e x = {n(mc.MA3_X[1])} m.",
        ["A referência é o problema de valores próprios 2×2 construído a partir "
         "da matriz de flexibilidade da consola,",
         "f_ij = x²·(3y − x)/(6EI), com x = min(xᵢ, xⱼ) e y = max(xᵢ, xⱼ),",
         "resolvendo det(F·M − ω⁻²·I) = 0. É uma verificação cruzada genuína: a "
         "solução de referência não usa nada do modelo de elementos finitos.",
         f"f₁ = {n(mc.MA3_F[0], 4)} Hz;  f₂ = {n(mc.MA3_F[1], 4)} Hz;  "
         f"f₂/f₁ = {n(mc.MA3_F[1] / mc.MA3_F[0], 4)}"],
        c.quantities,
        ["Verifica-se ainda a ortogonalidade dos modos em relação à massa "
         "(φ₁ᵀ·M·φ₂ = 0) e a ordenação crescente das frequências."]))

    # ── m-a4 ────────────────────────────────────────────────────────────
    c = _case("m-a4", mc.MODAL_CASES)
    S.append(Section(
        "m-a4", "Bloco rígido sobre fundação elástica — translação e rotação",
        f"Bloco rígido de comprimento L = {n(mc.MA4_L)} m e massa "
        f"M = {n(mc.MA4_M)} t sobre uma fundação k = {sci(mc.MA4_K)} kN/m², "
        f"discretizado em 4 elementos. Cada nó recebe a massa da sua área de "
        f"influência e a respectiva inércia de rotação mᵢ·h²/12.",
        ["No meio contínuo os dois modos rígidos são **degenerados**:",
         f"translação ω² = kL/M e rotação ω² = (kL³/12)/(ML²/12) = kL/M, ou "
         f"seja f = {n(mc.MA4_F_HEAVE, 4)} Hz em ambos.",
         "No modelo discreto a rotação não coincide, porque a inércia de "
         "rotação tem duas parcelas — as translações dos nós e o *spin* de cada "
         "segmento em torno do seu centróide:",
         "ω²_rot = (k/m̄)·S/(S + M·h²/12), com S = Σ mᵢ·xᵢ²,",
         f"o que dá f_rot = {n(mc.ma4_f_rock(4), 4)} Hz com 4 elementos, "
         f"convergindo para o valor do contínuo quando h → 0 (2, 4 e 8 "
         f"elementos: {n((mc.ma4_f_rock(2) / mc.MA4_F_HEAVE - 1) * 100, 1)} %, "
         f"{n((mc.ma4_f_rock(4) / mc.MA4_F_HEAVE - 1) * 100, 1)} %, "
         f"{n((mc.ma4_f_rock(8) / mc.MA4_F_HEAVE - 1) * 100, 2)} %)."],
        c.quantities,
        ["**A inércia de rotação tem de ser dada explicitamente.** Sem ela, o "
         "modo de rotação passa a ser suportado pelos graus de liberdade de "
         "rotação, que não têm massa e são regularizados numericamente; o "
         "condicionamento do problema de valores próprios degrada-se e a "
         "frequência obtida varia com a rigidez do bloco e até com a "
         "biblioteca de álgebra linear da máquina. Com mtz = mᵢ·h²/12 o mesmo "
         "resultado é estável a 10⁻⁵."]))

    # ── m-b1 ────────────────────────────────────────────────────────────
    S.append(Section(
        "m-b1-16", "Viga simplesmente apoiada — massa distribuída",
        f"Viga de vão L = {n(mc.MB_L)} m com massa distribuída obtida do peso "
        f"próprio (γ = {n(mc.GAMMA_C)} kN/m³, A = {n(vc.A_BAR, 4)} m² → "
        f"m̄ = γA/g = {n(mc.MBAR, 4)} t/m), discretizada em 16 elementos.",
        ["Solução de Euler-Bernoulli:",
         "ω_n = (nπ/L)²·√(EI/m̄),  f_n = ω_n/2π",
         f"f₁ = {n(mc.MB1_F[0], 4)} Hz;  f₂ = {n(mc.MB1_F[1], 4)} Hz;  "
         f"f₃ = {n(mc.MB1_F[2], 4)} Hz"],
        [_freq_q("f₁ (1.º modo de flexão)", mc.MB1_F[0], 0),
         _freq_q("f₂ (2.º modo de flexão)", mc.MB1_F[1], 1),
         _freq_q("f₃ (3.º modo de flexão)", mc.MB1_F[2], 2)],
        ["Com massa distribuída a convergência é O(h²) e faz-se **por baixo**: "
         "a matriz de massa concentrada não pode sobrestimar a frequência. Com "
         "4, 8 e 16 elementos o erro no 1.º modo é −0,031 %, −0,002 % e "
         "−0,000 %; nos modos superiores é maior, como seria de esperar.",
         "Atenção à ordenação: o programa ordena os modos por frequência, pelo "
         "que os modos axiais aparecem intercalados com os de flexão — nesta "
         "viga o 3.º modo devolvido é axial (142,9 Hz), não flexural."]))

    # ── m-b2 ────────────────────────────────────────────────────────────
    S.append(Section(
        "m-b2-16", "Consola — massa distribuída",
        f"A mesma viga, agora em consola, discretizada em 16 elementos. É o "
        f"caso mais exigente da série em termos de convergência.",
        ["f_n = (β_n·L)²·√(EI/m̄L⁴)/2π, com β_n·L raízes de cos·cosh + 1 = 0:",
         f"β₁L = {n(mc.MB2_BETA[0], 4)} → f₁ = {n(mc.MB2_F[0], 4)} Hz",
         f"β₂L = {n(mc.MB2_BETA[1], 4)} → f₂ = {n(mc.MB2_F[1], 4)} Hz",
         f"β₃L = {n(mc.MB2_BETA[2], 4)} → f₃ = {n(mc.MB2_F[2], 4)} Hz"],
        [_freq_q("f₁", mc.MB2_F[0], 0),
         _freq_q("f₂", mc.MB2_F[1], 1),
         _freq_q("f₃", mc.MB2_F[2], 2)],
        ["Convergência O(h²) por baixo: com 4, 8 e 16 elementos o erro no 1.º "
         "modo é −2,8 %, −0,71 % e −0,18 %, dividindo-se por ≈4 a cada "
         "refinamento. Para os modos superiores a regra prática habitual — "
         "pelo menos 6 a 8 elementos por meia-onda do modo — continua a valer."]))

    # ── m-b3 ────────────────────────────────────────────────────────────
    S.append(Section(
        "m-b3-32", "Viga livre-livre",
        f"A mesma viga sem apoios, em 32 elementos. Como um modelo sem apoios "
        f"nem sequer se resolve estaticamente (matriz de rigidez singular), "
        f"usa-se o recurso clássico de molas muito flexíveis: as frequências de "
        f"corpo rígido ficam três ordens de grandeza abaixo da primeira "
        f"frequência elástica e não a perturbam.",
        ["β_n·L são as raízes de cos·cosh − 1 = 0:",
         f"β₁L = {n(mc.MB3_BETA[0], 4)} → f₁ = {n(mc.MB3_F[0], 4)} Hz",
         f"β₂L = {n(mc.MB3_BETA[1], 4)} → f₂ = {n(mc.MB3_F[1], 4)} Hz"],
        [_freq_q("f₁ (1.º modo elástico)", mc.MB3_F[0], 0, skip_rigid=True),
         _freq_q("f₂ (2.º modo elástico)", mc.MB3_F[1], 1, skip_rigid=True)],
        ["Além da convergência, verifica-se a separação: os modos de corpo "
         "rígido têm de ficar abaixo de 1 % da primeira frequência elástica. "
         "O número de modos de corpo rígido efectivamente listados varia com a "
         "malha, porque o filtro interno de valores próprios nulos é "
         "sensível — daí que a verificação seja feita por frequência e não por "
         "índice de modo."]))

    # ── m-b4 ────────────────────────────────────────────────────────────
    S.append(Section(
        "m-b4-32", "Barra em vibração axial",
        f"A mesma consola, olhando agora para os modos **axiais** (dominados "
        f"por u_x), em 32 elementos. ρ = γ/g = {n(mc.MB4_RHO, 4)} t/m³.",
        ["Barra encastrada-livre em vibração longitudinal:",
         "ω_n = (2n − 1)·π/(2L)·√(E/ρ)",
         f"c = √(E/ρ) = {n(math.sqrt(vc.E / mc.MB4_RHO), 0)} m/s",
         f"f₁ = {n(mc.MB4_F[0], 3)} Hz;  f₂ = {n(mc.MB4_F[1], 3)} Hz"],
        [_freq_q("f₁ (1.º modo axial)", mc.MB4_F[0], 0, kind="x"),
         _freq_q("f₂ (2.º modo axial)", mc.MB4_F[1], 1, kind="x")],
        ["Os modos axiais são identificados pela componente dominante da "
         "deformada, não pela ordem: numa viga horizontal deste tipo o 1.º modo "
         "axial surge muito acima dos primeiros modos de flexão."]))

    # ── m-b5 ────────────────────────────────────────────────────────────
    S.append(Section(
        "m-b5-32", "Viga sobre fundação elástica — vibração",
        f"Viga simplesmente apoiada de vão L = {n(mc.MB_L)} m com massa "
        f"distribuída, assente numa fundação de Winkler k = {sci(mc.MB5_K)} "
        f"kN/m². Cruza a análise modal com as molas de elemento.",
        ["A fundação acrescenta um termo constante ao quadrado da frequência:",
         "ω_n² = [EI·(nπ/L)⁴ + k]/m̄",
         f"f₁ = {n(mc.MB5_F[0], 4)} Hz;  f₂ = {n(mc.MB5_F[1], 4)} Hz"],
        [_freq_q("f₁", mc.MB5_F[0], 0),
         _freq_q("f₂", mc.MB5_F[1], 1)],
        ["Note-se o efeito da fundação: o 1.º modo sobe de "
         f"{n(mc.MB1_F[0], 2)} Hz para {n(mc.MB5_F[0], 2)} Hz, enquanto o 2.º "
         f"quase não se altera — o termo k é fixo e a rigidez de flexão cresce "
         f"com n⁴."]))

    # ── massas efectivas ────────────────────────────────────────────────
    def _meff(i):
        def extract(r, s, i=i):
            ys = [m for k, m in enumerate(mc.modes(r), start=1)
                  if mc.mode_kind(r, k) == "y"]
            return ys[i]["meff_y"]
        return extract

    S.append(Section(
        "m-b2-16", "Massas efectivas e factores de participação",
        f"Sobre o modelo da consola (16 elementos, massa total "
        f"m̄·L = {n(mc.MC_TOTAL, 4)} t), verificam-se as massas efectivas "
        f"modais, que são o que governa o corte basal numa análise sísmica e o "
        f"critério de truncatura de modos.",
        ["Para uma consola com massa uniforme, as fracções clássicas da massa "
         "total são 0,613, 0,188 e 0,065 nos três primeiros modos:",
         f"m_ef,1 = {n(mc.MC_CANT_FRACTIONS[0] * mc.MC_TOTAL, 4)} t;  "
         f"m_ef,2 = {n(mc.MC_CANT_FRACTIONS[1] * mc.MC_TOTAL, 4)} t;  "
         f"m_ef,3 = {n(mc.MC_CANT_FRACTIONS[2] * mc.MC_TOTAL, 4)} t",
         "Adicionalmente, a soma das massas efectivas sobre **todos** os modos "
         "tem de igualar a massa nos graus de liberdade livres."],
        [_q("Massa efectiva do 1.º modo", "t",
            mc.MC_CANT_FRACTIONS[0] * mc.MC_TOTAL, _meff(0)),
         _q("Massa efectiva do 2.º modo", "t",
            mc.MC_CANT_FRACTIONS[1] * mc.MC_TOTAL, _meff(1)),
         _q("Massa efectiva do 3.º modo", "t",
            mc.MC_CANT_FRACTIONS[2] * mc.MC_TOTAL, _meff(2))],
        ["**As percentagens de massa efectiva são relativas à massa dos graus "
         "de liberdade livres**, não à massa total do modelo: a massa agregada "
         "a um nó apoiado não pode vibrar e é excluída (transmite-se "
         "directamente à fundação). O denominador está escolhido para que as "
         "percentagens de todos os modos somem 100 %, o que faz da coluna um "
         "indicador fiável de truncatura de modos.",
         "A fracção excluída depende da malha, não da estrutura: numa viga "
         "simplesmente apoiada com n elementos os nós apoiados levam exactamente "
         "1/n da massa (6,3 % com 16 elementos). Para o critério dos 90 % do "
         "EC8, que é sobre a massa **total**, os resultados incluem também "
         "model_mass_x/y e restrained_mass_x/y; e quando a massa apoiada excede "
         "2 % o caso modal emite um aviso.",
         "Numa viga simplesmente apoiada os modos antissimétricos (2.º, 4.º, …) "
         "têm massa efectiva **exactamente nula** e o 1.º modo vale 8/π² = "
         "81,06 % da massa total — também verificado."]))

    # ── m-e1 espectro ───────────────────────────────────────────────────
    S.append(Section(
        "m-e1", "Espectro de resposta — verificação da cadeia de cálculo",
        f"O oscilador de 1 GL do caso m-a1 (m = {n(mc.ME_M)} t) sujeito a um "
        f"espectro **constante** Sa = {n(mc.ME_A0)} m/s², combinação SRSS. O "
        f"mesmo modelo contém um caso Linear com a carga estática equivalente "
        f"F = m·Sa = {n(mc.ME_M * mc.ME_A0)} kN.",
        ["Com espectro constante, Sd = Sa/ω² e a resposta modal Γ·Sd·φ colapsa "
         "exactamente na resposta estática sob F = m·Sa:",
         f"δ (direção Y) = m·Sa/(3EI/L³) = {n(mc.ME1_UY * 1000, 4)} mm",
         f"δ (direção X) = m·Sa/(EA/L) = {sci(mc.ME1_UX * 1000)} mm",
         "É a verificação mais apertada possível da cadeia Mass → Modal → "
         "Spectrum, porque compara duas vias de cálculo completamente "
         "diferentes dentro do mesmo modelo."],
        [_q("Deslocamento espectral em Y", "mm", mc.ME1_UY * 1000,
            lambda r, s: r["analysis_cases"]["SPY"]["displacements"]["B"][1],
            to_unit=1000.0),
         _q("Deslocamento estático equivalente em Y", "mm", mc.ME1_UY * 1000,
            lambda r, s: r["analysis_cases"]["LIN"]["displacements"]["B"][1],
            to_unit=1000.0),
         _q("Deslocamento espectral em X", "mm", mc.ME1_UX * 1000,
            lambda r, s: r["analysis_cases"]["SPX"]["displacements"]["B"][0],
            to_unit=1000.0)],
        ["Verificam-se ainda, nos testes automáticos: a direção XY como SRSS "
         "das direções X e Y; a coincidência de CQC com SRSS quando os modos "
         "estão bem separados (f₂/f₁ = 5,15 com ξ = 5 %); e a interpolação "
         "linear de Sa(T) entre pontos do espectro."]))

    # ── m-c1 — membrana (triângulos): CST vs ES-FEM ─────────────────────
    S.append(Section(
        "m-c1-es-16x4",
        "Membrana em consola (triângulos) — CST vs ES-FEM",
        f"Contraparte modal do Caso 4.2: uma parede-consola esbelta "
        f"(L = {n(mc.MC1_L)} m, h = {n(mc.MC1_H)} m, t = {n(mc.MC1_T)} m), "
        f"encastrada num bordo e malhada em triângulos T3. A massa é o peso "
        f"próprio distribuído (concentrada nos nós), igual em qualquer malha, "
        f"pelo que só a formulação de rigidez muda entre os dois modelos.",
        [f"Referência de teoria de vigas (Euler–Bernoulli, flexão no plano): "
         f"f₁ = (β₁²/2π)·√(EI/(m̄L⁴)) = {n(mc.MC1_F1_BEAM, 3)} Hz, com "
         f"β₁L = 1,875, I = t·h³/12 e m̄ = γ·t·h/g.",
         "É uma parede profunda (L/h = 8): a deformação por corte baixa o valor "
         "real face à viga de Euler–Bernoulli, pelo que este serve de referência "
         "de convergência e não como valor exacto a atingir."],
        [_q("f₁ (ES-FEM, 16×4)", "Hz", mc.MC1_F1_BEAM,
            lambda r, s: mc.freq(r, 1))],
        ["O CST é rígido a mais e **sobrestima** as frequências; a suavização "
         "por arestas do ES-FEM relaxa esse excesso, dando frequências mais "
         "baixas e mais próximas do valor convergido. Nas famílias de malha "
         "(m-c1-cst-8×2/16×4 e m-c1-es-8×2/16×4, com a referência ES-FEM 48×12) "
         "verifica-se que, à mesma malha, f₁(ES-FEM) < f₁(CST) e que o ES-FEM "
         "converge mais depressa.",
         "Ao contrário do NS-FEM (baseado nos nós), o ES-FEM é temporalmente "
         "estável — a rigidez tem exactamente os 3 modos de corpo rígido e "
         "nenhum modo espúrio de energia nula —, pelo que é adequado a análise "
         "modal. A matriz de massa é concentrada (lumped), como nos restantes "
         "casos, por isso o ganho de precisão vem do lado da rigidez."]))

    return S


# ═══ Curved geometry (its own chapter, inside Part I) ══════════════════════
def arc_sections() -> list[Section]:
    """Structures generated from a curved geometry object."""
    S = []

    # ── arco ────────────────────────────────────────────────────────────
    c = _case("arc", vc.CASES)
    S.append(Section(
        "arc", "Consola em quarto de círculo — carga na extremidade livre",
        f"Arco de raio R = {n(vc.ARC_R)} m descrevendo um quarto de círculo, "
        f"encastrado em (R, 0) e livre em (0, R), com carga vertical "
        f"P = {n(vc.ARC_P, 0)} kN na extremidade livre. O arco é definido como "
        f"objecto de geometria e discretizado em 64 barras rectas, pelo que este "
        f"caso valida ao mesmo tempo a geração de malha a partir de um objecto "
        f"curvo e o comportamento de uma estrutura de eixo curvo.",
        ["Solução analítica por Castigliano, considerando apenas a energia de "
         "flexão (peça esbelta):",
         f"δ_v = π·P·R³/(4·E·I) = {n(vc.ARC_TARGET, 4)} mm",
         f"Momento no encastramento: M = P·R = {n(vc.ARC_P * vc.ARC_R)} kN·m",
         f"Reação vertical: R_v = P = {n(vc.ARC_P)} kN",
         "O momento e a reação são de equilíbrio, logo exactos "
         "independentemente da discretização; a flecha é que depende dela."],
        c.quantities,
        ["A convergência faz-se **por cima** e não é monótona no sentido "
         "habitual: o modelo de cordas rectas inclui as deformações axial e de "
         "corte, que a solução analítica despreza, pelo que a flecha calculada "
         "é ligeiramente superior à teórica (16, 32 e 64 divisões: +0,03 %, "
         "+0,18 %, +0,22 %). O que converge não é para o valor de flexão pura, "
         "mas para a solução exacta do arco com axial e corte — que é "
         "efectivamente um pouco mais flexível.",
         "Por isso a tolerância deste caso é de 1,5 % e o que se verifica é a "
         "estabilização do valor, não a sua anulação."]))

    return S


# ═══ Allman triangle (its own chapter, inside Part II) ═════════════════════
def allman_sections() -> list[Section]:
    """The second plane element: a triangle with a drilling rotation per node."""
    S = []

    # ── Allman: patch de tração ─────────────────────────────────────────
    c = _case("3.1-allman", vc.CASES)
    S.append(Section(
        "3.1-allman", "Elemento de Allman — patch test de tração uniforme",
        "O mesmo patch test do Caso 4.1 (placa 2,00 × 1,00 m, t = 0,10 m, "
        "tração de 100 kN no bordo), agora com o triângulo de **Allman** em vez "
        "do CST. O elemento de Allman acrescenta um grau de liberdade de "
        "rotação no plano (*drilling*, θz) a cada nó, o que lhe dá um campo de "
        "deslocamentos mais rico do que o do CST a igual número de nós.",
        ["Estado uniaxial uniforme, idêntico ao do Caso 4.1:",
         "σ_x = F/(h·t) = 100/(1,00 × 0,10) = 1 000 kN/m² = 1,00 MPa",
         "σ_y = 0;  τ_xy = 0"],
        c.quantities,
        ["**O drilling tem de estar restringido para que o patch seja bem "
         "posto.** Num estado de extensão constante a rotação no plano é nula, "
         "pelo que prescrevê-la (θz = 0 em todos os nós) faz parte do patch "
         "test de um elemento com este grau de liberdade. Com θz fixo o "
         "elemento reproduz o estado exactamente; deixá-lo livre na fronteira "
         "torna o patch mal-posto — o que não é uma falha do elemento.",
         "O patch test canónico de Allman (1984), com um nó interior livre e a "
         "fronteira totalmente prescrita em u, v **e θ**, é verificado à parte "
         "na suite automática: translação de corpo rígido, rotação de corpo "
         "rígido e extensão constante uniaxial, biaxial e de corte puro são "
         "reproduzidas com erro ≤ 10⁻¹²."]))

    # ── Allman: deslocamento imposto ────────────────────────────────────
    c = _case("3.3-allman", vc.CASES)
    S.append(Section(
        "3.3-allman", "Elemento de Allman — deslocamento imposto num bordo",
        "O Caso 4.3 com o elemento de Allman: a mesma placa com um "
        "deslocamento imposto u_x = 1,00 mm no bordo x = L, o drilling "
        "restringido, e o apoio relaxado a u_x no bordo x = 0 (como no caso "
        "CST correspondente) para que o estado seja verdadeiramente uniaxial.",
        ["ε_x = δ/L = 0,001/2,00 = 5,0×10⁻⁴",
         "σ_x = E·ε_x = 30×10⁶ × 5,0×10⁻⁴ = 15 000 kN/m² = 15,0 MPa",
         "Reação total no bordo apoiado: R = σ_x·h·t = 15 000 × 1,00 × 0,10 = "
         "1 500 kN"],
        c.quantities,
        ["Tal como o caso anterior, é exacto em qualquer malha desde que o "
         "drilling esteja restringido."]))

    # ── Allman: térmico ─────────────────────────────────────────────────
    c = _case("3.4-allman", vc.CASES)
    S.append(Section(
        "3.4-allman", "Elemento de Allman — variação uniforme de temperatura",
        "O patch test térmico do Caso 4.4 com o elemento de Allman: placa "
        "livre de restrições redundantes sujeita a ΔT = +30 °C. Aqui o drilling "
        "fica **livre**, porque uma dilatação livre não tem rotação a "
        "restringir.",
        ["Dilatação livre: ε₀ = α·ΔT = 1,0×10⁻⁵ × 30 = 3,0×10⁻⁴",
         "σ = 0 em todos os elementos (não há restrição que impeça a dilatação)",
         "δ_x = ε₀·L = 3,0×10⁻⁴ × 2,00 = 0,60 mm;  "
         "δ_y = ε₀·h = 3,0×10⁻⁴ × 1,00 = 0,30 mm"],
        c.quantities,
        ["A carga térmica equivalente e a recuperação de tensões cobrem os 9 "
         "graus de liberdade do elemento (f_th = t·A·B₀ᵀ·D·ε₀; "
         "σ = D·(B·u − ε₀)). Uma tensão não nula neste caso indicaria que a "
         "deformação inicial não está a ser subtraída na relação constitutiva."]))

    # ── Allman vs CST em flexão ─────────────────────────────────────────
    S.append(Section(
        "3.2a-32x8", "Elemento de Allman em flexão — comparação com o CST",
        f"A consola-parede do Caso 4.2 (L = 4,00 m, h = 0,50 m, t = 0,20 m, "
        f"P = 50 kN na extremidade), agora com o elemento de Allman e o "
        f"drilling **livre** — que é onde este elemento ganha ao CST. Malha de "
        f"32 × 8 elementos.",
        [f"Alvo da teoria de vigas (Timoshenko, com deformação por corte): "
         f"δ = {n(vc.CASE_3_2_TARGET)} mm.",
         "Não é uma solução exacta de elasticidade plana, pelo que serve de "
         "referência para a convergência e não como valor a atingir "
         "exactamente."],
        [_q("Flecha na extremidade", "mm", vc.CASE_3_2_TARGET,
            lambda r, s: vc.case_3_2_tip_deflection(r, s))],
        ["É esta a razão de ser do grau de liberdade de rotação: à mesma malha "
         "de 32 × 8, o Allman fica a −1,6 % do alvo enquanto o CST fica a "
         "−10,2 %. Por outras palavras, o Allman com 16 × 4 (−5,3 %) já é "
         "melhor do que o CST com 32 × 8.",
         "Ambos convergem por baixo, porque ambos sobrestimam a rigidez de "
         "flexão: Allman com 8 × 2, 16 × 4 e 32 × 8 dá −16,8 %, −5,3 % e "
         "−1,6 %; CST com 16 × 4, 32 × 8 e 64 × 16 dá −30,8 %, −10,2 % e "
         "−2,8 %.",
         "Nota de modelação: em problemas de membrana com extensão constante "
         "convém restringir o drilling (casos anteriores); em problemas de "
         "flexão deve deixar-se livre, que é o que aqui se faz."]))

    # ── ES-FEM vs CST em flexão ─────────────────────────────────────────
    S.append(Section(
        "3.2e-32x8", "Elemento ES-FEM em flexão — comparação com o CST",
        f"A mesma consola-parede (L = 4,00 m, h = 0,50 m, t = 0,20 m, "
        f"P = 50 kN na extremidade), agora com o elemento **ES-FEM** "
        f"(CST suavizado por arestas, 2 GL/nó — sem grau de liberdade de "
        f"rotação). Malha de 32 × 8 elementos.",
        [f"Alvo da teoria de vigas (Timoshenko, com deformação por corte): "
         f"δ = {n(vc.CASE_3_2_TARGET)} mm.",
         "Como nos casos anteriores, é uma referência de convergência e não "
         "um valor de elasticidade plana a atingir exactamente."],
        [_q("Flecha na extremidade", "mm", vc.CASE_3_2_TARGET,
            lambda r, s: vc.case_3_2_tip_deflection(r, s))],
        ["O ES-FEM suaviza a rigidez excessiva do CST montando a rigidez sobre "
         "domínios de aresta, pelo que converge para a teoria de vigas mais "
         "depressa — como o Allman, mas sem introduzir o grau de liberdade de "
         "drilling. Ambos convergem por baixo (sobrestimam a rigidez de flexão).",
         "Comparação à mesma malha (famílias 3.2e-8×2/16×4/32×8 vs. as do CST "
         "3.2-16×4/32×8/64×16): o ES-FEM aproxima-se do alvo com menos "
         "elementos que o CST. É a vantagem principal de uma malha só de "
         "triângulos, sem mudar o número de graus de liberdade nem o solver."]))

    return S


# ═══ Plate bending (out-of-plane) — its own chapter, inside Part II ═════════
def plate_sections() -> list[Section]:
    """The plate-bending triangles (DKT thin-plate, MITC3 shear-deformable),
    validated against the classical Navier and Mindlin plate series."""
    S = []

    # ── P.1 — thin slab, DKT vs Navier ──────────────────────────────────
    c = _case("P.1", vc.PLATE_CASES)
    S.append(Section(
        "P.1", "Laje fina simplesmente apoiada — DKT vs série de Navier",
        f"Laje quadrada de {n(vc.PLATE_A, 0)} × {n(vc.PLATE_A, 0)} m, espessura "
        f"t = {n(vc.PLATE_THIN_T, 2)} m (vão/espessura = 100), simplesmente "
        f"apoiada nos quatro bordos e sob uma pressão uniforme "
        f"p = {n(abs(vc.PLATE_PZ), 0)} kN/m². Malha de {vc.PLATE_N} × "
        f"{vc.PLATE_N} com o triângulo **DKT** (placa fina de Kirchhoff). O "
        f"apoio é do tipo *hard* (w = 0 e rotação tangencial nula no bordo), "
        f"que é a condição que a série de Navier assume.",
        ["Flecha central da placa fina (Kirchhoff), série de Navier:",
         "w = (16 p / π⁶ D) · Σ_{m,k ímpares} sin(mπ/2)sin(kπ/2) / "
         "[m·k·(m²/a² + k²/a²)²],  com D = E·t³/12(1−ν²)",
         f"w = {n(vc.PLATE_THIN_W * 1000.0, 3)} mm"],
        c.quantities,
        ["O DKT não tem deformação por corte transverso, pelo que é o elemento "
         "natural para placa fina: à malha usada reproduz a série de Navier com "
         "erro inferior a 0,1 %.",
         "A recuperação por elemento dá também os momentos mx, my, mxy, os "
         "momentos principais e os momentos de dimensionamento de Wood-Armer."]))

    # ── P.2 — thick slab, MITC3 vs Mindlin ──────────────────────────────
    c = _case("P.2", vc.PLATE_CASES)
    shear_pct = (vc.PLATE_THICK_W / vc.PLATE_THICK_KIRCHHOFF_W - 1.0) * 100.0
    S.append(Section(
        "P.2", "Laje espessa simplesmente apoiada — MITC3 vs teoria de Mindlin",
        f"A mesma laje, agora **espessa**: t = {n(vc.PLATE_THICK_T, 2)} m "
        f"(vão/espessura = 10), com o triângulo **MITC3** (Mindlin-Reissner, "
        f"deformável ao corte). A carga, o apoio (*hard*) e a malha são os do "
        f"caso anterior. É o caso que o DKT **não** consegue representar, por "
        f"não ter deformação por corte transverso.",
        ["Flecha central da placa de Mindlin (1.ª ordem de corte), série de "
         "Navier com o fator de corte:",
         "w = Σ (16 p / π² m k)/(D λ²) · sin(mπ/2)sin(kπ/2) · "
         "(1 + λ D/(κ G t)),  λ = π²(m²/a² + k²/a²),  κ = 5/6",
         f"w = {n(vc.PLATE_THICK_W * 1000.0, 4)} mm  "
         f"(a teoria fina daria {n(vc.PLATE_THICK_KIRCHHOFF_W * 1000.0, 4)} mm, "
         f"−{n(shear_pct, 1)} % — a parcela de corte)"],
        c.quantities,
        [f"A flecha de Mindlin é {n(shear_pct, 1)} % superior à de Kirchhoff a "
         f"esta espessura: é a deflexão por corte transverso, que o MITC3 capta "
         f"(erro < 0,3 % vs. a série exacta) e o DKT ignora.",
         "O MITC3 evita o *shear locking* por interpolação mista das "
         "deformações de corte covariantes nos meios-lados; passa o *patch "
         "test* de momento constante em malha irregular (verificado à parte na "
         "suite automática) e é o elemento de placa por omissão."]))

    return S


# ═══ Grillage (plate-domain bars) — its own chapter, inside Part III ════════
def grillage_sections() -> list[Section]:
    """The grillage bar (out-of-plane bending + St-Venant torsion), validated by
    closed-form cantilever and crossed-grid solutions."""
    S = []

    # ── G.1 — cantilever bending ────────────────────────────────────────
    c = _case("G.1", vc.GRILLAGE_CASES)
    S.append(Section(
        "G.1", "Consola de grelha — carga transversal na ponta (flexão)",
        f"Barra de grelha encastrada-livre de comprimento L = "
        f"{n(vc.GRID_L, 0)} m (secção {n(vc.GRID_B, 2)} × {n(vc.GRID_H, 2)} m), "
        f"com uma carga transversal Fz = {n(abs(vc.GRID_F), 0)} kN na ponta. "
        f"No domínio de placa a barra flecte fora do plano; os graus de "
        f"liberdade do nó são (w, θx, θy).",
        ["Teoria de viga (flexão fora do plano):",
         f"δ = F·L³/3EI = {n(vc.GRID_TIP_W * 1000.0, 4)} mm",
         f"θ (rotação de flexão) = F·L²/2EI = "
         f"{n(vc.GRID_TIP_ROT * 1000.0, 4)} mrad"],
        c.quantities,
        ["Reproduzido exactamente (erro < 10⁻³ %): a flexão fora do plano da "
         "barra de grelha é a mesma da viga plana, reinterpretada nos graus de "
         "liberdade do domínio de placa."]))

    # ── G.2 — torsion ───────────────────────────────────────────────────
    c = _case("G.2", vc.GRILLAGE_CASES)
    S.append(Section(
        "G.2", "Consola de grelha — momento torsor na ponta (torção GJ)",
        f"A mesma consola, agora com um momento torsor Mx = "
        f"{n(vc.GRID_T, 0)} kN·m na ponta, em torno do eixo da barra. É a "
        f"rigidez que distingue a grelha da viga plana: a torção de St-Venant "
        f"GJ/L.",
        ["Torção de St-Venant:",
         f"θ_torção = T·L/GJ = {n(vc.GRID_TWIST * 1000.0, 4)} mrad,",
         f"com J = {sci(vc._GRID_J)} m⁴ (constante de St-Venant da secção "
         f"rectangular) e G = E/2(1+ν)."],
        c.quantities,
        ["Exacto. A constante de torção usada no valor analítico é a mesma que "
         "o elemento deriva da secção, pelo que o caso valida efectivamente a "
         "rigidez GJ e não uma identidade com um J escolhido à mão."]))

    # ── G.3 — crossed grid ──────────────────────────────────────────────
    c = _case("G.3", vc.GRILLAGE_CASES)
    S.append(Section(
        "G.3", "Grelha de duas vigas cruzadas — repartição da carga central",
        f"Duas vigas iguais, simplesmente apoiadas, de vão L = "
        f"{n(vc.GRID_LG, 0)} m, cruzando-se a meio vão e partilhando o nó "
        f"central, com uma carga Fz = {n(abs(vc.GRID_P), 0)} kN no cruzamento. "
        f"É a grelha de manual: a carga reparte-se pelas duas vigas na razão "
        f"das suas rigidezes — para vigas iguais, 50/50.",
        ["Cada viga leva metade da carga e comporta-se como viga simplesmente "
         "apoiada:",
         f"δ = (P/2)·L³/48EI = {n(vc.GRID_CROSS_W * 1000.0, 4)} mm"],
        c.quantities,
        ["Exacto para vigas iguais. O caso confirma o acoplamento no nó "
         "partilhado: sem ele cada viga levaria a carga toda e a flecha seria o "
         "dobro."]))

    return S


# ═══ Constraints (multi-point / linear DOF coupling) — its own chapter ═══════
def constraint_sections() -> list[Section]:
    """Multi-point constraints: a rigid link and an equal-DOF tie."""
    S = []

    # ── cn-1 — rigid offset ─────────────────────────────────────────────
    c = _case("cn-1", vc.CONSTRAINT_CASES)
    S.append(Section(
        "cn-1", "Ligação rígida — carga excêntrica transmitida por braço rígido",
        f"Pilar em consola de altura H = {n(vc.CN1_H)} m, encastrado na base. O "
        f"nó do topo (mestre) está ligado por uma **ligação rígida** a um nó "
        f"escravo afastado e = {n(vc.CN1_E)} m na horizontal, onde se aplica a "
        f"carga vertical P = {n(vc.CN1_P, 0)} kN. A ligação obriga o escravo a "
        f"acompanhar o topo como um corpo rígido, pelo que a carga excêntrica "
        f"chega ao topo do pilar como força axial mais um momento — exercitando "
        f"os três graus de liberdade (u_x, u_y, θz) da transferência rígida.",
        ["A carga em B (escravo) transfere-se ao topo A (mestre) como uma força "
         "vertical P e um momento M = P·e:",
         f"Momento na base: M = P·e = {n(vc.CN1_MB)} kN·m",
         f"Rotação do topo (flexão por momento de extremidade): "
         f"θ = M·H/(E·I) = {sci(vc.CN1_ROT)} rad",
         f"Deslocamento horizontal do topo: "
         f"u_x = M·H²/(2·E·I) = {n(vc.CN1_UX * 1000)} mm",
         "Cinemática de corpo rígido do escravo: θ_B = θ_A e "
         "u_x,B = u_x,A (o escravo está à mesma altura do mestre)."],
        c.quantities,
        ["A ligação rígida é imposta por penalização, pelo que os valores são "
         "reproduzidos com erro relativo da ordem de 10⁻⁶ — não à precisão da "
         "máquina — e resta uma reação horizontal espúria muito pequena "
         "(~10⁻⁵ kN). Por isso a verificação de equilíbrio destes casos usa "
         "tolerância relativa, e não a tolerância absoluta apertada dos "
         "restantes capítulos. A ligação mestre–escravo exacta fica prevista "
         "para uma versão futura."]))

    # ── cn-2 — equal-DOF load sharing ───────────────────────────────────
    c = _case("cn-2", vc.CONSTRAINT_CASES)
    S.append(Section(
        "cn-2", "Igualdade de graus de liberdade — repartição de carga",
        f"Duas consolas idênticas de vão L = {n(vc.CN2_L)} m, encastradas e "
        f"paralelas, com as pontas ligadas por uma **igualdade de grau de "
        f"liberdade** na translação vertical (u_y,B1 = u_y,B2). Aplicam-se cargas "
        f"desiguais — P₁ = {n(vc.CN2_P1, 0)} kN numa ponta e "
        f"P₂ = {n(vc.CN2_P2, 0)} kN na outra. A ligação obriga as duas pontas a "
        f"descer o mesmo, o que equivale a pô-las em paralelo.",
        [f"Rigidez de ponta de cada consola: k = 3EI/L³ = {n(vc.CN2_K, 0)} kN/m",
         f"Flecha partilhada: u_y = (P₁ + P₂)/(2k) = {n(vc.CN2_D * 1000)} mm",
         f"Reação vertical em cada apoio: R = (P₁ + P₂)/2 = {n(vc.CN2_R)} kN",
         "A repartição é o ponto do caso: cada apoio recebe a média das cargas, "
         "não a que lhe foi aplicada — a ligação move carga da consola mais "
         "carregada para a outra."],
        c.quantities,
        ["Sem a ligação, as duas pontas desceriam em proporção das suas cargas "
         "(uma três vezes mais do que a outra); com a ligação descem o mesmo. "
         "É a validação directa da igualdade de DOF: um vínculo linear entre "
         "graus de liberdade de nós distintos, imposto por penalização."]))

    return S


# ── Rows for the document's summary table ───────────────────────────────────
RESUMO = {
    "P.1": ("Placa fina (DKT)", "Pressão pz",
            "Flecha central vs. série de Navier (Kirchhoff)"),
    "P.2": ("Placa espessa (MITC3)", "Pressão pz",
            "Flecha central vs. série de Mindlin (corte)"),
    "G.1": ("Grelha — consola", "Carga transversal Fz",
            "Flecha e rotação na ponta (FL³/3EI, FL²/2EI)"),
    "G.2": ("Grelha — consola", "Momento torsor Mx",
            "Rotação de torção na ponta (TL/GJ)"),
    "G.3": ("Grelha — vigas cruzadas", "Carga central Fz",
            "Flecha no cruzamento; repartição 50/50"),
    "s-a1": ("Barra + mola nodal", "Carga pontual",
             "δ, N, força na mola, reação"),
    "s-a2": ("Barra + mola nodal", "Carga pontual",
             "Flecha, M, força na mola"),
    "s-a3": ("Barra + molas de rotação", "Carga distribuída",
             "M no apoio e a meio vão, θ, flecha"),
    "s-a4": ("Corpo rígido + molas", "Carga pontual",
             "Assentamentos, forças nas molas, ausência de acoplamento"),
    "s-b1": ("Corpo rígido + Winkler", "Carga distribuída",
             "Assentamento uniforme; momento espúrio do lumping"),
    "s-b2-80": ("Barra + Winkler", "Carga pontual",
                "Flecha e momento sob a carga; convergência"),
    "s-b3-80": ("Barra + Winkler", "Carga pontual no bordo",
                "Flecha no bordo, momento máximo; convergência"),
    "s-b5": ("Corpo rígido + Winkler local", "Carga distribuída",
             "Acoplamento x–y do bloco rodado"),
    "s-c1a": ("Barra + mola unilateral", "Carga pontual (não-linear)",
              "Flecha com mola activa"),
    "s-c1b": ("Barra + mola unilateral", "Carga pontual (não-linear)",
              "Flecha com mola inactiva; força nula"),
    "s-c2a": ("Barra + mola unilateral", "Carga pontual (não-linear)",
              "Flecha com tirante activo"),
    "s-c2b": ("Barra + mola unilateral", "Carga pontual (não-linear)",
              "Flecha com tirante em folga; força nula"),
    "s-c3-48": ("Corpo rígido + Winkler unilateral", "Carga excêntrica",
                "Comprimento de contacto, pressão máxima"),
    "m-a1": ("Barra + massa nodal", "Modal", "f₁ (flexão e axial), T, massa efectiva"),
    "m-a2": ("Barra + mola + massa", "Modal", "f₁ com rigidez equivalente"),
    "m-a3": ("Barra + 2 massas", "Modal", "f₁, f₂, ortogonalidade dos modos"),
    "m-a4": ("Corpo rígido + Winkler", "Modal", "Modos de translação e de rotação"),
    "m-b1-16": ("Barra, massa distribuída", "Modal", "3 primeiras frequências"),
    "m-b2-16": ("Consola, massa distribuída", "Modal",
                "3 primeiras frequências; massas efectivas"),
    "m-b3-32": ("Barra livre-livre", "Modal",
                "Frequências elásticas; separação dos modos de corpo rígido"),
    "m-b4-32": ("Barra, vibração axial", "Modal", "Frequências axiais"),
    "m-b5-32": ("Barra + Winkler", "Modal", "Frequências com fundação"),
    "m-e1": ("Barra + massa nodal", "Espectro de resposta",
             "Resposta espectral vs estática equivalente; SRSS, CQC, direções"),
    "arc": ("Barra (objecto arco)", "Carga pontual",
            "Flecha na ponta, momento e reação no encastramento"),
    "3.1-allman": ("Allman", "Carga distribuída (tração)",
                   "σ_x, σ_y (patch test com drilling restringido)"),
    "3.3-allman": ("Allman", "Assentamento de apoio",
                   "σ_x, reação total no bordo apoiado"),
    "3.4-allman": ("Allman", "Variação de temperatura",
                   "σ = 0 (patch térmico), δ_x, δ_y"),
    "3.2a-32x8": ("Allman", "Carga pontual (flexão)",
                  "Flecha na extremidade; comparação com o CST"),
    "3.2e-32x8": ("ES-FEM", "Carga pontual (flexão)",
                  "Flecha na extremidade; comparação com o CST"),
    "m-c1-es-16x4": ("ES-FEM", "Modal (peso próprio)",
                     "f₁ da membrana em consola; comparação com o CST"),
    "cn-1": ("Barra + ligação rígida", "Carga excêntrica",
             "u_x e θ no topo, momento na base, cinemática do escravo"),
    "cn-2": ("Barra + igualdade de DOF", "Carga pontual",
             "Flecha partilhada nas duas pontas, reação repartida"),
}
