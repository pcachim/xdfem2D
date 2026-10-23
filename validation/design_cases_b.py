"""Grupo B — casos de validação end-to-end (mesh → calculate() → design_*).

Ao contrário do grupo A (que chama `RCSection` / as funções de baixo nível
do eurocodepy diretamente, isolando a verificação da secção), este grupo
testa o caminho completo: ``Structure2D`` → ``calculate()`` (montagem,
resolução, pós-processamento) → ``design_concrete_sections`` /
``design_steel_members`` / ``design_timber_members``. É aqui que apanhamos
problemas de integração que os testes isolados do grupo A não veem: sinal
das forças na fronteira FE→design, malha vs. localização do momento máximo,
combinações de ações, k_mod/duração da ação, etc.

Nota de modelação importante descoberta a fazer este ficheiro:
``design_steel_members``/``design_timber_members`` só avaliam os esforços
nos **nós** de cada barra (início/fim), ao contrário de
``design_concrete_sections`` que já interpola ao longo do vão. Por isso B1
(viga RC) funciona com uma única barra, mas B4 (viga de madeira) precisa da
barra subdividida em vários elementos para que um nó caia a meio vão —
sem isso o momento máximo nunca é avaliado (ficaria M_Ed=0 na secção a meio
vão, um falso "PASS"). Fica documentado no caso para não se repetir o erro.

Grupo completo: B1 viga RC, B2 consola de aço, B3 pilar de aço em
compressão (encurvadura, caso de falha deliberado), B4 viga de madeira,
B5 laje simplesmente apoiada, B6 punçoamento.

Requer: xdfem2d + eurocodepy (código-fonte real, ver design_cases.py).
Correr:  python validation/design_cases_b.py
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
# B1 — Viga RC bi-apoiada, L=6 m, secção 0.30×0.50 (C25/30/A500),
# carga distribuída q=20 kN/m. Uma única barra (design_concrete_sections
# interpola ao longo do vão, não precisa de subdivisão).
# ═══════════════════════════════════════════════════════════════════════

def case_b1_viga_rc():
    from xdfem2d import Structure2D
    from xdfem2d.rc_design import design_concrete_sections

    s = Structure2D(domain="plane")
    s.add_material("C25_30", 31000e3, 25.0, material_type="Concrete",
                   design={"fck": 25.0, "fyk": 500.0})
    s.add_section("B30x50", "C25_30", b=0.30, h=0.50, section_type="Concrete")
    length = 6.0
    s.add_node("N1", 0.0, 0.0)
    s.add_node("N2", length, 0.0)
    s.add_bar_element("E1", "N1", "N2", "B30x50")
    s.add_support("pin", ux=True, uy=True, tz=False)
    s.add_support("roller", ux=False, uy=True, tz=False)
    s.assign_support("N1", "pin")
    s.assign_support("N2", "roller")
    s.add_load_case("LC1")
    q = 20.0
    s.add_distributed_load("E1", "LC1", fye=-q, fyd=-q)
    s.add_load_combination("ULS1", {"LC1": 1.0})
    res = s.calculate()

    ef = res["combinations"]["ULS1"]["element_forces"]["E1"]
    v_support = abs(ef["i"][1])
    m_theory, v_theory = q * length**2 / 8, q * length / 2
    _check("B1 viga RC: V no apoio (teoria da viga qL/2)", v_support, v_theory, 1e-6, " kN")

    rows = design_concrete_sections(s, res)
    governing = [r for r in rows if r.get("governing")][0]
    _check("B1 viga RC: M_Ed a meio vão (teoria qL²/8)", governing["M_Ed"], m_theory, 1e-6, " kNm")
    _check("B1 viga RC: As_bot", governing["As_bot"], 4.7666e-4, 5e-6, " m2")
    _check_true("B1 viga RC: localização correta (meio vão)",
               governing["location"] == "x=3.00m")


# ═══════════════════════════════════════════════════════════════════════
# B2 — Consola de aço, L=3 m, IPE300/S275, carga vertical P=20 kN na ponta.
# ═══════════════════════════════════════════════════════════════════════

def case_b2_consola_aco():
    from xdfem2d import Structure2D
    from xdfem2d.steel_design import design_steel_members

    s = Structure2D(domain="plane")
    s.add_material("S275", 210e6, 78.5, material_type="Steel", design={"fy": 275.0})
    s.add_section("IPE300", "S275", b=0.150, h=0.300, tw=0.0071, tf=0.0107, shape="I")
    length = 3.0
    s.add_node("N1", 0.0, 0.0)
    s.add_node("N2", length, 0.0)
    s.add_bar_element("E1", "N1", "N2", "IPE300")
    s.add_support("fix", ux=True, uy=True, tz=True)
    s.assign_support("N1", "fix")
    s.add_load_case("LC1")
    load = 20.0
    s.add_point_load("N2", "LC1", fy=-load)
    s.add_load_combination("ULS1", {"LC1": 1.0})
    res = s.calculate()

    ef = res["combinations"]["ULS1"]["element_forces"]["E1"]
    m_support, v_support = abs(ef["i"][2]), abs(ef["i"][1])
    _check("B2 consola aço: M no encastramento (teoria P·L)", m_support, load * length, 1e-6, " kNm")
    _check("B2 consola aço: V (teoria P)", v_support, load, 1e-6, " kN")

    out = design_steel_members(s, res)
    ratios = out["elements"]["E1"]
    _check("B2 consola aço: util bending (M/Mpl,y,Rd)", ratios["bending"], 0.362369, 0.001)
    _check_true("B2 consola aço: encurvadura calculada automaticamente "
               "(K de consola livre-encastrada)", "buckling" in ratios)


# ═══════════════════════════════════════════════════════════════════════
# B3 — Pilar de aço em compressão centrada, consola vertical L=4 m,
# IPE300/S275, N=300 kN no topo. Encastrado na base, topo livre — a
# encurvadura por flexão em torno do eixo fraco (z) domina e o pilar FALHA
# (utilização > 1.0). Caso deliberadamente escolhido para confirmar que o
# módulo sinaliza corretamente um pilar subdimensionado, não só os que
# passam.
# ═══════════════════════════════════════════════════════════════════════

def case_b3_pilar_aco_compressao():
    from xdfem2d import Structure2D
    from xdfem2d.steel_design import design_steel_members

    s = Structure2D(domain="plane")
    s.add_material("S275", 210e6, 78.5, material_type="Steel", design={"fy": 275.0})
    s.add_section("IPE300", "S275", b=0.150, h=0.300, tw=0.0071, tf=0.0107, shape="I")
    length = 4.0
    s.add_node("N1", 0.0, 0.0)
    s.add_node("N2", 0.0, length)
    s.add_bar_element("E1", "N1", "N2", "IPE300", is_column=True)
    s.add_support("fix", ux=True, uy=True, tz=True)
    s.assign_support("N1", "fix")
    s.add_load_case("LC1")
    axial = 300.0
    s.add_point_load("N2", "LC1", fy=-axial)
    s.add_load_combination("ULS1", {"LC1": 1.0})
    res = s.calculate()

    ef = res["combinations"]["ULS1"]["element_forces"]["E1"]
    n_ed = abs(ef["i"][0])
    _check("B3 pilar aço: N_Ed transmitido (teoria N aplicado)", n_ed, axial, 1e-6, " kN")

    out = design_steel_members(s, res)
    member = out["members"][0]
    _check_true("B3 pilar aço: chi_z << 1.0 (encurvadura eixo fraco domina, "
               "pilar em consola)", member["chi_z"] < 0.20,
               f"chi_z={member['chi_z']:.4f}")
    _check_true("B3 pilar aço: utilização > 1.0 (pilar subdimensionado — "
               "o módulo deteta corretamente a falha)",
               member["utilization"] > 1.0,
               f"utilizacao={member['utilization']:.3f}")
    _check_true("B3 pilar aço: passed == False", bool(member["passed"]) is False)


# ═══════════════════════════════════════════════════════════════════════
# B4 — Viga de madeira bi-apoiada, L=4 m, secção 0.10×0.20 (C24),
# carga distribuída q=3 kN/m (ação permanente por omissão → kmod=0.6).
#
# ATENÇÃO DE MODELAÇÃO: subdividida em 4 barras (nós a 0/1/2/3/4 m) para que
# um nó caia a meio vão — ver nota no topo do ficheiro.
# ═══════════════════════════════════════════════════════════════════════

def case_b4_viga_madeira():
    from xdfem2d import Structure2D
    from xdfem2d.timber_design import design_timber_members

    s = Structure2D(domain="plane")
    s.add_material("C24", 11e6, 4.2, material_type="Timber", design={"class": "C24"})
    s.add_section("T1", "C24", b=0.10, h=0.20, shape="Rectangular")
    length, n = 4.0, 4
    for i in range(n + 1):
        s.add_node(f"N{i}", length * i / n, 0.0)
    for i in range(n):
        s.add_bar_element(f"E{i}", f"N{i}", f"N{i + 1}", "T1")
    s.add_support("pin", ux=True, uy=True, tz=False)
    s.add_support("roller", ux=False, uy=True, tz=False)
    s.assign_support("N0", "pin")
    s.assign_support(f"N{n}", "roller")
    s.add_load_case("LC1")
    q = 3.0
    for i in range(n):
        s.add_distributed_load(f"E{i}", "LC1", fye=-q, fyd=-q)
    s.add_load_combination("ULS1", {"LC1": 1.0})
    res = s.calculate()

    out = design_timber_members(s, res)
    m_theory, v_theory = q * length**2 / 8, q * length / 2
    member = out["members"][0]
    _check("B4 viga madeira: My_Ed a meio vão (teoria qL²/8)",
          member["My_Ed"], m_theory, 1e-6, " kNm")
    _check_true("B4 viga madeira: elemento a meio vão (E1/E2) governa a flexão",
               out["elements"]["E1"]["bending"] >= out["elements"]["E0"]["bending"])
    _check("B4 viga madeira: util bending a meio vão (kmod=0.6, ação permanente)",
          max(out["elements"]["E1"]["bending"], out["elements"]["E2"]["bending"]),
          0.8125, 0.001)
    _check("B4 viga madeira: util shear no apoio (V=qL/2, kmod=0.6)",
          out["elements"]["E0"]["shear"], 0.24375, 0.001)


# ═══════════════════════════════════════════════════════════════════════
# B5 — Laje simplesmente apoiada, quadrada a=4 m, espessura t=0.20 m
# (C25/30), carga uniforme q=5 kN/m2, malha DKT via geo_rectangle
# (target_size=0.4). Ao contrário de design_steel_members/design_timber_
# members, design_concrete_slabs precisa da malha COMPILADA
# (expand_geometry(s)) — o objeto de geometria (s.tri_elements) está vazio
# antes disso; ficou documentado porque calou uma primeira tentativa (rows
# vinha vazio, sem erro).
#
# Não há solução fechada simples para comparar (é uma placa, não uma viga),
# por isso o critério de correção aqui é duplo: (1) simetria exata — o
# problema é quadrado e a solução tem de ser simétrica em x/y, mesh-
# -independente; (2) ordem de grandeza — o coeficiente de momento no centro
# (mx/(q·a²)) tem de estar próximo do valor clássico de placa fina
# simplesmente apoiada em todo o contorno (Timoshenko, ≈0.048 para ν=0.3;
# a malha DKT grosseira usada aqui, com ν=0.2, deve ficar dentro de ~15%
# desse valor de referência).
# ═══════════════════════════════════════════════════════════════════════

def case_b5_laje_simplesmente_apoiada():
    from xdfem2d import Structure2D
    from xdfem2d.geo_expand import expand_geometry
    from xdfem2d.rc_design import design_concrete_slabs

    s = Structure2D(domain="plate")
    s.add_material("C25_30", 31000e3, 25.0, material_type="Concrete",
                   design={"fck": 25.0, "fyk": 500.0}, poisson=0.2)
    s.add_tri_section("S20", "C25_30", thickness=0.20, formulation="DKT",
                      rc_cover=0.025)
    a = 4.0
    s.add_geo_rectangle("R1", (0.0, 0.0), (a, a), tri_section_name="S20",
                        target_size=0.4, prefer_quad=False)
    s.add_support("SIMPLE", ux=True, uy=False, tz=False)
    s.support_object_edge("R1", "all", "SIMPLE")
    s.add_load_case("LC1")
    q = 5.0
    s.add_area_load("R1", "LC1", pz=-q)
    s.add_load_combination("ULS1", {"LC1": 1.0})
    res = s.calculate()

    compiled, _trace = expand_geometry(s)     # ← malha compilada, não `s`
    rows = design_concrete_slabs(compiled, res)
    _check_true("B5 laje: design produziu resultados (malha compilada "
               "passada corretamente)", len(rows) > 0)

    mx_max = max(r["mx_bot"] for r in rows)
    my_max = max(r["my_bot"] for r in rows)
    asx_max = max(r["Asx_bot"] for r in rows)
    asy_max = max(r["Asy_bot"] for r in rows)
    _check("B5 laje: simetria mx_bot vs my_bot (secção quadrada)",
          mx_max, my_max, 1e-9, " kNm/m")
    _check("B5 laje: simetria Asx_bot vs Asy_bot", asx_max, asy_max, 1e-12, " m2/m")

    coef = mx_max / (q * a**2)
    ref_coef = 0.0479          # Timoshenko, placa fina s.a., ν=0.3
    _check_true("B5 laje: coeficiente mx/(q·a²) próximo do valor clássico "
               f"de placa fina (got={coef:.4f}, Timoshenko≈{ref_coef})",
               abs(coef - ref_coef) / ref_coef < 0.15,
               f"desvio={abs(coef - ref_coef) / ref_coef:.1%}")


# ═══════════════════════════════════════════════════════════════════════
# B6 — Punçoamento, laje 8×8 m (t=0.20 m, C25/30) apoiada nos 4 bordos +
# um pilar interior no centro (0.30×0.30 m), carga q=10 kN/m2 via
# self_weight_factor (peso próprio equivalente: unit_weight=q/t, truque
# para obter uma carga de área uniforme sem depender de geo_rectangle —
# aqui a malha é construída à mão, nó a nó, precisamente para ter acesso
# direto ao nó central sem o problema de IDs gerados pelo expand_geometry
# do B5).
#
# Verificação de correção mais forte que qualquer valor de referência
# tabelado: equilíbrio estático exato — a soma de TODAS as reações tem de
# ser exatamente q·área, independente da malha. Depois confirma-se
# v_Ed = N_Ed/(u1·d) (fórmula EC2 §6.4.3 sem β, β=1.0 aqui por simetria).
# ═══════════════════════════════════════════════════════════════════════

def case_b6_puncoamento():
    from xdfem2d import Structure2D
    from xdfem2d.punching import design_punching

    s = Structure2D(domain="plate")
    q, t = 10.0, 0.20
    unit_weight = q / t                        # self-weight == q
    s.add_material("C25_30", 31000e3, unit_weight, material_type="Concrete",
                   design={"fck": 25.0, "fyk": 500.0}, poisson=0.2)
    s.add_tri_section("S20", "C25_30", thickness=t, formulation="DKT",
                      rc_cover=0.025)

    a, n = 8.0, 8

    def nid(i, j):
        return f"N{i}_{j}"

    for i in range(n + 1):
        for j in range(n + 1):
            s.add_node(nid(i, j), a * i / n, a * j / n)
    tid = 0
    for i in range(n):
        for j in range(n):
            p00, p10, p01, p11 = nid(i, j), nid(i + 1, j), nid(i, j + 1), nid(i + 1, j + 1)
            s.add_tri_element(f"T{tid}", p00, p10, p11, "S20"); tid += 1
            s.add_tri_element(f"T{tid}", p00, p11, p01, "S20"); tid += 1

    s.add_support("SIMPLE", ux=True, uy=False, tz=False)
    for i in range(n + 1):
        s.assign_support(nid(i, 0), "SIMPLE")
        s.assign_support(nid(i, n), "SIMPLE")
    for j in range(n + 1):
        s.assign_support(nid(0, j), "SIMPLE")
        s.assign_support(nid(n, j), "SIMPLE")
    center = nid(n // 2, n // 2)
    s.assign_support(center, "SIMPLE")

    s.add_load_case("LC1", self_weight_factor=1.0)
    s.add_load_combination("ULS1", {"LC1": 1.0})
    res = s.calculate()

    reac = res["combinations"]["ULS1"]["reactions"]
    total_reac = sum(r[0] for r in reac.values())
    _check("B6 punçoamento: equilíbrio estático (ΣR = q·área, exato, "
          "independente da malha)", total_reac, q * a * a, 1e-6, " kN")

    s.add_punch_column("C1", center, shape="rectangular", bx=0.30, by=0.30,
                       position="center")
    rows = design_punching(s, res)
    row = rows[0]
    n_ed_theory = row["N_Ed"]        # confirmado pelo equilíbrio acima
    v_ed_formula = n_ed_theory / (row["u1"] * row["d"]) / 1000.0
    _check("B6 punçoamento: v_Ed = N_Ed/(u1·d) (β=1.0, sem excentricidade)",
          row["v_Ed"], v_ed_formula, 1e-6, " MPa")
    _check_true("B6 punçoamento: utilização < 1.0 (secção passa, "
               f"util={row['utilization']:.3f})", row["utilization"] < 1.0)
    _check_true("B6 punçoamento: needs_reinf coerente com utilização < 1.0",
               row["needs_reinf"] is False)


def run() -> bool:
    cases = [v for k, v in sorted(globals().items()) if k.startswith("case_")]
    print(f"A executar {len(cases)} casos end-to-end...\n")
    for fn in cases:
        print(f"--- {fn.__name__}: {(fn.__doc__ or '').strip().splitlines()[0] if fn.__doc__ else ''} ---")
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
