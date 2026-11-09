"""Construction sequences — correções B1–B6 (ver devs/sequences_analise_plano.md).

B1: remoção de elementos → forças de libertação aplicadas à estrutura restante
    e sem esforços "fantasma" do elemento removido.
B2: ids de fase duplicados → erro.
B3: sequência vazia → erro claro (em vez de final=None).
B4: caso repetido em fases sucessivas → só carrega entidades novas (sem dupla
    contagem).
B5: caso inexistente em applied_cases → erro.
B6: initial_state respeitado na 1.ª fase; proibido nas seguintes (é computado).
"""
import unittest

from context import Structure2D
from xdfem2d.models import (ConstructionPhase, ConstructionSequence,
                            ElementInitialState)
from xdfem2d.phasing import solve_sequence


def _beam(n2_support=False):
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    s.add_node("N1", 0.0, 0.0); s.add_node("N2", 5.0, 0.0); s.add_node("N3", 10.0, 0.0)
    s.add_bar_element("E1", "N1", "N2", "S"); s.add_bar_element("E2", "N2", "N3", "S")
    s.add_support("PIN", ux=True, uy=True); s.add_support("ROLLER", uy=True)
    s.assign_support("N1", "PIN"); s.assign_support("N3", "ROLLER")
    if n2_support:
        s.assign_support("N2", "ROLLER")
    s.add_load_case("Q")
    s.add_distributed_load("E1", "Q", fye=-10.0, fyd=-10.0)
    s.add_distributed_load("E2", "Q", fye=-10.0, fyd=-10.0)
    return s


class TestValidation(unittest.TestCase):
    def test_empty_sequence_raises(self):
        with self.assertRaises(ValueError):
            solve_sequence(_beam(), ConstructionSequence(id="S", phases=[]))

    def test_duplicate_phase_ids_raise(self):
        p1 = ConstructionPhase(id="P", applied_cases=["Q"])
        p2 = ConstructionPhase(id="P", applied_cases=[])
        with self.assertRaises(ValueError):
            solve_sequence(_beam(), ConstructionSequence(id="S", phases=[p1, p2]))

    def test_unknown_applied_case_raises(self):
        p = ConstructionPhase(id="P1", applied_cases=["TYPO"])
        with self.assertRaises(ValueError):
            solve_sequence(_beam(), ConstructionSequence(id="S", phases=[p]))

    def test_initial_state_on_later_phase_raises(self):
        p1 = ConstructionPhase(id="P1", applied_cases=["Q"])
        p2 = ConstructionPhase(
            id="P2", applied_cases=[],
            initial_state={"E1": ElementInitialState(i=(1, 0, 0), j=(0, 0, 0))})
        with self.assertRaises(ValueError):
            solve_sequence(_beam(), ConstructionSequence(id="S", phases=[p1, p2]))

    def test_initial_state_on_first_phase_is_used(self):
        s = _beam()
        seed = {"E1": ElementInitialState(i=(7.0, 0.0, 0.0), j=(-7.0, 0.0, 0.0))}
        p = ConstructionPhase(id="P1", applied_cases=[], initial_state=seed)
        out = solve_sequence(s, ConstructionSequence(id="S", phases=[p]))
        self.assertAlmostEqual(
            out["final"]["element_forces"]["E1"]["i"][0], 7.0)


class TestRemoval(unittest.TestCase):
    """Remover uma escora carregada: o estado final tem de igualar a solução
    directa sem escora (identidade da elasticidade linear), e o elemento
    removido não pode aparecer nos esforços finais."""

    def _propped(self):
        s = _beam()
        s.add_node("N9", 5.0, -3.0)
        s.add_support("FIX", ux=True, uy=True, tz=True)
        s.assign_support("N9", "FIX")
        s.add_bar_element("E9", "N9", "N2", "S")
        return s

    def test_prop_removal_matches_unpropped_solution(self):
        s = self._propped()
        p1 = ConstructionPhase(id="P1",
                               active_elements={"E1", "E2", "E9"},
                               applied_cases=["Q"])
        p2 = ConstructionPhase(id="P2",
                               active_elements={"E1", "E2"},
                               applied_cases=[])
        out = solve_sequence(s, ConstructionSequence(id="S", phases=[p1, p2]))
        final = out["phases"]["P2"]["element_forces"]

        ref = _beam().calculate()["element_forces"]["Q"]
        for eid in ("E1", "E2"):
            for end in ("i", "j"):
                for k in range(3):
                    self.assertAlmostEqual(final[eid][end][k],
                                           ref[eid][end][k], places=6,
                                           msg=f"{eid}.{end}[{k}]")

    def test_removed_element_not_in_final_forces(self):
        s = self._propped()
        p1 = ConstructionPhase(id="P1",
                               active_elements={"E1", "E2", "E9"},
                               applied_cases=["Q"])
        p2 = ConstructionPhase(id="P2",
                               active_elements={"E1", "E2"},
                               applied_cases=[])
        out = solve_sequence(s, ConstructionSequence(id="S", phases=[p1, p2]))
        self.assertNotIn("E9", out["phases"]["P2"]["element_forces"])

    def test_removal_with_no_locked_forces_is_noop(self):
        # Elemento removido sem estado instalado (nunca carregado): nada a
        # libertar; a fase seguinte é um incremento normal.
        s = self._propped()
        p1 = ConstructionPhase(id="P1",
                               active_elements={"E1", "E2", "E9"},
                               applied_cases=[])          # sem cargas
        p2 = ConstructionPhase(id="P2",
                               active_elements={"E1", "E2"},
                               applied_cases=["Q"])
        out = solve_sequence(s, ConstructionSequence(id="S", phases=[p1, p2]))
        ref = _beam().calculate()["element_forces"]["Q"]
        self.assertAlmostEqual(
            out["final"]["element_forces"]["E1"]["i"][1],
            ref["E1"]["i"][1], places=6)


class TestRepeatedCase(unittest.TestCase):
    def test_same_case_same_geometry_gives_zero_increment(self):
        # Reaplicar o mesmo caso sem elementos novos não pode duplicar cargas.
        s = _beam()
        p1 = ConstructionPhase(id="P1", applied_cases=["Q"])
        p2 = ConstructionPhase(id="P2", applied_cases=["Q"])
        out = solve_sequence(s, ConstructionSequence(id="S", phases=[p1, p2]))
        f1 = out["phases"]["P1"]["element_forces"]
        f2 = out["phases"]["P2"]["element_forces"]
        for eid in ("E1", "E2"):
            for k in range(3):
                self.assertAlmostEqual(f2[eid]["i"][k], f1[eid]["i"][k],
                                       places=9)

    def test_growth_with_repeated_case_matches_manual_staging(self):
        # Fase 1: só E1 (apoio em N2) com Q; fase 2: junta E2 com o MESMO caso.
        # O correto: E1 carrega Q uma vez; na fase 2 só E2 recebe as cargas de Q.
        s = _beam(n2_support=True)
        p1 = ConstructionPhase(id="P1", active_elements={"E1"},
                               applied_cases=["Q"])
        p2 = ConstructionPhase(id="P2", active_elements={"E1", "E2"},
                               applied_cases=["Q"])
        out = solve_sequence(s, ConstructionSequence(id="S", phases=[p1, p2]))

        # Composição manual do resultado esperado.
        sub1 = s.with_active({"E1"})
        r1 = sub1.calculate()["element_forces"]["Q"]
        sub2 = s.with_active({"E1", "E2"})
        sub2.distributed_loads = [dl for dl in sub2.distributed_loads
                                  if dl.element_id != "E1"]   # Q só em E2
        r2 = sub2.calculate()["element_forces"]["Q"]
        exp_e1 = [r1["E1"]["i"][k] + r2["E1"]["i"][k] for k in range(3)]

        got = out["final"]["element_forces"]["E1"]["i"]
        for k in range(3):
            self.assertAlmostEqual(got[k], exp_e1[k], places=6)

    def test_distinct_cases_are_not_masked(self):
        # Casos diferentes em fases sucessivas continuam a somar normalmente.
        s = _beam()
        s.add_load_case("Q2"); s.add_distributed_load("E1", "Q2", fye=-10.0, fyd=-10.0)
        p1 = ConstructionPhase(id="P1", applied_cases=["Q"])
        p2 = ConstructionPhase(id="P2", applied_cases=["Q2"])
        out = solve_sequence(s, ConstructionSequence(id="S", phases=[p1, p2]))
        f1 = out["phases"]["P1"]["element_forces"]["E1"]["i"][1]
        f2 = out["phases"]["P2"]["element_forces"]["E1"]["i"][1]
        self.assertGreater(abs(f2), abs(f1) * 1.2)


class TestCaseStripping(unittest.TestCase):
    """Ação 4 — cada fase resolve só os seus applied_cases; casos, analysis
    cases e combinações alheios não podem alterar o resultado (nem custar)."""

    def test_extra_cases_do_not_change_results(self):
        s1 = _beam()
        p = [ConstructionPhase(id="P1", active_elements={"E1", "E2"},
                               applied_cases=["Q"])]
        ref = solve_sequence(s1, ConstructionSequence(id="S", phases=list(p)))

        s2 = _beam()
        for c in range(6):                      # ballast the model
            s2.add_load_case(f"X{c}")
            s2.add_distributed_load("E1", f"X{c}", fye=-3.0, fyd=-3.0)
        s2.add_load_combination("CMB", {"Q": 1.5, "X0": 1.0})
        out = solve_sequence(s2, ConstructionSequence(id="S", phases=list(p)))

        for eid in ("E1", "E2"):
            for end in ("i", "j"):
                for k in range(3):
                    self.assertAlmostEqual(
                        out["final"]["element_forces"][eid][end][k],
                        ref["final"]["element_forces"][eid][end][k], places=9)

    def test_phase_with_no_cases_is_zero_increment(self):
        s = _beam(n2_support=True)
        p1 = ConstructionPhase(id="P1", active_elements={"E1"},
                               applied_cases=["Q"])
        p2 = ConstructionPhase(id="P2", active_elements={"E1", "E2"},
                               applied_cases=[])       # growth only, no loads
        out = solve_sequence(s, ConstructionSequence(id="S", phases=[p1, p2]))
        f1 = out["phases"]["P1"]["element_forces"]["E1"]["i"]
        f2 = out["phases"]["P2"]["element_forces"]["E1"]["i"]
        for k in range(3):
            self.assertAlmostEqual(f2[k], f1[k], places=9)
        # the new element exists in the final state, unloaded
        self.assertIn("E2", out["final"]["element_forces"])
        for k in range(3):
            self.assertAlmostEqual(
                out["final"]["element_forces"]["E2"]["i"][k], 0.0, places=9)


class TestPhasesFromScenes(unittest.TestCase):
    """Ação 6 — gerar fases a partir de scenes (subconjuntos nomeados)."""

    SCENES = [
        {"name": "Tramo 1", "elements": {"E1"}},
        {"name": "Tramo 2", "elements": {"E2", "GHOST"}},   # id inexistente
    ]

    def test_cumulative_generation(self):
        from xdfem2d.workflows import phases_from_scenes
        phases = phases_from_scenes(self.SCENES, {"E1", "E2"})
        self.assertEqual([p.id for p in phases], ["Tramo 1", "Tramo 2"])
        self.assertEqual(phases[0].active_elements, {"E1"})
        self.assertEqual(phases[1].active_elements, {"E1", "E2"})   # união
        self.assertEqual(phases[0].applied_cases, [])

    def test_exact_generation(self):
        from xdfem2d.workflows import phases_from_scenes
        phases = phases_from_scenes(self.SCENES, {"E1", "E2"},
                                    cumulative=False)
        self.assertEqual(phases[1].active_elements, {"E2"})

    def test_generated_sequence_solves(self):
        from xdfem2d.workflows import phases_from_scenes
        import dataclasses
        s = _beam(n2_support=True)
        phases = phases_from_scenes(self.SCENES,
                                    {e.id for e in s.bar_elements})
        phases = [dataclasses.replace(p, applied_cases=["Q"])
                  for p in phases]
        out = solve_sequence(s, ConstructionSequence(id="S", phases=phases))
        self.assertIn("Tramo 2", out["phases"])
        self.assertIn("E2", out["final"]["element_forces"])


class TestPhasesFromStages(unittest.TestCase):
    """Ação 7 — atributo `stage` por elemento: fase k = {e : stage(e) ≤ k}."""

    def test_assign_and_generate(self):
        from xdfem2d.workflows import (assign_stage, stage_numbers,
                                       phases_from_stages)
        s = _beam(n2_support=True)
        self.assertEqual(stage_numbers(s), [1])          # default
        n = assign_stage(s, ["E2", "GHOST"], 3)          # id inexistente ignora
        self.assertEqual(n, 1)
        self.assertEqual(stage_numbers(s), [1, 3])       # gaps permitidos
        phases = phases_from_stages(s, cases_by_stage={1: ["Q"]})
        self.assertEqual([p.id for p in phases], ["Stage 1", "Stage 3"])
        self.assertEqual(phases[0].active_elements, {"E1"})
        self.assertEqual(phases[1].active_elements, {"E1", "E2"})   # monótono
        self.assertEqual(phases[0].applied_cases, ["Q"])
        self.assertEqual(phases[1].applied_cases, [])

    def test_invalid_stage_raises(self):
        from xdfem2d.workflows import assign_stage
        with self.assertRaises(ValueError):
            assign_stage(_beam(), ["E1"], 0)

    def test_stage_roundtrip(self):
        import os, tempfile
        from xdfem2d.workflows import assign_stage
        s = _beam()
        assign_stage(s, ["E2"], 4)
        fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
        try:
            s.save(path); r = Structure2D.load(path)
        finally:
            os.remove(path)
        self.assertEqual(r.bar_elements_by_id["E2"].stage, 4)
        self.assertEqual(r.bar_elements_by_id["E1"].stage, 1)

    def test_generated_sequence_solves(self):
        import dataclasses
        from xdfem2d.workflows import assign_stage, phases_from_stages
        s = _beam(n2_support=True)
        assign_stage(s, ["E2"], 2)
        phases = [dataclasses.replace(p, applied_cases=["Q"])
                  for p in phases_from_stages(s)]
        out = solve_sequence(s, ConstructionSequence(id="S", phases=phases))
        self.assertEqual(list(out["phases"]), ["Stage 1", "Stage 2"])
        self.assertIn("E2", out["final"]["element_forces"])
        # E1 não é recarregado na fase 2 (caso repetido → só entidades novas),
        # mas recebe a redistribuição por continuidade da carga nova em E2:
        # v2 = v1 + incremento da viga contínua carregada só em E2.
        v1 = out["phases"]["Stage 1"]["element_forces"]["E1"]["i"][1]
        v2 = out["phases"]["Stage 2"]["element_forces"]["E1"]["i"][1]
        self.assertNotEqual(v1, 0.0)
        sub2 = s.with_active({"E1", "E2"})
        sub2.distributed_loads = [dl for dl in sub2.distributed_loads
                                  if dl.element_id != "E1"]   # Q só em E2
        inc = sub2.calculate()["element_forces"]["Q"]["E1"]["i"][1]
        self.assertAlmostEqual(v2, v1 + inc, places=9)


if __name__ == "__main__":
    unittest.main()
