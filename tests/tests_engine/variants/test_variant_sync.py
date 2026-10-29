"""Base ↔ variant synchronisation (Fase 2).

Rule under test: a variant's own model is authoritative for its ACTIONS
(loads / cases / combinations) and for its SUPPORTS and springs; its GEOMETRY
and PROPERTIES follow the base filtered by the variant's declared config.
Editing the base must therefore propagate — via ``sync_variant_with_base`` —
into every variant model, without touching the variant's own actions, and the
supports the base changes reach the nodes the variant has not changed.
"""
import unittest

from context import Structure2D
from xdfem2d.models import Variant, SupportSet
from xdfem2d.workflows import (make_variant_model, apply_variant_config,
                               sync_variant_with_base, sync_all_variants,
                               variant_base_diff, new_variant,
                               reset_variant_supports, copy_variant_supports,
                               current_restraints)


def _beam():
    s = Structure2D()
    s.add_material("M", elastic_modulus=30e6, unit_weight=0.0)
    s.add_section("S", "M", b=0.3, h=0.6)
    s.add_node("N1", 0.0, 0.0); s.add_node("N2", 5.0, 0.0); s.add_node("N3", 10.0, 0.0)
    s.add_bar_element("E1", "N1", "N2", "S"); s.add_bar_element("E2", "N2", "N3", "S")
    s.add_support("PIN", ux=True, uy=True); s.add_support("ROLLER", uy=True)
    s.assign_support("N1", "PIN"); s.assign_support("N3", "ROLLER")
    s.add_load_case("LC"); s.add_distributed_load("E1", "LC", fye=-10.0, fyd=-10.0)
    return s


def _restraint_of(model, node_id):
    """(ux, uy, tz) at *node_id* — support sets now reuse a canonical/shared
    definition (PIN/FIXED/...) rather than one named after the node itself,
    so tests can no longer index ``model.supports[node_id]`` directly."""
    a = next(a for a in model.support_assignments if a.node_id == node_id)
    s = model.supports[a.support_name]
    return (s.ux, s.uy, s.tz)


def _register(base, vid="V1", active=None):
    v = new_variant(base, vid, active_elements=active)
    base.add_variant(v)
    return v


class TestGeometrySync(unittest.TestCase):
    def test_new_base_bar_appears_in_full_variant(self):
        base = _beam()
        v = _register(base)                          # active=None → follow all
        base.add_node("N4", 15.0, 0.0)
        base.add_bar_element("E3", "N3", "N4", "S")
        self.assertTrue(sync_variant_with_base(base, v))
        self.assertIn("E3", v.model.bar_elements_by_id)
        self.assertIn("N4", v.model.nodes)

    def test_new_base_bar_stays_out_of_restricted_variant(self):
        base = _beam()
        v = _register(base, active={"E1"})
        base.add_node("N4", 15.0, 0.0)
        base.add_bar_element("E3", "N3", "N4", "S")
        sync_variant_with_base(base, v)
        self.assertEqual(sorted(v.model.bar_elements_by_id), ["E1"])

    def test_deleted_base_bar_removed_only_by_destructive_sync(self):
        # The guarded AUTO-sync is non-destructive (it cannot tell a bar the
        # user added to the variant from one deleted from the base); removals
        # happen only on the explicit destructive reconcile.
        base = _beam()
        v = _register(base)
        base.remove_element("E2")
        self.assertTrue(sync_variant_with_base(base, v))
        self.assertIn("E2", v.model.bar_elements_by_id)       # kept (auto)
        self.assertTrue(sync_variant_with_base(base, v, force=True,
                                               destructive=True))
        self.assertNotIn("E2", v.model.bar_elements_by_id)    # removed
        self.assertFalse([dl for dl in v.model.distributed_loads
                          if dl.element_id == "E2"])

    def test_deleted_base_bar_in_declared_active_set(self):
        # A stale id in active_elements must not resurrect/keep a bar that no
        # longer exists in the base entity space (explicit reconcile).
        base = _beam()
        v = _register(base, active={"E1", "E2"})
        base.remove_element("E2")
        sync_variant_with_base(base, v, force=True, destructive=True)
        self.assertEqual(sorted(v.model.bar_elements_by_id), ["E1"])

    def test_moved_node_follows_base(self):
        base = _beam()
        v = _register(base)
        base.nodes["N2"].x = 6.0
        self.assertTrue(sync_variant_with_base(base, v))
        self.assertEqual(v.model.nodes["N2"].x, 6.0)

    def test_changed_section_follows_base(self):
        base = _beam()
        v = _register(base)
        base.sections["S"].h = 0.9
        self.assertTrue(sync_variant_with_base(base, v))
        self.assertEqual(v.model.sections["S"].h, 0.9)

    def test_changed_hinge_follows_base(self):
        base = _beam()
        v = _register(base)
        base.bar_elements_by_id["E1"].hinge_j = True
        self.assertTrue(sync_variant_with_base(base, v))
        self.assertTrue(v.model.bar_elements_by_id["E1"].hinge_j)


class TestSupportSync(unittest.TestCase):
    """The supports and springs of a variant are its own. What the base
    changes reaches the nodes the variant has not changed; a node the user
    changed on the variant keeps its value."""

    def test_a_base_support_reaches_a_node_the_variant_left_alone(self):
        base = _beam()
        v = _register(base)
        base.add_support("R2", uy=True); base.assign_support("N2", "R2")
        self.assertTrue(sync_variant_with_base(base, v))
        self.assertEqual(_restraint_of(v.model, "N2"), (False, True, False))

    def test_a_base_support_taken_away_goes_from_the_variant_too(self):
        base = _beam()
        v = _register(base)
        base.support_assignments = [a for a in base.support_assignments
                                    if a.node_id != "N3"]
        sync_variant_with_base(base, v)
        self.assertEqual(sorted(current_restraints(v.model)), ["N1"])

    def test_a_support_the_user_changed_on_the_variant_is_kept(self):
        base = _beam()
        v = _register(base)
        # the variant frees N3 …
        v.model.support_assignments = [a for a in v.model.support_assignments
                                       if a.node_id != "N3"]
        # … and the base changes N3 and N2
        base.support_assignments = [a for a in base.support_assignments
                                    if a.node_id != "N3"]
        base.add_support("FX", ux=True, uy=True, tz=True); base.assign_support("N3", "FX")
        base.add_support("R2", uy=True); base.assign_support("N2", "R2")
        d = variant_base_diff(base, v)
        self.assertIn(("supports", "N3"), d['conflicts'])    # both changed N3 … differently
        sync_variant_with_base(base, v)
        got = current_restraints(v.model)
        self.assertNotIn("N3", got)                          # the variant's choice stays
        self.assertEqual(got["N2"], (False, True, False))    # the untouched one follows

    def test_the_variants_own_support_survives_base_edits_elsewhere(self):
        base = _beam()
        v = _register(base)
        v.model.add_support("FIXO", ux=True, uy=True, tz=True)
        v.model.support_assignments = [a for a in v.model.support_assignments
                                       if a.node_id != "N1"]
        v.model.assign_support("N1", "FIXO")
        base.nodes["N2"].y = 0.5
        base.add_distributed_load("E2", "LC", fye=-1.0, fyd=-1.0)
        sync_variant_with_base(base, v)
        self.assertEqual(_restraint_of(v.model, "N1"), (True, True, True))

    def test_a_base_spring_follows_the_variant_that_did_not_set_one(self):
        base = _beam()
        v = _register(base)
        base.add_node_spring("N2", ky=1000.0)
        self.assertTrue(sync_variant_with_base(base, v))
        self.assertEqual(v.model.node_springs["N2"].ky, 1000.0)

    def test_a_spring_the_variant_set_is_kept(self):
        base = _beam()
        v = _register(base)
        v.model.add_node_spring("N2", ky=50.0)
        base.add_node_spring("N2", ky=1000.0)
        sync_variant_with_base(base, v)
        self.assertEqual(v.model.node_springs["N2"].ky, 50.0)

    def test_a_node_the_variant_does_not_have_is_skipped(self):
        base = _beam()
        v = _register(base, active={"E1", "E2"})
        base.add_node("N4", 15.0, 0.0)
        base.add_bar_element("E3", "N3", "N4", "S")         # not active in the variant
        base.add_support("R4", uy=True); base.assign_support("N4", "R4")
        sync_variant_with_base(base, v)
        self.assertNotIn("N4", v.model.nodes)
        self.assertNotIn("N4", current_restraints(v.model))
        v.model.calculate()                                  # and the model is sound

    def test_syncing_never_leaves_a_support_on_a_missing_node(self):
        base = _beam()
        v = _register(base)
        base.add_support("R2", uy=True); base.assign_support("N2", "R2")
        sync_variant_with_base(base, v)
        self.assertTrue(set(current_restraints(v.model)) <= set(v.model.nodes))

    def test_reset_makes_the_variants_supports_the_bases_and_they_follow_again(self):
        base = _beam()
        v = _register(base)
        v.model.support_assignments = []                     # a variant with no supports of its own
        self.assertTrue(variant_base_diff(base, v)['overrides'])
        reset_variant_supports(base, v)
        self.assertEqual(current_restraints(v.model), current_restraints(base))
        self.assertEqual(variant_base_diff(base, v)['overrides'], [])
        base.add_support("R2", uy=True); base.assign_support("N2", "R2")
        sync_variant_with_base(base, v)
        self.assertIn("N2", current_restraints(v.model))     # it follows the base again

    def test_copying_from_another_variant_makes_them_the_variants_own(self):
        base = _beam()
        v1 = _register(base, "V1")
        v2 = _register(base, "V2")
        v1.model.add_support("FIXO", ux=True, uy=True, tz=True)
        v1.model.support_assignments = [a for a in v1.model.support_assignments
                                        if a.node_id != "N1"]
        v1.model.assign_support("N1", "FIXO")
        copy_variant_supports(v2, v1.model)
        self.assertEqual(current_restraints(v2.model), current_restraints(v1.model))
        d = variant_base_diff(base, v2)
        self.assertEqual([k for k in d['overrides'] if k[0] == 'supports'],
                         [("supports", "N1")])               # N1 differs from the base: the choice

    def test_the_first_sync_takes_the_model_as_it_is(self):
        base = _beam()
        v = Variant(id="V1", model=make_variant_model(base))  # no record of the base yet
        v.model.support_assignments = []
        base.add_variant(v)
        sync_variant_with_base(base, v)
        self.assertEqual(current_restraints(v.model), {})
        self.assertEqual(v.base_state['supports'],
                         {n: list(r) for n, r in current_restraints(base).items()})


class TestLoadRestore(unittest.TestCase):
    def test_reactivated_bar_recovers_base_loads(self):
        base = _beam()
        v = _register(base, active={"E1"})     # E2 (and its loads) excluded
        base.add_distributed_load("E2", "LC", fye=-7.0, fyd=-7.0)
        # Reactivate everything.
        v.active_elements = None
        self.assertTrue(sync_variant_with_base(base, v))
        loads_e2 = [dl for dl in v.model.distributed_loads
                    if dl.element_id == "E2"]
        self.assertEqual(len(loads_e2), 1)
        self.assertEqual(loads_e2[0].fye, -7.0)

    def test_a_case_the_base_gains_arrives_with_its_loads(self):
        """It used to stay out (the variant's cases were its own and nothing reached them): now a case the base adds
        comes to a variant that follows the base's actions, loads included."""
        base = _beam()
        v = _register(base, active={"E1"})
        base.add_load_case("SNOW")
        base.add_distributed_load("E2", "SNOW", fye=-3.0, fyd=-3.0)
        v.active_elements = None
        sync_variant_with_base(base, v)
        self.assertIn("SNOW", v.model.load_cases_by_id)
        self.assertEqual([dl.fye for dl in v.model.distributed_loads
                          if dl.element_id == "E2" and dl.load_case_id == "SNOW"], [-3.0])

    def test_variant_own_loads_preserved_by_sync(self):
        base = _beam()
        v = _register(base)
        v.model.add_distributed_load("E2", "LC", fye=-99.0, fyd=-99.0)  # variant's own action
        base.add_node("N4", 15.0, 0.0)
        base.add_bar_element("E3", "N3", "N4", "S")     # base edit
        sync_variant_with_base(base, v)
        own = [dl for dl in v.model.distributed_loads
               if dl.element_id == "E2" and dl.fye == -99.0]
        self.assertEqual(len(own), 1)


class TestSyncGuard(unittest.TestCase):
    def test_sync_is_noop_when_base_unchanged(self):
        base = _beam()
        v = _register(base)
        self.assertTrue(sync_variant_with_base(base, v))    # first sync runs
        self.assertFalse(sync_variant_with_base(base, v))   # guarded no-op
        base.nodes["N2"].y = 1.0
        self.assertTrue(sync_variant_with_base(base, v))    # base changed

    def test_variant_without_model_is_ignored(self):
        base = _beam()
        base.add_variant(Variant(id="V0"))
        self.assertFalse(sync_variant_with_base(base, base.variants["V0"]))

    def test_sync_all_variants(self):
        base = _beam()
        _register(base, "V1")
        _register(base, "V2", active={"E1"})
        base.add_node("N4", 15.0, 0.0)
        base.add_bar_element("E3", "N3", "N4", "S")
        out = sync_all_variants(base)
        self.assertEqual(set(out), {"V1", "V2"})
        self.assertIn("E3", base.variants["V1"].model.bar_elements_by_id)
        self.assertNotIn("E3", base.variants["V2"].model.bar_elements_by_id)

    def test_synced_variant_solves(self):
        base = _beam()
        v = _register(base)
        base.add_node("N4", 15.0, 0.0)
        base.add_bar_element("E3", "N3", "N4", "S")
        base.add_support("R2", uy=True); base.assign_support("N4", "R2")
        sync_variant_with_base(base, v)
        res = base.derive_variant(v).calculate()
        self.assertIn("E3", res["element_forces"]["LC"])


class TestUserEditsSurviveAutoSync(unittest.TestCase):
    """Regressão (2026-08-01): o auto-sync apagava apoios/geometria editados
    diretamente no modelo da variante — as vistas 'deixavam de alternar'
    porque todas as variantes ficavam iguais à base."""

    def _customize_supports(self, v):
        """Simulate the user editing supports on the ACTIVE variant's model
        through the normal editors (no support set involved)."""
        m = v.model
        m.support_assignments = []
        m.add_support("FIXO", ux=True, uy=True, tz=True)
        m.assign_support("N1", "FIXO")
        m.assign_support("N2", "ROLLER")

    def _sup(self, m):
        return sorted((a.node_id, a.support_name)
                      for a in m.support_assignments)

    def test_first_sync_keeps_custom_supports(self):
        # Scenario A: variant with no recorded sync signature (old file /
        # freshly rebuilt Variant). The first auto-sync used to wipe.
        base = _beam()
        v = _register(base)                 # no _synced_base_sig
        self._customize_supports(v)
        before = self._sup(v.model)
        sync_variant_with_base(base, v)     # what selection/Run trigger
        self.assertEqual(self._sup(v.model), before)

    def test_base_support_edit_keeps_custom_supports(self):
        # Scenario B: sig recorded, then the user edits supports on the BASE.
        base = _beam()
        v = _register(base)
        sync_variant_with_base(base, v)     # record sig
        self._customize_supports(v)
        base.add_support("EXTRA", uy=True)
        base.assign_support("N2", "EXTRA")  # base change → sig change
        before = self._sup(v.model)
        self.assertTrue(sync_variant_with_base(base, v))
        self.assertEqual(self._sup(v.model), before)

    def test_variant_local_bar_survives_auto_sync(self):
        # A bar the user added ONLY to the variant's model must survive the
        # auto-sync (it is not distinguishable from a base deletion, so only
        # the explicit destructive reconcile may remove it).
        base = _beam()
        v = _register(base)
        sync_variant_with_base(base, v)
        v.model.add_node("NX", 5.0, 3.0)
        v.model.add_bar_element("EX", "N2", "NX", "S")
        base.nodes["N2"].y = 0.5            # base change → sig change
        self.assertTrue(sync_variant_with_base(base, v))
        self.assertIn("EX", v.model.bar_elements_by_id)
        sync_variant_with_base(base, v, force=True, destructive=True)
        self.assertNotIn("EX", v.model.bar_elements_by_id)

    def test_combination_validity_does_not_wipe(self):
        from xdfem2d.workflows import combination_validity
        base = _beam()
        v = _register(base)
        self._customize_supports(v)
        before = self._sup(v.model)
        info = combination_validity(base, ["BASE", "V1"], "LinearSum")
        self.assertEqual(self._sup(v.model), before)
        # and the verdict reflects the real difference in supports
        self.assertFalse(info["same_stiffness"])

    def test_run_combination_does_not_wipe(self):
        from xdfem2d.workflows import run_variant_combination
        base = _beam()
        v = _register(base)
        self._customize_supports(v)
        before = self._sup(v.model)
        run_variant_combination(base, [("BASE", "LC", 1.0), ("V1", "LC", 1.0)],
                                "Envelope", allow_geometry_mismatch=True)
        self.assertEqual(self._sup(v.model), before)


class TestEmptyModelRepair(unittest.TestCase):
    """Regressão (2026-08-01): variante a mostrar 'só os apoios' — modelo sem
    barras (ficheiro da era do bug) que a guarda do sync nunca reparava, e
    active_elements=set() a esvaziar o modelo derivado."""

    def test_empty_active_set_means_all(self):
        base = _beam()
        v = Variant(id="V1", active_elements=set())      # acidente de dados
        sub = base.derive_variant(v)
        self.assertEqual(sorted(sub.bar_elements_by_id), ["E1", "E2"])

    def test_empty_active_set_normalised_on_roundtrip(self):
        import os, tempfile
        base = _beam()
        base.add_variant(Variant(id="V1", active_elements=set()))
        fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
        try:
            base.save(path); r = Structure2D.load(path)
        finally:
            os.remove(path)
        self.assertIsNone(r.variants["V1"].active_elements)

    def test_barless_model_repaired_despite_sig_guard(self):
        base = _beam()
        v = _register(base)
        sync_variant_with_base(base, v)                  # regista a assinatura
        # dano: modelo perde todas as barras (como nos ficheiros afetados)
        for eid in list(v.model.bar_elements_by_id):
            v.model.remove_element(eid)
        self.assertFalse(v.model.bar_elements)
        # a guarda daria no-op (base intocada) — o reparo tem de vencer
        self.assertTrue(sync_variant_with_base(base, v))
        self.assertEqual(sorted(v.model.bar_elements_by_id), ["E1", "E2"])
        # e os apoios existentes do modelo não foram tocados (não destrutivo)
        self.assertTrue(v.model.support_assignments)


class TestVariantBaseDiff(unittest.TestCase):
    """variant_base_diff reports what a sync would change, without mutating."""

    def test_in_sync_reports_nothing(self):
        base = _beam()
        v = _register(base)
        sync_variant_with_base(base, v)
        d = variant_base_diff(base, v)
        self.assertTrue(d['in_sync'])
        self.assertEqual(d['add'], []); self.assertEqual(d['remove'], [])
        self.assertEqual(d['overrides'], []); self.assertEqual(d['follow'], [])
        self.assertEqual(d['conflicts'], [])

    def test_reports_added_and_removed_elements(self):
        base = _beam()
        v = _register(base)                 # model has E1, E2
        base.add_node("N4", 15.0, 0.0)
        base.add_bar_element("E3", "N3", "N4", "S")
        v.active_elements = {"E1", "E3"}    # E2 out, E3 in
        d = variant_base_diff(base, v)
        self.assertFalse(d['in_sync'])
        self.assertEqual(d['add'], ["E3"])
        self.assertEqual(d['remove'], ["E2"])
        # pure: nothing changed on the model
        self.assertEqual(sorted(v.model.bar_elements_by_id), ["E1", "E2"])

    def test_reports_what_follows_and_what_the_variant_chose(self):
        base = _beam()
        v = _register(base)
        v.model.support_assignments = [a for a in v.model.support_assignments
                                       if a.node_id != "N1"]       # the variant's choice
        base.add_support("R2", uy=True); base.assign_support("N2", "R2")   # the base's change
        d = variant_base_diff(base, v)
        self.assertFalse(d['in_sync'])
        self.assertIn(("supports", "N1"), d['overrides'])
        self.assertEqual(d['follow'], [("supports", "N2")])
        self.assertEqual(d['conflicts'], [])
        # pure: nothing changed on the model
        self.assertNotIn("N2", current_restraints(v.model))

    def test_property_only_change_flags_out_of_sync(self):
        base = _beam()
        v = _register(base)
        sync_variant_with_base(base, v)
        base.sections["S"].h = 0.9          # no add/remove, no supports
        d = variant_base_diff(base, v)
        self.assertFalse(d['in_sync'])
        self.assertEqual(d['add'], []); self.assertEqual(d['remove'], [])

    def test_modelless_variant_is_trivially_in_sync(self):
        base = _beam()
        base.add_variant(Variant(id="V0"))
        d = variant_base_diff(base, base.variants["V0"])
        self.assertTrue(d['in_sync'])


class TestSyncSigPersistence(unittest.TestCase):
    """The sync signature survives a save/load round-trip, so reopening a file
    whose base did not change skips the redundant first reconcile."""

    def test_roundtrip_keeps_sync_guard(self):
        import os, tempfile
        base = _beam()
        v = _register(base)
        self.assertTrue(sync_variant_with_base(base, v))    # records the sig
        fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
        try:
            base.save(path); r = Structure2D.load(path)
        finally:
            os.remove(path)
        rv = r.variants["V1"]
        self.assertEqual(getattr(rv, '_synced_base_sig', None),
                         v._synced_base_sig)
        # Base unchanged after reload → guarded no-op.
        self.assertFalse(sync_variant_with_base(r, rv))

    def test_roundtrip_then_base_edit_triggers_sync(self):
        import os, tempfile
        base = _beam()
        v = _register(base)
        sync_variant_with_base(base, v)
        fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
        try:
            base.save(path); r = Structure2D.load(path)
        finally:
            os.remove(path)
        r.nodes["N2"].x = 6.0
        rv = r.variants["V1"]
        self.assertTrue(sync_variant_with_base(r, rv))
        self.assertEqual(rv.model.nodes["N2"].x, 6.0)


class TestConfigBuild(unittest.TestCase):
    """Regression for P2: a model-less variant's config must drive the model."""

    def test_model_built_from_declared_config(self):
        base = _beam()
        base.add_support_set(SupportSet(
            id="SS1", restraints={"N1": (True, True, False),
                                  "N2": (False, True, False)}))
        m = make_variant_model(base, active_elements={"E1"},
                               support_set=base.support_sets["SS1"])
        self.assertEqual(sorted(m.bar_elements_by_id), ["E1"])
        # The set changes what it lists (N1, N2); the base's support at N3 stays.
        self.assertEqual(sorted(a.node_id for a in m.support_assignments),
                         ["N1", "N2", "N3"])
        self.assertEqual(_restraint_of(m, "N1"), (True, True, False))
        self.assertEqual(_restraint_of(m, "N3"), (False, True, False))

class TestSupportSetValidation(unittest.TestCase):
    """A support set now serves construction phases only; a set that names a
    node the model does not have is reported when it is made and by the model
    check, instead of failing later in the solver."""

    def test_deriving_with_a_set_that_lists_a_missing_node_names_it(self):
        base = _beam()
        ss = SupportSet(id="SS", restraints={"N1": (True, True, False),
                                             "N99": (True, True, False)})
        with self.assertRaises(ValueError) as cm:
            make_variant_model(base, support_set=ss)
        self.assertIn("N99", str(cm.exception))
        self.assertIn("SS", str(cm.exception))

    def test_the_model_check_reports_it(self):
        from xdfem2d.model_check import model_check
        base = _beam()
        base.add_support_set(SupportSet(id="SS", restraints={"N1": (True, True, False),
                                                             "N99": (True, True, False)}))
        kinds = [i['type'] for i in model_check(base)]
        self.assertIn('support_set_node', kinds)
        base.support_sets["SS"].restraints.pop("N99")
        self.assertNotIn('support_set_node', [i['type'] for i in model_check(base)])

    def test_an_empty_set_changes_nothing(self):
        """A set holds what it changes: an empty one leaves the supports as they are (it used to wipe them)."""
        base = _beam()
        m = make_variant_model(base, support_set=SupportSet(id="E"))
        self.assertEqual(current_restraints(m), current_restraints(base))


class TestVariantEditsToBarsSurviveTheBaseChanging(unittest.TestCase):
    """What the user changes on a variant's bars (a hinge, a section, a node's position) stays when the base changes;
    what the base changes reaches the bars the variant did not change. It used to be copied over from the base on
    every sync, so a hinge added on a variant was gone as soon as anything was edited on the base."""

    def test_a_hinge_added_on_the_variant_stays_when_the_base_is_edited(self):
        base = _beam()
        v = _register(base)
        sync_variant_with_base(base, v)
        v.model.bar_elements_by_id["E1"].hinge_j = True              # the user, on the variant
        base.nodes["N3"].x = 11.0                                    # anything on the base
        self.assertTrue(sync_variant_with_base(base, v))
        self.assertTrue(v.model.bar_elements_by_id["E1"].hinge_j)
        self.assertEqual(v.model.nodes["N3"].x, 11.0)                # and the base's change arrived

    def test_the_base_changing_another_field_of_the_same_bar_still_arrives(self):
        base = _beam()
        base.add_section("S2", "M", b=0.4, h=0.8)
        v = _register(base)
        v.model.bar_elements_by_id["E1"].hinge_j = True              # variant: the hinge
        base.bar_elements_by_id["E1"].section_name = "S2"            # base: the section
        sync_variant_with_base(base, v)
        e = v.model.bar_elements_by_id["E1"]
        self.assertTrue(e.hinge_j)
        self.assertEqual(e.section_name, "S2")
        self.assertIn("S2", v.model.sections)                        # with its definition

    def test_when_both_changed_the_same_field_the_variant_keeps_its_own(self):
        base = _beam()
        v = _register(base)
        v.model.bar_elements_by_id["E1"].hinge_j = True
        base.bar_elements_by_id["E1"].hinge_j = False
        base.bar_elements_by_id["E1"].hinge_i = True                 # a different one, so the base did change
        d = variant_base_diff(base, v)
        sync_variant_with_base(base, v)
        e = v.model.bar_elements_by_id["E1"]
        self.assertTrue(e.hinge_j)                                   # the variant's
        self.assertTrue(e.hinge_i)                                   # the base's other field

    def test_a_section_property_edited_on_the_variant_stays(self):
        base = _beam()
        v = _register(base)
        v.model.sections["S"].h = 0.9
        base.nodes["N2"].y = 0.5                                     # a base edit, so a sync runs
        sync_variant_with_base(base, v)
        self.assertEqual(v.model.sections["S"].h, 0.9)
        self.assertEqual(base.sections["S"].h, 0.6)                  # the base is untouched

    def test_a_section_property_changed_on_the_base_reaches_a_variant_that_did_not_touch_it(self):
        base = _beam()
        v = _register(base)
        base.sections["S"].h = 0.9
        sync_variant_with_base(base, v)
        self.assertEqual(v.model.sections["S"].h, 0.9)

    def test_the_explicit_sync_keeps_the_variants_edits_too(self):
        base = _beam()
        v = _register(base)
        v.model.bar_elements_by_id["E2"].hinge_i = True
        base.nodes["N2"].y = 0.5
        sync_variant_with_base(base, v, force=True, destructive=True)     # "Sync with base" / "Save variant"
        self.assertTrue(v.model.bar_elements_by_id["E2"].hinge_i)

    def test_the_diff_reports_the_edit_as_the_variants_choice(self):
        base = _beam()
        v = _register(base)
        v.model.bar_elements_by_id["E1"].hinge_j = True
        d = variant_base_diff(base, v)
        self.assertIn(("bars", "E1|hinge_j"), d['overrides'])

    def test_a_save_and_load_keeps_the_edit_and_the_record(self):
        import os, tempfile
        base = _beam()
        v = _register(base)
        sync_variant_with_base(base, v)
        v.model.bar_elements_by_id["E1"].hinge_j = True
        fd, path = tempfile.mkstemp(suffix=".json"); os.close(fd)
        try:
            base.save(path); r = Structure2D.load(path)
        finally:
            os.remove(path)
        rv = r.variants["V1"]
        self.assertTrue(rv.model.bar_elements_by_id["E1"].hinge_j)
        r.nodes["N3"].x = 11.0
        sync_variant_with_base(r, rv)
        self.assertTrue(rv.model.bar_elements_by_id["E1"].hinge_j)   # still, after the base changed
        self.assertEqual(rv.model.nodes["N3"].x, 11.0)


class TestVariantCombinationsSurviveAX2dSave(unittest.TestCase):
    """A .x2d is read in two goes (variants.json, then sequences.json). The second, without combinations, used to
    wipe the ones the first had loaded: a combination across variants was saved and never came back."""

    def _roundtrip(self, s):
        import os, tempfile
        from xdfem2d.file_io import save_x2d, load_x2d
        d = tempfile.mkdtemp()
        p = os.path.join(d, "m.x2d")
        save_x2d(s, None, p)
        return load_x2d(p)[0]

    def test_a_combination_across_variants_is_still_there_after_save_and_open(self):
        base = _beam()
        _register(base)
        base.variant_combinations["C0"] = {"op": "Envelope",
                                           "terms": [("BASE", "LC", 1.0), ("V1", "LC", 1.0)]}
        r = self._roundtrip(base)
        self.assertEqual(list(r.variant_combinations), ["C0"])
        self.assertEqual(r.variant_combinations["C0"]["terms"],
                         [("BASE", "LC", 1.0), ("V1", "LC", 1.0)])

    def test_with_construction_sequences_too(self):
        from xdfem2d.models import ConstructionPhase, ConstructionSequence
        base = _beam()
        _register(base)
        base.variant_combinations["C0"] = {"op": "LinearSum", "terms": [("V1", "LC", 1.0)]}
        base.add_construction_sequence(ConstructionSequence(
            id="SEQ", phases=[ConstructionPhase(id="P", active_elements={"E1"}, applied_cases=["LC"])]))
        r = self._roundtrip(base)
        self.assertIn("C0", r.variant_combinations)
        self.assertIn("SEQ", r.construction_sequences)

    def test_no_combinations_is_still_fine(self):
        base = _beam()
        self.assertEqual(self._roundtrip(base).variant_combinations, {})


class TestVariantActionsFollowTheBase(unittest.TestCase):
    """The variant's load cases, analysis cases, combinations and loads follow the base where the variant has not
    changed the same thing, and keep what the variant did. They used to be a copy made once: a case or a load the base
    gained afterwards never reached the variant."""

    def test_a_new_base_case_and_its_loads_reach_the_variant(self):
        base = _beam()
        v = _register(base)
        sync_variant_with_base(base, v)
        base.add_load_case("Q")
        base.add_distributed_load("E2", "Q", fye=-5.0, fyd=-5.0)
        base.nodes["N3"].x = 11.0                                     # so that a sync runs
        sync_variant_with_base(base, v)
        self.assertIn("Q", v.model.load_cases_by_id)
        self.assertEqual([d.fye for d in v.model.distributed_loads
                          if d.element_id == "E2" and d.load_case_id == "Q"], [-5.0])

    def test_a_load_the_base_edits_reaches_a_variant_that_did_not_touch_it(self):
        base = _beam()
        v = _register(base)
        sync_variant_with_base(base, v)
        base.distributed_loads[0].fye = base.distributed_loads[0].fyd = -20.0
        sync_variant_with_base(base, v)
        self.assertEqual([d.fye for d in v.model.distributed_loads
                          if d.element_id == "E1" and d.load_case_id == "LC"], [-20.0])

    def test_a_load_the_variant_changed_keeps_its_value(self):
        base = _beam()
        v = _register(base)
        sync_variant_with_base(base, v)
        mine = [d for d in v.model.distributed_loads if d.element_id == "E1"][0]
        mine.fye = mine.fyd = -99.0                                   # the variant's own
        base.distributed_loads[0].fye = base.distributed_loads[0].fyd = -20.0
        d = variant_base_diff(base, v)
        self.assertTrue(d['conflicts'])
        sync_variant_with_base(base, v)
        self.assertEqual([x.fye for x in v.model.distributed_loads if x.element_id == "E1"], [-99.0])

    def test_the_variants_own_case_and_loads_stay(self):
        base = _beam()
        v = _register(base)
        sync_variant_with_base(base, v)
        v.model.add_load_case("W")
        v.model.add_distributed_load("E1", "W", fye=-3.0, fyd=-3.0)
        base.nodes["N3"].x = 11.0
        sync_variant_with_base(base, v, force=True, destructive=True)
        self.assertIn("W", v.model.load_cases_by_id)
        self.assertEqual([d.fye for d in v.model.distributed_loads if d.load_case_id == "W"], [-3.0])

    def test_a_load_added_to_an_existing_group_arrives_without_duplicating_the_rest(self):
        base = _beam()
        v = _register(base)
        sync_variant_with_base(base, v)
        base.add_distributed_load("E1", "LC", fye=-1.0, fyd=-1.0)    # a second load on the same bar and case
        sync_variant_with_base(base, v)
        got = sorted(d.fye for d in v.model.distributed_loads if d.element_id == "E1")
        self.assertEqual(got, sorted(d.fye for d in base.distributed_loads if d.element_id == "E1"))

    def test_a_case_the_base_removes_goes_from_a_variant_that_left_it_alone(self):
        base = _beam()
        base.add_load_case("OLD")
        v = _register(base)
        sync_variant_with_base(base, v)
        self.assertIn("OLD", v.model.load_cases_by_id)
        base.remove_load_case("OLD")
        sync_variant_with_base(base, v)
        self.assertNotIn("OLD", v.model.load_cases_by_id)

    def test_a_removed_base_case_the_variant_changed_stays(self):
        base = _beam()
        base.add_load_case("OLD")
        v = _register(base)
        sync_variant_with_base(base, v)
        v.model.load_cases_by_id["OLD"].self_weight_factor = 2.0       # the variant used it
        base.remove_load_case("OLD")
        sync_variant_with_base(base, v)
        self.assertIn("OLD", v.model.load_cases_by_id)

    def test_a_clean_variant_does_not_take_the_bases_actions(self):
        base = _beam()
        v = new_variant(base, "V1", inherit=False)
        base.add_variant(v)
        self.assertFalse(v.inherit_actions)
        self.assertEqual(v.model.load_cases, [])
        base.add_load_case("Q")
        base.add_distributed_load("E2", "Q", fye=-5.0, fyd=-5.0)
        base.nodes["N3"].x = 11.0
        sync_variant_with_base(base, v)
        self.assertEqual(v.model.load_cases, [])                       # still its own, empty
        self.assertEqual(v.model.nodes["N3"].x, 11.0)                  # while the geometry followed

    def test_it_survives_a_save_and_open(self):
        import os, tempfile
        from xdfem2d.file_io import save_x2d, load_x2d
        base = _beam()
        v = new_variant(base, "V1", inherit=False)
        base.add_variant(v)
        p = os.path.join(tempfile.mkdtemp(), "m.x2d")
        save_x2d(base, None, p)
        r = load_x2d(p)[0]
        self.assertFalse(r.variants["V1"].inherit_actions)
        self.assertIsNotNone(r.variants["V1"].base_state)


class TestVariantCasesAreMarkedSolved(unittest.TestCase):
    """A variant is solved on a copy of its model, so Run never marked the cases of the model itself: they stayed
    pending and the Case selector hid them while their results were there (only the combinations showed)."""

    def test_running_all_variants_marks_their_cases_solved(self):
        from xdfem2d.workflows import solve_all_variants
        base = _beam()
        base.add_analysis_case("LC", "Linear", {"LC": 1.0}) if "LC" not in base.analysis_cases_by_id else None
        v = _register(base)
        for ac in v.model.analysis_cases:
            ac.solved = False
        self.assertTrue(v.model.analysis_cases)
        run = solve_all_variants(base)
        self.assertFalse(run.variant_errors)
        self.assertTrue(all(ac.solved for ac in v.model.analysis_cases))

    def test_what_has_been_run_is_not_something_the_variant_follows(self):
        base = _beam()
        v = _register(base)
        sync_variant_with_base(base, v)
        for ac in v.model.analysis_cases:
            ac.solved = True
        for ac in base.analysis_cases:
            ac.solved = False                                          # e.g. the base was unlocked
        base.nodes["N3"].x = 11.0                                      # so that a sync runs
        sync_variant_with_base(base, v)
        self.assertTrue(all(ac.solved for ac in v.model.analysis_cases))
        self.assertEqual(variant_base_diff(base, v)['overrides'], [])
