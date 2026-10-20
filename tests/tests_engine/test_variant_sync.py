"""Base ↔ variant synchronisation (Fase 2).

Rule under test: a variant's own model is authoritative for its ACTIONS
(loads / cases / combinations), but its GEOMETRY, PROPERTIES and SUPPORTS
follow the base filtered by the variant's declared config. Editing the base
must therefore propagate — via ``sync_variant_with_base`` — into every
variant model, without touching the variant's own actions.
"""
import unittest

from context import Structure2D
from xdfem2d.models import Variant, SupportSet
from xdfem2d.workflows import (make_variant_model, apply_variant_config,
                               sync_variant_with_base, sync_all_variants,
                               variant_base_diff)


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


def _register(base, vid="V1", active=None, sset_id=None):
    v = Variant(id=vid, active_elements=active, support_set_id=sset_id,
                model=make_variant_model(
                    base, active_elements=active,
                    support_set=(base.support_sets.get(sset_id)
                                 if sset_id else None)))
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
    def test_edited_support_set_reapplied(self):
        base = _beam()
        base.add_support_set(SupportSet(
            id="SS1", restraints={"N1": (True, True, True),
                                  "N3": (False, True, False)}))
        v = _register(base, sset_id="SS1")
        self.assertTrue(_restraint_of(v.model, "N1")[2])
        # Edit the set (replace, as the GUI does) → variant must follow.
        base.remove_support_set("SS1")
        base.add_support_set(SupportSet(
            id="SS1", restraints={"N1": (True, True, False),
                                  "N2": (False, True, False),
                                  "N3": (False, True, False)}))
        v.support_set_id = "SS1"           # remove_support_set cleared it
        self.assertTrue(sync_variant_with_base(base, v))
        sup_nodes = sorted(a.node_id for a in v.model.support_assignments)
        self.assertEqual(sup_nodes, ["N1", "N2", "N3"])
        self.assertFalse(_restraint_of(v.model, "N1")[2])

    def test_removed_support_set_keeps_supports_until_explicit_sync(self):
        base = _beam()
        base.add_support_set(SupportSet(
            id="SS1", restraints={"N1": (True, True, True),
                                  "N2": (False, True, False)}))
        v = _register(base, sset_id="SS1")
        base.remove_support_set("SS1")     # clears v.support_set_id
        self.assertIsNone(v.support_set_id)
        # Auto-sync must NOT wipe: the set's supports stay as the variant's own.
        self.assertTrue(sync_variant_with_base(base, v))
        sup_nodes = sorted(a.node_id for a in v.model.support_assignments)
        self.assertEqual(sup_nodes, ["N1", "N2"])
        # The explicit destructive reconcile restores the base supports.
        sync_variant_with_base(base, v, force=True, destructive=True)
        sup_nodes = sorted(a.node_id for a in v.model.support_assignments)
        self.assertEqual(sup_nodes, ["N1", "N3"])

    def test_base_spring_follows_on_destructive_sync_only(self):
        base = _beam()
        v = _register(base)
        base.add_node_spring("N2", ky=1000.0)
        self.assertTrue(sync_variant_with_base(base, v))      # auto: keeps
        self.assertNotIn("N2", v.model.node_springs)
        sync_variant_with_base(base, v, force=True, destructive=True)
        self.assertIn("N2", v.model.node_springs)
        self.assertEqual(v.model.node_springs["N2"].ky, 1000.0)


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

    def test_restore_skips_cases_missing_in_variant(self):
        base = _beam()
        v = _register(base, active={"E1"})
        base.add_load_case("SNOW")
        base.add_distributed_load("E2", "SNOW", fye=-3.0, fyd=-3.0)   # case absent in variant
        v.active_elements = None
        sync_variant_with_base(base, v)
        self.assertFalse([dl for dl in v.model.distributed_loads
                          if dl.element_id == "E2" and dl.load_case_id == "SNOW"])

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
        self.assertFalse(d['supports_changed'])

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

    def test_reports_support_change(self):
        base = _beam()
        base.add_support_set(SupportSet(
            id="SS1", restraints={"N1": (True, True, True)}))
        v = _register(base)                 # built with base supports
        v.support_set_id = "SS1"
        d = variant_base_diff(base, v)
        self.assertFalse(d['in_sync'])
        self.assertTrue(d['supports_changed'])

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
        self.assertEqual(sorted(a.node_id for a in m.support_assignments),
                         ["N1", "N2"])


if __name__ == "__main__":
    unittest.main()
