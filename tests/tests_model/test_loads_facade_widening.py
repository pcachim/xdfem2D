"""Widening of the loads simple facade (dev/XDFEM2D_ENGINE.md §8, §14).

create_bar_uniform_load / create_area_uniform_load (both since unified into
a single create_uniform_load, dispatching per target's actual kind — bar-like
or area-like — instead of splitting the facade along typology) and
create_bar_point_load / create_node_load keep the same idea their add_*
predecessors had: a magnitude + a named direction instead of signed
components and a reference frame. create_uniform_load's signature is
``(targets, load_case_id, value, direction='down', coord_sys='global')`` —
unlike its two predecessors, load_case_id is a required positional (no
get-or-create default case for this one), targets accept a single ref/id, a
list of ids, or a list mixing bar-like and area-like ids in one call, and it
returns a dict mapping each resolved target id to how it was classified
('bar'/'area') rather than the raw load object(s) — the atomic
validate-then-apply design means there is no per-target object to hand back
that means the same thing across a mixed-kind call.

Checked the same way as the create_* invariant tests: dict equality on the
serialised form (structure_io._to_dict) between the old explicit call style
and the new permissive one.
"""
from __future__ import annotations

import unittest

from context import Structure2D  # noqa: F401  (puts src/ on the path)

from xdfem2d.structure_io import _to_dict


def _bar_model():
    s = Structure2D()
    s.add_node("1", 0.0, 0.0)
    s.add_node("2", 5.0, 0.0)
    s.add_material("M", 30e6, 25.0)
    s.add_section("S", "M", 0.3, 0.5)
    s.add_bar_element("B1", "1", "2", "S")
    return s


class TestPositionalCallShape(unittest.TestCase):
    def test_positional_call_style(self):
        """The exact call shape every existing script/test uses today."""
        s = _bar_model()
        s.add_load_case("LC1")
        result = s.create_uniform_load("B1", "LC1", 10, "down")
        self.assertEqual(result, {"B1": "bar"})
        self.assertAlmostEqual(s.distributed_loads[-1].fye, -10)


class TestListTargetsAreLossless(unittest.TestCase):
    def test_list_target_matches_two_single_calls(self):
        s_old = _bar_model()
        s_old.add_node("3", 10.0, 0.0)
        s_old.add_bar_element("B2", "2", "3", "S")
        s_old.add_load_case("LC1")
        s_old.create_uniform_load("B1", "LC1", 10, "down")
        s_old.create_uniform_load("B2", "LC1", 10, "down")

        s_new = _bar_model()
        s_new.add_node("3", 10.0, 0.0)
        s_new.add_bar_element("B2", "2", "3", "S")
        s_new.add_load_case("LC1")
        s_new.create_uniform_load(["B1", "B2"], "LC1", 10, "down")

        self.assertEqual(_to_dict(s_old), _to_dict(s_new))

    def test_single_target_and_one_item_list_are_both_accepted(self):
        s = _bar_model()
        s.add_load_case("LC1")
        single = s.create_uniform_load("B1", "LC1", 10, "down")
        self.assertEqual(single, {"B1": "bar"})
        many = s.create_uniform_load(["B1"], "LC1", 10, "down")
        self.assertEqual(many, {"B1": "bar"})

    def test_targets_may_be_the_objects_themselves_not_only_ids(self):
        """Late addition to the design: targets may hold the actual element/
        object instances, mixed freely with bare id strings."""
        s = _bar_model()
        s.add_load_case("LC1")
        bar_obj = s.bar_elements_by_id["B1"]
        result = s.create_uniform_load(bar_obj, "LC1", 10, "down")
        self.assertEqual(result, {"B1": "bar"})


class TestBarLoadDispatchesToGeoLineObject(unittest.TestCase):
    def _line_model(self):
        s = Structure2D()
        s.add_material("C", 30e6, 0.0)
        s.add_section("S", "C", b=0.3, h=0.5)
        s.add_node("L.p0", 0, 0)
        s.add_node("L.p1", 6, 0)
        s.add_geo_line("L", 0, 0, 6, 0, section_name="S", divisions=2)
        return s

    def test_matches_a_direct_add_line_distributed_load_call(self):
        s_old = self._line_model()
        s_old.add_load_case("LC1")
        s_old.add_line_distributed_load("L", "LC1", fy=-10)

        s_new = self._line_model()
        s_new.add_load_case("LC1")
        s_new.create_uniform_load("L", "LC1", 10, "down")

        self.assertEqual(_to_dict(s_old), _to_dict(s_new))

    def test_perp_and_axial_are_rejected_for_a_line_object(self):
        s = self._line_model()
        s.add_load_case("LC1")
        with self.assertRaises(ValueError):
            s.create_uniform_load("L", "LC1", 10, direction="perp")

    def test_unknown_target_raises(self):
        s = _bar_model()
        s.add_load_case("LC1")
        with self.assertRaises(ValueError):
            s.create_uniform_load("does-not-exist", "LC1", 10)


class TestMixedBarAndAreaTargetsInOneCall(unittest.TestCase):
    def test_a_bar_and_a_tri_target_dispatch_independently(self):
        s = Structure2D(domain="plate")
        s.add_node("1", 0.0, 0.0)
        s.add_node("2", 5.0, 0.0)
        s.add_node("3", 0.0, 5.0)
        s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        s.add_bar_element("B1", "1", "2", "S")
        s.add_tri_section("TS", "M", thickness=0.2)
        s.add_tri_element("T1", "1", "2", "3", "TS")
        s.add_load_case("LC1")

        result = s.create_uniform_load(["B1", "T1"], "LC1", 5, "down")
        self.assertEqual(result, {"B1": "bar", "T1": "area"})
        self.assertEqual(len(s.distributed_loads), 1)
        self.assertEqual(len(s.tri_area_loads), 1)

    def test_perp_direction_on_an_area_target_is_rejected_atomically(self):
        """direction validity is target-kind-dependent and checked for every
        target before anything is applied — a bad direction on ONE area
        target in a mixed list must leave the whole call with no side
        effects on any target."""
        s = Structure2D(domain="plate")
        s.add_node("1", 0.0, 0.0)
        s.add_node("2", 5.0, 0.0)
        s.add_node("3", 0.0, 5.0)
        s.add_material("M", 30e6, 25.0)
        s.add_section("S", "M", 0.3, 0.5)
        s.add_bar_element("B1", "1", "2", "S")
        s.add_tri_section("TS", "M", thickness=0.2)
        s.add_tri_element("T1", "1", "2", "3", "TS")
        s.add_load_case("LC1")

        with self.assertRaises(ValueError):
            s.create_uniform_load(["B1", "T1"], "LC1", 5, "perp")
        self.assertEqual(s.distributed_loads, [])
        self.assertEqual(s.tri_area_loads, [])


class TestSelfWeightAndTemperature(unittest.TestCase):
    def test_add_self_weight_matches_add_load_case_factor(self):
        s_old = _bar_model()
        s_old.add_load_case("LC1", self_weight_factor=1.0)

        s_new = _bar_model()
        s_new.create_self_weight(1.0)   # default case, created on demand

        self.assertEqual(_to_dict(s_old), _to_dict(s_new))

    def test_add_self_weight_on_an_existing_case_mutates_it_in_place(self):
        s = _bar_model()
        s.add_load_case("LC1", self_weight_factor=0.0)
        s.create_self_weight(0.9, load_case="LC1")
        self.assertEqual(len(s.load_cases), 1)
        self.assertAlmostEqual(s.load_cases_by_id["LC1"].self_weight_factor, 0.9)

    def test_add_temperature_on_a_bar_matches_add_temperature_load(self):
        s_old = _bar_model()
        s_old.add_load_case("LC1")
        s_old.add_temperature_load("B1", "LC1", delta_t_uniform=5, delta_t_gradient=2)

        s_new = _bar_model()
        s_new.add_load_case("LC1")
        s_new.create_temperature("B1", uniform=5, gradient=2, load_case="LC1")

        self.assertEqual(_to_dict(s_old), _to_dict(s_new))

    def test_add_temperature_on_a_geo_line_matches_add_line_temperature_load(self):
        s_old = Structure2D()
        s_old.add_material("C", 30e6, 0.0)
        s_old.add_section("S", "C", b=0.3, h=0.5)
        s_old.add_node("L.p0", 0, 0)
        s_old.add_node("L.p1", 6, 0)
        s_old.add_geo_line("L", 0, 0, 6, 0, section_name="S", divisions=2)
        s_old.add_load_case("LC1")
        s_old.add_line_temperature_load("L", "LC1", dt_uniform=4, dt_gradient=1)

        s_new = Structure2D()
        s_new.add_material("C", 30e6, 0.0)
        s_new.add_section("S", "C", b=0.3, h=0.5)
        s_new.add_node("L.p0", 0, 0)
        s_new.add_node("L.p1", 6, 0)
        s_new.add_geo_line("L", 0, 0, 6, 0, section_name="S", divisions=2)
        s_new.add_load_case("LC1")
        s_new.create_temperature("L", uniform=4, gradient=1, load_case="LC1")

        self.assertEqual(_to_dict(s_old), _to_dict(s_new))

    def test_add_temperature_unknown_target_raises(self):
        s = _bar_model()
        with self.assertRaises(ValueError):
            s.create_temperature("does-not-exist", uniform=5)


if __name__ == "__main__":
    unittest.main()
