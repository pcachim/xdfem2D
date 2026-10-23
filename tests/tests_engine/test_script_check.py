"""Reading a model-building script without running it.

This exists for one mistake, and the mistake is mine: writing a test in this
project, with the source open beside me, I called
``add_distributed_load(wy=-10.0)``. The parameters are ``fye`` and ``fyd``. A
language model working from its memory of other FEM libraries makes that error
constantly, and nothing says so — the script reads correctly and fails only
when the user runs it.

Two properties are held here above all. The check finds that class of mistake,
and it never runs anything: the whole value of a script the user reads before
executing is lost if something else executes it first.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from context import Structure2D

from xdfem2d import script_check as C
from xdfem2d import script_export as X
from xdfem2d.file_io import load_x2d

HEAD = "from xdfem2d import Structure2D\ns = Structure2D()\n"


def _check(body: str):
    return C.check(HEAD + body)


class TestTheMistakeItExistsFor(unittest.TestCase):

    def test_a_parameter_from_another_library(self):
        found = _check("s.add_distributed_load('E1', 'LC', wy=-10.0)\n")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['kind'], 'arguments')
        self.assertIn('wy', found[0]['msg'])

    def test_the_real_parameters_are_named_in_the_message(self):
        """So the reader can fix it without going to look."""
        msg = _check("s.add_distributed_load('E1', 'LC', wy=-10.0)\n")[0]['msg']
        self.assertIn('fye', msg)
        self.assertIn('fyd', msg)

    def test_the_correct_call_is_silent(self):
        self.assertEqual(
            _check("s.add_distributed_load('E1', 'LC', fye=-10.0, fyd=-10.0)\n"),
            [])


class TestWhatElseItCatches(unittest.TestCase):

    def test_a_method_that_does_not_exist(self):
        found = _check("s.add_beam('E1', 'N1', 'N2')\n")
        self.assertEqual(found[0]['kind'], 'unknown_method')
        self.assertIn('add_beam', found[0]['msg'])

    def test_a_missing_required_argument(self):
        found = _check("s.add_node('N1')\n")
        self.assertEqual(found[0]['kind'], 'arguments')
        self.assertIn("'x'", found[0]['msg'])

    def test_too_many_positional_arguments(self):
        found = _check("s.add_node('N1', 0.0, 0.0, 0.0)\n")
        self.assertEqual(found[0]['kind'], 'arguments')

    def test_a_file_that_is_not_valid_python(self):
        found = C.check("s.add_node('N1',\n")
        self.assertEqual(found[0]['kind'], 'syntax')

    def test_findings_are_reported_in_file_order(self):
        found = _check("s.add_beam('a')\n"
                       "s.add_node('N1')\n"
                       "s.add_wobble()\n")
        self.assertEqual([p['line'] for p in found], [3, 4, 5])


class TestTheSuggestionIsWorthReading(unittest.TestCase):
    """A confident wrong suggestion is worse than none: it reads as
    knowledge."""

    def test_a_shortened_name_points_at_the_real_one(self):
        found = _check("s.add_dist_load('E1', 'LC')\n")
        self.assertIn('add_distributed_load', found[0]['msg'])

    def test_character_similarity_alone_would_have_got_this_wrong(self):
        """'add_dist_load' scores 0.81 against add_point_load and 0.79 against
        add_distributed_load, so the obvious implementation sends the reader
        to the wrong method."""
        self.assertNotIn('add_point_load',
                         _check("s.add_dist_load('E1', 'LC')\n")[0]['msg'])

    def test_a_typo_is_recognised(self):
        self.assertIn('add_node', _check("s.add_nod('N1', 0, 0)\n")[0]['msg'])

    def test_a_name_missing_an_inner_word_points_at_the_real_one(self):
        """'add_rectangle' is every word of 'add_geo_rectangle', in order, with
        only 'geo' left out — a small model reaches for it. The positional match
        misses that (the inserted word shifts everything after it); the
        subsequence check catches it."""
        msg = _check("s.add_rectangle(0, 0, 3, 5)\n")[0]['msg']
        self.assertIn('add_geo_rectangle', msg)

    def test_an_invented_name_gets_no_suggestion(self):
        """add_wall_load is not a misspelling of anything; sending the reader
        to add_point_load would be inventing an intention."""
        msg = _check("s.add_wall_load('W1')\n")[0]['msg']
        self.assertIn('add_wall_load', msg)
        self.assertNotIn('did you mean', msg)


class TestWhatItLeavesAlone(unittest.TestCase):

    def test_data_attributes_are_not_methods(self):
        """s.geometry_objects[id] = obj is how the loader builds objects, and
        the exporter writes it."""
        self.assertEqual(_check("print(s.nodes['N1'].x)\n"), [])

    def test_calls_on_other_objects_are_none_of_its_business(self):
        self.assertEqual(_check("import math\nmath.hypot(3, 4)\n"), [])

    def test_a_call_assembled_at_run_time_is_not_guessed_at(self):
        """**kwargs cannot be checked without evaluating, and evaluating is
        the one thing this must never do."""
        self.assertEqual(_check("kw = {'x': 1.0}\ns.add_node('N1', **kw)\n"), [])

    def test_a_method_with_its_own_kwargs_accepts_them(self):
        self.assertEqual(
            _check("s.add_analysis_case('PD', 'GeometricNonlinear', {}, "
                   "max_iterations=50)\n"), [])


class TestAttributeAccessThatIsNotACall(unittest.TestCase):
    """Found for real: a model wrote ``for e in model.mesh['bars']:`` —
    Structure2D has no ``mesh`` attribute at all (bar elements live at
    ``bar_elements``/``bar_elements_by_id``), and the checker gave no
    error, because ``model.mesh`` is never an ``ast.Call`` — only the
    per-call loop existed before this, and it only ever looks at a Call's
    own ``.func``. See dev/XDFEM2D_ENGINE.md §20.
    """

    def test_an_invented_attribute_used_as_a_value_is_caught(self):
        found = _check("for e in s.mesh['bars']:\n    pass\n")
        kinds = {p['kind'] for p in found}
        self.assertIn('unknown_attribute', kinds)
        msg = next(p['msg'] for p in found if p['kind'] == 'unknown_attribute')
        self.assertIn('mesh', msg)

    def test_a_real_data_attribute_is_still_silent(self):
        """The previous hand-maintained _DATA_ATTRS list was missing
        'domain', 'cuts' and about a dozen others — computed now from a
        real instance instead, so this must never false-positive on them."""
        self.assertEqual(_check("print(s.domain)\n"), [])
        self.assertEqual(_check("print(len(s.cuts))\n"), [])
        self.assertEqual(_check("print(len(s.tri_edge_loads))\n"), [])

    def test_a_bound_method_reference_is_not_an_unknown_attribute(self):
        """`f = s.pin` (no call) names a real method — not calling it is
        not a mistake this check is entitled to have an opinion on."""
        self.assertEqual(_check("f = s.pin\n"), [])

    def test_a_normal_call_is_not_double_reported(self):
        """s.add_beam('E1') is already reported once, as unknown_method
        (with its own closest-match search over methods only) — the
        attribute-access check must not add a second finding for the same
        node."""
        found = _check("s.add_beam('E1')\n")
        self.assertEqual([p['kind'] for p in found], ['unknown_method'])

    def test_a_near_miss_data_attribute_gets_a_suggestion(self):
        found = _check("print(s.material)\n")
        msg = next(p['msg'] for p in found if p['kind'] == 'unknown_attribute')
        self.assertIn('materials', msg)


class TestTheCreateLayerIsRecognisedAsCreating(unittest.TestCase):
    """create_load_case was the one create_* registered in _CREATES; every
    other create_* had the identical gap, just never found until a real
    script wrote create_bar_element(..., id='E1') and referenced 'E1' from
    add_distributed_load two lines later — a genuine dangling-reference
    false positive on an id the script plainly creates. See
    dev/XDFEM2D_ENGINE.md §22.
    """

    def test_create_bar_element_id_is_recognised(self):
        # The exact shape reported by the user: create_bar_element(...,
        # id='E1') then add_distributed_load('E1', ...).
        self.assertEqual(_check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_node(5.0, 0.0, id='N2')\n"
            "s.create_rc_section('S1', 0.3, 0.5)\n"
            "s.create_bar_element(node_i='N1', node_j='N2', section='S1', "
            "id='E1')\n"
            "s.add_load_case('LC1')\n"
            "s.add_distributed_load('E1', 'LC1', fye=-10.0, fyd=-10.0)\n"),
            [])

    def test_create_area_element_id_is_recognised(self):
        found = _check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_node(4.0, 0.0, id='N2')\n"
            "s.create_node(4.0, 3.0, id='N3')\n"
            "s.create_area_element('T1', 'N1', 'N2', 'N3', id='E1')\n"
            "s.assign_support('N1', 'PIN')\n")
        # 'E1' is created by create_area_element — must never show up as a
        # dangling reference, whatever else this particular snippet reports.
        self.assertFalse(any("id='E1'" in p.get('msg', '') for p in found))

    def test_create_rc_section_name_is_recognised(self):
        self.assertEqual(_check(
            "s.create_rc_section('S1', 0.3, 0.5)\n"
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_node(5.0, 0.0, id='N2')\n"
            "s.add_bar_element('E1', 'N1', 'N2', 'S1')\n"), [])

    def test_create_support_name_is_recognised(self):
        self.assertEqual(_check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_support(['N1'], ux=True, uy=True, name='SUP1')\n"
            "s.assign_support('N1', 'SUP1')\n"), [])

    def test_create_analysis_case_id_is_recognised(self):
        self.assertEqual(_check(
            "s.create_load_case('LC1')\n"
            "s.create_analysis_case('AC1', 'Linear', {'LC1': 1.0})\n"), [])


class TestCreateBarElementAndCreateSupportReferencesAreChecked(unittest.TestCase):
    """create_bar_element's node_i/node_j/section and create_support's
    nodes had no _REFERS coverage at all — unlike the fundamental
    add_bar_element, which already catches a dangling node/section. Nodes
    and sections are always statically, explicitly created (never a
    runtime-mesh id the way a tri/quad id legitimately can be), so this is
    a genuine gap, not a case for the deliberate exclusions elsewhere in
    _REFERS."""

    def test_a_dangling_node_i_is_reported(self):
        found = _check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_node(5.0, 0.0, id='N2')\n"
            "s.create_rc_section('S1', 0.3, 0.5)\n"
            "s.create_bar_element(section='S1', node_i='NX', node_j='N2')\n")
        self.assertTrue(any(p['kind'] == 'unknown_id' and 'NX' in p['msg']
                            for p in found))

    def test_a_dangling_node_j_is_reported(self):
        found = _check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_node(5.0, 0.0, id='N2')\n"
            "s.create_rc_section('S1', 0.3, 0.5)\n"
            "s.create_bar_element(section='S1', node_i='N1', node_j='NX')\n")
        self.assertTrue(any(p['kind'] == 'unknown_id' and 'NX' in p['msg']
                            for p in found))

    def test_a_dangling_section_is_reported(self):
        found = _check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_node(5.0, 0.0, id='N2')\n"
            "s.create_bar_element(section='SX', node_i='N1', node_j='N2')\n")
        self.assertTrue(any(p['kind'] == 'unknown_id' and 'SX' in p['msg']
                            for p in found))

    def test_real_bar_element_references_are_silent(self):
        self.assertEqual(_check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_node(5.0, 0.0, id='N2')\n"
            "s.create_rc_section('S1', 0.3, 0.5)\n"
            "s.create_bar_element(section='S1', node_i='N1', node_j='N2')\n"
            "s.pin('N1')\n"), [])

    def test_one_bad_id_in_a_create_support_list_is_reported(self):
        found = _check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_support(['N1', 'NX'], ux=True, uy=True, name='SUP1')\n")
        self.assertTrue(any(p['kind'] == 'unknown_id' and 'NX' in p['msg']
                            for p in found))
        self.assertFalse(any('N1' in p['msg'] and p['kind'] == 'unknown_id'
                             for p in found))

    def test_every_bad_id_in_a_create_support_list_is_reported(self):
        found = [p for p in _check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_support(['NX', 'NY'], ux=True, uy=True, name='SUP1')\n")
            if p['kind'] == 'unknown_id']
        self.assertTrue(any('NX' in p['msg'] for p in found))
        self.assertTrue(any('NY' in p['msg'] for p in found))

    def test_a_real_create_support_node_list_is_silent(self):
        self.assertEqual(_check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_node(5.0, 0.0, id='N2')\n"
            "s.create_support(['N1', 'N2'], ux=True, uy=True, name='SUP1')\n"
            "s.assign_support('N1', 'SUP1')\n"
            "s.assign_support('N2', 'SUP1')\n"), [])


class TestAreaElementLoadAndEdgeReferencesAreChecked(unittest.TestCase):
    """create_area_element/add_area_element/add_tri_element/add_quad_element's
    node_i/node_j/node_k/node_l/section(_name), create_bar_point_load's
    elements, create_node_load's nodes, create_support_settlement's nodes,
    and create_edge_load's object_id had no _REFERS coverage — the same gap
    already closed for create_bar_element and create_support above."""

    def test_a_dangling_node_in_create_area_element_triangle_is_reported(self):
        found = _check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_node(5.0, 0.0, id='N2')\n"
            "s.create_node(0.0, 5.0, id='N3')\n"
            "s.create_area_section('AS1', 'Mat', thickness=0.2)\n"
            "s.create_area_element(section='AS1', node_i='N1', node_j='N2', "
            "node_k='NX')\n")
        self.assertTrue(any(p['kind'] == 'unknown_id' and 'NX' in p['msg']
                            for p in found))

    def test_a_real_create_area_element_triangle_is_silent_node_l_none(self):
        """3-node (triangle) call: node_l is omitted/None and must not be
        reported as a dangling reference."""
        self.assertEqual(_check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_node(5.0, 0.0, id='N2')\n"
            "s.create_node(0.0, 5.0, id='N3')\n"
            "s.create_area_section('AS1', 'Mat', thickness=0.2)\n"
            "s.create_area_element(section='AS1', node_i='N1', node_j='N2', "
            "node_k='N3')\n"), [])

    def test_a_dangling_node_l_in_create_area_element_quad_is_reported(self):
        found = _check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_node(5.0, 0.0, id='N2')\n"
            "s.create_node(5.0, 5.0, id='N3')\n"
            "s.create_node(0.0, 5.0, id='N4')\n"
            "s.create_area_section('AS1', 'Mat', thickness=0.2)\n"
            "s.create_area_element(section='AS1', node_i='N1', node_j='N2', "
            "node_k='N3', node_l='NX')\n")
        self.assertTrue(any(p['kind'] == 'unknown_id' and 'NX' in p['msg']
                            for p in found))

    def test_a_real_create_area_element_quad_is_silent(self):
        self.assertEqual(_check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_node(5.0, 0.0, id='N2')\n"
            "s.create_node(5.0, 5.0, id='N3')\n"
            "s.create_node(0.0, 5.0, id='N4')\n"
            "s.create_area_section('AS1', 'Mat', thickness=0.2)\n"
            "s.create_area_element(section='AS1', node_i='N1', node_j='N2', "
            "node_k='N3', node_l='N4')\n"), [])

    def test_a_dangling_section_in_add_area_element_is_reported(self):
        found = _check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_node(5.0, 0.0, id='N2')\n"
            "s.create_node(0.0, 5.0, id='N3')\n"
            "s.add_area_element('T1', 'N1', 'N2', 'N3', section_name='SX')\n")
        self.assertTrue(any(p['kind'] == 'unknown_id' and 'SX' in p['msg']
                            for p in found))

    def test_a_dangling_node_in_create_bar_point_load_list_is_reported(self):
        found = _check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_node(5.0, 0.0, id='N2')\n"
            "s.create_rc_section('S1', 0.3, 0.5)\n"
            "s.create_bar_element(section='S1', node_i='N1', node_j='N2', "
            "id='E1')\n"
            "s.create_load_case('LC1')\n"
            "s.create_bar_point_load(['E1', 'EX'], 'LC1', p=10.0)\n")
        self.assertTrue(any(p['kind'] == 'unknown_id' and 'EX' in p['msg']
                            for p in found))

    def test_a_real_create_bar_point_load_is_silent(self):
        self.assertEqual(_check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_node(5.0, 0.0, id='N2')\n"
            "s.create_rc_section('S1', 0.3, 0.5)\n"
            "s.create_bar_element(section='S1', node_i='N1', node_j='N2', "
            "id='E1')\n"
            "s.create_load_case('LC1')\n"
            "s.create_bar_point_load('E1', 'LC1', p=10.0)\n"), [])

    def test_a_dangling_node_in_create_node_load_is_reported(self):
        found = _check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_load_case('LC1')\n"
            "s.create_node_load('NX', 'LC1', p=10.0)\n")
        self.assertTrue(any(p['kind'] == 'unknown_id' and 'NX' in p['msg']
                            for p in found))

    def test_a_real_create_node_load_is_silent(self):
        self.assertEqual(_check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_load_case('LC1')\n"
            "s.create_node_load('N1', 'LC1', p=10.0)\n"), [])

    def test_a_dangling_node_in_create_support_settlement_is_reported(self):
        found = _check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_load_case('LC1')\n"
            "s.create_support_settlement(['N1', 'NX'], 'LC1', uy=-0.01)\n")
        self.assertTrue(any(p['kind'] == 'unknown_id' and 'NX' in p['msg']
                            for p in found))

    def test_a_real_create_support_settlement_is_silent(self):
        self.assertEqual(_check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_load_case('LC1')\n"
            "s.create_support_settlement('N1', 'LC1', uy=-0.01)\n"), [])

    def test_a_dangling_object_id_in_create_edge_load_is_reported(self):
        found = _check(
            "s.add_geo_rectangle('Laje1', (0.0, 0.0), (3.0, 5.0), "
            "section_name='Slab')\n"
            "s.create_load_case('LC1')\n"
            "s.create_edge_load('LajeX', 'top', 'LC1', p=10.0)\n")
        self.assertTrue(any(p['kind'] == 'unknown_id' and 'LajeX' in p['msg']
                            for p in found))

    def test_a_real_create_edge_load_object_id_is_silent(self):
        self.assertEqual(_check(
            "s.add_geo_rectangle('Laje1', (0.0, 0.0), (3.0, 5.0), "
            "section_name='Slab')\n"
            "s.create_load_case('LC1')\n"
            "s.create_edge_load('Laje1', 'top', 'LC1', p=10.0)\n"), [])


class TestTheLoadsFacadeReferencesAreChecked(unittest.TestCase):
    """The create_* loads facade (create_uniform_load and siblings)
    had no _REFERS coverage at all — a load_case_id/load_case left
    dangling by a typo went unreported even though the equivalent
    fundamental add_* calls already catch it. See
    dev/XDFEM2D_ENGINE.md §22."""

    def test_a_dangling_load_case_id_is_reported(self):
        found = _check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_node(5.0, 0.0, id='N2')\n"
            "s.create_rc_section('S1', 0.3, 0.5)\n"
            "s.create_bar_element('S1', 'N1', 'N2', id='E1')\n"
            "s.create_uniform_load('E1', 'LCX', 10, 'down')\n")
        self.assertEqual(found[0]['kind'], 'unknown_id')
        self.assertIn('LCX', found[0]['msg'])

    def test_a_real_load_case_is_silent(self):
        self.assertEqual(_check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_node(5.0, 0.0, id='N2')\n"
            "s.create_rc_section('S1', 0.3, 0.5)\n"
            "s.create_bar_element('S1', 'N1', 'N2', id='E1')\n"
            "s.create_load_case('LC1')\n"
            "s.create_uniform_load('E1', 'LC1', 10, 'down')\n"), [])

    def test_create_self_weight_is_checked(self):
        """create_self_weight actually raises KeyError at run time for a
        not-yet-created literal case — an even stronger reason to catch
        it before the script runs."""
        found = _check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_self_weight(1.0, 'LCX')\n")
        self.assertEqual(found[0]['kind'], 'unknown_id')

    def test_create_temperature_uses_its_own_keyword_spelling(self):
        """The parameter is load_case, not load_case_id — different
        keyword, same idea."""
        found = _check(
            "s.create_node(0.0, 0.0, id='N1')\n"
            "s.create_node(5.0, 0.0, id='N2')\n"
            "s.create_rc_section('S1', 0.3, 0.5)\n"
            "s.create_bar_element('S1', 'N1', 'N2', id='E1')\n"
            "s.create_temperature('E1', uniform=5.0, load_case='LCX')\n")
        self.assertEqual(found[0]['kind'], 'unknown_id')
        self.assertIn('LCX', found[0]['msg'])


class TestWhichVariableHoldsTheModel(unittest.TestCase):

    def test_the_usual_construction(self):
        self.assertTrue(C.check("from xdfem2d import Structure2D\n"
                                "model = Structure2D()\n"
                                "model.add_beam('E1')\n"))

    def test_a_model_returned_by_build(self):
        """What the exporter writes: build() then calculate()."""
        self.assertTrue(C.check("m = build()\nm.add_beam('E1')\n"))

    def test_an_annotated_parameter(self):
        self.assertTrue(C.check(
            "def edit(s: Structure2D):\n    s.add_beam('E1')\n"))

    def test_an_unknown_variable_is_left_alone(self):
        """Better to miss one than to report a mistake that is not there — the
        reader who sees a false finding learns to ignore the true ones.

        The silence is about the *call*: nothing is said about add_beam on a
        variable that may hold anything. With a real model in the file there is
        no finding at all.
        """
        self.assertEqual(_check("whatever.add_beam('E1')\n"), [])


class TestAScriptThatBuildsNoModel(unittest.TestCase):
    """The failure the benchmark found, and the worst one this could have.

    granite4.1:8b was asked for a beam and wrote ``import xdfem2d as struc``
    followed by ``struc.Material(…)``, ``struc.Node(…)``, ``struc.solve()`` —
    an API invented whole. Not one call was on a Structure2D, so there was
    nothing to compare, and the checker replied "no problems found". The model
    passed that on as having verified the script.

    A silent pass on a script that cannot run is worse than no checker: it is
    the checker's authority lent to a fabrication.
    """

    def test_a_script_with_no_structure_is_reported(self):
        found = C.check("import xdfem2d as struc\n"
                        "mat = struc.Material(name='M', E=3.3e7)\n"
                        "struc.solve()\n")
        self.assertEqual(found[0]['kind'], 'no_model')

    def test_the_message_says_how_a_model_is_actually_made(self):
        """A finding the reader cannot act on only tells them to give up."""
        msg = C.check("struc.solve()\n")[0]['msg']
        self.assertIn('Structure2D()', msg)
        self.assertIn('add_node', msg)

    def test_the_exact_reply_that_scored_zero_is_now_caught(self):
        self.assertTrue(C.check(
            "import xdfem2d as struc\n"
            "n0 = struc.Node(id='n0', pos=[0.0, 0.0])\n"
            "struc.add_bar_element(element_id='Be1', node_i='n0')\n"))

    def test_a_file_with_no_calls_at_all_is_not_nagged(self):
        """A constants file or a stub is not a broken model script."""
        self.assertEqual(C.check("E = 30e6\nNAME = 'beam'\n"), [])

    def test_a_real_script_is_still_silent(self):
        self.assertEqual(_check("s.add_node('N1', 0.0, 0.0)\n"), [])


class TestToolNamesDoNotBelongInScripts(unittest.TestCase):
    """The same run wrote ``struc.model_check_tool(kind='', limit=10)`` into
    the file. A tool exists in the conversation and nowhere else, so this is a
    category error rather than a typo, and no suggestion would be honest."""

    def test_a_tool_call_is_reported(self):
        found = _check("s.add_node('N1', 0.0, 0.0)\n"
                       "out = model_check_tool(kind='')\n")
        self.assertEqual(found[0]['kind'], 'tool_in_script')

    def test_it_is_caught_as_an_attribute_too(self):
        found = _check("x = struc.reactions_tool(case='ULS')\n")
        kinds = {p['kind'] for p in found}
        self.assertIn('tool_in_script', kinds)

    def test_the_message_explains_the_distinction(self):
        msg = _check("out = model_check_tool()\n")[0]['msg']
        self.assertIn('conversation', msg)

    def test_a_method_ending_in_tool_would_be_a_problem_here(self):
        """Guarding the rule: no Structure2D method ends in '_tool', so the
        suffix is unambiguous. If one is ever added, this fails and says so."""
        import inspect as _i
        from xdfem2d.structure import Structure2D
        named = [n for n, _ in _i.getmembers(Structure2D, _i.isfunction)
                 if n.endswith('_tool')]
        self.assertEqual(named, [])


class TestNoneWhereAStringIsRequired(unittest.TestCase):
    """``bind`` accepts ``id=None``, so the checker used to call
    ``add_node(id=None, x=6.0, y=6.0)`` "all valid". It fails when run."""

    def test_none_for_a_required_string_is_reported(self):
        found = _check("s.add_node(id=None, x=6.0, y=6.0)\n")
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0]['kind'], 'arguments')
        self.assertIn("'id' cannot be None", found[0]['msg'])

    def test_positional_none_is_reported_too(self):
        self.assertEqual(len(_check("s.add_node(None, 6.0, 6.0)\n")), 1)

    def test_none_where_it_is_allowed_is_left_alone(self):
        # create_node(x, y, id: str | None = None)
        self.assertEqual(_check("s.create_node(6.0, 6.0, id=None)\n"), [])

    def test_a_real_string_is_left_alone(self):
        self.assertEqual(_check("s.add_node('N1', 6.0, 6.0)\n"), [])

    def test_the_increment_check_reports_it_as_well(self):
        found = C.check_increment("model.add_node(id=None, x=6.0, y=6.0)\n")
        self.assertEqual(len(found), 1)
        self.assertIn("'id' cannot be None", found[0]['msg'])


class TestCompleteness(unittest.TestCase):
    """A whole build() script with no support assigned is a mechanism — the one
    presence check robust enough to run on generated Python. Fragments and
    single-call snippets are left alone."""

    _WHOLE = (
        "from xdfem2d import Structure2D\n"
        "def build():\n"
        "    s = Structure2D()\n"
        "    s.add_material('M', 30e6, 25.0)\n"
        "    s.add_section('S', 'M', 0.3, 0.5)\n"
        "    s.add_node('N1', 0, 0); s.add_node('N2', 5, 0)\n"
        "    s.add_bar_element('E1', 'N1', 'N2', 'S')\n"
        "{support}"
        "    s.add_load_case('LC')\n"
        "    s.add_analysis_case('ULS', 'Linear', {{'LC': 1.35}})\n"
        "    return s\n"
    )

    def test_a_whole_script_with_no_support_is_flagged(self):
        found = C.check(self._WHOLE.format(support=""))
        self.assertTrue(any(p['kind'] == 'incomplete' for p in found))
        self.assertIn('mechanism',
                      next(p['msg'] for p in found if p['kind'] == 'incomplete'))

    def test_a_pin_clears_it(self):
        found = C.check(self._WHOLE.format(support="    s.pin('N1')\n"))
        self.assertEqual([p for p in found if p['kind'] == 'incomplete'], [])

    def test_assign_support_clears_it(self):
        body = ("    s.add_support('PIN', ux=True, uy=True)\n"
                "    s.assign_support('N1', 'PIN')\n")
        found = C.check(self._WHOLE.format(support=body))
        self.assertEqual([p for p in found if p['kind'] == 'incomplete'], [])

    def test_a_fragment_is_not_judged_incomplete(self):
        # No build() and no calculate(): a snippet, not a whole model.
        found = C.check("from xdfem2d import Structure2D\n"
                        "s = Structure2D()\n"
                        "s.add_node('N1', 0, 0)\n")
        self.assertEqual(found, [])


class TestItNeverRuns(unittest.TestCase):
    """The property that makes this safe to point at a model's output."""

    def test_a_script_with_side_effects_is_only_read(self):
        marker = Path('/tmp/xdfem2d_script_check_must_not_exist')
        if marker.exists():
            marker.unlink()
        C.check(f"from pathlib import Path\n"
                f"Path({str(marker)!r}).write_text('ran')\n"
                f"s = Structure2D()\n"
                f"s.add_node('N1', 0, 0)\n")
        self.assertFalse(marker.exists(), "the checker executed the script")

    def test_it_does_not_import_what_the_script_imports(self):
        found = C.check("import a_module_that_does_not_exist\n"
                        "s = Structure2D()\n"
                        "s.add_node('N1', 0, 0)\n")
        self.assertEqual(found, [])

    def test_the_module_never_calls_exec_or_eval(self):
        src = (Path(__file__).resolve().parent.parent.parent / 'src' / 'xdfem2d'
               / 'script_check.py').read_text(encoding='utf-8')
        # These must never appear as *calls* in the module. They may appear as
        # string literals inside _FORBIDDEN_CALLS (e.g. '__import__' in the
        # set of names to detect in user scripts) — that is legitimate. We
        # check for the call form "name(" rather than the bare name.
        for forbidden in ('exec(', 'eval(', 'compile(', 'importlib',
                          'subprocess'):
            self.assertNotIn(forbidden, src)
        # __import__ must not be *called* — appearing as a string literal to
        # detect it in user code is fine.
        import ast as _ast
        tree = _ast.parse(src)
        for node in _ast.walk(tree):
            if isinstance(node, _ast.Call):
                fn = node.func
                name = getattr(fn, 'id', None) or getattr(fn, 'attr', None)
                self.assertNotEqual(
                    name, '__import__',
                    f"script_check.py calls __import__ at line {node.lineno}")


class TestAgainstWhatWeGenerate(unittest.TestCase):
    """The exporter is the reference implementation: everything it writes must
    pass, or one of the two is wrong."""

    def test_every_example_exports_to_a_clean_script(self):
        import glob
        paths = sorted(glob.glob(str(Path(__file__).resolve().parent.parent.parent
                                     / 'examples' / '*.x2d')))
        self.assertTrue(paths)
        for path in paths:
            with self.subTest(model=Path(path).name):
                struc, _, _ = load_x2d(path)
                self.assertEqual(C.check(X.to_python(struc)), [])

    def test_a_deliberately_broken_export_is_caught(self):
        """Proof the previous test can fail."""
        s = Structure2D()
        s.add_material('M', elastic_modulus=30e6, unit_weight=25.0)
        s.add_node('N1', 0.0, 0.0)
        text = X.to_python(s).replace("s.add_node('N1'", "s.add_nodes('N1'")
        self.assertTrue(C.check(text))


class TestEditorSecurityChecks(unittest.TestCase):
    """check_editor() enforces sandbox rules on top of the normal checks."""

    # --- allowed scripts ---------------------------------------------------

    def test_allowed_imports_are_silent(self):
        for mod in ('xdfem2d', 'math', 'numpy', 'itertools', 'functools'):
            with self.subTest(mod=mod):
                src = f"import {mod}\n" + HEAD + "s.pin('N1')\n"
                found = C.check_editor(src)
                kinds = {p['kind'] for p in found}
                self.assertNotIn('forbidden_import', kinds)

    def test_from_import_allowed_module_is_silent(self):
        src = "from xdfem2d import Structure2D\n" + HEAD + "s.pin('N1')\n"
        found = C.check_editor(src)
        kinds = {p['kind'] for p in found}
        self.assertNotIn('forbidden_import', kinds)

    def test_normal_script_passes_security(self):
        src = HEAD + "s.add_node('N1', 0, 0)\n"
        found = C.check_editor(src)
        self.assertFalse(any(p['kind'].startswith('forbidden') or
                             p['kind'] == 'dunder_access' for p in found))

    # --- forbidden imports -------------------------------------------------

    def test_os_import_is_rejected(self):
        src = "import os\n" + HEAD
        found = C.check_editor(src)
        self.assertTrue(any(p['kind'] == 'forbidden_import' for p in found))

    def test_sys_import_is_rejected(self):
        src = "import sys\n" + HEAD
        found = C.check_editor(src)
        self.assertTrue(any(p['kind'] == 'forbidden_import' for p in found))

    def test_subprocess_import_is_rejected(self):
        src = "import subprocess\n" + HEAD
        found = C.check_editor(src)
        self.assertTrue(any(p['kind'] == 'forbidden_import' for p in found))

    def test_from_os_path_is_rejected(self):
        src = "from os.path import join\n" + HEAD
        found = C.check_editor(src)
        self.assertTrue(any(p['kind'] == 'forbidden_import' for p in found))

    def test_forbidden_import_carries_allowlist_in_message(self):
        src = "import os\n" + HEAD
        found = C.check_editor(src)
        msg = next(p['msg'] for p in found if p['kind'] == 'forbidden_import')
        self.assertIn('xdfem2d', msg)
        self.assertIn('math', msg)

    # --- forbidden builtins ------------------------------------------------

    def test_open_call_is_rejected(self):
        src = HEAD + "open('/etc/passwd')\n"
        found = C.check_editor(src)
        self.assertTrue(any(p['kind'] == 'forbidden_builtin' for p in found))

    def test_exec_call_is_rejected(self):
        src = HEAD + "exec('import os')\n"
        found = C.check_editor(src)
        self.assertTrue(any(p['kind'] == 'forbidden_builtin' for p in found))

    def test_eval_call_is_rejected(self):
        src = HEAD + "eval('1+1')\n"
        found = C.check_editor(src)
        self.assertTrue(any(p['kind'] == 'forbidden_builtin' for p in found))

    def test_compile_call_is_rejected(self):
        src = HEAD + "compile('pass', '<s>', 'exec')\n"
        found = C.check_editor(src)
        self.assertTrue(any(p['kind'] == 'forbidden_builtin' for p in found))

    def test_dunder_import_call_is_rejected(self):
        src = HEAD + "__import__('os')\n"
        found = C.check_editor(src)
        self.assertTrue(any(p['kind'] == 'forbidden_builtin' for p in found))

    # --- dunder access -----------------------------------------------------

    def test_dunder_attribute_is_rejected(self):
        src = HEAD + "x = s.__class__\n"
        found = C.check_editor(src)
        self.assertTrue(any(p['kind'] == 'dunder_access' for p in found))

    def test_globals_dunder_is_rejected(self):
        src = HEAD + "x = s.__globals__\n"
        found = C.check_editor(src)
        self.assertTrue(any(p['kind'] == 'dunder_access' for p in found))

    def test_subclasses_dunder_is_rejected(self):
        src = HEAD + "x = ().__class__.__subclasses__()\n"
        found = C.check_editor(src)
        self.assertTrue(any(p['kind'] == 'dunder_access' for p in found))

    # --- security findings short-circuit further checks --------------------

    def test_security_error_stops_further_checks(self):
        """If security fails the correctness checks do not run."""
        # This has a bad method AND a forbidden import — only security should appear.
        src = "import os\n" + HEAD + "s.totally_fake_method()\n"
        found = C.check_editor(src)
        kinds = {p['kind'] for p in found}
        self.assertIn('forbidden_import', kinds)
        self.assertNotIn('unknown_method', kinds)


class TestTheSummaryIsHonest(unittest.TestCase):

    def test_a_clean_script_says_what_was_not_checked(self):
        text = C.summary([])
        self.assertIn('names and arguments only', text)
        self.assertIn('not values', text)

    def test_findings_carry_their_line(self):
        self.assertIn('line 3', C.summary(_check("s.add_beam('E1')\n")))


class TestIncrementObjectIdIsChecked(unittest.TestCase):
    """An edge method (support_edge, pin_edge, …) names a geometry object; an
    id that does not exist should be caught before it KeyErrors at apply time,
    including a number written where the string id belongs (object_id=1 for a
    'Laje1')."""

    def _slab(self):
        s = Structure2D(domain='plate')
        s.add_material('Mat', elastic_modulus=3e7, unit_weight=25.0)
        s.add_tri_section('Slab', 'Mat', thickness=0.2)
        s.add_geo_rectangle('Laje1', (0.0, 0.0), (3.0, 5.0),
                            section_name='Slab')
        return s

    def _msgs(self, src, struc):
        return [p['msg'] for p in C.check_increment(src, struc=struc)]

    def test_the_real_id_passes(self):
        self.assertEqual(
            self._msgs("model.support_edge('Laje1', 'right', 'SIMPLE')\n",
                       self._slab()), [])

    def test_a_number_where_the_id_belongs_is_flagged(self):
        msg = self._msgs(
            "model.support_edge(object_id=1, edge='top', kind='CLAMPED')\n",
            self._slab())
        self.assertTrue(msg and "object_id='1'" in msg[0])

    def test_an_unknown_string_id_is_flagged(self):
        msg = self._msgs("model.pin_edge('Laje9', 'left')\n", self._slab())
        self.assertTrue(msg and 'Laje9' in msg[0])


if __name__ == '__main__':
    unittest.main()



class TestIncrementNamesWhatDoesNotExist(unittest.TestCase):
    """An id or a variable an increment cannot have, found before it is applied."""

    def setUp(self):
        from xdfem2d import Structure2D
        self.s = Structure2D()
        self.s.create_rc_section('S1', b=0.3, h=0.5)
        self.s.add_node('N1', 0, 0)
        self.s.add_node('N2', 6, 0)
        self.s.add_bar_element('E1', 'N1', 'N2', 'S1')

    def _kinds(self, code, struc=None):
        return [p['kind'] for p in C.check_increment(
            code, struc=struc or self.s)]

    def test_a_load_on_a_bar_the_model_does_not_have(self):
        code = "model.create_load_case('LC')\nmodel.create_uniform_load('B1', 'LC', 20, 'down')"
        probs = C.check_increment(code, struc=self.s)
        self.assertEqual([p['kind'] for p in probs], ['unknown_id'])
        self.assertIn('E1', probs[0]['msg'])
        self.assertIn('geometry_tool', probs[0]['msg'])

    def test_a_load_on_a_bar_it_has_is_fine(self):
        code = "model.create_load_case('LC')\nmodel.create_uniform_load('E1', 'LC', 20, 'down')"
        self.assertEqual(self._kinds(code), [])
        code = "model.create_load_case('LC')\nmodel.create_bar_distributed_load(['E1'], 'LC', 5, 5)"
        self.assertEqual(self._kinds(code), [])

    def test_a_list_names_the_one_that_is_missing(self):
        code = "model.create_load_case('LC')\nmodel.create_uniform_load(['E1', 'X9'], 'LC', 20, 'down')"
        probs = C.check_increment(code, struc=self.s)
        self.assertEqual(len(probs), 1)
        self.assertIn("'X9'", probs[0]['msg'])

    def test_a_bar_made_in_the_same_increment_has_an_id_that_cannot_be_seen(self):
        code = ("sec = model.create_rc_section('S2', b=0.3, h=0.5)\n"
                "n1 = model.create_node(0, 3)\nn2 = model.create_node(6, 3)\n"
                "model.create_bar_element(sec, n1, n2)\n"
                "model.create_load_case('LC')\n"
                "model.create_uniform_load('B1', 'LC', 20, 'down')")
        self.assertEqual(self._kinds(code), [])

    def test_a_model_with_an_area_is_left_alone(self):
        from xdfem2d import Structure2D
        wall = Structure2D()
        wall.create_polygon([0, 0, 5, 2.5], thickness=0.25, id='W1')
        code = "model.create_load_case('LC')\nmodel.create_uniform_load('T7', 'LC', 5, 'down')"
        self.assertEqual(self._kinds(code, struc=wall), [])

    def test_no_model_open_is_left_alone(self):
        code = "model.create_load_case('LC')\nmodel.create_uniform_load('B1', 'LC', 20, 'down')"
        self.assertNotIn('unknown_id', [p['kind'] for p in
                                        C.check_increment(code, struc=None)
                                        if 'targets' in p['msg']])

    def test_a_variable_from_an_earlier_reply(self):
        code = "model.create_load_case('LC')\nmodel.create_uniform_load(b1, 'LC', 20, 'down')"
        probs = C.check_increment(code, struc=self.s)
        self.assertEqual([p['kind'] for p in probs], ['undefined_name'])
        self.assertIn("'b1'", probs[0]['msg'])
        code = "model.pin([n1, n2])"
        self.assertEqual(self._kinds(code), ['undefined_name', 'undefined_name'])

    def test_a_coefficient_key_that_is_no_load_case(self):
        self.s.add_load_case('G')
        code = "model.create_analysis_case('AC1', coefficients={'Permanent': 1.0})"
        probs = C.check_increment(code, struc=self.s)
        self.assertEqual([p['kind'] for p in probs], ['unknown_id'])
        self.assertIn("'Permanent'", probs[0]['msg'])
        self.assertIn('G', probs[0]['msg'])
        code = "model.add_analysis_case('AC1', 'Linear', {'G': 1.0, 'Q': 1.0})"
        probs = C.check_increment(code, struc=self.s)
        self.assertEqual(len(probs), 1)
        self.assertIn("'Q'", probs[0]['msg'])

    def test_coefficient_keys_that_exist_or_are_created_are_fine(self):
        self.s.add_load_case('G')
        code = "model.create_analysis_case('AC1', coefficients={'G': 1.0})"
        self.assertEqual(self._kinds(code), [])
        code = ("model.create_load_case('Q')\n"
                "model.create_analysis_case('AC2', coefficients={'G': 1.0, 'Q': 1.0})")
        self.assertEqual(self._kinds(code), [])

    def test_coefficient_keys_are_left_alone_when_a_case_is_auto_numbered(self):
        code = ("model.create_load_case()\n"
                "model.create_analysis_case('AC1', coefficients={'LC1': 1.0})")
        self.assertEqual(self._kinds(code), [])

    def test_a_section_name_that_does_not_exist(self):
        # "adiciona material aço S355" came back as section_name = 'S355' on
        # four bars with no such section anywhere, and the check said valid.
        code = "model.bar_elements_by_id['B1'].section_name = 'S355'"
        probs = C.check_increment(code, struc=self.s)
        self.assertEqual([p['kind'] for p in probs], ['unknown_id'])
        self.assertIn("'S355'", probs[0]['msg'])

    def test_a_section_name_that_exists_or_is_created_is_fine(self):
        name = next(iter(self.s.sections))
        code = f"model.bar_elements_by_id['B1'].section_name = {name!r}"
        self.assertEqual(self._kinds(code), [])
        code = ("model.create_steel_section('S2', 'IPE300', grade='S355')\n"
                "model.bar_elements_by_id['B1'].section_name = 'S2'")
        self.assertEqual(self._kinds(code), [])
        code = ("model.bar_elements_by_id['B1'].section_name = name")
        self.assertNotIn('unknown_id', self._kinds(code))

    def test_a_modal_or_sequence_case_is_not_read_as_load_case_weights(self):
        code = "model.create_analysis_case('AC1', 'Sequence', {'AC0': 1.0})"
        self.assertEqual(self._kinds(code), [])

    def test_a_variable_the_increment_defines_is_fine(self):
        code = ("n1 = model.create_node(0, 3)\nn2 = model.create_node(6, 3)\n"
                "model.pin(n1)\nmodel.roller(n2, free='x')\n"
                "for n in (n1, n2):\n    model.create_node_load(n, 'LC', 1, 'down')\n"
                "import math\nmodel.create_node(math.pi, 0)")
        self.assertNotIn('undefined_name', self._kinds(code))


class TestIntegerIndexOnAnIdDict(unittest.TestCase):
    """`model.nodes[0]` is a KeyError: the dict is keyed by id as text."""

    def kinds(self, code, struc=None):
        return [p['kind'] for p in C.check_increment(code, struc=struc)]

    def test_an_integer_index_is_reported(self):
        code = ("model.create_node(0.0, 0.0)\n"
                "model.create_bar_element('S1', model.nodes[0], model.nodes[1])")
        probs = [p for p in C.check_increment(code) if p['kind'] == 'integer_index']
        self.assertEqual(len(probs), 2)
        self.assertIn('keyed by id', probs[0]['msg'])
        self.assertIn('model.nodes[0]', probs[0]['msg'])

    def test_it_holds_for_the_other_id_dicts_and_negative_indexes(self):
        for attr in ('bar_elements_by_id', 'sections', 'materials',
                     'tri_elements_by_id'):
            self.assertIn('integer_index',
                          self.kinds(f"x = model.{attr}[0]"), attr)
        self.assertIn('integer_index', self.kinds("x = model.nodes[-1]"))

    def test_an_id_as_text_is_fine(self):
        self.assertNotIn('integer_index', self.kinds("x = model.nodes['N1']"))

    def test_a_variable_may_hold_an_id(self):
        self.assertNotIn('integer_index',
                         self.kinds("i = 'N1'\nx = model.nodes[i]"))

    def test_a_list_attribute_is_not_an_id_dict(self):
        # load_cases is a list, so [0] is fine
        self.assertNotIn('integer_index', self.kinds("x = model.load_cases[0]"))


class TestCreateElementSectionMustExist(unittest.TestCase):

    def setUp(self):
        from xdfem2d import Structure2D
        self.s = Structure2D()
        self.s.create_rc_section('S1', b=0.3, h=0.5)
        self.n = "n1 = model.create_node(0, 0)\nn2 = model.create_node(7, 0)\n"

    def kinds(self, code):
        return [p['kind'] for p in C.check_increment(code, struc=self.s)]

    def test_a_section_that_is_nowhere(self):
        probs = C.check_increment(
            self.n + "model.create_bar_element('Sec', n1, n2)", struc=self.s)
        self.assertEqual([p['kind'] for p in probs], ['unknown_id'])
        self.assertIn("'Sec'", probs[0]['msg'])

    def test_the_section_as_a_keyword(self):
        self.assertEqual(self.kinds(
            self.n + "model.create_bar_element(section='Sec', node_i=n1, node_j=n2)"),
            ['unknown_id'])

    def test_a_section_in_the_model_or_created_here_is_fine(self):
        self.assertEqual(self.kinds(
            self.n + "model.create_bar_element('S1', n1, n2)"), [])
        self.assertEqual(self.kinds(
            self.n + "sec = model.create_steel_section('S2', 'IPE300')\n"
            "model.create_bar_element('S2', n1, n2)"), [])

    def test_a_section_object_is_not_checked(self):
        self.assertEqual(self.kinds(
            self.n + "sec = model.create_rc_section('S3', b=0.3, h=0.5)\n"
            "model.create_bar_element(sec, n1, n2)"), [])
