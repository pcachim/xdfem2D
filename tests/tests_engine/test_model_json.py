"""A model as JSON: the template, and what "complete" means.

This exists to test one idea. Five language models asked to build a beam
produced four that could not be solved, and never because a name was wrong —
always because a step was missing. Four of five defined supports and assigned
them to nothing.

The idea is that a template makes the missing step visible: support_assignments
is a key you can see sitting empty, where assign_support is a call nothing
reminds you to make. Whether that actually helps is a question for the
benchmark; what is tested here is that the piece is sound enough to measure
with.

Two properties carry it. The template must load and solve exactly as it is —
a template that does not work is one nobody trusts — and missing_pieces must
find the absence without inventing correctness it cannot judge.
"""
from __future__ import annotations

import json
import unittest

from context import Structure2D  # noqa: F401  (puts src/ on the path)

from xdfem2d import model_json as M


class TestTheTemplateWorksAsGiven(unittest.TestCase):

    def test_it_is_complete_by_its_own_check(self):
        self.assertEqual(M.missing_pieces(M.template(as_text=False)), [])

    def test_it_loads_and_solves(self):
        """The property that makes it worth handing to anyone. Checked by
        solving, not by inspection: every failed script of the five passed
        inspection of one kind or another."""
        struc = M.load(M.template(as_text=False))
        results = struc.calculate()
        total = sum(v[1] for v in
                    results['analysis_cases']['G']['reactions'].values())
        # 20 kN/m over 6 m, plus self-weight (self_weight_factor 1.0):
        # 0.3 x 0.5 m section, C30/37 (unit_weight 25 kN/m3), over 6 m
        # = 22.5 kN. 120.0 + 22.5 = 142.5.
        self.assertAlmostEqual(total, 20.0 * 6.0 + 0.3 * 0.5 * 25.0 * 6.0,
                               places=6)

    def test_the_text_form_is_valid_json(self):
        self.assertEqual(json.loads(M.template()), M.BEAM)

    def test_editing_it_gives_a_different_beam_that_still_solves(self):
        """What a model is being asked to do with it."""
        d = M.template(as_text=False)
        d['nodes'][1]['x'] = 10.0
        d['distributed_loads'][0]['fye'] = -30.0
        d['distributed_loads'][0]['fyd'] = -30.0
        self.assertEqual(M.missing_pieces(d), [])
        r = M.load(d).calculate()
        total = sum(v[1] for v in
                    r['analysis_cases']['G']['reactions'].values())
        # Same section, now 10 m long: self-weight 0.3 x 0.5 x 25.0 x 10 = 37.5 kN.
        self.assertAlmostEqual(total, 30.0 * 10.0 + 0.3 * 0.5 * 25.0 * 10.0,
                               places=6)

    def test_a_returned_copy_cannot_corrupt_the_template(self):
        d = M.template(as_text=False)
        d['nodes'].clear()
        self.assertTrue(M.BEAM['nodes'])


class TestTheMistakeItExistsFor(unittest.TestCase):
    """Supports defined, assigned to nothing. Four of five replies."""

    def test_it_is_reported(self):
        d = M.template(as_text=False)
        d['support_assignments'] = []
        found = ' '.join(M.missing_pieces(d))
        self.assertIn('support_assignments', found)

    def test_the_message_explains_the_pair(self):
        d = M.template(as_text=False)
        d['support_assignments'] = []
        msg = next(m for m in M.missing_pieces(d) if 'floats' in m)
        self.assertIn('node id', msg)

    def test_the_key_is_in_the_template_at_all(self):
        """The whole hypothesis in one assertion: it can only be filled in if
        it is visible."""
        self.assertIn('support_assignments', M.template())


class TestDanglingReferences(unittest.TestCase):

    def test_a_section_that_does_not_exist(self):
        d = M.template(as_text=False)
        d['bar_elements'][0]['section_name'] = 'NotThere'
        self.assertTrue(any('NotThere' in m for m in M.missing_pieces(d)))

    def test_a_load_case_that_was_never_created(self):
        """The other repeated failure: a load pointing at nothing."""
        d = M.template(as_text=False)
        d['load_cases'] = []
        found = ' '.join(M.missing_pieces(d))
        self.assertIn('load_cases', found)

    def test_a_bar_between_nodes_that_are_not_there(self):
        d = M.template(as_text=False)
        d['bar_elements'][0]['node_j'] = 'N99'
        self.assertTrue(any('N99' in m for m in M.missing_pieces(d)))

    def test_a_support_assigned_to_a_name_never_defined(self):
        d = M.template(as_text=False)
        d['support_assignments'][0]['support_name'] = 'GHOST'
        self.assertTrue(any('GHOST' in m for m in M.missing_pieces(d)))


class TestWhatItRefusesToJudge(unittest.TestCase):
    """The honest half, and the reason this is not called validation.

    Everything below is a model that is complete and wrong. If any of these
    ever starts being reported, the function has taken on engineering
    judgement it does not have, and a clean result would start being read as
    "the model is right".
    """

    def test_a_load_pointing_upwards_passes(self):
        d = M.template(as_text=False)
        d['distributed_loads'][0]['fye'] = 20.0
        d['distributed_loads'][0]['fyd'] = 20.0
        self.assertEqual(M.missing_pieces(d), [])

    def test_a_support_that_restrains_nothing_passes(self):
        d = M.template(as_text=False)
        for s in d['supports']:
            s['ux'] = s['uy'] = s['tz'] = False
        self.assertEqual(M.missing_pieces(d), [])

    def test_an_absurd_section_passes(self):
        d = M.template(as_text=False)
        d['sections'][0]['h'] = 0.001
        self.assertEqual(M.missing_pieces(d), [])


class TestItRunsNothing(unittest.TestCase):
    """The property that lets a model reply go straight into the application,
    where a script may not."""

    def test_the_module_never_executes(self):
        from pathlib import Path
        src = (Path(__file__).resolve().parent.parent.parent / 'src' / 'xdfem2d'
               / 'model_json.py').read_text(encoding='utf-8')
        for forbidden in ('exec(', 'eval(', 'compile(', 'subprocess',
                          '__import__'):
            self.assertNotIn(forbidden, src)

    def test_a_payload_with_side_effects_is_only_parsed(self):
        marker = '/tmp/xdfem2d_model_json_must_not_exist'
        from pathlib import Path
        if Path(marker).exists():
            Path(marker).unlink()
        M.missing_pieces({'nodes': [{'id': f"__import__('os').system('touch "
                                          f"{marker}')", 'x': 0, 'y': 0}]})
        self.assertFalse(Path(marker).exists())


if __name__ == '__main__':
    unittest.main()


class TestWhatModelsActuallyWrite(unittest.TestCase):
    """JSON as it arrives, not as the specification has it.

    granite produced a model with `/* Load case ... */` in the middle. JSON has
    no comments, json.loads rejected the whole thing with a character offset,
    and the user saw nothing work. The intent was unambiguous; refusing it
    served nobody.

    Tolerant on the way in, strict on the way out — what we write is always
    plain JSON.
    """

    def test_a_block_comment_is_tolerated(self):
        text = ('{\n  /* the load case */\n  "nodes": '
                '[{"id": "N1", "x": 0, "y": 0}]\n}')
        self.assertEqual(M.parse(text)['nodes'][0]['id'], 'N1')

    def test_a_line_comment_is_tolerated(self):
        text = '{\n  // nodes\n  "nodes": [{"id": "N1", "x": 0, "y": 0}]\n}'
        self.assertEqual(len(M.parse(text)['nodes']), 1)

    def test_a_trailing_comma_is_tolerated(self):
        self.assertEqual(M.parse('{"nodes": [1, 2,],}')['nodes'], [1, 2])

    def test_a_comment_marker_inside_a_string_survives(self):
        """An id is user data. Stripping // out of one would silently rename
        it, which is worse than refusing the file."""
        self.assertEqual(M.parse('{"nodes": [{"id": "N//1"}]}')
                         ['nodes'][0]['id'], 'N//1')

    def test_valid_json_takes_the_fast_path_unchanged(self):
        self.assertEqual(M.parse(M.template()), M.BEAM)

    def test_something_that_is_not_json_at_all_still_raises(self):
        with self.assertRaises(ValueError):
            M.parse('this is a sentence')


class TestOmittedFieldsMeanTheirDefault(unittest.TestCase):
    """A restraint left out means free, which is what add_support already says.

    Requiring all three turned an omitted 'tz' into `KeyError: 'tz'` — raised
    from inside the loader, naming a key, about a value nobody had to give.
    """

    def test_a_support_without_tz(self):
        d = M.template(as_text=False)
        for s in d['supports']:
            s.pop('tz', None)
        struc = M.load(d)
        self.assertFalse(struc.supports['PIN'].tz)

    def test_a_support_with_only_the_restraint_that_matters(self):
        d = M.template(as_text=False)
        d['supports'] = [{'name': 'PIN', 'ux': True, 'uy': True},
                         {'name': 'ROLLER_X', 'uy': True}]
        struc = M.load(d)
        self.assertTrue(struc.supports['PIN'].ux)
        self.assertFalse(struc.supports['ROLLER_X'].ux)
        struc.calculate()

    def test_the_action_type_may_be_a_letter_or_a_word(self):
        for value in ('G', 'Permanent', 'G – Permanent'):
            with self.subTest(action_type=value):
                d = M.template(as_text=False)
                d['load_cases'][0]['action_type'] = value
                self.assertEqual(M.missing_pieces(d), [])
                M.load(d).calculate()


class TestTheGraniteModelAsItArrived(unittest.TestCase):
    """The whole chain on one real reply: comments, a missing tz, and an
    action type the enum used to refuse. It now loads and solves — and it is
    still the wrong structure, which no amount of tolerance can fix."""

    REPLY = '''{
  "nodes": [
    { "id": "N0", "x": 0, "y": 0 }, { "id": "N1", "x": 6, "y": 0 },
    { "id": "N2", "x": 12, "y": 0 }, { "id": "N3", "x": 18, "y": 0 }
  ],
  "materials": [{ "name": "C30", "elastic_modulus": 33000000, "unit_weight": 25 }],
  "sections": [{ "name": "RECT", "material_name": "C30", "b": 0.24, "h": 0.24 }],
  "bar_elements": [
    { "id": "E1", "node_i": "N0", "node_j": "N1", "section_name": "RECT" },
    { "id": "E2", "node_i": "N1", "node_j": "N2", "section_name": "RECT" },
    { "id": "E3", "node_i": "N2", "node_j": "N3", "section_name": "RECT" }
  ],
  "supports": [
    { "name": "FIXED", "ux": true, "uy": true },
    { "name": "PIN", "ux": false, "uy": true }
  ],
  "support_assignments": [
    { "node_id": "N0", "support_name": "FIXED" },
    { "node_id": "N3", "support_name": "PIN" }
  ],
  /* Load case - the uniform load is entered per element */
  "load_cases": [{ "id": "G", "self_weight_factor": 0, "action_type": "Permanent" }],
  "distributed_loads": [
    { "element_id": "E1", "load_case_id": "G", "fye": -10, "fyd": -10 },
    { "element_id": "E2", "load_case_id": "G", "fye": -10, "fyd": -10 },
    { "element_id": "E3", "load_case_id": "G", "fye": -10, "fyd": -10 }
  ],
  "analysis_cases": [{ "id": "ULS", "analysis_type": "Linear",
                       "coefficients": { "G": 1 } }]
}'''

    def test_it_parses_loads_and_solves(self):
        data = M.parse(self.REPLY)
        self.assertEqual(M.missing_pieces(data), [])
        struc = M.load(data)
        results = struc.calculate()
        total = sum(v[1] for v in
                    results['analysis_cases']['ULS']['reactions'].values())
        self.assertAlmostEqual(total, 10.0 * 18.0, places=6)

    def test_and_it_is_the_wrong_structure(self):
        """Asked for a three-span continuous beam. Two of the four nodes carry
        no support, so it is an 18 m simply supported beam — complete, valid,
        and not what was requested. Nothing static can know that."""
        struc = M.load(M.parse(self.REPLY))
        supported = {a.node_id for a in struc.support_assignments}
        self.assertEqual(sorted(set(struc.nodes) - supported), ['N1', 'N2'])


class TestTheWallTemplate(unittest.TestCase):
    """A wall is not a beam with different numbers in it.

    The first version of this module had one template and one rule, both
    written around bars: 'sections' and 'bar_elements' were required outright.
    A wall has neither — it is triangles with a thickness — so a wall that
    solved was reported incomplete, and the button offering to open it never
    appeared. The same beam-shaped assumption that produced six beam fixtures
    and no wall.
    """

    def test_it_is_complete_by_its_own_check(self):
        self.assertEqual(M.missing_pieces(M.template('wall', as_text=False)),
                         [])

    def test_it_solves_and_the_base_holds_the_load(self):
        """100 kN pushes the top sideways; the base must push back by 100."""
        struc = M.load(M.template('wall', as_text=False))
        results = struc.calculate()
        rx = sum(v[0] for v in results['reactions']['G'].values())
        self.assertAlmostEqual(rx, -100.0, places=6)

    def test_it_is_made_of_triangles_and_says_so(self):
        d = M.template('wall', as_text=False)
        self.assertNotIn('sections', d)
        self.assertNotIn('bar_elements', d)
        self.assertEqual(len(d['tri_elements']), 4)
        self.assertIn('thickness', d['tri_sections'][0])

    def test_the_kinds_are_named_and_differ(self):
        self.assertEqual(sorted(M.TEMPLATES),
                         ['beam', 'grillage', 'slab', 'wall'])
        self.assertNotEqual(M.template('wall'), M.template('beam'))

    def test_the_plate_templates_are_plate_and_solve(self):
        """slab and grillage are plate-domain models that must load and solve —
        and, unlike a 4-corner panel, actually deflect somewhere."""
        for kind in ('slab', 'grillage'):
            d = M.template(kind, as_text=False)
            self.assertEqual(d['domain'], 'plate')
            struc = M.load(d)
            self.assertEqual(struc.domain, 'plate')
            res = struc.calculate()
            case = next(iter(res['analysis_cases']))
            disp = res['analysis_cases'][case]['displacements']
            wmax = max(abs(v[0]) for v in disp.values())
            self.assertGreater(wmax, 0.0)
        # A slab's load is a pressure, not a bar's distributed_loads.
        self.assertIn('tri_area_loads', M.template('slab', as_text=False))
        self.assertIn('formulation',
                      M.template('slab', as_text=False)['tri_sections'][0])

    def test_an_unknown_kind_falls_back_rather_than_raising(self):
        """Called with whatever a language model wrote in a tool argument. A
        KeyError there costs the user the answer over a word."""
        self.assertEqual(M.template('WALL', as_text=False),
                         M.template('wall', as_text=False))
        self.assertEqual(M.template('nonsense', as_text=False),
                         M.template('beam', as_text=False))


class TestSectionNameMeansTwoThings(unittest.TestCase):
    """'section_name' is a bar section in bar_elements and a wall section in
    tri_elements. Resolving it against one global set of names reported every
    valid wall as referring to a section that does not exist."""

    def test_a_wall_section_is_looked_up_among_wall_sections(self):
        self.assertEqual(M.missing_pieces(M.template('wall', as_text=False)),
                         [])

    def test_a_wall_pointing_at_a_bar_section_is_still_caught(self):
        d = M.template('wall', as_text=False)
        d['sections'] = [{'name': 'S1', 'material_name': 'C30/37',
                          'b': 0.3, 'h': 0.5}]
        d['tri_elements'][0]['section_name'] = 'S1'   # a bar section
        self.assertEqual(M.missing_pieces(d),
                         ["tri_elements: section_name='S1' is not in "
                          "tri_sections"])

    def test_a_bar_pointing_at_a_wall_section_is_caught_too(self):
        d = M.template('beam', as_text=False)
        d['tri_sections'] = [{'name': 'W1', 'material_name': 'C30/37',
                              'thickness': 0.2}]
        d['bar_elements'][0]['section_name'] = 'W1'
        self.assertEqual(M.missing_pieces(d),
                         ["bar_elements: section_name='W1' is not in sections"])


class TestGeometryObjectsCountAsSomethingToAnalyse(unittest.TestCase):
    """A region with a target size, meshed into triangles when the model is
    solved. It is how a wall is actually written — the explicit triangles in
    the template are there to show what one needs, not to be typed out."""

    def _geo(self):
        """The real shape, which is not the obvious one: a rectangle names four
        nodes that must exist, and carries no coordinates of its own."""
        d = M.template('wall', as_text=False)
        del d['tri_elements']
        d['nodes'] = [
            {'id': 'R1.p0', 'x': 0.0, 'y': 0.0},
            {'id': 'R1.p1', 'x': 4.0, 'y': 0.0},
            {'id': 'R1.p2', 'x': 4.0, 'y': 3.0},
            {'id': 'R1.p3', 'x': 0.0, 'y': 3.0},
        ]
        d['geometry_objects'] = [
            {'kind': 'rectangle', 'id': 'R1', 'tri_section_name': 'W1',
             'target_size': 1.0,
             'node_ids': ['R1.p0', 'R1.p1', 'R1.p2', 'R1.p3']},
        ]
        d['support_assignments'] = [
            {'node_id': 'R1.p0', 'support_name': 'PIN'},
            {'node_id': 'R1.p1', 'support_name': 'PIN'},
        ]
        d['point_loads'] = [
            {'node_id': 'R1.p3', 'load_case_id': 'G', 'fx': 100.0, 'fy': 0.0,
             'mz': 0.0},
        ]
        return d

    def test_a_model_of_regions_is_complete(self):
        self.assertEqual(M.missing_pieces(self._geo()), [])

    def test_it_meshes_and_solves(self):
        """Complete has to mean something: the region must become elements."""
        from xdfem2d.geo_expand import expand_geometry
        mesh, _ = expand_geometry(M.load(self._geo()))
        self.assertGreater(len(mesh.tri_elements), 0)
        results = mesh.calculate()
        rx = sum(v[0] for v in results['reactions']['G'].values())
        self.assertAlmostEqual(rx, -100.0, places=6)

    def test_a_region_with_no_corners_is_caught(self):
        """The trap, and the one I walked into: writing the corners inline as
        p0/p1 reads perfectly, passes every reference check — there are no
        references — and meshes into nothing at all."""
        d = self._geo()
        d['geometry_objects'] = [
            {'id': 'R1', 'kind': 'rectangle', 'tri_section_name': 'W1',
             'target_size': 1.0, 'p0': [0.0, 0.0], 'p1': [4.0, 3.0]},
        ]
        said = M.missing_pieces(d)
        self.assertTrue(any("has no 'node_ids'" in s for s in said), said)

    def test_a_rectangle_with_the_wrong_number_of_corners_is_caught(self):
        d = self._geo()
        d['geometry_objects'][0]['node_ids'] = ['R1.p0', 'R1.p1', 'R1.p2']
        self.assertIn("geometry_objects: rectangle 'R1' is defined by 3 "
                      "node(s) and needs at least 4 — as it stands it produces "
                      "no elements", M.missing_pieces(d))

    def test_a_region_naming_no_section_is_caught(self):
        d = self._geo()
        d['geometry_objects'][0]['tri_section_name'] = 'FANTASMA'
        self.assertIn("geometry_objects: tri_section_name='FANTASMA' is not in "
                      "tri_sections", M.missing_pieces(d))

    def test_nothing_to_analyse_at_all_is_reported_once(self):
        d = M.template('wall', as_text=False)
        del d['tri_elements']
        said = M.missing_pieces(d)
        self.assertEqual(len(said), 1)
        self.assertIn('bar_elements', said[0])
        self.assertIn('geometry_objects', said[0])


class TestAFieldThatIsSimplyAbsent(unittest.TestCase):
    """Dangling was checked; missing was not.

    The reference check only looked at fields that were present, so a
    section_name pointing at nothing was caught and a tri_element with no
    section_name at all was called complete. granite wrote four of those. The
    model was declared fine and the loader died with KeyError 'material_name',
    which tells a user nothing about what to fix.

    The same hole had been closed the day before for a geometry object with no
    node_ids — and closed there only, as a special case, instead of as the
    class it belongs to. That is the mistake worth not repeating.
    """

    def test_a_triangle_with_no_section_is_caught(self):
        d = M.template('wall', as_text=False)
        for t in d['tri_elements']:
            del t['section_name']
        said = M.missing_pieces(d)
        self.assertEqual(len(said), 4)
        self.assertIn("'T1' has no 'section_name'", said[0])

    def test_a_bar_with_no_section_is_caught(self):
        d = M.template('beam', as_text=False)
        del d['bar_elements'][0]['section_name']
        self.assertIn("bar_elements: 'E1' has no 'section_name' — it must name "
                      "one of sections", M.missing_pieces(d))

    def test_a_section_with_no_material_is_caught(self):
        d = M.template('beam', as_text=False)
        del d['sections'][0]['material_name']
        self.assertTrue(any('material' in s for s in M.missing_pieces(d)))

    def test_a_load_with_no_case_is_caught(self):
        d = M.template('beam', as_text=False)
        del d['distributed_loads'][0]['load_case_id']
        self.assertTrue(any('load_case' in s for s in M.missing_pieces(d)))

    def test_the_model_that_found_it_does_not_pass(self):
        """granite's wall, reduced to the part that mattered."""
        d = M.template('wall', as_text=False)
        for t in d['tri_elements']:
            t.pop('section_name', None)
        self.assertNotEqual(M.missing_pieces(d), [])
        with self.assertRaises(Exception):
            M.load(d).calculate()


class TestGeometryObjectsAreTwoFamilies(unittest.TestCase):
    """A line, arc or polyline becomes bars and carries 'section_name'. A
    rectangle or surface becomes triangles and carries 'tri_section_name'.

    Written first as though they were all regions, which told a shipped
    example that its arc had no wall section. An arc has no wall section.
    """

    def _obj(self, **kw):
        d = M.template('wall', as_text=False)
        del d['tri_elements']
        d['sections'] = [{'name': 'S1', 'material_name': 'C30/37',
                          'b': 0.3, 'h': 0.5}]
        d['nodes'] = [{'id': f'P{i}', 'x': float(i), 'y': 0.0}
                      for i in range(4)]
        d['support_assignments'] = [{'node_id': 'P0', 'support_name': 'PIN'}]
        d['point_loads'] = [{'node_id': 'P1', 'load_case_id': 'G',
                             'fx': 1.0, 'fy': 0.0, 'mz': 0.0}]
        obj = {'id': 'G1', 'node_ids': ['P0', 'P1', 'P2', 'P3']}
        obj.update(kw)
        d['geometry_objects'] = [obj]
        return d

    def test_an_arc_names_a_bar_section(self):
        d = self._obj(kind='arc', section_name='S1',
                      node_ids=['P0', 'P1', 'P2'])
        self.assertEqual(M.missing_pieces(d), [])

    def test_an_arc_is_not_asked_for_a_wall_section(self):
        said = M.missing_pieces(self._obj(kind='arc', section_name='S1',
                                          node_ids=['P0', 'P1', 'P2']))
        self.assertFalse(any('tri_section' in s for s in said), said)

    def test_a_rectangle_names_a_wall_section(self):
        self.assertEqual(
            M.missing_pieces(self._obj(kind='rectangle',
                                       tri_section_name='W1')), [])

    def test_a_rectangle_naming_a_bar_section_is_caught(self):
        said = M.missing_pieces(self._obj(kind='rectangle',
                                          tri_section_name='S1'))
        self.assertTrue(any('is not in tri_sections' in s for s in said), said)

    def test_too_few_defining_nodes_is_caught(self):
        """Two opposite corners read like a rectangle and mesh into nothing."""
        said = M.missing_pieces(self._obj(kind='rectangle',
                                          tri_section_name='W1',
                                          node_ids=['P0', 'P2']))
        self.assertTrue(any('produces no elements' in s for s in said), said)

    def test_an_unknown_kind_is_named_with_the_real_ones(self):
        said = M.missing_pieces(self._obj(kind='blob', tri_section_name='W1'))
        self.assertTrue(any('rectangle' in s and 'blob' in s for s in said),
                        said)


class TestItDoesNotShoutAtTheRealExamples(unittest.TestCase):
    """Every rule here is one I wrote from a failure, and the way to find out
    whether it overreaches is to point it at the files that ship. Two of them
    complained the first time: one was a false positive (the arc), one was a
    real defect (an object producing no elements)."""

    def test_the_shipped_examples_are_complete(self):
        import glob
        from pathlib import Path

        from xdfem2d.file_io import load_x2d
        from xdfem2d.structure_io import _to_dict

        root = Path(__file__).resolve().parent.parent.parent
        files = sorted(glob.glob(str(root / 'examples' / '*.x2d')))
        self.assertTrue(files, 'no examples found — the check is vacuous')
        known_bad = {'example-frame_and_wall.x2d'}   # see the test below
        for path in files:
            name = Path(path).name
            if name in known_bad:
                continue
            with self.subTest(example=name):
                struc = load_x2d(path)
                struc = struc[0] if isinstance(struc, tuple) else struc
                self.assertEqual(M.missing_pieces(_to_dict(struc)), [])

    def test_the_one_it_complains_about_really_is_broken(self):
        """A rule that fires on a shipped file is either wrong or right, and
        the only way to know is to check. This one is right: the object it
        names contributes nothing — the mesh is identical without it."""
        from pathlib import Path

        from xdfem2d.file_io import load_x2d
        from xdfem2d.geo_expand import expand_geometry

        root = Path(__file__).resolve().parent.parent.parent
        struc = load_x2d(str(root / 'examples' / 'example-frame_and_wall.x2d'))
        struc = struc[0] if isinstance(struc, tuple) else struc
        before, _ = expand_geometry(struc)
        del struc.geometry_objects['Wall']
        after, _ = expand_geometry(struc)
        self.assertEqual(len(before.tri_elements), len(after.tri_elements))
