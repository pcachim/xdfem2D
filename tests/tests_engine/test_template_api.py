"""create_from_template: a standard structure in one call, on an empty model.

The assistant's small models write valid calls and get the problem wrong (three
nodes for three spans, a running total that does not add up, a support left
out). The template builders do that arithmetic, and this call is how a script
reaches them. These tests keep the call honest: every kind builds and solves,
the numbers come out as stated, a mistake is reported with what the call takes,
and the increment checker reads the rest of a reply against the model the
template makes.
"""
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)
from xdfem2d import script_check as C
from xdfem2d import template_api as TA


def _solved(struc):
    """Results of the first analysis case, expanding geometry objects."""
    if getattr(struc, 'geometry_objects', None):
        from xdfem2d.geo_expand import expand_geometry
        struc = expand_geometry(struc)[0]
    return struc.calculate()


class TestEveryKindBuilds(unittest.TestCase):
    def test_each_kind_builds_with_its_defaults_and_solves(self):
        for kind in TA.template_kinds():
            with self.subTest(kind=kind):
                s = Structure2D()
                self.assertIs(s.create_from_template(kind), s)
                self.assertTrue(s.nodes)
                self.assertIn('SW', s.load_cases_by_id)
                self.assertIn('displacements', _solved(s))

    def test_a_linear_kind_is_bars_and_an_area_kind_is_one_object(self):
        beam = Structure2D(); beam.create_from_template('portal_frame')
        self.assertEqual(len(beam.bar_elements), 3)
        self.assertFalse(beam.geometry_objects)
        wall = Structure2D(); wall.create_from_template('wall')
        self.assertEqual(list(wall.geometry_objects), ['R'])
        self.assertFalse(wall.bar_elements)

    def test_a_slab_is_a_plate_model_and_a_wall_is_not(self):
        slab = Structure2D(); slab.create_from_template('slab')
        wall = Structure2D(); wall.create_from_template('wall')
        self.assertEqual(slab.domain, 'plate')
        self.assertNotEqual(wall.domain, 'plate')


class TestTheNumbersComeOutAsStated(unittest.TestCase):
    def test_spans_are_added_up(self):
        s = Structure2D()
        s.create_from_template('continuous_beam', spans=[4, 5, 6])
        self.assertEqual([n.x for n in s.nodes.values()], [0.0, 4.0, 9.0, 15.0])
        self.assertEqual(len(s.bar_elements), 3)

    def test_a_count_and_a_common_span(self):
        s = Structure2D()
        s.create_from_template('continuous_beam', spans=3, span=6)
        self.assertEqual([n.x for n in s.nodes.values()], [0.0, 6.0, 12.0, 18.0])

    def test_the_span_list_may_come_as_span(self):
        s = Structure2D()
        s.create_from_template('continuous_beam', spans=3, span=[4, 5, 6])
        self.assertEqual(len(s.nodes), 4)

    def test_every_node_of_a_continuous_beam_is_supported_by_default(self):
        s = Structure2D()
        s.create_from_template('continuous_beam', spans=[4, 5, 6])
        self.assertEqual({a.node_id for a in s.support_assignments},
                         set(s.nodes))

    def test_end_supports_can_be_chosen(self):
        s = Structure2D()
        s.create_from_template('continuous_beam', spans=[4, 5],
                               left='fixed', intermediate='free', right='free')
        self.assertEqual({a.node_id for a in s.support_assignments}, {'N0'})

    def test_the_section_is_the_one_asked_for(self):
        s = Structure2D()
        s.create_from_template('beam', span=6, b=0.25, h=0.6, material='C25/30')
        sec = s.sections['Sec']
        self.assertEqual((sec.b, sec.h), (0.25, 0.6))
        mat = s.materials[sec.material_name]
        self.assertEqual(mat.design['class_conc'], 'C25/30')

    def test_a_steel_profile_replaces_the_rectangle(self):
        s = Structure2D()
        s.create_from_template('portal_frame', span=8, height=5,
                               profile='IPE300', material='S355')
        self.assertEqual({sec.profile_name for sec in s.sections.values()},
                         {'IPE300'})
        self.assertEqual(len(s.materials), 1)
        self.assertIn('displacements', _solved(s))

    def test_a_slab_pressure_is_its_own_case_and_none_means_none(self):
        s = Structure2D()
        s.create_from_template('slab', Lx=5, Ly=4, pressure=5)
        self.assertIn('Q', s.load_cases_by_id)
        self.assertEqual(s.surface_area_loads[0].pz, -5.0)
        t = Structure2D()
        t.create_from_template('slab', Lx=5, Ly=4)
        self.assertNotIn('Q', t.load_cases_by_id)
        self.assertFalse(t.surface_area_loads)

    def test_slab_edges_per_side(self):
        s = Structure2D()
        s.create_from_template('slab', edges={'bottom': 'clamped', 'top': 'clamped'})
        sup = s.geometry_objects['R'].edge_supports
        self.assertEqual(sup, ['CLAMPED', None, 'CLAMPED', None])

    def test_a_wall_is_fixed_along_its_base(self):
        s = Structure2D()
        s.create_from_template('wall', width=5, height=2.5, thickness=0.25)
        s.add_load_case('LC')
        s.add_point_load('R.p2', 'LC', fx=80.0)
        results = _solved(s)
        total = abs(sum(v[0] for v in results['reactions']['LC'].values()))
        self.assertAlmostEqual(total, 80.0, places=3)

    def test_a_known_alias_names_the_kind(self):
        s = Structure2D()
        s.create_from_template('Warren truss', panels=4)
        self.assertTrue(s.bar_elements)
        s = Structure2D()
        s.create_from_template('portal')
        self.assertEqual(len(s.bar_elements), 3)


class TestTheKindsAddedLater(unittest.TestCase):
    """The deep beam, the frame with a wall, the special slabs and the grillages
    are the application's own templates; each is one call here."""

    def _built(self, kind, **params):
        s = Structure2D()
        s.create_from_template(kind, **params)
        return s

    def test_the_domain_is_the_one_of_the_template(self):
        for kind, domain in (('wall_beam', 'plane'), ('wall_frame', 'plane'),
                             ('slab_ribbed', 'plate'), ('flat_slab', 'plate'),
                             ('slab_on_grade', 'plate'), ('slab_sector', 'plate'),
                             ('beam_grid', 'plate'), ('bridge_grillage', 'plate')):
            with self.subTest(kind=kind):
                self.assertEqual(self._built(kind).domain, domain)

    def test_a_deep_beam_is_pinned_on_the_left_and_rolling_on_the_right(self):
        s = self._built('wall_beam', width=8, height=4)
        self.assertEqual(list(s.geometry_objects), ['R'])
        got = {a.node_id: a.support_name for a in s.support_assignments}
        self.assertEqual(got, {'R.p0': 'Pin', 'R.p1': 'Roller'})

    def test_a_frame_with_a_wall_has_both_and_fixes_every_base(self):
        s = self._built('wall_frame', bays=2, floors=3, bay_width=[5, 4],
                        floor_height=3, wall_width=3, wall_side='right')
        self.assertTrue(s.geometry_objects)
        self.assertGreater(len(s.nodes), 0)
        self.assertEqual({a.support_name for a in s.support_assignments}, {'Fixed'})
        self.assertGreaterEqual(len(s.support_assignments), 3)   # 3 column bases

    def test_the_wall_of_the_frame_can_be_on_either_side(self):
        left = self._built('wall_frame', wall_side='left')
        right = self._built('wall_frame', wall_side='right')
        self.assertLess(min(n.x for n in left.nodes.values()), 0.0)
        self.assertGreater(max(n.x for n in right.nodes.values()), 5.0)

    def test_a_flat_slab_is_carried_on_its_columns_only(self):
        s = self._built('flat_slab', Lx=6, Ly=6, bays_x=3, bays_y=2)
        self.assertEqual(len(s.support_assignments), (3 + 1) * (2 + 1))
        self.assertEqual(len(s.geometry_objects), 3 * 2)

    def test_a_flat_slab_takes_a_list_of_spans(self):
        s = self._built('flat_slab', Lx=[5, 6, 7], Ly=6, bays_y=1)
        self.assertEqual(len(s.geometry_objects), 3)

    def test_a_slab_on_grade_has_springs_and_no_supports(self):
        s = self._built('slab_on_grade', kz=20000)
        self.assertFalse(s.support_assignments)
        self.assertEqual(s.surface_area_springs[0].kz, 20000)

    def test_a_sector_takes_its_supports_per_edge(self):
        s = self._built('slab_sector', rmax=4, angle=180, outer='fixed',
                        radial='pin')
        self.assertEqual(s.domain, 'plate')
        # the outer arc fixed (CLAMPED), the radial sides pinned (SIMPLE)
        self.assertEqual({a.support_name for a in s.support_assignments},
                         {'CLAMPED', 'SIMPLE'})

    def test_a_grid_and_a_bridge_deck_count_what_they_are_asked(self):
        g = self._built('beam_grid', Lx=6, Ly=4, bays_x=3, bays_y=2)
        self.assertEqual(len(g.nodes), (3 + 1) * (2 + 1))
        b = self._built('bridge_grillage', girders=3, stations=4)
        self.assertEqual(len(b.nodes), (4 + 1) * 3)

    def test_the_concrete_class_reaches_the_design(self):
        s = self._built('slab_ribbed', material='C25/30')
        mat = next(iter(s.materials.values()))
        self.assertEqual(mat.design.get('class_conc'), 'C25/30')

    def test_a_pressure_is_a_downward_load_case(self):
        for kind in ('slab_ribbed', 'flat_slab', 'slab_on_grade', 'slab_sector'):
            with self.subTest(kind=kind):
                s = self._built(kind, pressure=10)
                self.assertGreater(len(s.load_cases_by_id), 1)

    def test_every_one_solves(self):
        for kind in ('wall_beam', 'wall_frame', 'slab_ribbed', 'flat_slab',
                     'slab_on_grade', 'slab_sector', 'beam_grid',
                     'bridge_grillage'):
            with self.subTest(kind=kind):
                self.assertIn('displacements', _solved(self._built(kind)))

    def test_the_names_a_caller_may_write(self):
        for alias, kind in (('deep_beam', 'wall_beam'), ('mixed', 'wall_frame'),
                            ('ribbed_slab', 'slab_ribbed'), ('winkler', 'slab_on_grade'),
                            ('grillage', 'beam_grid'), ('bridge', 'bridge_grillage')):
            self.assertEqual(TA.normalize_kind(alias), kind)

    def test_a_mistake_says_what_the_kind_takes(self):
        for kind, params, word in (
                ('wall_frame', {'wall_side': 'middle'}, 'wall_side'),
                ('slab_sector', {'rmin': 6, 'rmax': 4}, 'rmax'),
                ('slab_sector', {'angle': 400}, 'angle'),
                ('slab_sector', {'outer': 'roller'}, 'outer'),
                ('bridge_grillage', {'girders': 1}, 'girders'),
                ('beam_grid', {'bays_x': 0}, 'bays_x'),
                ('flat_slab', {'thick': 0.2}, 'thick')):
            with self.subTest(kind=kind, params=params):
                with self.assertRaises(ValueError) as cm:
                    Structure2D().create_from_template(kind, **params)
                self.assertIn(word, str(cm.exception))

    def test_a_plate_template_is_refused_on_a_plane_model_that_is_not_empty(self):
        s = Structure2D()
        s.add_node('N0', 0, 0)
        with self.assertRaises(ValueError):
            s.create_from_template('flat_slab')

    def test_an_area_kind_still_fills_a_plate_model_a_bar_kind_does_not(self):
        plate = Structure2D(domain='plate')
        self.assertEqual(TA.domain_mismatch('beam_grid', plate), '')
        self.assertNotEqual(TA.domain_mismatch('wall_frame', plate), '')


class TestMistakesAreSaidClearly(unittest.TestCase):
    def test_an_unknown_kind_lists_the_kinds(self):
        with self.assertRaises(ValueError) as cm:
            Structure2D().create_from_template('dam')
        self.assertIn('continuous_beam', str(cm.exception))

    def test_an_unknown_parameter_lists_what_the_kind_takes(self):
        with self.assertRaises(ValueError) as cm:
            Structure2D().create_from_template('wall', thick=0.25)
        msg = str(cm.exception)
        self.assertIn("'thick'", msg)
        self.assertIn('thickness', msg)

    def test_a_value_that_cannot_be_used(self):
        for kind, params in (('beam', {'span': -3}),
                             ('beam', {'support': 'hinged'}),
                             ('continuous_beam', {'spans': []}),
                             ('continuous_beam', {'spans': [4, 0]}),
                             ('wall', {'width': 'wide'}),
                             ('slab', {'edges': {'north': 'simply'}})):
            with self.subTest(kind=kind, params=params):
                with self.assertRaises(ValueError):
                    Structure2D().create_from_template(kind, **params)

    def test_a_model_that_already_has_geometry_is_refused_and_left_alone(self):
        s = Structure2D()
        s.create_node(0, 0)
        with self.assertRaises(ValueError) as cm:
            s.create_from_template('beam')
        self.assertIn('create_node', str(cm.exception))
        self.assertEqual(len(s.nodes), 1)

    def test_a_model_with_only_definitions_is_still_empty(self):
        s = Structure2D()
        s.create_rc_section('S1', b=0.3, h=0.5)
        s.create_from_template('beam')
        self.assertTrue(s.bar_elements)


class TestTheIncrementCheckerReadsTheTemplate(unittest.TestCase):
    def _kinds(self, code, struc=None):
        return [p['kind'] for p in C.check_increment(
            code, struc=struc if struc is not None else Structure2D())]

    def test_the_ids_the_template_makes_are_known_to_the_lines_after_it(self):
        code = ("model.create_from_template('continuous_beam', spans=[4, 5, 6])\n"
                "model.create_load_case('LC')\n"
                "model.create_uniform_load(['E0', 'E1', 'E2'], 'LC', 15, 'down')\n"
                "model.create_analysis_case('ULS', 'Linear', {'LC': 1.0, 'SW': 1.0})")
        self.assertEqual(self._kinds(code), [])

    def test_a_wall_corner_is_known(self):
        code = ("model.create_from_template('wall', width=5, height=2.5, thickness=0.25)\n"
                "model.create_load_case('LC')\n"
                "model.create_node_load('R.p2', 'LC', 80, 'right')")
        self.assertEqual(self._kinds(code), [])

    def test_a_bar_the_template_did_not_make_is_reported(self):
        code = ("model.create_from_template('continuous_beam', spans=[4, 5, 6])\n"
                "model.create_load_case('LC')\n"
                "model.create_uniform_load('B7', 'LC', 15, 'down')")
        probs = C.check_increment(code, struc=Structure2D())
        self.assertEqual([p['kind'] for p in probs], ['unknown_id'])
        self.assertIn('E0', probs[0]['msg'])

    def test_an_unknown_parameter_is_reported_before_anything_runs(self):
        probs = C.check_increment(
            "model.create_from_template('beam', length=6)", struc=Structure2D())
        self.assertEqual([p['kind'] for p in probs], ['arguments'])
        self.assertIn('span', probs[0]['msg'])

    def test_a_model_with_geometry_is_reported(self):
        s = Structure2D()
        s.create_node(0, 0)
        self.assertEqual(
            self._kinds("model.create_from_template('beam')", struc=s),
            ['template_not_empty'])

    def test_two_calls_are_reported(self):
        code = ("model.create_from_template('beam')\n"
                "model.create_from_template('portal_frame')")
        self.assertEqual(self._kinds(code), ['template_twice'])

    def test_apply_increment_fills_the_model(self):
        result = C.apply_increment(
            Structure2D(),
            "model.create_from_template('continuous_beam', spans=[4, 5, 6])\n"
            "model.create_load_case('LC')\n"
            "model.create_uniform_load(['E0', 'E1', 'E2'], 'LC', 15, 'down')")
        self.assertTrue(result['ok'], result)
        self.assertEqual(len(result['struc'].nodes), 4)

    def test_apply_increment_refuses_a_model_that_has_geometry(self):
        s = Structure2D()
        s.create_node(0, 0)
        result = C.apply_increment(s, "model.create_from_template('beam')")
        self.assertFalse(result['ok'])
        self.assertEqual(len(s.nodes), 1)


if __name__ == '__main__':
    unittest.main()


# --- applying a template over a model that already has geometry -------------

_PORTAL = "model.create_from_template('portal_frame', span=6, height=3)\n"


def _portal():
    from xdfem2d.structure import Structure2D
    from xdfem2d import script_check
    return script_check.apply_increment(Structure2D(), _PORTAL)['struc']


def test_template_replace_model():
    from xdfem2d import script_check
    base = _portal()
    assert not script_check.apply_increment(base, _PORTAL)['ok']
    rep = script_check.apply_increment(base, _PORTAL, replace_model=True)
    assert rep['ok'] and len(rep['struc'].nodes) == len(base.nodes)


def test_template_insert_joins_coincident_nodes():
    from xdfem2d import script_check
    base = _portal()
    rep = script_check.apply_increment(base, _PORTAL, insert_at=(6, 0), join=True)
    assert rep['ok'], rep
    assert len(rep['report'].welded) == 2
    assert len(rep['struc'].nodes) == len(base.nodes) + 2
    assert len(rep['inserted']['nodes']) == 2
    assert len(rep['inserted']['elements']) == 2      # shared column not duplicated
    assert len(base.nodes) == 4                       # base untouched


def test_template_insert_without_join_keeps_nodes_separate():
    from xdfem2d import script_check
    base = _portal()
    rep = script_check.apply_increment(base, _PORTAL, insert_at=(6, 0), join=False)
    assert rep['ok'], rep
    assert not rep['report'].welded
    assert len(rep['struc'].nodes) == 2 * len(base.nodes)
    ids = list(rep['struc'].nodes)
    assert len(ids) == len(set(ids))                  # clashing ids renumbered


def test_template_insert_moves_lower_left_corner():
    from xdfem2d import script_check
    base = _portal()
    rep = script_check.apply_increment(base, _PORTAL, insert_at=(20, 5))
    new = [rep['struc'].nodes[i] for i in rep['inserted']['nodes']]
    assert min(n.x for n in new) == 20 and min(n.y for n in new) == 5


def test_template_insert_domain_mismatch_is_an_error():
    from xdfem2d import script_check
    rep = script_check.apply_increment(
        _portal(), "model.create_from_template('slab', Lx=4, Ly=4)\n",
        insert_at=(0, 0))
    assert not rep['ok'] and rep['error']


def test_reply_check_accepts_template_over_existing_model():
    from xdfem2d import script_check
    base = _portal()
    assert script_check.check_increment(_PORTAL, struc=base)          # refused
    assert not script_check.check_increment_for_apply(_PORTAL, struc=base)


class TestSupportCallsReplace(unittest.TestCase):
    """pin/fix/roller/symm change a node's support; they do not stack on it."""

    def _beam(self):
        from xdfem2d.templates import default_structure
        s = default_structure(domain='plane')
        s.create_from_template('continuous_beam', spans=[4, 5, 6], b=0.3, h=0.5)
        return s

    def test_roller_replaces_the_pin(self):
        s = self._beam()
        s.roller('N1', free='x')
        self.assertEqual([a.support_name for a in s.support_assignments
                          if a.node_id == 'N1'], ['ROLLER-X'])

    def test_a_horizontal_load_is_no_longer_taken_at_the_changed_node(self):
        s = self._beam()
        s.roller('N1', free='x')
        s.add_point_load('N1', s.load_cases[0].id, fx=10.0)
        rx = s.calculate()['reactions'][s.load_cases[0].id]['N1'][0]
        self.assertAlmostEqual(float(rx), 0.0)

    def test_other_nodes_keep_their_support(self):
        s = self._beam()
        s.fix('N1')
        self.assertEqual(sorted(a.node_id for a in s.support_assignments),
                         ['N0', 'N1', 'N2', 'N3'])
