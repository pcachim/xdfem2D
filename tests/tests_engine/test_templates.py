"""Tests for the parametric model templates (xdfem2d.templates).

Each builder must return a Structure2D with the expected topology that solves
without error.
"""
import unittest

from context import Structure2D  # noqa: F401  (adds src/ to sys.path)
from xdfem2d import templates as T

E, G = 2.0e7, 25.0


def _solves(struc):
    """A model solves if it has at least one load case and calculate() runs."""
    if not struc.load_cases:
        struc.add_load_case("LC", self_weight_factor=1.0)
    return struc.calculate()


class TestTopology(unittest.TestCase):
    def test_portal_frame(self):
        s = T.portal_frame(6.0, 3.0, 0.3, 0.5, E, G)
        self.assertEqual(len(s.nodes), 4)
        self.assertEqual(len(s.bar_elements), 3)   # 2 columns + 1 beam

    def test_continuous_beam(self):
        s = T.continuous_beam(3, 5.0, 0.0, 0.3, 0.6, E, G)
        self.assertEqual(len(s.bar_elements), 3)   # 3 spans
        self.assertEqual(len(s.nodes), 4)

    def test_continuous_beam_per_position_supports(self):
        # free left, pin intermediates, fixed right — canonical names, and the
        # free end carries no support.
        s = T.continuous_beam(3, 5.0, 0.0, 0.3, 0.6, E, G,
                              left='free', intermediate='pin', right='fixed')
        by_node = {a.node_id: a.support_name for a in s.support_assignments}
        self.assertNotIn('N0', by_node)              # free left end
        self.assertEqual(by_node['N1'], 'PIN')
        self.assertEqual(by_node['N2'], 'PIN')
        self.assertEqual(by_node['N3'], 'FIXED')
        self.assertIn("displacements", _solves(s))

    def test_single_span_beam_ignores_intermediate(self):
        s = T.continuous_beam(1, 6.0, 0.0, 0.3, 0.6, E, G,
                              left='pin', intermediate='fixed', right='pin')
        self.assertEqual({a.support_name for a in s.support_assignments}, {'PIN'})

    def test_frame_grid(self):
        bays, floors = 2, 3
        s = T.frame(bays, floors, 5.0, 3.0, 0.3, 0.5, E, G)
        self.assertEqual(len(s.nodes), (bays + 1) * (floors + 1))

    def test_truss_panel_count(self):
        s = T.truss_pratt(6, 12.0, 2.0, 0.1, 0.1, 2.0e8, 78.0)
        self.assertGreater(len(s.bar_elements), 6)
        self.assertEqual(len(s.nodes), 14)         # 7 top + 7 bottom chord nodes
        # a real shaped section, not just an area
        self.assertGreater(s.sections['Bar'].inertia, 0.0)

    def test_sector_per_edge_supports(self):
        import math as _m
        # Annulus: outer arc fixed, inner arc pinned, radial sides free.
        s = T.slab_sector(1.0, 5.0, 90.0, 0.6, 0.2,
                          outer='fixed', inner='pin', radial='free')
        byname = {}
        for a in s.support_assignments:
            n = s.nodes[a.node_id]
            byname.setdefault(a.support_name, []).append(round(_m.hypot(n.x, n.y), 2))
        self.assertIn('CLAMPED', byname)                 # outer = fixed
        self.assertIn('SIMPLE', byname)                  # inner = pin
        self.assertTrue(all(abs(r - 5.0) < 1e-6 for r in byname['CLAMPED']))
        self.assertTrue(all(abs(r - 1.0) < 1e-6 for r in byname['SIMPLE']))
        self.assertIn("displacements", _solves(s))

    def test_sector_full_disc_only_outer(self):
        # A full disc has only the outer edge; inner/radial are ignored.
        s = T.slab_sector(0.0, 5.0, 360.0, 0.8, 0.2,
                          outer='pin', inner='fixed', radial='fixed')
        self.assertEqual({a.support_name for a in s.support_assignments}, {'SIMPLE'})
        self.assertIn("displacements", _solves(s))

    def test_town_lattice_pitch_and_span(self):
        # Town lattice: chord pitch rounds to a whole panel count, k-spanning
        # crossing diagonals (2 per (n-k+1)), end posts only, and it solves.
        span, pitch, k = 16.0, 1.0, 3
        s = T.truss_town(span, 2.0, pitch, k, 0.05, 0.1, E, G)
        n = sum(1 for e in s.bar_elements if e.id.startswith('BC'))
        self.assertEqual(n, round(span / pitch))          # pitch → rounded count
        nH = sum(1 for e in s.bar_elements if e.id.startswith('DH'))
        nP = sum(1 for e in s.bar_elements if e.id.startswith('DP'))
        self.assertEqual(nH, n - k + 1)
        self.assertEqual(nP, n - k + 1)
        # End posts only (no interior verticals): exactly two VP members.
        self.assertEqual(sum(1 for e in s.bar_elements
                             if e.id.startswith('VP')), 2)
        self.assertIn("displacements", _solves(s))

    def test_long_truss_x_braced_any_n(self):
        # Long (X-braced): both diagonals in every panel, verticals kept. Two
        # diagonals per panel (Howe "/" + Pratt "\"), symmetric for any n, and
        # solves (statically indeterminate).
        for n in (3, 4, 5):
            s = T.truss_long(n, 12.0, 2.0, 0.1, 0.1, E, G)
            npan = sum(1 for e in s.bar_elements if e.id.startswith('BC'))
            nH = sum(1 for e in s.bar_elements if e.id.startswith('DH'))
            nP = sum(1 for e in s.bar_elements if e.id.startswith('DP'))
            self.assertEqual(nH, npan)               # one "/" per panel
            self.assertEqual(nP, npan)               # one "\" per panel
            self.assertIn("displacements", _solves(s))

    def test_howe_pratt_even_panels_symmetric(self):
        # Howe/Pratt keep an even panel count (odd rounds up), so the single
        # diagonal per panel is always mirror-symmetric and the label holds.
        flip = {'/': '\\', '\\': '/'}
        for fn in (T.truss_howe, T.truss_pratt):
            for n in (3, 5, 7):                    # odd requests
                s = fn(n, 12.0, 2.0, 0.1, 0.1, E, G)
                npan = sum(1 for e in s.bar_elements if e.id.startswith('BC'))
                self.assertEqual(npan % 2, 0)      # rounded to even
                seq = []
                for i in range(npan):
                    e = next(x for x in s.bar_elements if x.id == f'D{i}')
                    ni, nj = s.nodes[e.node_i], s.nodes[e.node_j]
                    lo, hi = (ni, nj) if ni.y < nj.y else (nj, ni)
                    seq.append('/' if lo.x < hi.x else '\\')
                self.assertEqual(seq, [flip[c] for c in reversed(seq)])
                self.assertIn("displacements", _solves(s))

    def test_kingpost_odd_inverts_diagonals(self):
        # An odd panel count (n >= 3) inverts every diagonal relative to the
        # even case, so n and n+1 give distinct trusses (same node count).
        def _diags(n):
            s = T.truss_kingpost(n, 12.0, 3.0, 0.1, 0.2, E, G)
            out = []
            for e in s.bar_elements:
                if e.id.startswith('D'):
                    ni, nj = s.nodes[e.node_i], s.nodes[e.node_j]
                    lo, hi = (ni, nj) if ni.y < nj.y else (nj, ni)
                    out.append('/' if lo.x < hi.x else '\\')
            return out

        even, odd = _diags(4), _diags(3)
        self.assertEqual(len(even), len(odd))            # same panel count
        self.assertTrue(odd)                             # n >= 3 has diagonals
        # Each diagonal is flipped between the even and the odd truss.
        self.assertEqual(odd, ['/' if d == '\\' else '\\' for d in even])
        # n = 5 / n = 6 likewise, and both still solve (not mechanisms).
        self.assertEqual(len(_diags(5)), len(_diags(6)))
        s = T.truss_kingpost(3, 12.0, 3.0, 0.1, 0.2, E, G)
        self.assertIn("displacements", _solves(s))

    def test_arch_segment_count(self):
        n = 8
        s = T.arch(n, 10.0, 3.0, 0.3, 0.4, E, G)
        self.assertEqual(len(s.bar_elements), n)
        self.assertEqual(len(s.nodes), n + 1)

    def test_beam_support_conditions(self):
        for key in ('simply', 'cantilever', 'fixed', 'propped'):
            s = T.beam(key, 6.0, 0.3, 0.5, E, G)
            self.assertEqual(len(s.bar_elements), 4)      # default subdivision
            self.assertEqual(len(s.nodes), 5)
            n_assign = len(s.support_assignments)
            self.assertEqual(n_assign, 1 if key == 'cantilever' else 2)

    def test_wall_mesh(self):
        nx, ny = 6, 4
        s = T.wall(4.0, 3.0, nx, ny, 0.2, E, G)
        self.assertEqual(len(s.nodes), (nx + 1) * (ny + 1))
        self.assertEqual(len(s.tri_elements), 2 * nx * ny)
        self.assertEqual(len(s.tri_sections), 1)
        # whole base fixed
        self.assertEqual(len(s.support_assignments), nx + 1)

    def test_wall_quad_mesh(self):
        nx, ny = 6, 4
        s = T.wall(4.0, 3.0, nx, ny, 0.2, E, G, prefer_quad=True)
        self.assertEqual(len(s.tri_elements), 0)
        self.assertEqual(len(s.quad_elements), nx * ny)   # one quad per cell
        sec = s.quad_elements[0].section_name
        self.assertEqual(s.quad_sections[sec].formulation, "QM6")
        self.assertIn("displacements", _solves(s))

    def test_slab_quad_mesh(self):
        nx, ny = 4, 3
        s = T.slab("simply", 5.0, 4.0, 0.2, nx, ny, prefer_quad=True)
        self.assertEqual(len(s.tri_elements), 0)
        self.assertEqual(len(s.quad_elements), nx * ny)
        sec = s.quad_elements[0].section_name
        self.assertEqual(s.quad_sections[sec].formulation, "MITC4")
        # Every quad carries the self-weight area load.
        self.assertEqual(len(s.quad_area_loads), nx * ny)
        self.assertIn("displacements", _solves(s))

    def test_wall_frame_quad_mesh(self):
        tri = T.wall_frame(2, 2, 5.0, 3.0, 2.0, 3, 0.3, 0.5, E, G, 0.2)
        quad = T.wall_frame(2, 2, 5.0, 3.0, 2.0, 3, 0.3, 0.5, E, G, 0.2,
                            prefer_quad=True)
        self.assertEqual(len(quad.tri_elements), 0)
        self.assertGreater(len(quad.quad_elements), 0)
        self.assertEqual(len(quad.quad_elements), len(tri.tri_elements) // 2)
        self.assertIn("displacements", _solves(quad))

    def test_flat_slab_variable_bays_columns_on_lines(self):
        # Variable bays via a list set the bay count and the exact column lines.
        s = T.flat_slab([6.0, 4.0, 6.0], [5.0, 5.0], 2, 2, 0.22, 2.0)
        cols = s.support_assignments
        self.assertEqual(len(cols), 4 * 3)      # (3+1)×(2+1) column lines
        xs = sorted({round(s.nodes[a.node_id].x, 6) for a in cols})
        ys = sorted({round(s.nodes[a.node_id].y, 6) for a in cols})
        self.assertEqual(xs, [0.0, 6.0, 10.0, 16.0])
        self.assertEqual(ys, [0.0, 5.0, 10.0])
        self.assertIn("displacements", _solves(s))

    def test_flat_slab_max_element_size(self):
        # Each 8 m bay split into ceil(8/2)=4 → 8×8 cells over 2×2 bays.
        s = T.flat_slab(8.0, 8.0, 2, 2, 0.22, 2.0)
        self.assertEqual(len(s.nodes), 9 * 9)

    def test_slab_sector_mesh_quality(self):
        import math as _m
        # The sector uses the unstructured mesher, not a polar grid — every
        # triangle should have a healthy minimum angle even for a narrow sector
        # (the old polar mesh produced ~2.5° slivers here).
        s = T.slab_sector(0.0, 5.0, 20.0, 0.4, 0.2)   # narrow solid sector

        def _min_angle(t):
            P = [(s.nodes[t.node_i].x, s.nodes[t.node_i].y),
                 (s.nodes[t.node_j].x, s.nodes[t.node_j].y),
                 (s.nodes[t.node_k].x, s.nodes[t.node_k].y)]

            def ang(o, x, y):
                v1 = (x[0] - o[0], x[1] - o[1]); v2 = (y[0] - o[0], y[1] - o[1])
                d = ((v1[0] * v2[0] + v1[1] * v2[1])
                     / (_m.hypot(*v1) * _m.hypot(*v2) + 1e-30))
                return _m.degrees(_m.acos(max(-1.0, min(1.0, d))))
            return min(ang(P[0], P[1], P[2]), ang(P[1], P[0], P[2]),
                       ang(P[2], P[0], P[1]))

        self.assertGreater(min(_min_angle(t) for t in s.tri_elements), 12.0)
        self.assertIn("displacements", _solves(s))

    def test_slab_per_edge_supports(self):
        # bottom fixed → CLAMPED, left/right pin → SIMPLE, top free → none.
        edges = {'bottom': 'fixed', 'right': 'pin', 'left': 'pin', 'top': 'free'}
        s = T.slab(edges, 4.0, 3.0, 0.2, 4, 3)
        names = {a.support_name for a in s.support_assignments}
        self.assertEqual(names, {'CLAMPED', 'SIMPLE'})
        self.assertIn("displacements", _solves(s))

    def test_slab_string_shortcut_backcompat(self):
        s = T.slab('clamped', 5.0, 5.0, 0.2, 4, 4)
        self.assertEqual({a.support_name for a in s.support_assignments},
                         {'CLAMPED'})

    def test_slab_ribbed_quad_mesh(self):
        s = T.slab_ribbed(5.0, 5.0, 0.15, 4, 4, 0.3, 0.5, E, G,
                          prefer_quad=True)
        self.assertEqual(len(s.tri_elements), 0)
        self.assertEqual(len(s.quad_elements), 4 * 4)
        self.assertEqual(s.quad_sections[s.quad_elements[0].section_name]
                         .formulation, "MITC4")
        self.assertIn("displacements", _solves(s))

    def test_object_variants_build_objects_and_solve(self):
        cases = [
            T.beam_obj('cantilever', 4.0, 0.3, 0.5, E, G, shape='Rectangular'),
            T.continuous_beam_obj(3, 5.0, 0.3, 0.5, E, G, shape='Rectangular'),
            T.arch_obj(8, 10.0, 3.0, 0.3, 0.4, E, G, shape='Rectangular'),
            T.frame_obj(2, 2, 5.0, 3.0, 0.3, 0.5, E, G, shape='Rectangular'),
            T.wall_obj(4.0, 3.0, 6, 4, 0.2, E, G, 0.2, 'ES-FEM'),
        ]
        for s in cases:
            self.assertGreater(len(s.geometry_objects), 0)  # parametric objects
            r = s.calculate()
            self.assertIn("displacements", r)

    def test_object_slab_wall_variants_honour_quad_choice(self):
        """prefer_quad must reach the object variants' meshed elements: the
        surface objects mesh into quads when asked, triangles otherwise
        (regression — the GUI dispatch used to drop prefer_quad for objects)."""
        from xdfem2d.geo_expand import expand_geometry
        makers = {
            'slab_edges_obj': lambda pq: T.slab_edges_obj('simply', prefer_quad=pq),
            'flat_slab_obj':  lambda pq: T.flat_slab_obj(prefer_quad=pq),
            'slab_on_grade_obj': lambda pq: T.slab_on_grade_obj(prefer_quad=pq),
            'slab_ribbed_obj': lambda pq: T.slab_ribbed_obj(prefer_quad=pq),
            'wall_obj': lambda pq: T.wall_obj(4.0, 3.0, 4, 3, 0.2, E, G, 0.2,
                                              prefer_quad=pq),
        }
        for name, mk in makers.items():
            mt, _ = expand_geometry(mk(False))
            mq, _ = expand_geometry(mk(True))
            self.assertGreater(len(mt.tri_elements), 0, name)
            self.assertEqual(len(getattr(mt, 'quad_elements', [])), 0, name)
            self.assertGreater(len(mq.quad_elements), 0, name)
            self.assertEqual(len(mq.tri_elements), 0, name)

    def test_slab_pressure_is_a_separate_imposed_case(self):
        """A slab's surface pressure lives in its own imposed case 'Q' (not
        lumped into the self-weight case 'SW'), with 'SW' left as self-weight
        only, and 'Q' folded into the ULS/SLS combinations (1.5 / 1.0)."""
        for mk in (lambda: T.slab('simply'), lambda: T.slab_edges_obj('simply'),
                   lambda: T.flat_slab(), lambda: T.slab_on_grade()):
            s = mk()
            ids = [lc.id for lc in s.load_cases]
            self.assertIn('SW', ids)
            self.assertIn('Q', ids)
            r = s.calculate()
            # Self-weight alone still sags (it is a real load, not empty).
            self.assertLess(min(d[0] for d in r['displacements']['SW'].values()), 0.0)
            # The pressure drives the 'Q' case.
            self.assertLess(min(d[0] for d in r['displacements']['Q'].values()), 0.0)
            combos = {c.id: c.coefficients for c in s.load_combinations}
            self.assertEqual(combos['ULS-1.35G'].get('Q'), 1.5)
            self.assertEqual(combos['SLS-G'].get('Q'), 1.0)

    def test_object_beam_matches_explicit(self):
        se = T.beam('cantilever', 4.0, 0.3, 0.5, E, G, shape='Rectangular')
        so = T.beam_obj('cantilever', 4.0, 0.3, 0.5, E, G, shape='Rectangular')
        te = min(se.calculate()['displacements']['SW'][n][1] for n in se.nodes)
        ro = so.calculate()
        to = min(ro['displacements']['SW'][n][1] for n in so.nodes)
        self.assertAlmostEqual(te, to, places=6)

    def test_default_structure_is_usable(self):
        s = T.default_structure()
        self.assertGreaterEqual(len(s.materials), 1)
        self.assertGreaterEqual(len(s.sections), 1)
        # A CST section too: creating a surface object needs one, so without it
        # the first Add surface on a brand-new model cannot even start.
        self.assertGreaterEqual(len(s.tri_sections), 1)
        for ts in s.tri_sections.values():
            self.assertIn(ts.material_name, s.materials)


class TestSolvable(unittest.TestCase):
    """Every template, once supported and loaded, must solve."""

    def test_all_templates_solve(self):
        builders = [
            T.beam('simply', 6.0, 0.3, 0.5, E, G),
            T.beam('cantilever', 4.0, 0.3, 0.5, E, G),
            T.beam('fixed', 6.0, 0.3, 0.5, E, G),
            T.beam('propped', 6.0, 0.3, 0.5, E, G),
            T.wall(4.0, 3.0, 6, 4, 0.2, E, G),
            T.portal_frame(6.0, 3.0, 0.3, 0.5, E, G),
            T.continuous_beam(3, 5.0, 0.0, 0.3, 0.6, E, G),
            T.frame(2, 2, 5.0, 3.0, 0.3, 0.5, E, G),
            T.truss_warren(6, 12.0, 2.0, 0.1, 0.1, 2.0e8, 78.0),
            T.truss_howe(6, 12.0, 2.0, 0.1, 0.1, 2.0e8, 78.0),
            T.truss_pratt(6, 12.0, 2.0, 0.1, 0.1, 2.0e8, 78.0),
            T.arch(8, 10.0, 3.0, 0.3, 0.4, E, G),
        ]
        for s in builders:
            r = _solves(s)
            self.assertIn("displacements", r)


class TestMaterialTypePropagation(unittest.TestCase):
    """The 'Mat' material every bar-shape template creates must carry the
    material_type the caller asked for -- it used to always default to
    MaterialType.CONCRETE (Structure2D.add_material's own default),
    regardless of what a GUI 'Steel'/'Timber' selection intended. See dev/
    RC_COLUMN_DESIGN.md (New from Template + concrete section shape)."""

    def test_default_is_concrete(self):
        """The material's own NAME reflects the type too -- 'Conc' by
        default -- not just a generic 'Mat' that happens to carry
        material_type='Concrete' internally."""
        s = T.portal_frame(6.0, 3.0, 0.3, 0.5, E, G)
        self.assertIn('Conc', s.materials)
        self.assertEqual(s.materials['Conc'].material_type.value, 'Concrete')

    def test_steel_is_propagated(self):
        s = T.portal_frame(6.0, 3.0, 0.3, 0.5, 2.1e8, 78.5,
                           shape='I', material_type='Steel')
        self.assertIn('Steel', s.materials)
        self.assertEqual(s.materials['Steel'].material_type.value, 'Steel')

    def test_timber_is_propagated(self):
        s = T.frame(2, 2, 5.0, 3.0, 0.3, 0.5, 1.1e7, 4.6,
                    material_type='Timber')
        self.assertIn('Timber', s.materials)
        self.assertEqual(s.materials['Timber'].material_type.value, 'Timber')

    def test_truss_propagates_material_type(self):
        s = T.truss_warren(6, 12.0, 2.0, 0.1, 0.1, 2.0e8, 78.0,
                           shape='I', material_type='Steel')
        self.assertIn('Steel', s.materials)
        self.assertEqual(s.materials['Steel'].material_type.value, 'Steel')

    def test_sections_reference_the_named_material(self):
        """Sections must follow the renamed material, not the old 'Mat'."""
        s = T.frame(2, 2, 5.0, 3.0, 0.3, 0.5, E, G)
        for sec in s.sections.values():
            self.assertEqual(sec.material_name, 'Conc')


if __name__ == "__main__":
    unittest.main(verbosity=2)
