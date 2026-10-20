"""The model written back out as Python.

One test carries this module: generate the script, run it, and compare the
result with the model it came from, key by key against the save format. Every
example that ships with the application goes through it. A generator can be
wrong in a hundred small ways that all read plausibly, and only executing it
tells them apart — which is the same argument that made this exporter worth
writing at all.

The second theme is the promise in the module docstring: nothing is omitted
silently. What cannot be expressed is declared, and the tests check that
declaration against the save format, so adding a feature to the model without
deciding what the exporter does with it fails here.
"""
from __future__ import annotations

import glob
import os
import unittest
from pathlib import Path

from context import Structure2D

from xdfem2d import script_export as X
from xdfem2d.file_io import load_x2d
from xdfem2d.structure_io import _to_dict

EXAMPLES = sorted(glob.glob(str(Path(__file__).resolve().parent.parent.parent
                                / 'examples' / '*.x2d')))
E = 30e6


def _rebuild(struc, source=""):
    """Run the generated script and return the model it builds."""
    src = X.to_python(struc, source=source)
    ns: dict = {}
    exec(compile(src, '<generated>', 'exec'), ns)     # noqa: S102
    return ns['build'](), src


def _plate_model():
    """A plate model exercising every plate-only export path: domain, a DKT
    section, an area (pressure) load, a Winkler area spring, a thermal gradient,
    a grillage bar whose section carries a torsion override J, and (dev/
    IMPLEMENT_QUAD.md Phase 3/4) a quad section/element/area load alongside
    the triangles, reusing the same four corner nodes (already CCW)."""
    s = Structure2D(domain='plate')
    s.add_material('M', elastic_modulus=30e6, unit_weight=25.0, poisson=0.2,
                   alpha=1e-5)
    s.add_plate_section('Sl', 'M', thickness=0.2)
    s.add_section('Rib', 'M', b=0.3, h=0.5, torsion_override=0.004)
    s.add_quad_section('Qs', 'M', thickness=0.2, formulation='MITC4')
    for i, (x, y) in enumerate([(0, 0), (4, 0), (4, 3), (0, 3)]):
        s.add_node(f'N{i}', float(x), float(y))
    s.add_tri_element('T1', 'N0', 'N1', 'N2', 'Sl')
    s.add_tri_element('T2', 'N0', 'N2', 'N3', 'Sl')
    s.add_quad_element('Q1', 'N0', 'N1', 'N2', 'N3', 'Qs')
    s.add_bar_element('B1', 'N0', 'N1', 'Rib')
    s.add_support('SS', w=True)
    for n in ('N0', 'N1', 'N2', 'N3'):
        s.assign_support(n, 'SS')
    # create_analysis_case=False: the explicit add_analysis_case('G', ...)
    # below now collides with add_load_case's own auto-created twin
    # otherwise, since add_analysis_case refuses a duplicate id (26/09/2026).
    s.add_load_case('G', self_weight_factor=1.0, create_analysis_case=False)
    s.add_area_load('T1', 'G', pz=-5.0)
    s.add_area_load('Q1', 'G', pz=-5.0)
    s.add_area_spring('T2', kz=1000.0)
    s.add_tri_temperature_load('T1', 'G', dt_gradient=10.0)
    s.add_analysis_case('G', 'Linear', {'G': 1.0})
    return s


class TestPlateRoundTrip(unittest.TestCase):
    """A plate model must survive the round trip like a plane one, including the
    plate-only features (domain, DKT section, area load/spring, thermal
    gradient, torsion override)."""

    def test_it_rebuilds_identically(self):
        struc = _plate_model()
        rebuilt, src = _rebuild(struc)
        before, after = _to_dict(struc), _to_dict(rebuilt)
        for key in sorted(X.HANDLED):
            self.assertEqual(before.get(key), after.get(key),
                             f"{key} differs after the round trip")
        # The domain is carried and the section keeps its J override.
        self.assertEqual(after['domain'], 'plate')
        self.assertIn("domain='plate'", src)

    def test_the_checker_passes_the_generated_plate_script(self):
        from xdfem2d import script_check
        src = X.to_python(_plate_model())
        self.assertEqual(script_check.check(src), [])


class TestRoundTrip(unittest.TestCase):
    """Generate, run, compare. The test the module exists to pass."""

    def test_every_example_rebuilds_identically(self):
        self.assertTrue(EXAMPLES, "no examples found to test against")
        for path in EXAMPLES:
            with self.subTest(model=os.path.basename(path)):
                struc, _, _ = load_x2d(path)
                rebuilt, _ = _rebuild(struc, os.path.basename(path))
                before, after = _to_dict(struc), _to_dict(rebuilt)
                for key in sorted(X.HANDLED):
                    self.assertEqual(before.get(key), after.get(key),
                                     f"{key} differs after the round trip")

    def test_the_rebuilt_model_solves_to_the_same_numbers(self):
        """Equal dictionaries are not quite the point; equal results are."""
        struc, _, _ = load_x2d(str(Path(__file__).resolve().parent.parent.parent
                                   / 'examples'
                                   / 'example-continuous_beam.x2d'))
        rebuilt, _ = _rebuild(struc)
        a = struc.calculate()['combinations']
        b = rebuilt.calculate()['combinations']
        self.assertEqual(sorted(a), sorted(b))
        for combo in a:
            for node, value in a[combo]['reactions'].items():
                for i, v in enumerate(value):
                    self.assertAlmostEqual(v, b[combo]['reactions'][node][i],
                                           places=9)

    def test_geometry_objects_keep_their_nodes(self):
        """Rebuilt from node ids, not from coordinates.

        Through the public constructors an object looks for a node already
        sitting at each defining point, and in a model with two nodes at the
        same place it finds whichever comes first: the frame-and-wall example
        came back with its wall standing on 'WB_13' where it had stood on
        'Rect1.p1'. Same geometry, different ids, and every result keyed by
        them.
        """
        path = (Path(__file__).resolve().parent.parent.parent / 'examples'
                / 'example-frame_and_wall.x2d')
        struc, _, _ = load_x2d(str(path))
        rebuilt, _ = _rebuild(struc)
        for oid, obj in struc.geometry_objects.items():
            self.assertEqual(list(obj.node_ids),
                             list(rebuilt.geometry_objects[oid].node_ids))


class TestRunningItLeavesSomethingYouCanOpen(unittest.TestCase):
    """The script built and solved a model and then threw it away.

    The loop is worth closing: a variation written by editing the script is of
    little use if the only way back into the application is to rebuild it by
    hand. Running the file now writes an .x2d beside itself.
    """

    def test_the_script_is_actually_run_here(self):
        """Not the generated text inspected — the file executed as __main__,
        which is the only way to know the save line works."""
        import subprocess
        import sys
        import tempfile

        struc, _, _ = load_x2d(str(Path(__file__).resolve().parent.parent.parent
                                   / 'examples'
                                   / 'example-continuous_beam.x2d'))
        with tempfile.TemporaryDirectory() as tmp:
            script = Path(tmp) / 'model.py'
            script.write_text(X.to_python(struc), encoding='utf-8')
            src = str(Path(__file__).resolve().parent.parent.parent / 'src')
            proc = subprocess.run([sys.executable, str(script)],
                                  capture_output=True, text=True,
                                  env={'PYTHONPATH': src, 'PATH': '/usr/bin:/bin'})
            self.assertEqual(proc.returncode, 0, proc.stderr)
            written = Path(tmp) / 'model.x2d'
            self.assertTrue(written.exists(), proc.stdout + proc.stderr)
            self.assertIn('saved:', proc.stdout)

            # And what it wrote opens, with the model it was given.
            # The default __main__ block saves the model only (no analysis),
            # so results is None or empty — the nodes are what matter.
            back, results, _ = load_x2d(str(written))
            self.assertEqual(sorted(back.nodes), sorted(struc.nodes))

    def test_it_writes_under_its_own_name(self):
        """So running it twice overwrites its own output and never something
        the user was keeping."""
        text = X.to_python(_beam())
        self.assertIn("Path(__file__).with_suffix('.x2d')", text)

    def test_the_header_says_that_running_it_writes_a_file(self):
        """A side effect the reader learns about from the code alone is a
        surprise; this one is announced before the first line of it."""
        head = X.to_python(_beam()).split('"""')[1]
        self.assertIn('.x2d', head)


class TestNothingIsOmittedSilently(unittest.TestCase):

    def test_the_declaration_covers_the_save_format(self):
        """Every key of the save format is either written or declared missing.

        This is the maintenance contract. Add a feature to the model, and this
        test fails until someone decides whether the exporter writes it — which
        is better than a script that quietly leaves it out.
        """
        struc, _, _ = load_x2d(str(Path(__file__).resolve().parent.parent.parent
                                   / 'examples'
                                   / 'example_2level_2bay_full.x2d'))
        keys = set(_to_dict(struc))
        undecided = keys - X.HANDLED - set(X.DECLARED_UNSUPPORTED)
        self.assertEqual(undecided, set(),
                         "new save-format keys: write them or declare them "
                         "in DECLARED_UNSUPPORTED")

    def test_the_two_sets_do_not_overlap(self):
        self.assertEqual(X.HANDLED & set(X.DECLARED_UNSUPPORTED), set())

    def test_a_model_with_nothing_missing_says_nothing(self):
        s = _beam()
        self.assertEqual(X.unsupported_in(s), [])
        self.assertNotIn('NOT INCLUDED', X.to_python(s))

    def test_a_missing_feature_is_named_in_the_file(self):
        from xdfem2d.models import SupportSet
        s = _beam()
        s.add_support_set(SupportSet(id='SS1', description='provisional'))
        self.assertIn('support sets', X.unsupported_in(s))
        text = X.to_python(s)
        self.assertIn('NOT INCLUDED', text)
        self.assertIn('support sets', text)

    def test_the_warning_is_inside_the_docstring(self):
        """So it cannot be scrolled past as a comment, and survives a paste."""
        from xdfem2d.models import SupportSet
        s = _beam()
        s.add_support_set(SupportSet(id='SS1'))
        text = X.to_python(s)
        self.assertLess(text.index('NOT INCLUDED'), text.index('"""', 3) + 4)


class TestItIsReadable(unittest.TestCase):
    """A script nobody reads is only a slower .x2d."""

    def test_default_arguments_are_left_out(self):
        """add_section takes fourteen parameters and two of them matter."""
        s = _beam()
        line = _line_with(X.to_python(s), 'model.add_section(')
        self.assertIn("0.3, 0.6", line)
        self.assertNotIn('rc_bar_phi', line)
        self.assertNotIn('area_override', line)

    def test_a_changed_argument_is_kept(self):
        s = _beam()
        s.sections['S'].rc_bar_phi = 25.0
        self.assertIn('rc_bar_phi=25.0',
                      _line_with(X.to_python(s), 'model.add_section('))

    def test_a_static_case_carries_no_modal_parameters(self):
        """num_modes and damping are AnalysisCase defaults, not choices, and
        four of them on every Linear case read as meaning."""
        s = _beam()
        s.add_analysis_case('ULS', 'Linear', {'LC': 1.35})
        line = _line_with(X.to_python(s), "model.add_analysis_case('ULS'")
        self.assertNotIn('num_modes', line)
        self.assertNotIn('damping', line)

    def test_a_non_default_analysis_field_is_not_lost(self):
        """The whitelist this replaced had no place for max_iterations, and a
        P-Delta case exported with 50 came back with 100."""
        s = _beam()
        s.add_analysis_case('PD', 'GeometricNonlinear', {'LC': 1.0},
                            max_iterations=50)
        self.assertIn('max_iterations=50',
                      _line_with(X.to_python(s), "model.add_analysis_case('PD'"))

    def test_sections_are_headed_with_their_counts(self):
        text = X.to_python(_beam())
        self.assertIn('# ── Nodes (2)', text)
        self.assertIn('# ── Materials (1)', text)

    def test_the_source_file_is_named_when_known(self):
        self.assertIn('Source: beam.x2d', X.to_python(_beam(), source='beam.x2d'))

    def test_the_project_metadata_reaches_the_header_and_the_model(self):
        """The header wrote 'name', 'author' and 'description'; project_info
        holds 'Project name', 'Owner' and 'Designer'. Nothing matched, so no
        model ever had its metadata in the header — and, listed as handled but
        never written, it was silently dropped from the rebuilt model too.

        Invisible until an example was saved with the fields filled in, which
        is the argument for testing against real files rather than fixtures.
        """
        s = _beam()
        s.project_info['Project name'] = 'Footbridge'
        s.project_info['Designer'] = 'A. Engineer'
        text = X.to_python(s)
        self.assertIn('Project name: Footbridge', text)
        self.assertIn('Designer: A. Engineer', text)
        rebuilt, _ = _rebuild(s)
        self.assertEqual(rebuilt.project_info, s.project_info)

    def test_empty_metadata_fields_are_left_out(self):
        """Four blank lines under the title, and five assignments in build(),
        for a dictionary that starts out empty on every new model."""
        text = X.to_python(_beam())
        self.assertNotIn('Location:', text)
        self.assertNotIn('project_info[', text)


class TestItIsValidPython(unittest.TestCase):

    def test_an_empty_model_still_produces_a_runnable_file(self):
        rebuilt, src = _rebuild(Structure2D())
        self.assertEqual(len(rebuilt.nodes), 0)
        self.assertIn('def build()', src)

    def test_floats_round_trip_exactly(self):
        """A coordinate written rounded is a different model."""
        s = Structure2D()
        s.add_node('N1', 0.1 + 0.2, 1.0 / 3.0)
        rebuilt, _ = _rebuild(s)
        self.assertEqual(rebuilt.nodes['N1'].x, 0.1 + 0.2)
        self.assertEqual(rebuilt.nodes['N1'].y, 1.0 / 3.0)

    def test_quotes_in_an_id_do_not_break_the_file(self):
        s = Structure2D()
        s.add_node("it's", 0.0, 0.0)
        rebuilt, _ = _rebuild(s)
        self.assertIn("it's", rebuilt.nodes)


def _beam() -> Structure2D:
    s = Structure2D()
    s.add_material('M', elastic_modulus=E, unit_weight=25.0)
    s.add_section('S', 'M', b=0.3, h=0.6)
    s.add_node('N1', 0.0, 0.0)
    s.add_node('N2', 6.0, 0.0)
    s.add_bar_element('E1', 'N1', 'N2', 'S')
    s.add_support('PIN', ux=True, uy=True)
    s.assign_support('N1', 'PIN')
    s.add_load_case('LC')
    return s


def _line_with(text: str, needle: str) -> str:
    for line in text.splitlines():
        if needle in line:
            return line
    raise AssertionError(f"no line containing {needle!r}")


if __name__ == '__main__':
    unittest.main()
