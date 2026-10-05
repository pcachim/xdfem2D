"""Tests for the script editor execution engine.

These tests exercise the execution logic of ScriptRunWorker and the
integration between check_editor() and the runner — without starting Qt.

The worker runs on a QThread normally, but all the interesting behaviour is in
its run() method.  We test that directly by extracting the logic into a helper
(_run_script) that mirrors what run() does but is callable from plain Python.

What is covered:
  - Mode 1 (build): script defines build() → returns Structure2D
  - Mode 2 (snippet): script operates on injected s
  - deepcopy isolation: snippet failure doesn't mutate the original model
  - print() output captured and separated from the result
  - Syntax errors reported cleanly (no exec)
  - Runtime exceptions reported without crashing the host
  - SAFE_BUILTINS blocks open/exec at runtime
  - from-import inside the script works (needs __import__ in builtins)
  - check_editor() stops forbidden imports / dunder / dangerous builtins
    before exec is ever called
  - Round-trip: a script_export output reruns to an equivalent model
"""
from __future__ import annotations

import sys
import os
import unittest

# Make src/ importable without an install
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(_ROOT, 'src'))

from xdfem2d import Structure2D, script_check, script_export
from xdfem2d.script_runner import SAFE_BUILTINS, run_script as _run_script


# ---------------------------------------------------------------------------
# Build mode
# ---------------------------------------------------------------------------

class TestBuildMode(unittest.TestCase):

    def _build(self, body: str) -> Structure2D:
        code = f"from xdfem2d import Structure2D\ndef build():\n{body}\n"
        result, _ = _run_script(code)
        return result

    def test_simple_build_returns_structure(self):
        s = self._build(
            "    s = Structure2D()\n"
            "    s.add_node('N1', 0, 0)\n"
            "    return s\n"
        )
        self.assertIsInstance(s, Structure2D)
        self.assertIn('N1', s.nodes)

    def test_build_with_two_nodes(self):
        s = self._build(
            "    s = Structure2D()\n"
            "    s.add_node('N1', 0, 0)\n"
            "    s.add_node('N2', 5, 0)\n"
            "    return s\n"
        )
        self.assertEqual(len(s.nodes), 2)

    def test_build_mode_detected_by_def_build(self):
        """Detection must work regardless of leading whitespace."""
        code = "\n\ndef build():\n    s = Structure2D()\n    return s\n"
        result, _ = _run_script(code)
        self.assertIsInstance(result, Structure2D)

    def test_from_import_inside_build_script(self):
        """from xdfem2d import Structure2D must work inside the restricted ns."""
        code = (
            "from xdfem2d import Structure2D\n"
            "def build():\n"
            "    s = Structure2D()\n"
            "    s.add_node('A', 1, 2)\n"
            "    return s\n"
        )
        result, _ = _run_script(code)
        self.assertIn('A', result.nodes)

    def test_math_available_in_build(self):
        code = (
            "import math\n"
            "def build():\n"
            "    s = Structure2D()\n"
            "    s.add_node('N1', math.cos(0), math.sin(0))\n"
            "    return s\n"
        )
        result, _ = _run_script(code)
        self.assertEqual(len(result.nodes), 1)

    def test_loop_generates_nodes(self):
        code = (
            "def build():\n"
            "    s = Structure2D()\n"
            "    for i in range(5):\n"
            "        s.add_node(f'N{i}', i * 3.0, 0)\n"
            "    return s\n"
        )
        result, _ = _run_script(code)
        self.assertEqual(len(result.nodes), 5)

    def test_print_is_captured_not_lost(self):
        code = (
            "def build():\n"
            "    print('building…')\n"
            "    s = Structure2D()\n"
            "    return s\n"
        )
        _, out = _run_script(code)
        self.assertIn('building', out)

    def test_syntax_error_raises(self):
        with self.assertRaises(SyntaxError):
            _run_script("def build(:\n    pass\n")

    def test_runtime_error_propagates(self):
        code = (
            "def build():\n"
            "    raise ValueError('deliberate')\n"
        )
        with self.assertRaises(ValueError):
            _run_script(code)


# ---------------------------------------------------------------------------
# Snippet mode
# ---------------------------------------------------------------------------

class TestSnippetMode(unittest.TestCase):

    def _snip(self, body: str, base: Structure2D | None = None) -> Structure2D:
        result, _ = _run_script(body, struc=base)
        return result

    def test_snippet_receives_injected_model(self):
        base = Structure2D()
        base.add_node('A', 0, 0)
        result = self._snip("model.add_node('B', 3, 0)", base)
        self.assertIn('A', result.nodes)
        self.assertIn('B', result.nodes)

    def test_snippet_does_not_mutate_original(self):
        base = Structure2D()
        base.add_node('A', 0, 0)
        self._snip("model.add_node('B', 3, 0)", base)
        self.assertEqual(len(base.nodes), 1,
                         "original model was mutated — deepcopy failed")

    def test_snippet_on_empty_model(self):
        result = self._snip("model.add_node('X', 7, 0)")
        self.assertIn('X', result.nodes)

    def test_snippet_modification_with_loop(self):
        base = Structure2D()
        base.add_material('S275', elastic_modulus=210e6, unit_weight=78.5, alpha=1.2e-5)
        base.add_section('HEB200', material_name='S275', b=0.2, h=0.2)
        for i in range(4):
            base.add_node(f'N{i}', i * 3.0, 0)
        for i in range(3):
            base.add_bar_element(f'B{i}', f'N{i}', f'N{i+1}', 'HEB200')

        code = (
            "for b in model.bar_elements:\n"
            "    b.section_name = 'HEB200'\n"   # no-op but exercises iteration
        )
        result = self._snip(code, base)
        self.assertEqual(len(result.bar_elements), 3)

    def test_snippet_print_captured(self):
        _, out = _run_script("print('snippet output')")
        self.assertIn('snippet output', out)

    def test_snippet_syntax_error_raises(self):
        with self.assertRaises(SyntaxError):
            _run_script("model.add_node(")

    def test_snippet_runtime_error_does_not_mutate_original(self):
        base = Structure2D()
        base.add_node('A', 0, 0)
        with self.assertRaises(Exception):
            _run_script("model.add_node('B', 0, 0)\nraise RuntimeError('oops')", base)
        self.assertEqual(len(base.nodes), 1)


# ---------------------------------------------------------------------------
# Safe builtins — runtime blocking
# ---------------------------------------------------------------------------

class TestSafeBuiltinsAtRuntime(unittest.TestCase):
    """These functions must not be reachable even if check_editor is bypassed."""

    def _exec_raw(self, code: str) -> None:
        """Exec directly in the restricted namespace — bypasses check_editor."""
        ns = {'__builtins__': SAFE_BUILTINS}
        exec(compile(code, '<t>', 'exec'), ns)  # noqa: S102

    def test_open_is_not_available(self):
        with self.assertRaises(NameError):
            self._exec_raw("open('/etc/passwd')")

    def test_exec_is_not_available(self):
        with self.assertRaises(NameError):
            self._exec_raw("exec('x=1')")

    def test_eval_is_not_available(self):
        with self.assertRaises(NameError):
            self._exec_raw("eval('1+1')")

    def test_compile_is_not_available(self):
        with self.assertRaises(NameError):
            self._exec_raw("compile('pass','<s>','exec')")

    def test_breakpoint_is_not_available(self):
        with self.assertRaises(NameError):
            self._exec_raw("breakpoint()")

    def test_safe_builtins_contain_expected_names(self):
        for name in ('abs', 'len', 'range', 'print', 'list', 'dict',
                     'min', 'max', 'sum', 'enumerate', 'zip', 'sorted'):
            with self.subTest(name=name):
                self.assertIn(name, SAFE_BUILTINS)

    def test_import_machinery_works(self):
        """__import__ is present so that import statements inside scripts work."""
        self.assertIn('__import__', SAFE_BUILTINS)


# ---------------------------------------------------------------------------
# Integration with check_editor() — pre-run AST gate
# ---------------------------------------------------------------------------

class TestCheckEditorGate(unittest.TestCase):
    """check_editor() must catch forbidden patterns before any exec."""

    def _kinds(self, code: str) -> set[str]:
        return {p['kind'] for p in script_check.check_editor(code)}

    def test_clean_build_script_passes(self):
        code = (
            "from xdfem2d import Structure2D\n"
            "def build():\n"
            "    s = Structure2D()\n"
            "    s.add_node('N1', 0, 0)\n"
            "    s.pin('N1')\n"
            "    return s\n"
        )
        self.assertFalse(script_check.check_editor(code))

    def test_forbidden_import_blocked(self):
        code = "import os\ns = Structure2D()\n"
        self.assertIn('forbidden_import', self._kinds(code))

    def test_forbidden_import_from_blocked(self):
        code = "from pathlib import Path\ns = Structure2D()\n"
        self.assertIn('forbidden_import', self._kinds(code))

    def test_open_call_blocked(self):
        code = "from xdfem2d import Structure2D\nopen('/etc/passwd')\n"
        self.assertIn('forbidden_builtin', self._kinds(code))

    def test_dunder_access_blocked(self):
        code = "from xdfem2d import Structure2D\nx = ().__class__.__bases__\n"
        self.assertIn('dunder_access', self._kinds(code))

    def test_security_problems_prevent_correctness_check(self):
        """A script with a security violation must not surface method errors."""
        code = "import os\nfrom xdfem2d import Structure2D\ns = Structure2D()\ns.fake_method()\n"
        kinds = self._kinds(code)
        self.assertIn('forbidden_import', kinds)
        self.assertNotIn('unknown_method', kinds)

    def test_syntax_error_surfaced_by_check_editor(self):
        kinds = self._kinds("def build(:\n    pass\n")
        self.assertIn('syntax', kinds)


# ---------------------------------------------------------------------------
# Round-trip: export → re-run produces equivalent model
# ---------------------------------------------------------------------------

class TestRoundTrip(unittest.TestCase):
    """A script produced by script_export must re-execute cleanly."""

    def _roundtrip(self, struc: Structure2D) -> Structure2D:
        code = script_export.to_python(struc)
        # The exported script defines build()
        result, _ = _run_script(code)
        return result

    def _beam(self) -> Structure2D:
        s = Structure2D()
        s.add_material('S275', elastic_modulus=210e6, unit_weight=78.5, alpha=1.2e-5)
        s.add_section('HEB200', material_name='S275', b=0.2, h=0.2)
        s.add_node('N1', 0, 0)
        s.add_node('N2', 6, 0)
        s.add_bar_element('B1', 'N1', 'N2', 'HEB200')
        s.pin('N1')
        s.roller('N2')
        s.add_load_case('G', action_type='permanent')
        s.add_point_load('N2', 'G', fy=-50)
        return s

    def test_beam_nodes_survive_roundtrip(self):
        orig = self._beam()
        rebuilt = self._roundtrip(orig)
        self.assertEqual(set(rebuilt.nodes.keys()), set(orig.nodes.keys()))

    def test_beam_elements_survive_roundtrip(self):
        orig = self._beam()
        rebuilt = self._roundtrip(orig)
        orig_ids = {e.id for e in orig.bar_elements}
        new_ids  = {e.id for e in rebuilt.bar_elements}
        self.assertEqual(new_ids, orig_ids)

    def test_beam_load_cases_survive_roundtrip(self):
        orig = self._beam()
        rebuilt = self._roundtrip(orig)
        self.assertEqual({lc.id for lc in rebuilt.load_cases},
                         {lc.id for lc in orig.load_cases})

    def test_export_passes_check_editor(self):
        """The exporter must never produce code that fails the security gate."""
        code = script_export.to_python(self._beam())
        problems = script_check.check_editor(code)
        security_kinds = {'forbidden_import', 'forbidden_builtin', 'dunder_access'}
        self.assertFalse(
            [p for p in problems if p['kind'] in security_kinds],
            "script_export produced code that fails check_editor security rules"
        )


if __name__ == '__main__':
    unittest.main()
