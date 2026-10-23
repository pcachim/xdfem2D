"""The engine still works when the optional extras are not installed.

``pip install xdfem2d`` brings numpy, scipy and eurocodepy. Reporting
(matplotlib, openpyxl, python-docx) and DXF exchange (ezdxf) are extras,
because they are reached from two modules and only from inside the functions
that need them — someone using this as a solver in a script or a service should
not have to install a Word library to do it.

That promise holds only while those imports stay lazy. One module-level
``import matplotlib`` in report_io turns a missing extra into an ImportError on
``import xdfem2d``, and every core install breaks — for a feature the user
never asked for. Nothing else in the suite would notice, because the developer
machine has all of them.

So this makes the modules genuinely unimportable, the way a core install does,
and then builds and solves a model. The technique is borrowed from
``build_app.py``, where the same class of failure already shipped once: a
dependency imported something at module scope, the packaged application
excluded it, and the app died on launch while every test passed.
"""
from __future__ import annotations

import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
SRC = ROOT / 'src'

EXTRAS = ('matplotlib', 'openpyxl', 'docx', 'ezdxf')

# Run in a subprocess: the blocker has to be installed before the first import
# of the package, and this suite has already imported it.
_SCRIPT = r"""
import importlib.abc, importlib.machinery, sys

BLOCKED = set(sys.argv[1].split(','))


class _Blocker(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name.split('.')[0] in BLOCKED:
            raise ImportError(f"{name} is not installed (simulated)")
        return None


sys.meta_path.insert(0, _Blocker())

import xdfem2d
from xdfem2d import Structure2D

s = Structure2D()
s.add_material('M', elastic_modulus=30e6, unit_weight=25.0)
s.add_section('S', 'M', b=0.3, h=0.5)
s.add_node('N1', 0.0, 0.0)
s.add_node('N2', 6.0, 0.0)
s.add_bar_element('E1', 'N1', 'N2', 'S')
s.add_support('PIN', ux=True, uy=True)
s.add_support('ROLL', uy=True)
s.assign_support('N1', 'PIN')
s.assign_support('N2', 'ROLL')
s.add_load_case('LC')
s.add_distributed_load('E1', 'LC', fye=-20.0, fyd=-20.0)
s.add_analysis_case('LIN', 'Linear', {'LC': 1.0})
r = s.calculate()

ry = sum(v[1] for v in r['analysis_cases']['LIN']['reactions'].values())
print('REACTION', round(ry, 6))

# Saving and loading a model is core, not an extra: an engine that cannot read
# its own files is not usable headless.
import tempfile, os
from xdfem2d.file_io import save_x2d, load_x2d
with tempfile.TemporaryDirectory() as tmp:
    p = os.path.join(tmp, 'm.x2d')
    save_x2d(s, r, p)
    back, _, _ = load_x2d(p)
    print('ROUNDTRIP', len(back.nodes))
"""


def _run(blocked: tuple[str, ...]):
    return subprocess.run(
        [sys.executable, '-c', _SCRIPT, ','.join(blocked)],
        capture_output=True, text=True,
        env={'PYTHONPATH': str(SRC), 'PATH': '/usr/bin:/bin'})


class TestTheCoreInstallIsEnough(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.proc = _run(EXTRAS)

    def test_the_package_imports_without_them(self):
        self.assertEqual(self.proc.returncode, 0, self.proc.stderr[-2000:])

    def test_a_model_solves_without_them(self):
        """20 kN/m over 6 m, carried by two supports.

        Self weight is not in it: add_load_case defaults to
        self_weight_factor=0.0, so the unit weight given to the material is
        declared and not applied. I expected 142.5 here and the engine was
        right.
        """
        self.assertIn('REACTION', self.proc.stdout, self.proc.stderr[-2000:])
        value = float(self.proc.stdout.split('REACTION')[1].split()[0])
        self.assertAlmostEqual(value, 20.0 * 6.0, places=6)

    def test_a_model_saves_and_loads_without_them(self):
        self.assertIn('ROUNDTRIP 2', self.proc.stdout, self.proc.stderr[-2000:])


class TestTheBlockerActuallyBlocks(unittest.TestCase):
    """Without this the suite above passes on a machine that simply has the
    packages, which is every developer machine and therefore the one case
    where a green result means nothing."""

    def test_a_blocked_module_really_cannot_be_imported(self):
        proc = subprocess.run(
            [sys.executable, '-c',
             _SCRIPT.replace('import xdfem2d\n', 'import matplotlib\n', 1),
             ','.join(EXTRAS)],
            capture_output=True, text=True,
            env={'PYTHONPATH': str(SRC), 'PATH': '/usr/bin:/bin'})
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn('simulated', proc.stderr)


class TestTheExtrasAreDeclared(unittest.TestCase):

    def setUp(self):
        try:
            import tomllib
        except ModuleNotFoundError:
            try:
                import tomli as tomllib
            except ModuleNotFoundError:
                self.skipTest('no TOML reader')
        with open(ROOT / 'pyproject.toml', 'rb') as fh:
            self.data = tomllib.load(fh)

    def test_the_core_is_only_what_solving_needs(self):
        core = {d.split('>')[0].split('[')[0].strip()
                for d in self.data['project']['dependencies']}
        self.assertEqual(core, {'numpy', 'scipy', 'eurocodepy'})

    def test_every_extra_module_is_behind_an_extra(self):
        declared = set()
        for names in self.data['project']['optional-dependencies'].values():
            declared |= {d.split('>')[0].split('[')[0].strip() for d in names}
        # python-docx is imported as `docx`; the rest match their import name.
        self.assertLessEqual({'matplotlib', 'openpyxl', 'python-docx', 'ezdxf'},
                             declared)

    def test_there_is_one_name_that_installs_everything(self):
        """So a user who does not want to reason about it does not have to."""
        self.assertIn('all', self.data['project']['optional-dependencies'])


if __name__ == '__main__':
    unittest.main()
