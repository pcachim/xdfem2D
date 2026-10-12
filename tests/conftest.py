"""Test bootstrap shared by every suite under ``tests/``.

Puts the engine/app source (``src/``) and the shared helpers directory
(``tests/`` itself, which holds ``context.py``) on ``sys.path`` so that the
tests in ``tests_engine/``, ``tests_app/`` and ``tests_model/`` can all do
``import context`` / ``from xdfem2d import …`` regardless of the subfolder they
live in.
"""
import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path

# The suite opens windows, loads model files and saves preferences. Pointed at
# the real home, that wrote the user's own ~/.xdfem2d/preferences.json: pytest
# temp directories ended up in "recent models" and "last used folder". So every
# test run gets a throwaway home, for the paths Python builds from ``~``
# (``Path.home()``, ``os.path.expanduser``). The HOME environment variable is
# left alone on purpose: the macOS keychain is found through it, and the tests
# of the provider keys need the real one. This has to run here, before any
# module computes a path from ``~`` at import (user_config.CONFIG_PATH does).
# XDFEM2D_TEST_REAL_HOME=1 turns it off, for debugging against real settings.
if not os.environ.get("XDFEM2D_TEST_REAL_HOME"):
    # One scratch directory for the whole run, removed when it ends. It is also
    # where every ``tempfile.mkdtemp()`` / ``TemporaryDirectory`` of the suite
    # goes (and pytest's own tmp_path directories), so a test that forgets to
    # clean up after itself no longer leaves anything in the system temp
    # folder: 2,300 such folders had piled up there.
    _SCRATCH = tempfile.mkdtemp(prefix="xdfem2d-tests-")
    atexit.register(shutil.rmtree, _SCRATCH, True)
    tempfile.tempdir = _SCRATCH
    os.environ["TMPDIR"] = _SCRATCH
    _HOME = os.path.join(_SCRATCH, "home")
    os.makedirs(_HOME)
    _real_expanduser = os.path.expanduser

    def _expanduser(path):
        path = os.fspath(path)
        if path == "~" or path.startswith(("~/", "~\\")):
            return _HOME + path[1:]
        return _real_expanduser(path)

    os.path.expanduser = _expanduser
    Path.home = classmethod(lambda cls: Path(_HOME))

_ROOT = Path(__file__).resolve().parent.parent      # repo root
_TESTS = Path(__file__).resolve().parent             # tests/

for _p in (_ROOT / "src", _TESTS):
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)
