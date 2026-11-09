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

# The windows of the suite start a thread that lists the models of the local Ollama. Left on the real server (the
# developer's own, on 127.0.0.1:11434) that thread answers whenever it likes, and a window destroyed while it is waiting
# aborts Qt ("QThread: Destroyed while thread is still running"): a crash that came and went with the number of windows
# alive. Tests do not talk to a real server, so the host is a closed port and the listing fails at once. The ollama
# client reads OLLAMA_HOST when it is imported, so this has to be here, before any test module is. A test that needs a
# real server sets XDFEM2D_TEST_REAL_OLLAMA=1.
if not os.environ.get("XDFEM2D_TEST_REAL_OLLAMA"):
    os.environ["OLLAMA_HOST"] = "http://127.0.0.1:9"

_ROOT = Path(__file__).resolve().parent.parent      # repo root
_TESTS = Path(__file__).resolve().parent             # tests/

# tests/ is grouped in sub-folders (tests_app/assistant, tests_engine/io, ...) and tools/ in bench/, router/ and
# model/; test and tool modules import their siblings by bare name, so every one of those folders is on sys.path.
# (Only the leaf folders: with tests_app/ itself on the path, tests_app/mcp would hide the ``mcp`` package.)
_dirs = [_ROOT / "src", _TESTS, _TESTS / "helpers", _TESTS / "tests_tools"]
for _top in ("tests_app", "tests_engine", "tests_model"):
    _dirs += [p for p in sorted((_TESTS / _top).rglob("*")) if p.is_dir() and p.name != "__pycache__"
              and p.name != "fixtures"]
_dirs += [_ROOT / "tools", _ROOT / "tools" / "bench", _ROOT / "tools" / "router", _ROOT / "tools" / "router" / "text_router", _ROOT / "tools" / "model", _ROOT / "tools" / "archive"]
for _p in _dirs:
    _s = str(_p)
    if _s not in sys.path:
        sys.path.insert(0, _s)


# ── the real state of the user must not be written by a test run ─────────────────────────────────────────────────────────
# The isolation above covers this process. A test that starts the application or a server in another process, or a tool run from
# a test, can still write the real ~/.xdfem2d/state.json, and the temporary models then pile up in File > Open recent. The run
# says so at the end instead of leaving it for the menu to show.
_REAL_STATE = Path(os.path.expanduser("~")).joinpath(".xdfem2d", "state.json") if os.environ.get("XDFEM2D_TEST_REAL_HOME") \
    else Path(_real_expanduser("~")).joinpath(".xdfem2d", "state.json")


def temporary_recents(state_file) -> set:
    """The recent models of *state_file* that live in a temporary folder (the system one, or the one of a test run)."""
    import json
    try:
        recent = json.loads(Path(state_file).read_text(encoding="utf-8")).get("recent_models", [])
    except (OSError, ValueError, AttributeError):
        return set()
    marks = ("/T/tmp", "/xdfem2d-tests-", "/private/tmp/", "/tmp/", "\\Temp\\")
    return {p for p in recent if isinstance(p, str) and any(m in p for m in marks)}


_RECENTS_BEFORE: set = set()


def pytest_sessionstart(session):
    global _RECENTS_BEFORE
    _RECENTS_BEFORE = temporary_recents(_REAL_STATE)


def pytest_terminal_summary(terminalreporter):
    new = temporary_recents(_REAL_STATE) - _RECENTS_BEFORE
    if new:
        terminalreporter.write_sep("!", "this run wrote temporary models in your real File > Open recent")
        for p in sorted(new):
            terminalreporter.write_line("  " + p)
        terminalreporter.write_line("  (a test or a tool started a process with the real home; "
                                    "~/.xdfem2d/state.json has to be cleaned: File > Open recent > Clear)")
