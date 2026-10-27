"""Script execution engine for the xdfem2D script editor.

This module is intentionally Qt-free so it can be tested without a display.
The application's script-editor dialog imports from here and wraps
the worker in a QThread.

Public API
----------
SAFE_BUILTINS : dict
    Restricted __builtins__ namespace used when exec-ing user scripts.
run_script(code, struc) -> (Structure2D, str)
    Execute *code* in the restricted namespace.  Returns the resulting
    Structure2D and any text printed by the script.  Raises on syntax
    error or runtime exception.
"""
from __future__ import annotations

import copy
import io

__all__ = ['SAFE_BUILTINS', 'run_script']


# ---------------------------------------------------------------------------
# Restricted builtins
# ---------------------------------------------------------------------------

def _make_safe_builtins() -> dict:
    """A reduced __builtins__ dict for user-script execution.

    Includes builtins that are useful in geometry/model scripts and harmless.
    Excludes open, exec, eval, compile and anything that touches the file
    system or gives access to interpreter internals.

    ``__import__`` is included because Python's import machinery needs it to
    execute ``from xdfem2d import Structure2D`` statements inside a script.
    Direct calls to ``__import__()`` are blocked earlier by check_editor() at
    the AST level before exec is ever reached.
    """
    safe_names = {
        'abs', 'all', 'any', 'ascii', 'bin', 'bool', 'bytes',
        'callable', 'chr', 'complex', 'dict', 'divmod',
        'enumerate', 'filter', 'float', 'format', 'frozenset',
        'getattr', 'setattr', 'delattr',
        'hasattr', 'hash', 'hex', 'int', 'isinstance', 'issubclass',
        'iter', 'len', 'list', 'map', 'max', 'min', 'next', 'object',
        'oct', 'ord', 'pow', 'print', 'range', 'repr', 'reversed',
        'round', 'set', 'slice', 'sorted', 'str', 'sum', 'super',
        'tuple', 'type', 'zip',
        # Exceptions — needed for try/except in scripts
        'Exception', 'ValueError', 'TypeError', 'KeyError',
        'IndexError', 'AttributeError', 'RuntimeError', 'StopIteration',
        'NotImplementedError', 'ArithmeticError', 'ZeroDivisionError',
        'True', 'False', 'None',
    }
    import builtins as _b
    result = {name: getattr(_b, name)
              for name in safe_names if hasattr(_b, name)}
    result['__import__'] = _b.__import__
    # Shiboken (PySide6) patches the real builtins with ``__orig_import__``
    # when it installs its import hook.  Without it in our restricted dict
    # any ``import`` statement executed from a QThread causes a fatal crash:
    #   libshiboken: builtins has no "__orig_import__" function
    if hasattr(_b, '__orig_import__'):
        result['__orig_import__'] = _b.__orig_import__
    return result


SAFE_BUILTINS: dict = _make_safe_builtins()


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------

def run_script(code: str, struc=None) -> tuple:
    """Execute *code* in the restricted namespace.

    Parameters
    ----------
    code:
        Python source to execute.
    struc:
        Current Structure2D.  Used as the base for snippet mode (deep-copied
        before injection so a failing snippet never mutates the original).

    Returns
    -------
    (Structure2D, captured_stdout: str)

    Raises
    ------
    SyntaxError
        If the script is not valid Python.
    Any exception raised by the script itself.
    """
    import math
    import itertools
    import functools

    # A model's own decoding sometimes substitutes a typographic hyphen/minus
    # or curly quote for the ASCII a Python parser expects (see
    # xdfem2d.script_check.normalize_source for the transcript that found
    # this and the full rationale). check_increment/check already apply the
    # same normalization before reporting problems; without it here too, a
    # snippet that "checks out" could still fail to compile when actually
    # run, on a character the check no longer sees as a problem.
    # is_build_mode is the single source of truth in xdfem2d.script_check —
    # ScriptEditorDialog._run needs the exact same decision (which of check()/
    # check_increment() to validate with) and used to have no way to make it,
    # always running the whole-script check even on a plain increment.
    from xdfem2d.script_check import normalize_source, is_build_mode as _is_build_mode
    code = normalize_source(code)

    buf = io.StringIO()
    build_mode = _is_build_mode(code)

    ns: dict = {
        '__builtins__': SAFE_BUILTINS,
        '__name__': '<script>',   # ensures if __name__ == '__main__': is skipped
        'math': math,
        'itertools': itertools,
        'functools': functools,
        'print': lambda *a, **kw: print(*a, **kw, file=buf),
    }

    # Lazy imports — keep this module importable even if xdfem2d or numpy are
    # not yet on sys.path (e.g. during early bootstrap or testing).
    try:
        from xdfem2d import Structure2D
        import xdfem2d as _xd
        ns['Structure2D'] = Structure2D
        ns['xdfem2d'] = _xd
    except ImportError:
        pass

    try:
        import numpy as _np
        ns['numpy'] = _np
        ns['np'] = _np
    except ImportError:
        pass

    code_obj = compile(code, '<script>', 'exec')   # raises SyntaxError

    if build_mode:
        exec(code_obj, ns)  # noqa: S102
        result = ns['build']()
    else:
        from xdfem2d import Structure2D as _S2D
        ns['model'] = copy.deepcopy(struc) if struc is not None else _S2D()
        exec(code_obj, ns)  # noqa: S102
        result = ns['model']

    return result, buf.getvalue()
