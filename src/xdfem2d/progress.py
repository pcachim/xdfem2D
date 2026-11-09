"""Progress reporting for the long design loops, without coupling the engine to a GUI.

A caller that wants progress (the app runs the design on a worker thread) installs
a callback with :func:`listen`; the engine wraps its element loops in
:func:`track`. With no listener, :func:`track` is a plain pass-through.

The callback is per thread, so it is only seen by the thread that installed it.
It is called as ``callback(done, total, label)`` and must be cheap and thread-safe
(the app's callback just emits a Qt signal).
"""
from __future__ import annotations

import contextlib
import threading

_state = threading.local()


@contextlib.contextmanager
def listen(callback):
    """Install *callback* for the current thread for the duration of the block."""
    previous = getattr(_state, "callback", None)
    _state.callback = callback
    try:
        yield
    finally:
        _state.callback = previous


def report(done: int, total: int, label: str = "") -> None:
    cb = getattr(_state, "callback", None)
    if cb is not None:
        cb(done, total, label)


def track(items, label: str = ""):
    """Iterate *items*, reporting ``(done, total, label)`` before each one."""
    seq = items if hasattr(items, "__len__") else list(items)
    total = len(seq)
    for i, item in enumerate(seq):
        report(i, total, label)
        yield item
    report(total, total, label)
