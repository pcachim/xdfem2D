"""Version of the ``.x2d`` container, and the conversions between versions.

A ``.x2d`` carries a ``manifest.json`` with ``format_version``, a whole number
(1, 2, 3, ...) that goes up when a file written by the new version could not be
read as it is by the old one. A file without a manifest was written before
versions existed and is version 1.

Reading goes one way only: the program reads its own version and every older
one, never a newer one (it refuses, with a message, rather than guess). There is
one reader, of the current version; an older file is first brought up to date by
a chain of small conversions ``N -> N+1`` on the parsed JSON, before anything is
built from it. To add a version:

1. raise :data:`FORMAT_VERSION`;
2. write a function decorated with ``@migration(old_version)`` that changes the
   :class:`Documents` in place (see :meth:`Documents.models` for the models that
   sit inside ``variants.json`` and ``snapshots.json``);
3. freeze a file of the old version under ``tests/data/formats/v<old>/``.

A change that only *adds* a field, with a default that means what the old file
meant, needs no new version: the reader tolerates its absence. A change of a
name, a structure, a meaning, or of a default that is no longer equivalent, does.
"""
from __future__ import annotations

import json
from typing import Callable, Iterator

#: The version this program writes, and the newest it reads.
FORMAT_VERSION = 1

MANIFEST = 'manifest.json'
#: The version of a file with no manifest (written before versions existed).
UNVERSIONED = 1

#: The JSON entries a conversion may change. ``results.json`` is not among them:
#: results follow the model they were computed on, so a conversion that changes
#: the model sets :attr:`Documents.drop_results` instead of converting them.
DOCUMENT_ENTRIES = ('structure.json', 'variants.json', 'sequences.json',
                    'snapshots.json', 'beam_detail.json')


class X2DFormatError(ValueError):
    """A ``.x2d`` the program cannot read because of its format version."""


class Documents:
    """The JSON entries of a ``.x2d`` that conversions work on, parsed.

    ``entries`` maps the entry name to its parsed JSON; an entry the file does
    not have, or that could not be parsed, is absent."""

    def __init__(self, entries: dict | None = None):
        self.entries: dict = dict(entries or {})
        #: Set by a conversion that changes the model: the stored results no
        #: longer describe it and are not loaded.
        self.drop_results = False
        #: One line per conversion applied, for the user.
        self.notes: list[str] = []

    def models(self) -> Iterator[dict]:
        """Every model dict in the file: the main one, the own model of each
        variant and the model of each snapshot. A conversion of the model
        applies to all of them."""
        s = self.entries.get('structure.json')
        if isinstance(s, dict):
            yield s
        v = self.entries.get('variants.json')
        if isinstance(v, dict):
            for var in v.get('variants') or []:
                if isinstance(var, dict) and isinstance(var.get('model'), dict):
                    yield var['model']
        for sn in self.entries.get('snapshots.json') or []:
            if isinstance(sn, dict) and isinstance(sn.get('model'), dict):
                yield sn['model']


_MIGRATIONS: dict[int, Callable[[Documents], None]] = {}


def migration(from_version: int):
    """Register the conversion of a file of ``from_version`` to the next one."""
    def register(fn: Callable[[Documents], None]):
        if from_version in _MIGRATIONS:
            raise ValueError(f"two conversions from version {from_version}")
        _MIGRATIONS[from_version] = fn
        return fn
    return register


def migrate(docs: Documents, from_version: int,
            to_version: int | None = None) -> Documents:
    """Bring *docs* from ``from_version`` up to ``to_version`` (default: the
    current one) by applying the conversions in order."""
    target = FORMAT_VERSION if to_version is None else to_version
    for v in range(from_version, target):
        fn = _MIGRATIONS.get(v)
        if fn is None:
            raise X2DFormatError(
                f"This file is of format version {from_version} and there is "
                f"no conversion from version {v}.")
        fn(docs)
        docs.notes.append(f"converted from format version {v} to {v + 1}")
    return docs


def manifest_bytes(entries: list[str]) -> bytes:
    """The ``manifest.json`` of a file about to be written."""
    from . import __version__
    return json.dumps({'format': 'x2d', 'format_version': FORMAT_VERSION,
                       'app_version': __version__,
                       'entries': sorted(entries)}, indent=2).encode()


def read_version(zf) -> int:
    """The format version of the open archive *zf*. Raises
    :class:`X2DFormatError` for a manifest that cannot be read, or for a version
    newer than this program's."""
    if MANIFEST not in zf.namelist():
        return UNVERSIONED
    try:
        m = json.loads(zf.read(MANIFEST))
        version = m['format_version']
    except (ValueError, KeyError, TypeError) as e:
        raise X2DFormatError(f"manifest.json cannot be read ({e}).") from e
    if isinstance(version, bool) or not isinstance(version, int) or version < 1:
        raise X2DFormatError(
            f"manifest.json has a format_version that is not a whole number "
            f"from 1: {version!r}.")
    if version > FORMAT_VERSION:
        who = m.get('app_version')
        raise X2DFormatError(
            f"This file is of format version {version}, written by "
            f"{'xdfem2D ' + str(who) if who else 'a newer xdfem2D'}; this "
            f"program reads up to version {FORMAT_VERSION}. Update xdfem2D to "
            "open it.")
    return version
