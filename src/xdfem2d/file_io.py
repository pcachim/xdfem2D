"""
xdfem2D native project file I/O (``.x2d``).

A ``.x2d`` file is a ZIP archive containing:
  * ``manifest.json``  — the format version (see :mod:`xdfem2d.x2d_format`)
  * ``structure.json`` — the model (nodes, elements, loads, …)
  * ``variants.json``  — support sets, variants and variant combinations
  * ``sequences.json`` — construction sequences (phasing)
  * ``beam_detail.json`` — the user's bar decisions per beam (optional)
  * ``results.json``   — optional solved results (diagrams as lists)
  * ``view.json``      — optional GUI display options + selection
  * ``stiffness_<id>.npy`` — optional stored ANLG stiffness matrices

This module is GUI-free so it can be used from scripts and tests. The GUI
imports :func:`save_x2d` / :func:`load_x2d` from here.
"""
from __future__ import annotations

import io
import json
import zipfile
from pathlib import Path

from . import x2d_format

import numpy as np


class NpEncoder(json.JSONEncoder):
    """JSON encoder that converts NumPy arrays/scalars to plain Python types."""

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


#: The date of every entry written (the ZIP format cannot hold an earlier one).
_ZIP_DATE = (1980, 1, 1, 0, 0, 0)

# Backwards-compatible private alias (the GUI used ``_NpEncoder``).
_NpEncoder = NpEncoder


def _results_to_dict(results: dict) -> dict:
    """Deep-copy results, converting NumPy arrays to lists for JSON storage."""
    return json.loads(json.dumps(results, cls=NpEncoder))


def _results_from_dict(data: dict) -> dict:
    """Restore NumPy arrays in element_distribution / combo_distribution,
    including the per-case distributions nested inside analysis_cases and
    combinations (otherwise diagram plotting receives plain lists)."""
    def _restore_dist(dist: dict):
        if not isinstance(dist, dict):
            return
        for elem_d in dist.values():
            if isinstance(elem_d, dict):
                for k in ('x', 'N', 'V', 'M', 'N_min', 'V_min', 'M_min'):
                    if k in elem_d:
                        elem_d[k] = np.array(elem_d[k])

    for key in ('element_distribution', 'combo_distribution'):
        for case_d in data.get(key, {}).values():
            _restore_dist(case_d)

    for container in ('analysis_cases', 'combinations'):
        for case_res in data.get(container, {}).values():
            if isinstance(case_res, dict):
                _restore_dist(case_res.get('element_distribution', {}))

    # Files saved before the punching rows used the same spelling as every
    # other design result ('utilization', not 'utilisation').
    punch = data.get('punching')
    if isinstance(punch, dict):
        for row in punch.get('rows') or []:
            if isinstance(row, dict) and 'utilisation' in row:
                row.setdefault('utilization', row.pop('utilisation'))

    # Files saved before 'Asc' was renamed to 'Nc' (23/09/2026): the name
    # looked like a reinforcement area next to Asx/Asy, but it is a concrete
    # compression force per unit length (kN/m), not an area -- see
    # rc_design.py's design_concrete_membranes/design_concrete_quads.
    tri_reinf = data.get('tri_reinforcement')
    if isinstance(tri_reinf, dict):
        for row in tri_reinf.get('rows') or []:
            if isinstance(row, dict) and 'Asc' in row:
                row.setdefault('Nc', row.pop('Asc'))

    return data


def save_x2d(struc, results: dict | None, path: str | Path,
             view: dict | None = None, object_mesh=None, snapshots=None):
    """Write a ``.x2d`` file (ZIP) with structure.json, optional results.json,
    any ANLG stiffness matrices, and an optional view.json (display options +
    selection).

    When the model has geometry objects and *object_mesh* (the compiled
    Structure2D that produced the results) is given, it is also stored as
    ``object_mesh.json`` — a snapshot tied to the results so that reopening a
    computed file maps the results onto exactly the mesh they were computed on,
    even if the expansion algorithm changes in a later version (Option B).

    *snapshots* (the GUI's named snapshots, ``[{'name', 'time', 'data'}]`` with
    ``data`` the zlib-compressed model JSON, no results) go to
    ``snapshots.json``; ``time`` is the epoch, ``created`` its readable form."""
    from xdfem2d.structure_io import _to_dict, variant_overlays_to_dict
    path = Path(path)
    struct_dict = _to_dict(struc)
    # Variants / support sets / construction sequences / variant combinations
    # live in their own entries (variants.json / sequences.json) rather than
    # inline in structure.json — a variant carries a full nested copy of the
    # model, so keeping it out of structure.json keeps the core model file
    # small and easy to read/diff on its own.
    overlays = variant_overlays_to_dict(struc)
    for key in overlays:
        struct_dict.pop(key, None)
    # The user's bar decisions get their own entry (like variants / sequences)
    # — not results.json (lost with the analysis) nor structure.json.
    detail_dict = struct_dict.pop('beam_detail', None)
    # indent= switches json to its pure-Python encoder, ~10x slower: fine for a
    # small hand-readable model, not for a few thousand mesh nodes/elements.
    _n_items = sum(len(v) for v in struct_dict.values()
                   if isinstance(v, (list, dict)))
    struct_bytes = json.dumps(struct_dict,
                              indent=2 if _n_items <= 3000 else None).encode()
    variants_bytes = json.dumps({
        'support_sets': overlays['support_sets'],
        'variants': overlays['variants'],
        'variant_combinations': overlays['variant_combinations'],
    }, indent=2).encode()
    sequences_bytes = json.dumps({
        'construction_sequences': overlays['construction_sequences'],
    }, indent=2).encode()

    files: list[tuple[str, bytes]] = []

    def put(name: str, data: bytes):
        files.append((name, data))

    put('structure.json', struct_bytes)
    put('variants.json', variants_bytes)
    put('sequences.json', sequences_bytes)
    if snapshots:
        import time as _time, zlib
        put('snapshots.json', json.dumps([{
            'name': sn['name'], 'time': sn['time'],
            'created': _time.strftime('%Y-%m-%d %H:%M:%S',
                                      _time.localtime(sn['time'])),
            'model': json.loads(zlib.decompress(sn['data']))}
            for sn in snapshots]).encode())
    if detail_dict:
        put('beam_detail.json', json.dumps(detail_dict, indent=2).encode())
    if view is not None:
        put('view.json', json.dumps(view, indent=2).encode())
    if (results is not None and object_mesh is not None
            and getattr(struc, 'geometry_objects', None)):
        try:
            from xdfem2d.geo_expand import EXPANSION_VERSION
        except Exception:
            EXPANSION_VERSION = 0
        cache = {'version': EXPANSION_VERSION, 'mesh': _to_dict(object_mesh)}
        put('object_mesh.json', json.dumps(cache).encode())
    if results is not None:
        # Exclude stored_stiffness (NumPy arrays) from the JSON payload, and
        # drop the per-load-case results — load cases are load definitions
        # only; their solved response lives in the matching analysis cases.
        _lc_keys = ('displacements', 'reactions', 'element_forces',
                    'element_distribution', 'spring_forces')
        res_copy = {k: ({} if k in _lc_keys else v)
                    for k, v in results.items() if k != 'stored_stiffness'}
        # Machine data, not meant to be read: no indent (see structure.json).
        put('results.json', json.dumps(_results_to_dict(res_copy),
                                       cls=NpEncoder).encode())

        # Save each ANLG stiffness matrix as a compressed .npy entry
        for anlg_id, K_mat in results.get('stored_stiffness', {}).items():
            buf = io.BytesIO()
            np.save(buf, K_mat)
            put(f'stiffness_{anlg_id}.npy', buf.getvalue())

    # The manifest comes first. Entries carry a fixed date, so saving the same
    # model twice gives the same bytes (a file kept under version control only
    # shows what changed).
    files.insert(0, (x2d_format.MANIFEST,
                     x2d_format.manifest_bytes([n for n, _ in files])))
    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files:
            zi = zipfile.ZipInfo(name, date_time=_ZIP_DATE)
            zi.compress_type = zipfile.ZIP_DEFLATED
            zf.writestr(zi, data)


#: Entries a file can lose without failing to open: the user's bar decisions
#: and the named snapshots. A broken variants.json or sequences.json raises.
_TOLERATED = ('snapshots.json', 'beam_detail.json')


def _read_documents(zf) -> tuple[x2d_format.Documents, int]:
    """Parse the JSON entries conversions work on and bring them up to the
    current format version. Returns the documents and the version the file was
    written in. An entry that is not valid JSON is left out (``structure.json``
    is not optional: its error is raised)."""
    version = x2d_format.read_version(zf)
    names = set(zf.namelist())
    entries = {'structure.json': json.loads(zf.read('structure.json'))}
    for name in x2d_format.DOCUMENT_ENTRIES[1:]:
        if name not in names:
            continue
        try:
            entries[name] = json.loads(zf.read(name))
        except json.JSONDecodeError as e:
            if name not in _TOLERATED:
                raise
            entries[f'{name}#error'] = str(e)
    docs = x2d_format.Documents(entries)
    x2d_format.migrate(docs, version)
    return docs, version


def load_x2d(path: str | Path):
    """Read a ``.x2d`` file. Returns ``(struc, results_or_None, view_or_None)``.
    Any stored stiffness matrices (``stiffness_<id>.npy``) are loaded into
    ``results['stored_stiffness']``.

    A file of an older format version is converted on the way in
    (``struc._format_converted_from`` is that version, ``None`` otherwise); one
    of a newer version raises :class:`xdfem2d.x2d_format.X2DFormatError`."""
    from xdfem2d.structure_io import _from_dict, load_variant_overlays
    path = Path(path)
    with zipfile.ZipFile(path, 'r') as zf:
        docs, version = _read_documents(zf)
        struc = _from_dict(docs.entries['structure.json'])
        struc._format_converted_from = (
            version if version != x2d_format.FORMAT_VERSION else None)
        if 'variants.json' in docs.entries:
            load_variant_overlays(struc, docs.entries['variants.json'])
        if 'sequences.json' in docs.entries:
            load_variant_overlays(struc, docs.entries['sequences.json'])
        if 'beam_detail.json' in docs.entries:
            from xdfem2d.structure_io import _beam_detail_from_dict
            _beam_detail_from_dict(struc, docs.entries['beam_detail.json'])
        elif 'beam_detail.json#error' in docs.entries:
            struc.beam_detail_load_error = (
                f"beam_detail.json: {docs.entries['beam_detail.json#error']}")
        # Named snapshots (GUI): handed over on the structure, like the mesh cache.
        struc._loaded_snapshots = []
        if 'snapshots.json' in docs.entries:
            try:
                import zlib
                struc._loaded_snapshots = [{
                    'name': d['name'], 'time': float(d['time']),
                    'data': zlib.compress(json.dumps(d['model']).encode())}
                    for d in docs.entries['snapshots.json']]
            except Exception:
                struc._loaded_snapshots = []
        view = None
        if 'view.json' in zf.namelist():
            try:
                view = json.loads(zf.read('view.json'))
            except Exception:
                view = None
        results = None
        if 'results.json' in zf.namelist() and not docs.drop_results:
            results = _results_from_dict(json.loads(zf.read('results.json')))
            stored: dict = {}
            for name in zf.namelist():
                if name.startswith('stiffness_') and name.endswith('.npy'):
                    anlg_id = name[len('stiffness_'):-len('.npy')]
                    buf = io.BytesIO(zf.read(name))
                    stored[anlg_id] = np.load(buf)
            if stored:
                results['stored_stiffness'] = stored
        # Compiled-mesh cache (Option B): the exact mesh the stored results were
        # computed on. Attach it transiently so the GUI draws results on it
        # instead of regenerating.
        if 'object_mesh.json' in zf.namelist() and not docs.drop_results:
            try:
                cache = json.loads(zf.read('object_mesh.json'))
                struc._cached_object_mesh = _from_dict(cache['mesh'])
            except Exception:
                struc._cached_object_mesh = None
    return struc, results, view
