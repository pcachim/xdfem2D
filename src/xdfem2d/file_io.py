"""
xdfem2D native project file I/O (``.x2d``).

A ``.x2d`` file is a ZIP archive containing:
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
             view: dict | None = None, object_mesh=None):
    """Write a ``.x2d`` file (ZIP) with structure.json, optional results.json,
    any ANLG stiffness matrices, and an optional view.json (display options +
    selection).

    When the model has geometry objects and *object_mesh* (the compiled
    Structure2D that produced the results) is given, it is also stored as
    ``object_mesh.json`` — a snapshot tied to the results so that reopening a
    computed file maps the results onto exactly the mesh they were computed on,
    even if the expansion algorithm changes in a later version (Option B)."""
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
    struct_bytes = json.dumps(struct_dict, indent=2).encode()
    variants_bytes = json.dumps({
        'support_sets': overlays['support_sets'],
        'variants': overlays['variants'],
        'variant_combinations': overlays['variant_combinations'],
    }, indent=2).encode()
    sequences_bytes = json.dumps({
        'construction_sequences': overlays['construction_sequences'],
    }, indent=2).encode()

    with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.writestr('structure.json', struct_bytes)
        zf.writestr('variants.json', variants_bytes)
        zf.writestr('sequences.json', sequences_bytes)
        if detail_dict:
            zf.writestr('beam_detail.json',
                        json.dumps(detail_dict, indent=2).encode())
        if view is not None:
            zf.writestr('view.json', json.dumps(view, indent=2).encode())
        if (results is not None and object_mesh is not None
                and getattr(struc, 'geometry_objects', None)):
            try:
                from xdfem2d.geo_expand import EXPANSION_VERSION
            except Exception:
                EXPANSION_VERSION = 0
            cache = {'version': EXPANSION_VERSION, 'mesh': _to_dict(object_mesh)}
            zf.writestr('object_mesh.json', json.dumps(cache).encode())
        if results is not None:
            # Exclude stored_stiffness (NumPy arrays) from the JSON payload, and
            # drop the per-load-case results — load cases are load definitions
            # only; their solved response lives in the matching analysis cases.
            _lc_keys = ('displacements', 'reactions', 'element_forces',
                        'element_distribution', 'spring_forces')
            res_copy = {k: ({} if k in _lc_keys else v)
                        for k, v in results.items() if k != 'stored_stiffness'}
            res_bytes = json.dumps(_results_to_dict(res_copy),
                                   cls=NpEncoder, indent=2).encode()
            zf.writestr('results.json', res_bytes)

            # Save each ANLG stiffness matrix as a compressed .npy entry
            for anlg_id, K_mat in results.get('stored_stiffness', {}).items():
                buf = io.BytesIO()
                np.save(buf, K_mat)
                zf.writestr(f'stiffness_{anlg_id}.npy', buf.getvalue())


def load_x2d(path: str | Path):
    """Read a ``.x2d`` file. Returns ``(struc, results_or_None, view_or_None)``.
    Any stored stiffness matrices (``stiffness_<id>.npy``) are loaded into
    ``results['stored_stiffness']``."""
    from xdfem2d.structure_io import _from_dict, load_variant_overlays
    path = Path(path)
    with zipfile.ZipFile(path, 'r') as zf:
        struct_data = json.loads(zf.read('structure.json'))
        struc = _from_dict(struct_data)
        if 'variants.json' in zf.namelist():
            load_variant_overlays(struc, json.loads(zf.read('variants.json')))
        if 'sequences.json' in zf.namelist():
            load_variant_overlays(struc, json.loads(zf.read('sequences.json')))
        if 'beam_detail.json' in zf.namelist():
            from xdfem2d.structure_io import _beam_detail_from_dict
            try:
                _beam_detail_from_dict(
                    struc, json.loads(zf.read('beam_detail.json')))
            except json.JSONDecodeError as e:
                struc.beam_detail_load_error = f"beam_detail.json: {e}"
        view = None
        if 'view.json' in zf.namelist():
            try:
                view = json.loads(zf.read('view.json'))
            except Exception:
                view = None
        results = None
        if 'results.json' in zf.namelist():
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
        if 'object_mesh.json' in zf.namelist():
            try:
                cache = json.loads(zf.read('object_mesh.json'))
                struc._cached_object_mesh = _from_dict(cache['mesh'])
            except Exception:
                struc._cached_object_mesh = None
    return struc, results, view
