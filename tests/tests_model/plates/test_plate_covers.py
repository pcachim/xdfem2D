"""Phase 9A — per-face/direction covers for slab reinforcement.

A slab is reinforced top and bottom, each in x and y, at different depths, so a
plate section carries four covers (top/bottom × x/y). Each is optional and falls
back to the single ``rc_cover`` — so a plane model, or any model that only sets
the single cover, is unchanged. These check the model, the resolver, and the
round trips (file and script).
"""
import context  # noqa: F401

from xdfem2d import Structure2D
from xdfem2d.structure_io import _from_dict, _to_dict
from xdfem2d import script_check
from xdfem2d.script_export import to_python


def _plate():
    s = Structure2D(domain='plate')
    s.add_material('C', 30e6, 25.0, poisson=0.2)
    return s


# ── The resolver ────────────────────────────────────────────────────────────

def test_single_cover_propagates_to_four():
    s = _plate()
    s.add_tri_section('S', 'C', thickness=0.2, formulation='DKT', rc_cover=0.05)
    assert s.tri_sections['S'].resolved_covers() == {
        'top_x': 0.05, 'top_y': 0.05, 'bot_x': 0.05, 'bot_y': 0.05}


def test_four_distinct_covers_are_kept():
    s = _plate()
    s.add_plate_section('S', 'C', thickness=0.25, rc_cover=0.05,
                        rc_cover_top_x=0.03, rc_cover_top_y=0.04,
                        rc_cover_bot_x=0.035, rc_cover_bot_y=0.045)
    assert s.tri_sections['S'].resolved_covers() == {
        'top_x': 0.03, 'top_y': 0.04, 'bot_x': 0.035, 'bot_y': 0.045}


def test_partial_covers_fall_back_to_the_single_one():
    s = _plate()
    # Only the bottom-x cover set; the rest fall back to rc_cover.
    s.add_plate_section('S', 'C', thickness=0.25, rc_cover=0.05,
                        rc_cover_bot_x=0.03)
    c = s.tri_sections['S'].resolved_covers()
    assert c == {'top_x': 0.05, 'top_y': 0.05, 'bot_x': 0.03, 'bot_y': 0.05}


# ── Round trips ─────────────────────────────────────────────────────────────

def test_file_round_trip_preserves_four_covers():
    s = _plate()
    s.add_plate_section('S', 'C', thickness=0.25,
                        rc_cover_top_x=0.03, rc_cover_top_y=0.04,
                        rc_cover_bot_x=0.035, rc_cover_bot_y=0.045)
    s2 = _from_dict(_to_dict(s))
    assert s2.tri_sections['S'].resolved_covers() == \
        s.tri_sections['S'].resolved_covers()


def test_script_round_trip_and_check_clean():
    s = _plate()
    s.add_plate_section('S', 'C', thickness=0.25,
                        rc_cover_top_x=0.03, rc_cover_bot_y=0.045)
    # A minimal, actually-supported model: check_script's completeness rule
    # flags any whole (build()-shaped) script with no support as a mechanism,
    # and _plate() alone has no nodes/elements/supports at all.
    s.add_node('N1', 0.0, 0.0)
    s.add_node('N2', 1.0, 0.0)
    s.add_node('N3', 0.0, 1.0)
    s.add_tri_element('T1', 'N1', 'N2', 'N3', 'S')
    s.add_support('FIX', w=True, tx=True, ty=True)
    s.assign_support('N1', 'FIX')
    src = to_python(s)
    assert script_check.check(src) == []
    ns = {}
    exec(compile(src, '<gen>', 'exec'), ns)          # noqa: S102
    s2 = ns['build']()
    assert s2.tri_sections['S'].resolved_covers() == \
        s.tri_sections['S'].resolved_covers()


def test_legacy_file_without_the_keys_still_loads():
    s = _plate()
    s.add_tri_section('W', 'C', thickness=0.1, formulation='DKT', rc_cover=0.05)
    d = _to_dict(s)
    for k in ('rc_cover_top_x', 'rc_cover_top_y',
              'rc_cover_bot_x', 'rc_cover_bot_y'):
        d['tri_sections'][0].pop(k, None)            # an older file's shape
    s2 = _from_dict(d)
    assert s2.tri_sections['W'].resolved_covers() == {
        'top_x': 0.05, 'top_y': 0.05, 'bot_x': 0.05, 'bot_y': 0.05}
