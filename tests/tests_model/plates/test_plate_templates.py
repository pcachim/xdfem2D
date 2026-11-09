"""Plate-domain templates: simply supported / clamped slabs, a rib-stiffened
slab, and a beam grillage. Each must build a valid plate model that solves.
"""
import context  # noqa: F401
from xdfem2d.templates import (
    _tpl_slab, _tpl_slab_ribbed, _tpl_beam_grid,
    _tpl_flat_slab, _tpl_slab_on_grade, _tpl_bridge_grillage, _tpl_slab_sector,
    _tpl_slab_ribbed_obj, _tpl_flat_slab_obj, _tpl_slab_on_grade_obj,
    _tpl_slab_sector_obj, _tpl_beam_grid_obj, _tpl_bridge_grillage_obj,
)
from xdfem2d.geo_expand import expand_geometry


def _solved(s):
    return s.domain, s.domain_problems(), s.calculate()


def test_simply_supported_slab_builds_and_solves():
    dom, probs, r = _solved(_tpl_slab('simply'))
    assert dom == 'plate' and probs == []
    assert min(d[0] for d in r['displacements']['SW'].values()) < 0.0   # sags


def test_clamped_slab_is_stiffer_than_simply_supported():
    ss = _tpl_slab('simply', pz=-5.0).calculate()
    cl = _tpl_slab('clamped', pz=-5.0).calculate()
    w_ss = min(d[0] for d in ss['displacements']['SW'].values())
    w_cl = min(d[0] for d in cl['displacements']['SW'].values())
    assert abs(w_cl) < abs(w_ss)          # clamped deflects less


def test_rib_stiffened_slab_has_both_slab_and_beams():
    s = _tpl_slab_ribbed()
    assert s.domain == 'plate'
    assert s.tri_elements and s.bar_elements     # DKT slab + grillage ribs
    assert s.domain_problems() == []
    s.calculate()


def test_beam_grid_is_a_grillage():
    s = _tpl_beam_grid(nx=3, ny=3)
    assert s.domain == 'plate'
    assert s.bar_elements and not s.tri_elements
    assert s.domain_problems() == []
    r = s.calculate()
    assert min(d[0] for d in r['displacements']['SW'].values()) < 0.0


def test_flat_slab_is_supported_only_on_columns():
    s = _tpl_flat_slab(Lx=4.0, Ly=4.0, ncx=2, ncy=2, max_size=2.0)
    assert s.domain == 'plate' and s.tri_elements
    # (ncx+1)×(ncy+1) column supports, all w-only.
    assert len(s.support_assignments) == 9
    # Columns land exactly on the bay lines (0, 4, 8) — not snapped/approximate.
    xs = sorted({round(s.nodes[a.node_id].x, 6) for a in s.support_assignments})
    assert xs == [0.0, 4.0, 8.0]
    assert s.domain_problems() == []
    r = s.calculate()
    assert min(d[0] for d in r['displacements']['SW'].values()) < 0.0


def test_slab_on_grade_stands_on_springs_alone():
    s = _tpl_slab_on_grade(nx=8, ny=8)
    assert s.domain == 'plate'
    assert not s.support_assignments            # no supports — Winkler carries it
    assert s.tri_area_springs                   # elastic foundation present
    assert s.domain_problems() == []
    s.calculate()


def test_bridge_grillage_supported_at_both_abutments():
    s = _tpl_bridge_grillage(ng=4, ncross=6)
    assert s.domain == 'plate'
    assert s.bar_elements and not s.tri_elements
    assert len(s.support_assignments) == 8      # 4 girders × 2 ends
    assert s.domain_problems() == []
    r = s.calculate()
    assert min(d[0] for d in r['displacements']['SW'].values()) < 0.0


def test_slab_sector_annulus_and_full_disc():
    ann = _tpl_slab_sector(rmin=1.0, rmax=5.0, theta_deg=90.0, max_size=0.6)
    assert ann.domain == 'plate' and ann.tri_elements
    assert ann.domain_problems() == []
    ann.calculate()
    # rmin = 0, θ = 360 → a solid full disc as a single closed outline: no
    # duplicated radial seam (each coordinate belongs to exactly one node).
    disc = _tpl_slab_sector(rmin=0.0, rmax=5.0, theta_deg=360.0, max_size=0.8)
    coords = {}
    for nid, n in disc.nodes.items():
        coords.setdefault((round(n.x, 6), round(n.y, 6)), []).append(nid)
    assert all(len(v) == 1 for v in coords.values())   # no coincident nodes
    assert disc.domain_problems() == []
    r = disc.calculate()
    assert min(d[0] for d in r['displacements']['SW'].values()) < 0.0


# ── Object (parametric geometry) variants ───────────────────────────────────
# Each builds from geometry objects (rectangle/polygon surfaces or line grids)
# that mesh at solve time; they must expand cleanly and sag under load.

def _obj_solves_and_sags(s):
    assert s.geometry_objects                 # really built from objects
    assert s.domain == 'plate'
    assert s.domain_problems() == []
    r = s.calculate()
    return min(d[0] for d in r['displacements']['SW'].values())


def test_rib_stiffened_slab_obj_meshes_slab_and_ribs():
    s = _tpl_slab_ribbed_obj()
    compiled, _ = expand_geometry(s)
    assert compiled.tri_elements and compiled.bar_elements   # slab + edge beams
    assert _obj_solves_and_sags(s) < 0.0


def test_flat_slab_obj_is_a_grid_of_rectangle_panels():
    s = _tpl_flat_slab_obj(ncx=2, ncy=2, max_size=1.0)
    assert len(s.geometry_objects) == 4                      # one panel per bay
    # (ncx+1)×(ncy+1) point-column supports, all w-only, no edge propagation.
    assert len(s.support_assignments) == 9
    assert _obj_solves_and_sags(s) < 0.0


def test_slab_on_grade_obj_stands_on_a_surface_area_spring():
    s = _tpl_slab_on_grade_obj(nx=10, ny=10)
    assert not s.support_assignments                         # Winkler carries it
    assert s.surface_area_springs                            # spring on the object
    assert _obj_solves_and_sags(s) < 0.0


def test_slab_sector_obj_annulus_pie_and_disc_have_no_degenerate_tris():
    for kw in (dict(rmin=1.0, rmax=5.0, theta_deg=90.0),          # annulus
               dict(rmin=0.0, rmax=5.0, theta_deg=90.0),          # pie slice
               dict(rmin=0.0, rmax=5.0, theta_deg=360.0, max_size=0.6)):  # disc
        s = _tpl_slab_sector_obj(**kw)
        compiled, _ = expand_geometry(s)

        def _area(t):
            a, b, c = (compiled.nodes[t.node_i], compiled.nodes[t.node_j],
                       compiled.nodes[t.node_k])
            return abs((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) / 2

        assert min(_area(t) for t in compiled.tri_elements) > 1e-9
        assert _obj_solves_and_sags(s) < 0.0


def test_beam_grid_obj_matches_the_explicit_grillage():
    s = _tpl_beam_grid_obj(nx=3, ny=3)
    compiled, _ = expand_geometry(s)
    assert compiled.bar_elements and not compiled.tri_elements
    w_obj = _obj_solves_and_sags(s)
    w_exp = min(d[0] for d in _tpl_beam_grid(nx=3, ny=3)
                .calculate()['displacements']['SW'].values())
    assert abs(w_obj - w_exp) < 1e-9          # identical bar topology


def test_bridge_grillage_obj_matches_the_explicit_grillage():
    s = _tpl_bridge_grillage_obj(ng=4, ncross=6)
    w_obj = _obj_solves_and_sags(s)
    w_exp = min(d[0] for d in _tpl_bridge_grillage(ng=4, ncross=6)
                .calculate()['displacements']['SW'].values())
    assert abs(w_obj - w_exp) < 1e-9
