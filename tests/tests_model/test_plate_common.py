"""Phase 8A — the shared plate machinery (xdfem2d.plate_common).

The thin (DKT) element and the shear-deformable ones (MITC3, later DST) share
their transverse-shear constitutive, the Wood-Armer design moments, and the
assembly of the reported result dict. These check that shared code directly, and
that the DKT — refactored to use it — still reports exactly the same keys and
numbers it did before.
"""
import context  # noqa: F401

import numpy as np

from xdfem2d import model_json as MJ
from xdfem2d.plate_common import (SHEAR_CORRECTION, bending_D, plate_moment_result,
                                  shear_D, wood_armer)
from xdfem2d.tri_elements import plane_D


# ── Shear constitutive ──────────────────────────────────────────────────────

def test_shear_D_is_k_G_t_isotropic():
    E, nu, t = 30e6, 0.2, 0.2
    G = E / (2.0 * (1.0 + nu))
    ds = shear_D(E, nu, t)
    assert np.allclose(ds, np.array([[SHEAR_CORRECTION * G * t, 0.0],
                                     [0.0, SHEAR_CORRECTION * G * t]]))
    # A custom correction factor flows through.
    assert np.isclose(shear_D(E, nu, t, k=1.0)[0, 0], G * t)


def test_bending_D_matches_the_plate_rigidity():
    E, nu, t = 30e6, 0.2, 0.25
    assert np.allclose(bending_D(E, nu, t),
                       (t ** 3 / 12.0) * plane_D(E, nu, plane_strain=False))


# ── Wood-Armer (moved here, still importable from tri_elements_dkt) ──────────

def test_wood_armer_import_path_is_preserved():
    from xdfem2d.tri_elements_dkt import wood_armer as wa_dkt
    assert wa_dkt is wood_armer

def test_wood_armer_simple_and_crossing():
    # Both moments sagging: bottom m* = m + |mxy|; top clamps at ≤ 0 (no top
    # steel needed), so mx_top = my_top = 0.
    assert wood_armer(10.0, 8.0, 2.0) == (12.0, 10.0, 0.0, 0.0)
    # Bottom crossing: mx + |mxy| < 0 → mx_bot = 0, my corrected by mxy²/mx.
    mx_bot, my_bot, mx_top, my_top = wood_armer(-5.0, 20.0, 1.0)
    assert mx_bot == 0.0


# ── The shared result dict ──────────────────────────────────────────────────

def test_plate_moment_result_keys_and_principals():
    r = plate_moment_result(10.0, 2.0, 3.0, 0.5, -0.7, 'MITC3')
    assert set(r) == {'mx', 'my', 'mxy', 'm1', 'm2', 'theta',
                      'vx', 'vy', 'mx_bot', 'my_bot', 'mx_top', 'my_top',
                      'formulation'}
    # Principal moments: mean ± radius.
    c, rad = (10.0 + 2.0) / 2.0, np.hypot((10.0 - 2.0) / 2.0, 3.0)
    assert np.isclose(r['m1'], c + rad) and np.isclose(r['m2'], c - rad)
    # The shears are passed through verbatim (element-provided), and the
    # formulation is stamped so a mixed model can be read.
    assert r['vx'] == 0.5 and r['vy'] == -0.7
    assert r['formulation'] == 'MITC3'


# ── DKT still reports exactly the same shape after the refactor ─────────────

def test_dkt_still_reports_the_full_shared_key_set():
    # Build an explicit DKT slab (the template now defaults to MITC3).
    data = MJ.template('slab', as_text=False)
    for ts in data['tri_sections']:
        ts['formulation'] = 'DKT'
    s = MJ.load(data)
    res = s.calculate()
    ts = res['analysis_cases'][next(iter(res['analysis_cases']))]['tri_stress']
    one = next(iter(ts.values()))
    assert set(one) == {'mx', 'my', 'mxy', 'm1', 'm2', 'theta',
                        'vx', 'vy', 'mx_bot', 'my_bot', 'mx_top', 'my_top',
                        'formulation'}
    assert one['formulation'] == 'DKT'
