"""DKT4 / MITC4 plate quadrilateral kernel tests (dev/IMPLEMENT_QUAD.md Phase 2).

Pure kernel tests, independent of ``Structure2D`` — same Phase 1/2 scoping as
``test_quad_elements.py``: no data model or assembly wiring yet, only the
``quad_elements_dkt4``/``quad_elements_mitc4`` stiffness/recovery functions.

Three checks, per the phase plan:

* constant-curvature patch test (exact for both — DKT4 by construction from
  already-tested DKT triangles, MITC4 by the same mixed-interpolation
  argument that makes MITC3 pass it);
* simply-supported square plate under UDL vs. the standard Navier/Timoshenko
  table value (alpha=0.00406, nu=0.3);
* MITC4 thin/thick locking check — the deflection ratio (FE / thin-theory
  exact) must stay near 1 as t/L shrinks over 3 orders of magnitude, where a
  naive Mindlin quad would lock and the ratio would collapse toward 0.
"""
import unittest

import numpy as np

from context import assert_close  # noqa: F401  (adds src/ to sys.path)
from xdfem2d.quad_elements_dkt4 import dkt4_stiffness, dkt4_moment_entry
from xdfem2d.quad_elements_mitc4 import mitc4_stiffness, mitc4_moment_entry

E = 30.0e9      # Pa
NU_PATCH = 0.2  # patch test: any nu works, kept distinct from the SS-plate nu
T_PATCH = 0.15


def _kirchhoff_dofs(coords, kx, ky, kxy):
    """Nodal (w, θx, θy) consistent with the constant curvature
    (w,xx, w,yy, 2w,xy) = (-kx, -ky, -kxy) — the sign the B matrices in both
    kernels are built in (κ = [-θy,x, θx,y, θx,x-θy,y] with θx=w,y, θy=-w,x)."""
    u = []
    for x, y in coords:
        w = -0.5 * (kx * x * x + ky * y * y + kxy * x * y)
        dwdx = -(kx * x + 0.5 * kxy * y)
        dwdy = -(ky * y + 0.5 * kxy * x)
        u += [w, dwdy, -dwdx]
    return np.array(u)


class TestDKT4MITC4PatchTest(unittest.TestCase):
    """Constant-curvature patch test on a distorted (non-rectangular) quad."""

    coords = [(0.0, 0.0), (3.0, 0.2), (2.8, 2.5), (-0.3, 2.1)]
    kx, ky, kxy = 2.0e-3, -1.0e-3, 1.5e-3

    def _expected_moments(self):
        Db = (T_PATCH ** 3 / 12.0) * (E / (1.0 - NU_PATCH ** 2)) * np.array([
            [1.0, NU_PATCH, 0.0],
            [NU_PATCH, 1.0, 0.0],
            [0.0, 0.0, (1.0 - NU_PATCH) / 2.0],
        ])
        # actual curvature is (-kx, -ky, -kxy), see _kirchhoff_dofs.
        return Db @ np.array([-self.kx, -self.ky, -self.kxy])

    def test_dkt4_recovers_constant_moment(self):
        self._check(dkt4_moment_entry, 'DKT4')

    def test_mitc4_recovers_constant_moment(self):
        self._check(mitc4_moment_entry, 'MITC4')

    def _check(self, entry_fn, name):
        u12 = _kirchhoff_dofs(self.coords, self.kx, self.ky, self.kxy)
        res = entry_fn(self.coords, E, NU_PATCH, T_PATCH, u12)
        m0 = self._expected_moments()
        for key, expected in (('mx', m0[0]), ('my', m0[1]), ('mxy', m0[2])):
            assert_close(self, res[key], expected, rel=1e-6, abs_tol=1e-3,
                         msg=f"{name} {key}")


def _ss_plate_deflection(N, a, t, E, nu, q, kernel, is_dkt4):
    """Simply-supported (soft, w=0 on the boundary) square plate under a
    lumped UDL, meshed N x N, manually assembled (Phase 1/2 has no
    Structure2D/assembly wiring for quads yet). Returns the centre w."""
    h = a / N
    nn = N + 1

    def node(i, j):
        return i * nn + j

    ndof = 3 * nn * nn
    K = np.zeros((ndof, ndof))
    for i in range(N):
        for j in range(N):
            coords = [(i * h, j * h), ((i + 1) * h, j * h),
                      ((i + 1) * h, (j + 1) * h), (i * h, (j + 1) * h)]
            if is_dkt4:
                k, _Db, _area = kernel(coords, E, nu, t)
            else:
                k, _Bb, _Db, _area = kernel(coords, E, nu, t)
            dofs = []
            for nd in (node(i, j), node(i + 1, j), node(i + 1, j + 1), node(i, j + 1)):
                dofs += [3 * nd, 3 * nd + 1, 3 * nd + 2]
            for a1 in range(12):
                for b1 in range(12):
                    K[dofs[a1], dofs[b1]] += k[a1, b1]

    F = np.zeros(ndof)
    for i in range(nn):
        for j in range(nn):
            wx = 1.0 if 0 < i < N else 0.5
            wy = 1.0 if 0 < j < N else 0.5
            F[3 * node(i, j)] = -q * wx * wy * h * h

    fixed = {3 * node(i, j) for i in range(nn) for j in range(nn)
             if i in (0, N) or j in (0, N)}
    free = [d for d in range(ndof) if d not in fixed]
    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    return u[3 * node(N // 2, N // 2)]


class TestSimplySupportedPlateBenchmark(unittest.TestCase):
    """UDL square plate vs. the classical Navier/Timoshenko table value
    (alpha=0.00406 for nu=0.3, thin plate), N=8 mesh, moderate thickness."""

    A, T, NU, Q, N = 4.0, 0.1, 0.3, 10_000.0, 8
    ALPHA = 0.00406

    def _exact(self):
        D = E * self.T ** 3 / (12.0 * (1.0 - self.NU ** 2))
        return -self.ALPHA * self.Q * self.A ** 4 / D

    def test_mitc4_within_1pct(self):
        w = _ss_plate_deflection(self.N, self.A, self.T, E, self.NU, self.Q,
                                  mitc4_stiffness, is_dkt4=False)
        assert_close(self, w, self._exact(), rel=0.01, msg="MITC4 SS-plate centre w")

    def test_dkt4_within_5pct(self):
        # DKT4 is a coarser composite element (see quad_elements_dkt4 module
        # docstring) — looser tolerance than MITC4's, still a real check.
        w = _ss_plate_deflection(self.N, self.A, self.T, E, self.NU, self.Q,
                                  dkt4_stiffness, is_dkt4=True)
        assert_close(self, w, self._exact(), rel=0.05, msg="DKT4 SS-plate centre w")


class TestMITC4ThinThickLocking(unittest.TestCase):
    """The deflection ratio (FE / thin-theory exact) must stay near 1 as
    a/t grows from 10 (thick — some genuine deviation from thin theory is
    expected there) to 5000 (very thin) — a naive Mindlin quad without the
    MITC shear treatment would instead collapse toward 0 as t shrinks."""

    A, NU, Q, N = 4.0, 0.3, 10_000.0, 8
    ALPHA = 0.00406

    def test_ratio_stable_across_thickness(self):
        ratios = []
        for t in (0.1, 0.02, 0.004, 0.0008):
            D = E * t ** 3 / (12.0 * (1.0 - self.NU ** 2))
            w_exact = -self.ALPHA * self.Q * self.A ** 4 / D
            w = _ss_plate_deflection(self.N, self.A, t, E, self.NU, self.Q,
                                      mitc4_stiffness, is_dkt4=False)
            ratios.append(w / w_exact)
        for r in ratios:
            self.assertGreater(r, 0.9, f"ratios={ratios}")
            self.assertLess(r, 1.1, f"ratios={ratios}")
        # No locking trend: the thinnest case must not be meaningfully worse
        # than the thickest one (a locking element's ratio drops with t).
        self.assertGreater(ratios[-1], ratios[0] - 0.05, f"ratios={ratios}")


if __name__ == '__main__':
    unittest.main()
