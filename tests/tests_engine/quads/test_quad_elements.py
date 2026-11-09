"""Q4 / QM6 membrane quadrilateral kernel tests (dev/IMPLEMENT_QUAD.md Phase 1).

Pure kernel tests, independent of ``Structure2D`` — Phase 1 adds no data
model or assembly wiring, only ``quad_elements.py``'s stiffness/stress
functions (mirrors how ``tri_elements_dkt.py`` etc. is unit-tested at the
kernel level before ``assembly.py`` ever calls it).

Three checks, per the phase plan:

* constant-stress patch test (exact for both Q4 and QM6 — the non-negotiable
  check for any incompatible-mode element);
* pure-bending cantilever vs. beam theory (QM6 close/exact, plain Q4 visibly
  too stiff — the test that justifies choosing QM6 as the default, per
  dev/IMPLEMENT_QUAD.md §1);
* mesh convergence of the same cantilever, showing Q4's error shrinking with
  refinement while QM6 stays essentially exact throughout.
"""
import unittest

import numpy as np

from context import assert_close  # noqa: F401  (adds src/ to sys.path)
from xdfem2d.quad_elements import (
    q4_stiffness, qm6_stiffness,
    q4_stress_at_centroid, qm6_stress_at_centroid,
)

E = 30.0e9      # Pa
NU = 0.2
T = 0.1         # m


def _patch_test_displacements(coords, ex0, ey0, gxy0):
    """Nodal displacement vector for the linear field consistent with the
    constant strain state (ex0, ey0, gxy0), the standard single-element
    patch-test loading."""
    ue = []
    for x, y in coords:
        u = ex0 * x + 0.5 * gxy0 * y
        v = ey0 * y + 0.5 * gxy0 * x
        ue += [u, v]
    return np.array(ue)


def _cantilever_tip_deflection(formulation: str, n: int, L: float, h: float,
                                M_tip: float) -> float:
    """Assemble a 1-row x n-column rectangular mesh of quad elements by hand
    (Phase 1 has no Structure2D/assembly wiring yet), fix the left edge, load
    the tip with an equal-and-opposite Fx pair producing a pure end moment
    M_tip, solve, and return the tip's centroidal vertical deflection."""
    kernel = q4_stiffness if formulation == 'q4' else qm6_stiffness
    nnodes = 2 * (n + 1)
    ndof = 2 * nnodes
    K = np.zeros((ndof, ndof))

    def dof(node):
        return 2 * node

    for e in range(n):
        ni, nj, nk, nl = 2 * e, 2 * e + 2, 2 * e + 3, 2 * e + 1
        x0, x1 = e * L / n, (e + 1) * L / n
        coords = [(x0, 0.0), (x1, 0.0), (x1, h), (x0, h)]
        k, _gauss, _D, _area = kernel(coords, E, NU, T)
        dofs = []
        for nd in (ni, nj, nk, nl):
            dofs += [dof(nd), dof(nd) + 1]
        for a in range(8):
            for b in range(8):
                K[dofs[a], dofs[b]] += k[a, b]

    fixed = [dof(0), dof(0) + 1, dof(1), dof(1) + 1]
    free = [i for i in range(ndof) if i not in fixed]

    F = np.zeros(ndof)
    bot, top = 2 * n, 2 * n + 1
    Fx = M_tip / h                          # equal & opposite -> moment M_tip
    F[dof(top)] = Fx
    F[dof(bot)] = -Fx

    u = np.zeros(ndof)
    u[free] = np.linalg.solve(K[np.ix_(free, free)], F[free])
    return 0.5 * (u[dof(top) + 1] + u[dof(bot) + 1])


class TestQ4QM6PatchTest(unittest.TestCase):
    """Single-element constant-stress patch test on a deliberately distorted
    (non-rectangular) quad — the Taylor–Wilson consistency correction in the
    incompatible modes is exactly what this guards against failing."""

    coords = [(0.0, 0.0), (3.0, 0.2), (2.8, 2.5), (-0.3, 2.1)]
    ex0, ey0, gxy0 = 3.0e-4, -1.5e-4, 2.0e-4

    def test_q4_recovers_constant_stress(self):
        self._check('q4', q4_stiffness, q4_stress_at_centroid)

    def test_qm6_recovers_constant_stress(self):
        self._check('qm6', qm6_stiffness, qm6_stress_at_centroid)

    def _check(self, name, kernel, stress_fn):
        ue = _patch_test_displacements(self.coords, self.ex0, self.ey0, self.gxy0)
        k, gauss, D, _area = kernel(self.coords, E, NU, T)
        sig0 = D @ np.array([self.ex0, self.ey0, self.gxy0])

        res = stress_fn(self.coords, E, NU, T, ue)
        for key, expected in (('sx', sig0[0]), ('sy', sig0[1]), ('txy', sig0[2])):
            assert_close(self, res[key], expected, rel=1e-9, abs_tol=1e-3,
                         msg=f"{name} centroid {key}")

        # Every Gauss point, not just the centroid, must see the same strain.
        for xi, eta, B, detJ in gauss:
            eps = B @ ue
            for i, expected in enumerate((self.ex0, self.ey0, self.gxy0)):
                assert_close(self, eps[i], expected, rel=1e-9, abs_tol=1e-9,
                             msg=f"{name} strain[{i}] at gauss ({xi:.3f},{eta:.3f})")

        # Stiffness matrix must be symmetric and give zero force under a
        # rigid-body translation.
        self.assertLessEqual(np.max(np.abs(k - k.T)), 1e-6 * np.max(np.abs(k)))
        rigid = np.tile([1.0, 0.0], 4)
        self.assertLessEqual(np.max(np.abs(k @ rigid)), 1e-6 * np.max(np.abs(k)))


class TestQ4QM6PureBending(unittest.TestCase):
    """Cantilever under a pure tip moment (no shear) — the classic
    demonstration of Q4 bending (shear) locking vs. QM6's fix. Rectangular
    mesh, 1 element through the depth, so QM6's incompatible modes are exact
    per element for this constant-curvature state."""

    L, H = 2.0, 0.2
    M_TIP = 1000.0

    def _exact(self):
        I = T * self.H ** 3 / 12.0
        return self.M_TIP * self.L ** 2 / (2.0 * E * I)

    def test_qm6_matches_beam_theory(self):
        exact = self._exact()
        for n in (2, 4, 8):
            d = _cantilever_tip_deflection('qm6', n, self.L, self.H, self.M_TIP)
            assert_close(self, abs(d), exact, rel=1e-6,
                         msg=f"QM6 tip deflection, n={n}")

    def test_q4_locks_and_underpredicts(self):
        exact = self._exact()
        for n in (2, 4, 8):
            d = _cantilever_tip_deflection('q4', n, self.L, self.H, self.M_TIP)
            ratio = abs(d) / exact
            self.assertLess(ratio, 0.7,
                             f"Q4 should visibly lock (n={n}): ratio={ratio:.3f}")

    def test_q4_error_shrinks_with_refinement(self):
        """Q4's locking is a per-element artefact of the coarse bilinear
        field along the beam axis; refining along the length (more, shorter
        elements) reduces — but does not eliminate as fast as QM6 does — the
        parasitic-shear error."""
        exact = self._exact()
        ratios = [abs(_cantilever_tip_deflection('q4', n, self.L, self.H,
                                                  self.M_TIP)) / exact
                  for n in (2, 4, 8)]
        self.assertLess(ratios[0], ratios[1])
        self.assertLess(ratios[1], ratios[2])


if __name__ == '__main__':
    unittest.main()
