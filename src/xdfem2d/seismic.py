"""
Seismic design helpers for xdfem2D.

The EN 1998-1 spectrum ordinates are delegated to ``eurocodepy.ec8`` (a required
dependency) so the formulas live in one place; this module only samples them
into the ``[[T, Sa], ...]`` point list the rest of xdfem2D uses.

GUI-free so it can be used from scripts and tests; the GUI imports from here.
"""
from __future__ import annotations


def _sample_periods(TB, TC, TD, T_max, n):
    """Evenly spaced periods (n+1 points) plus the spectrum corner periods."""
    return sorted(
        {0.0, TB, TC, TD, T_max}
        | {round(i / n * T_max, 5) for i in range(n + 1)}
    )


def ec8_spectrum(ag, S, TB, TC, TD, xi_pct, T_max=4.0, n=80):
    """EC8:2004 horizontal *elastic* response spectrum Se(T).

    Ordinates come from :func:`eurocodepy.ec8.calc_elastic_spectrum`.

    Parameters
    ----------
    ag : design ground acceleration [m/s²]
    S  : soil factor
    TB, TC, TD : spectrum corner periods [s]
    xi_pct : viscous damping ratio [%]
    T_max : largest period to tabulate [s]
    n : number of evenly spaced sample periods (corner periods are added)

    Returns ``[[T, Se], ...]`` with T in s and Se in m/s².
    """
    from eurocodepy.ec8 import calc_elastic_spectrum
    return [[round(T, 6),
             round(calc_elastic_spectrum(T, ag, S, TB, TC, TD, xi_pct), 6)]
            for T in _sample_periods(TB, TC, TD, T_max, n)
            if 0.0 <= T <= T_max + 1e-9]


def ec8_design_spectrum(ag, S, TB, TC, TD, q, beta=0.2, T_max=4.0, n=80):
    """EC8:2004 horizontal *design* response spectrum Sd(T).

    Ordinates come from :func:`eurocodepy.ec8.calc_spectrum`, which uses the
    behaviour factor ``q`` and the lower-bound factor ``beta`` (EN 1998-1
    §3.2.2.5).

    Parameters
    ----------
    ag : design ground acceleration [m/s²]
    S  : soil factor
    TB, TC, TD : spectrum corner periods [s]
    q  : behaviour factor
    beta : lower-bound factor for the horizontal design spectrum (default 0.2)
    T_max : largest period to tabulate [s]
    n : number of evenly spaced sample periods (corner periods are added)

    Returns ``[[T, Sd], ...]`` with T in s and Sd in m/s².
    """
    from eurocodepy.ec8 import calc_spectrum
    return [[round(T, 6),
             round(calc_spectrum(T, ag, S, q, TB, TC, TD, beta), 6)]
            for T in _sample_periods(TB, TC, TD, T_max, n)
            if 0.0 <= T <= T_max + 1e-9]


# Backwards-compatible private aliases (the GUI used these names).
_ec8_spectrum = ec8_spectrum
_ec8_design_spectrum = ec8_design_spectrum
