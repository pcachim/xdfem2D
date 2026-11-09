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


#: Spectrum sets of ``eurocodepy.ec8.get_spectrum_parameters``: the zones each one has (the table of ag, S, TB, TC, TD and of the
#: importance factors lives in eurocodepy, not here). CEN: the EN 1998-1 values, ag as a fraction of g; PT: the Portuguese annex.
EC8_CODES = {
    'CEN-1': ('EN 1998-1, type 1', ('.1g', '.2g', '.3g', '.4g', '.5g', '.6g', '.7g', '.8g', '.9g', '1_0g')),
    'CEN-2': ('EN 1998-1, type 2', ('.1g', '.2g', '.3g', '.4g', '.5g', '.6g', '.7g', '.8g', '.9g', '1_0g')),
    'PT-1': ('Portugal NA, type 1 (distant)', ('1_1', '1_2', '1_3', '1_4', '1_5', '1_6')),
    'PT-2': ('Portugal NA, type 2 (near)', ('2.1', '2.2', '2.3', '2.4', '2.5')),
}
EC8_IMPORTANCE = ('i', 'ii', 'iii', 'iv')


def ec8_parameters(code, soil, zone, importance='ii'):
    """The parameters of the EC8 spectrum of a code, ground type, zone and importance class.

    ``code`` is a key of :data:`EC8_CODES`, ``soil`` 'A'..'E', ``zone`` one of its zones (``'1.3'`` and ``'1_3'`` are the same),
    ``importance`` 'i'..'iv'. Returns ``{'ag', 'S', 'TB', 'TC', 'TD'}``, ag in m/s² already multiplied by the importance factor
    and S already reduced for the ag as the annex says. The values are those of ``eurocodepy.ec8.get_spectrum_parameters``.
    """
    from eurocodepy.ec8 import get_spectrum_parameters
    if code not in EC8_CODES:
        raise ValueError(f"unknown spectrum code {code!r}: {', '.join(EC8_CODES)}")
    zones = EC8_CODES[code][1]
    key = next((z for z in (zone, str(zone).replace('.', '_', 1)) if z in zones), None)
    if key is None:
        raise ValueError(f"{code} has no zone {zone!r}: {', '.join(zones)}")
    if str(soil).upper() not in 'ABCDE' or len(str(soil)) != 1:
        raise ValueError(f"ground type {soil!r} is not A, B, C, D or E")
    if str(importance).lower() not in EC8_IMPORTANCE:
        raise ValueError(f"importance class {importance!r} is not i, ii, iii or iv")
    S, ag, TB, TC, TD = get_spectrum_parameters(code, str(importance).lower(), str(soil).upper(), key)
    return {'ag': float(ag), 'S': float(S), 'TB': float(TB), 'TC': float(TC), 'TD': float(TD)}


# Backwards-compatible private aliases (the GUI used these names).
_ec8_spectrum = ec8_spectrum
_ec8_design_spectrum = ec8_design_spectrum
