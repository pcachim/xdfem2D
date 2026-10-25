"""Malformed span distributions degrade visibly (log + placeholder), not silently."""
import logging

import context  # noqa: F401  (adds src/ to sys.path)

from xdfem2d import rc_design, report_io


def test_critical_points_bad_distribution_warns_and_uses_ends(caplog):
    ef = {'i': [1.0, 2.0, 3.0], 'j': [1.0, 2.0, -3.0]}
    with caplog.at_level(logging.WARNING, logger="xdfem2d.rc_design"):
        pts = rc_design._critical_points(ef, {'N': [1.0]})   # no 'V', 'M', 'x'
    assert [p[0] for p in pts] == ['i', 'j']
    assert any("span distribution unusable" in r.getMessage()
               for r in caplog.records)


def test_critical_points_good_distribution_adds_interior_points(caplog):
    ef = {'i': [0.0, 5.0, 0.0], 'j': [0.0, -5.0, 0.0]}
    dist = {'x': [0.0, 1.0, 2.0], 'N': [0.0] * 3, 'V': [5.0, 0.0, -5.0],
            'M': [0.0, 6.0, 0.0]}
    with caplog.at_level(logging.WARNING, logger="xdfem2d.rc_design"):
        pts = rc_design._critical_points(ef, dist)
    assert not caplog.records
    assert any(p[1] == 'Asxb,max' for p in pts)


def test_span_extremes_keeps_bar_row_when_distribution_is_bad(caplog):
    results = {'analysis_cases': {'G': {'element_distribution': {
        'B1': {'N': [0.0, 0.0], 'V': [1.0, -1.0], 'M': [0.0, 2.0]},
        'B2': {'N': [0.0]},                                  # malformed
    }}}}
    with caplog.at_level(logging.WARNING, logger="xdfem2d.report_io"):
        tables = report_io._results_tables(results)
    rows = dict((t[0], t[2]) for t in tables)["Span extremes"]
    by_bar = {r[1]: r for r in rows}
    assert by_bar['B1'][2:] == [0.0, 1.0, 2.0, 0.0]
    assert by_bar['B2'][2:] == ["n/a"] * 4
    assert any("B2" in r.getMessage() for r in caplog.records)
