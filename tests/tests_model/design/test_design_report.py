"""Renderers for design calculation reports (xdfem2d.design_report).

Pure tests on synthetic report dicts — no Qt, no eurocodepy, no solved model —
so they exercise normalisation, the Markdown / HTML renderers, the selection
filter and the schema-version guard in isolation.
"""
import context  # noqa: F401

from xdfem2d import design_report as dr


def _step(symbol, value, **kw):
    base = dict(symbol=symbol, value=value, unit="—", clause="", expr="",
                latex="", subst="", note="", ok=None)
    base.update(kw)
    return base


def _report(title, steps):
    return {"schema_version": dr.EXPECTED_SCHEMA, "title": title, "meta": {},
            "sections": [{"title": "Sec", "steps": steps}]}


REP = _report("Steel member M1 — EN 1993-1-1 §6.3.3", [
    _step("χ_y", 0.971, clause="EN 1993-1-1 §6.3.1.2",
          latex=r"\chi=\frac{1}{\Phi+\sqrt{\Phi^2-\bar\lambda^2}}"),
    _step("Eq. (6.62)", 0.806, clause="EN 1993-1-1 Eq. 6.62",
          subst="0.149+0.948", ok=True),
])


def test_reports_from_dict_and_list():
    assert list(dr.reports_from({"reports": {"M1": REP}})) == ["M1"]
    assert list(dr.reports_from([{"element": "B3", "report": REP},
                                 {"element": "B4"}])) == ["B3"]
    assert dr.reports_from([{"element": "B4"}]) == {}


def test_render_markdown():
    md = dr.render_markdown({"M1": REP}, "Design report")
    assert "# Design report" in md
    # The step symbol is bolded via LaTeX's own \mathbf{}, not Markdown's
    # **…** — a "**" glued onto a "$" delimiter trips up pandoc/MathJax
    # (see _mathify_bold's docstring in design_report.py).
    assert r"$\mathbf{χ_{y}}$" in md and "✓ verified" in md
    # LaTeX is emitted as a display equation ($$…$$) for math viewers / pandoc.
    assert r"$$ \chi=" in md
    assert isinstance(dr.pandoc_available(), bool)


def test_render_html_has_mathjax_and_latex_and_toc():
    html = dr.render_html({"M1": REP, "M2": REP}, "Design report")
    assert "MathJax" in html
    assert r"\chi" in html                 # latex passed through
    assert "verified" in html
    assert "Contents" in html and 'href="#r0"' in html   # TOC for >1 report


def test_filter_reports_concrete_by_bar():
    reps = {"B3": REP, "B4": REP}

    class _S:
        punch_columns = []
    kept = dr.filter_reports(reps, "concrete", _S(), selected_bars={"B3"})
    assert list(kept) == ["B3"]
    # empty selection → unchanged
    assert dr.filter_reports(reps, "concrete", _S(), selected_bars=set()) == reps


def _report_meta(title, meta):
    r = _report(title, [_step("x", 1.0)])
    r["meta"] = meta
    return r


def test_report_base_id_from_meta():
    # base id is read from the material-specific meta key
    assert dr.report_base_id(_report_meta("m", {"member": "M1"})) == "M1"
    assert dr.report_base_id(_report_meta("e", {"element": "B7"})) == "B7"
    assert dr.report_base_id(_report_meta("c", {"column": "C3"})) == "C3"
    # no id in meta → empty string (caller falls back to the dict key)
    assert dr.report_base_id(_report_meta("x", {})) == ""
    assert dr.report_base_id({}) == ""


def test_per_combination_key_and_merge():
    assert dr.per_combination_key("M1", "ULS1") == "M1 · ULS1"
    r_a = _report_meta("M1 a", {"member": "M1", "combination": "ULS1"})
    r_b = _report_meta("M1 b", {"member": "M1", "combination": "ULS2"})
    r_c = _report_meta("M2 a", {"member": "M2", "combination": "ULS1"})
    merged = dr.merge_per_combination({"ULS1": {"M1": r_a, "M2": r_c},
                                       "ULS2": {"M1": r_b}})
    # one entry per (member, combination), keyed by the composite key
    assert set(merged) == {"M1 · ULS1", "M1 · ULS2", "M2 · ULS1"}
    # the base id is still recoverable from each entry's meta
    assert dr.report_base_id(merged["M1 · ULS2"]) == "M1"
    # empty / missing per-combo dicts are tolerated
    assert dr.merge_per_combination({"ULS1": {}, "ULS2": None}) == {}


def test_reports_from_keys_by_triangle():
    # Concrete-area rows are keyed by their triangle id.
    assert list(dr.reports_from([{"triangle": "T5", "report": REP},
                                 {"triangle": "T6"}])) == ["T5"]


def test_filter_reports_concrete_area_by_triangle():
    reps = {"T5": REP, "T6": REP}

    class _S:
        punch_columns = []
    kept = dr.filter_reports(reps, "concrete_area", _S(), selected_tris={"T5"})
    assert list(kept) == ["T5"]
    # empty triangle selection → unchanged
    assert dr.filter_reports(reps, "concrete_area", _S(),
                             selected_tris=set()) == reps


def test_report_base_id_from_triangle_meta():
    r = _report_meta("area", {"triangle": "T9"})
    assert dr.report_base_id(r) == "T9"


def test_schema_guard():
    bad = dict(REP)
    bad = {**REP, "schema_version": dr.EXPECTED_SCHEMA + 99}
    assert dr.schema_mismatches({"M1": bad}) == ["M1"]
    assert dr.schema_mismatches({"M1": REP}) == []
    # the HTML banner shows up on mismatch
    html = dr.render_html({"M1": bad}, "t")
    assert "different schema version" in html


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
