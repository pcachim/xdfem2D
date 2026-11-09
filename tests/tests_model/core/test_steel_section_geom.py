"""The steel report's cross-section geometry (xdfem2d.steel_design).

Pure mapping test: the section shape → figure ``section_geom`` used by the design
report's cross-section drawing. No Qt, no eurocodepy, no solved model.
"""
import context  # noqa: F401

from types import SimpleNamespace

from xdfem2d.models import SectionShape
from xdfem2d.steel_design import _steel_section_geom


def _sec(shape, b, h, tw, tf):
    return SimpleNamespace(shape=shape, b=b, h=h, tw=tw, tf=tf)


def test_i_profile_geom():
    g = _steel_section_geom(_sec(SectionShape.I, 0.15, 0.30, 0.0071, 0.0107))
    assert g["shape"] == "I"
    assert g["b"] == 0.15 and g["h"] == 0.30
    assert g["tw"] == 0.0071 and g["tf"] == 0.0107


def test_rhs_vs_shs():
    rhs = _steel_section_geom(
        _sec(SectionShape.RECTANGULAR_HOLLOW, 0.20, 0.10, 0.01, 0.01))
    shs = _steel_section_geom(
        _sec(SectionShape.RECTANGULAR_HOLLOW, 0.20, 0.20, 0.01, 0.01))
    assert rhs["shape"] == "RHS"
    assert shs["shape"] == "SHS"       # square hollow when b == h


def test_chs_carries_diameter():
    g = _steel_section_geom(
        _sec(SectionShape.CIRCULAR_HOLLOW, 0.219, 0.219, 0.01, 0.01))
    assert g["shape"] == "CHS"
    assert g["d"] == 0.219


if __name__ == "__main__":
    import sys
    import pytest
    sys.exit(pytest.main([__file__, "-q"]))
