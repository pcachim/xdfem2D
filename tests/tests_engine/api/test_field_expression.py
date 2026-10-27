"""Field expressions f(x, y): safe evaluation, validation and no silent zero."""
import logging
import math

import context  # noqa: F401  (adds src/ to sys.path)
import pytest

from xdfem2d import models
from xdfem2d.models import check_field_expression, evaluate_field


@pytest.mark.parametrize("expr, x, y, expected", [
    ("x*y", 2.0, 3.0, 6.0),
    ("sin(x) + 0.5*y", 0.5, 2.0, math.sin(0.5) + 1.0),
    ("-x**2 + pi", 2.0, 0.0, -4.0 + math.pi),
    ("max(x, y) if x > 0 else 0", 1.0, 4.0, 4.0),
    ("hypot(x, y)", 3.0, 4.0, 5.0),
])
def test_valid_expressions(expr, x, y, expected):
    check_field_expression(expr)
    assert evaluate_field(expr, x, y) == pytest.approx(expected)


@pytest.mark.parametrize("expr", [
    "().__class__.__bases__[0].__subclasses__()",
    "__import__('os').system('true')",
    "x.real",
    "[1, 2][0]",
    "(lambda: 1)()",
    "open('f')",
    "'text'",
    "x +",
    "unknown(x)",
    "sin(x=1)",
])
def test_unsafe_or_invalid_expressions_rejected(expr):
    with pytest.raises(ValueError):
        check_field_expression(expr)
    assert evaluate_field(expr, 1.0, 1.0) == 0.0


def test_failure_is_not_silent(caplog):
    models._warn_field_once.cache_clear()
    with caplog.at_level(logging.WARNING, logger="xdfem2d.models"):
        assert evaluate_field("log(x - 1)", 1.0, 0.0) == 0.0
        assert evaluate_field("log(x - 1)", 1.0, 0.0) == 0.0
    msgs = [r.getMessage() for r in caplog.records]
    assert len(msgs) == 1 and "log(x - 1)" in msgs[0]
