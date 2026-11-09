"""One material per class: Add material, the templates and create_*_material
all go through databases.material_from_grade."""
import context  # noqa: F401  (adds src/ to sys.path)
import pytest

from xdfem2d import Structure2D  # noqa: E402
from xdfem2d.databases import material_from_grade  # noqa: E402

KEYS = ("elastic_modulus", "unit_weight", "alpha", "poisson")


def _same(m, p):
    assert all(getattr(m, k) == p[k] for k in KEYS)
    assert m.material_type.value == p["material_type"]
    assert m.design == p["design"]


def test_timber_class_carries_its_strengths():
    p = material_from_grade("Timber", "GL24h")
    assert p["design"] == {"class": "GL24h", "fmk": 24.0, "fvk": 3.5,
                           "fc0k": 24.0, "ft0k": 19.2, "rhok": 385.0}


def test_steel_unit_weight_is_785_and_class_values_follow_the_class():
    p = material_from_grade("Steel", "S355")
    assert p["unit_weight"] == 78.5
    assert (p["design"]["fy"], p["design"]["fu"]) == (355.0, 490.0)


def test_concrete_takes_the_reinforcement_class():
    p = material_from_grade("Concrete", "C20/25", reinforcement="B500B")
    assert p["design"] == {"class_conc": "C20/25", "fck": 20.0,
                           "class_reinf": "B500B", "fyk": 500.0}


@pytest.mark.parametrize("mt, grade", [("Concrete", "C25/30"),
                                       ("Steel", "S355"), ("Timber", "C24")])
def test_create_material_is_the_same_material(mt, grade):
    s = Structure2D()
    make = {"Concrete": lambda: s.create_rc_material(grade, "A500NR", name="m"),
            "Steel": lambda: s.create_steel_material(grade, name="m"),
            "Timber": lambda: s.create_timber_material(grade, name="m")}[mt]
    _same(make(), material_from_grade(mt, grade))


def test_add_material_naming_a_class_gets_that_classes_strengths():
    s = Structure2D()
    s.add_material("t", 1.0, 1.0, material_type="Timber",
                   design={"class": "GL24h"})
    assert s.materials["t"].design == material_from_grade("Timber", "GL24h")["design"]


def test_no_design_gives_the_default_class_in_full():
    s = Structure2D()
    s.add_material("t", 1.0, 1.0, material_type="Timber")
    assert s.materials["t"].design == material_from_grade("Timber", "C24")["design"]


def test_unknown_class_names_the_valid_ones():
    with pytest.raises(ValueError, match="valid grades: .*S355"):
        material_from_grade("Steel", "S999")
    with pytest.raises(ValueError, match="valid classes: .*C30/37"):
        material_from_grade("Concrete", "C99/99")
