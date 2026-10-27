"""apply_increment joins a create_node that lands on an existing node, within
the coincident-node tolerance, so the increment's bars connect to the model."""
import unittest

from context import Structure2D
from xdfem2d.script_check import apply_increment

CODE = """n1 = model.create_node(0.0, 0.0)
n2 = model.create_node(6.0, 4.0)
sec = model.create_rc_section('S1', b=0.3, h=0.5)
model.create_bar_element(sec, n1, n2)
"""


def _base():
    s = Structure2D()
    s.add_node("N0", 0, 0)
    s.add_node("N1", 6, 0)
    return s


class WeldOnApply(unittest.TestCase):
    def test_new_node_on_existing_one_is_that_node(self):
        r = apply_increment(_base(), CODE, weld_tol=1e-3)
        self.assertTrue(r['ok'])
        st = r['struc']
        self.assertEqual(sorted(st.nodes), ['N0', 'N1', 'N2'])   # no duplicate
        self.assertEqual([(e.node_i, e.node_j) for e in st.bar_elements],
                         [('N0', 'N2')])
        self.assertEqual(r['welded'], ['N0'])

    def test_within_tolerance_only(self):
        code = "model.create_node(0.0004, 0.0)\nmodel.create_node(0.002, 0.0)\n"
        r = apply_increment(_base(), code, weld_tol=1e-3)
        self.assertEqual(sorted(r['struc'].nodes), ['N0', 'N1', 'N2'])  # 0.4 mm joined, 2 mm not
        self.assertEqual(r['welded'], ['N0'])

    def test_off_with_none(self):
        r = apply_increment(_base(), CODE, weld_tol=None)
        self.assertEqual(len(r['struc'].nodes), 4)
        self.assertEqual(r['welded'], [])

    def test_leaves_no_trace_on_the_models(self):
        base = _base()
        r = apply_increment(base, CODE)
        for m in (base, r['struc']):
            self.assertFalse(hasattr(m, '_script_weld_tol'))
        self.assertEqual(len(base.nodes), 2)             # the original untouched

    def test_explicit_id_is_kept(self):
        r = apply_increment(_base(), "model.create_node(0.0, 0.0, id='A')\n")
        self.assertIn('A', r['struc'].nodes)


if __name__ == "__main__":
    unittest.main()
