"""Physical-member recognition (xdfem2d.member_utils): collinear pass-through
bar chains merge into one member (length = sum), and the chain breaks at T/cross
joints, corners, supports, hinges and section changes.

Shared by steel, timber and RC-column design (each groups its bar elements
into physical members via :func:`identify_members` before checking
buckling/2nd-order effects).
"""
import context  # noqa: F401
from xdfem2d import Structure2D
from xdfem2d.member_utils import identify_members


def _struc():
    s = Structure2D()
    s.add_material("St", 210e6, 78.5)
    s.add_section("S", "St", 0.15, 0.30, shape="I", tw=0.0071, tf=0.0107)
    s.add_section("S2", "St", 0.2, 0.4, shape="I", tw=0.008, tf=0.012)
    return s


def test_three_collinear_bars_are_one_member():
    s = _struc()
    for i, (x, y) in enumerate([(0, 0), (0, 3), (0, 6), (0, 9)]):
        s.add_node(f"N{i}", x, y)
    s.add_bar_element("B0", "N0", "N1", "S")
    s.add_bar_element("B1", "N1", "N2", "S")
    s.add_bar_element("B2", "N2", "N3", "S")
    members, by_bar = identify_members(s)
    assert len(members) == 1
    m = members[0]
    assert set(m.bar_ids) == {"B0", "B1", "B2"}
    assert m.length == 9.0                       # 3 + 3 + 3
    assert by_bar["B1"].length == 9.0
    assert m.end_nodes == ("N0", "N3")


def test_t_joint_splits_the_column():
    # A column N0-N1-N2 with a beam framing in at the mid node N1 (3 bars there).
    s = _struc()
    for i, (x, y) in enumerate([(0, 0), (0, 3), (0, 6)]):
        s.add_node(f"N{i}", x, y)
    s.add_node("NB", 4, 3)                        # beam end
    s.add_bar_element("C0", "N0", "N1", "S")
    s.add_bar_element("C1", "N1", "N2", "S")
    s.add_bar_element("BM", "N1", "NB", "S")
    members, _ = identify_members(s)
    # N1 has 3 incident bars ⇒ boundary: column halves + beam = 3 members.
    lengths = sorted(m.length for m in members)
    assert len(members) == 3
    assert lengths == [3.0, 3.0, 4.0]


def test_corner_is_not_merged():
    s = _struc()
    s.add_node("N0", 0, 0); s.add_node("N1", 0, 3); s.add_node("N2", 4, 3)
    s.add_bar_element("V", "N0", "N1", "S")       # vertical
    s.add_bar_element("H", "N1", "N2", "S")       # horizontal (kink at N1)
    members, _ = identify_members(s)
    assert len(members) == 2


def test_support_breaks_the_member():
    s = _struc()
    for i, (x, y) in enumerate([(0, 0), (0, 3), (0, 6)]):
        s.add_node(f"N{i}", x, y)
    s.add_bar_element("B0", "N0", "N1", "S")
    s.add_bar_element("B1", "N1", "N2", "S")
    s.add_support("Pin", ux=True, uy=True, tz=False)
    s.assign_support("N1", "Pin")                # braced intermediate point
    members, _ = identify_members(s)
    assert len(members) == 2


def test_hinge_breaks_the_member():
    s = _struc()
    for i, (x, y) in enumerate([(0, 0), (0, 3), (0, 6)]):
        s.add_node(f"N{i}", x, y)
    s.add_bar_element("B0", "N0", "N1", "S")
    s.add_bar_element("B1", "N1", "N2", "S", hinge_i=True)  # released at N1
    members, _ = identify_members(s)
    assert len(members) == 2


def test_section_change_breaks_the_member():
    s = _struc()
    for i, (x, y) in enumerate([(0, 0), (0, 3), (0, 6)]):
        s.add_node(f"N{i}", x, y)
    s.add_bar_element("B0", "N0", "N1", "S")
    s.add_bar_element("B1", "N1", "N2", "S2")     # different section
    members, _ = identify_members(s)
    assert len(members) == 2


def test_single_bar_is_its_own_member():
    s = _struc()
    s.add_node("N0", 0, 0); s.add_node("N1", 5, 0)
    s.add_bar_element("B0", "N0", "N1", "S")
    members, by_bar = identify_members(s)
    assert len(members) == 1
    assert members[0].length == 5.0
    assert by_bar["B0"].length == 5.0
