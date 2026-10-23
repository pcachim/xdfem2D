"""Physical-member recognition (material-neutral).

A frame model discretises a physical member (a column running through several
floors, a continuous beam) into one or more collinear :class:`BarElement`s. The
Eurocode 3/5/2 checks (steel/timber buckling, RC column slenderness) need the
length of the *whole* member (the buckling length is ``K * L`` with ``L`` the
physical length), not of each element.

:func:`identify_members` walks the bar topology and merges chains of collinear
bar elements that pass straight through a node, breaking the chain wherever the
member physically ends. A node is a **member boundary** when any of the
following holds:

* a number of incident bars other than two (a free end / support end with one
  bar, or a T / cross joint with three or more) -- this is the "T" rule;
* the two incident bars are not collinear (a corner / kink);
* the two incident bars use different sections (the check assumes a uniform
  member);
* one of the incident bars is released (a moment hinge) at that node;
* the node carries a support (a braced / restrained point).

Everything else is a *pass-through* node and the two bars belong to the same
physical member.

This module has no material-specific logic; it is shared by
:mod:`xdfem2d.steel_design`, :mod:`xdfem2d.timber_design` and
:mod:`xdfem2d.rc_design` (RC columns).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

# Collinear if the turn at the shared node is below this angle [rad] (~1°).
_ANGLE_TOL = math.radians(1.0)


@dataclass
class Member:
    """A physical member — an ordered chain of collinear bar elements."""

    id: str
    bar_ids: list = field(default_factory=list)   # ordered along the member
    node_ids: list = field(default_factory=list)  # ordered nodes (len = bars+1)
    section_name: str = ""
    length: float = 0.0                            # total length [m]

    @property
    def end_nodes(self) -> tuple:
        """The two extreme nodes of the member."""
        return (self.node_ids[0], self.node_ids[-1]) if self.node_ids else ()


def _bar_length(struc, bar) -> float:
    ni, nj = struc.nodes.get(bar.node_i), struc.nodes.get(bar.node_j)
    if ni is None or nj is None:
        return 0.0
    return math.hypot(nj.x - ni.x, nj.y - ni.y)


def _unit_from(struc, node_id, bar):
    """Outgoing unit vector of *bar* seen from *node_id* (its shared end)."""
    other = bar.node_j if bar.node_i == node_id else bar.node_i
    a, b = struc.nodes[node_id], struc.nodes[other]
    dx, dy = b.x - a.x, b.y - a.y
    L = math.hypot(dx, dy)
    return (dx / L, dy / L) if L > 0 else (0.0, 0.0)


def _supported_nodes(struc) -> set:
    return {a.node_id for a in getattr(struc, "support_assignments", [])}


def _hinge_at(bar, node_id) -> bool:
    """True if *bar* is released (moment hinge) at its end on *node_id*."""
    if bar.node_i == node_id:
        return bool(getattr(bar, "hinge_i", False))
    if bar.node_j == node_id:
        return bool(getattr(bar, "hinge_j", False))
    return False


def identify_members(struc) -> tuple:
    """Group the bar elements of *struc* into physical members.

    Args:
        struc: A ``Structure2D`` with ``nodes`` and ``bar_elements``.

    Returns:
        ``(members, by_bar)`` where *members* is a list of :class:`Member` and
        *by_bar* maps every bar id to its :class:`Member` (so the design routine
        can look up a member's length from any of its elements).

    """
    bars = {b.id: b for b in struc.bar_elements}
    incident: dict[str, list[str]] = {}
    for b in struc.bar_elements:
        incident.setdefault(b.node_i, []).append(b.id)
        incident.setdefault(b.node_j, []).append(b.id)
    supported = _supported_nodes(struc)

    def passthrough(node_id: str) -> bool:
        ids = incident.get(node_id, [])
        if len(ids) != 2:
            return False                       # free end, or T / cross joint
        if node_id in supported:
            return False                       # braced / restrained point
        b1, b2 = bars[ids[0]], bars[ids[1]]
        if b1.section_name != b2.section_name:
            return False                       # non-uniform member
        if _hinge_at(b1, node_id) or _hinge_at(b2, node_id):
            return False                       # released connection
        u1, u2 = _unit_from(struc, node_id, b1), _unit_from(struc, node_id, b2)
        dot = u1[0] * u2[0] + u1[1] * u2[1]
        # Pass straight through ⇒ the two outgoing directions are opposite.
        return dot < 0.0 and (math.pi - math.acos(max(-1.0, min(1.0, dot)))) <= _ANGLE_TOL

    def next_bar(node_id: str, came_from: str):
        ids = incident.get(node_id, [])
        if len(ids) != 2 or not passthrough(node_id):
            return None
        other = ids[0] if ids[1] == came_from else ids[1]
        return other

    visited: set[str] = set()
    members: list[Member] = []
    by_bar: dict[str, Member] = {}
    counter = 0

    for start in struc.bar_elements:
        if start.id in visited:
            continue
        # Walk backwards from node_i, then forwards from node_j, to order the
        # chain from one physical end to the other.
        chain = [start.id]
        visited.add(start.id)

        # extend towards node_i
        cur, frm = start.node_i, start.id
        while True:
            nxt = next_bar(cur, frm)
            if nxt is None or nxt in visited:
                break
            visited.add(nxt)
            chain.insert(0, nxt)
            b = bars[nxt]
            cur = b.node_i if b.node_j == cur else b.node_j
            frm = nxt
        # extend towards node_j
        cur, frm = start.node_j, start.id
        while True:
            nxt = next_bar(cur, frm)
            if nxt is None or nxt in visited:
                break
            visited.add(nxt)
            chain.append(nxt)
            b = bars[nxt]
            cur = b.node_i if b.node_j == cur else b.node_j
            frm = nxt

        # Ordered node list along the chain.
        node_seq: list[str] = []
        for k, bid in enumerate(chain):
            b = bars[bid]
            if k == 0:
                # orient the first bar so the chain starts at its free end
                if len(chain) > 1:
                    nb = bars[chain[1]]
                    shared = b.node_j if b.node_j in (nb.node_i, nb.node_j) else b.node_i
                    first = b.node_i if shared == b.node_j else b.node_j
                    node_seq = [first, shared]
                else:
                    node_seq = [b.node_i, b.node_j]
            else:
                prev = node_seq[-1]
                nxt_node = b.node_j if b.node_i == prev else b.node_i
                node_seq.append(nxt_node)

        length = sum(_bar_length(struc, bars[bid]) for bid in chain)
        counter += 1
        m = Member(id=f"M{counter}", bar_ids=chain, node_ids=node_seq,
                   section_name=bars[chain[0]].section_name, length=length)
        members.append(m)
        for bid in chain:
            by_bar[bid] = m

    return members, by_bar


# ── Free-end (cantilever) detection ──────────────────────────────────────────

def _incident_bar_count(struc) -> dict:
    """How many bar elements touch each node."""
    from collections import Counter
    c: Counter = Counter()
    for b in struc.bar_elements:
        c[b.node_i] += 1
        c[b.node_j] += 1
    return c


def _constraint_nodes(struc) -> set:
    """All node ids referenced by any enabled constraint (terms/master/slaves/
    equal-dof nodes)."""
    out: set = set()
    for c in getattr(struc, "constraints", {}).values():
        if not getattr(c, "enabled", True):
            continue
        for term in getattr(c, "terms", None) or []:
            if term:
                out.add(term[0])
        if getattr(c, "master", ""):
            out.add(c.master)
        out.update(getattr(c, "slaves", None) or [])
        out.update(getattr(c, "nodes", None) or [])
    return out


def _node_is_free(struc, node_id, incident, constrained) -> bool:
    """True when *nothing* is attached at ``node_id``: exactly one incident bar
    (the member's terminal bar) and no support, node spring, triangle, or
    constraint. This is the physical free end of a cantilever."""
    if incident.get(node_id, 0) != 1:
        return False
    if any(a.node_id == node_id
           for a in getattr(struc, "support_assignments", [])):
        return False
    if node_id in getattr(struc, "node_springs", {}):
        return False
    if node_id in constrained:
        return False
    for t in getattr(struc, "tri_elements", []):
        if node_id in (t.node_i, t.node_j, t.node_k):
            return False
    return True


def member_free_ends(struc, member, incident=None, constrained=None) -> tuple:
    """(free_i, free_j): whether each extreme node of *member* is truly free.

    A terminal bar carrying a foundation (element) spring is on an elastic
    support, so that end is not treated as free."""
    if not member.node_ids or not member.bar_ids:
        return (False, False)
    if incident is None:
        incident = _incident_bar_count(struc)
    if constrained is None:
        constrained = _constraint_nodes(struc)
    springs = getattr(struc, "element_springs", {})
    fi = (_node_is_free(struc, member.node_ids[0], incident, constrained)
          and member.bar_ids[0] not in springs)
    fj = (_node_is_free(struc, member.node_ids[-1], incident, constrained)
          and member.bar_ids[-1] not in springs)
    return (fi, fj)


def member_has_free_end(struc, member, incident=None, constrained=None) -> bool:
    fi, fj = member_free_ends(struc, member, incident, constrained)
    return fi or fj


def cantilever_k_for(struc, member, prefs, incident=None, constrained=None):
    """The auto buckling-length factor K for a free-end (cantilever) member, or
    ``None`` when the feature is off or the member has no free end.

    Applies to the in-plane and out-of-plane flexural buckling (Ky, Kz): a truly
    free tip is unrestrained in both planes. K_LT is left to the caller."""
    prefs = prefs or {}
    if not prefs.get("cantilever_auto", True):
        return None
    if not member_has_free_end(struc, member, incident, constrained):
        return None
    return float(prefs.get("cantilever_k", 2.0))
