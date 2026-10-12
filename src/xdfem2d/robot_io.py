"""Export a xdfem2D model to the Autodesk Robot (Robot Millennium / Robot 97)
text interchange format (``.str``).

The xdfem2D planar model (X horizontal, Y vertical) is written as a Robot
**plane frame** (``FRAme PLAne``) in the X–Z plane: xdfem2D X → Robot X and
xdfem2D Y → Robot Z (Z is Robot's in-plane vertical axis). The file is a
keyword-driven text format; section headers are recognised by their first three
letters (``NODes``, ``ELEments``, ``PROperties`` …), and the file is written in
UTF-16 as Robot expects.

Mapping
-------
* nodes            -> ``NODes``  (id  X  Z)
* bar elements     -> ``ELEments``  (id  node_i  node_j)
* sections         -> ``PROperties``: one property block per xdfem2D section,
                      named after the section, with a rectangular ``BF``/``HT``
                      (width/height) geometry
* supports         -> ``SUPports``: node lists grouped by fixed DOF
                      (ux→UX, uy→UZ, tz→RY)
* load cases       -> ``LOAds`` / ``CASe # n``: self-weight, element uniform and
                      trapezoidal loads (fx→PX, fy→PZ) and nodal forces/moments
                      (fx→FX, fy→FZ, mz→CY)
* combinations     -> ``COMbination`` (linear, case/factor pairs)

!!! note
    The ``.str`` text format is a legacy Robot format and its full syntax lives
    in Robot's bundled ``R97mod01.doc`` (not published online). This exporter is
    best-effort and faithful to the structure of real Robot 97 files; verify a
    simple case after importing into Robot, in particular the section geometry
    (exported as a rectangular BF/HT) and the moment/sign conventions.
"""
from __future__ import annotations

from pathlib import Path


def _n(v: float) -> str:
    """Coordinate-style number: dot decimals, trailing zeros trimmed (matches
    how Robot writes node coordinates, e.g. 0, 0.23, 2.3)."""
    if v == 0 or v == 0.0:
        return "0"
    s = f"{v:.6f}".rstrip("0").rstrip(".")
    return s or "0"


def _f3(v: float) -> str:
    """Section-geometry number: 3 decimals (Robot writes BF 1.000 HT 0.300)."""
    return f"{v:.3f}"


def _f6(v: float) -> str:
    """Load number: 6 decimals (Robot writes PZ=-294.000000)."""
    return f"{v:.6f}"


def _compact_ids(ids: list) -> str:
    """Render a list of integer ids, collapsing consecutive runs to ``a to b``
    (Robot's compact list syntax). Ids that are not integers are written as-is."""
    try:
        nums = sorted({int(i) for i in ids})
    except (TypeError, ValueError):
        return " ".join(str(i) for i in ids)
    out: list[str] = []
    i = 0
    while i < len(nums):
        j = i
        while j + 1 < len(nums) and nums[j + 1] == nums[j] + 1:
            j += 1
        if j - i >= 2:                      # a run of >= 3 → "a to b"
            out.append(f"{nums[i]}to{nums[j]}")
        else:
            out.extend(str(nums[k]) for k in range(i, j + 1))
        i = j + 1
    return " ".join(out)


def to_str(struc) -> str:
    """Return the Robot ``.str`` text for *struc* (a Structure2D)."""
    # Geometry objects have no equivalent in .str — export the compiled mesh.
    if getattr(struc, "geometry_objects", None):
        from .geo_expand import expand_geometry
        struc = expand_geometry(struc)[0]
    L: list[str] = []
    n_nodes = len(struc.nodes)
    n_elems = len(struc.bar_elements)

    # Robot identifies nodes, elements and load cases by integer number. xdfem2D
    # ids are arbitrary strings, so map each to a sequential integer and use the
    # numbers everywhere they are referenced.
    node_num = {nid: i for i, nid in enumerate(struc.nodes, 1)}
    elem_num = {e.id: i for i, e in enumerate(struc.bar_elements, 1)}
    case_num = {lc.id: i for i, lc in enumerate(struc.load_cases, 1)}

    # ── Header ────────────────────────────────────────────────────────
    L += [
        "ROBOT97",
        "",
        "FRAme PLAne",
        "",
        "NUMbering DIScontinuous",
        "",
        f"NODes {n_nodes}  ELEments {n_elems}",
        "",
        "UNIts",
        "LENgth=m\tForce=kN",
        "",
    ]

    # ── Nodes (xdfem2D X,Y → Robot X,Z) ───────────────────────────────
    L.append("NODes")
    for nd in struc.nodes.values():
        L.append(f"{node_num[nd.id]}\t{_n(nd.x)}\t{_n(nd.y)}")
    L.append("")

    # ── Elements ──────────────────────────────────────────────────────
    L.append("ELEments")
    for e in struc.bar_elements:
        L.append(f"{elem_num[e.id]}\t{node_num[e.node_i]}\t{node_num[e.node_j]}")
    L.append("")

    # ── Properties ────────────────────────────────────────────────────
    # In the Robot .str format the quoted name in a PROperties block is the
    # MATERIAL/family; each section is defined by its geometry line below it.
    # (Putting a section name there breaks Robot when the name collides with a
    # keyword prefix — e.g. "BASE" → keyword "BAS".) So group by material.
    L.append("PROperties")
    elems_by_sec: dict[str, list] = {}
    for e in struc.bar_elements:
        elems_by_sec.setdefault(e.section_name, []).append(elem_num[e.id])
    secs_by_mat: dict[str, list] = {}
    for sname, sec in struc.sections.items():
        if elems_by_sec.get(sname):
            mat = getattr(sec, "material_name", "") or "MAT"
            secs_by_mat.setdefault(mat, []).append(sname)
    for mat, snames in secs_by_mat.items():
        L.append(f'"{mat}"')
        for sname in snames:
            sec = struc.sections[sname]
            b = getattr(sec, "b", 0.0) or 0.0
            h = getattr(sec, "h", 0.0) or 0.0
            L.append(f"{_compact_ids(elems_by_sec[sname])}  "
                     f"BF {_f3(b)} HT {_f3(h)}")
    L.append("")

    # ── Supports (grouped by fixed DOF: ux→UX, uy→UZ, tz→RY) ──────────
    supp_by_node: dict[str, object] = {}
    for a in struc.support_assignments:
        s = struc.supports.get(a.support_name)
        if s is not None:
            supp_by_node[a.node_id] = s
    groups: dict[tuple, list] = {}
    for nid, s in supp_by_node.items():
        key = (bool(s.ux), bool(s.uy), bool(s.tz))
        if any(key):
            groups.setdefault(key, []).append(node_num[nid])
    if groups:
        L.append("SUPports")
        for (ux, uy, tz), nids in groups.items():
            # Robot writes a fully-fixed support as a bare node list; only add
            # direction flags for a partial restraint.
            if ux and uy and tz:
                L.append(_compact_ids(nids))
            else:
                flags = " ".join(f for f, on in
                                 (("UX", ux), ("UZ", uy), ("RY", tz)) if on)
                L.append(f"{_compact_ids(nids)}  {flags}")
        L.append("")

    # ── Loads ─────────────────────────────────────────────────────────
    L += _loads_block(struc, node_num, elem_num, case_num)

    # ── Combinations (linear, case/factor pairs) ─────────────────────
    # Robot numbers combinations too; start them above the load-case numbers to
    # avoid collision, and reference cases by their mapped number.
    combo_no = len(case_num)
    for combo in getattr(struc, "load_combinations", []):
        if getattr(combo, "combo_type", "LinearSum") != "LinearSum":
            continue
        try:
            coeffs = struc.expand_combination_coefficients(combo)
        except Exception:
            coeffs = dict(getattr(combo, "coefficients", {}))
        pairs = "".join(f"{case_num[cid]} {_n(f)} "
                        for cid, f in coeffs.items() if f and cid in case_num)
        if not pairs:
            continue
        combo_no += 1
        kind = "SLS" if str(combo.id).upper().startswith("SLS") else "ULS"
        L += [f"COMbination # {combo_no} {combo.id}", f" {kind}", pairs, ""]

    L.append("END")
    return "\n".join(L) + "\n"


def _loads_block(struc, node_num, elem_num, case_num) -> list[str]:
    """Build the LOAds section: one CASe per load case with its self-weight,
    element (uniform / trapezoidal) and nodal loads."""
    L = ["LOAds", ""]
    for lc in struc.load_cases:
        L.append(f"CASe # {case_num[lc.id]} {lc.id}")

        if getattr(lc, "self_weight_factor", 0.0):
            L.append("SELf-weight")

        # Distributed loads on elements (fx→PX, fy→PZ). Uniform when start==end,
        # otherwise a trapezoid via the "X=0 .. TILl X=1 .. RElative" form.
        dl_lines: list[str] = []
        for dl in struc.distributed_loads:
            if dl.load_case_id != lc.id:
                continue
            comps = (("PX", dl.fxe, dl.fxd), ("PZ", dl.fye, dl.fyd))
            for tag, a, b in comps:
                if not a and not b:
                    continue
                enum = elem_num.get(dl.element_id)
                if enum is None:
                    continue
                if abs(a - b) < 1e-12:
                    dl_lines.append(f"\t{enum} {tag}={_f6(a)}")
                else:
                    dl_lines.append(
                        f"\t{enum}    X=0.0 {tag}={_f6(a)} "
                        f"TILl X=1.000 {tag}={_f6(b)}      RElative")
        # Uniform element point loads are not a native Robot bar load; skip.
        if dl_lines:
            L.append("ELEments")
            L.extend(dl_lines)

        # Nodal forces / moments (fx→FX, fy→FZ, mz→CY).
        nd_lines: list[str] = []
        for pl in struc.point_loads:
            if pl.load_case_id != lc.id:
                continue
            parts = []
            if pl.fx:
                parts.append(f"FX={_f6(pl.fx)}")
            if pl.fy:
                parts.append(f"FZ={_f6(pl.fy)}")
            if getattr(pl, "mz", 0.0):
                parts.append(f"CY={_f6(pl.mz)}")
            if parts and pl.node_id in node_num:
                nd_lines.append(f"\t{node_num[pl.node_id]}  " + " ".join(parts))
        if nd_lines:
            L.append("NODes")
            L.extend(nd_lines)

        L.append("")
    return L


def save_str(struc, path: str | Path):
    """Write *struc* to a Robot ``.str`` text file at *path* (UTF-16)."""
    Path(path).write_text(to_str(struc), encoding="utf-16")
