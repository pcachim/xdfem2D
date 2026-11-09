"""Geometry of the beam-bars result for drawing (no GUI).

``layout_beams`` turns the rows of :func:`xdfem2d.rc_design.design_beam_bars`
(one row per zone) into one *group* per beam — a group is a ``beam_tag`` (its
continuous segments ``V1.1``, ``V1.2`` … side by side) or a single auto-detected
beam — with every zone placed on a common x axis [m] along the group:

    group  = {key, title, length, segments, breaks}
    segment= {name, x0, x1, spans}             # a continuous run of spans
    span   = {id, x0, x1, zones}
    zone   = {x0, x1, bottom, top, bottom_layers, top_layers, row}
    break  = {x, before, after}                # continuity break between segments

Segments of one tag are ordered by name (``V1.1`` before ``V1.2``) and abut;
the boundary between two of them is reported in ``breaks`` — a section change,
a hinge or a kink, where the bars are not continuous (see
dev/BEAM_BARS_DEFINITION.md §3.4).
"""
from __future__ import annotations

import re


def _natural(name: str):
    return [int(t) if t.isdigit() else t for t in re.split(r"(\d+)", str(name))]


def layout_beams(rows: list) -> list:
    groups: dict = {}
    for r in rows:
        key = r.get("beam_tag") or r["beam"]
        g = groups.setdefault(key, {"key": key, "title": key, "segs": {}})
        seg = g["segs"].setdefault(r["beam"], {"name": r["beam"], "spans": {}})
        seg["spans"].setdefault(r["span"], []).append(r)

    out = []
    for g in groups.values():
        x = 0.0
        segments = []
        for name in sorted(g["segs"], key=_natural):
            seg = g["segs"][name]
            s = {"name": name, "x0": x, "spans": []}
            for sid, zrows in seg["spans"].items():
                zrows = sorted(zrows, key=lambda z: z["x0"])
                length = max(z["x1"] for z in zrows)
                span = {"id": sid, "x0": x, "x1": x + length, "zones": []}
                for z in zrows:
                    span["zones"].append({
                        "x0": x + z["x0"], "x1": x + z["x1"],
                        "bottom": z.get("bottom", ""), "top": z.get("top", ""),
                        "bottom_layers": [tuple(l) for l in z.get("bottom_layers") or []],
                        "top_layers": [tuple(l) for l in z.get("top_layers") or []],
                        "row": z})
                s["spans"].append(span)
                x += length
            s["x1"] = x
            segments.append(s)
        breaks = [{"x": a["x1"], "before": a["name"], "after": b["name"]}
                  for a, b in zip(segments, segments[1:])]
        # display name: the segments' common name (``Floor 1`` for ``Floor
        # 1.1`` / ``Floor 1.2``), with the tag when it differs
        disp = (re.sub(r"\.\d+$", "", segments[0]["name"])
                if len(segments) > 1 else segments[0]["name"])
        title = disp if disp == g["key"] else f"{disp} ({g['key']})"
        out.append({"key": g["key"], "title": title, "length": x,
                    "segments": segments, "breaks": breaks})
    return out
