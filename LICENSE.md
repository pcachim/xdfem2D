xdfem2d — GNU Lesser General Public License v3
================================================

This licence covers the **calculation engine**: the `xdfem2d` Python package,
published on PyPI under that name.

It does **not** cover the xdfem2D desktop application — the graphical interface
and the assistant, which form a separate package — distributed as freeware in
compiled form only, under its own terms (see [LICENSE.md](LICENSE.md)). Nothing
in those terms restricts the rights granted here.

(Described by package rather than by directory on purpose: this file travels
with the engine to a repository where the application's directories do not
exist, and a licence that points at a path the reader cannot find tells them
nothing.)

---

Copyright (c) 2026 Paulo Cachim

The xdfem2d calculation engine is free software: you can redistribute it
and/or modify it under the terms of the GNU Lesser General Public License
as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version.

The xdfem2d calculation engine is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU Lesser
General Public License for more details.

You should have received a copy of the GNU Lesser General Public License
along with this program. If not, see <https://www.gnu.org/licenses/>.

Full LGPL v3 text: <https://www.gnu.org/licenses/lgpl-3.0.html>

---

Engineering notice
------------------

The warranty disclaimer above is the operative legal text, and the following
adds nothing to it — but it is the part that matters in practice, so it is
stated plainly rather than left to be inferred from capital letters.

xdfem2d computes structural analysis and reinforcement design results. Those
results are not verified, certified, or checked against any code of practice
by their being produced. The software is an aid to calculation and not a
substitute for engineering judgement. Anyone using its output in the design,
verification, or construction of a real structure is responsible for
independently verifying it, and remains the engineer of record.

Third-party components
----------------------

The engine depends on other packages, each under its own licence — NumPy,
SciPy, eurocodepy and others listed in pyproject.toml. They are
dependencies, not part of this Software, and are not relicensed by this
file. See NOTICE.md for the components bundled with the compiled
application, where the distinction matters a great deal more.
