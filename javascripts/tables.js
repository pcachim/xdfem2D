/* A column whose text is short stays on one line (up to 44 characters for the first column, 28 for the others): the browser shares the width of a table among its columns by the text of the
   longest cell, so a column of short labels beside long text is wrapped to several lines. */
function fitTables() {
  document.querySelectorAll(".md-typeset table:not([class])").forEach(function (table) {
    var rows = Array.prototype.slice.call(table.rows);
    var columns = Math.max.apply(null, rows.map(function (r) { return r.cells.length; }));
    for (var c = 0; c < columns; c++) {
      var longest = 0;
      rows.forEach(function (r) { if (r.cells[c]) longest = Math.max(longest, r.cells[c].textContent.trim().length); });
      if (longest <= (c === 0 ? 44 : 28)) rows.forEach(function (r) { if (r.cells[c]) r.cells[c].classList.add("nowrap-col"); });
    }
  });
}
if (typeof document$ !== "undefined") document$.subscribe(fitTables);   // Material's instant navigation
else document.addEventListener("DOMContentLoaded", fitTables);
