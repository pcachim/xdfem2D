"""Fill the validation document with the current FEM results.

Reads the template ``validation/Validacao_Programa_MEF.docx``, re-solves every
model, and writes the "Resultado MEF" and "Erro (%)" columns of each comparison
table. The output is ``validation/Validacao_Programa_MEF_resultados.docx`` — a
regenerable record of how the engine compares to the closed-form solutions.

Run:  python validation/make_report.py
Requires python-docx (pip install python-docx).
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from xdfem2d import load_x2d, save_x2d          # noqa: E402
import validation_cases as vc                   # noqa: E402

VAL = ROOT / "validation"
MODELS = VAL / "models"
TEMPLATE = VAL / "Validacao_Programa_MEF.docx"
OUTPUT = VAL / "Validacao_Programa_MEF_resultados.docx"
ICON = ROOT / "assets" / "xdfem2d_logo.png"   # optional: skipped if absent

# Where each case's quantities land: (docx table index, first data row).
# Table row 0 is the header; 2.5a and 2.5b share table 5.
ROW_START = {"2.5b": 3}


def _fmt(v: float) -> str:
    """Portuguese-style number (comma decimal, space thousands)."""
    if abs(v) < 1e-12:
        return "0,00"
    if abs(v) >= 1e5:
        s = f"{v:.3e}"
    elif abs(v) >= 1:
        s = f"{v:,.2f}"                       # thousands + 2 decimals
    else:
        s = f"{v:.3g}"                        # 3 significant figures
    return s.replace(",", "§").replace(".", ",").replace("§", " ")


def _fmt_err(measured: float, analytical: float) -> str:
    if abs(analytical) < 1e-9:
        return "—"
    return f"{(measured - analytical) / analytical * 100:.2f}".replace(".", ",")


def _solve(model_id: str):
    struc, _, _ = load_x2d(MODELS / f"{model_id}.x2d")
    return struc, struc.calculate()


_MARKUP = __import__("re").compile(r"(\*\*[^*]+\*\*|\*[^*]+\*)")


def _add_rich(paragraph, text):
    """Write *text* into *paragraph*, turning **bold** and *italic* markers into
    real runs — the case narratives use them for the points worth emphasising,
    and Word would otherwise show the asterisks."""
    for piece in _MARKUP.split(text):
        if not piece:
            continue
        if piece.startswith("**") and piece.endswith("**"):
            paragraph.add_run(piece[2:-2]).bold = True
        elif piece.startswith("*") and piece.endswith("*") and len(piece) > 2:
            paragraph.add_run(piece[1:-1]).italic = True
        else:
            paragraph.add_run(piece)
    return paragraph


# Paragraph spacing (EMU) copied from the hand-written template chapters, so the
# generated chapters (arcs, Allman, springs, dynamics) match them exactly.
_SP_CHAPTER = (254000, 127000)     # "N. Título"           (Heading 1)
_SP_CASE = (190500, 95250)         # "Caso N.i — …"        (Heading 3)
_SP_SUB = (127000, 63500)          # "Solução analítica" … (Heading 3)
_SA_BODY = 76200                   # description / intro paragraph
_SA_TIGHT = 38100                  # analytical formula lines

# Direct formatting copied from the hand-written chapters. The template uses
# almost no styles: its headings are bold + dark blue at the RUN level (the
# Heading styles alone render lighter and non-bold) and its tables are
# formatted cell by cell in the XML (the table style is the empty "Normal
# Table"). The generated content replicates the same direct formatting so
# both kinds of chapter look identical.
_AZUL = "1F4E79"                   # heading text / table header fill
_BORDA = "AAAAAA"                  # cell border grey
_ZEBRA = "F2F2F2"                  # light grey of the last ("Erro") column


def _sp(p, before=None, after=None):
    """Set paragraph space before/after (EMU), matching the template."""
    from docx.shared import Emu
    if before is not None:
        p.paragraph_format.space_before = Emu(before)
    if after is not None:
        p.paragraph_format.space_after = Emu(after)
    return p


def _para(doc, text, after=_SA_BODY):
    return _sp(_add_rich(doc.add_paragraph(), text), after=after)


def _fmt_heading(p, color=True):
    """Bold (+ dark blue) at run level, matching the hand-written headings."""
    from docx.shared import RGBColor
    for r in p.runs:
        r.font.bold = True
        if color:
            r.font.color.rgb = RGBColor(0x1F, 0x4E, 0x79)
    return p


def _cell_pr(width, header=False, fill=None):
    """The template's tcPr: grey borders, margins, vertical centring."""
    from docx.oxml.ns import nsdecls
    m = 80 if header else 60
    borders = "".join(
        f'<w:{side} w:val="single" w:sz="2" w:space="0" w:color="{_BORDA}"/>'
        for side in ("top", "left", "bottom", "right"))
    shd = f'<w:shd w:val="clear" w:color="auto" w:fill="{fill}"/>' if fill else ""
    return (f'<w:tcPr {nsdecls("w")}>'
            f'<w:tcW w:w="{width}" w:type="dxa"/>'
            f'<w:tcBorders>{borders}</w:tcBorders>{shd}'
            f'<w:tcMar>'
            f'<w:top w:w="{m}" w:type="dxa"/><w:left w:w="100" w:type="dxa"/>'
            f'<w:bottom w:w="{m}" w:type="dxa"/><w:right w:w="100" w:type="dxa"/>'
            f'</w:tcMar><w:vAlign w:val="center"/></w:tcPr>')


def _format_cell(cell, width=2250, header=False, fill=None):
    """Format one cell like the template: tcPr above, 10 pt centred text
    (white bold on header cells)."""
    from docx.oxml import parse_xml
    from docx.shared import Pt, RGBColor
    from docx.enum.text import WD_ALIGN_PARAGRAPH
    tc = cell._tc
    if tc.tcPr is not None:
        tc.remove(tc.tcPr)
    tc.insert(0, parse_xml(_cell_pr(width, header, fill)))
    for p in cell.paragraphs:
        p.paragraph_format.alignment = WD_ALIGN_PARAGRAPH.CENTER
        for r in p.runs:
            r.font.size = Pt(10)
            if header:
                r.font.bold = True
                r.font.color.rgb = RGBColor(0xFF, 0xFF, 0xFF)


def _format_table(table):
    """Format a whole table like the template's comparison tables: fixed
    9000 dxa width, black outer grid, blue header row, grey last column."""
    from docx.oxml import parse_xml
    from docx.oxml.ns import nsdecls
    borders = "".join(
        f'<w:{side} w:val="single" w:sz="4" w:space="0" w:color="auto"/>'
        for side in ("top", "left", "bottom", "right", "insideH", "insideV"))
    tbl_pr = parse_xml(
        f'<w:tblPr {nsdecls("w")}>'
        f'<w:tblW w:w="9000" w:type="dxa"/>'
        f'<w:tblBorders>{borders}</w:tblBorders>'
        f'<w:tblCellMar><w:left w:w="10" w:type="dxa"/>'
        f'<w:right w:w="10" w:type="dxa"/></w:tblCellMar>'
        f'<w:tblLook w:val="0000" w:firstRow="0" w:lastRow="0"'
        f' w:firstColumn="0" w:lastColumn="0" w:noHBand="0" w:noVBand="0"/>'
        f'</w:tblPr>')
    table._tbl.replace(table._tbl.tblPr, tbl_pr)
    ncols = len(table.columns)
    width = 9000 // ncols
    for i, row in enumerate(table.rows):
        for j, cell in enumerate(row.cells):
            fill = _AZUL if i == 0 else (_ZEBRA if j == ncols - 1 else None)
            _format_cell(cell, width, header=(i == 0), fill=fill)


def _set_cell(table, row, col, text):
    """Replace a cell's text, KEEPING its paragraph properties (centring) and
    matching the 10 pt of the neighbouring hand-written cells — ``cell.text``
    would discard both."""
    from docx.shared import Pt
    cell = table.rows[row].cells[col]
    for extra in cell.paragraphs[1:]:
        extra._p.getparent().remove(extra._p)
    p = cell.paragraphs[0]
    for r in list(p.runs):
        r._r.getparent().remove(r._r)
    p.add_run(text).font.size = Pt(10)


def main():
    try:
        import docx
    except ImportError:
        raise SystemExit("python-docx is required: pip install python-docx")

    doc = docx.Document(str(TEMPLATE))
    tables = doc.tables

    # App icon at the very top of the first (cover) page, centred.
    if ICON.exists():
        from docx.shared import Inches
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        logo = doc.paragraphs[0].insert_paragraph_before()
        logo.alignment = WD_ALIGN_PARAGRAPH.CENTER
        logo.add_run().add_picture(str(ICON), width=Inches(1.4))

    # Standard cases (2.1–3.4, except 3.2): fill each quantity row.
    for c in vc.CASES:
        if c.doc_table < 0:
            continue
        table = tables[c.doc_table]
        start = ROW_START.get(c.id, 1)
        struc, res = _solve(c.id)
        for k, q in enumerate(c.quantities):
            m = q.measured(res, struc)
            a = q.analytical
            # The document reports deflections/rotations as magnitudes
            # (convention: positive downward). Match it for signed=False rows.
            if not q.signed:
                m, a = abs(m), abs(a)
            _set_cell(table, start + k, 2, _fmt(m))
            _set_cell(table, start + k, 3, _fmt_err(m, a))

    # Case 3.2 (table 8): finest mesh, informational error vs beam theory.
    t = tables[8]
    struc, res = _solve(vc.CASE_3_2_MESHES[-1][0])
    d = vc.case_3_2_tip_deflection(res, struc)
    sx = vc.case_3_2_max_sx(res, struc)
    txy = vc.case_3_2_max_txy(res, struc)
    _set_cell(t, 1, 2, _fmt(d));   _set_cell(t, 1, 3, _fmt_err(d, 17.26))
    _set_cell(t, 2, 2, _fmt(sx));  _set_cell(t, 2, 3, _fmt_err(sx, 24.0))
    # The CST peak τ_xy is a corner concentration at the load/support, not the
    # beam-theory mid-section value — report it but flag the error as N/A.
    _set_cell(t, 3, 2, _fmt(txy)); _set_cell(t, 3, 3, "n/a (pico de canto)")

    # ── Restructure into five parts ─────────────────────────────────────
    # The template is written as: 1. metodologia / 2. Parte I (barra) /
    # 3. Parte II (CST) / 4. quadro-resumo. The generated chapters turn it into
    #   Parte I  → 2. barras            3. arcos
    #   Parte II → 4. CST               5. Allman
    #   Parte III→ 6. molas
    #   Parte IV → 7. modal e espectro
    #   Parte V  → 8. quadro-resumo e conclusões
    # so the part labels become their own dividers, the chapters are renumbered
    # and everything generated is INSERTED BEFORE the summary chapter, which
    # therefore ends up last without having to be moved.
    import report_sections as rs                      # noqa: E402
    arcos = rs.arc_sections()
    allman = rs.allman_sections()
    plate = rs.plate_sections()
    grillage = rs.grillage_sections()
    springs = rs.spring_sections()
    dynamics = rs.modal_sections()
    constraints = rs.constraint_sections()

    h_barras = _find_heading(doc, "2. Parte I")
    h_cst = _find_heading(doc, "3. Parte II")
    h_resumo = _find_heading(doc, "4. Quadro-resumo")
    h_recom = _find_heading(doc, "4.1 Recomendações")

    _retitle(h_barras, "2. Elementos de barra (N, V, M)")
    _retitle(h_cst, "4. Elementos triangulares CST (estado plano de tensão)")
    _retitle(h_resumo, "11. Quadro-resumo dos casos de validação")
    _retitle(h_recom, "11.1 Recomendações finais")
    # The CST cases were numbered 3.x in the template and are now chapter 4.
    _renumber_cst_cases(doc, tables)

    _insert_divider(doc, h_barras, "Parte I — Elementos de barra")
    _insert_part(doc, h_cst, 3, "Estruturas de eixo curvo", _CH_ARC_INTRO, arcos)
    _insert_divider(doc, h_cst, "Parte II — Elementos triangulares "
                                "(estado plano de tensão)")
    _insert_part(doc, h_resumo, 5, "Elemento triangular de Allman",
                 _CH_ALLMAN_INTRO, allman)
    _insert_divider(doc, h_resumo, "Parte III — Elementos de placa e grelha "
                                   "(flexão fora do plano)")
    _insert_part(doc, h_resumo, 6, "Elementos de placa (DKT e MITC3)",
                 _CH_PLATE_INTRO, plate)
    _insert_part(doc, h_resumo, 7, "Elementos de grelha (barras de placa)",
                 _CH_GRILLAGE_INTRO, grillage)
    _insert_divider(doc, h_resumo, "Parte IV — Molas (apoios elásticos e "
                                   "fundação de Winkler)")
    _insert_part(doc, h_resumo, 8, "Molas de nó e de elemento",
                 _CH_SPRINGS_INTRO, springs)
    _insert_divider(doc, h_resumo, "Parte V — Análise dinâmica")
    _insert_part(doc, h_resumo, 9, "Análise modal e espectro de resposta",
                 _CH_MODAL_INTRO, dynamics)
    _insert_divider(doc, h_resumo, "Parte VI — Ligações (constraints)")
    _insert_part(doc, h_resumo, 10, "Ligações multiponto entre graus de liberdade",
                 _CH_CONSTRAINTS_INTRO, constraints)
    _insert_divider(doc, h_resumo, "Parte VII — Conclusões")

    _extend_summary_table(tables[11], ((3, arcos), (5, allman), (6, plate),
                                       (7, grillage), (8, springs),
                                       (9, dynamics), (10, constraints)))

    doc.save(str(OUTPUT))
    print(f"wrote {OUTPUT.relative_to(ROOT)}")


# ── Generated chapters ──────────────────────────────────────────────────────
_CH_SPRINGS_INTRO = [
    "Os casos desta parte validam as molas — de nó (kx, ky, kt) e de elemento "
    "(fundação de Winkler distribuída ao longo da barra) — e não constam da "
    "versão original deste plano. Foram acrescentados depois, com o mesmo "
    "critério: solução analítica fechada, comparação grandeza a grandeza e "
    "verificação de equilíbrio global.",
    "As duas famílias de molas têm naturezas numéricas distintas, e isso "
    "determina o que se pode exigir de cada caso. A mola de nó entra "
    "directamente na diagonal da matriz de rigidez: toda a solução fechada do "
    "tipo «estrutura + apoio elástico» é reproduzida exactamente. A mola de "
    "elemento é agregada de forma concentrada (k·L/2 em cada nó extremo) e não "
    "como matriz de fundação consistente: um estado uniforme continua exacto, "
    "mas um estado variável converge apenas em O(h²) e valida-se por "
    "convergência em três malhas.",
    "Critério de aceitação: erro relativo inferior a 1 % nos casos de solução "
    "fechada (na prática obtém-se muito melhor do que isso) e convergência "
    "monótona demonstrada nos casos de estado variável.",
]

_CH_MODAL_INTRO = [
    "Esta parte valida a análise modal (frequências próprias, modos, massas "
    "efectivas) e a análise por espectro de resposta.",
    "Três propriedades do programa condicionam o que pode ser verificado "
    "exactamente e o que só pode ser verificado por convergência. Primeira: a "
    "matriz de massa é diagonal (concentrada), construída a partir do valor "
    "absoluto de Fy dos casos de carga referidos pelo caso Mass, mais as massas "
    "nodais concentradas — logo, uma estrutura cuja massa esteja toda "
    "concentrada em nós é reproduzida exactamente, enquanto a massa distribuída "
    "converge em O(h²) e por baixo. Segunda: não há inércia de rotação, a menos "
    "que se dê uma massa nodal de rotação (mtz) explícita. Terceira: a massa "
    "derivada de um caso de carga é aplicada igualmente às direções X e Y, pelo "
    "que uma carga ascendente contribui com a mesma massa que uma descendente e "
    "uma carga horizontal não contribui com massa nenhuma.",
    "Critério de aceitação: erro inferior a 1 % nos casos de solução fechada e "
    "convergência de ordem 2 demonstrada nos casos de massa distribuída.",
]


_CH_CONSTRAINTS_INTRO = [
    "Esta parte valida as **ligações multiponto** (constraints): vínculos "
    "lineares que relacionam graus de liberdade de nós distintos, ao contrário "
    "dos apoios e das molas, que actuam sobre um grau de liberdade de cada vez. "
    "Cobrem-se as duas formas de uso corrente — a ligação rígida (um nó escravo "
    "acompanha um nó mestre como corpo rígido) e a igualdade de graus de "
    "liberdade (um conjunto de nós partilha o mesmo valor de uma componente).",
    "O programa impõe as ligações por **penalização**: cada equação de vínculo "
    "acrescenta à matriz de rigidez um termo muito rígido que a força a "
    "verificar-se. É o método mais simples e robusto para a primeira versão, "
    "mas é aproximado — os resultados batem a solução fechada com erro relativo "
    "da ordem de 10⁻⁶, e não à precisão da máquina, restando uma reação espúria "
    "muito pequena. Por isso a verificação de equilíbrio destes casos usa "
    "tolerância relativa. A ligação mestre–escravo exacta (por transformação "
    "de graus de liberdade), que elimina esta aproximação, está prevista para "
    "uma versão futura.",
    "Critério de aceitação: erro relativo inferior a 1 % face à solução fechada "
    "e cinemática de corpo rígido do nó escravo reproduzida exactamente.",
]


_CH_ARC_INTRO = [
    "Os casos anteriores usam barras rectas definidas nó a nó. Este capítulo "
    "valida a outra via de modelação de barras: a definição da estrutura por um "
    "**objecto de geometria** curvo, que o programa discretiza automaticamente "
    "em barras rectas na altura do cálculo.",
    "São, portanto, duas verificações numa só — a geração da malha a partir do "
    "objecto e o comportamento de uma peça de eixo curvo, cuja solução "
    "analítica se obtém por Castigliano. É também o único caso do plano em que "
    "a convergência se faz por cima, pelo motivo explicado adiante.",
]

_CH_ALLMAN_INTRO = [
    "O programa dispõe de um segundo elemento plano: o triângulo de **Allman**, "
    "que acrescenta a cada nó um grau de liberdade de rotação no plano "
    "(*drilling*, θz) além das duas translações do CST.",
    "O seu interesse está na flexão: com o mesmo número de nós, o campo de "
    "deslocamentos mais rico reduz substancialmente a rigidez excessiva que o "
    "CST exibe em problemas dominados por flexão — como se vê comparando o "
    "último caso deste capítulo com o Caso 4.2. Em contrapartida, o grau de "
    "liberdade adicional exige cuidado nos patch tests: em estados de extensão "
    "constante a rotação no plano é nula e tem de ser prescrita para que o "
    "teste fique bem posto. Os casos seguintes cobrem as duas situações, "
    "repetindo os patches do capítulo anterior com o novo elemento.",
]

_CH_PLATE_INTRO = [
    "No domínio de **placa** (flexão fora do plano) o programa dispõe de dois "
    "triângulos de 3 nós e 9 graus de liberdade (w, θx, θy): o **DKT** "
    "(Discrete Kirchhoff Triangle), de placa fina, e o **MITC3** "
    "(Mindlin-Reissner com interpolação mista), deformável ao corte e válido "
    "também para placa espessa. O MITC3 é o elemento por omissão.",
    "A validação usa a mesma laje quadrada simplesmente apoiada sob pressão "
    "uniforme em dois regimes: fino, contra a série clássica de Navier "
    "(Kirchhoff), com o DKT; e espesso, contra a série exacta de Mindlin (1.ª "
    "ordem de corte), com o MITC3 — a situação que o DKT não representa por não "
    "ter deformação por corte transverso. Ambos partilham nós, graus de "
    "liberdade, recuperação de esforços e momentos de Wood-Armer, pelo que "
    "trocar de elemento é apenas mudar a formulação da secção.",
]

_CH_GRILLAGE_INTRO = [
    "A grelha é a outra metade do domínio de placa: barras que flectem fora do "
    "plano. Uma barra de grelha usa os mesmos três graus de liberdade por nó "
    "(w, θx, θy) e acrescenta à flexão fora do plano (EI) a rigidez de torção "
    "de St-Venant (GJ), que a viga plana não tem.",
    "Os casos seguintes validam as três respostas em fórmula fechada: a flexão "
    "de uma consola sob carga transversal, a torção da mesma consola sob um "
    "momento torsor (que valida directamente GJ) e a grelha clássica de duas "
    "vigas cruzadas, onde a carga se reparte pelas duas na razão das rigidezes. "
    "Todos são reproduzidos exactamente.",
]


# ── Document restructuring helpers ─────────────────────────────────────────
def _find_heading(doc, prefix):
    """The paragraph whose text starts with *prefix* (template anchors)."""
    for p in doc.paragraphs:
        if p.text.strip().startswith(prefix):
            return p
    raise SystemExit(f"template: não encontrei o título que começa por {prefix!r}")


def _retitle(paragraph, novo):
    """Replace a heading's text, keeping its style and its first run's format."""
    runs = paragraph.runs
    if runs:
        runs[0].text = novo
        for r in runs[1:]:
            r.text = ""
    else:
        paragraph.add_run(novo)


def _renumber_cst_cases(doc, tables):
    """The CST chapter was numbered 3 in the template and is now chapter 4, so
    its case labels (and the references to them in the prose and in the summary
    table) have to follow. Only «3.1»–«3.4» are touched — a decimal such as
    3,125×10⁻³ has a comma and never matches."""
    import re
    pat = re.compile(r"\b3\.([1-4])\b")

    def fix(text):
        return pat.sub(lambda m: f"4.{m.group(1)}", text)

    for p in doc.paragraphs:
        if not pat.search(p.text):
            continue
        for r in p.runs:                      # run-level keeps the formatting
            if pat.search(r.text):
                r.text = fix(r.text)
        if pat.search(p.text):                # split across runs: rewrite whole
            _retitle(p, fix(p.text))
    for t in tables:
        for row in t.rows:
            for cell in row.cells:
                if pat.search(cell.text):
                    for p in cell.paragraphs:
                        if pat.search(p.text):
                            _retitle(p, fix(p.text))


def _insert_divider(doc, anchor, titulo):
    """A part divider — its own heading, on a new page, before *anchor*."""
    h = _fmt_heading(_sp(doc.add_heading(titulo, level=1), *_SP_CHAPTER))
    h.paragraph_format.page_break_before = True
    anchor._p.addprevious(h._p)


def _tail_len(body):
    """1 if the body ends with the section properties, else 0.

    python-docx appends paragraphs and tables BEFORE that trailing ``w:sectPr``,
    so it has to be excluded when locating what was just added."""
    kids = list(body)
    return 1 if kids and kids[-1].tag.endswith("}sectPr") else 0


def _insert_part(doc, anchor, num, titulo, intro, sections):
    """Build a generated chapter and move it in front of *anchor*.

    The new elements are located by POSITION, not by identity: lxml creates its
    element proxies on demand, so ``id()`` (or ``in``) on them is not stable
    across two iterations of the same tree — using it silently relocates
    unrelated blocks of the template."""
    body = doc.element.body
    tail = _tail_len(body)
    n0 = len(body) - tail
    _append_part(doc, num, titulo, intro, sections)
    novos = list(body)[n0:len(body) - tail]
    for el in novos:
        anchor._p.addprevious(el)


def _append_part(doc, num, titulo, intro, sections):
    """Append a whole generated chapter: heading, intro, one block per case.

    Spacing (space_before/after) is set to match the hand-written template
    chapters, so a generated chapter looks identical to Chapter 2 / 4."""
    _fmt_heading(_sp(doc.add_heading(f"{num}. {titulo}", level=1), *_SP_CHAPTER))
    for p in intro:
        _para(doc, p)
    for i, sec in enumerate(sections, start=1):
        struc, res = _solve(sec.model_id)
        # The hand-written case headings start with two spaces — keep it.
        _fmt_heading(_sp(doc.add_heading(
            f"  Caso {num}.{i} — {sec.titulo}", level=3), *_SP_CASE))
        _para(doc, sec.descricao)
        _fmt_heading(_sp(doc.add_heading("Solução analítica", level=3),
                         *_SP_SUB), color=False)
        for line in sec.analitica:
            _para(doc, line, after=_SA_TIGHT)
        if sec.quantities:
            _fmt_heading(_sp(doc.add_heading("Comparação de resultados",
                                             level=3), *_SP_SUB), color=False)
            _results_table(doc, sec, res, struc)
        for nota in sec.notas:
            _para(doc, f"Nota: {nota}" if len(sec.notas) == 1 else f"— {nota}")


def _extend_summary_table(table, parts):
    """Add the generated cases to the document's summary table (Caso / Elemento
    / Ação / Grandezas-chave), so it covers the whole plan and not only the
    chapters that were written by hand.

    Rows are placed in chapter order: the arc chapter sits between the bar and
    the CST chapters, so its rows cannot simply be appended."""
    def _chapter_of(row):
        txt = row.cells[0].text.strip()
        try:
            return int(txt.split(".")[0])
        except ValueError:
            return -1                      # header row

    for num, sections in parts:
        anchor = next((r for r in table.rows if _chapter_of(r) > num), None)
        for i, sec in enumerate(sections, start=1):
            elemento, accao, chave = sec.resumo
            row = table.add_row()
            for j, txt in enumerate((f"{num}.{i}", elemento, accao, chave)):
                cell = row.cells[j]
                cell.text = ""
                cell.paragraphs[0].add_run(txt)
                # add_row() creates bare cells: give them the same borders,
                # margins and 10 pt centred text as the hand-written rows.
                _format_cell(cell, fill=_ZEBRA if j == 3 else None)
            if anchor is not None:
                anchor._tr.addprevious(row._tr)


def _results_table(doc, sec, res, struc):
    """The 4-column comparison table used throughout the document."""
    table = doc.add_table(rows=1, cols=4)
    hdr = table.rows[0].cells
    for i, txt in enumerate(("Grandeza", "Solução analítica",
                             "Resultado MEF", "Erro (%)")):
        hdr[i].text = ""
        hdr[i].paragraphs[0].add_run(txt)
    for q in sec.quantities:
        a = q.analytical
        try:
            m = q.measured(res, struc)
        except Exception as exc:                   # noqa: BLE001
            m = float("nan")
            print(f"  ! {sec.model_id} / {q.label}: {exc}")
        if not q.signed:
            a, m = abs(a), abs(m)
        row = table.add_row().cells
        row[0].text = f"{q.label} ({q.unit})"
        row[1].text = _fmt(a)
        row[2].text = _fmt(m)
        row[3].text = _fmt_err(m, a)
    _format_table(table)


if __name__ == "__main__":
    main()
