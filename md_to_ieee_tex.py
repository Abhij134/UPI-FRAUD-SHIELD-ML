"""
md_to_ieee_tex.py
=================
Converts UPI_Fraud_Shield_ML_Paper_Final.md → UPI_Fraud_Shield_ML_Paper_Final.tex
in IEEE Transactions two-column format WITHOUT any LLM rewriting.

All text is taken verbatim from the Markdown; only structural markup is translated.

Architecture
------------
1. Read the Markdown line-by-line, accumulating blocks.
2. Identify block types:   heading, table, figure, reference, blockquote, list, code-fence, blank, text.
3. Emit a corresponding LaTeX fragment for each block.
4. Frame everything with the IEEEtran preamble / document envelope.

Unicode → LaTeX
---------------
All Unicode symbols commonly found in the paper are mapped to LaTeX equivalents
(₹, ×, →, ≥, ≤, —, …, etc.).

Tables
------
Every table row is emitted; nothing is truncated.  The script uses booktabs
(\\toprule, \\midrule, \\bottomrule).  Tables with >4 columns use table* (full width).
"""

import re
import sys

INPUT  = "UPI_Fraud_Shield_ML_Paper_Final.md"
OUTPUT = "UPI_Fraud_Shield_ML_Paper_Final.tex"

# ──────────────────────────────────────────────────────────────
# 1.  Unicode → LaTeX character map
# ──────────────────────────────────────────────────────────────
UNICODE_MAP = [
    # rupee, arrows, math relations
    ("\u20b9",  r"\rupee{}"),   # ₹
    ("\u2192",  r"$\rightarrow$"),   # →
    ("\u2190",  r"$\leftarrow$"),    # ←
    ("\u2191",  r"$\uparrow$"),      # ↑
    ("\u2193",  r"$\downarrow$"),    # ↓
    ("\u2265",  r"$\geq$"),          # ≥
    ("\u2264",  r"$\leq$"),          # ≤
    ("\u2260",  r"$\neq$"),          # ≠
    ("\u2248",  r"$\approx$"),       # ≈
    ("\u00b1",  r"$\pm$"),           # ±
    ("\u00d7",  r"$\times$"),        # ×
    # dashes, quotes
    ("\u2014",  "---"),              # em dash
    ("\u2013",  "--"),               # en dash
    ("\u2012",  "--"),               # figure dash
    ("\u2018",  "`"),                # left single quote
    ("\u2019",  "'"),                # right single quote
    ("\u201c",  "``"),               # left double quote
    ("\u201d",  "''"),               # right double quote
    ("\u2026",  r"\ldots{}"),        # ellipsis
    ("\u2012",  "--"),               # figure dash
    ("\u2011",  "-"),                # non-breaking hyphen
    ("\u00a0",  " "),                # non-breaking space
    # math / greek
    ("\u0394",  r"$\Delta$"),        # Δ
    ("\u03b1",  r"$\alpha$"),        # α
    ("\u03b2",  r"$\beta$"),         # β
    ("\u03b3",  r"$\gamma$"),        # γ
    ("\u03bb",  r"$\lambda$"),       # λ
    ("\u03bc",  r"$\mu$"),           # μ
    ("\u03c3",  r"$\sigma$"),        # σ
    ("\u2208",  r"$\in$"),           # ∈
    ("\u2209",  r"$\notin$"),        # ∉
    ("\u2229",  r"$\cap$"),          # ∩
    ("\u222a",  r"$\cup$"),          # ∪
    ("\u2205",  r"$\emptyset$"),     # ∅
    ("\u2200",  r"$\forall$"),       # ∀
    ("\u2203",  r"$\exists$"),       # ∃
    ("\u221e",  r"$\infty$"),        # ∞
    ("\u2212",  r"$-$"),             # minus sign
    ("\u00b2",  r"$^{2}$"),          # superscript 2
    ("\u00b3",  r"$^{3}$"),          # superscript 3
    # box drawing chars (used in preamble comments)
    ("\u2500",  "-"),                # ─
    ("\u2502",  "|"),                # │
    ("\u250c",  "+"),                # ┌
    ("\u2510",  "+"),                # ┐
    ("\u2514",  "+"),                # └
    ("\u2518",  "+"),                # ┘
    # arrows
    ("\u21d2",  r"$\Rightarrow$"),   # ⇒
    ("\u21d4",  r"$\Leftrightarrow$"),# ⇔
    # checkmark
    ("\u2713",  r"\checkmark"),      # ✓
    ("\u2714",  r"\checkmark"),      # ✔
    ("\u2717",  r"$\times$"),        # ✗
    # run placeholder (should already be gone)
    ("\u27e8run\u27e9", r"\textit{(run)}"),
    # subscript/superscript
    ("\u00b9",  r"$^{1}$"),
    ("\u2070",  r"$^{0}$"),
    # degree
    ("\u00b0",  r"$^{\circ}$"),
    # fraction slash
    ("\u2044",  "/"),
]

def esc_tex(text: str, math_ok: bool = True) -> str:
    """
    Escape raw text for LaTeX.
    If math_ok=True we first apply unicode replacements that generate $...$ fragments,
    then handle regular special characters.
    We are careful NOT to double-escape & inside table cells (handled by caller).
    """
    # Apply unicode map first
    for uni, latex in UNICODE_MAP:
        text = text.replace(uni, latex)

    # Escape LaTeX specials that remain outside math zones:
    # We do this segment-by-segment to avoid clobbering already-emitted \$...\$ zones.
    segments = re.split(r'(\$[^$]*?\$)', text)   # split on inline math
    result = []
    for seg in segments:
        if seg.startswith('$') and seg.endswith('$'):
            result.append(seg)   # pass math through unchanged
        else:
            # Escape in order of longest/safest first
            seg = seg.replace('\\', r'\textbackslash{}')
            seg = seg.replace('{', r'\{').replace('}', r'\}')
            seg = seg.replace('#', r'\#')
            seg = seg.replace('%', r'\%')
            seg = seg.replace('^', r'\^{}')
            seg = seg.replace('~', r'\textasciitilde{}')
            seg = seg.replace('_', r'\_')
            seg = seg.replace('|', r'\textbar{}')
            seg = seg.replace('<', r'\textless{}')
            seg = seg.replace('>', r'\textgreater{}')
            result.append(seg)
    return ''.join(result)


# ──────────────────────────────────────────────────────────────
# 2.  Inline Markdown → LaTeX (bold, italic, code, links, math)
# ──────────────────────────────────────────────────────────────
def inline_md_to_tex(text: str) -> str:
    """Convert inline Markdown markup to LaTeX commands."""

    # We protect math spans first
    math_spans = {}
    def protect_math(m):
        key = f"MATHSPAN{len(math_spans)}MATH"
        math_spans[key] = m.group(0)
        return key
        
    # Preserve display math $$...$$ first
    text = re.sub(r'\$\$(.+?)\$\$', lambda m: r'\[' + m.group(1) + r'\]', text, flags=re.DOTALL)
    
    # Protect \[ ... \] blocks (treat as display math)
    text = re.sub(r'\\\[(.*?)\\\]', protect_math, text, flags=re.DOTALL)

    # Protect Inline math $...$
    text = re.sub(r'\$[^$\n]+?\$', protect_math, text)

    # Protect Inline code `...`
    code_spans = {}
    def protect_code(m):
        key = f"CODESPAN{len(code_spans)}CODE"
        code_spans[key] = m.group(1)
        return key
    text = re.sub(r'`([^`]+)`', protect_code, text)

    # Escape LaTeX special chars outside of math/code
    text = text.replace('\\', r'\textbackslash{}')
    text = text.replace('&', r'\&')
    text = text.replace('%', r'\%')
    text = text.replace('#', r'\#')

    # Bold-italic ***...***
    text = re.sub(r'\*{3}(.+?)\*{3}', r'\\textbf{\\textit{\1}}', text)
    # Bold **...**
    text = re.sub(r'\*{2}(.+?)\*{2}', r'\\textbf{\1}', text)
    # Italic *...* or _..._
    text = re.sub(r'\*(.+?)\*', r'\\textit{\1}', text)
    text = re.sub(r'(?<![a-zA-Z\\])_(.+?)_(?![a-zA-Z\\])', r'\\textit{\1}', text)
    
    # Escape any remaining literal underscores
    text = re.sub(r'(?<!\\)_', r'\\textunderscore{}', text)

    # [text](url) hyperlinks
    text = re.sub(r'\[([^\]]+)\]\(([^)]+)\)', r'\\href{\2}{\1}', text)
    # Citation-style references like [1], [2,3], etc.
    text = re.sub(r'\[(\d+(?:,\s*\d+)*)\]', lambda m: r'\cite{' + 'ref' + re.sub(r'[,\s]+', ',ref', m.group(1)) + r'}', text)

    # Restore code spans (and escape latex specials inside them)
    for key, val in code_spans.items():
        # Inside texttt, we must escape _, &, %, #
        val = val.replace('\\', r'\textbackslash{}')
        val = val.replace('_', r'\textunderscore{}')
        val = val.replace('&', r'\&')
        val = val.replace('%', r'\%')
        val = val.replace('#', r'\#')
        text = text.replace(key, r'\texttt{' + val + '}')

    # Restore math spans
    for key, val in math_spans.items():
        text = text.replace(key, val)

    return text


# ──────────────────────────────────────────────────────────────
# 3.  Table parser
# ──────────────────────────────────────────────────────────────
def parse_and_emit_table(rows: list[str], wide: bool) -> str:
    """
    rows: list of raw Markdown table lines (including separator line).
    wide: if True, use table* environment (two-column span).
    Returns a LaTeX table environment string.
    """
    # Filter out separator rows (|----|)
    data_rows = [r for r in rows if not re.match(r'^\s*\|[\s\-|:]+\|\s*$', r)]

    if not data_rows:
        return ""

    # Parse cells
    def parse_cells(line: str) -> list[str]:
        line = line.strip()
        if line.startswith('|'):
            line = line[1:]
        if line.endswith('|'):
            line = line[:-1]
        return [c.strip() for c in line.split('|')]

    header = data_rows[0]
    body   = data_rows[1:]

    header_cells = parse_cells(header)
    ncols = len(header_cells)

    env   = "table*" if wide else "table"
    colspec = "l" + "p{2.8cm}" * (ncols - 1) if ncols > 3 else "l" * ncols
    # Better column spec based on count
    if ncols == 2:
        colspec = "lp{5cm}"
    elif ncols == 3:
        colspec = "lll"
    elif ncols == 4:
        colspec = "lp{3cm}p{3cm}l"
    elif ncols == 5:
        colspec = "lp{2.5cm}p{2.5cm}p{2cm}l"
    elif ncols >= 6:
        colspec = ("l" + "p{1.9cm}" * (ncols - 1))

    lines = []
    lines.append(f"\\begin{{{env}}}[!htbp]")
    lines.append("  \\centering")
    lines.append("  \\footnotesize")
    lines.append(f"  \\begin{{tabular}}{{{colspec}}}")
    lines.append("    \\toprule")

    # Header row
    def cell(c):
        return inline_md_to_tex(c)

    header_tex = " & ".join(f"\\textbf{{{cell(c)}}}" for c in header_cells)
    lines.append(f"    {header_tex} \\\\")
    lines.append("    \\midrule")

    # Body rows
    for row in body:
        cells = parse_cells(row)
        # Pad / trim to ncols
        while len(cells) < ncols:
            cells.append("")
        cells = cells[:ncols]
        row_tex = " & ".join(cell(c) for c in cells)
        lines.append(f"    {row_tex} \\\\")

    lines.append("    \\bottomrule")
    lines.append(f"  \\end{{tabular}}")
    lines.append(f"\\end{{{env}}}")
    lines.append("")

    return "\n".join(lines)


# ──────────────────────────────────────────────────────────────
# 4.  Figure emitter
# ──────────────────────────────────────────────────────────────
_fig_counter = [0]

def emit_figure(alt: str, path: str) -> str:
    _fig_counter[0] += 1
    n = _fig_counter[0]
    label = f"fig:{n}"
    # Ensure path uses forward slashes and no leading ./
    path = path.replace('\\', '/').lstrip('./')
    return (
        f"\\begin{{figure}}[!htbp]\n"
        f"  \\centering\n"
        f"  \\includegraphics[width=\\columnwidth]{{{path}}}\n"
        f"  \\caption{{{inline_md_to_tex(alt)}}}\n"
        f"  \\label{{{label}}}\n"
        f"\\end{{figure}}\n"
    )


# ──────────────────────────────────────────────────────────────
# 5.  Reference parser
# ──────────────────────────────────────────────────────────────
def parse_reference_line(line: str) -> tuple[str, str] | None:
    """
    Parse a line like:  [3] Author, "Title," ...
    Returns (key, text) or None.
    """
    m = re.match(r'^\[(\d+)\]\s+(.*)', line.strip())
    if m:
        num = m.group(1)
        text = m.group(2)
        return f"ref{num}", text
    return None


# ──────────────────────────────────────────────────────────────
# 6.  IEEE Preamble
# ──────────────────────────────────────────────────────────────
PREAMBLE = r"""% ==============================================================
%  UPI Fraud Shield ML -- IEEE Transactions Manuscript
%  Auto-generated by md_to_ieee_tex.py (do not edit directly)
% ==============================================================
\documentclass[journal,10pt,twocolumn]{IEEEtran}

% -- Packages --------------------------------------------------
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{amsmath,amssymb}
\usepackage{booktabs}          % professional table rules
\usepackage{hyperref}
\usepackage{xcolor}
\usepackage{cite}
\usepackage{url}
\usepackage{textcomp}
\usepackage{microtype}
\usepackage{balance}
\usepackage{upgreek}

% Rupee symbol (ASCII-safe definition)
\newcommand{\rupee}{\text{\textsf{Rs.}}}

% -- Hyperref colours ------------------------------------------
\hypersetup{
  colorlinks = true,
  linkcolor  = blue,
  citecolor  = blue,
  urlcolor   = blue
}

% --------------------------------------------------------------
\begin{document}
"""

POSTAMBLE = r"""
\end{document}
"""


# ──────────────────────────────────────────────────────────────
# 7.  Main conversion loop
# ──────────────────────────────────────────────────────────────
def convert(src: str) -> str:
    lines = src.splitlines()
    out   = []

    out.append(PREAMBLE)

    # State
    in_table        = False
    table_rows      = []
    in_references   = False
    in_abstract     = False
    in_code_fence   = False
    code_lang       = ""
    code_lines      = []
    in_blockquote   = False
    bq_lines        = []
    title_emitted   = False
    abstract_tex    = []
    after_abstract  = False
    author_lines    = []
    in_itemize      = False
    in_enumerate    = False
    list_stack      = []   # "itemize" | "enumerate"
    section_count   = [0]
    section_name    = [""]

    def flush_table():
        nonlocal in_table, table_rows
        if table_rows:
            # Determine if wide: use table* if >4 columns or if we're inside a data-heavy section
            sample = [r for r in table_rows if not re.match(r'^\s*\|[\s\-|:]+\|\s*$', r)]
            ncols = len(sample[0].split('|')) - 2 if sample else 1
            wide = ncols >= 5
            out.append(parse_and_emit_table(table_rows, wide))
        table_rows = []
        in_table   = False

    def flush_blockquote():
        nonlocal in_blockquote, bq_lines
        if bq_lines:
            out.append(r"\begin{quote}")
            for bl in bq_lines:
                out.append(r"\small " + inline_md_to_tex(bl))
            out.append(r"\end{quote}")
            out.append("")
        bq_lines       = []
        in_blockquote  = False

    def flush_code():
        nonlocal in_code_fence, code_lines, code_lang
        if code_lines:
            out.append(r"\begin{verbatim}")
            for cl in code_lines:
                out.append(cl)
            out.append(r"\end{verbatim}")
            out.append("")
        code_lines    = []
        in_code_fence = False
        code_lang     = ""

    def close_lists():
        nonlocal in_itemize, in_enumerate, list_stack
        while list_stack:
            env = list_stack.pop()
            out.append(f"\\end{{{env}}}")
        in_itemize   = False
        in_enumerate = False

    i = 0
    while i < len(lines):
        line = lines[i]
        raw  = line                         # for table row check
        stripped = line.rstrip('\r')

        # ── Code fence ────────────────────────────────────────
        if stripped.startswith('```'):
            if in_table:   flush_table()
            if in_blockquote: flush_blockquote()
            if in_code_fence:
                flush_code()
            else:
                code_lang = stripped[3:].strip()
                in_code_fence = True
            i += 1
            continue

        if in_code_fence:
            code_lines.append(stripped)
            i += 1
            continue

        # ── Block quote ───────────────────────────────────────
        if stripped.startswith('>'):
            if in_table:   flush_table()
            close_lists()
            in_blockquote = True
            bq_text = re.sub(r'^>\s?', '', stripped)
            bq_lines.append(bq_text)
            i += 1
            continue
        else:
            if in_blockquote:
                flush_blockquote()

        # ── Table row detection ───────────────────────────────
        is_table_row = bool(re.match(r'^\s*\|', stripped)) and '|' in stripped

        if is_table_row:
            close_lists()
            in_table = True
            table_rows.append(stripped)
            i += 1
            continue
        else:
            if in_table:
                flush_table()

        # ── Horizontal rule ───────────────────────────────────
        if re.match(r'^[\-\*]{3,}\s*$', stripped):
            out.append(r"\medskip\noindent\rule{\columnwidth}{0.4pt}\medskip")
            i += 1
            continue

        # ── Headings ──────────────────────────────────────────
        h_match = re.match(r'^(#{1,4})\s+(.*)', stripped)
        if h_match:
            close_lists()
            level  = len(h_match.group(1))
            htext  = h_match.group(2).strip()

            # Title (single #)
            if level == 1 and not title_emitted:
                out.append(r"\title{" + inline_md_to_tex(htext) + r"}")
                title_emitted = True
                # Collect author block (next non-blank lines until ## Abstract)
                i += 1
                while i < len(lines):
                    l2 = lines[i].rstrip('\r')
                    if re.match(r'^##\s+', l2):
                        break
                    if l2.strip():
                        author_lines.append(l2.strip())
                    i += 1
                # Emit author / affiliation
                if author_lines:
                    out.append(r"\author{" + " \\\\\n".join(
                        inline_md_to_tex(al) for al in author_lines
                    ) + r"}")
                out.append(r"\maketitle")
                out.append("")
                continue

            # Abstract
            if level == 2 and htext.lower() == 'abstract':
                in_abstract = True
                # Collect abstract text
                i += 1
                abs_lines = []
                while i < len(lines):
                    l2 = lines[i].rstrip('\r')
                    if re.match(r'^##?\s+', l2):
                        break
                    abs_lines.append(l2.strip())
                    i += 1
                abstract_body = " ".join(x for x in abs_lines if x)
                out.append(r"\begin{abstract}")
                out.append(inline_md_to_tex(abstract_body))
                out.append(r"\end{abstract}")
                out.append("")
                in_abstract   = False
                after_abstract = True
                continue

            # Index terms (after abstract)
            if level == 2 and 'index term' in htext.lower():
                i += 1
                term_lines = []
                while i < len(lines):
                    l2 = lines[i].rstrip('\r')
                    if re.match(r'^##?\s+', l2):
                        break
                    if l2.strip():
                        term_lines.append(l2.strip())
                    i += 1
                terms = " ".join(term_lines)
                out.append(r"\begin{IEEEkeywords}")
                out.append(inline_md_to_tex(terms))
                out.append(r"\end{IEEEkeywords}")
                out.append("")
                continue

            # References section
            if level == 2 and 'reference' in htext.lower():
                in_references = True
                out.append(r"\begin{thebibliography}{99}")
                i += 1
                continue

            # Appendix
            if level == 2 and 'appendix' in htext.lower():
                out.append(r"\appendix")
                out.append(r"\section{" + inline_md_to_tex(re.sub(r'^Appendix\s+[A-Z]\.\s*', '', htext)) + r"}")
                i += 1
                continue

            # Map level → command
            cmds = {1: r'\section', 2: r'\section', 3: r'\subsection', 4: r'\subsubsection'}
            cmd = cmds.get(level, r'\paragraph')
            # Strip leading roman-numeral section labels (I., II.3, etc.) — keep them for IEEE
            out.append(f"{cmd}{{{inline_md_to_tex(htext)}}}")
            out.append("")
            i += 1
            continue

        # ── Reference list item ───────────────────────────────
        if in_references:
            ref = parse_reference_line(stripped)
            if ref:
                key, text = ref
                # Convert italic markers in reference text
                text = re.sub(r'\*(.+?)\*', r'\\textit{\1}', text)
                text = text.replace('_', r'\_')
                out.append(f"\\bibitem{{{key}}}")
                out.append(text)
                out.append("")
            elif stripped == '' or stripped == '---':
                pass
            else:
                # continuation of previous bibitem or closing
                if stripped.startswith('[') and not re.match(r'^\[\d+\]', stripped):
                    pass
                elif stripped:
                    out.append(inline_md_to_tex(stripped))
            i += 1
            continue

        # ── Figure (Markdown image) ────────────────────────────
        fig_match = re.match(r'^\s*!\[([^\]]*)\]\(([^)]+)\)\s*$', stripped)
        if fig_match:
            close_lists()
            out.append(emit_figure(fig_match.group(1), fig_match.group(2)))
            i += 1
            continue

        # ── Lists ─────────────────────────────────────────────
        # Unordered
        ul_match = re.match(r'^(\s*)[\*\-\+]\s+(.*)', stripped)
        # Ordered
        ol_match = re.match(r'^(\s*)\d+\.\s+(.*)', stripped)

        if ul_match:
            text = ul_match.group(2)
            if not list_stack or list_stack[-1] != 'itemize':
                if list_stack and list_stack[-1] == 'enumerate':
                    out.append(r"\end{enumerate}")
                    list_stack.pop()
                out.append(r"\begin{itemize}")
                list_stack.append('itemize')
            out.append(r"  \item " + inline_md_to_tex(text))
            i += 1
            continue
        elif ol_match:
            text = ol_match.group(2)
            if not list_stack or list_stack[-1] != 'enumerate':
                if list_stack and list_stack[-1] == 'itemize':
                    out.append(r"\end{itemize}")
                    list_stack.pop()
                out.append(r"\begin{enumerate}")
                list_stack.append('enumerate')
            out.append(r"  \item " + inline_md_to_tex(text))
            i += 1
            continue
        else:
            if list_stack and not stripped:
                # blank line after list — close it
                close_lists()
            elif list_stack and stripped and not ul_match and not ol_match:
                # non-list content — close open list
                close_lists()

        # ── Blank line ────────────────────────────────────────
        if not stripped:
            out.append("")
            i += 1
            continue

        # ── Regular paragraph text ────────────────────────────
        out.append(inline_md_to_tex(stripped))
        i += 1

    # Flush any open environments
    flush_table()
    flush_blockquote()
    flush_code()
    close_lists()

    if in_references:
        out.append(r"\end{thebibliography}")
        out.append("")

    out.append(POSTAMBLE)
    return "\n".join(out)


# ──────────────────────────────────────────────────────────────
# 8.  Entry point
# ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    with open(INPUT, "r", encoding="utf-8") as f:
        md_text = f.read()

    tex = convert(md_text)

    with open(OUTPUT, "w", encoding="utf-8") as f:
        f.write(tex)

    print(f"[SUCCESS] Wrote {OUTPUT}  ({len(tex):,} chars, {tex.count(chr(10)):,} lines)")
