"""
Fix two classes of errors in UPI_Fraud_Shield_ML_Paper_Final.tex:

1. TABLE COLUMN MISMATCH: colspec was being truncated by the generator
   (regex stopped at first '}' inside 'p{...}'). Fix: recount actual
   columns from the header row and build a proper colspec.

2. MATH MODE ERRORS: stray inline formatting commands were left inside
   \[...\] display math blocks, corrupting them. Fix: clean those up.

This script reads the .tex, surgically patches every tabular environment
to match its actual column count, then fixes malformed display-math blocks.
"""

import re

src = open("UPI_Fraud_Shield_ML_Paper_Final.tex", encoding="utf-8").read()

# ─── STEP 1: Fix every \begin{tabular}{...} whose colspec doesn't match ──────
# Strategy: for each tabular, count '&' in the header row to get actual ncols,
# then rebuild a valid colspec.

def col_widths(ncols):
    """Return a reasonable column spec for ncols columns."""
    if ncols == 1:
        return "l"
    elif ncols == 2:
        return "lp{6cm}"
    elif ncols == 3:
        return "llp{5cm}"
    elif ncols == 4:
        return "lp{3cm}p{3cm}l"
    elif ncols == 5:
        return "lp{2.8cm}p{2.5cm}p{2.5cm}l"
    elif ncols == 6:
        return "lp{2.3cm}p{2cm}p{2cm}p{2cm}l"
    elif ncols == 7:
        return "lp{2cm}p{2cm}p{1.8cm}p{1.8cm}p{1.8cm}l"
    elif ncols == 8:
        return "lp{1.8cm}p{1.8cm}p{1.5cm}p{1.5cm}p{1.5cm}p{1.5cm}l"
    else:
        return "l" * ncols

TAB_START = re.compile(r'\\begin\{tabular\}\{[^}]*\}')  # Note: may be truncated
TABULAR_BLOCK = re.compile(
    r'(\\begin\{tabular\}\{)([^\n]+?)(\})(.*?)(\\end\{tabular\})',
    re.DOTALL
)

def fix_tabular(m):
    old_spec = m.group(2)
    body     = m.group(4)
    
    # Find the header row (first non-toprule line with & and \\)
    header_line = None
    for line in body.splitlines():
        ls = line.strip()
        if '&' in ls and '\\\\' in ls and 'toprule' not in ls and 'midrule' not in ls:
            header_line = ls
            break
    
    if header_line is None:
        return m.group(0)  # Can't fix, leave as-is
    
    actual_ncols = header_line.count('&') + 1
    needed_spec  = col_widths(actual_ncols)
    
    if old_spec == needed_spec:
        return m.group(0)  # Already correct
    
    # Rebuild with corrected colspec
    return m.group(1) + needed_spec + m.group(3) + m.group(4) + m.group(5)

fixed = TABULAR_BLOCK.sub(fix_tabular, src)

# ─── STEP 2: Fix malformed display-math blocks ───────────────────────────────
# The problematic line (272) contains \textit{...} inside \[...\], which is
# caused by the inline_md_to_tex running on text that already had LaTeX \[ \].
# We clean display-math blocks by removing italic/bold commands that snuck in.
# But specifically the admissibility criterion equation has mixed text/math.
# Replace the specific broken line with a clean version.

BAD_MATH = r'\[\text{(i)}\ \ X_j\ \text{is}\ \mathcal{F}\textit{{\tau^-}\text{-measurable}\quad\text{and}\quad \text{(ii)}\ \ P}{\text{train}}(X_j\mid y)=P_{\text{deploy}}(X_j\mid y). \tag{7}\]'

GOOD_MATH = (
    r'\begin{align}' + '\n'
    r'  &\text{(i)}\ X_j \text{ is } \mathcal{F}_{\tau^-}\text{-measurable,}' + '\n'
    r'  \quad\text{and}\nonumber\\' + '\n'
    r'  &\text{(ii)}\ P_{\text{train}}(X_j \mid y) = P_{\text{deploy}}(X_j \mid y). \tag{7}' + '\n'
    r'\end{align}'
)

fixed = fixed.replace(BAD_MATH, GOOD_MATH)

# ─── STEP 3: Verify column-count fix worked ────────────────────────────────
remaining = []
for m in TABULAR_BLOCK.finditer(fixed):
    spec = m.group(2)
    body = m.group(4)
    for line in body.splitlines():
        ls = line.strip()
        if '&' in ls and '\\\\' in ls and 'toprule' not in ls and 'midrule' not in ls:
            ncols_from_spec = len(re.findall(r'[lcr]|p\{[^}]+\}', spec))
            ncols_from_row  = ls.count('&') + 1
            if ncols_from_spec != ncols_from_row:
                remaining.append(f"spec={spec!r} row_cols={ncols_from_row}: {ls[:80]}")
            break

if remaining:
    print(f"WARNING: {len(remaining)} tables still mismatched:")
    for r in remaining[:10]:
        print(" ", r)
else:
    print("All tabular column specs match their header rows.")

open("UPI_Fraud_Shield_ML_Paper_Final.tex", "w", encoding="utf-8").write(fixed)
print("Wrote fixed UPI_Fraud_Shield_ML_Paper_Final.tex")
