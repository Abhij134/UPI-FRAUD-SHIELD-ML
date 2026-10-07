"""Post-process UPI_Fraud_Shield_ML_Paper_Final.tex to sweep remaining non-ASCII."""

# Characters that are FINE to keep (accented letters in UTF-8 with inputenc)
KEEP_CHARS = set("àáâãäåæçèéêëìíîïðñòóôõöùúûüýþÿ"
                 "ÀÁÂÃÄÅÆÇÈÉÊËÌÍÎÏÐÑÒÓÔÕÖÙÚÛÜÝÞŸ"
                 "ăćčďěłňřšťžŽŠėîĖ")

EXTRA_MAP = {
    "\u00b1": r"$\pm$",
    "\u00d7": r"$\times$",
    "\u2013": "--",
    "\u2014": "---",
    "\u20b9": r"\rupee{}",
    "\u2192": r"$\rightarrow$",
    "\u2212": r"$-$",
    "\u2248": r"$\approx$",
    "\u2260": r"$\neq$",
    "\u2264": r"$\leq$",
    "\u2265": r"$\geq$",
    "\u2018": "`",
    "\u2019": "'",
    "\u201c": "``",
    "\u201d": "''",
    "\u2026": r"\ldots{}",
    "\u00a0": " ",
    "\u2500": "-",
    "\u2502": "|",
}

text = open("UPI_Fraud_Shield_ML_Paper_Final.tex", encoding="utf-8").read()

for uni, latex in EXTRA_MAP.items():
    text = text.replace(uni, latex)

open("UPI_Fraud_Shield_ML_Paper_Final.tex", "w", encoding="utf-8").write(text)
print("Post-processing done.")

# Final check
import unicodedata
bad = [(ord(c), unicodedata.name(c,'?')) for c in set(text)
       if ord(c) > 127 and c not in KEEP_CHARS]
if bad:
    print(f"Still {len(bad)} non-ASCII chars remaining:")
    for cp, name in sorted(bad):
        print(f"  U+{cp:04X}  {name}")
else:
    print("All non-ASCII resolved. LaTeX file is 100% clean.")
