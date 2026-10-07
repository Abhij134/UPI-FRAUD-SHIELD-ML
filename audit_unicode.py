import unicodedata

text = open('UPI_Fraud_Shield_ML_Paper_Final.tex', encoding='utf-8').read()
seen = {}
for i, c in enumerate(text):
    if ord(c) > 127 and c not in seen:
        ctx = text[max(0,i-30):i+30]
        seen[c] = (ord(c), unicodedata.name(c,'?'), ctx)

lines = []
for c, (cp, name, ctx) in sorted(seen.items(), key=lambda x: x[1][0]):
    safe_ctx = ctx.encode('ascii', 'backslashreplace').decode('ascii')
    lines.append(f'U+{cp:04X}  {name}:  ...{safe_ctx}...')

out = '\n'.join(lines)
open('remaining_unicode.txt', 'w', encoding='utf-8').write(out)
print(f'{len(seen)} distinct non-ASCII chars; see remaining_unicode.txt')
