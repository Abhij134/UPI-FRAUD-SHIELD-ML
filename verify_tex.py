text = open('UPI_Fraud_Shield_ML_Paper_Final.tex', encoding='utf-8').read()
lines = text.splitlines()
import re

results = []

# Check 1: IEEEtran documentclass
has_docclass = r'\documentclass[journal,10pt,twocolumn]{IEEEtran}' in text
results.append(f"[1] IEEEtran documentclass:    {'PASS' if has_docclass else 'FAIL'}")

# Check 2: Ends with \end{document}
last_nonempty = next((l for l in reversed(lines) if l.strip()), '')
has_end_doc = r'\end{document}' in last_nonempty
results.append(f"[2] Ends with end document:    {'PASS' if has_end_doc else 'FAIL'} | last line: {repr(last_nonempty[:70])}")

# Check 3: Data dictionary key features
keys = ['amount_log','otp_request_device_consistency','keyboard_input_speed',
        'transaction_amount_vs_sender_history','geographic_disparity']
found = sum(1 for k in keys if k in text)
results.append(f"[3] Key feature cols in text:  {found}/{len(keys)} {'PASS' if found==len(keys) else 'FAIL'}")

# Check 4: Booktabs
has_booktabs = '\\toprule' in text
results.append(f"[4] Booktabs toprule:          {'PASS' if has_booktabs else 'FAIL'}")

# Check 5 & 6: Bibliography
has_bib = r'\begin{thebibliography}' in text
bibitem_count = text.count(r'\bibitem')
results.append(f"[5] Bibliography environment:  {'PASS' if has_bib else 'FAIL'}")
results.append(f"[6] bibitem entries:           {bibitem_count} (expected ~51)")

# Check 7: Residual raw unicode
bad = [c for c in ['\u20b9','\u2192','\u2265','\u2264','\u2014','\u2013','\xd7'] if c in text]
results.append(f"[7] Residual unescaped unicode: {'FAIL: ' + str(bad) if bad else 'PASS (none found)'}")

# Check 8: Figures
fig_count = text.count(r'\begin{figure}')
results.append(f"[8] Figure environments:       {fig_count}")

# Check 9 & 10
results.append(f"[9] Total lines:               {len(lines)}")
results.append(f"[10] Total chars:              {len(text):,}")

with open('verify_tex_report.txt', 'w', encoding='utf-8') as rpt:
    rpt.write('\n'.join(results) + '\n')
    if bad:
        rpt.write(f'   Bad chars: {repr(bad)}\n')
print("Report written to verify_tex_report.txt")

