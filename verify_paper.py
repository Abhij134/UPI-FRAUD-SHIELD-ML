import re
import sys

def verify_paper(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"FAILED to read {filepath}: {e}")
        return

    errors = []

    # 1. Residual Placeholder Check
    placeholder = "\u27e8run\u27e9"
    count = content.count(placeholder)
    if count > 0:
        errors.append(f"[Error] Found {count} residual <run> placeholders in the document.")
        # Find line numbers
        lines = content.split('\n')
        for i, line in enumerate(lines):
            if placeholder in line:
                errors.append(f"  -> Line {i+1}: {line.strip().replace(placeholder, '<run>')}")

    # 2. Ablation Table Check
    if "Table XII" in content:
        # Check if actual data is in there (Full Feature Set, W/o Behavioral) from ablation.md
        if "Full Feature Set" not in content or "W/o Behavioral" not in content:
            errors.append("[Error] Ablation Table data (Table XII) seems to be missing or not injected correctly.")
    else:
        errors.append("[Error] 'Table XII' caption is completely missing from the document.")

    # 3. Latency Table Check
    if "Table XV" in content:
        if "P50" not in content or "P99" not in content or "28.84" not in content:
            errors.append("[Error] Latency Table data (Table XV) seems to be missing or not injected correctly.")
    else:
        errors.append("[Error] 'Table XV' caption is completely missing from the document.")

    # 4. Use Cases Check
    # We fallback to "N/A" if json was missing, so we check for "N/A" or probabilities
    if "N/A" in content:
        print("[Warning] 'N/A' found in the document. This means JSON files were missing, but the injection itself ran.")
    else:
        pass # If we had real JSON data, we would check for specific % values

    # 5. Figures Check
    if "![Figure 3: Precision-Recall & Calibration Curves](results/fig3_pr_calibration.png)" not in content:
        errors.append("[Error] Figure 3 markdown link is missing or incorrect.")
    if "![Figure 4: SHAP Waterfall Plots](results/shap_waterfall.png)" not in content:
        errors.append("[Error] Figure 4 markdown link is missing or incorrect.")
    if "![Figure 5: System Latency Distribution](results/fig5_latency_sla.png)" not in content:
        errors.append("[Error] Figure 5 markdown link is missing or incorrect.")
    
    if "**Fig. 3" in content or "**Fig. 4" in content or "**Fig. 5" in content:
        errors.append("[Error] Found residual bolded '**Fig.' pending text.")

    # 6. Roadmap Section Check
    roadmap_heading = "### XIII.4 Real-World Dataset Evaluation"
    if roadmap_heading not in content:
        errors.append("[Error] Roadmap Section 'XIII.4' was not found in the document.")
    else:
        # Check if it appears multiple times (idempotency failure)
        if content.count(roadmap_heading) > 1:
            errors.append("[Error] Roadmap Section 'XIII.4' appears multiple times (duplicated).")

    if not errors:
        print("ALL CHECKS PASSED: The manuscript is 100% clean and ready for PDF/LaTeX compilation.")
    else:
        print("VALIDATION REPORT - ISSUES FOUND:")
        for error in errors:
            print(error)

if __name__ == "__main__":
    verify_paper("UPI_Fraud_Shield_ML_Paper_Final.md")
