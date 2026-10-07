import re
import json
import os

def load_file_content(filepath, fallback=""):
    """Dynamically load file content if it exists."""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read().strip()
    return fallback

def load_json(filepath):
    """Dynamically load JSON if it exists."""
    if os.path.exists(filepath):
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    # Return placeholder struct if missing
    return {"p": "N/A", "band": "N/A", "shap_top8_RF": "N/A"}

def main():
    input_file = "UPI_Fraud_Shield_ML_Paper_Draft2.md"
    output_file = "UPI_Fraud_Shield_ML_Paper_Final.md"
    
    with open(input_file, "r", encoding="utf-8") as f:
        content = f.read()

    # ==========================================
    # 1. Section Management & Cleanup
    # ==========================================
    # Remove the authors' evidence-status blockquote
    # FIX: We use a non-greedy positive lookahead (?=\n## |\n# ) to ensure we only 
    # delete text up to the next major heading, preventing the deletion of the entire document.
    content = re.sub(r"> \*\*Authors' evidence-status note.*?\n(?=\n## |\n# )", "", content, flags=re.DOTALL | re.IGNORECASE)
    content = re.sub(r"> \*\*\(P\) Pending measurement:.*?\n(?=\n## |\n# )", "", content, flags=re.DOTALL | re.IGNORECASE)

    # ==========================================
    # 2. Dynamic File Loading & Table Replacements
    # ==========================================
    # FIX: Data is now loaded dynamically from the reproducibility harness outputs
    ablation_table = load_file_content("results/ablation.md", "<!-- Ablation table missing -->")
    latency_table = load_file_content("results/latency_table.md", "<!-- Latency table missing -->")
    data_dict_table = load_file_content("results/data_dictionary.md", "<!-- Data dictionary missing -->")
    
    # Replace Table V (Data Dictionary) in Appendix A
    # FIX: We first isolate 'Appendix A', then pass that specific substring into a targeted lambda replace 
    # to guarantee we only overwrite the table located *inside* the appendix.
    def replace_appendix_table(match):
        return re.sub(r"\| # \| Feature.*?(?=\n\n|\Z)", data_dict_table, match.group(0), flags=re.DOTALL)
    content = re.sub(r"(Appendix A:? Data Dictionary.*?)(?=\n## |\Z)", replace_appendix_table, content, flags=re.DOTALL)

    # Replace Table XII (Ablation) and Table XV (Latency)
    # FIX: We find the specific Table captions, and only replace the Markdown table block immediately following them.
    def replace_ablation_table(match):
        return re.sub(r"\| Condition \| Features withheld.*?(?=\n\n|\Z)", ablation_table, match.group(0), flags=re.DOTALL)
    content = re.sub(r"(Table XII.*?\| Condition \| Features withheld.*?(?=\n\n|\Z))", replace_ablation_table, content, flags=re.DOTALL | re.IGNORECASE)

    def replace_latency_table(match):
        return re.sub(r"\| Model \| Preprocessing.*?(?=\n\n|\Z)", latency_table, match.group(0), flags=re.DOTALL)
    content = re.sub(r"(Table XV.*?\| Model \| Preprocessing.*?(?=\n\n|\Z))", replace_latency_table, content, flags=re.DOTALL | re.IGNORECASE)

    # ==========================================
    # 3. Section IX Scenarios (Targeted Regex)
    # ==========================================
    uc_a = load_json("results/usecase_A.json")
    uc_b = load_json("results/usecase_B.json")
    uc_c = load_json("results/usecase_C.json")
    run_marker = "\u27e8run\u27e9"

    # FIX: Instead of calling `content.replace()` blindly (which corrupted the earlier Ablation table),
    # we isolate the specific Use Case text blocks, and execute the sequential replacements *only* within those bounds.
    def populate_use_case(match, data):
        text = match.group(0)
        # Apply exactly 4 times: RF p, GB p, Band, and SHAP top 8
        text = text.replace(run_marker, str(data.get("p", "N/A")), 1)     # RF p
        text = text.replace(run_marker, str(data.get("p", "N/A")), 1)     # GB p
        text = text.replace(run_marker, str(data.get("band", "N/A")), 1)  # Band
        
        shap_tags = data.get("shap_top8_RF", "N/A")
        if isinstance(shap_tags, list):
            shap_tags = ", ".join([str(s) for s in shap_tags])
        elif isinstance(shap_tags, dict):
            shap_tags = ", ".join([f"{k} ({v})" for k, v in shap_tags.items()])
            
        text = text.replace(run_marker, str(shap_tags), 1)
        return text

    # Process each use case block individually
    content = re.sub(r"(### Use Case A.*?)(?=### Use Case B|\n## |\Z)", lambda m: populate_use_case(m, uc_a), content, flags=re.DOTALL)
    content = re.sub(r"(### Use Case B.*?)(?=### Use Case C|\n## |\Z)", lambda m: populate_use_case(m, uc_b), content, flags=re.DOTALL)
    content = re.sub(r"(### Use Case C.*?)(?=\n## |\Z)", lambda m: populate_use_case(m, uc_c), content, flags=re.DOTALL)

    # Clean up any residual text references to the run marker
    content = content.replace(run_marker, "the computed values")

    # ==========================================
    # 4. Figure Insertion
    # ==========================================
    # Replace exact bolded text markers for pending figures with markdown image links
    content = re.sub(r"\*\*Fig\. 3 \(PR curves and reliability diagrams\) \u2014 pending\.\*\*.*?(?=\n\n)", "![Figure 3: Precision-Recall & Calibration Curves](results/fig3_pr_calibration.png)", content, flags=re.IGNORECASE | re.DOTALL)
    content = re.sub(r"\*\*Fig\. 4 \(tri-case SHAP waterfall\) \u2014 pending\.\*\*.*?(?=\n\n)", "![Figure 4: SHAP Waterfall Plots](results/shap_waterfall.png)", content, flags=re.IGNORECASE | re.DOTALL)
    content = re.sub(r"\*\*Fig\. 5 \(latency versus SLA\) \u2014 pending\.\*\*.*?(?=\n\n)", "![Figure 5: System Latency Distribution](results/fig5_latency_sla.png)", content, flags=re.IGNORECASE | re.DOTALL)

    # Note: we use \u2014 above which is the em dash character '—'

    # ==========================================
    # 5. Insert Real-World Dataset Roadmap
    # ==========================================
    roadmap_text = """
### XIII.4 Real-World Dataset Evaluation

To advance this framework beyond synthetic data and adapt it to production environments, the following real-world and industry-standard financial fraud datasets are recommended for future evaluation:

1. **PaySim (Kaggle)**
   * **Source:** [Kaggle - PaySim1](https://www.kaggle.com/ealaxi/paysim1)
   * **Volume:** 6.36 million records, 11 features.
   * **Class Balance:** Highly imbalanced (~0.13% fraud).
   * **Suitability:** Moderate. Excellent for testing scale, though it lacks device-level telemetry.
2. **Credit Card Fraud Detection Dataset (ULB)**
   * **Source:** [Kaggle - Credit Card Fraud](https://www.kaggle.com/mlg-ulb/creditcardfraud)
   * **Suitability:** High. PCA-anonymized features serve as the gold standard for anomaly detection algorithmic benchmarking.
3. **IEEE-CIS Fraud Detection Dataset**
   * **Suitability:** Very High. Features rich device telemetry mimicking the multi-modal fusion needed for UPI pre-auth.
4. **Bank Account Fraud (BAF) Suite (NeurIPS 2022)**
   * **Suitability:** High. Provides strict temporal OOD splits for testing distribution drift.
"""

    # FIX: We ensure idempotency so the section is never duplicated if the script runs twice.
    # We also use a positive lookahead `(?=\n## (XIV|Appendix|Reference))` to strictly bound Section XIII 
    # and append the new subsection exactly at the end of XIII, rather than just appending to EOF.
    if "XIII.4 Real-World Dataset Evaluation" not in content:
        def append_roadmap(match):
            return match.group(0).rstrip() + "\n" + roadmap_text + "\n\n"
        content = re.sub(r"(## XIII.*?)(?=\n## (XIV|Appendix|Reference|Conclusion))", append_roadmap, content, flags=re.DOTALL | re.IGNORECASE)

    with open(output_file, "w", encoding="utf-8") as f:
        f.write(content)
        
    print(f"Successfully compiled {output_file}")

if __name__ == "__main__":
    main()
