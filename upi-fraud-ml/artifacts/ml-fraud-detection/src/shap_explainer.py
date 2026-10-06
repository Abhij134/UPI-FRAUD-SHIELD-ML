import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import io

FEATURE_LABELS = {
    # Transaction
    "amount_log":                             "Txn Amount (log ₹)",
    "session_duration":                        "Session Duration (s)",
    "authentication_attempts":                 "Auth Attempts",
    "transaction_amount_vs_sender_history":    "Amount vs Sender History",
    "geographic_disparity":                    "Geographic Disparity (km)",
    "transaction_time_of_day":                 "Hour of Day",
    "merchant_category_code":                  "Merchant Category (MCC)",
    "session_source":                          "Session Source",
    "time_between_link_click_and_transaction": "Link→Txn Time (s)",
    # Behavioural biometrics (previously missing)
    "dns_lookup_age":                          "DNS Age (days)",
    "input_timing_consistency":               "Input Timing Consistency",
    "app_switching_frequency":                 "App Switches",
    "keyboard_input_speed":                    "Keyboard Speed (kps)",
    "screen_active_time":                      "Screen Active Time (s)",
    "geographic_location_vs_ip":               "GPS vs IP Mismatch",
    "background_data_usage":                   "Background Data (MB)",
    # OTP & auth
    "time_between_otp_generation_and_input":   "OTP Delay (s)",
    "pin_entry_speed":                         "PIN Entry Speed",
    "otp_request_frequency":                   "OTP Request Frequency",
    "otp_request_device_consistency":          "OTP Device Consistent",
    # Transaction history
    "transaction_velocity":                    "Txns Last 24h",
    "failed_transaction_count":                "Failed Txns (7d)",
    # Encoded categoricals
    "authorization_method":                    "Auth Method",
    "transaction_type":                        "Transaction Type",
    # Collect request
    "request_amount_roundness":                "Amount Roundness",
    "request_frequency":                       "Request Frequency (7d)",
    "request_acceptance_rate":                 "Request Accept Rate",
    "time_to_respond_to_request":              "Response Time (s)",
    "requester_account_age":                   "Requester Account Age",
    "relationship_to_requester":               "Relationship to Requester",
    # UPI Handle (previously missing)
    "upi_handle_age":                          "UPI Handle Age (days)",
    "handle_similarity_score":                 "Handle Similarity Score",
    "handle_contains_official_terms":          "Official Terms in Handle",
    "handle_transaction_history":              "Handle Txn History",
    "business_name_match":                     "Business Name Match",
    "social_media_presence":                   "Social Media Presence",
}

PRECAUTIONS = {
    "transaction_amount_vs_sender_history": {
        "threshold": 3.0,
        "high": "⚠️ Amount is unusually high vs your spending pattern. Verify before confirming.",
    },
    "transaction_velocity": {
        "threshold": 15,
        "high": "🔁 Too many transactions in 24 hours. Your account may be compromised.",
    },
    "failed_transaction_count": {
        "threshold": 3,
        "high": "❌ Multiple recent failed transactions detected — common before a fraud attempt.",
    },
    "otp_request_frequency": {
        "threshold": 3,
        "high": "📩 High OTP request rate. If you didn't request these, block access immediately.",
    },
    "time_between_otp_generation_and_input": {
        "threshold": 120,
        "high": "⏱️ OTP entered very late — may indicate manual relay by a fraudster.",
    },
    "geographic_disparity": {
        "threshold": 500,
        "high": "📍 Transaction origin is far from your usual city. Verify it's you.",
    },
    "geographic_location_vs_ip": {
        "threshold": 0.6,
        "high": "🌐 Your GPS location doesn't match your IP location — possible proxy/VPN.",
    },
    "handle_similarity_score": {
        "threshold": 0.7,
        "high": "🎭 Payee UPI handle looks similar to an official bank/gov handle — check carefully.",
    },
    "handle_contains_official_terms": {
        "threshold": 0.5,
        "high": "🏛️ Payee handle contains official-sounding terms (bank, npci, gov). Verify it's legitimate.",
    },
    "authentication_attempts": {
        "threshold": 2,
        "high": "🔐 Multiple failed auth attempts — someone may be trying to brute-force your PIN.",
    },
    "requester_account_age": {
        "threshold": None,
        "reverse": True,
        "reverse_threshold": 30,
        "high": "🆕 Requester's account is very new — new accounts are frequently used for mule fraud.",
    }
}

GENERAL_PRECAUTIONS = [
    "🔒 Never share your UPI PIN or OTP with anyone — not even bank officials.",
    "📞 If someone asks you to 'receive money', it's a scam. Decline and report.",
    "✅ Always double-check the receiver's UPI ID before confirming a payment.",
    "📱 Keep your UPI app updated and enable biometric lock.",
    "🚫 Avoid using UPI on public Wi-Fi. Use mobile data instead.",
]


def get_shap_values(model, X_input: pd.DataFrame, X_background: pd.DataFrame, model_name: str):
    try:
        if model_name in ["Random Forest", "Gradient Boosting", "Decision Tree"]:
            bg = X_background.sample(min(100, len(X_background)), random_state=42)
            explainer = shap.TreeExplainer(model, data=bg)
        else:
            explainer = shap.KernelExplainer(
                model.predict_proba,
                shap.sample(X_background, 50),
            )
        shap_vals = explainer(X_input)
        if hasattr(shap_vals, "values") and len(shap_vals.values.shape) == 3:
            return shap_vals[..., 1]
        return shap_vals
    except Exception:
        return None


def plot_shap_waterfall(shap_vals, feature_names, feature_values, base_value: float, fraud_prob: float) -> bytes:
    vals = shap_vals if isinstance(shap_vals, np.ndarray) else shap_vals.values.flatten()
    feat_labels = [FEATURE_LABELS.get(f, f) for f in feature_names]

    top_idx = np.argsort(np.abs(vals))[::-1][:12]
    top_vals = vals[top_idx]
    top_labels = [feat_labels[i] for i in top_idx]
    top_feat_vals = [feature_values[i] for i in top_idx]

    order = np.argsort(top_vals)
    plot_vals   = top_vals[order]
    plot_labels = [top_labels[i] for i in order]
    plot_fvals  = [top_feat_vals[i] for i in order]

    fig, ax = plt.subplots(figsize=(10, 6))
    fig.patch.set_facecolor("#0e1117")
    ax.set_facecolor("#161b22")

    colors = ["#EF553B" if v > 0 else "#00CC96" for v in plot_vals]
    bars = ax.barh(range(len(plot_vals)), plot_vals, color=colors, alpha=0.85, height=0.6, edgecolor="#333")

    for i, (bar, val, fval) in enumerate(zip(bars, plot_vals, plot_fvals)):
        lbl = f"{val:+.3f}  (val={fval:.2f})"
        xp  = bar.get_width() + 0.002 if val >= 0 else bar.get_width() - 0.002
        ha  = "left" if val >= 0 else "right"
        ax.text(xp, i, lbl, va="center", ha=ha, color="white", fontsize=9)

    ax.set_yticks(range(len(plot_labels)))
    ax.set_yticklabels(plot_labels, color="white", fontsize=10)
    ax.axvline(0, color="white", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("SHAP Value (impact on fraud probability)", color="white", fontsize=11)
    ax.set_title(
        f"SHAP Explanation — Base: {base_value:.3f} → Predicted: {fraud_prob:.3f}",
        color="white", fontsize=13, fontweight="bold", pad=12,
    )
    ax.tick_params(colors="white")
    for sp in ax.spines.values():
        sp.set_color("#444")

    red_patch   = mpatches.Patch(color="#EF553B", label="Increases fraud risk")
    green_patch = mpatches.Patch(color="#00CC96", label="Reduces fraud risk")
    ax.legend(handles=[red_patch, green_patch], facecolor="#0e1117", labelcolor="white",
              fontsize=9, loc="lower right")

    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format="png", dpi=130, bbox_inches="tight",
                facecolor="#0e1117", edgecolor="none")
    buf.seek(0)
    plt.close()
    return buf.read()


def get_precautions(feature_names: list, feature_values: list) -> list:
    out = []
    for feat, val in zip(feature_names, feature_values):
        rule = PRECAUTIONS.get(feat)
        if not rule:
            continue
        if rule.get("reverse"):
            if val < rule["reverse_threshold"] and rule.get("high"):
                out.append(rule["high"])
        elif rule.get("threshold") is not None and val > rule["threshold"] and rule.get("high"):
            out.append(rule["high"])
    return out
