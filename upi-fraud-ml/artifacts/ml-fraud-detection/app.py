import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from imblearn.over_sampling import SMOTE

from src.data_generator import (
    get_train_test_data,
    SESSION_SOURCES, AUTH_METHODS, TRANSACTION_TYPES,
    RELATIONSHIP_OPTIONS, BUSINESS_MATCH_OPTIONS, SOCIAL_OPTIONS,
)
from src.feature_engineering import (
    get_scaler,
)
from src.models import MODEL_REGISTRY, train_model, predict_model, get_feature_importance
from src.evaluator import compute_metrics, get_roc_curve
from src.auth import login, register
from src.history import save_prediction, get_history, clear_history
from src.shap_explainer import (
    get_shap_values, plot_shap_waterfall, get_precautions,
    GENERAL_PRECAUTIONS, FEATURE_LABELS,
)

st.set_page_config(
    page_title="UPI Fraud Shield ML",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
html, body, [data-testid="stAppViewContainer"] {
    background: #080c14 !important;
    color: #e2e8f0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stHeader"] { background: transparent !important; }
[data-testid="stSidebar"] { background: #0d1117 !important; }
[data-testid="stSidebarContent"] { background: #0d1117 !important; }

.stButton > button {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.6rem 1.4rem !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 25px rgba(99,102,241,0.5) !important;
}

.stTextInput > div > div > input,
.stNumberInput > div > div > input {
    background: #0d1117 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
    padding: 0.5rem 0.75rem !important;
}
.stTextInput > div > div > input:focus,
.stNumberInput > div > div > input:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}
.stSelectbox > div > div {
    background: #0d1117 !important;
    border: 1px solid #2d3748 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}
label[data-testid="stWidgetLabel"] > div > p,
label[data-testid="stWidgetLabel"] p {
    font-size: 0.82rem !important;
    color: #94a3b8 !important;
    font-weight: 500 !important;
    margin-bottom: 2px !important;
}

.stTabs [data-baseweb="tab-list"] {
    background: #0d1117 !important;
    border-radius: 12px !important;
    padding: 4px !important;
    gap: 4px !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: #94a3b8 !important;
    border-radius: 8px !important;
    font-weight: 500 !important;
    padding: 0.4rem 1rem !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}

.input-section {
    background: linear-gradient(135deg, #0f1420, #141a28);
    border: 1px solid #1e2a3a;
    border-radius: 14px;
    padding: 1.2rem 1.4rem 0.8rem;
    margin-bottom: 1rem;
}
.input-section-title {
    font-size: 0.78rem;
    font-weight: 700;
    color: #6366f1;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 0.8rem;
    display: flex;
    align-items: center;
    gap: 6px;
}

.metric-card {
    background: linear-gradient(135deg, #161b27, #1a2035);
    border: 1px solid #2d3748;
    border-radius: 14px;
    padding: 1.2rem 1.5rem;
    text-align: center;
}
.metric-val   { font-size: 2rem; font-weight: 700; }
.metric-lbl   { font-size: 0.78rem; color: #94a3b8; margin-top: 4px; text-transform: uppercase; letter-spacing: 0.05em; }
.metric-delta { font-size: 0.78rem; margin-top: 4px; color: #64748b; }

.risk-approved   { background:#064e3b; color:#34d399; border:1px solid #059669; }
.risk-suspicious { background:#78350f; color:#fbbf24; border:1px solid #d97706; }
.risk-high       { background:#7c2d12; color:#fb923c; border:1px solid #ea580c; }
.risk-blocked    { background:#450a0a; color:#f87171; border:1px solid #dc2626; }
.risk-badge {
    padding: 0.45rem 1.4rem;
    border-radius: 999px;
    font-size: 1rem;
    font-weight: 700;
    display: inline-block;
    letter-spacing: 0.06em;
}

.auth-card {
    background: linear-gradient(135deg, rgba(22,27,39,0.96), rgba(26,32,53,0.96));
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 20px;
    padding: 2.2rem;
    box-shadow: 0 25px 60px rgba(0,0,0,0.5);
}

.section-header {
    font-size: 1rem;
    font-weight: 700;
    color: #a5b4fc;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.6rem;
    border-left: 3px solid #6366f1;
    padding-left: 0.6rem;
}

.precaution-box {
    background: rgba(220,38,38,0.12);
    border-left: 4px solid #dc2626;
    border-radius: 0 10px 10px 0;
    padding: 0.7rem 1rem;
    margin: 0.35rem 0;
    color: #fca5a5;
    font-size: 0.9rem;
    line-height: 1.5;
}
.precaution-general {
    background: rgba(99,102,241,0.1);
    border-left: 4px solid #6366f1;
    border-radius: 0 10px 10px 0;
    padding: 0.65rem 1rem;
    margin: 0.3rem 0;
    color: #c7d2fe;
    font-size: 0.87rem;
    line-height: 1.5;
}

.hist-row {
    background: #0f1420;
    border: 1px solid #1e2a3a;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    margin-bottom: 0.5rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

.hdivider { height:1px; background:linear-gradient(90deg,transparent,#2d3748,transparent); margin:1rem 0; border:none; }

#MainMenu, footer, .stDeployButton { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state ──────────────────────────────────────────────────────────────
for k, v in [
    ("logged_in", False), ("username", None), ("user_data", {}),
    ("prediction_done", False), ("last_result", None),
]:
    if k not in st.session_state:
        st.session_state[k] = v


# ── ML pipeline (cached) ───────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Training ML models on real fraud data…")
def load_pipeline():
    X_train, X_test, y_train, y_test, label_encoders, feature_cols = get_train_test_data()
    print("\nFEATURE COLS:")
    print(feature_cols)

    # SMOTE only on training data to handle class imbalance
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)

    scaler = get_scaler(X_train_sm)
    results = {}
    for name in MODEL_REGISTRY:
        model   = train_model(name, X_train_sm, y_train_sm, scaler)
        y_pred, y_prob = predict_model(name, model, X_test, scaler)
        metrics = compute_metrics(y_test, y_pred, y_prob)
        fpr, tpr, _ = get_roc_curve(y_test, y_prob)
        imp     = get_feature_importance(name, model, feature_cols)
        results[name] = {
            "model":    model,
            "metrics":  metrics,
            "roc":      (fpr, tpr),
            "importance": imp,
        }
    return results, scaler, X_train, feature_cols, label_encoders


def risk_band(prob: float):
    if prob < 0.20:
        return "APPROVED",   "risk-approved",   "✅", "#34d399"
    elif prob < 0.50:
        return "SUSPICIOUS", "risk-suspicious", "⚠️", "#fbbf24"
    elif prob < 0.80:
        return "HIGH RISK",  "risk-high",       "🔴", "#fb923c"
    else:
        return "BLOCKED",    "risk-blocked",    "🚫", "#f87171"


# ══════════════════════════════════════════════════════════════════════════════
# AUTH
# ══════════════════════════════════════════════════════════════════════════════
def show_auth():
    _, center, _ = st.columns([1, 1.5, 1])
    with center:
        st.markdown("""
        <div style='text-align:center;padding:2rem 0 1.5rem;'>
            <div style='font-size:3.2rem;margin-bottom:0.5rem;'>🛡️</div>
            <div style='font-size:1.9rem;font-weight:800;
                background:linear-gradient(135deg,#6366f1,#a78bfa,#38bdf8);
                -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
                UPI Fraud Shield
            </div>
            <div style='color:#475569;font-size:0.88rem;margin-top:0.4rem;'>
                ML-powered real-time fraud detection &amp; explainability
            </div>
        </div>
        """, unsafe_allow_html=True)

        tab_l, tab_r = st.tabs(["Login", "Create Account"])

        with tab_l:
            st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:1rem;font-weight:700;color:#94a3b8;margin-bottom:1rem;'>Welcome back</div>", unsafe_allow_html=True)
            uname = st.text_input("Username", placeholder="Enter your username", key="li_user")
            pwd   = st.text_input("Password", type="password", placeholder="Enter your password", key="li_pwd")
            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            if st.button("Login  →", use_container_width=True, key="btn_login"):
                if not uname or not pwd:
                    st.error("Please fill in all fields.")
                else:
                    ok, msg, udata = login(uname, pwd)
                    if ok:
                        st.session_state.logged_in = True
                        st.session_state.username  = uname.strip().lower()
                        st.session_state.user_data = udata
                        st.rerun()
                    else:
                        st.error(msg)
            st.markdown("</div>", unsafe_allow_html=True)

        with tab_r:
            st.markdown("<div class='auth-card'>", unsafe_allow_html=True)
            st.markdown("<div style='font-size:1rem;font-weight:700;color:#94a3b8;margin-bottom:1rem;'>Create your account</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            r_uname = c1.text_input("Username", placeholder="e.g. rahul_123", key="reg_user")
            r_email = c2.text_input("Email",    placeholder="you@email.com",  key="reg_email")
            r_pwd   = st.text_input("Password",         type="password", placeholder="Min 6 characters", key="reg_pwd")
            r_pwd2  = st.text_input("Confirm Password", type="password", placeholder="Re-enter password", key="reg_pwd2")
            st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
            if st.button("Create Account  →", use_container_width=True, key="btn_reg"):
                if not all([r_uname, r_email, r_pwd, r_pwd2]):
                    st.error("Please fill in all fields.")
                elif r_pwd != r_pwd2:
                    st.error("Passwords do not match.")
                else:
                    ok, msg = register(r_uname, r_email, r_pwd)
                    if ok:
                     st.success(msg)
                    else:
                        st.error(msg)
            st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("<div style='text-align:center;color:#334155;font-size:0.77rem;margin-top:1.5rem;'>Passwords hashed with SHA-256 · Predictions stored per user</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
def show_dashboard():
    results, scaler, X_bg, feature_cols, label_encoders = load_pipeline()

    # Header columns contain only header elements.
    # IMPORTANT: do not create st.tabs() inside n2. If tab content then creates
    # left/right columns and those create c1/c2 columns, Streamlit sees more
    # than one level of column nesting and raises StreamlitAPIException.
    n1, _, n3 = st.columns([2.2, 5, 2.2])
    with n1:
        st.markdown("""
        <div style='display:flex;align-items:center;gap:10px;padding-top:0.4rem;'>
            <span style='font-size:1.7rem;'>🛡️</span>
            <div>
                <div style='font-weight:800;font-size:1.05rem;color:#a5b4fc;'>UPI Fraud Shield</div>
                <div style='font-size:0.7rem;color:#475569;'>ML · Real-time · SHAP</div>
            </div>
        </div>""", unsafe_allow_html=True)
    with n3:
        uname = st.session_state.username
        ud    = st.session_state.user_data
        st.markdown(f"""
        <div style='text-align:right;padding-top:0.3rem;'>
            <span style='color:#a5b4fc;font-weight:600;font-size:0.95rem;'>👤 {uname}</span><br>
            <span style='color:#475569;font-size:0.73rem;'>{ud.get("email","")}</span>
        </div>""", unsafe_allow_html=True)
        if st.button("Logout", key="logout_btn"):
            st.session_state.logged_in = False
            st.session_state.username  = None
            st.session_state.prediction_done = False
            st.rerun()

    # Tabs must be top-level so the dashboard can safely use one nested
    # columns level (left/right -> c1/c2) inside the Predict tab.
    tabs = st.tabs(["Predict Transaction", "My History", "Analytics", "Model Performance"])

    st.markdown("<hr style='border-color:#1e2a3a;margin:0.5rem 0 1.2rem;'>", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 1 — PREDICT
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[0]:
        left, right = st.columns([1.1, 1.9], gap="large")

        # ── LEFT: INPUTS ──────────────────────────────────────────────────────
        with left:
            st.markdown("<div class='section-header'>Transaction Details</div>", unsafe_allow_html=True)

            # Group 1: Transaction basics
            st.markdown("<div class='input-section'>", unsafe_allow_html=True)
            st.markdown("<div class='input-section-title'>Transaction</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            amount           = c1.number_input("Amount (₹)", min_value=1.0, max_value=500000.0, value=12000.0, step=500.0, key="amt")
            txn_type         = c2.selectbox("Transaction Type", TRANSACTION_TYPES, key="txn_type")
            c1, c2 = st.columns(2)
            merchant_cat     = c1.number_input("Merchant Category Code", min_value=1000, max_value=9999, value=5411, key="mcc")
            txn_time         = c2.slider("Hour of Day (0–23)", 0, 23, 14, key="txn_time")
            session_source   = st.selectbox("Session Source", SESSION_SOURCES, key="sess_src")
            st.markdown("</div>", unsafe_allow_html=True)

            # Group 2: Session & Auth
            st.markdown("<div class='input-section'>", unsafe_allow_html=True)
            st.markdown("<div class='input-section-title'>Session & Authentication</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            session_dur      = c1.number_input("Session Duration (s)", min_value=1, max_value=3600, value=120, key="sess_dur")
            auth_attempts    = c2.number_input("Auth Attempts", min_value=1, max_value=10, value=1, key="auth_att")
            c1, c2 = st.columns(2)
            auth_method      = c1.selectbox("Auth Method", AUTH_METHODS, key="auth_meth")
            otp_delay        = c2.number_input("OTP Entry Delay (s)", min_value=0.0, max_value=300.0, value=12.0, step=1.0, key="otp_delay")
            c1, c2 = st.columns(2)
            otp_freq         = c1.number_input("OTP Requests (last 10 min)", min_value=0, max_value=20, value=1, key="otp_freq")
            otp_dev_consist  = c2.selectbox("OTP on Same Device?", [("Yes", 1), ("No", 0)], format_func=lambda x: x[0], key="otp_dev")[1]
            link_to_txn      = st.number_input("Link→Txn Time (s)", min_value=0.0, max_value=3600.0, value=5.0, step=1.0, key="link_txn")
            st.markdown("</div>", unsafe_allow_html=True)

            # Group 3: Behavioural signals
            st.markdown("<div class='input-section'>", unsafe_allow_html=True)
            st.markdown("<div class='input-section-title'>Behavioural Signals</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            pin_speed        = c1.number_input("PIN Entry Speed (kps)", min_value=0.0, max_value=5.0, value=1.2, step=0.1, key="pin_spd")
            c1, c2 = st.columns(2)
            kbd_speed        = c1.number_input("Keyboard Speed (kps)", min_value=0.0, max_value=10.0, value=2.5, step=0.1, key="kbd_spd")
            app_switches     = c2.number_input("App Switches During Session", min_value=0, max_value=50, value=1, key="app_sw")
            c1, c2 = st.columns(2)
            screen_time      = c1.number_input("Screen Active Time (s)", min_value=0, max_value=3600, value=180, key="scr_time")
            bg_data          = c2.number_input("Background Data (MB)", min_value=0.0, max_value=100.0, value=0.2, step=0.1, key="bg_data")
            st.markdown("</div>", unsafe_allow_html=True)

            # Group 4: Geography & Network
            st.markdown("<div class='input-section'>", unsafe_allow_html=True)
            st.markdown("<div class='input-section-title'>Geography & Network</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            geo_disp         = c1.number_input("Geographic Disparity (km)", min_value=0.0, max_value=10000.0, value=0.0, step=10.0, key="geo_disp")
            geo_vs_ip        = c2.number_input("GPS vs IP Mismatch (0–1)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="geo_ip")
            dns_age          = st.number_input("Payee DNS Age (days)", min_value=0, max_value=9999, value=365, key="dns_age")
            st.markdown("</div>", unsafe_allow_html=True)

            # Group 5: Transaction history
            st.markdown("<div class='input-section'>", unsafe_allow_html=True)
            st.markdown("<div class='input-section-title'>Transaction History</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            amt_vs_hist      = c1.number_input("Amount vs Sender History (ratio)", min_value=0.0, max_value=50.0, value=1.0, step=0.1, key="amt_hist")
            txn_velocity     = c2.number_input("Txns Last 24h", min_value=0, max_value=100, value=3, key="txn_vel")
            failed_txns      = st.number_input("Failed Txns (last 7 days)", min_value=0, max_value=50, value=0, key="fail_txn")
            st.markdown("</div>", unsafe_allow_html=True)

            # Group 6: Collect request info
            st.markdown("<div class='input-section'>", unsafe_allow_html=True)
            st.markdown("<div class='input-section-title'>Collect Request Info</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            req_roundness    = c1.number_input("Amount Roundness (0–1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="req_round")
            req_freq         = c2.number_input("Requests Received (7d)", min_value=0, max_value=100, value=0, key="req_freq")
            c1, c2 = st.columns(2)
            req_accept_rate  = c1.number_input("Request Accept Rate (0–1)", min_value=0.0, max_value=1.0, value=0.5, step=0.01, key="req_acc")
            resp_time        = c2.number_input("Response Time (s)", min_value=0.0, max_value=3600.0, value=30.0, step=1.0, key="resp_time")
            c1, c2 = st.columns(2)
            req_acc_age      = c1.number_input("Requester Account Age (days)", min_value=0, max_value=3650, value=365, key="req_age")
            relationship     = c2.selectbox("Relationship to Requester", RELATIONSHIP_OPTIONS, key="relation")
            st.markdown("</div>", unsafe_allow_html=True)

            # Group 7: UPI Handle
            st.markdown("<div class='input-section'>", unsafe_allow_html=True)
            st.markdown("<div class='input-section-title'>UPI Handle Intelligence</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            handle_age       = c1.number_input("UPI Handle Age (days)", min_value=0, max_value=3650, value=365, key="h_age")
            handle_sim       = c2.number_input("Handle Similarity Score (0–1)", min_value=0.0, max_value=1.0, value=0.0, step=0.01, key="h_sim")
            c1, c2 = st.columns(2)
            handle_official  = c1.selectbox("Contains Official Terms?", [("No", 0), ("Yes", 1)], format_func=lambda x: x[0], key="h_off")[1]
            handle_hist      = c2.number_input("Handle Transaction History", min_value=0, max_value=100000, value=500, key="h_hist")
            c1, c2 = st.columns(2)
            biz_match        = c1.selectbox("Business Name Match", BUSINESS_MATCH_OPTIONS, key="biz_match")
            social_pres      = c2.selectbox("Social Media Presence", SOCIAL_OPTIONS, key="social")
            st.markdown("</div>", unsafe_allow_html=True)

            # Model selector
            st.markdown("<div class='input-section'>", unsafe_allow_html=True)
            st.markdown("<div class='input-section-title'>ML Model</div>", unsafe_allow_html=True)
            model_choice = st.selectbox("Select Model", list(MODEL_REGISTRY.keys()), index=2, key="model_inp",
                                        label_visibility="collapsed")
            st.markdown("</div>", unsafe_allow_html=True)

            analyze = st.button("Analyze Transaction", use_container_width=True, type="primary")

        # ── RIGHT: RESULTS ────────────────────────────────────────────────────
        with right:
            if analyze or st.session_state.prediction_done:
                # ── Encode categoricals for prediction ────────────────────────
                def _encode_cat(col: str, val: str) -> int:
                    le = label_encoders.get(col)
                    if le is None:
                        return 0
                    known = set(le.classes_)
                    v = val if val in known else le.classes_[0]
                    return int(le.transform([v])[0])

                amount_log = float(np.log1p(amount))

                feat_dict = {
                    "amount_log":                             amount_log,
                    "session_duration":                        float(session_dur),
                    "authentication_attempts":                 float(auth_attempts),
                    "transaction_amount_vs_sender_history":    float(amt_vs_hist),
                    "geographic_disparity":                    float(geo_disp),
                    "transaction_time_of_day":                 float(txn_time),
                    "merchant_category_code":                  float(merchant_cat),
                    "session_source":                          float(_encode_cat("session_source", session_source)),
                    "time_between_link_click_and_transaction": float(link_to_txn),
                    "screen_active_time":                      float(screen_time),
                    "time_between_otp_generation_and_input":   float(otp_delay),
                    "pin_entry_speed":                         float(pin_speed),
                    "otp_request_frequency":                   float(otp_freq),
                    "otp_request_device_consistency":          float(otp_dev_consist),
                    "transaction_velocity":                    float(txn_velocity),
                    "failed_transaction_count":                float(failed_txns),
                    "authorization_method":                    float(_encode_cat("authorization_method", auth_method)),
                    "transaction_type":                        float(_encode_cat("transaction_type", txn_type)),
                    "request_amount_roundness":                float(req_roundness),
                    "request_frequency":                       float(req_freq),
                    "request_acceptance_rate":                 float(req_accept_rate),
                    "time_to_respond_to_request":              float(resp_time),
                    "requester_account_age":                   float(req_acc_age),
                    "relationship_to_requester":               float(_encode_cat("relationship_to_requester", relationship)),
                    
                }

                X_input = pd.DataFrame([feat_dict])[feature_cols]
                cfg     = MODEL_REGISTRY[model_choice]
                X_sc    = scaler.transform(X_input) if cfg["needs_scaling"] else X_input.values
                X_bg_sc = scaler.transform(X_bg[feature_cols]) if cfg["needs_scaling"] else X_bg[feature_cols]

                model   = results[model_choice]["model"]
                prob    = float(model.predict_proba(X_sc)[0][1])
                label, badge_cls, icon, color = risk_band(prob)

                if analyze:
                    st.session_state.prediction_done = True
                    record = {
                        "amount":           f"₹{amount:,.0f}",
                        "txn_type":         txn_type,
                        "hour":             txn_time,
                        "session_source":   session_source,
                        "auth_method":      auth_method,
                        "model":            model_choice,
                        "probability":      round(prob * 100, 2),
                        "risk_label":       label,
                        "txn_velocity":     txn_velocity,
                        "failed_txns":      failed_txns,
                    }
                    save_prediction(st.session_state.username, record)

                # ── Risk Banner ──
                st.markdown(f"""
                <div style='background:linear-gradient(135deg,#161b27,#1a2035);
                            border:1px solid #2d3748;border-radius:16px;
                            padding:1.4rem 1.2rem;margin-bottom:1rem;text-align:center;'>
                    <div style='color:#64748b;font-size:0.82rem;margin-bottom:0.4rem;'>
                        ₹{amount:,.0f} · {txn_type} · {session_source} · {model_choice}
                    </div>
                    <div style='font-size:3.2rem;font-weight:900;color:{color};line-height:1.1;'>{prob*100:.1f}%</div>
                    <div style='color:#64748b;font-size:0.8rem;margin:0.2rem 0 0.6rem;'>Fraud Probability</div>
                    <span class='risk-badge {badge_cls}'>{icon} &nbsp;{label}</span>
                </div>""", unsafe_allow_html=True)

                # ── Gauge ──
                fig = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=prob * 100,
                    number={"suffix": "%", "font": {"size": 28, "color": color}},
                    gauge={
                        "axis": {"range": [0, 100], "tickcolor": "#475569", "tickfont": {"color": "#94a3b8"}},
                        "bar":  {"color": color, "thickness": 0.22},
                        "bgcolor": "#161b27",
                        "bordercolor": "#2d3748",
                        "steps": [
                            {"range": [0,  20], "color": "#064e3b"},
                            {"range": [20, 50], "color": "#422006"},
                            {"range": [50, 80], "color": "#431407"},
                            {"range": [80,100], "color": "#450a0a"},
                        ],
                        "threshold": {"line": {"color": color, "width": 4}, "value": prob * 100},
                    },
                ))
                fig.update_layout(height=220, margin=dict(t=15, b=5, l=20, r=20),
                                  paper_bgcolor="#0e1117", font_color="#e2e8f0")
                st.plotly_chart(fig, use_container_width=True)

                # ── All-model compare ──
                with st.expander("Compare all 5 models"):
                    rows = []
                    for mn, res in results.items():
                        cfg2 = MODEL_REGISTRY[mn]
                        X2   = scaler.transform(X_input) if cfg2["needs_scaling"] else X_input.values
                        p2   = float(res["model"].predict_proba(X2)[0][1])
                        lbl2, _, ic2, col2 = risk_band(p2)
                        rows.append({"Model": mn, "Fraud %": round(p2*100,1), "Risk": f"{ic2} {lbl2}", "Color": col2})
                    for r in sorted(rows, key=lambda x: -x["Fraud %"]):
                        st.markdown(f"""
                        <div style='background:#0f1420;border:1px solid #1e2a3a;border-radius:8px;
                                    padding:0.6rem 1rem;margin-bottom:0.4rem;
                                    display:flex;justify-content:space-between;align-items:center;'>
                            <span style='color:#94a3b8;font-size:0.9rem;'>{r["Model"]}</span>
                            <span style='color:{r["Color"]};font-weight:700;font-size:0.9rem;'>{r["Fraud %"]}% &nbsp; {r["Risk"]}</span>
                        </div>""", unsafe_allow_html=True)

                # ── Key signals ──
                feat_vals_raw = list(X_input.iloc[0])
                feat_names_raw = list(X_input.columns)
                disp_vals = {
                    FEATURE_LABELS.get(f, f): round(v, 3)
                    for f, v in zip(feat_names_raw, feat_vals_raw)
                }
                top_feats = sorted(disp_vals.items(), key=lambda x: abs(x[1]), reverse=True)[:8]

                st.markdown("<div class='section-header' style='margin-top:1rem;'>Key Signals</div>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                for i, (fname, fval) in enumerate(top_feats):
                    col = c1 if i % 2 == 0 else c2
                    col.markdown(f"""
                    <div class='metric-card' style='margin-bottom:0.5rem;'>
                        <div class='metric-val' style='font-size:1.3rem;color:#a5b4fc;'>{fval}</div>
                        <div class='metric-lbl'>{fname}</div>
                    </div>""", unsafe_allow_html=True)

                # ── SHAP ──
                st.markdown("<div class='section-header' style='margin-top:1rem;'>SHAP Explanation</div>", unsafe_allow_html=True)
                with st.spinner("Computing SHAP values…"):
                    X_in_df = pd.DataFrame(X_sc if cfg["needs_scaling"] else X_input.values,
                                           columns=feature_cols)
                    X_bg_df = pd.DataFrame(X_bg_sc if cfg["needs_scaling"] else X_bg[feature_cols].values,
                                           columns=feature_cols)
                    shap_vals = get_shap_values(model, X_in_df, X_bg_df, model_choice)
                if shap_vals is not None:
                    try:
                        bv = float(shap_vals.base_values[0]) if hasattr(shap_vals, "base_values") else 0.0
                        sv = shap_vals.values[0] if hasattr(shap_vals, "values") else np.array(shap_vals[0])
                        img = plot_shap_waterfall(sv, feature_cols, feat_vals_raw, bv, prob)
                        st.image(img, use_container_width=True)
                    except Exception:
                        st.info("SHAP chart unavailable for this model configuration.")
                else:
                    st.info("SHAP chart unavailable for this model configuration.")

                # ── Precautions ──
                prec = get_precautions(feat_names_raw, feat_vals_raw)
                if prec or prob > 0.2:
                    st.markdown("<div class='section-header' style='margin-top:1rem;'>Precautions</div>", unsafe_allow_html=True)
                    for p in prec:
                        st.markdown(f"<div class='precaution-box'>{p}</div>", unsafe_allow_html=True)
                    for gp in GENERAL_PRECAUTIONS:
                        st.markdown(f"<div class='precaution-general'>{gp}</div>", unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='text-align:center;padding:4rem 2rem;color:#334155;'>
                    <div style='font-size:3rem;margin-bottom:1rem;'>🛡️</div>
                    <div style='font-size:1.1rem;font-weight:600;color:#475569;'>
                        Fill in the transaction details and click<br><b style='color:#6366f1;'>Analyze Transaction</b>
                    </div>
                </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 2 — HISTORY
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[1]:
        st.markdown("<div class='section-header'>Your Prediction History</div>", unsafe_allow_html=True)
        history = get_history(st.session_state.username)
        if not history:
            st.info("No predictions yet. Analyze a transaction to get started.")
        else:
            if st.button("Clear History", key="clear_hist"):
                clear_history(st.session_state.username)
                st.rerun()
            for rec in reversed(history[-50:]):
                color = {"APPROVED":"#34d399","SUSPICIOUS":"#fbbf24","HIGH RISK":"#fb923c","BLOCKED":"#f87171"}.get(rec.get("risk_label",""), "#94a3b8")
                st.markdown(f"""
                <div class='hist-row'>
                    <div>
                        <span style='color:#e2e8f0;font-weight:600;'>{rec.get("amount","—")}</span>
                        <span style='color:#475569;font-size:0.8rem;'> · {rec.get("txn_type","—")} · {rec.get("model","—")}</span>
                    </div>
                    <span style='color:{color};font-weight:700;font-size:0.95rem;'>
                        {rec.get("probability","—")}% &nbsp; {rec.get("risk_label","—")}
                    </span>
                </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 3 — ANALYTICS
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[2]:
        st.markdown("<div class='section-header'>Model Analytics</div>", unsafe_allow_html=True)
        best_name = max(results, key=lambda n: results[n]["metrics"]["AUC-ROC"])
        best_res  = results[best_name]

        # ROC curves
        fig_roc = go.Figure()
        for name, res in results.items():
            fpr, tpr = res["roc"]
            auc = res["metrics"]["AUC-ROC"]
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr, mode="lines", name=f"{name} (AUC={auc})",
                line=dict(width=2, color=MODEL_REGISTRY[name]["color"]),
            ))
        fig_roc.add_trace(go.Scatter(x=[0,1], y=[0,1], mode="lines", name="Random",
                                     line=dict(dash="dash", color="#475569")))
        fig_roc.update_layout(
            title="ROC Curves — All Models",
            xaxis_title="False Positive Rate", yaxis_title="True Positive Rate",
            paper_bgcolor="#0e1117", plot_bgcolor="#0d1117",
            font_color="#e2e8f0", legend=dict(bgcolor="#0d1117"),
            height=380,
        )
        st.plotly_chart(fig_roc, use_container_width=True)

        # Feature importance (best model)
        imp = best_res["importance"]
        if imp:
            top_imp = dict(list(imp.items())[:15])
            labels  = [FEATURE_LABELS.get(f, f) for f in top_imp]
            fig_imp = go.Figure(go.Bar(
                x=list(top_imp.values()), y=labels, orientation="h",
                marker_color="#6366f1", marker_line_color="#8b5cf6",
            ))
            fig_imp.update_layout(
                title=f"Feature Importance — {best_name}",
                xaxis_title="Importance", yaxis_autorange="reversed",
                paper_bgcolor="#0e1117", plot_bgcolor="#0d1117",
                font_color="#e2e8f0", height=420,
            )
            st.plotly_chart(fig_imp, use_container_width=True)

    # ══════════════════════════════════════════════════════════════════════════
    # TAB 4 — MODEL PERFORMANCE
    # ══════════════════════════════════════════════════════════════════════════
    with tabs[3]:
        st.markdown("<div class='section-header'>Model Performance Metrics</div>", unsafe_allow_html=True)
        metric_keys = ["Accuracy", "Precision", "Recall", "F1 Score", "AUC-ROC"]
        for name, res in results.items():
            m = res["metrics"]
            cfg_col = MODEL_REGISTRY[name]["color"]
            st.markdown(f"<div style='font-size:0.9rem;font-weight:700;color:{cfg_col};margin:1rem 0 0.4rem;'>{name}</div>", unsafe_allow_html=True)
            cols = st.columns(len(metric_keys))
            for col, key in zip(cols, metric_keys):
                val = m[key]
                col.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-val' style='font-size:1.4rem;color:{cfg_col};'>{val:.4f}</div>
                    <div class='metric-lbl'>{key}</div>
                </div>""", unsafe_allow_html=True)
            cm = [[m["True Negatives"], m["False Positives"]], [m["False Negatives"], m["True Positives"]]]
            fig_cm = px.imshow(
                cm,
                labels=dict(x="Predicted", y="Actual", color="Count"),
                x=["Legit", "Fraud"], y=["Legit", "Fraud"],
                color_continuous_scale="Blues", text_auto=True,
            )
            fig_cm.update_layout(
                height=220, margin=dict(t=10, b=10, l=10, r=10),
                paper_bgcolor="#0e1117", font_color="#e2e8f0",
            )
            st.plotly_chart(fig_cm, use_container_width=True)
            st.markdown("<hr class='hdivider'>", unsafe_allow_html=True)


# ── Entry point ─────────────────────────────────────────────────────────────────
if st.session_state.logged_in:
    show_dashboard()
else:
    show_auth()
