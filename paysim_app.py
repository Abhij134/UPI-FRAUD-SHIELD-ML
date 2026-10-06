import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="PaySim Fraud Shield", page_icon="🛡️", layout="wide")

# Custom CSS for dark premium theme
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}
.stApp {
    background-color: #0f111a;
    color: #e2e8f0;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 700;
    background: linear-gradient(90deg, #818cf8 0%, #c084fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.5rem;
}
.auth-card {
    background: rgba(30, 41, 59, 0.5);
    border: 1px solid rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 2rem;
    margin-top: 1rem;
    backdrop-filter: blur(10px);
}
.risk-badge {
    padding: 0.5rem 1.5rem;
    border-radius: 8px;
    font-weight: 700;
    font-size: 1.1rem;
    display: inline-block;
    margin-top: 1rem;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.risk-high { background: rgba(239,68,68,0.2); color: #f87171; border: 1px solid rgba(239,68,68,0.3); }
.risk-low { background: rgba(16,185,129,0.2); color: #34d399; border: 1px solid rgba(16,185,129,0.3); }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_paysim_artifacts():
    model_path = "upi-fraud-ml/notebooks/models/paysim_rf.pkl"
    feats_path = "upi-fraud-ml/notebooks/models/paysim_features.pkl"
    
    if not os.path.exists(model_path) or not os.path.exists(feats_path):
        return None, None
        
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(feats_path, "rb") as f:
        features = pickle.load(f)
        
    return model, features

st.markdown('<div class="hero-title">🛡️ PaySim Enterprise Fraud Shield</div>', unsafe_allow_html=True)
st.markdown("Real-time transaction authorization powered by Random Forest on the 6M+ row PaySim Ledger.")

model, features = load_paysim_artifacts()

if model is None:
    st.error("Model artifacts not found! Run `python train_paysim.py` first.")
    st.stop()

st.sidebar.header("Transaction Details")

# Input fields
col1, col2 = st.columns([1, 2])

with st.sidebar:
    st.subheader("Ledger Entries")
    txn_type = st.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"])
    amount = st.number_input("Amount (₹)", min_value=1.0, value=50000.0, step=1000.0)
    
    st.markdown("---")
    oldbalanceOrg = st.number_input("Sender Initial Balance", min_value=0.0, value=50000.0)
    newbalanceOrig = st.number_input("Sender Final Balance", min_value=0.0, value=0.0)
    
    st.markdown("---")
    oldbalanceDest = st.number_input("Receiver Initial Balance", min_value=0.0, value=10000.0)
    newbalanceDest = st.number_input("Receiver Final Balance", min_value=0.0, value=10000.0)
    
    st.markdown("---")
    hour_of_day = st.slider("Hour of Day", 0, 23, 14)
    
    analyze = st.button("Authorize Transaction", type="primary", use_container_width=True)

if analyze:
    # Feature Engineering
    errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg
    errorBalanceDest = oldbalanceDest + amount - newbalanceDest
    
    type_CASH_OUT = 1 if txn_type == "CASH_OUT" else 0
    type_DEBIT = 1 if txn_type == "DEBIT" else 0
    type_PAYMENT = 1 if txn_type == "PAYMENT" else 0
    type_TRANSFER = 1 if txn_type == "TRANSFER" else 0
    
    feat_dict = {
        'amount': amount,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'errorBalanceOrig': errorBalanceOrig,
        'errorBalanceDest': errorBalanceDest,
        'hour_of_day': hour_of_day,
        'type_CASH_OUT': type_CASH_OUT,
        'type_DEBIT': type_DEBIT,
        'type_PAYMENT': type_PAYMENT,
        'type_TRANSFER': type_TRANSFER
    }
    
    X_input = pd.DataFrame([feat_dict])[features]
    prob = model.predict_proba(X_input)[0][1]
    
    with col1:
        st.markdown('<div class="auth-card">', unsafe_allow_html=True)
        st.subheader("Decision Engine")
        
        if prob > 0.5:
            st.markdown(f'<div class="risk-badge risk-high">BLOCKED ({prob*100:.1f}% Fraud Risk)</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-badge risk-low">APPROVED ({(1-prob)*100:.1f}% Safe)</div>', unsafe_allow_html=True)
            
        st.markdown(f"**Sender Balance Error:** ₹{errorBalanceOrig:,.2f}")
        st.markdown(f"**Receiver Balance Error:** ₹{errorBalanceDest:,.2f}")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col2:
        st.subheader("Ledger Anomaly Explanation (SHAP)")
        with st.spinner("Calculating SHAP values..."):
            explainer = shap.TreeExplainer(model)
            shap_values = explainer(X_input)
            
            fig, ax = plt.subplots(figsize=(8, 4))
            # Black background for dark mode SHAP
            fig.patch.set_facecolor('#0f111a')
            ax.set_facecolor('#0f111a')
            ax.tick_params(colors='white')
            ax.xaxis.label.set_color('white')
            ax.yaxis.label.set_color('white')
            
            # Waterfall for multi-class (class 1 is fraud)
            shap.plots.waterfall(shap_values[0, :, 1], show=False)
            st.pyplot(fig, transparent=True)
