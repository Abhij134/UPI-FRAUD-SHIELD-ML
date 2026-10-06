import os

with open("upi-fraud-ml/artifacts/ml-fraud-detection/app.py", "r", encoding="utf-8") as f:
    code = f.read()

# 1. Path fix
code = code.replace(
    'sys.path.insert(0, os.path.dirname(__file__))',
    'sys.path.insert(0, os.path.abspath(\'upi-fraud-ml/artifacts/ml-fraud-detection\'))'
)

# 2. Add pickle
code = code.replace('import pandas as pd', 'import pandas as pd\nimport pickle')

# 3. Add load_paysim_artifacts and replace load_pipeline
load_fn = '''
@st.cache_resource
def load_paysim_artifacts():
    import os
    model_path = "upi-fraud-ml/notebooks/models/paysim_all_models.pkl"
    if not os.path.exists(model_path): return None, None, None, None, None, None, None
    with open(model_path, "rb") as f: models = pickle.load(f)
    with open("upi-fraud-ml/notebooks/models/paysim_scaler.pkl", "rb") as f: scaler = pickle.load(f)
    with open("upi-fraud-ml/notebooks/models/paysim_all_features.pkl", "rb") as f: features = pickle.load(f)
    return None, None, None, None, models, scaler, features

def load_pipeline():
    pass
'''
code = code.replace('from src.feature_engineering import get_scaler, FEATURE_COLS', load_fn)
code = code.replace(
    'X_train, X_test, y_train, y_test, label_encoders, feature_cols, trained_models, scaler = load_pipeline()',
    '_, _, _, _, trained_models, scaler, feature_cols = load_paysim_artifacts()'
)
code = code.replace(
    'results, scaler, X_bg, feature_cols, label_encoders = load_pipeline()',
    '_, _, X_bg, _, trained_models, scaler, feature_cols = load_paysim_artifacts()'
)

# 4. Replace Left Inputs
left_str = "with left:\n            st.markdown(\"<div class='section-header'>Transaction Details</div>\", unsafe_allow_html=True)"
right_str = "        # ── RIGHT: RESULTS ────────────────────────────────────────────────────"

paysim_inputs = """
        with left:
            st.markdown("<div class='section-header'>Ledger Entries</div>", unsafe_allow_html=True)
            
            st.markdown("<div class='input-section'>", unsafe_allow_html=True)
            st.markdown("<div class='input-section-title'>💳 Transaction Basics</div>", unsafe_allow_html=True)
            c1, c2 = st.columns(2)
            txn_type = c1.selectbox("Transaction Type", ["TRANSFER", "CASH_OUT", "CASH_IN", "PAYMENT", "DEBIT"], key="txn_type")
            amount = c2.number_input("Amount (₹)", min_value=1.0, value=50000.0, step=1000.0, key="amt")
            hour_of_day = st.slider("Hour of Day", 0, 23, 14, key="hour")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='input-section'>", unsafe_allow_html=True)
            st.markdown("<div class='input-section-title'>🏦 Sender Details</div>", unsafe_allow_html=True)
            oldbalanceOrg = st.number_input("Sender Initial Balance", min_value=0.0, value=50000.0, key="oldOrg")
            newbalanceOrig = st.number_input("Sender Final Balance", min_value=0.0, value=0.0, key="newOrg")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='input-section'>", unsafe_allow_html=True)
            st.markdown("<div class='input-section-title'>🏦 Receiver Details</div>", unsafe_allow_html=True)
            oldbalanceDest = st.number_input("Receiver Initial Balance", min_value=0.0, value=10000.0, key="oldDest")
            newbalanceDest = st.number_input("Receiver Final Balance", min_value=0.0, value=10000.0, key="newDest")
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown("<div class='input-section'>", unsafe_allow_html=True)
            st.markdown("<div class='input-section-title'>🤖 ML Model</div>", unsafe_allow_html=True)
            
            model_choices = list(trained_models.keys()) if trained_models else ["Random Forest"]
            model_choice = st.selectbox("Select Model", model_choices, index=0, key="model_inp", label_visibility="collapsed")
            if trained_models:
                st.markdown(f"<div style='font-size:0.75rem;color:#64748b;margin-top:4px;font-family:Inter,sans-serif;'>{trained_models[model_choice]['description']}</div>", unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            analyze = st.button("🔍  Analyze Transaction", use_container_width=True, type="primary")

"""

start_idx = code.find(left_str)
end_idx = code.find(right_str)
if start_idx != -1 and end_idx != -1:
    code = code[:start_idx] + paysim_inputs + code[end_idx:]

# 5. Replace feat_dict
encode_str = "def _encode_cat("
predict_str = "X_input = pd.DataFrame([feat_dict])[feature_cols]"

paysim_feat = """
                errorBalanceOrig = newbalanceOrig + amount - oldbalanceOrg
                errorBalanceDest = oldbalanceDest + amount - newbalanceDest
                feat_dict = {
                    'amount': amount,
                    'oldbalanceOrg': oldbalanceOrg,
                    'newbalanceOrig': newbalanceOrig,
                    'oldbalanceDest': oldbalanceDest,
                    'newbalanceDest': newbalanceDest,
                    'errorBalanceOrig': errorBalanceOrig,
                    'errorBalanceDest': errorBalanceDest,
                    'hour_of_day': hour_of_day,
                    'type_CASH_OUT': 1 if txn_type == 'CASH_OUT' else 0,
                    'type_DEBIT': 1 if txn_type == 'DEBIT' else 0,
                    'type_PAYMENT': 1 if txn_type == 'PAYMENT' else 0,
                    'type_TRANSFER': 1 if txn_type == 'TRANSFER' else 0
                }
                
                # Replace model calling logic
                if trained_models is None:
                    st.error("Models not found! Run train_paysim_all.py first.")
                    st.stop()
                    
                cfg = trained_models[model_choice]
                mdl = cfg["model"]
                
                # We do not use X_bg for SHAP in PaySim (we use tree explainer or simple explainer)
                """

s_idx = code.find(encode_str)
e_idx = code.find(predict_str)
if s_idx != -1 and e_idx != -1:
    code = code[:s_idx] + paysim_feat + code[e_idx:]

# Fix Prediction execution
code = code.replace(
    'X_input_sc = scaler.transform(X_input)',
    'X_input_sc = scaler.transform(X_input) if cfg["needs_scaling"] else X_input'
)
code = code.replace(
    'prob = float(trained_models[model_choice]["model"].predict_proba(X_input_sc)[0][1])',
    'prob = float(mdl.predict_proba(X_input_sc)[0][1])'
)

# 6. SHAP fix
code = code.replace('shap.plots.waterfall(shap_values[0], show=False)', 'shap.plots.waterfall(shap_values[0, :, 1], show=False)')

# Remove X_bg dependency in SHAP because we don't have X_bg loaded here (it's too large)
code = code.replace('shap_values = get_shap_values(mdl, X_input_sc, X_bg)', 'explainer = shap.TreeExplainer(mdl); shap_values = explainer(X_input_sc)')
code = code.replace('shap_values_lr = get_shap_values(mdl, X_input_sc, X_bg)', 'explainer_lr = shap.LinearExplainer(mdl, scaler.transform(pd.DataFrame([feat_dict]*100)[feature_cols])); shap_values_lr = explainer_lr(X_input_sc)')

# Ensure X_bg is mocked
code = code.replace('results, scaler, X_bg, feature_cols, label_encoders = load_pipeline()', '_, _, X_bg, _, trained_models, scaler, feature_cols = load_paysim_artifacts()\n    X_bg = pd.DataFrame([feat_dict]*10)[feature_cols] if "feat_dict" in locals() else None')

with open("paysim_app_full.py", "w", encoding="utf-8") as f:
    f.write(code)

print("App generation successful!")
