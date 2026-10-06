import sys
sys.path.append("p:/Researchproject/upi-fraud-ml-fixed/upi-fraud-ml/artifacts/ml-fraud-detection")
import pandas as pd
from src.data_generator import get_train_test_data
from src.feature_engineering import get_scaler
from src.models import train_model, MODEL_REGISTRY, predict_model
import numpy as np

X_train, X_test, y_train, y_test, label_encoders, feature_cols = get_train_test_data()
scaler = get_scaler(X_train)
model = train_model("Random Forest", X_train, y_train, scaler)
results = {"Random Forest": {"model": model, "needs_scaling": MODEL_REGISTRY["Random Forest"]["needs_scaling"]}}

def _encode_cat(col: str, val: str) -> int:
    le = label_encoders.get(col)
    if le is None:
        return 0
    known = set(le.classes_)
    v = val if val in known else le.classes_[0]
    return int(le.transform([v])[0])

# Simulate a 12000 rs transaction with exact defaults from UI
feat_dict = {
    "amount_log":                             float(np.log1p(12000.0)),
    "session_duration":                        float(120),
    "authentication_attempts":                 float(1),
    "transaction_amount_vs_sender_history":    float(1.0),
    "geographic_disparity":                    float(0.0),
    "transaction_time_of_day":                 float(14),
    "merchant_category_code":                  float(_encode_cat("merchant_category_code", str(int(5411)))),
    "session_source":                          float(_encode_cat("session_source", "app")),
    "time_between_link_click_and_transaction": float(5.0),
    "dns_lookup_age":                          float(365),
    "input_timing_consistency":               float(0.8),
    "app_switching_frequency":                 float(1),
    "keyboard_input_speed":                    float(2.5),
    "screen_active_time":                      float(180),
    "geographic_location_vs_ip":               float(0.0),
    "background_data_usage":                   float(0.2),
    "time_between_otp_generation_and_input":   float(12.0),
    "pin_entry_speed":                         float(1.2),
    "otp_request_frequency":                   float(1),
    "otp_request_device_consistency":          float(0),
    "transaction_velocity":                    float(3),
    "failed_transaction_count":                float(0),
    "authorization_method":                    float(_encode_cat("authorization_method", "pin")),
    "transaction_type":                        float(_encode_cat("transaction_type", "payment")),
    "request_amount_roundness":                float(0.5),
    "request_frequency":                       float(0),
    "request_acceptance_rate":                 float(0.5),
    "time_to_respond_to_request":              float(30.0),
    "requester_account_age":                   float(365),
    "relationship_to_requester":               float(_encode_cat("relationship_to_requester", "known")),
    "upi_handle_age":                          float(365),
    "handle_similarity_score":                 float(0.0),
    "handle_contains_official_terms":          float(0),
    "handle_transaction_history":              float(500),
    "business_name_match":                     float(_encode_cat("business_name_match", "match")),
    "social_media_presence":                   float(_encode_cat("social_media_presence", "verified")),
}
available_cols = [c for c in feature_cols if c in feat_dict]
X_input = pd.DataFrame([feat_dict])[available_cols]

cfg = results["Random Forest"]
X_sc = scaler.transform(X_input) if cfg["needs_scaling"] else X_input.values
prob = cfg["model"].predict_proba(X_sc)[0][1]
print(f"Random Forest (No SMOTE): {prob * 100:.2f}% Fraud")
