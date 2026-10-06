import sys
import os
sys.path.append(os.path.join("p:/Researchproject/upi-fraud-ml-fixed", "upi-fraud-ml", "artifacts", "ml-fraud-detection"))
import pandas as pd
import numpy as np
from app import load_pipeline
from src.models import MODEL_REGISTRY

results, scaler, X_bg, feature_cols, label_encoders = load_pipeline()

def _encode_cat(col: str, val: str) -> int:
    le = label_encoders.get(col)
    if le is None:
        return 0
    known = set(le.classes_)
    v = val if val in known else le.classes_[0]
    return int(le.transform([v])[0])

# Simulate a known NON-FRAUD transaction
feat_dict = {
    "amount_log":                             float(np.log1p(1960.75)),
    "session_duration":                        float(234),
    "authentication_attempts":                 float(1),
    "transaction_amount_vs_sender_history":    float(1.27),
    "geographic_disparity":                    float(11080.8),
    "transaction_time_of_day":                 float(23),
    "merchant_category_code":                  float(_encode_cat("merchant_category_code", "food")),
    "session_source":                          float(_encode_cat("session_source", "app")),
    "time_between_link_click_and_transaction": float(0.0),
    "dns_lookup_age":                          float(0),
    "input_timing_consistency":               float(0.9),
    "app_switching_frequency":                 float(0),
    "keyboard_input_speed":                    float(1.17),
    "screen_active_time":                      float(262),
    "geographic_location_vs_ip":               float(0.0),
    "background_data_usage":                   float(0.33),
    "time_between_otp_generation_and_input":   float(0.0),
    "pin_entry_speed":                         float(1.25),
    "otp_request_frequency":                   float(0),
    "otp_request_device_consistency":          float(1),
    "transaction_velocity":                    float(0),
    "failed_transaction_count":                float(0),
    "authorization_method":                    float(_encode_cat("authorization_method", "pin")),
    "transaction_type":                        float(_encode_cat("transaction_type", "payment")),
    "request_amount_roundness":                float(1.0),
    "request_frequency":                       float(0),
    "request_acceptance_rate":                 float(0.0),
    "time_to_respond_to_request":              float(0.0),
    "requester_account_age":                   float(0),
    "relationship_to_requester":               float(_encode_cat("relationship_to_requester", "unknown")),
    "upi_handle_age":                          float(0),
    "handle_similarity_score":                 float(0.0),
    "handle_contains_official_terms":          float(0),
    "handle_transaction_history":              float(0),
    "business_name_match":                     float(_encode_cat("business_name_match", "none")),
    "social_media_presence":                   float(_encode_cat("social_media_presence", "none")),
}

available_cols = [c for c in feature_cols if c in feat_dict]
X_input = pd.DataFrame([feat_dict])[available_cols]

for name, cfg in MODEL_REGISTRY.items():
    X_sc = scaler.transform(X_input) if cfg["needs_scaling"] else X_input.values
    prob = results[name]["model"].predict_proba(X_sc)[0][1]
    print(f"{name}: {prob * 100:.2f}% Fraud")
