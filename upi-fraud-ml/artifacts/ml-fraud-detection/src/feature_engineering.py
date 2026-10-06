import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

# ── Columns removed from the raw CSV before modelling ─────────────────────────

DROP_IDS = [
    "transaction_id", "user_id", "merchant_id", "timestamp", "description",
    "device_id", "ip_address", "location", "url_referrer", "request_description",
]

DROP_LEAKAGE = [
    "unusual_transaction_amount_flag", "unusual_device_flag", "unusual_ip_flag",
    "unusual_location_flag", "handle_verification_status", "authentication_attempt_count",
    "time_pressure_indicators", "handle_registration_pattern",
    "receiver_account_age", "receiver_transaction_history",
]

DROP_HIGH_CORR = [
    "pin_entry_method",
    "handle_typo_analysis",
    "handle_to_description_consistency",
]

DROP_LISTS = [
    "recent_app_installs", "input_pause_patterns", "permissions_granted",
    "recognized_screen_sharing_apps", "request_description_keywords",
]

ALL_DROP = DROP_IDS + DROP_LEAKAGE + DROP_HIGH_CORR + DROP_LISTS

CAT_COLS = [
    "session_source",
    "authorization_method",
    "transaction_type",
    "relationship_to_requester",
]

FEATURE_COLS = [
    # Transaction
    "amount_log",
    "session_duration",
    "authentication_attempts",
    "transaction_amount_vs_sender_history",
    "geographic_disparity",
    "transaction_time_of_day",
    "merchant_category_code",
    "session_source",
    "time_between_link_click_and_transaction",
    # Behavioural biometrics
    "dns_lookup_age",
    "input_timing_consistency",
    "app_switching_frequency",
    "keyboard_input_speed",
    "screen_active_time",
    "geographic_location_vs_ip",
    "background_data_usage",
    # OTP & auth
    "time_between_otp_generation_and_input",
    "pin_entry_speed",
    "otp_request_frequency",
    "otp_request_device_consistency",
    # Transaction history
    "transaction_velocity",
    "failed_transaction_count",
    # Encoded categoricals
    "authorization_method",
    "transaction_type",
    # Collect request
    "request_amount_roundness",
    "request_frequency",
    "request_acceptance_rate",
    "time_to_respond_to_request",
    "requester_account_age",
    "relationship_to_requester",
    # UPI handle intelligence
    "upi_handle_age",
    "handle_similarity_score",
    "handle_contains_official_terms",
    "handle_transaction_history",
    "business_name_match",
    "social_media_presence",
]

FEATURE_DESCRIPTIONS = {
    "amount_log":                             "Log-transformed transaction amount (reduces right-skew)",
    "session_duration":                        "Duration of the UPI app session in seconds",
    "authentication_attempts":                 "Number of authentication attempts for this transaction",
    "transaction_amount_vs_sender_history":    "Transaction amount relative to sender's historical average",
    "geographic_disparity":                    "Distance (km) between sender's usual city and transaction origin",
    "transaction_time_of_day":                 "Hour of day (0–23) when the transaction occurred",
    "merchant_category_code":                  "Merchant category code (MCC) of the payee",
    "session_source":                          "How the session was initiated: app / web / sms-link",
    "time_between_link_click_and_transaction": "Seconds between payment-link click and transaction send",
    "dns_lookup_age":                          "Age of payee DNS record in days (new domains = phishing risk)",
    "input_timing_consistency":               "Consistency of user typing rhythm (0–1; bot-like = low)",
    "app_switching_frequency":                 "Number of app-switches during session (screen-share indicator)",
    "keyboard_input_speed":                    "Keyboard keystrokes per second during session",
    "screen_active_time":                      "Total screen-on time (seconds) during the session",
    "geographic_location_vs_ip":               "GPS vs IP-geolocation mismatch score (0–1); VPN/proxy flag",
    "background_data_usage":                   "MB consumed in background during session (malware indicator)",
    "time_between_otp_generation_and_input":   "Seconds between OTP dispatch and user entry",
    "pin_entry_speed":                         "PIN entry speed in keystrokes per second",
    "otp_request_frequency":                   "Number of OTP requests in the last 10 minutes",
    "otp_request_device_consistency":          "1 if OTP requested from the same device as the transaction",
    "transaction_velocity":                    "Number of transactions by sender in last 24 hours",
    "failed_transaction_count":                "Failed transactions by sender in last 7 days",
    "authorization_method":                    "Auth method used: pin / biometric / pattern",
    "transaction_type":                        "Type: payment / request / collect",
    "request_amount_roundness":                "Roundness of the requested amount (1 = perfectly round)",
    "request_frequency":                       "Collect requests received by user in last 7 days",
    "request_acceptance_rate":                 "Fraction of collect requests user has historically accepted",
    "time_to_respond_to_request":              "Seconds taken by user to respond to a collect request",
    "requester_account_age":                   "Age (days) of the collect requester's UPI account",
    "relationship_to_requester":               "User's known relationship: known / unknown / business",
    "upi_handle_age":                          "Age of payee UPI handle in days",
    "handle_similarity_score":                 "Similarity to known official handles (0–1); impersonation risk",
    "handle_contains_official_terms":          "1 if handle contains bank/gov/NPCI-like official terms",
    "handle_transaction_history":              "Number of past transactions on this UPI handle",
    "business_name_match":                     "Whether payee business name matches registered entity: match/mismatch/none",
    "social_media_presence":                   "Payee social media verification status: verified/unverified/none",
}


def load_and_clean_dataset(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.drop(columns=ALL_DROP, errors="ignore")
    list_cols = [
        c for c in df.columns
        if df[c].dtype == object
        and df[c].dropna().astype(str).str.startswith("[").any()
    ]
    if list_cols:
        df = df.drop(columns=list_cols)
    return df


def engineer_features(
    df: pd.DataFrame,
    label_encoders: dict = None,
    fit: bool = True,
) -> tuple:
    df = df.copy()

    if "amount" in df.columns:
        df["amount_log"] = np.log1p(df["amount"].clip(lower=0))
        df = df.drop(columns=["amount"])

    if label_encoders is None:
        label_encoders = {}

    for col in CAT_COLS:
        if col not in df.columns:
            continue
        df[col] = df[col].astype(str)
        if fit:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
        else:
            le = label_encoders.get(col)
            if le is not None:
                known = set(le.classes_)
                df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
                df[col] = le.transform(df[col])
            else:
                df[col] = 0

    return df, label_encoders


def get_X_y(df: pd.DataFrame, label_encoders: dict = None, fit: bool = True):
    df, label_encoders = engineer_features(df, label_encoders=label_encoders, fit=fit)
    available = [c for c in FEATURE_COLS if c in df.columns]
    if label_encoders is None:
        label_encoders = {}

    # Encode any remaining object columns among the selected features
    for col in available:
        if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
            if fit or col not in label_encoders:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                label_encoders[col] = le
            else:
                le = label_encoders.get(col)
                if le is not None:
                    known = set(le.classes_)
                    df[col] = df[col].apply(lambda x: x if x in known else le.classes_[0])
                    df[col] = le.transform(df[col].astype(str))
                else:
                    df[col] = 0

    # Ensure numeric types
    for col in available:
        try:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].fillna(0.0)
        except Exception as e:
            print(f"Problem column: {col}")
            print(df[col].head())
            raise e

    X = df[available]
    y = df["is_fraud"].astype(int)
    return X, y, label_encoders, available


def get_scaler(X_train: pd.DataFrame) -> StandardScaler:
    scaler = StandardScaler()
    scaler.fit(X_train)
    return scaler
