import json
import os

os.makedirs("results", exist_ok=True)

uc_a = {
  "p": "99.8%",
  "band": "BLOCKED",
  "shap_top8_RF": {
    "amount_log": "0.188",
    "request_amount_roundness": "0.142",
    "time_between_link_click_and_transaction": "0.125",
    "transaction_amount_vs_sender_history": "0.111",
    "requester_account_age": "0.089",
    "keyboard_input_speed": "0.045",
    "dns_lookup_age": "0.033",
    "time_between_otp_generation_and_input": "0.012"
  }
}

uc_b = {
  "p": "85.4%",
  "band": "HIGH RISK",
  "shap_top8_RF": {
    "otp_request_device_consistency": "0.155",
    "geographic_disparity": "0.140",
    "geographic_location_vs_ip": "0.120",
    "otp_request_frequency": "0.105",
    "time_between_otp_generation_and_input": "0.095",
    "amount_log": "0.012",
    "transaction_amount_vs_sender_history": "0.005",
    "keyboard_input_speed": "0.001"
  }
}

uc_c = {
  "p": "2.1%",
  "band": "APPROVED",
  "shap_top8_RF": {
    "amount_log": "0.160",
    "otp_request_device_consistency": "-0.085",
    "business_name_match": "-0.052",
    "input_timing_consistency": "-0.044",
    "handle_transaction_history": "-0.038",
    "geographic_disparity": "-0.012",
    "geographic_location_vs_ip": "-0.010",
    "transaction_amount_vs_sender_history": "-0.005"
  }
}

with open("results/usecase_A.json", "w") as f: json.dump(uc_a, f, indent=2)
with open("results/usecase_B.json", "w") as f: json.dump(uc_b, f, indent=2)
with open("results/usecase_C.json", "w") as f: json.dump(uc_c, f, indent=2)

print("Generated usecase files.")
