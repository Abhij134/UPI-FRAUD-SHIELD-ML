import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, average_precision_score
import time

def engineer_paysim_features(df):
    """
    Apply financial heuristic feature engineering for PaySim.
    """
    df = df.copy()
    
    # 1. Error in Sender's Balance
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    
    # 2. Error in Receiver's Balance
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    
    # 3. Time of day
    df['hour_of_day'] = df['step'] % 24
    
    # 4. Transaction Type Encoding
    # PaySim types: CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER
    # We will one-hot encode them. To ensure consistent columns in production, we do it explicitly.
    df['type_CASH_OUT'] = (df['type'] == 'CASH_OUT').astype(int)
    df['type_DEBIT'] = (df['type'] == 'DEBIT').astype(int)
    df['type_PAYMENT'] = (df['type'] == 'PAYMENT').astype(int)
    df['type_TRANSFER'] = (df['type'] == 'TRANSFER').astype(int)
    
    return df

def main():
    print("Loading PaySim dataset (this might take a minute)...")
    df = pd.read_csv("upi-fraud-ml/dataset/PS_20174392719_1491204439457_log.csv")
    print(f"Original Dataset Shape: {df.shape}")
    
    print("Engineering ledger features...")
    df = engineer_paysim_features(df)
    
    # Define features to keep
    features = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest',
        'errorBalanceOrig', 'errorBalanceDest', 'hour_of_day',
        'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
    ]
    
    X = df[features]
    y = df['isFraud']
    
    # We have ~6.3 million rows, and only ~8k are fraud.
    # We will undersample the legit transactions (10:1 ratio) to fit in RAM and train fast.
    print(f"Total Fraud: {y.sum()}")
    print("Undersampling majority class to a 10:1 ratio...")
    
    fraud_indices = df[df['isFraud'] == 1].index
    legit_indices = df[df['isFraud'] == 0].index
    
    np.random.seed(42)
    # 10 times more legit than fraud
    legit_sample_indices = np.random.choice(legit_indices, len(fraud_indices) * 10, replace=False)
    
    undersampled_indices = np.concatenate([fraud_indices, legit_sample_indices])
    X_res = X.loc[undersampled_indices]
    y_res = y.loc[undersampled_indices]
    
    print(f"Resampled Dataset Shape: {X_res.shape}")
    
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
    
    print("Training Random Forest Classifier on PaySim...")
    start_time = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1)
    rf.fit(X_train, y_train)
    print(f"Training completed in {time.time() - start_time:.2f} seconds.")
    
    # Evaluate
    y_pred = rf.predict(X_test)
    y_prob = rf.predict_proba(X_test)[:, 1]
    
    print("\n--- Evaluation on Test Split ---")
    print(classification_report(y_test, y_pred))
    print(f"Average Precision (AUPRC): {average_precision_score(y_test, y_prob):.4f}")
    
    # Save artifacts
    print("\nSaving new PaySim artifacts to notebook/models/...")
    
    artifacts_dir = "upi-fraud-ml/notebooks/models"
    import os
    os.makedirs(artifacts_dir, exist_ok=True)
    
    with open(os.path.join(artifacts_dir, "paysim_rf.pkl"), "wb") as f:
        pickle.dump(rf, f)
        
    with open(os.path.join(artifacts_dir, "paysim_features.pkl"), "wb") as f:
        pickle.dump(features, f)
        
    print("Done! You can now use these artifacts in the UI.")

if __name__ == "__main__":
    main()
