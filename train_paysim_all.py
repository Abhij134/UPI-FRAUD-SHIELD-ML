import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, average_precision_score
import time
import os

def engineer_paysim_features(df):
    df = df.copy()
    df['errorBalanceOrig'] = df['newbalanceOrig'] + df['amount'] - df['oldbalanceOrg']
    df['errorBalanceDest'] = df['oldbalanceDest'] + df['amount'] - df['newbalanceDest']
    df['hour_of_day'] = df['step'] % 24
    df['type_CASH_OUT'] = (df['type'] == 'CASH_OUT').astype(int)
    df['type_DEBIT'] = (df['type'] == 'DEBIT').astype(int)
    df['type_PAYMENT'] = (df['type'] == 'PAYMENT').astype(int)
    df['type_TRANSFER'] = (df['type'] == 'TRANSFER').astype(int)
    return df

def main():
    print("Loading PaySim dataset...")
    df = pd.read_csv("upi-fraud-ml/dataset/PS_20174392719_1491204439457_log.csv")
    df = engineer_paysim_features(df)
    
    features = [
        'amount', 'oldbalanceOrg', 'newbalanceOrig', 
        'oldbalanceDest', 'newbalanceDest',
        'errorBalanceOrig', 'errorBalanceDest', 'hour_of_day',
        'type_CASH_OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
    ]
    
    X = df[features]
    y = df['isFraud']
    
    fraud_indices = df[df['isFraud'] == 1].index
    legit_indices = df[df['isFraud'] == 0].index
    
    np.random.seed(42)
    legit_sample_indices = np.random.choice(legit_indices, len(fraud_indices) * 10, replace=False)
    undersampled_indices = np.concatenate([fraud_indices, legit_sample_indices])
    
    X_res = X.loc[undersampled_indices]
    y_res = y.loc[undersampled_indices]
    
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42, stratify=y_res)
    
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)
    
    models = {
        "Logistic Regression": {
            "model": LogisticRegression(max_iter=1000, C=0.5, class_weight="balanced", random_state=42),
            "needs_scaling": True,
            "color": "#636EFA",
            "description": "Baseline linear model. Fast and interpretable. Uses log-odds to score fraud risk."
        },
        "Decision Tree": {
            "model": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20, class_weight="balanced", random_state=42),
            "needs_scaling": False,
            "color": "#EF553B",
            "description": "Rule-based tree. Highly interpretable. max_depth=6 prevents overfitting."
        },
        "Random Forest": {
            "model": RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced', n_jobs=-1),
            "needs_scaling": False,
            "color": "#00CC96",
            "description": "Ensemble of 100 trees. Robust to outliers. Best overall trade-off on fraud data."
        },
        "Gradient Boosting": {
            "model": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42),
            "needs_scaling": False,
            "color": "#AB63FA",
            "description": "Sequential boosting. High accuracy on tabular data."
        },
        "Neural Network": {
            "model": MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=200, alpha=0.01, random_state=42, early_stopping=True),
            "needs_scaling": True,
            "color": "#FFA15A",
            "description": "2-layer MLP. Learns non-linear patterns."
        }
    }
    
    print("Training and Evaluating 5 Models...")
    results = {}
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
    
    for name, cfg in models.items():
        print(f"Training {name}...")
        start = time.time()
        X_train_fold = X_train_sc if cfg["needs_scaling"] else X_train
        X_test_fold = X_test_sc if cfg["needs_scaling"] else X_test
        
        mdl = cfg["model"]
        mdl.fit(X_train_fold, y_train)
        
        # Evaluate
        y_pred = mdl.predict(X_test_fold)
        y_prob = mdl.predict_proba(X_test_fold)[:, 1]
        
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        
        # Feature importance mock for non-tree models
        if hasattr(mdl, "feature_importances_"):
            imp = pd.Series(mdl.feature_importances_, index=features).sort_values(ascending=False).head(10)
        elif hasattr(mdl, "coef_"):
            imp = pd.Series(np.abs(mdl.coef_[0]), index=features).sort_values(ascending=False).head(10)
        else:
            imp = pd.Series(1.0/len(features), index=features).sort_values(ascending=False).head(10)
            
        results[name] = {
            "metrics": {
                "Accuracy": acc, "Precision": prec, "Recall": rec, "F1-Score": f1, "AUC-ROC": auc,
                "True Negatives": int(tn), "False Positives": int(fp), "False Negatives": int(fn), "True Positives": int(tp)
            },
            "roc": (fpr, tpr),
            "importance": imp
        }
        
        print(f"Done in {time.time() - start:.2f}s")
        
    artifacts_dir = "upi-fraud-ml/notebooks/models"
    os.makedirs(artifacts_dir, exist_ok=True)
    
    with open(os.path.join(artifacts_dir, "paysim_all_models.pkl"), "wb") as f:
        pickle.dump(models, f)
    with open(os.path.join(artifacts_dir, "paysim_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(artifacts_dir, "paysim_all_features.pkl"), "wb") as f:
        pickle.dump(features, f)
    with open(os.path.join(artifacts_dir, "paysim_results.pkl"), "wb") as f:
        pickle.dump(results, f)
        
    print("All models and results saved successfully.")

if __name__ == "__main__":
    main()
