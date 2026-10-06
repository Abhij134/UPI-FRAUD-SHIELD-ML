import pandas as pd
import numpy as np
df = pd.read_csv('p:/Researchproject/upi-fraud-ml-fixed/upi-fraud-ml/dataset/fraud_dataset.csv')
print("Non-fraud amount mean:", df[df['is_fraud']==0]['amount'].mean())
print("Fraud amount mean:", df[df['is_fraud']==1]['amount'].mean())
print("Non-fraud amount_log mean:", np.log1p(df[df['is_fraud']==0]['amount']).mean())
print("Fraud amount_log mean:", np.log1p(df[df['is_fraud']==1]['amount']).mean())
