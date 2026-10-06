import pandas as pd
df = pd.read_csv('p:/Researchproject/upi-fraud-ml-fixed/upi-fraud-ml/dataset/fraud_dataset.csv')
print("Class counts:")
print(df['is_fraud'].value_counts())
