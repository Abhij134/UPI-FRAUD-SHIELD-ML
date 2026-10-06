import pandas as pd
df = pd.read_csv('p:/Researchproject/upi-fraud-ml-fixed/upi-fraud-ml/dataset/fraud_dataset.csv')
print("Non-fraud sample:")
nf = df[df['is_fraud']==0].iloc[0]
for k, v in nf.items():
    print(f"{k}: {v}")
