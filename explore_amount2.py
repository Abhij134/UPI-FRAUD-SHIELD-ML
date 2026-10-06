import pandas as pd
df = pd.read_csv('p:/Researchproject/upi-fraud-ml-fixed/upi-fraud-ml/dataset/fraud_dataset.csv')
print("Min amount:", df['amount'].min())
print("Max amount:", df['amount'].max())
print("1st quartile:", df['amount'].quantile(0.25))
