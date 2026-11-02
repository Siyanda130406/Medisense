import pandas as pd
df = pd.read_csv('data/processed/sa_clinic_database_real.csv')
print('Columns:', df.columns.tolist())
print()
print('First row:')
print(df.iloc[0].to_dict())
