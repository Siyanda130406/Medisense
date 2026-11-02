import pandas as pd

df = pd.read_csv('data/processed/medical_qa_database.csv')
print('Total rows:', len(df))
print('Columns:', df.columns.tolist())

row = df[df['question'].str.contains('flu', case=False, na=False)].iloc[0]
print()
print('Q:', row['question'])
print('A length:', len(row['answer']))
print('A first 600 chars:')
print(row['answer'][:600])