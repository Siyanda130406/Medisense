import pandas as pd

df = pd.read_csv('data/processed/chatbot_qa.csv')
print('Rows:', len(df))
print('Columns:', df.columns.tolist())
print()
for i in range(3):
    row = df.iloc[i]
    print(f"--- Row {i+1} ---")
    print("Q:", row.get('question', ''))
    print("A length:", len(str(row.get('answer', ''))))
    print("A:", str(row.get('answer', ''))[:400])
    print("Tags:", row.get('tags', ''))
    print()