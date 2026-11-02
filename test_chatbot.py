import pandas as pd
from modules import medisense_ml as m

# Load the datasets
df_sym = pd.read_csv('data/processed/symptom_descriptions.csv')
df_dis = pd.read_csv('data/processed/master_disease_database.csv')
df_cb = pd.read_csv('data/processed/chatbot_qa.csv')
df_mq = pd.read_csv('data/processed/medical_qa_database.csv')

# Initialize ML
m.initialize_ml(
    df_symptom_descriptions=df_sym,
    df_disease_master=df_dis,
    df_chatbot_qa=df_cb,
    df_medical_qa=df_mq,
)

print()
print("=" * 60)
print("TEST 1: symptom extraction")
print("=" * 60)
result = m.extract_symptoms_public('my head is pounding')
print("Result:", result)

print()
print("=" * 60)
print("TEST 2: semantic search")
print("=" * 60)
r = m.semantic_search_qa('my head is pounding', top_k=3)
if not r:
    print("  (no results)")
for item in r:
    print(f"  [{item['score']}] {item['question']}")

print()
print("=" * 60)
print("TEST 3: bilingual search (full pipeline)")
print("=" * 60)
b = m.bilingual_search('my head is pounding')
print("  language:", b['language'])
print("  normalized:", b['normalized_query'])
print("  results count:", len(b['results']))
if b['top_answer']:
    print("  top answer:", b['top_answer'][:150])
else:
    print("  top answer: (none)")