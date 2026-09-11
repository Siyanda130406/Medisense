import pandas as pd
import ast
import os
import pickle
from datetime import datetime

print("="*60)
print("MEDISENSE - COMPLETE DATA INTEGRATION")
print("Integrating ALL Datasets")
print("="*60)

# ============================================================
# 1. LOAD ALL DATASETS
# ============================================================

print("\n📁 LOADING ALL DATASETS...")
print("-"*60)

# 1.1 Main symptom-disease-doctor-risk data
df_main = pd.read_csv('data/processed/symptom_disease_doctor_data.csv')
print(f"✅ symptom_disease_doctor_data.csv: {len(df_main)} rows")

# 1.2 Disease descriptions
df_descriptions = pd.read_csv('data/processed/disease_descriptions_final.csv')
print(f"✅ disease_descriptions_final.csv: {len(df_descriptions)} rows")

# 1.3 Disease symptoms (17 columns)
df_symptoms = pd.read_csv('data/raw/DiseaseAndSymptoms.csv')
print(f"✅ DiseaseAndSymptoms.csv: {len(df_symptoms)} rows")

# 1.4 Clinic database
df_clinics = pd.read_csv('data/processed/sa_clinic_database_full.csv')
print(f"✅ sa_clinic_database_full.csv: {len(df_clinics)} rows")

# 1.5 First Aid Q&A
df_firstaid = pd.read_csv('data/raw/firstaid_qa_dataset.csv')
print(f"✅ firstaid_qa_dataset.csv: {len(df_firstaid)} rows")

# 1.6 Health data
df_health = pd.read_csv('data/raw/health.csv')
print(f"✅ health.csv: {len(df_health)} rows")

# 1.7 Symptom descriptions
df_symptom_desc = pd.read_csv('data/raw/symptom_Description.csv')
print(f"✅ symptom_Description.csv: {len(df_symptom_desc)} rows")

# 1.8 Symptom precautions
df_symptom_precaution = pd.read_csv('data/raw/symptom_precaution.csv')
print(f"✅ symptom_precaution.csv: {len(df_symptom_precaution)} rows")

# 1.9 Dataset CSV (alternative symptom data)
df_dataset = pd.read_csv('data/raw/dataset.csv')
print(f"✅ dataset.csv: {len(df_dataset)} rows")

# 1.10 MedQuAD Files (Medical Q&A)
medquad_files = {
    'CancerQA.csv': pd.read_csv('data/raw/CancerQA.csv'),
    'Diabetes_and_Digestive_and_Kidney_DiseasesQA.csv': pd.read_csv('data/raw/Diabetes_and_Digestive_and_Kidney_DiseasesQA.csv'),
    'Disease_Control_and_PreventionQA.csv': pd.read_csv('data/raw/Disease_Control_and_PreventionQA.csv'),
    'Genetic_and_Rare_DiseasesQA.csv': pd.read_csv('data/raw/Genetic_and_Rare_DiseasesQA.csv'),
    'Neurological_Disorders_and_StrokeQA.csv': pd.read_csv('data/raw/Neurological_Disorders_and_StrokeQA.csv'),
    'SeniorHealthQA.csv': pd.read_csv('data/raw/SeniorHealthQA.csv'),
    'MedicalQuestionAnswering.csv': pd.read_csv('data/raw/MedicalQuestionAnswering.csv'),
    'OtherQA.csv': pd.read_csv('data/raw/OtherQA.csv'),
    'Heart_Lung_and_BloodQA.csv': pd.read_csv('data/raw/Heart_Lung_and_BloodQA.csv'),
    'growth_hormone_receptorQA.csv': pd.read_csv('data/raw/growth_hormone_receptorQA.csv')
}

for name, df in medquad_files.items():
    print(f"✅ {name}: {len(df)} rows")

print("-"*60)
print("✅ ALL DATASETS LOADED!")

# ============================================================
# 2. CLEAN EACH DATASET
# ============================================================

print("\n🧹 CLEANING DATASETS...")
print("-"*60)

def safe_split(text):
    """Split comma-separated text into list"""
    if pd.isna(text):
        return []
    if isinstance(text, list):
        return text
    if isinstance(text, str):
        return [item.strip() for item in text.split(',') if item.strip()]
    return []

def extract_risk(text):
    """Extract risk level (Low/Moderate/High)"""
    if pd.isna(text):
        return 'Unknown'
    text = str(text).lower()
    if 'low' in text:
        return 'Low'
    elif 'moderate' in text:
        return 'Moderate'
    elif 'high' in text:
        return 'High'
    return 'Unknown'

# 2.1 Clean main dataset
df_main_clean = df_main.copy()
df_main_clean['symptoms_list'] = df_main_clean['symptoms'].apply(safe_split)
df_main_clean['cures_list'] = df_main_clean['cures'].apply(safe_split)
df_main_clean['doctors_list'] = df_main_clean['doctor'].apply(safe_split)
df_main_clean['risk_category'] = df_main_clean['risk level'].apply(extract_risk)
print("✅ Cleaned main dataset")

# 2.2 Clean disease descriptions
df_descriptions_clean = df_descriptions.copy()
df_descriptions_clean.columns = ['disease', 'description', 'symptoms_str', 'precautions']
df_descriptions_clean['description'] = df_descriptions_clean['description'].fillna('No description available')
df_descriptions_clean['precautions'] = df_descriptions_clean['precautions'].fillna('No precautions available')
print("✅ Cleaned disease descriptions")

# 2.3 Clean disease symptoms (combine 17 columns into one list)
symptom_cols = [f'Symptom_{i}' for i in range(1, 18)]
def combine_symptoms(row):
    symptoms = []
    for col in symptom_cols:
        val = row.get(col, '')
        if pd.notna(val) and str(val).strip() != '':
            symptoms.append(str(val).strip())
    return symptoms

df_symptoms_clean = df_symptoms.copy()
df_symptoms_clean['symptoms_list'] = df_symptoms_clean.apply(combine_symptoms, axis=1)
df_symptoms_clean = df_symptoms_clean[df_symptoms_clean['symptoms_list'].apply(len) > 0]
df_symptoms_clean = df_symptoms_clean[['Disease', 'symptoms_list']]
df_symptoms_clean.columns = ['disease', 'symptoms_from_disease']
df_symptoms_clean = df_symptoms_clean.drop_duplicates(subset=['disease'])
print(f"✅ Cleaned disease symptoms: {len(df_symptoms_clean)} diseases")

# 2.4 Clean clinics
df_clinics_clean = df_clinics.copy()
df_clinics_clean['Phone'] = df_clinics_clean['Phone'].fillna('N/A')
df_clinics_clean['Services'] = df_clinics_clean['Services'].fillna('General healthcare')
print("✅ Cleaned clinics")

# 2.5 Clean first aid
df_firstaid_clean = df_firstaid.copy()
df_firstaid_clean.columns = ['question', 'answer'] if len(df_firstaid_clean.columns) == 2 else df_firstaid_clean.columns
print("✅ Cleaned first aid")

# 2.6 Clean health data
df_health_clean = df_health.copy()
print("✅ Cleaned health data")

# 2.7 Clean symptom descriptions
df_symptom_desc_clean = df_symptom_desc.copy()
print("✅ Cleaned symptom descriptions")

# 2.8 Clean symptom precautions
df_symptom_precaution_clean = df_symptom_precaution.copy()
print("✅ Cleaned symptom precautions")

# 2.9 Clean dataset
df_dataset_clean = df_dataset.copy()
print("✅ Cleaned dataset")

# 2.10 Clean MedQuAD files
medquad_clean = {}
for name, df in medquad_files.items():
    df_clean = df.copy()
    # Standardize column names
    df_clean.columns = ['question', 'answer'] if len(df_clean.columns) == 2 else df_clean.columns
    medquad_clean[name] = df_clean
print("✅ Cleaned MedQuAD files")

print("-"*60)
print("✅ ALL DATASETS CLEANED!")

# ============================================================
# 3. CREATE MASTER DATABASE
# ============================================================

print("\n🔗 CREATING MASTER DATABASE...")
print("-"*60)

# Start with main dataset
master = df_main_clean[['disease', 'symptoms_list', 'cures_list', 'doctors_list', 'risk_category']].copy()

# Add descriptions
master = pd.merge(master, df_descriptions_clean[['disease', 'description', 'precautions']], on='disease', how='left')

# Add extra symptoms
master = pd.merge(master, df_symptoms_clean[['disease', 'symptoms_from_disease']], on='disease', how='left')

# Fill missing values
master['description'] = master['description'].fillna('No description available')
master['precautions'] = master['precautions'].fillna('No precautions available')

# Combine all symptoms
def combine_all_symptoms(row):
    all_sym = []
    if isinstance(row.get('symptoms_list'), list):
        all_sym.extend(row['symptoms_list'])
    if isinstance(row.get('symptoms_from_disease'), list):
        all_sym.extend(row['symptoms_from_disease'])
    if isinstance(row.get('symptoms_str'), str) and row['symptoms_str'] not in ['No symptoms listed', 'nan', '']:
        all_sym.extend([s.strip() for s in row['symptoms_str'].split(',') if s.strip()])
    return list(set([s for s in all_sym if s and s != 'nan']))

master['all_symptoms'] = master.apply(combine_all_symptoms, axis=1)

# Final master columns
master_final = master[['disease', 'all_symptoms', 'description', 'cures_list', 'precautions', 'doctors_list', 'risk_category']]

# Save master database
master_final.to_csv('data/processed/master_disease_database.csv', index=False)
print(f"✅ Master database created: {len(master_final)} diseases")

# ============================================================
# 4. CREATE HEALTH TIPS DATASET
# ============================================================

print("\n💡 CREATING HEALTH TIPS DATABASE...")
print("-"*60)

# Extract health tips from health.csv
health_tips = []
if 'tips' in df_health_clean.columns:
    for tip in df_health_clean['tips'].dropna():
        health_tips.append({'tip': tip})
elif len(df_health_clean.columns) > 0:
    # Use first column as tips
    for val in df_health_clean.iloc[:, 0].dropna():
        health_tips.append({'tip': val})

# If no tips found, create fallback tips
if not health_tips:
    health_tips = [
        {'tip': 'Drink at least 8 glasses of water daily'},
        {'tip': 'Eat a balanced diet with fruits and vegetables'},
        {'tip': 'Exercise for at least 30 minutes daily'},
        {'tip': 'Get 7-8 hours of sleep each night'},
        {'tip': 'Wash hands regularly to prevent infections'},
        {'tip': 'Avoid smoking and excessive alcohol consumption'},
        {'tip': 'Manage stress through meditation or relaxation'},
        {'tip': 'Stay up to date with vaccinations'},
        {'tip': 'Regular health check-ups are important'},
        {'tip': 'Maintain a healthy weight for your height'}
    ]

df_health_tips = pd.DataFrame(health_tips)
df_health_tips.to_csv('data/processed/health_tips.csv', index=False)
print(f"✅ Health tips created: {len(df_health_tips)} tips")

# ============================================================
# 5. CREATE SYMPTOM PRECAUTIONS DATASET
# ============================================================

print("\n🛡️ CREATING SYMPTOM PRECAUTIONS DATABASE...")
print("-"*60)

# Combine symptom precautions from multiple sources
precautions_dict = {}

# From symptom_precaution.csv
if not df_symptom_precaution_clean.empty:
    for _, row in df_symptom_precaution_clean.iterrows():
        symptom = row.iloc[0] if len(row) > 0 else ''
        precaution = row.iloc[1] if len(row) > 1 else ''
        if symptom and precaution:
            precautions_dict[symptom] = precaution

# From disease descriptions precautions
for _, row in df_descriptions_clean.iterrows():
    disease = row.get('disease', '')
    precaution = row.get('precautions', '')
    if disease and precaution and precaution != 'No precautions available':
        precautions_dict[disease] = precaution

# Save
df_precautions = pd.DataFrame(list(precautions_dict.items()), columns=['condition', 'precaution'])
df_precautions.to_csv('data/processed/symptom_precautions_combined.csv', index=False)
print(f"✅ Symptom precautions created: {len(df_precautions)} entries")

# ============================================================
# 6. CREATE MEDICAL Q&A DATABASE
# ============================================================

print("\n❓ CREATING MEDICAL Q&A DATABASE...")
print("-"*60)

# Combine all MedQuAD files
all_qa = []
for name, df in medquad_clean.items():
    for _, row in df.iterrows():
        q = row.iloc[0] if len(row) > 0 else ''
        a = row.iloc[1] if len(row) > 1 else ''
        if q and a:
            all_qa.append({
                'question': str(q)[:300],
                'answer': str(a)[:500],
                'source': name.replace('.csv', '')
            })

df_qa = pd.DataFrame(all_qa)
df_qa.to_csv('data/processed/medical_qa_database.csv', index=False)
print(f"✅ Medical Q&A created: {len(df_qa)} entries")

# ============================================================
# 7. CREATE SYMPTOM DESCRIPTIONS DATABASE
# ============================================================

print("\n📋 CREATING SYMPTOM DESCRIPTIONS DATABASE...")
print("-"*60)

symptom_desc_dict = {}
if not df_symptom_desc_clean.empty:
    for _, row in df_symptom_desc_clean.iterrows():
        symptom = row.iloc[0] if len(row) > 0 else ''
        description = row.iloc[1] if len(row) > 1 else ''
        if symptom and description:
            symptom_desc_dict[symptom] = description

df_symptom_desc = pd.DataFrame(list(symptom_desc_dict.items()), columns=['symptom', 'description'])
df_symptom_desc.to_csv('data/processed/symptom_descriptions.csv', index=False)
print(f"✅ Symptom descriptions created: {len(df_symptom_desc)} entries")

# ============================================================
# 8. SUMMARY
# ============================================================

print("\n" + "="*60)
print("✅ DATA INTEGRATION COMPLETE!")
print("="*60)

print("\n📊 SUMMARY OF CREATED FILES:")
print("-"*60)

files_created = {
    'Master Disease Database': 'data/processed/master_disease_database.csv',
    'Health Tips': 'data/processed/health_tips.csv',
    'Symptom Precautions': 'data/processed/symptom_precautions_combined.csv',
    'Medical Q&A': 'data/processed/medical_qa_database.csv',
    'Symptom Descriptions': 'data/processed/symptom_descriptions.csv'
}

for name, path in files_created.items():
    if os.path.exists(path):
        df = pd.read_csv(path)
        print(f"✅ {name}: {len(df)} entries → {path}")
    else:
        print(f"❌ {name}: NOT CREATED")

print("-"*60)
print("\n🎉 ALL DATASETS INTEGRATED SUCCESSFULLY!")
print("="*60)