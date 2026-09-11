
# MEDISENSE HEALTH EDUCATION MODULE
import pandas as pd
import joblib

drive_path = '/content/drive/MyDrive/MediSense_Project/'

def load_health_data():
    """Load health education data"""
    return pd.read_csv(drive_path + 'disease_descriptions_final.csv')

def get_disease_info(disease_name, df):
    """Get disease information"""
    disease_lower = disease_name.lower().strip()
    for idx, row in df.iterrows():
        if disease_lower in row['Disease'].lower():
            return {
                'disease': row['Disease'],
                'description': row['Description'],
                'symptoms': row['Symptoms'],
                'precautions': row['Precautions']
            }
    return None

def health_education_response(user_input, df):
    """Generate health education response"""
    disease_keywords = ['flu', 'malaria', 'covid', 'tuberculosis', 'cold', 'pneumonia',
                       'gastroenteritis', 'hepatitis', 'hypertension', 'diabetes',
                       'hiv', 'aids', 'stroke', 'heart disease', 'asthma', 'arthritis',
                       'migraine', 'allergy', 'bronchitis', 'sinusitis', 'uti']
    
    user_lower = user_input.lower()
    found = None
    for disease in disease_keywords:
        if disease in user_lower:
            found = disease
            break
    
    if not found:
        return "Please ask about a specific disease. Example: 'Tell me about malaria'"
    
    info = get_disease_info(found, df)
    if not info:
        return f"Sorry, I don't have information about '{found}'."
    
    return f"""
DISEASE INFORMATION: {info['disease']}

DESCRIPTION:
{info['description']}

COMMON SYMPTOMS:
{info['symptoms']}

PRECAUTIONS:
{info['precautions']}

This is educational information only. Consult a healthcare professional.
"""
