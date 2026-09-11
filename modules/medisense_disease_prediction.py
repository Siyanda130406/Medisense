
# MEDISENSE DISEASE PREDICTION MODULE
import pandas as pd
import numpy as np
import joblib

drive_path = '/content/drive/MyDrive/MediSense_Project/'

def load_model():
    """Load disease prediction model"""
    model = joblib.load(drive_path + 'disease_prediction_model.pkl')
    symptom_columns = joblib.load(drive_path + 'symptom_columns.pkl')
    disease_info = pd.read_csv(drive_path + 'disease_descriptions_final.csv')
    return model, symptom_columns, disease_info

def predict_disease(symptoms_list, model, symptom_columns, disease_info):
    """Predict disease based on symptoms"""
    features = pd.DataFrame(0, index=[0], columns=symptom_columns)
    for symptom in symptoms_list:
        symptom_clean = symptom.lower().strip()
        if symptom_clean in symptom_columns:
            features[symptom_clean] = 1
    
    prediction = model.predict(features)[0]
    probabilities = model.predict_proba(features)[0]
    confidence = max(probabilities) * 100
    
    # Get disease information
    disease_info_row = disease_info[disease_info['Disease'] == prediction]
    description = disease_info_row['Description'].iloc[0] if not disease_info_row.empty else ""
    
    return {
        'disease': prediction,
        'confidence': confidence,
        'description': description
    }
