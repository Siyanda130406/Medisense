
# MEDISENSE NLP SYMPTOM CHECKER
# Functions for symptom checking and chatbot responses

import re
import pandas as pd
import joblib

def load_symptom_checker():
    """Load the symptom checker dictionaries"""
    disease_symptoms = joblib.load('disease_symptoms_dict.pkl')
    disease_precautions = joblib.load('disease_precaution_dict.pkl')
    return disease_symptoms, disease_precautions

def check_symptoms(patient_symptoms, disease_symptoms_dict, disease_precaution_dict):
    """Check symptoms and suggest possible diseases"""
    patient_symptoms = [s.lower().strip() for s in patient_symptoms]
    results = []
    
    for disease, disease_symptoms in disease_symptoms_dict.items():
        matching = [s for s in patient_symptoms if s in disease_symptoms]
        if matching:
            match_percentage = (len(matching) / len(disease_symptoms)) * 100
            precautions = disease_precaution_dict.get(disease, [])
            results.append({
                'disease': disease,
                'matching_symptoms': matching,
                'match_percentage': match_percentage,
                'precautions': precautions
            })
    
    results.sort(key=lambda x: x['match_percentage'], reverse=True)
    return results

def chatbot_response(user_input, disease_symptoms_dict, disease_precaution_dict):
    """Generate chatbot response for patient symptoms"""
    symptom_keywords = ['fever', 'cough', 'headache', 'vomiting', 'diarrhea', 
                       'nausea', 'sneezing', 'runny nose', 'chest pain', 'sore throat',
                       'pain', 'rash', 'dizziness', 'fatigue', 'chills']
    
    user_lower = user_input.lower()
    found_symptoms = [k for k in symptom_keywords if k in user_lower]
    
    if not found_symptoms:
        return "Please describe your symptoms. For example: 'I have a fever and cough'"
    
    results = check_symptoms(found_symptoms, disease_symptoms_dict, disease_precaution_dict)
    
    if not results:
        return "I couldn't find matches for your symptoms. Please consult a healthcare professional."
    
    response = f"I found these possible conditions based on your symptoms ({', '.join(found_symptoms)}):\n\n"
    
    for i, result in enumerate(results[:3]):
        response += f"**{i+1}. {result['disease']}**\n"
        response += f"   - Matching symptoms: {', '.join(result['matching_symptoms'])}\n"
        response += f"   - Match confidence: {result['match_percentage']:.1f}%\n"
        if result['precautions']:
            response += f"   - Precautions: {', '.join(result['precautions'][:2])}\n"
        response += "\n"
    
    response += "\n⚠️ This is not medical advice. Please consult a healthcare professional."
    return response
