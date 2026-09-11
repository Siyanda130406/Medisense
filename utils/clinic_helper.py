import pandas as pd
import os

def get_clinic_data():
    """Load clinic data from CSV"""
    base_path = os.path.dirname(os.path.dirname(__file__))
    clinic_path = os.path.join(base_path, 'data', 'processed', 'sa_clinic_database_full.csv')
    
    try:
        df = pd.read_csv(clinic_path)
        return df
    except:
        # Return sample clinics if file not found
        return pd.DataFrame([
            {'Clinic_ID': 1, 'Clinic_Name': 'Empangeni Clinic', 'City': 'Empangeni', 'Area': 'Empangeni', 'Province': 'KwaZulu-Natal', 'Phone': '035 123 4567', 'Latitude': -28.75, 'Longitude': 31.95},
            {'Clinic_ID': 2, 'Clinic_Name': 'Ngwelezane Clinic', 'City': 'Empangeni', 'Area': 'Ngwelezane', 'Province': 'KwaZulu-Natal', 'Phone': '035 789 0123', 'Latitude': -28.72, 'Longitude': 31.92},
            {'Clinic_ID': 3, 'Clinic_Name': 'Richards Bay Clinic', 'City': 'Richards Bay', 'Area': 'Richards Bay', 'Province': 'KwaZulu-Natal', 'Phone': '035 456 7890', 'Latitude': -28.78, 'Longitude': 32.05}
        ])

def get_clinics_by_city(city):
    """Get clinics in a specific city"""
    df = get_clinic_data()
    return df[df['City'].str.lower().str.contains(city.lower(), na=False)]

def get_clinic_by_id(clinic_id):
    """Get clinic by ID"""
    df = get_clinic_data()
    result = df[df['Clinic_ID'] == clinic_id]
    if not result.empty:
        return result.iloc[0].to_dict()
    return None

def get_all_clinics():
    """Get all clinics as list of dictionaries"""
    df = get_clinic_data()
    return df.to_dict('records')

def get_clinic_options():
    """Get clinic options for dropdown (ID, Name)"""
    df = get_clinic_data()
    return [(row['Clinic_ID'], row['Clinic_Name']) for _, row in df.iterrows()]