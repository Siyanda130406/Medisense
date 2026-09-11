
# MEDISENSE CLINIC FINDER MODULE
import pandas as pd
from math import radians, sin, cos, sqrt, asin

drive_path = '/content/drive/MyDrive/MediSense_Project/'

def load_clinics():
    return pd.read_csv(drive_path + 'sa_clinic_database_full.csv')

def calculate_distance(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * 6371

def find_nearby_clinics(lat, lng, radius_km=20):
    df_clinics = load_clinics()
    results = []
    for idx, clinic in df_clinics.iterrows():
        distance = calculate_distance(lat, lng, clinic['Latitude'], clinic['Longitude'])
        if distance <= radius_km:
            results.append({
                'Clinic_ID': clinic['Clinic_ID'],
                'Clinic_Name': clinic['Clinic_Name'],
                'Clinic_Type': clinic['Clinic_Type'],
                'City': clinic['City'],
                'Province': clinic['Province'],
                'Distance_km': round(distance, 2),
                'Phone': clinic['Phone'],
                'Services': clinic['Services'],
                'Emergency_Services': clinic['Emergency_Services']
            })
    return sorted(results, key=lambda x: x['Distance_km'])

def find_clinics_for_disease(disease_name, lat, lng, radius_km=30):
    nearby = find_nearby_clinics(lat, lng, radius_km)
    disease_lower = disease_name.lower().strip()
    suitable = []
    for clinic in nearby:
        if disease_lower in clinic['Treatable_Conditions'].lower():
            suitable.append(clinic)
    return suitable
