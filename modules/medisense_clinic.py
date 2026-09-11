
# MEDISENSE CLINIC MODULE
import pandas as pd
import numpy as np
from math import radians, sin, cos, sqrt, asin

drive_path = '/content/drive/MyDrive/MediSense_Project/'

def load_clinics():
    """Load clinic data"""
    return pd.read_csv(drive_path + 'sa_clinic_database_full.csv')

def calculate_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two coordinates in km"""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    c = 2 * asin(sqrt(a))
    return c * 6371

def find_nearby_clinics(lat, lng, radius_km=20, df_clinics=None):
    """Find clinics within a certain radius"""
    if df_clinics is None:
        df_clinics = load_clinics()
    
    results = []
    for idx, clinic in df_clinics.iterrows():
        distance = calculate_distance(lat, lng, clinic['Latitude'], clinic['Longitude'])
        if distance <= radius_km:
            clinic_copy = clinic.to_dict()
            clinic_copy['Distance_km'] = round(distance, 2)
            results.append(clinic_copy)
    
    return sorted(results, key=lambda x: x['Distance_km'])

def find_emergency_clinics(lat, lng, radius_km=50, df_clinics=None):
    """Find clinics with emergency services"""
    if df_clinics is None:
        df_clinics = load_clinics()
    
    nearby = find_nearby_clinics(lat, lng, radius_km, df_clinics)
    return [c for c in nearby if c['Emergency_Services'] == 'Yes']
