from app import app
from flask import render_template, session, redirect, url_for, request, flash, Response, jsonify
from datetime import datetime, timedelta
import sqlite3
import os
import hashlib
import pandas as pd
import json
import csv
from io import StringIO
import joblib
import pickle
import numpy as np
import random
import re
import secrets
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from functools import wraps

# ===== FIX #8: Spell correction library =====
from difflib import get_close_matches
# ===== END FIX #8 =====

# ============================================================
# FIX #10: DB path helper (persistent data dir for online hosting)
# ============================================================
def _get_db_dir():
    env_dir = os.environ.get('MEDISENSE_DATA_DIR', '').strip()
    if env_dir:
        os.makedirs(env_dir, exist_ok=True)
        return env_dir
    os.makedirs('database', exist_ok=True)
    return 'database'

DB_PATH = os.path.join(_get_db_dir(), 'medisense_users.db')
# ============================================================
# END FIX #10
# ============================================================

# ============================================================
# LOAD ALL DATASETS
# ============================================================

print("="*60)
print("LOADING MEDISENSE DATASETS")
print("="*60)

def safe_load_csv(file_path):
    try:
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            print(f"✅ Loaded: {os.path.basename(file_path)} ({len(df)} rows)")
            return df
        else:
            print(f"❌ File not found: {file_path}")
            return None
    except Exception as e:
        print(f"❌ Error loading {file_path}: {e}")
        return None

base_dir = os.path.dirname(os.path.dirname(__file__))

# Load all datasets
df_master = safe_load_csv(os.path.join(base_dir, 'data', 'processed', 'master_disease_database.csv'))
df_health_tips = safe_load_csv(os.path.join(base_dir, 'data', 'processed', 'health_tips.csv'))
df_medical_qa = safe_load_csv(os.path.join(base_dir, 'data', 'processed', 'medical_qa_database.csv'))
df_symptom_precautions = safe_load_csv(os.path.join(base_dir, 'data', 'processed', 'symptom_precautions_combined.csv'))
df_symptom_descriptions = safe_load_csv(os.path.join(base_dir, 'data', 'processed', 'symptom_descriptions.csv'))
df_clinics = safe_load_csv(os.path.join(base_dir, 'data', 'processed', 'sa_clinic_database_real.csv'))
df_firstaid = safe_load_csv(os.path.join(base_dir, 'data', 'raw', 'firstaid_qa_dataset.csv'))

# ============================================================
# LOAD MEDICINE DATASETS
# ============================================================

print("="*60)
print("LOADING MEDICINE DATASETS")
print("="*60)

df_medicines_master = safe_load_csv(os.path.join(base_dir, 'data', 'processed', 'medisense_master_medicines.csv'))
df_disease_to_medicine = safe_load_csv(os.path.join(base_dir, 'data', 'processed', 'disease_to_medicine_master.csv'))
df_medicine_to_disease = safe_load_csv(os.path.join(base_dir, 'data', 'processed', 'medicine_to_disease_master.csv'))

if df_medicines_master is not None:
    print(f"✅ Loaded Master Medicines: {len(df_medicines_master)} medicines")
if df_disease_to_medicine is not None:
    print(f"✅ Loaded Disease→Medicine: {len(df_disease_to_medicine)} diseases")
if df_medicine_to_disease is not None:
    print(f"✅ Loaded Medicine→Disease: {len(df_medicine_to_disease)} medicines")

# ML Model - RandomForest (original)
model_path = os.path.join(base_dir, 'models', 'medisense_model.pkl')
scaler_path = os.path.join(base_dir, 'models', 'medisense_scaler.pkl')
try:
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    model_loaded = True
    print("✅ Loaded RandomForest ML Model")
except:
    model = None
    scaler = None
    model_loaded = False
    print("❌ Could not load ML Model")

# ===== MULTIPLE ML MODELS (XGBoost, Neural Network, Ensemble) =====
model_xgb = None
model_nn = None
model_ensemble = None
models_loaded = False
try:
    xgb_path = os.path.join(base_dir, 'models', 'xgb_model.pkl')
    nn_path = os.path.join(base_dir, 'models', 'nn_model.pkl')
    ensemble_path = os.path.join(base_dir, 'models', 'ensemble_model.pkl')
    
    if os.path.exists(xgb_path):
        model_xgb = joblib.load(xgb_path)
        print("✅ Loaded XGBoost Model")
    if os.path.exists(nn_path):
        model_nn = joblib.load(nn_path)
        print("✅ Loaded Neural Network Model")
    if os.path.exists(ensemble_path):
        model_ensemble = joblib.load(ensemble_path)
        print("✅ Loaded Ensemble Model")
    
    if model_xgb or model_nn or model_ensemble:
        models_loaded = True
except Exception as e:
    print(f"ℹ️ Additional ML models not found: {e}")

print("="*60)
print("ALL DATASETS LOADED")
print("="*60)

# ============================================================
# TRANSLATIONS - English to isiZulu (Full Dictionary)
# ============================================================

translations = {
    'zulu': {
        # Navigation
        'Dashboard': 'I-Dashboard',
        'Profile': 'I-Profile',
        'Book Appointment': 'Bhuka Ukubonisana',
        'Health & Symptoms': 'Impilo Nezimpawu',
        'Find Clinics': 'Thola Imitholampilo',
        'My Statistics': 'Izibalo Zami',
        'No-Show History': 'Umlando Wokungaveli',
        'Manual Booking': 'Ukubhuka Ngesandla',
        'Manage Slots': 'Phatha Izikhathi',
        'Export Data': 'Khipha Idatha',
        'Search Patients': 'Sesha Iziguli',
        'User Management': 'Ukuphathwa Kwabasebenzisi',
        'Staff Management': 'Ukuphathwa Kwabasebenzi',
        'Clinic Management': 'Ukuphathwa Kwemitholampilo',
        'All Appointments': 'Zonke Izibonisano',
        'Reports': 'Imibiko',
        'Activity Log': 'Umlando Wemisebenzi',
        'Settings': 'Izilungiselelo',
        'Terms Management': 'Ukuphathwa Kwemigomo',
        'Login': 'Ngena',
        'Sign Up': 'Bhalisa',
        'Logout': 'Phuma',
        'Welcome back': 'Siyakwamukela',
        
        # User Fields
        'Email': 'I-imeyili',
        'Password': 'Iphasiwedi',
        'Full Name': 'Igama Eliphelele',
        'Phone': 'Ucingo',
        'Role': 'Indima',
        'Staff Code': 'Ikhodi Yabasebenzi',
        'Age': 'Iminyaka',
        'Location': 'Indawo',
        'Health Conditions': 'Izimo Zempilo',
        'Language': 'Ulimi',
        'Conditions': 'Izimo',
        'Visits': 'Ukuvakasha',
        'Gender': 'Ubulili',
        'ID Number': 'Inombolo Kamazisi',
        'South African ID Number': 'Inombolo Kamazisi YaseNingizimu Afrika',
        'Select Gender': 'Khetha Ubulili',
        'Male': 'Owesilisa',
        'Female': 'Owesifazane',
        'Other': 'Okunye',
        'Prefer not to say': 'Ngikhetha Ukungasho',
        
        # Roles
        'Patient': 'Isiguli',
        'Nurse': 'Umhlengikazi',
        'Staff': 'Umsebenzi',
        'Admin': 'Umlawuli',
        
        # Appointment
        'Appointment': 'Ukubonisana',
        'Date': 'Usuku',
        'Time': 'Isikhathi',
        'Status': 'Isimo',
        'Reason': 'Isizathu',
        'Clinic': 'Umtholampilo',
        'Scheduled': 'Kuhleliwe',
        'Completed': 'Kuphothuliwe',
        'Cancelled': 'Kukhanseliwe',
        'No-Show': 'Akavelanga',
        'Checked-in': 'Ungene',
        'Waiting': 'Uyalinda',
        'Today': 'Namuhla',
        'Tomorrow': 'Kusasa',
        'Available': 'Iyatholakala',
        'Unavailable': 'Ayitholakali',
        
        # Risk Levels
        'Risk': 'Ingozi',
        'Low': 'Ephansi',
        'Medium': 'Emaphakathi',
        'High': 'Ephezulu',
        'Very High': 'Ephezulu Kakhulu',
        'Emergency': 'Isimo esiphuthumayo',
        'Urgent': 'Ngokushesha',
        'Routine': 'Okujwayelekile',
        
        # Common Actions
        'Submit': 'Thumela',
        'Cancel': 'Khansela',
        'Save': 'Gcina',
        'Update': 'Buyekeza',
        'Delete': 'Susa',
        'Edit': 'Hlela',
        'View': 'Buka',
        'Search': 'Sesha',
        'Book': 'Bhuka',
        'Check-in': 'Ngena',
        'Check-out': 'Phuma',
        'Send': 'Thumela',
        'Accept': 'Yamukela',
        'Decline': 'Nqaba',
        'Reset': 'Setha Kabusha',
        'Confirm': 'Qinisekisa',
        'Create': 'Dala',
        'Remove': 'Susa',
        'Add': 'Faka',
        'Close': 'Vala',
        'Open': 'Vula',
        'Load More': 'Layisha Okunye',
        'Showing': 'Kuboniswa',
        'of': 'kwe',
        'results': 'imiphumela',
        'Did you mean': 'Ngabe usho',
        'Read More': 'Funda Kabanzi',
        'Show Less': 'Bonisa Okuncane',
        'Q&A Only': 'Imibuzo Nezimpendulo Kuphela',
        'All Results': 'Yonke Imiphumela',
        'more': 'okunye',
        'No more results': 'Ayikho eminye imiphumela',
        
        # Messages
        'Welcome': 'Siyakwamukela',
        'Success': 'Kuphumelele',
        'Error': 'Iphutha',
        'Warning': 'Isixwayiso',
        'Info': 'Ulwazi',
        'Please wait': 'Sicela ulinde',
        'Loading': 'Kulayishwa',
        'No results found': 'Akukho okutholakele',
        'Please try again': 'Sicela uzame futhi',
        
        # Health
        'Symptoms': 'Izimpawu',
        'Disease': 'Isifo',
        'Treatment': 'Ukwelashwa',
        'Precautions': 'Izinyathelo zokuphepha',
        'Description': 'Incazelo',
        'Doctor': 'Udokotela',
        'Medicine': 'Umuthi',
        'Hospital': 'Isibhedlela',
        'First Aid': 'Usizo Lokuqala',
        'Health Tips': 'Amacebiso Ezempilo',
        'Diseases': 'Izifo',
        'Q&A': 'Imibuzo Nezimpendulo',
        'Related Symptoms': 'Izimpawu Ezihlobene',
        'Treatments': 'Izindlela Zokwelapha',
        'Recommended Doctors': 'Odokotela Abanconyiwe',
        'Risk Categories': 'Izigaba Zengozi',
        'Found information for': 'Ulwazi olutholakele nge',
        'Try searching for': 'Zama ukusesha',
        'Start your search': 'Qala ukusesha kwakho',
        'Search for diseases, symptoms, or health information': 'Sesha izifo, izimpawu, noma ulwazi lwezempilo',
        'Search diseases, symptoms, or conditions...': 'Sesha izifo, izimpawu, noma izimo...',
        'No exact match found': 'Akukho okufana ncamashi okutholakele',
        'Click to search for': 'Chofoza ukuze useshe',
        
        # Staff Related
        "Today's Appointments": 'Izibonisano Zanamuhla',
        'Patient Name': 'Igama Lesiguli',
        'Patient Phone': 'Ucingo Lwesiguli',
        'No phone': 'Akukho ucingo',
        'Walk-in': 'Ungene Ngaphandle Kokubhuka',
        'Staff Dashboard': 'Ideshibhodi Yabasebenzi',
        'Clinic Name': 'Igama Lomtholampilo',
        'Clinic Address': 'Ikheli Lomtholampilo',
        'Select Clinic': 'Khetha Umtholampilo',
        
        # Admin Related
        'Total Users': 'Ingqikithi Yabasebenzisi',
        'Total Staff': 'Ingqikithi Yabasebenzi',
        'Active Users': 'Abasebenzisi Abasebenzayo',
        'Recent Users': 'Abasebenzisi Bakamuva',
        'User Role': 'Indima Yomsebenzisi',
        'Verify Staff': 'Qinisekisa Abasebenzi',
        'Add Clinic': 'Faka Umtholampilo',
        'Edit Clinic': 'Hlela Umtholampilo',
        'Delete Clinic': 'Susa Umtholampilo',
        'System Settings': 'Izilungiselelo Zesistimu',
        
        # Search Enhancement
        'Showing results for': 'Kuboniswa imiphumela ye',
        'Load more results': 'Layisha imiphumela eyengeziwe',
        'showing': 'kuboniswa',
        
        # Footer
        'All rights reserved': 'Wonke amalungelo agodliwe',
        'Empowering healthcare through technology': 'Ukuthuthukisa ezempilo ngobuchwepheshe',
        
        # ===== FIRST AID TRANSLATIONS =====
        'First Aid Guide': 'Umhlahlandlela Wosizo Lokuqala',
        'Quick reference for medical emergencies': 'Inkomba esheshayo yezimo eziphuthumayo zezempilo',
        'Search first aid...': 'Sesha usizo lokuqala...',
        'View Details': 'Buka Imininingwane',
        'What to do:': 'Okumele ukwenze:',
        'No first aid results found': 'Ayikho imiphumela yosizo lokuqala etholakele',
        'Try adjusting your search terms': 'Zama ukulungisa amagama akho okusesha',
        'First Aid Categories': 'Izigaba Zosizo Lokuqala',
        'All': 'Konke',
        
        # ===== NEW FIRST AID TERMS =====
        'Heart Attack': 'Ukuhlaselwa Inhliziyo',
        'Stroke': 'Ukushaywa Umgogodla',
        'Choking': 'Ukuminywa',
        'Severe Bleeding': 'Ukopha Kakhulu',
        'Allergic Reaction': 'Ukungezwani Komzimba',
        'Seizure / Fits': 'Isithuthwane / Ukuqhwagwa',
        'Poisoning': 'Ubuthi',
        'Drowning': 'Ukuminza',
        'Burns & Scalds': 'Ukusha Nokushiswa',
        'Fractures & Sprains': 'Ukuphuka Nokukhubazeka',
        'Heat Stroke & Dehydration': 'Ukushisa Ngokweqile Nokuphelelwa Amanzi',
        'Hypothermia': 'Ukubanda Ngokweqile',
        'Diabetic Emergency': 'Isimo Esiphuthumayo Sikashukela',
        'Asthma Attack': 'Ukuhlaselwa Isifuba',
        'Cuts & Scrapes': 'Ukusikeka Nokuklwebheka',
        'Insect Bites & Stings': 'Ukulunywa Izikelemu Nezinambuzane',
        'Nosebleeds': 'Ukopha Emakhaleni',
        'Headache & Migraine': 'Ikhanda Elibuhlungu / I-Migraine',
        'Food Poisoning': 'Ubuthi Bokudla',
        'Fever Management': 'Ukuphatha Umkhuhlane',
        
        # ===== EMERGENCY DESCRIPTIONS =====
        'Emergency signs & response': 'Izimpawu eziphuthumayo nendlela yokusabela',
        'FAST - Face, Arms, Speech, Time': 'FAST - Ubuso, Izingalo, Inkulumo, Isikhathi',
        'Heimlich maneuver for adults & children': 'Indlela ye-Heimlich kubantu abadala nezingane',
        'How to stop heavy bleeding': 'Indlela yokumisa ukopha okukhulu',
        'Anaphylaxis emergency response': 'Indlela yokusabela ekungezwani komzimba okukhulu',
        'What to do during a seizure': 'Okumele ukwenze ngesikhathi somuntu equphumayo',
        'Immediate steps for poisoning': 'Izinyathelo zokuqala zobuthi',
        'Rescue & CPR for drowning': 'Ukutakula nokwenza i-CPR komunye umuntu',
        
        # ===== URGENT DESCRIPTIONS =====
        'First aid for burns': 'Usizo lokuqala lokusha',
        'Handle broken bones & sprains': 'Ukuphatha amathambo aphukile nokukhubazeka',
        'Emergency cooling response': 'Indlela yokupholisa ngokushesha',
        'Warming a person safely': 'Ukufudumeza umuntu ngokuphepha',
        'Low/high blood sugar response': 'Indlela yokusabela kushukela ophansi/nophezulu',
        'Help someone having an asthma attack': 'Siza umuntu ohlaselwa isifuba',
        
        # ===== ROUTINE DESCRIPTIONS =====
        'Clean and dress minor wounds': 'Hlanza futhi ufake izinxibo emanxebeni amancane',
        'Treatment for bites and stings': 'Ukwelapha ukulunywa nokugazwa',
        'How to stop a nosebleed': 'Indlela yokumisa ukopha emakhaleni',
        'Relief for headaches': 'Ukusiza ikhanda elibuhlungu',
        'Symptoms and home care': 'Izimpawu nokunakekela ekhaya',
        'How to manage a fever': 'Indlela yokuphatha umkhuhlane',
        
        # ===== FIRST AID STEPS =====
        'Call emergency services immediately (10177)': 'Shayela abezimo eziphuthumayo ngokushesha (10177)',
        'Make the person sit down and rest': 'Yenza umuntu ahlale phansi futhi aphumule',
        'Loosen tight clothing': 'Khulula izingubo eziqinile',
        'Give aspirin if available and not allergic': 'Nika i-aspirin uma ikhona futhi engenawo umlenze',
        'Stay with them until help arrives': 'Hlala nabo kuze kufike usizo',
        'Check FAST: Face drooping, Arm weakness, Speech difficulty': 'Hlola FAST: Ubuso buyajika, Izingalo zibuthakathaka, Inkulumo inzima',
        'Note the time symptoms started': 'Bhala isikhathi izimpawu eziqale ngaso',
        'Keep the person calm and comfortable': 'Gcina umuntu ezolile futhi ekhululekile',
        'Do not give food or drink': 'Ungabaniki ukudla noma iziphuzo',
        'Ask if they can speak or cough': 'Buza ukuthi bayakhuluma noma bayakhwehlela',
        'If not, perform Heimlich maneuver': 'Uma kungenjalo, yenza indlela ye-Heimlich',
        'Stand behind them, make a fist above navel': 'Yima ngemuva kwabo, yenza isibhakela ngaphezu kwenkaba',
        'Thrust inward and upward': 'Faka ngaphakathi nangaphezulu',
        'If unconscious, start CPR': 'Uma equleka, qala i-CPR',
        'Apply direct pressure to the wound': 'Faka ingcindezi eqondile enxebeni',
        'Use a clean cloth or bandage': 'Sebenzisa indwangu ehlanzekile noma ibhandeshi',
        'Elevate the wound above the heart if possible': 'Phakamisa inxeba ngaphezu kwenhliziyo uma kungenzeka',
        'Keep pressure until bleeding stops': 'Gcina ingcindezi kuze kuyeke ukopha',
        'Seek medical help immediately': 'Funa usizo lwezokwelapha ngokushesha',
        'Give epinephrine auto-injector if available': 'Nika umjovo we-epinephrine uma ukhona',
        'Keep the person lying down with feet elevated': 'Gcina umuntu elele phansi nezinyawo ziphakanyisiwe',
        'Monitor breathing and pulse': 'Qaphela ukuphefumula nokushaya kwenhliziyo',
        'Clear the area of dangerous objects': 'Susa izinto eziyingozi endaweni',
        'Cushion the head with something soft': 'Beka ikhanda entweni ethambile',
        'Do NOT restrain the person': 'UNGAWAHLI umuntu',
        'Do NOT put anything in their mouth': 'UNGAFAKI lutho emlonyeni wabo',
        'Time the seizure, call 10177 if over 5 minutes': 'Bala isikhathi sokuqhwagwa, shayela u-10177 uma kungaphezu kwemizuzu emi-5',
        'Call Poison Control (0861 555 777)': 'Shayela u-Poison Control (0861 555 777)',
        'Do NOT induce vomiting unless instructed': 'UNGAZENZI ukuhlanza ngaphandle uma utshelwe',
        'Check for signs of poisoning': 'Hlola izimpawu zobuthi',
        'Bring the poison container with you to hospital': 'Letha isitsha sobuthi esibhedlela',
        'Call for help immediately': 'Shayela usizo ngokushesha',
        'If trained, start CPR': 'Uma uqeqeshiwe, qala i-CPR',
        'Check for breathing and pulse': 'Hlola ukuphefumula nokushaya kwenhliziyo',
        'Begin rescue breathing': 'Qala ukuphefumula ngosizo',
        'Continue CPR until help arrives': 'Qhubeka nge-CPR kuze kufike usizo',
        'Cool the burn with running cool water for 10-15 minutes': 'Pholisa ukusha ngamanzi apholile imizuzu eyi-10-15',
        'Remove tight items before swelling starts': 'Susa izinto eziqinile ngaphambi kokuvuvukala',
        'Cover with a sterile dressing': 'Mboza ngendwangu ehlanzekile',
        'Do NOT break blisters': 'UNGAZIQEBI amabhamuza',
        'Seek medical help for severe burns': 'Funa usizo lwezokwelapha ngokusha okukhulu',
        'Do NOT move the person unless necessary': 'UNGAHAMBI umuntu ngaphandle uma kudingeka',
        'Immobilize the injured area': 'Yenza indawo elimele inganyakazi',
        'Apply cold pack to reduce swelling': 'Faka ipaki ebandayo ukunciphisa ukufutha',
        'Elevate if possible': 'Phakamisa uma kungenzeka',
        'Move to a cool place out of the sun': 'Thuthela endaweni epholile ngaphandle kwelanga',
        'Remove excess clothing': 'Susa izingubo eziningi',
        'Apply cool water to skin': 'Faka amanzi apholile esikhumbeni',
        'Fan the person': 'Futhisa umuntu',
        'If conscious, give water to drink': 'Uma eqaphile, mnike amanzi okudla',
        'Move to a warm place': 'Thuthela endaweni efudumele',
        'Remove wet clothing': 'Susa izingubo ezimanzi',
        'Wrap in blankets or dry clothing': 'Goqoza ngokhukho noma izingubo ezomile',
        'Give warm drinks if conscious': 'Nika iziphuzo ezifudumele uma eqaphile',
        'Check if they have diabetes supplies': 'Hlola ukuthi banemithi yesifo sikashukela',
        'If conscious, give sugar/sweet drink': 'Uma eqaphile, mnika ushukela/isiphuzo esimnandi',
        'If unconscious, do NOT give anything by mouth': 'Uma equlekile, UNGAMNIKI lutho ngomlomo',
        'Help them sit upright': 'Bamsiza ukuthi ahlale aqonde',
        'Use their inhaler if available': 'Sebenzisa isiphefumuli sabo uma sikhona',
        'If no improvement, call 10177': 'Uma kungekho phucuko, shayela u-10177',
        'Stay calm and reassure them': 'Hlala uzolile futhi ubavume',
        'Wash hands before treating the wound': 'Hlanza izandla ngaphambi kokwelapha inxeba',
        'Clean with soap and water': 'Hlanza ngensipho namanzi',
        'Apply pressure to stop bleeding': 'Faka ingcindezi ukumisa ukopha',
        'Apply antibiotic ointment': 'Faka i-antibiotic ointment',
        'Cover with sterile bandage': 'Mboza ngendwangu ehlanzekile',
        'Remove the stinger if present': 'Susa isihlobo uma sikhona',
        'Wash the area with soap and water': 'Hlanza indawo ngensipho namanzi',
        'Apply ice pack to reduce swelling': 'Faka ipaki yeqhwa ukunciphisa ukufutha',
        'Use antihistamine if needed': 'Sebenzisa i-antihistamine uma kudingeka',
        'Watch for allergic reaction': 'Qaphela ukungezwani komzimba',
        'Sit forward, not backward': 'Hlala ubheke phambili, hhayi emuva',
        'Pinch the soft part of the nose': 'Bamba ingxenye ethambile yekhala',
        'Apply cold compress to nose bridge': 'Faka compress ebandayo esiqondeni sekhala',
        'Breathe through the mouth': 'Phefumula ngomlomo',
        'Seek help if bleeding persists': 'Funa usizo uma ukopha kuqhubeka',
        'Rest in a quiet, dark room': 'Phumula ekamelweni elithule, elimnyama',
        'Apply cold or warm compress': 'Faka i-compress ebandayo noma efudumele',
        'Drink water': 'Phuza amanzi',
        'Use pain relievers if needed': 'Sebenzisa imithi yobuhlungu uma kudingeka',
        'Avoid triggers like bright lights': 'Gwema izinto ezivusa amalangabi njengezibani ezikhanyayo',
        'Drink plenty of fluids': 'Phuza uketshezi oluningi',
        'Rest and avoid solid foods initially': 'Phumula futhi ugweme ukudla okuqinile ekuqaleni',
        'Gradually return to eating bland foods': 'Kancane kancane buyela ekudleni ukudla okungena saladi',
        'Monitor for severe symptoms': 'Qaphela izimpawu ezinzima',
        'Seek medical help if severe': 'Funa usizo lwezokwelapha uma kunzima',
        'Rest and stay hydrated': 'Phumula futhi uphuze amanzi amaningi',
        'Take paracetamol for fever': 'Thatha i-paracetamol ngomkhuhlane',
        'Use lukewarm sponging': 'Sebenzisa isiponji esifudumele',
        'Remove excess clothing': 'Susa izingubo eziningi',
        'Seek help if fever persists': 'Funa usizo uma umkhuhlane uqhubeka',
        
        # ===== ADMIN DASHBOARD TRANSLATIONS =====
        'Online': 'Ku-inthanethi',
        'System': 'Isistimu',
        'Patients': 'Iziguli',
        'Staff': 'Abasebenzi',
        'Appointments by Day': 'Izibonisano Ngosuku',
        'Appointment Status': 'Isimo Sokubonisana',
        'Quick Actions': 'Izenzo Ezisheshayo',
        'Manage Users': 'Phatha Abasebenzisi',
        'Manage Clinics': 'Phatha Imitholampilo',
        'All Appointments': 'Zonke Izibonisano',
        'Staff Management': 'Ukuphathwa Kwabasebenzi',
        'Reports': 'Imibiko',
        'System Settings': 'Izilungiselelo Zesistimu',
        'Terms Management': 'Ukuphathwa Kwemigomo',
        'Activity Log': 'Umlando Wemisebenzi',
        'Check No-Shows': 'Hlola Abangavelanga',
        'System Information': 'Ulwazi Lwesistimu',
        'Appointments': 'Izibonisano',
        'Clinics': 'Imitholampilo',
        'Name': 'Igama',
        'Getting directions...': 'Ukuthola iziqondiso...',
        'Open in Google Maps': 'Vula ku-Google Maps',
        'Clinic Location': 'Indawo Yomtholampilo',
        'Get Directions': 'Thola Iziqondiso'
    }
}

translations['en'] = {}

def translate_text(text, lang):
    if lang == 'en' or lang not in translations:
        return text
    trans = translations.get(lang, {})
    return trans.get(text, text)

# ============================================================
# FIX #1: SA ID VALIDATION
# ============================================================
def validate_sa_id(id_number):
    """Return (is_valid, message). Validates format, date, and Luhn checksum."""
    if not id_number:
        return False, "ID number is required."
    id_number = str(id_number).strip()
    if not id_number.isdigit():
        return False, "ID number must contain only digits."
    if len(id_number) != 13:
        return False, f"ID number must be exactly 13 digits (you entered {len(id_number)})."
    year = int(id_number[0:2])
    month = int(id_number[2:4])
    day = int(id_number[4:6])
    if month < 1 or month > 12:
        return False, f"Invalid month '{month:02d}'. Month must be 01-12."
    if day < 1 or day > 31:
        return False, f"Invalid day '{day:02d}'. Day must be 01-31."
    is_leap = (year % 4 == 0 and (year % 100 != 0 or year % 400 == 0))
    days_in_month = [31, 29 if is_leap else 28, 31, 30, 31, 30,
                     31, 31, 30, 31, 30, 31]
    if day > days_in_month[month - 1]:
        return False, f"Invalid day '{day:02d}' for month {month:02d}."
    total = 0
    for i, ch in enumerate(id_number):
        d = int(ch)
        if i % 2 == 1:
            d *= 2
            if d > 9:
                d -= 9
        total += d
    if total % 10 != 0:
        return False, "Invalid ID number (checksum failed). Please check the digits."
    return True, "Valid"

def extract_dob_from_sa_id(id_number):
    """Return (year, month, day) from a valid SA ID, or (None, None, None)."""
    if not id_number or len(str(id_number)) != 13:
        return None, None, None
    yy = int(str(id_number)[0:2])
    mm = int(str(id_number)[2:4])
    dd = int(str(id_number)[4:6])
    current_yy = datetime.now().year % 100
    year = 2000 + yy if yy <= current_yy else 1900 + yy
    return year, mm, dd
# ============================================================
# END FIX #1
# ============================================================

# ============================================================
# FIX #8: SPELL CORRECTION HELPERS
# ============================================================
_TERM_CACHE = None

def _build_term_cache():
    global _TERM_CACHE
    if _TERM_CACHE is not None:
        return _TERM_CACHE
    terms = set()
    try:
        if df_master is not None and 'disease' in df_master.columns:
            for t in df_master['disease'].dropna().astype(str):
                terms.add(t.strip().lower())
        if df_symptom_descriptions is not None and 'symptom' in df_symptom_descriptions.columns:
            for t in df_symptom_descriptions['symptom'].dropna().astype(str):
                terms.add(t.strip().lower())
        if df_symptom_precautions is not None and 'condition' in df_symptom_precautions.columns:
            for t in df_symptom_precautions['condition'].dropna().astype(str):
                terms.add(t.strip().lower())
        if df_disease_to_medicine is not None and 'disease' in df_disease_to_medicine.columns:
            for t in df_disease_to_medicine['disease'].dropna().astype(str):
                terms.add(t.strip().lower())
        if df_medicine_to_disease is not None and 'medicine' in df_medicine_to_disease.columns:
            for t in df_medicine_to_disease['medicine'].dropna().astype(str):
                terms.add(t.strip().lower())
        if df_clinics is not None:
            for col in ['Clinic_Name', 'City', 'Area', 'Province', 'District']:
                if col in df_clinics.columns:
                    for t in df_clinics[col].dropna().astype(str):
                        terms.add(t.strip().lower())
    except Exception as e:
        print(f"[spell] cache error: {e}")
    _TERM_CACHE = list(terms)
    return _TERM_CACHE

def correct_spelling(query, n=1, cutoff=0.72):
    """Return the best spelling suggestion for `query`, or None."""
    if not query or len(query.strip()) < 3:
        return None
    terms = _build_term_cache()
    if not terms:
        return None
    q = query.strip().lower()
    if q in terms:
        return None
    matches = get_close_matches(q, terms, n=n, cutoff=cutoff)
    if not matches:
        words = q.split()
        corrected_words = []
        changed = False
        for w in words:
            if len(w) < 3:
                corrected_words.append(w)
                continue
            m = get_close_matches(w, terms, n=1, cutoff=cutoff)
            if m:
                corrected_words.append(m[0])
                changed = True
            else:
                corrected_words.append(w)
        if changed:
            return ' '.join(corrected_words)
        return None
    return matches[0]
# ============================================================
# END FIX #8
# ============================================================

# ============================================================
# FIX #9: HEALTH SCORE (starts at 0)
# ============================================================
def calculate_health_score(age, health_conditions, appointments,
                           profile_complete=False, has_id=False,
                           has_emergency=False, has_allergies=False):
    """Health score starts at 0. Grows only from real user data."""
    score = 0
    if age and 1 <= age <= 120:
        score += 5
    if profile_complete:
        score += 5
    if has_id:
        score += 5
    if has_emergency:
        score += 5
    if health_conditions is not None and health_conditions != '':
        score += 5
    if has_allergies is not None and has_allergies != '':
        score += 5
    total = len(appointments)
    completed = sum(1 for a in appointments if a['status'] == 'Completed')
    score += min(completed * 5, 40)
    checked_in = sum(1 for a in appointments if a['status'] == 'Checked-in')
    score += min(checked_in * 3, 15)
    no_shows = sum(1 for a in appointments if a['status'] == 'No-Show')
    score -= min(no_shows * 10, 30)
    cancelled = sum(1 for a in appointments if a['status'] == 'Cancelled')
    score -= min(cancelled * 2, 10)
    if total >= 3: score += 5
    if total >= 5: score += 5
    if total >= 10: score += 5
    score = max(0, min(score, 100))
    return score

def get_health_score_category(score):
    if score >= 80: return 'Excellent'
    if score >= 60: return 'Good'
    if score >= 40: return 'Fair'
    if score >= 20: return 'Needs Attention'
    return 'New'
# ============================================================
# END FIX #9
# ============================================================

# ============================================================
# DATABASE INIT
# ============================================================

def default_terms_content():
    return """
    <h5>1. Acceptance of Terms</h5>
    <p>By using MediSense, you agree to comply with these Terms and Conditions.</p>
    
    <h5>2. User Accounts</h5>
    <p>You must provide accurate information when creating an account. You are responsible for maintaining the confidentiality of your password.</p>
    
    <h5>3. Medical Disclaimer</h5>
    <p><strong>IMPORTANT:</strong> MediSense provides health information and appointment management only. It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare provider for medical concerns.</p>
    
    <h5>4. Appointment Booking</h5>
    <p>You agree to arrive on time for appointments. Repeated no-shows may result in restricted access to booking services.</p>
    
    <h5>5. Privacy Policy</h5>
    <p>Your personal and health information will be kept confidential and used only for healthcare services and appointment management.</p>
    
    <h5>6. SMS Notifications</h5>
    <p>By providing your phone number, you consent to receive SMS reminders about appointments.</p>
    
    <h5>7. User Conduct</h5>
    <p>You agree to use MediSense for legitimate healthcare purposes only. Any misuse or abuse will result in account termination.</p>
    
    <h5>8. Liability</h5>
    <p>MediSense is not liable for any damages arising from the use of this platform. All healthcare decisions should be made with professional guidance.</p>
    
    <h5>9. Changes to Terms</h5>
    <p>MediSense reserves the right to update these terms. Continued use constitutes acceptance of updated terms.</p>
    
    <h5>10. Contact</h5>
    <p>For questions, contact us at: support@medisense.com</p>
    """

def init_database():
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    cursor = conn.cursor()
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT NOT NULL,
        phone TEXT,
        role TEXT DEFAULT 'patient',
        staff_code TEXT,
        is_verified INTEGER DEFAULT 0,
        is_active INTEGER DEFAULT 1,
        age INTEGER DEFAULT 30,
        health_conditions TEXT,
        location TEXT,
        staff_clinic TEXT,
        language TEXT DEFAULT 'en',
        reset_token TEXT,
        reset_expires TIMESTAMP,
        terms_accepted INTEGER DEFAULT 0,
        terms_accepted_at TIMESTAMP,
        terms_version TEXT DEFAULT '1.0',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP,
        emergency_name TEXT,
        emergency_phone TEXT,
        emergency_relationship TEXT,
        allergies TEXT,
        gender TEXT,
        id_number TEXT
    )
    ''')
    
    cursor.execute("PRAGMA table_info(users)")
    columns = [col[1] for col in cursor.fetchall()]
    for col in ['staff_code', 'is_verified', 'age', 'health_conditions', 'location', 'staff_clinic', 'language', 'reset_token', 'reset_expires', 'terms_accepted', 'terms_accepted_at', 'terms_version', 'emergency_name', 'emergency_phone', 'emergency_relationship', 'allergies', 'gender', 'id_number']:
        if col not in columns:
            cursor.execute(f"ALTER TABLE users ADD COLUMN {col} TEXT DEFAULT ''")
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_email TEXT,
        patient_name TEXT,
        patient_phone TEXT,
        clinic_name TEXT,
        appointment_date TEXT,
        appointment_time TEXT,
        status TEXT DEFAULT 'Scheduled',
        reason TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS unavailable_slots (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        clinic_name TEXT NOT NULL,
        appointment_date TEXT NOT NULL,
        appointment_time TEXT NOT NULL,
        reason TEXT,
        created_by INTEGER,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(clinic_name, appointment_date, appointment_time)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS symptom_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        search_term TEXT,
        result TEXT,
        urgency_level TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS health_journal (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        symptom TEXT,
        severity INTEGER DEFAULT 1,
        notes TEXT,
        recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')
    
    # ===== FIX #7: Chatbot history table =====
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS chatbot_history (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id INTEGER,
        question TEXT,
        answer TEXT,
        source TEXT,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        FOREIGN KEY (user_id) REFERENCES users(id)
    )
    ''')
    # ===== END FIX #7 =====
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS terms_versions (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        version TEXT NOT NULL,
        content TEXT NOT NULL,
        effective_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        created_by INTEGER,
        is_active INTEGER DEFAULT 1
    )
    ''')
    
    cursor.execute("SELECT COUNT(*) FROM terms_versions")
    if cursor.fetchone()[0] == 0:
        cursor.execute('''
        INSERT INTO terms_versions (version, content, created_by, is_active)
        VALUES (?, ?, ?, ?)
        ''', ('1.0', default_terms_content(), 1, 1))
    
    cursor.execute("SELECT COUNT(*) FROM users WHERE email = 'admin@medisense.com'")
    if cursor.fetchone()[0] == 0:
        password_hash = hashlib.sha256("Admin123!".encode()).hexdigest()
        cursor.execute('''
        INSERT INTO users (email, password_hash, full_name, phone, role, staff_code, is_verified, is_active, terms_accepted)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', ('admin@medisense.com', password_hash, 'System Administrator', '0821234567', 'admin', 'ADMIN2026', 1, 1, 1))
    
    conn.commit()
    conn.close()
    print("Database initialized")

init_database()

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def get_clinics_list():
    if df_clinics is not None and 'Clinic_Name' in df_clinics.columns:
        return df_clinics['Clinic_Name'].unique().tolist()
    return ['Empangeni Clinic', 'Ngwelezane Clinic', 'Richards Bay Clinic']

def calculate_risk_score(age, health_conditions, appointments):
    health_count = len([c for c in health_conditions.split(',') if c.strip()]) if health_conditions else 0
    no_show_history = sum(1 for a in appointments if a['status'] == 'No-Show')
    risk_score = 15
    if age < 30: risk_score += 10
    elif age > 60: risk_score += 5
    if health_count > 1: risk_score += 15
    elif health_count > 0: risk_score += 10
    if no_show_history > 0: risk_score += 20 * min(no_show_history, 3)
    if health_count >= 3: risk_score += 10
    return min(risk_score, 95)

def get_risk_category(risk_score):
    if risk_score < 30: return 'Low'
    elif risk_score < 50: return 'Medium'
    elif risk_score < 70: return 'High'
    else: return 'Very High'

def predict_no_show(user_email):
    """Predict if patient will no-show based on history using ML"""
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT 
        COUNT(*) as total_appointments,
        SUM(CASE WHEN status = 'No-Show' THEN 1 ELSE 0 END) as no_shows,
        SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed,
        SUM(CASE WHEN status = 'Cancelled' THEN 1 ELSE 0 END) as cancelled
    FROM appointments 
    WHERE patient_email = ?
    ''', (user_email,))
    
    data = cursor.fetchone()
    conn.close()
    
    if not data or data['total_appointments'] == 0:
        return {'risk': 0, 'category': 'Low', 'confidence': 0, 'predictions': []}
    
    no_show_rate = data['no_shows'] / data['total_appointments'] if data['total_appointments'] > 0 else 0
    completion_rate = data['completed'] / data['total_appointments'] if data['total_appointments'] > 0 else 0
    cancellation_rate = data['cancelled'] / data['total_appointments'] if data['total_appointments'] > 0 else 0
    
    if models_loaded and model_ensemble is not None:
        try:
            features = np.array([[no_show_rate, completion_rate, cancellation_rate, data['total_appointments']]])
            prediction = model_ensemble.predict_proba(features)[0][1]
        except:
            prediction = no_show_rate
    else:
        prediction = no_show_rate
    
    confidence = min(0.9, 0.5 + (data['total_appointments'] / 20))
    
    if prediction < 0.2:
        category = 'Low'
    elif prediction < 0.4:
        category = 'Medium'
    elif prediction < 0.6:
        category = 'High'
    else:
        category = 'Very High'
    
    predictions = []
    if model_loaded:
        predictions.append({'model': 'RandomForest', 'score': round(no_show_rate * 100, 1)})
    if model_xgb is not None:
        try:
            xgb_pred = model_xgb.predict_proba([[no_show_rate, completion_rate, cancellation_rate, data['total_appointments']]])[0][1]
            predictions.append({'model': 'XGBoost', 'score': round(xgb_pred * 100, 1)})
        except:
            pass
    if model_nn is not None:
        try:
            nn_pred = model_nn.predict_proba([[no_show_rate, completion_rate, cancellation_rate, data['total_appointments']]])[0][1]
            predictions.append({'model': 'Neural Network', 'score': round(nn_pred * 100, 1)})
        except:
            pass
    
    return {
        'risk': round(prediction * 100, 1),
        'category': category,
        'confidence': round(confidence * 100, 1),
        'predictions': predictions,
        'total_appointments': data['total_appointments'],
        'no_shows': data['no_shows'],
        'completed': data['completed'],
        'cancelled': data['cancelled']
    }

def get_nearest_clinic(location):
    if df_clinics is not None and location:
        try:
            location_lower = location.lower()
            matching = df_clinics[
                df_clinics['City'].str.lower().str.contains(location_lower, na=False) |
                df_clinics['Area'].str.lower().str.contains(location_lower, na=False)
            ]
            if not matching.empty:
                row = matching.iloc[0]
                return {
                    'name': row['Clinic_Name'], 
                    'phone': row['Phone'] if 'Phone' in row else 'N/A', 
                    'distance': '~1.2 km',
                    'lat': row.get('Latitude', None),
                    'lng': row.get('Longitude', None)
                }
        except:
            pass
    return {'name': 'Empangeni Clinic', 'phone': '035 123 4567', 'distance': 'Please update your location'}

def get_random_health_tip():
    if df_health_tips is not None and len(df_health_tips) > 0:
        try:
            tip = df_health_tips.sample(1).iloc[0]
            return tip['tip'] if 'tip' in tip else str(tip.iloc[0])
        except:
            pass
    return "Stay healthy! Regular check-ups and a balanced diet are important."

def get_available_times(clinic, date):
    all_times = []
    for hour in range(8, 17):
        for minute in ['00', '30']:
            all_times.append(f"{hour:02d}:{minute}")
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT appointment_time FROM appointments 
    WHERE clinic_name = ? AND appointment_date = ? AND status != 'Cancelled'
    ''', (clinic, date))
    booked = [row[0] for row in cursor.fetchall()]
    
    cursor.execute('''
    SELECT appointment_time FROM unavailable_slots 
    WHERE clinic_name = ? AND appointment_date = ?
    ''', (clinic, date))
    unavailable = [row[0] for row in cursor.fetchall()]
    
    conn.close()
    
    available = []
    for time in all_times:
        if time not in booked and time not in unavailable:
            available.append(time)
    
    return available

# ============================================================
# NLP SYMPTOM ANALYSIS
# ============================================================

def analyze_symptoms_nlp(text):
    """Extract symptoms using NLP - simple version without spacy"""
    if not text:
        return []
    
    symptom_keywords = [
        'pain', 'ache', 'cough', 'fever', 'headache', 'dizziness', 
        'nausea', 'vomiting', 'diarrhea', 'rash', 'itching', 
        'swelling', 'bleeding', 'fatigue', 'weakness', 'chest pain',
        'shortness of breath', 'difficulty breathing', 'sore throat',
        'runny nose', 'congestion', 'muscle ache', 'joint pain',
        'back pain', 'stomach ache', 'abdominal pain', 'cramps',
        'chills', 'sweating', 'loss of appetite', 'weight loss',
        'coughing', 'sneezing', 'allergies', 'asthma', 'wheezing'
    ]
    
    text_lower = text.lower()
    found = []
    
    for keyword in symptom_keywords:
        if keyword in text_lower:
            found.append(keyword)
    
    words = text_lower.split()
    for i in range(len(words) - 1):
        phrase = words[i] + ' ' + words[i+1]
        if phrase in ['chest pain', 'shortness breath', 'sore throat', 'runny nose', 'muscle ache', 'joint pain', 'back pain', 'stomach ache', 'abdominal pain', 'headache pain', 'cough fever']:
            if phrase not in found:
                found.append(phrase)
    
    return list(set(found))[:5]

# ============================================================
# AUTO NO-SHOW CHECKER
# ============================================================

def check_expired_appointments():
    now = datetime.now()
    today = now.strftime('%Y-%m-%d')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, patient_name, patient_email, appointment_time, clinic_name
    FROM appointments 
    WHERE appointment_date = ? 
    AND status = 'Scheduled'
    ''', (today,))
    
    all_appointments = cursor.fetchall()
    expired_count = 0
    
    for appt in all_appointments:
        appt_datetime = datetime.strptime(f"{today} {appt['appointment_time']}", '%Y-%m-%d %H:%M')
        minutes_passed = (now - appt_datetime).total_seconds() / 60
        
        if minutes_passed >= 45:
            cursor.execute('''
            UPDATE appointments 
            SET status = 'No-Show' 
            WHERE id = ?
            ''', (appt['id'],))
            conn.commit()
            expired_count += 1
            print(f"⏰ Auto No-Show: {appt['patient_name']} at {appt['clinic_name']} ({appt['appointment_time']}) - {minutes_passed:.0f} mins passed")
    
    conn.close()
    
    if expired_count > 0:
        print(f"📊 Total auto no-shows marked: {expired_count}")
    
    return expired_count

# ============================================================
# FIX #5: EMAIL HELPER — uses SMTP_USERNAME + SMTP_PASSWORD env vars
# ============================================================

def generate_reset_token():
    return secrets.token_urlsafe(32)

def send_reset_email(email, reset_token):
    try:
        # ===== FIX #5: read SMTP creds from env, don't hardcode =====
        sender_email = os.environ.get('SMTP_USERNAME', '').strip()
        sender_password = os.environ.get('SMTP_PASSWORD', '').strip()
        
        if not sender_email or not sender_password:
            print("[email] SMTP_USERNAME / SMTP_PASSWORD not set. Printing token instead.")
            print(f"[email] Reset token for {email}: {reset_token}")
            return False
        # ===== END FIX #5 =====
        
        reset_link = url_for('reset_password', token=reset_token, _external=True)
        
        subject = "MediSense - Password Reset"
        body = f"""
        Hello,

        You requested a password reset for your MediSense account.

        Click the link below to reset your password:
        {reset_link}

        This link will expire in 15 minutes.

        If you did not request this, please ignore this email.

        Regards,
        MediSense Team
        """
        
        msg = MIMEMultipart()
        msg['From'] = sender_email
        msg['To'] = email
        msg['Subject'] = subject
        msg.attach(MIMEText(body, 'plain'))
        
        with smtplib.SMTP('smtp.gmail.com', 587) as server:
            server.starttls()
            server.login(sender_email, sender_password)
            server.send_message(msg)
        
        print(f"✅ Reset email sent to {email}")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        return False

# ============================================================
# TERMS CHECK DECORATOR
# ============================================================

def require_terms_acceptance(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        exempt_routes = ['view_terms', 'accept_terms', 'logout', 'login', 'signup', 'forgot_password', 'reset_password', 'home', 'ussd',
                         # ===== FIX #3: exempt public routes =====
                         'public_home', 'public_symptoms', 'public_clinics', 'public_first_aid',
                         # ===== END FIX #3 =====
                        ]
        if request.endpoint in exempt_routes:
            return f(*args, **kwargs)
        
        if 'user_id' not in session:
            return f(*args, **kwargs)
        
        # ===== FIX #10: use DB_PATH =====
        conn = sqlite3.connect(DB_PATH)
        # ===== END FIX #10 =====
        cursor = conn.cursor()
        cursor.execute('SELECT terms_accepted FROM users WHERE id = ?', (session['user_id'],))
        result = cursor.fetchone()
        conn.close()
        
        if result and result[0] == 0:
            flash('Please accept the Terms and Conditions to continue.', 'warning')
            return redirect(url_for('view_terms'))
        
        return f(*args, **kwargs)
    return decorated_function

# ============================================================
# SEARCH FUNCTIONS - ENHANCED WITH NLP
# ============================================================

def get_disease_suggestions(search_term):
    if not search_term:
        return []
    
    search_lower = search_term.lower().strip()
    all_terms = []
    
    if df_master is not None and 'disease' in df_master.columns:
        disease_list = df_master['disease'].dropna().tolist()
        all_terms.extend(disease_list)
    
    if df_symptom_descriptions is not None and 'symptom' in df_symptom_descriptions.columns:
        symptom_list = df_symptom_descriptions['symptom'].dropna().tolist()
        all_terms.extend(symptom_list)
    
    if df_symptom_precautions is not None and 'condition' in df_symptom_precautions.columns:
        condition_list = df_symptom_precautions['condition'].dropna().tolist()
        all_terms.extend(condition_list)
    
    all_terms = list(set([term for term in all_terms if term and str(term).strip()]))
    
    suggestions = []
    search_words = search_lower.split()
    
    for term in all_terms:
        term_lower = str(term).lower()
        if search_lower in term_lower:
            suggestions.append(str(term))
        elif any(word in term_lower for word in search_words if len(word) > 3):
            suggestions.append(str(term))
        elif len(search_lower) > 3 and len(term_lower) > 3:
            matches = sum(1 for i in range(min(len(search_lower), len(term_lower))) 
                         if i < len(search_lower) and i < len(term_lower) and search_lower[i] == term_lower[i])
            similarity = matches / max(len(search_lower), len(term_lower))
            if similarity > 0.6:
                suggestions.append(str(term))
    
    suggestions = list(set(suggestions))[:3]
    return suggestions

def is_question(search_term):
    if not search_term:
        return False
    
    search_lower = search_term.lower().strip()
    if search_lower.endswith('?'):
        return True
    
    question_words = [
        'what', 'how', 'why', 'when', 'where', 'who', 'whom', 'whose', 'which',
        'does', 'do', 'did', 'is', 'are', 'was', 'were', 'has', 'have', 'had',
        'can', 'could', 'will', 'would', 'shall', 'should', 'may', 'might', 'must'
    ]
    
    first_word = search_lower.split()[0] if search_lower.split() else ''
    if first_word in question_words:
        return True
    
    for word in question_words:
        if search_lower.startswith(word + ' '):
            return True
    
    return False

def is_short_keyword(search_term):
    if not search_term:
        return False
    if is_question(search_term):
        return False
    word_count = len(search_term.strip().split())
    return word_count <= 3

def search_medical_qa(search_term, limit=6, offset=0):
    """Search medical Q&A with pagination"""
    if df_medical_qa is None:
        return [], 0
    
    search_lower = search_term.lower().strip()
    results = []
    
    try:
        matches = df_medical_qa[df_medical_qa['question'].astype(str).str.lower().str.contains(search_lower, na=False)]
        if matches.empty:
            matches = df_medical_qa[df_medical_qa['answer'].astype(str).str.lower().str.contains(search_lower, na=False)]
        
        for _, row in matches.iterrows():
            results.append({
                'question': row.get('question', ''),
                'answer': row.get('answer', ''),
                'source': row.get('source', 'Medical Q&A')
            })
    except Exception as e:
        print(f"Q&A search error: {e}")
    
    unique_results = []
    seen = set()
    for r in results:
        q_text = r.get('question', '')
        if q_text and q_text not in seen:
            seen.add(q_text)
            unique_results.append(r)
    
    total_count = len(unique_results)
    paginated_results = unique_results[offset:offset+limit]
    
    return paginated_results, total_count

def extract_health_terms(query):
    if not query:
        return []
    
    query_lower = query.lower().strip()
    extracted = []
    health_terms = set()
    
    if df_master is not None and 'disease' in df_master.columns:
        for disease in df_master['disease'].dropna():
            health_terms.add(disease.lower().strip())
    
    if df_symptom_descriptions is not None:
        for symptom in df_symptom_descriptions['symptom'].dropna():
            health_terms.add(symptom.lower().strip())
    
    if df_symptom_precautions is not None:
        for condition in df_symptom_precautions['condition'].dropna():
            health_terms.add(condition.lower().strip())
    
    nlp_symptoms = analyze_symptoms_nlp(query)
    for symptom in nlp_symptoms:
        if symptom not in extracted:
            extracted.append(symptom)
    
    words = query_lower.split()
    for i in range(len(words)):
        for j in range(2, 4):
            if i + j <= len(words):
                phrase = ' '.join(words[i:i+j])
                if phrase in health_terms and phrase not in extracted:
                    extracted.append(phrase)
    
    for word in words:
        word_clean = word.strip('.,!?()[]"\'')
        if word_clean in health_terms and word_clean not in extracted:
            extracted.append(word_clean)
    
    if not extracted:
        common = ['flu', 'fever', 'cough', 'headache', 'pain', 'cold', 'diabetes', 'asthma', 'allergy', 'infection', 'virus', 'disease', 'symptom', 'treatment', 'doctor', 'hospital', 'clinic', 'medicine', 'tb', 'hiv', 'covid']
        for term in common:
            if term in query_lower and term not in extracted:
                extracted.append(term)
    
    return list(set(extracted))[:5]

def get_urgency_level(symptoms, disease_info):
    emergency_keywords = ['chest pain', 'shortness of breath', 'difficulty breathing', 'severe headache', 
                         'unconscious', 'bleeding', 'stroke', 'heart attack', 'seizure', 'allergic reaction']
    urgent_keywords = ['high fever', 'persistent cough', 'vomiting', 'diarrhea', 'pain', 'infection']
    
    symptoms_lower = symptoms.lower() if symptoms else ''
    disease_lower = disease_info.lower() if disease_info else ''
    combined = f"{symptoms_lower} {disease_lower}"
    
    for keyword in emergency_keywords:
        if keyword in combined:
            return 'Emergency'
    
    for keyword in urgent_keywords:
        if keyword in combined:
            return 'Urgent'
    
    return 'Routine'

def search_master_database(search_term):
    if df_master is None:
        return []
    
    search_lower = search_term.lower().strip()
    results = []
    
    try:
        matches = df_master[df_master['disease'].astype(str).str.lower().str.contains(search_lower, na=False)]
        if matches.empty:
            matches = df_master[df_master['all_symptoms'].astype(str).str.lower().str.contains(search_lower, na=False)]
        
        for _, row in matches.iterrows():
            symptoms = row.get('all_symptoms', '')
            if isinstance(symptoms, str):
                try: symptoms = eval(symptoms)
                except: symptoms = []
            
            cures = row.get('cures_list', '')
            if isinstance(cures, str):
                try: cures = eval(cures)
                except: cures = []
            
            doctors = row.get('doctors_list', '')
            if isinstance(doctors, str):
                try: doctors = eval(doctors)
                except: doctors = []
            
            disease_name = row.get('disease', '')
            urgency = get_urgency_level(str(symptoms), disease_name)
            
            results.append({
                'disease': disease_name,
                'symptoms': symptoms,
                'description': row.get('description', 'No description available'),
                'treatments': cures,
                'precautions': row.get('precautions', 'No precautions available'),
                'doctors': doctors,
                'risk': row.get('risk_category', 'Unknown'),
                'urgency': urgency
            })
    except Exception as e:
        print(f"Search error: {e}")
    
    return results

def search_symptom_descriptions(search_term):
    if df_symptom_descriptions is None:
        return []
    
    search_lower = search_term.lower().strip()
    results = []
    
    try:
        matches = df_symptom_descriptions[df_symptom_descriptions['symptom'].astype(str).str.lower().str.contains(search_lower, na=False)]
        for _, row in matches.iterrows():
            results.append({
                'symptom': row.get('symptom', ''),
                'description': row.get('description', '')
            })
    except Exception as e:
        print(f"Symptom descriptions search error: {e}")
    
    return results[:3]

def search_symptom_precautions(search_term):
    if df_symptom_precautions is None:
        return []
    
    search_lower = search_term.lower().strip()
    results = []
    
    try:
        matches = df_symptom_precautions[df_symptom_precautions['condition'].astype(str).str.lower().str.contains(search_lower, na=False)]
        for _, row in matches.iterrows():
            results.append({
                'condition': row.get('condition', ''),
                'precaution': row.get('precaution', '')
            })
    except Exception as e:
        print(f"Precautions search error: {e}")
    
    return results[:3]

# ============================================================
# MEDICINE SEARCH FUNCTIONS
# ============================================================

def search_medicines_by_disease(search_term):
    if df_disease_to_medicine is None or not search_term:
        return []
    
    search_lower = search_term.lower().strip()
    results = []
    
    try:
        matches = df_disease_to_medicine[
            df_disease_to_medicine['disease'].astype(str).str.lower().str.contains(search_lower, na=False)
        ]
        
        for _, row in matches.iterrows():
            medicines = str(row.get('medicines', '')).split(', ')
            results.append({
                'disease': row.get('disease', ''),
                'medicines': [m.strip() for m in medicines if m and m != 'Not available']
            })
    except Exception as e:
        print(f"Medicine search error: {e}")
    
    return results

def search_diseases_by_medicine(search_term):
    if df_medicine_to_disease is None or not search_term:
        return []
    
    search_lower = search_term.lower().strip()
    results = []
    
    try:
        matches = df_medicine_to_disease[
            df_medicine_to_disease['medicine'].astype(str).str.lower().str.contains(search_lower, na=False)
        ]
        
        for _, row in matches.iterrows():
            diseases = str(row.get('diseases', '')).split(', ')
            results.append({
                'medicine': row.get('medicine', ''),
                'diseases': [d.strip() for d in diseases if d and d != 'Not available']
            })
    except Exception as e:
        print(f"Disease search error: {e}")
    
    return results

def get_medicine_details(medicine_name):
    if df_medicines_master is None or not medicine_name:
        return None
    
    try:
        matches = df_medicines_master[
            df_medicines_master['medicine'].astype(str).str.lower() == medicine_name.lower().strip()
        ]
        
        if len(matches) > 0:
            row = matches.iloc[0]
            return {
                'medicine': row.get('medicine', ''),
                'generic_name': row.get('generic_name', ''),
                'disease': row.get('disease', ''),
                'indications': row.get('indications', ''),
                'category': row.get('category', ''),
                'dosage': row.get('dosage', ''),
                'frequency': row.get('frequency', ''),
                'duration': row.get('duration', ''),
                'route': row.get('route', ''),
                'strength': row.get('strength', ''),
                'side_effects': row.get('side_effects', ''),
                'precautions': row.get('precautions', ''),
                'prescription': row.get('prescription', ''),
                'manufacturer': row.get('manufacturer', '')
            }
    except Exception as e:
        print(f"Medicine details error: {e}")
    
    return None

# ============================================================
# AUTHENTICATION ROUTES
# ============================================================

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        confirm = request.form.get('confirm_password', '').strip()
        full_name = request.form.get('full_name', '').strip()
        phone = request.form.get('phone', '').strip()
        role = request.form.get('role', 'patient').strip()
        staff_code = request.form.get('staff_code', '').strip()
        terms = request.form.get('terms')
        
        phone_clean = re.sub(r'[^0-9]', '', phone)
        if phone_clean and (len(phone_clean) != 10 or not phone_clean.startswith('0')):
            flash('Please enter a valid 10-digit phone number starting with 0 (e.g., 0821234567)', 'error')
            return render_template('signup.html', lang='en', translate_text=translate_text)
        phone = phone_clean
        
        if not email or '@' not in email:
            flash('Please enter a valid email', 'error')
            return render_template('signup.html', lang='en', translate_text=translate_text)
        
        if len(password) < 6:
            flash('Password must be at least 6 characters', 'error')
            return render_template('signup.html', lang='en', translate_text=translate_text)
        
        if password != confirm:
            flash('Passwords do not match', 'error')
            return render_template('signup.html', lang='en', translate_text=translate_text)
        
        if not full_name:
            flash('Please enter your full name', 'error')
            return render_template('signup.html', lang='en', translate_text=translate_text)
        
        if role in ['nurse', 'admin'] and not staff_code:
            flash('Staff code required for nurse/admin', 'error')
            return render_template('signup.html', lang='en', translate_text=translate_text)
        
        if not terms:
            flash('You must agree to the Terms and Conditions.', 'error')
            return render_template('signup.html', lang='en', translate_text=translate_text)
        
        # ===== FIX #10: use DB_PATH =====
        conn = sqlite3.connect(DB_PATH)
        # ===== END FIX #10 =====
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        if cursor.fetchone():
            conn.close()
            flash('Email already registered', 'error')
            return render_template('signup.html', lang='en', translate_text=translate_text)
        
        cursor.execute('''
        SELECT version FROM terms_versions 
        WHERE is_active = 1 
        ORDER BY effective_date DESC 
        LIMIT 1
        ''')
        terms_result = cursor.fetchone()
        terms_version = terms_result[0] if terms_result else '1.0'
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute('''
        INSERT INTO users (email, password_hash, full_name, phone, role, staff_code, is_verified, is_active, terms_accepted, terms_version)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (email, password_hash, full_name, phone, role, staff_code, 1, 1, 1, terms_version))
        conn.commit()
        user_id = cursor.lastrowid
        conn.close()
        
        session['user_id'] = user_id
        session['email'] = email
        session['full_name'] = full_name
        session['role'] = role
        session['language'] = 'en'
        
        flash('Account created! Please complete your profile.', 'success')
        
        if role == 'patient':
            return redirect(url_for('patient_profile_setup'))
        elif role == 'admin':
            return redirect(url_for('admin_dashboard'))
        else:
            return redirect(url_for('staff_profile_setup'))
    
    lang = session.get('language', 'en')
    return render_template('signup.html', lang=lang, translate_text=translate_text)

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        if not email or not password:
            flash('Please fill in all fields', 'error')
            lang = session.get('language', 'en')
            return render_template('login.html', lang=lang, translate_text=translate_text)
        
        # ===== FIX #10: use DB_PATH =====
        conn = sqlite3.connect(DB_PATH)
        # ===== END FIX #10 =====
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
        SELECT id, email, full_name, phone, role, staff_clinic, age, health_conditions, location, is_active, password_hash, terms_accepted, language, gender, id_number
        FROM users WHERE email = ?
        ''', (email,))
        user = cursor.fetchone()
        conn.close()
        
        if user and user['password_hash'] == hashlib.sha256(password.encode()).hexdigest():
            if user['is_active'] == 1:
                session['user_id'] = user['id']
                session['email'] = user['email']
                session['full_name'] = user['full_name']
                session['role'] = user['role']
                session['staff_clinic'] = user['staff_clinic'] or ''
                session['age'] = user['age'] or 30
                session['health_conditions'] = user['health_conditions'] or ''
                session['location'] = user['location'] or ''
                session['language'] = user['language'] if user['language'] else 'en'
                session['gender'] = user['gender'] or ''
                session['id_number'] = user['id_number'] or ''
                
                flash(f'Welcome back, {user["full_name"]}!', 'success')
                
                if user['terms_accepted'] == 0:
                    flash('Please accept the Terms and Conditions to continue.', 'warning')
                    return redirect(url_for('view_terms'))
                
                if user['role'] == 'patient':
                    if user['age'] is None or user['age'] == 0 or not user['location']:
                        flash('Please complete your profile first.', 'warning')
                        return redirect(url_for('patient_profile_setup'))
                    return redirect(url_for('patient_dashboard'))
                elif user['role'] in ['nurse', 'staff']:
                    if not user['staff_clinic']:
                        flash('Please select your clinic first.', 'warning')
                        return redirect(url_for('staff_profile_setup'))
                    return redirect(url_for('staff_dashboard'))
                else:
                    return redirect(url_for('admin_dashboard'))
            else:
                flash('Account inactive', 'error')
        else:
            flash('Invalid credentials', 'error')
    
    lang = session.get('language', 'en')
    return render_template('login.html', lang=lang, translate_text=translate_text)

@app.route('/logout')
def logout():
    session.clear()
    flash('Logged out', 'info')
    return redirect(url_for('login'))

@app.route('/')
def home():
    if 'user_id' not in session:
        # ===== FIX #3: send public visitors to welcome page =====
        return redirect('/welcome')
        # ===== END FIX #3 =====
    role = session.get('role')
    if role == 'patient':
        return redirect('/patient/dashboard')
    elif role in ['nurse', 'staff']:
        return redirect('/staff/dashboard')
    elif role == 'admin':
        return redirect('/admin/dashboard')
    return redirect('/login')

# ============================================================
# FIX #3: PUBLIC ROUTES (no login required)
# ============================================================

@app.route('/welcome')
def public_home():
    clinics_count = len(df_clinics) if df_clinics is not None else 521
    diseases_count = len(df_master) if df_master is not None else 99
    medicines_count = len(df_medicines_master) if df_medicines_master is not None else 24034
    lang = session.get('language', 'en')
    return render_template('public_home.html',
                           clinics_count=clinics_count,
                           diseases_count=diseases_count,
                           medicines_count=medicines_count,
                           lang=lang, translate_text=translate_text)

@app.route('/public/symptoms', methods=['GET', 'POST'])
def public_symptoms():
    search_result = None
    search_term = ''
    if request.method == 'POST':
        search_term = request.form.get('search_term', '').strip()
        if search_term:
            search_result = {
                'found': False, 'search_term': search_term, 'did_you_mean': None,
                'diseases': [], 'message': ''
            }
            diseases = search_master_database(search_term)
            if diseases:
                search_result['found'] = True
                search_result['diseases'] = diseases[:3]
            else:
                suggestion = correct_spelling(search_term)
                if suggestion:
                    search_result['did_you_mean'] = suggestion
                    search_result['message'] = f'No results for "{search_term}". Did you mean "{suggestion}"?'
                else:
                    search_result['message'] = f'No results for "{search_term}".'
    lang = session.get('language', 'en')
    return render_template('public_symptoms.html',
                           search_result=search_result,
                           search_term=search_term,
                           lang=lang, translate_text=translate_text)

@app.route('/public/clinics', methods=['GET', 'POST'])
def public_clinics():
    search_location = ''
    search_results = None
    if request.method == 'POST':
        search_location = request.form.get('location', '').strip()
        if search_location and df_clinics is not None:
            df = df_clinics.copy()
            for col in ['Province', 'District', 'City', 'Area', 'Clinic_Name']:
                if col in df.columns:
                    df[col] = df[col].astype(str).str.lower()
            sl = search_location.lower()
            mask = (df['Province'].str.contains(sl, na=False) |
                    df['District'].str.contains(sl, na=False) |
                    df['City'].str.contains(sl, na=False) |
                    df['Area'].str.contains(sl, na=False) |
                    df['Clinic_Name'].str.contains(sl, na=False))
            results = df[mask].head(50)
            search_results = results.to_dict('records')
            for c in search_results:
                n = str(c.get('Clinic_Name', '')).replace(' ', '+')
                ci = str(c.get('City', '')).replace(' ', '+')
                p = str(c.get('Province', '')).replace(' ', '+')
                q = f"{n}+{ci}+{p}+South+Africa"
                c['maps_url'] = f"https://www.google.com/maps/search/?api=1&query={q}"
                c['directions_url'] = f"https://www.google.com/maps/dir/?api=1&destination={q}"
    lang = session.get('language', 'en')
    return render_template('public_clinics.html',
                           clinics=search_results,
                           search_location=search_location,
                           lang=lang, translate_text=translate_text)

@app.route('/public/first-aid')
def public_first_aid():
    first_aid_categories = [
        {'id': 'heart_attack', 'icon': 'fa-heart-pulse', 'title': 'Heart Attack', 'description': 'Emergency signs & response', 'color': 'danger'},
        {'id': 'stroke', 'icon': 'fa-brain', 'title': 'Stroke', 'description': 'FAST - Face, Arms, Speech, Time', 'color': 'danger'},
        {'id': 'choking', 'icon': 'fa-lungs', 'title': 'Choking', 'description': 'Heimlich maneuver for adults & children', 'color': 'danger'},
        {'id': 'severe_bleeding', 'icon': 'fa-droplet', 'title': 'Severe Bleeding', 'description': 'How to stop heavy bleeding', 'color': 'danger'},
        {'id': 'allergic_reaction', 'icon': 'fa-allergies', 'title': 'Allergic Reaction', 'description': 'Anaphylaxis emergency response', 'color': 'danger'},
        {'id': 'seizure', 'icon': 'fa-bolt', 'title': 'Seizure / Fits', 'description': 'What to do during a seizure', 'color': 'danger'},
        {'id': 'poisoning', 'icon': 'fa-skull-crossbones', 'title': 'Poisoning', 'description': 'Immediate steps for poisoning', 'color': 'danger'},
        {'id': 'drowning', 'icon': 'fa-water', 'title': 'Drowning', 'description': 'Rescue & CPR for drowning', 'color': 'danger'},
        {'id': 'burns', 'icon': 'fa-fire', 'title': 'Burns & Scalds', 'description': 'First aid for burns', 'color': 'warning'},
        {'id': 'fracture', 'icon': 'fa-bone', 'title': 'Fractures & Sprains', 'description': 'Handle broken bones & sprains', 'color': 'warning'},
        {'id': 'heatstroke', 'icon': 'fa-temperature-high', 'title': 'Heat Stroke & Dehydration', 'description': 'Emergency cooling response', 'color': 'warning'},
        {'id': 'hypothermia', 'icon': 'fa-snowflake', 'title': 'Hypothermia', 'description': 'Warming a person safely', 'color': 'warning'},
        {'id': 'diabetic_emergency', 'icon': 'fa-syringe', 'title': 'Diabetic Emergency', 'description': 'Low/high blood sugar response', 'color': 'warning'},
        {'id': 'asthma_attack', 'icon': 'fa-lungs', 'title': 'Asthma Attack', 'description': 'Help someone having an asthma attack', 'color': 'warning'},
        {'id': 'cuts_scrapes', 'icon': 'fa-bandage', 'title': 'Cuts & Scrapes', 'description': 'Clean and dress minor wounds', 'color': 'info'},
        {'id': 'insect_bites', 'icon': 'fa-bug', 'title': 'Insect Bites & Stings', 'description': 'Treatment for bites and stings', 'color': 'info'},
        {'id': 'nosebleed', 'icon': 'fa-nose', 'title': 'Nosebleeds', 'description': 'How to stop a nosebleed', 'color': 'info'},
        {'id': 'headache_migraine', 'icon': 'fa-head-side-virus', 'title': 'Headache & Migraine', 'description': 'Relief for headaches', 'color': 'info'},
        {'id': 'food_poisoning', 'icon': 'fa-utensils', 'title': 'Food Poisoning', 'description': 'Symptoms and home care', 'color': 'info'},
        {'id': 'fever_management', 'icon': 'fa-thermometer', 'title': 'Fever Management', 'description': 'How to manage a fever', 'color': 'info'},
    ]
    lang = session.get('language', 'en')
    return render_template('public_first_aid.html',
                           first_aid_categories=first_aid_categories,
                           lang=lang, translate_text=translate_text)

# ============================================================
# END FIX #3
# ============================================================

# ============================================================
# FORGOT PASSWORD & RESET
# ============================================================

@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'POST':
        email = request.form.get('email', '').strip()
        
        if not email:
            flash('Please enter your email address.', 'error')
            lang = session.get('language', 'en')
            return render_template('forgot_password.html', lang=lang, translate_text=translate_text)
        
        # ===== FIX #10: use DB_PATH =====
        conn = sqlite3.connect(DB_PATH)
        # ===== END FIX #10 =====
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM users WHERE email = ?", (email,))
        user = cursor.fetchone()
        conn.close()
        
        if not user:
            flash('No account found with that email address.', 'error')
            lang = session.get('language', 'en')
            return render_template('forgot_password.html', lang=lang, translate_text=translate_text)
        
        reset_token = generate_reset_token()
        expires = datetime.now() + timedelta(minutes=15)
        
        # ===== FIX #10: use DB_PATH =====
        conn = sqlite3.connect(DB_PATH)
        # ===== END FIX #10 =====
        cursor = conn.cursor()
        cursor.execute('''
        UPDATE users SET reset_token = ?, reset_expires = ?
        WHERE email = ?
        ''', (reset_token, expires, email))
        conn.commit()
        conn.close()
        
        if send_reset_email(email, reset_token):
            flash(f'Password reset link sent to {email}. Please check your inbox.', 'success')
        else:
            flash('Failed to send email. Please try again later.', 'error')
            lang = session.get('language', 'en')
            return render_template('forgot_password.html', lang=lang, translate_text=translate_text)
        
        return redirect(url_for('login'))
    
    lang = session.get('language', 'en')
    return render_template('forgot_password.html', lang=lang, translate_text=translate_text)

@app.route('/reset-password/<token>', methods=['GET', 'POST'])
def reset_password(token):
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
    SELECT email FROM users 
    WHERE reset_token = ? AND reset_expires > datetime('now')
    ''', (token,))
    user = cursor.fetchone()
    conn.close()
    
    if not user:
        flash('Invalid or expired reset link. Please request a new one.', 'error')
        return redirect(url_for('forgot_password'))
    
    if request.method == 'POST':
        new_password = request.form.get('new_password', '').strip()
        confirm_password = request.form.get('confirm_password', '').strip()
        
        if len(new_password) < 6:
            flash('Password must be at least 6 characters.', 'error')
            lang = session.get('language', 'en')
            return render_template('reset_password.html', token=token, lang=lang, translate_text=translate_text)
        
        if new_password != confirm_password:
            flash('Passwords do not match.', 'error')
            lang = session.get('language', 'en')
            return render_template('reset_password.html', token=token, lang=lang, translate_text=translate_text)
        
        # ===== FIX #10: use DB_PATH =====
        conn = sqlite3.connect(DB_PATH)
        # ===== END FIX #10 =====
        cursor = conn.cursor()
        password_hash = hashlib.sha256(new_password.encode()).hexdigest()
        cursor.execute('''
        UPDATE users SET password_hash = ?, reset_token = NULL, reset_expires = NULL
        WHERE reset_token = ?
        ''', (password_hash, token))
        conn.commit()
        conn.close()
        
        flash('Password reset successful! You can now login with your new password.', 'success')
        return redirect(url_for('login'))
    
    lang = session.get('language', 'en')
    return render_template('reset_password.html', token=token, lang=lang, translate_text=translate_text)

# ============================================================
# TERMS & CONDITIONS
# ============================================================

@app.route('/terms')
def view_terms():
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT version, content, effective_date 
    FROM terms_versions 
    WHERE is_active = 1 
    ORDER BY effective_date DESC 
    LIMIT 1
    ''')
    terms = cursor.fetchone()
    conn.close()
    
    if not terms:
        flash('Terms not available. Please contact support.', 'error')
        return redirect(url_for('home'))
    
    terms_accepted = False
    accepted_at = None
    if 'user_id' in session:
        # ===== FIX #10: use DB_PATH =====
        conn = sqlite3.connect(DB_PATH)
        # ===== END FIX #10 =====
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
        SELECT terms_accepted, terms_accepted_at, terms_version 
        FROM users 
        WHERE id = ?
        ''', (session['user_id'],))
        user = cursor.fetchone()
        conn.close()
        if user:
            terms_accepted = user['terms_accepted'] == 1
            accepted_at = user['terms_accepted_at']
    
    lang = session.get('language', 'en')
    return render_template('terms.html',
        content=terms['content'],
        version=terms['version'],
        effective_date=terms['effective_date'],
        terms_accepted=terms_accepted,
        accepted_at=accepted_at,
        lang=lang,
        translate_text=translate_text
    )

@app.route('/accept-terms', methods=['POST'])
def accept_terms():
    if 'user_id' not in session:
        flash('Please login first.', 'error')
        return redirect(url_for('login'))
    
    terms_agree = request.form.get('terms_agree')
    if not terms_agree:
        flash('You must agree to the terms to continue.', 'error')
        return redirect(url_for('view_terms'))
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    cursor = conn.cursor()
    cursor.execute('''
    SELECT version FROM terms_versions 
    WHERE is_active = 1 
    ORDER BY effective_date DESC 
    LIMIT 1
    ''')
    terms = cursor.fetchone()
    version = terms[0] if terms else '1.0'
    
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    cursor.execute('''
    UPDATE users 
    SET terms_accepted = 1, 
        terms_accepted_at = ?, 
        terms_version = ? 
    WHERE id = ?
    ''', (now, version, session['user_id']))
    conn.commit()
    conn.close()
    
    flash('Terms accepted successfully!', 'success')
    return redirect(url_for('home'))

@app.route('/admin/terms', methods=['GET', 'POST'])
def admin_terms():
    if 'user_id' not in session or session.get('role') != 'admin':
        flash('Admin access required.', 'error')
        return redirect(url_for('login'))
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if request.method == 'POST':
        action = request.form.get('action')
        
        if action == 'update':
            version = request.form.get('version', '').strip()
            content = request.form.get('content', '').strip()
            
            if not version or not content:
                flash('Version and content are required.', 'error')
                return render_template('admin_terms.html', terms=None, versions=[])
            
            cursor.execute("UPDATE terms_versions SET is_active = 0")
            
            cursor.execute('''
            INSERT INTO terms_versions (version, content, created_by, is_active)
            VALUES (?, ?, ?, ?)
            ''', (version, content, session['user_id'], 1))
            conn.commit()
            
            cursor.execute("UPDATE users SET terms_accepted = 0, terms_accepted_at = NULL")
            conn.commit()
            
            flash(f'Terms updated to version {version}. All users must re-accept.', 'success')
    
    cursor.execute('''
    SELECT version, content, effective_date 
    FROM terms_versions 
    WHERE is_active = 1 
    ORDER BY effective_date DESC 
    LIMIT 1
    ''')
    terms = cursor.fetchone()
    
    cursor.execute('''
    SELECT id, version, content, effective_date, is_active 
    FROM terms_versions 
    ORDER BY effective_date DESC
    ''')
    versions = cursor.fetchall()
    conn.close()
    
    lang = session.get('language', 'en')
    return render_template('admin_terms.html',
        terms=terms,
        versions=versions,
        default_content=default_terms_content(),
        lang=lang,
        translate_text=translate_text
    )

# ============================================================
# PATIENT PROFILE SETUP - FIX #1 (SA ID) + FIX #6 (allergies)
# ============================================================

@app.route('/patient/profile-setup', methods=['GET', 'POST'])
@require_terms_acceptance
def patient_profile_setup():
    if 'user_id' not in session:
        return redirect('/login')
    if session.get('role') != 'patient':
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT age, location, health_conditions FROM users WHERE id = ?', (session['user_id'],))
    user = cursor.fetchone()
    conn.close()
    
    if user and user['age'] and user['age'] > 0 and user['location']:
        flash('Profile already set up', 'info')
        return redirect('/patient/dashboard')
    
    if request.method == 'POST':
        age = request.form.get('age', '')
        location = request.form.get('location', '').strip()
        health_conditions = request.form.get('health_conditions', '').strip()
        emergency_name = request.form.get('emergency_name', '').strip()
        emergency_phone = request.form.get('emergency_phone', '').strip()
        emergency_relationship = request.form.get('emergency_relationship', '').strip()
        allergies = request.form.get('allergy_details', '').strip()
        gender = request.form.get('gender', '').strip()
        id_number = request.form.get('id_number', '').strip()
        
        if not age:
            flash('Please enter your age', 'error')
            lang = session.get('language', 'en')
            return render_template('patient_profile_setup.html', user=session, lang=lang, translate_text=translate_text)
        
        try:
            age = int(age)
            if age < 1 or age > 120:
                flash('Please enter a valid age (1-120)', 'error')
                lang = session.get('language', 'en')
                return render_template('patient_profile_setup.html', user=session, lang=lang, translate_text=translate_text)
        except:
            flash('Please enter a valid age', 'error')
            lang = session.get('language', 'en')
            return render_template('patient_profile_setup.html', user=session, lang=lang, translate_text=translate_text)
        
        if not location:
            flash('Please enter your location', 'error')
            lang = session.get('language', 'en')
            return render_template('patient_profile_setup.html', user=session, lang=lang, translate_text=translate_text)
        
        # ===== FIX #1: SA ID validation =====
        if id_number:
            valid, msg = validate_sa_id(id_number)
            if not valid:
                flash(f'ID Number: {msg}', 'error')
                lang = session.get('language', 'en')
                return render_template('patient_profile_setup.html', user=session, lang=lang, translate_text=translate_text)
        # ===== END FIX #1 =====
        
        if not gender:
            flash('Please select your gender.', 'error')
            lang = session.get('language', 'en')
            return render_template('patient_profile_setup.html', user=session, lang=lang, translate_text=translate_text)
        
        # ===== FIX #10: use DB_PATH =====
        conn = sqlite3.connect(DB_PATH)
        # ===== END FIX #10 =====
        cursor = conn.cursor()
        cursor.execute('''
        UPDATE users 
        SET age = ?, location = ?, health_conditions = ?,
            emergency_name = ?, emergency_phone = ?, 
            emergency_relationship = ?, allergies = ?,
            gender = ?, id_number = ?
        WHERE id = ?
        ''', (age, location, health_conditions, emergency_name, emergency_phone, 
              emergency_relationship, allergies, gender, id_number, session['user_id']))
        conn.commit()
        conn.close()
        
        session['age'] = age
        session['location'] = location
        session['health_conditions'] = health_conditions
        session['gender'] = gender
        session['id_number'] = id_number
        
        flash('Profile setup complete! Welcome to MediSense.', 'success')
        return redirect('/patient/dashboard')
    
    lang = session.get('language', 'en')
    return render_template('patient_profile_setup.html', 
        user=session,
        lang=lang,
        translate_text=translate_text
    )

# ============================================================
# STAFF PROFILE SETUP
# ============================================================

@app.route('/staff/profile-setup', methods=['GET', 'POST'])
@require_terms_acceptance
def staff_profile_setup():
    if 'user_id' not in session:
        return redirect('/login')
    if session.get('role') not in ['nurse', 'staff']:
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT staff_clinic FROM users WHERE id = ?', (session['user_id'],))
    user = cursor.fetchone()
    conn.close()
    
    if user and user['staff_clinic']:
        flash('Profile already set up', 'info')
        return redirect('/staff/dashboard')
    
    clinics = get_clinics_list()
    
    if request.method == 'POST':
        clinic = request.form.get('staff_clinic', '').strip()
        
        if not clinic:
            flash('Please select a clinic', 'error')
            lang = session.get('language', 'en')
            return render_template('staff_profile_setup.html', user=session, clinics=clinics, lang=lang, translate_text=translate_text)
        
        # ===== FIX #10: use DB_PATH =====
        conn = sqlite3.connect(DB_PATH)
        # ===== END FIX #10 =====
        cursor = conn.cursor()
        cursor.execute('''
        UPDATE users SET staff_clinic = ?
        WHERE id = ?
        ''', (clinic, session['user_id']))
        conn.commit()
        conn.close()
        
        session['staff_clinic'] = clinic
        
        flash('Profile setup complete! Welcome to MediSense.', 'success')
        return redirect('/staff/dashboard')
    
    lang = session.get('language', 'en')
    return render_template('staff_profile_setup.html', 
        user=session, 
        clinics=clinics,
        lang=lang,
        translate_text=translate_text
    )

# ============================================================
# PATIENT DASHBOARD - FIX #9: health score starts at 0
# ============================================================

@app.route('/patient/dashboard')
@require_terms_acceptance
def patient_dashboard():
    if 'user_id' not in session or session.get('role') != 'patient':
        return redirect('/login')

    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute('SELECT id, email, full_name, phone, role, age, health_conditions, location, gender, id_number, allergies, emergency_name, emergency_phone FROM users WHERE id = ?', (session['user_id'],))
    patient = cursor.fetchone()

    if patient is None:
        flash('User record not found. Please log in again.', 'error')
        session.clear()
        conn.close()
        return redirect('/login')

    cursor.execute('''
    SELECT id, clinic_name, appointment_date, appointment_time, status, reason
    FROM appointments
    WHERE patient_email = ?
    ORDER BY appointment_date DESC, appointment_time DESC
    ''', (session['email'],))
    appointments = cursor.fetchall()
    conn.close()

    age = patient['age'] if patient['age'] else 0
    health_conditions = patient['health_conditions'] if patient['health_conditions'] else ''
    health_count = len([c for c in health_conditions.split(',') if c.strip()]) if health_conditions else 0

    # ===== FIX #9: HEALTH SCORE — starts at 0, based on real user data =====
    # Breakdown (max 100):
    #  - Profile completeness: up to 30 points
    #  - Completed appointments: up to 40 points
    #  - Checked-in appointments: up to 15 points
    #  - Regular booking bonus: up to 15 points
    #  - No-show penalty: up to -30
    #  - Cancelled penalty: up to -10
    profile_complete = bool(age and patient['location'])
    has_id = bool(patient['id_number'])
    has_emergency = bool(patient['emergency_name'] and patient['emergency_phone'])
    has_allergies = patient['allergies'] or ''

    health_score = calculate_health_score(
        age, health_conditions, appointments,
        profile_complete=profile_complete,
        has_id=has_id,
        has_emergency=has_emergency,
        has_allergies=has_allergies
    )
    health_score_category = get_health_score_category(health_score)

    # Give the dashboard a short explanation for the tooltip
    score_breakdown = []
    if age and 1 <= age <= 120:
        score_breakdown.append("Age provided (+5)")
    if profile_complete:
        score_breakdown.append("Profile complete (+10)")
    if has_id:
        score_breakdown.append("ID number on file (+5)")
    if has_emergency:
        score_breakdown.append("Emergency contact on file (+5)")
    if health_conditions:
        score_breakdown.append("Health conditions recorded (+5)")
    if has_allergies:
        score_breakdown.append("Allergies recorded (+5)")
    completed_ct = sum(1 for a in appointments if a['status'] == 'Completed')
    if completed_ct:
        score_breakdown.append(f"{completed_ct} completed appointment(s) (+{min(completed_ct * 5, 40)})")
    checked_in_ct = sum(1 for a in appointments if a['status'] == 'Checked-in')
    if checked_in_ct:
        score_breakdown.append(f"{checked_in_ct} checked-in appointment(s) (+{min(checked_in_ct * 3, 15)})")
    ns_ct = sum(1 for a in appointments if a['status'] == 'No-Show')
    if ns_ct:
        score_breakdown.append(f"{ns_ct} no-show(s) (-{min(ns_ct * 10, 30)})")
    cancel_ct = sum(1 for a in appointments if a['status'] == 'Cancelled')
    if cancel_ct:
        score_breakdown.append(f"{cancel_ct} cancelled appointment(s) (-{min(cancel_ct * 2, 10)})")

    # Keep risk_score/risk_category for template backward-compat
    risk_score = health_score
    risk_category = health_score_category
    # ===== END FIX #9 =====

    nearest_clinic = get_nearest_clinic(patient['location'] if patient['location'] else None)
    health_tip = get_random_health_tip()
    no_show_prediction = predict_no_show(session['email'])

    lang = session.get('language', 'en')
    return render_template('patient_dashboard.html',
        user=dict(patient),
        appointments=appointments,
        total_visits=len(appointments),
        no_shows=sum(1 for a in appointments if a['status'] == 'No-Show'),
        no_show_count=sum(1 for a in appointments if a['status'] == 'No-Show'),
        completed_count=sum(1 for a in appointments if a['status'] == 'Completed'),
        cancelled_count=sum(1 for a in appointments if a['status'] == 'Cancelled'),
        total_appointments=len(appointments),
        health_score=health_score,
        health_score_category=health_score_category,
        score_breakdown=score_breakdown,
        risk_score=risk_score,
        risk_category=risk_category,
        nearest_clinic=nearest_clinic,
        age=age,
        health_conditions=health_conditions,
        health_count=health_count,
        health_tip=health_tip,
        no_show_prediction=no_show_prediction,
        lang=lang,
        translate_text=translate_text
    )

# ============================================================
# PATIENT - HEALTH & SYMPTOMS - FIX #7 (Q&A removed), FIX #8 (spell)
# ============================================================

@app.route('/patient/health-symptoms', methods=['GET', 'POST'])
@require_terms_acceptance
def patient_health_symptoms():
    if 'user_id' not in session or session.get('role') != 'patient':
        return redirect('/login')
    
    search_result = None
    search_term = ""
    search_history = []
    featured_diseases = ['Diabetes', 'HIV/AIDS', 'Heart Attack', 'Stroke', 'Tuberculosis', 'Cancer', 'Asthma', 'Malaria']
    trending_topics = ['Flu', 'Cough', 'Fever', 'Headache', 'COVID-19', 'Malaria']
    
    if df_master is not None and 'disease' in df_master.columns:
        try:
            featured_diseases = df_master['disease'].head(8).tolist()
        except:
            pass
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
    SELECT search_term, result, urgency_level, created_at 
    FROM symptom_history 
    WHERE user_id = ? 
    ORDER BY created_at DESC 
    LIMIT 5
    ''', (session['user_id'],))
    search_history = cursor.fetchall()
    conn.close()
    
    if request.method == 'POST':
        search_term = request.form.get('search_term', '').strip()
        
        if search_term:
            is_question_search = is_question(search_term)
            is_keyword_search = is_short_keyword(search_term) and not is_question_search
            
            nlp_symptoms = analyze_symptoms_nlp(search_term)
            extracted_terms = extract_health_terms(search_term)
            all_extracted = list(set(nlp_symptoms + extracted_terms))
            search_queries = all_extracted if all_extracted else [search_term]
            
            # ===== FIX #8: offer spelling correction =====
            corrected = None
            has_any_result = bool(search_master_database(search_term)) or bool(search_medicines_by_disease(search_term)) or bool(search_diseases_by_medicine(search_term))
            if not has_any_result:
                corrected = correct_spelling(search_term)
            # ===== END FIX #8 =====
            
            search_result = {
                'found': False,
                'search_term': search_term,
                # ===== FIX #8: did_you_mean =====
                'did_you_mean': corrected,
                # ===== END FIX #8 =====
                'extracted_terms': all_extracted,
                'nlp_symptoms': nlp_symptoms,
                'is_question': is_question_search,
                'is_keyword': is_keyword_search,
                'diseases': [],
                # ===== FIX #7: Q&A fields kept so old template doesn't break, but never populated =====
                'qa': [], 'qa_total': 0, 'qa_offset': 0, 'qa_limit': 0,
                # ===== END FIX #7 =====
                'precautions': [],
                'symptom_descriptions': [],
                'health_tips': [],
                'all_symptoms': [],
                'all_treatments': [],
                'all_doctors': [],
                'all_risks': [],
                'all_urgency': [],
                'has_disease_info': False,
                'has_qa_info': False,
                'has_symptom_info': False,
                'has_precaution_info': False,
                'has_health_tips': False,
                'suggestions': [],
                'has_medicine_info': False,
                'medicines': [],
                'all_medicines': [],
                'has_disease_by_medicine_info': False,
                'diseases_by_medicine': [],
                'all_diseases_by_medicine': []
            }
            
            all_diseases = []
            all_symptoms = []
            all_precautions = []
            all_tips = []
            
            # ===== FIX #7: no Q&A search inside the main search =====
            for query in search_queries:
                disease_results = search_master_database(query)
                if disease_results:
                    all_diseases.extend(disease_results)
                symptom_results = search_symptom_descriptions(query)
                if symptom_results:
                    all_symptoms.extend(symptom_results)
                precaution_results = search_symptom_precautions(query)
                if precaution_results:
                    all_precautions.extend(precaution_results)
                if df_health_tips is not None:
                    try:
                        tip_matches = df_health_tips[df_health_tips['tip'].astype(str).str.lower().str.contains(query.lower(), na=False)]
                        if not tip_matches.empty:
                            all_tips.extend(tip_matches['tip'].head(2).tolist())
                    except:
                        pass
            # ===== END FIX #7 =====
            
            medicine_results = search_medicines_by_disease(search_term)
            if medicine_results:
                search_result['has_medicine_info'] = True
                search_result['medicines'] = medicine_results
                search_result['found'] = True
                all_medicines = []
                for item in medicine_results:
                    all_medicines.extend(item.get('medicines', []))
                search_result['all_medicines'] = list(set(all_medicines))[:10]
            
            disease_results = search_diseases_by_medicine(search_term)
            if disease_results:
                search_result['has_disease_by_medicine_info'] = True
                search_result['diseases_by_medicine'] = disease_results
                search_result['found'] = True
                all_diseases_by_med = []
                for item in disease_results:
                    all_diseases_by_med.extend(item.get('diseases', []))
                search_result['all_diseases_by_medicine'] = list(set(all_diseases_by_med))[:10]
            
            unique_diseases = []
            seen = set()
            for d in all_diseases:
                name = d.get('disease', '')
                if name and name not in seen:
                    seen.add(name)
                    unique_diseases.append(d)
            
            results_found = unique_diseases or all_symptoms or all_precautions or all_tips or medicine_results or disease_results
            
            if not results_found:
                # ===== FIX #8: suggest correction if available =====
                if corrected:
                    search_result['message'] = f"No results for '{search_term}'. Did you mean '{corrected}'?"
                else:
                    suggestions = get_disease_suggestions(search_term)
                    if suggestions:
                        search_result['did_you_mean'] = suggestions[0]
                        search_result['suggestions'] = suggestions
                        if is_question_search:
                            search_result['message'] = f"No results found for your question '{search_term}'. Did you mean: {suggestions[0]}?"
                        else:
                            search_result['message'] = f"No exact match found for '{search_term}'. Did you mean: {suggestions[0]}?"
                    else:
                        if is_question_search:
                            search_result['message'] = f"No results found for your question '{search_term}'. Please try rephrasing or use a shorter search term."
                        else:
                            search_result['message'] = f"No information found for '{search_term}'. Please try a different term."
                # ===== END FIX #8 =====
                search_result['suggestions'] = featured_diseases[:6]
            else:
                search_result['found'] = True
                
                urgency_levels = []
                if unique_diseases:
                    search_result['has_disease_info'] = True
                    search_result['diseases'] = unique_diseases[:3]
                    for d in unique_diseases:
                        if d.get('symptoms'):
                            for s in (d['symptoms'] if isinstance(d['symptoms'], list) else []):
                                if s not in search_result['all_symptoms']:
                                    search_result['all_symptoms'].append(s)
                        if d.get('treatments'):
                            for t in (d['treatments'] if isinstance(d['treatments'], list) else []):
                                if t not in search_result['all_treatments']:
                                    search_result['all_treatments'].append(t)
                        if d.get('doctors'):
                            for doc in (d['doctors'] if isinstance(d['doctors'], list) else []):
                                if doc not in search_result['all_doctors']:
                                    search_result['all_doctors'].append(doc)
                        if d.get('risk') and d['risk'] not in search_result['all_risks']:
                            search_result['all_risks'].append(d['risk'])
                        if d.get('urgency') and d['urgency'] not in urgency_levels:
                            urgency_levels.append(d['urgency'])
                
                if all_symptoms:
                    search_result['has_symptom_info'] = True
                    search_result['symptom_descriptions'] = all_symptoms[:3]
                
                if all_precautions:
                    search_result['has_precaution_info'] = True
                    search_result['precautions'] = all_precautions[:3]
                
                if all_tips:
                    search_result['has_health_tips'] = True
                    search_result['health_tips'] = all_tips[:3]
                
                search_result['all_symptoms'] = search_result['all_symptoms'][:10]
                search_result['all_treatments'] = search_result['all_treatments'][:5]
                search_result['all_doctors'] = search_result['all_doctors'][:3]
                search_result['all_risks'] = search_result['all_risks'][:3]
                search_result['all_urgency'] = urgency_levels[:2]
                
                urgency = ', '.join(urgency_levels[:2]) if urgency_levels else 'Unknown'
                result_summary = search_result['diseases'][0]['disease'] if search_result['diseases'] else 'Health information'
                try:
                    # ===== FIX #10: use DB_PATH =====
                    conn = sqlite3.connect(DB_PATH)
                    # ===== END FIX #10 =====
                    cursor = conn.cursor()
                    cursor.execute('''
                    INSERT INTO symptom_history (user_id, search_term, result, urgency_level)
                    VALUES (?, ?, ?, ?)
                    ''', (session['user_id'], search_term, result_summary, urgency))
                    conn.commit()
                    conn.close()
                except:
                    pass
    
    lang = session.get('language', 'en')
    return render_template('patient_health_symptoms_enhanced.html',
        user=session,
        search_result=search_result,
        search_term=search_term,
        featured_diseases=featured_diseases,
        trending_topics=trending_topics,
        search_history=search_history,
        lang=lang,
        translate_text=translate_text
    )

# ============================================================
# FIX #7: CHATBOT ROUTE (Q&A moved here)
# ============================================================

@app.route('/patient/chatbot', methods=['GET', 'POST'])
@require_terms_acceptance
def patient_chatbot():
    if 'user_id' not in session or session.get('role') != 'patient':
        return redirect('/login')
    
    question = ''
    answer_data = None
    history = []
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
    SELECT question, answer, source, created_at 
    FROM chatbot_history 
    WHERE user_id = ? 
    ORDER BY created_at DESC 
    LIMIT 10
    ''', (session['user_id'],))
    history = cursor.fetchall()
    conn.close()
    
    if request.method == 'POST':
        question = request.form.get('question', '').strip()
        if question:
            corrected = correct_spelling(question)
            query = corrected if corrected else question
            results, _ = search_medical_qa(query, limit=3, offset=0)
            
            answer_data = {
                'question': question,
                'corrected': corrected if corrected and corrected.lower() != question.lower() else None,
                'results': results,
                'found': bool(results),
            }
            
            if not results:
                diseases = search_master_database(query)
                if diseases:
                    answer_data['results'] = [{
                        'question': f'Information about {diseases[0]["disease"]}',
                        'answer': diseases[0].get('description', 'No description available'),
                        'source': 'Disease Database',
                    }]
                    answer_data['found'] = True
            
            if answer_data['found']:
                try:
                    # ===== FIX #10: use DB_PATH =====
                    conn = sqlite3.connect(DB_PATH)
                    # ===== END FIX #10 =====
                    cursor = conn.cursor()
                    cursor.execute('''
                    INSERT INTO chatbot_history (user_id, question, answer, source)
                    VALUES (?, ?, ?, ?)
                    ''', (session['user_id'], question,
                          answer_data['results'][0]['answer'],
                          answer_data['results'][0]['source']))
                    conn.commit()
                    conn.close()
                except:
                    pass
    
    lang = session.get('language', 'en')
    return render_template('patient_chatbot.html',
        user=session,
        question=question,
        answer_data=answer_data,
        history=history,
        lang=lang,
        translate_text=translate_text
    )

@app.route('/load-more-qa', methods=['POST'])
@require_terms_acceptance
def load_more_qa():
    search_term = request.form.get('search_term', '').strip()
    offset = int(request.form.get('offset', 0))
    limit = 6
    if not search_term:
        return jsonify({'error': 'No search term provided'}), 400
    results, total = search_medical_qa(search_term, limit=limit, offset=offset)
    return jsonify({
        'results': results,
        'total': total,
        'offset': offset + len(results),
        'has_more': (offset + len(results)) < total,
        'lang': session.get('language', 'en')
    })

# ============================================================
# HEALTH JOURNAL
# ============================================================

@app.route('/patient/health-journal', methods=['GET', 'POST'])
@require_terms_acceptance
def patient_health_journal():
    if 'user_id' not in session or session.get('role') != 'patient':
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if request.method == 'POST':
        symptom = request.form.get('symptom', '').strip()
        severity = request.form.get('severity', 1)
        notes = request.form.get('notes', '').strip()
        
        if symptom:
            cursor.execute('''
            INSERT INTO health_journal (user_id, symptom, severity, notes)
            VALUES (?, ?, ?, ?)
            ''', (session['user_id'], symptom, severity, notes))
            conn.commit()
            flash('Symptom recorded in your health journal.', 'success')
    
    cursor.execute('''
    SELECT id, symptom, severity, notes, recorded_at 
    FROM health_journal 
    WHERE user_id = ? 
    ORDER BY recorded_at DESC
    ''', (session['user_id'],))
    journal_entries = cursor.fetchall()
    conn.close()
    
    lang = session.get('language', 'en')
    return render_template('patient_health_journal.html',
        user=session,
        journal_entries=journal_entries,
        lang=lang,
        translate_text=translate_text
    )

# ============================================================
# BOOK APPOINTMENT
# ============================================================

@app.route('/patient/book', methods=['GET', 'POST'])
@require_terms_acceptance
def patient_book():
    if 'user_id' not in session or session.get('role') != 'patient':
        return redirect('/login')
    
    clinics = get_clinics_list()
    today = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M')
    
    if request.method == 'POST':
        clinic = request.form.get('clinic')
        date = request.form.get('appointment_date')
        time = request.form.get('appointment_time')
        reason = request.form.get('reason')
        
        if not all([clinic, date, time, reason]):
            flash('Please fill in all fields', 'error')
            lang = session.get('language', 'en')
            return render_template('book_appointment.html', today=today, clinics=clinics, lang=lang, translate_text=translate_text)
        
        try:
            if date < today:
                flash('Cannot book appointments for past dates.', 'error')
                lang = session.get('language', 'en')
                return render_template('book_appointment.html', today=today, clinics=clinics, lang=lang, translate_text=translate_text)
            
            if date == today and time <= current_time:
                flash('Cannot book an appointment time that has already passed today.', 'error')
                lang = session.get('language', 'en')
                return render_template('book_appointment.html', today=today, clinics=clinics, lang=lang, translate_text=translate_text)
        except:
            flash('Invalid date or time format.', 'error')
            lang = session.get('language', 'en')
            return render_template('book_appointment.html', today=today, clinics=clinics, lang=lang, translate_text=translate_text)
        
        # ===== FIX #10: use DB_PATH =====
        conn = sqlite3.connect(DB_PATH)
        # ===== END FIX #10 =====
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id FROM appointments 
        WHERE clinic_name = ? AND appointment_date = ? AND appointment_time = ? 
        AND status != 'Cancelled'
        ''', (clinic, date, time))
        existing_appt = cursor.fetchone()
        
        if existing_appt:
            conn.close()
            flash('This time slot is already booked. Please select another time.', 'error')
            lang = session.get('language', 'en')
            return render_template('book_appointment.html', today=today, clinics=clinics, lang=lang, translate_text=translate_text)
        
        cursor.execute('''
        SELECT id FROM unavailable_slots 
        WHERE clinic_name = ? AND appointment_date = ? AND appointment_time = ?
        ''', (clinic, date, time))
        unavailable_slot = cursor.fetchone()
        
        if unavailable_slot:
            conn.close()
            flash('This time slot is unavailable. Please select another time.', 'error')
            lang = session.get('language', 'en')
            return render_template('book_appointment.html', today=today, clinics=clinics, lang=lang, translate_text=translate_text)
        
        cursor.execute('''
        INSERT INTO appointments (patient_email, patient_name, patient_phone, clinic_name, appointment_date, appointment_time, reason, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', (session['email'], session['full_name'], session.get('phone', ''), clinic, date, time, reason, 'Scheduled'))
        conn.commit()
        conn.close()
        
        flash(f'✅ Appointment booked at {clinic} on {date} at {time}.', 'success')
        return redirect('/patient/dashboard')
    
    lang = session.get('language', 'en')
    return render_template('book_appointment.html', 
        today=today, 
        clinics=clinics,
        lang=lang,
        translate_text=translate_text
    )

@app.route('/patient/cancel/<int:appt_id>', methods=['POST'])
@require_terms_acceptance
def patient_cancel(appt_id):
    if 'user_id' not in session or session.get('role') != 'patient':
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    cursor = conn.cursor()
    cursor.execute('UPDATE appointments SET status = "Cancelled" WHERE id = ? AND patient_email = ?', (appt_id, session['email']))
    conn.commit()
    conn.close()
    flash('Appointment cancelled', 'info')
    return redirect('/patient/dashboard')

# ============================================================
# PATIENT PROFILE - FIX #1 (SA ID) + FIX #6 (allergies)
# ============================================================

@app.route('/patient/profile', methods=['GET', 'POST'])
@require_terms_acceptance
def patient_profile():
    if 'user_id' not in session or session.get('role') != 'patient':
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if request.method == 'POST':
        age = request.form.get('age', 30)
        health_conditions = request.form.get('health_conditions', '').strip()
        location = request.form.get('location', '').strip()
        language = request.form.get('language', 'en')
        phone = request.form.get('phone', '').strip()
        gender = request.form.get('gender', '').strip()
        id_number = request.form.get('id_number', '').strip()
        # ===== FIX #6: allergies saved =====
        allergies = request.form.get('allergies', '').strip()
        # ===== END FIX #6 =====
        
        # ===== FIX #1: SA ID validation =====
        if id_number:
            valid, msg = validate_sa_id(id_number)
            if not valid:
                flash(f'ID Number: {msg}', 'error')
                return redirect('/patient/profile')
        # ===== END FIX #1 =====
        
        try:
            age = int(age)
        except:
            flash('Please enter a valid age', 'error')
            return redirect('/patient/profile')
        
        cursor.execute('''
        UPDATE users 
        SET age = ?, health_conditions = ?, location = ?, 
            language = ?, phone = ?, gender = ?, id_number = ?,
            allergies = ?
        WHERE id = ?
        ''', (age, health_conditions, location, language, phone, gender, id_number, allergies, session['user_id']))
        conn.commit()
        conn.close()
        
        session['age'] = age
        session['health_conditions'] = health_conditions
        session['location'] = location
        session['language'] = language
        session['gender'] = gender
        session['id_number'] = id_number
        
        flash('Profile updated successfully', 'success')
        return redirect('/patient/dashboard')
    
    cursor.execute('SELECT id, email, full_name, phone, age, health_conditions, location, gender, id_number, allergies, emergency_name, emergency_phone FROM users WHERE id = ?', (session['user_id'],))
    patient = cursor.fetchone()
    conn.close()
    lang = session.get('language', 'en')
    return render_template('patient_profile.html', 
        user=session, 
        patient=patient,
        lang=lang,
        translate_text=translate_text
    )

# ============================================================
# GOOGLE MAPS INTEGRATION
# ============================================================

@app.route('/get-clinic-location/<clinic_name>')
def get_clinic_location(clinic_name):
    if df_clinics is not None:
        try:
            row = df_clinics[df_clinics['Clinic_Name'].str.contains(clinic_name, case=False, na=False)].iloc[0]
            return jsonify({
                'name': row.get('Clinic_Name', ''),
                'lat': float(row.get('Latitude', 0)) if row.get('Latitude') else 0,
                'lng': float(row.get('Longitude', 0)) if row.get('Longitude') else 0,
                'address': row.get('Address', ''),
                'phone': row.get('Phone', ''),
                'city': row.get('City', ''),
                'area': row.get('Area', '')
            })
        except:
            pass
    return jsonify({'error': 'Clinic not found'}), 404

@app.route('/patient/clinics', methods=['GET', 'POST'])
@require_terms_acceptance
def patient_clinics():
    if 'user_id' not in session or session.get('role') != 'patient':
        return redirect('/login')
    
    search_location = ""
    search_results = None
    today = datetime.now().strftime('%Y-%m-%d')
    
    if request.method == 'POST':
        search_location = request.form.get('location', '').strip()
        if search_location and df_clinics is not None:
            try:
                df = df_clinics.copy()
                for col in ['Province', 'District', 'City', 'Area', 'Clinic_Name']:
                    if col in df.columns:
                        df[col] = df[col].astype(str).str.lower()
                
                search_lower = search_location.lower()
                mask = (
                    df['Province'].str.contains(search_lower, na=False) |
                    df['District'].str.contains(search_lower, na=False) |
                    df['City'].str.contains(search_lower, na=False) |
                    df['Area'].str.contains(search_lower, na=False) |
                    df['Clinic_Name'].str.contains(search_lower, na=False)
                )
                results = df[mask].head(50)
                search_results = results.to_dict('records')
                
                # ===== FIX #8: spell correction for clinic search =====
                if not search_results:
                    corrected = correct_spelling(search_location)
                    if corrected and corrected != search_location.lower():
                        flash(f'No results for "{search_location}". Showing results for "{corrected}".', 'info')
                        search_location = corrected
                        search_lower = corrected.lower()
                        mask = (
                            df['Province'].str.contains(search_lower, na=False) |
                            df['District'].str.contains(search_lower, na=False) |
                            df['City'].str.contains(search_lower, na=False) |
                            df['Area'].str.contains(search_lower, na=False) |
                            df['Clinic_Name'].str.contains(search_lower, na=False)
                        )
                        search_results = df[mask].head(50).to_dict('records')
                # ===== END FIX #8 =====
                
                for clinic in search_results:
                    clinic_name = str(clinic.get('Clinic_Name', '')).replace(' ', '+')
                    city = str(clinic.get('City', '')).replace(' ', '+')
                    province = str(clinic.get('Province', '')).replace(' ', '+')
                    query = f"{clinic_name}+{city}+{province}+South+Africa"
                    clinic['maps_url'] = f"https://www.google.com/maps/search/?api=1&query={query}"
                    clinic['directions_url'] = f"https://www.google.com/maps/dir/?api=1&destination={query}"
            except Exception as e:
                print(f"Clinic search error: {e}")
    
    lang = session.get('language', 'en')
    return render_template('patient_clinics.html',
        user=session,
        clinics=search_results,
        search_location=search_location,
        today=today,
        lang=lang,
        translate_text=translate_text
    )

@app.route('/patient/book-from-clinic', methods=['POST'])
@require_terms_acceptance
def book_from_clinic():
    if 'user_id' not in session or session.get('role') != 'patient':
        return redirect('/login')
    
    clinic = request.form.get('clinic', '').strip()
    date = request.form.get('appointment_date', '').strip()
    time = request.form.get('appointment_time', '').strip()
    reason = request.form.get('reason', '').strip()
    
    if not all([clinic, date, time, reason]):
        flash('Please fill in all fields', 'error')
        return redirect('/patient/clinics')
    
    today = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H:%M')
    
    try:
        if date < today:
            flash('Cannot book appointments for past dates.', 'error')
            return redirect('/patient/clinics')
        if date == today and time <= current_time:
            flash('Cannot book an appointment time that has already passed today.', 'error')
            return redirect('/patient/clinics')
    except:
        flash('Invalid date or time format.', 'error')
        return redirect('/patient/clinics')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id FROM appointments 
    WHERE clinic_name = ? AND appointment_date = ? AND appointment_time = ? 
    AND status != 'Cancelled'
    ''', (clinic, date, time))
    existing_appt = cursor.fetchone()
    
    if existing_appt:
        conn.close()
        flash('This time slot is already booked.', 'error')
        return redirect('/patient/clinics')
    
    cursor.execute('''
    SELECT id FROM unavailable_slots 
    WHERE clinic_name = ? AND appointment_date = ? AND appointment_time = ?
    ''', (clinic, date, time))
    unavailable_slot = cursor.fetchone()
    
    if unavailable_slot:
        conn.close()
        flash('This time slot is unavailable.', 'error')
        return redirect('/patient/clinics')
    
    cursor.execute('''
    INSERT INTO appointments (patient_email, patient_name, patient_phone, clinic_name, appointment_date, appointment_time, reason, status)
    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    ''', (session['email'], session['full_name'], '', clinic, date, time, reason, 'Scheduled'))
    conn.commit()
    conn.close()
    
    flash(f'✅ Appointment booked at {clinic} on {date} at {time}', 'success')
    return redirect('/patient/clinics')

@app.route('/patient/statistics')
@require_terms_acceptance
def patient_statistics():
    if 'user_id' not in session or session.get('role') != 'patient':
        return redirect('/login')
    return redirect('/patient/dashboard')

@app.route('/patient/no-show-history')
@require_terms_acceptance
def patient_no_show_history():
    if 'user_id' not in session or session.get('role') != 'patient':
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('SELECT id, email, full_name, phone, age, health_conditions, location FROM users WHERE id = ?', (session['user_id'],))
    patient = cursor.fetchone()
    cursor.execute('SELECT id, clinic_name, appointment_date, appointment_time, status, reason, created_at FROM appointments WHERE patient_email = ? AND status = "No-Show" ORDER BY appointment_date DESC', (session['email'],))
    no_show_appointments = cursor.fetchall()
    conn.close()
    
    no_show_prediction = predict_no_show(session['email'])
    
    lang = session.get('language', 'en')
    return render_template('patient_no_show_history.html',
        user=session,
        patient=patient,
        no_show_appointments=no_show_appointments,
        no_show_count=len(no_show_appointments),
        total_appointments=0,
        no_show_rate=0,
        risk_score=no_show_prediction['risk'],
        risk_category=no_show_prediction['category'],
        no_show_prediction=no_show_prediction,
        lang=lang,
        translate_text=translate_text
    )

# ============================================================
# PATIENT - MEDICINE SEARCH - FIX #8 (spell)
# ============================================================

@app.route('/patient/medicines', methods=['GET', 'POST'])
@require_terms_acceptance
def patient_medicines():
    if 'user_id' not in session or session.get('role') != 'patient':
        return redirect('/login')
    
    search_results = []
    search_term = ''
    search_mode = 'disease'
    search_performed = False
    
    if request.method == 'POST':
        search_term = request.form.get('search_term', '').strip()
        search_mode = request.form.get('search_mode', 'disease')
        
        if search_term:
            search_performed = True
            if search_mode == 'disease':
                search_results = search_medicines_by_disease(search_term)
            else:
                search_results = search_diseases_by_medicine(search_term)
            
            # ===== FIX #8: spell correction =====
            if not search_results:
                corrected = correct_spelling(search_term)
                if corrected:
                    flash(f'No results for "{search_term}". Did you mean "{corrected}"?', 'info')
                    if search_mode == 'disease':
                        search_results = search_medicines_by_disease(corrected)
                    else:
                        search_results = search_diseases_by_medicine(corrected)
            # ===== END FIX #8 =====
    
    lang = session.get('language', 'en')
    return render_template('patient_medicines.html',
        user=session,
        search_results=search_results,
        search_term=search_term,
        search_mode=search_mode,
        search_performed=search_performed,
        lang=lang,
        translate_text=translate_text
    )

# ============================================================
# PATIENT - FIRST AID
# ============================================================

@app.route('/patient/first-aid')
@require_terms_acceptance
def patient_first_aid():
    if 'user_id' not in session or session.get('role') != 'patient':
        return redirect('/login')
    
    first_aid_data = []
    if df_firstaid is not None:
        try:
            for _, row in df_firstaid.iterrows():
                q = row.iloc[0] if len(row) > 0 else ''
                a = row.iloc[1] if len(row) > 1 else ''
                if q and a:
                    first_aid_data.append({'question': str(q)[:200], 'answer': str(a)[:300]})
        except:
            pass
    
    first_aid_categories = [
        {'id': 'heart_attack', 'icon': 'fa-heart-pulse', 'title': 'Heart Attack', 'description': 'Emergency signs & response', 'color': 'danger'},
        {'id': 'stroke', 'icon': 'fa-brain', 'title': 'Stroke', 'description': 'FAST - Face, Arms, Speech, Time', 'color': 'danger'},
        {'id': 'choking', 'icon': 'fa-lungs', 'title': 'Choking', 'description': 'Heimlich maneuver for adults & children', 'color': 'danger'},
        {'id': 'severe_bleeding', 'icon': 'fa-droplet', 'title': 'Severe Bleeding', 'description': 'How to stop heavy bleeding', 'color': 'danger'},
        {'id': 'allergic_reaction', 'icon': 'fa-allergies', 'title': 'Allergic Reaction', 'description': 'Anaphylaxis emergency response', 'color': 'danger'},
        {'id': 'seizure', 'icon': 'fa-bolt', 'title': 'Seizure / Fits', 'description': 'What to do during a seizure', 'color': 'danger'},
        {'id': 'poisoning', 'icon': 'fa-skull-crossbones', 'title': 'Poisoning', 'description': 'Immediate steps for poisoning', 'color': 'danger'},
        {'id': 'drowning', 'icon': 'fa-water', 'title': 'Drowning', 'description': 'Rescue & CPR for drowning', 'color': 'danger'},
        {'id': 'burns', 'icon': 'fa-fire', 'title': 'Burns & Scalds', 'description': 'First aid for burns', 'color': 'warning'},
        {'id': 'fracture', 'icon': 'fa-bone', 'title': 'Fractures & Sprains', 'description': 'Handle broken bones & sprains', 'color': 'warning'},
        {'id': 'heatstroke', 'icon': 'fa-temperature-high', 'title': 'Heat Stroke & Dehydration', 'description': 'Emergency cooling response', 'color': 'warning'},
        {'id': 'hypothermia', 'icon': 'fa-snowflake', 'title': 'Hypothermia', 'description': 'Warming a person safely', 'color': 'warning'},
        {'id': 'diabetic_emergency', 'icon': 'fa-syringe', 'title': 'Diabetic Emergency', 'description': 'Low/high blood sugar response', 'color': 'warning'},
        {'id': 'asthma_attack', 'icon': 'fa-lungs', 'title': 'Asthma Attack', 'description': 'Help someone having an asthma attack', 'color': 'warning'},
        {'id': 'cuts_scrapes', 'icon': 'fa-bandage', 'title': 'Cuts & Scrapes', 'description': 'Clean and dress minor wounds', 'color': 'info'},
        {'id': 'insect_bites', 'icon': 'fa-bug', 'title': 'Insect Bites & Stings', 'description': 'Treatment for bites and stings', 'color': 'info'},
        {'id': 'nosebleed', 'icon': 'fa-nose', 'title': 'Nosebleeds', 'description': 'How to stop a nosebleed', 'color': 'info'},
        {'id': 'headache_migraine', 'icon': 'fa-head-side-virus', 'title': 'Headache & Migraine', 'description': 'Relief for headaches', 'color': 'info'},
        {'id': 'food_poisoning', 'icon': 'fa-utensils', 'title': 'Food Poisoning', 'description': 'Symptoms and home care', 'color': 'info'},
        {'id': 'fever_management', 'icon': 'fa-thermometer', 'title': 'Fever Management', 'description': 'How to manage a fever', 'color': 'info'},
    ]
    
    lang = session.get('language', 'en')
    return render_template('patient_first_aid.html',
        user=session,
        first_aid_data=first_aid_data[:10] if first_aid_data else [],
        first_aid_categories=first_aid_categories,
        lang=lang,
        translate_text=translate_text
    )

# ============================================================
# STAFF ROUTES
# ============================================================

@app.route('/staff/dashboard')
@require_terms_acceptance
def staff_dashboard():
    if 'user_id' not in session or session.get('role') not in ['nurse', 'staff']:
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT staff_clinic FROM users WHERE id = ?', (session['user_id'],))
    user = cursor.fetchone()
    conn.close()
    
    staff_clinic = user['staff_clinic'] if user else ''
    
    if not staff_clinic:
        flash('Please select your clinic first', 'warning')
        return redirect(url_for('staff_profile_setup'))
    
    today = datetime.now().strftime('%Y-%m-%d')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT 
        a.id, 
        a.patient_name, 
        a.patient_email, 
        a.clinic_name, 
        a.appointment_time, 
        a.status,
        COALESCE(u.phone, a.patient_phone, 'No phone') as patient_phone
    FROM appointments a
    LEFT JOIN users u ON LOWER(a.patient_email) = LOWER(u.email)
    WHERE a.appointment_date = ? AND a.clinic_name = ? AND a.status != 'Cancelled'
    ORDER BY a.appointment_time
    ''', (today, staff_clinic))
    appointments = cursor.fetchall()
    conn.close()
    
    lang = session.get('language', 'en')
    return render_template('staff_dashboard.html',
        user=session,
        appointments=appointments,
        today_count=len(appointments),
        waiting=sum(1 for a in appointments if a['status'] == 'Checked-in'),
        completed=sum(1 for a in appointments if a['status'] == 'Completed'),
        no_shows=sum(1 for a in appointments if a['status'] == 'No-Show'),
        high_risk_patients=[],
        high_risk_count=0,
        search_query='',
        search_results=None,
        staff_clinic=staff_clinic,
        today=today,
        lang=lang,
        translate_text=translate_text
    )

@app.route('/staff/checkin/<int:appt_id>', methods=['POST'])
@require_terms_acceptance
def staff_checkin(appt_id):
    if 'user_id' not in session or session.get('role') not in ['nurse', 'staff', 'admin']:
        return redirect('/login')
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    cursor = conn.cursor()
    cursor.execute('UPDATE appointments SET status = "Checked-in" WHERE id = ?', (appt_id,))
    conn.commit()
    conn.close()
    return redirect('/staff/dashboard')

@app.route('/staff/checkout/<int:appt_id>', methods=['POST'])
@require_terms_acceptance
def staff_checkout(appt_id):
    if 'user_id' not in session or session.get('role') not in ['nurse', 'staff', 'admin']:
        return redirect('/login')
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    cursor = conn.cursor()
    cursor.execute('UPDATE appointments SET status = "Completed" WHERE id = ?', (appt_id,))
    conn.commit()
    conn.close()
    return redirect('/staff/dashboard')

@app.route('/staff/noshow/<int:appt_id>', methods=['POST'])
@require_terms_acceptance
def staff_noshow(appt_id):
    if 'user_id' not in session or session.get('role') not in ['nurse', 'staff', 'admin']:
        return redirect('/login')
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    cursor = conn.cursor()
    cursor.execute('UPDATE appointments SET status = "No-Show" WHERE id = ?', (appt_id,))
    conn.commit()
    conn.close()
    return redirect('/staff/dashboard')

@app.route('/staff/send-reminder/<int:appt_id>', methods=['POST'])
def staff_send_reminder(appt_id):
    flash('Reminder sent', 'success')
    return redirect('/staff/dashboard')

@app.route('/staff/manual-book', methods=['GET', 'POST'])
@require_terms_acceptance
def staff_manual_book():
    if 'user_id' not in session or session.get('role') not in ['nurse', 'staff']:
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT staff_clinic FROM users WHERE id = ?', (session['user_id'],))
    user = cursor.fetchone()
    conn.close()
    
    staff_clinic = user['staff_clinic'] if user else ''
    if not staff_clinic:
        flash('Please select your clinic first', 'warning')
        return redirect(url_for('staff_profile_setup'))
    
    today = datetime.now().strftime('%Y-%m-%d')
    clinics = get_clinics_list()
    
    if request.method == 'POST':
        patient_name = request.form.get('patient_name', '').strip()
        patient_phone = request.form.get('patient_phone', '').strip()
        clinic = request.form.get('clinic', staff_clinic)
        date = request.form.get('appointment_date')
        time = request.form.get('appointment_time')
        reason = request.form.get('reason', 'Walk-in appointment')
        
        if not all([patient_name, patient_phone, clinic, date, time]):
            flash('Please fill in all required fields', 'error')
            lang = session.get('language', 'en')
            return render_template('staff_manual_book.html', today=today, clinics=clinics, staff_clinic=staff_clinic, lang=lang, translate_text=translate_text)
        
        # ===== FIX #10: use DB_PATH =====
        conn = sqlite3.connect(DB_PATH)
        # ===== END FIX #10 =====
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT id FROM appointments 
        WHERE clinic_name = ? AND appointment_date = ? AND appointment_time = ? 
        AND status != 'Cancelled'
        ''', (clinic, date, time))
        existing_appt = cursor.fetchone()
        
        if existing_appt:
            conn.close()
            flash('This time slot is already booked.', 'error')
            lang = session.get('language', 'en')
            return render_template('staff_manual_book.html', today=today, clinics=clinics, staff_clinic=staff_clinic, lang=lang, translate_text=translate_text)
        
        cursor.execute('''
        SELECT id FROM unavailable_slots 
        WHERE clinic_name = ? AND appointment_date = ? AND appointment_time = ?
        ''', (clinic, date, time))
        unavailable_slot = cursor.fetchone()
        
        if unavailable_slot:
            conn.close()
            flash('This time slot is marked as unavailable.', 'error')
            lang = session.get('language', 'en')
            return render_template('staff_manual_book.html', today=today, clinics=clinics, staff_clinic=staff_clinic, lang=lang, translate_text=translate_text)
        
        cursor.execute('''
        INSERT INTO appointments (patient_name, patient_phone, clinic_name, appointment_date, appointment_time, reason, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (patient_name, patient_phone, clinic, date, time, reason, 'Scheduled'))
        conn.commit()
        conn.close()
        
        flash(f'✅ Appointment booked for {patient_name} at {clinic} on {date} at {time}', 'success')
        return redirect('/staff/dashboard')
    
    lang = session.get('language', 'en')
    return render_template('staff_manual_book.html', 
        user=session, 
        today=today, 
        clinics=clinics, 
        staff_clinic=staff_clinic,
        lang=lang,
        translate_text=translate_text
    )

@app.route('/staff/manage-slots', methods=['GET', 'POST'])
@require_terms_acceptance
def staff_manage_slots():
    if 'user_id' not in session or session.get('role') not in ['nurse', 'staff']:
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT staff_clinic FROM users WHERE id = ?', (session['user_id'],))
    user = cursor.fetchone()
    conn.close()
    
    staff_clinic = user['staff_clinic'] if user else ''
    if not staff_clinic:
        flash('Please select your clinic first', 'warning')
        return redirect(url_for('staff_profile_setup'))
    
    today = datetime.now().strftime('%Y-%m-%d')
    selected_date = request.args.get('date', today)
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    cursor = conn.cursor()
    
    if request.method == 'POST':
        action = request.form.get('action')
        date = request.form.get('date')
        time = request.form.get('time')
        reason = request.form.get('reason', 'Staff unavailable')
        
        if action == 'add':
            cursor.execute('''
            INSERT OR IGNORE INTO unavailable_slots (clinic_name, appointment_date, appointment_time, reason, created_by)
            VALUES (?, ?, ?, ?, ?)
            ''', (staff_clinic, date, time, reason, session['user_id']))
            conn.commit()
            flash(f'Time slot {time} on {date} marked as unavailable', 'success')
        
        elif action == 'remove':
            cursor.execute('''
            DELETE FROM unavailable_slots 
            WHERE clinic_name = ? AND appointment_date = ? AND appointment_time = ?
            ''', (staff_clinic, date, time))
            conn.commit()
            flash(f'Time slot {time} on {date} is now available', 'success')
    
    cursor.execute('''
    SELECT appointment_time, patient_name, patient_phone 
    FROM appointments 
    WHERE clinic_name = ? AND appointment_date = ? AND status != 'Cancelled'
    ORDER BY appointment_time
    ''', (staff_clinic, selected_date))
    booked_rows = cursor.fetchall()
    
    booked_slots = []
    for row in booked_rows:
        booked_slots.append({
            'appointment_time': row[0],
            'patient_name': row[1],
            'patient_phone': row[2]
        })
    
    cursor.execute('''
    SELECT appointment_time, reason 
    FROM unavailable_slots 
    WHERE clinic_name = ? AND appointment_date = ?
    ORDER BY appointment_time
    ''', (staff_clinic, selected_date))
    unavailable_rows = cursor.fetchall()
    
    unavailable_slots = []
    for row in unavailable_rows:
        unavailable_slots.append({
            'appointment_time': row[0],
            'reason': row[1]
        })
    
    conn.close()
    
    time_slots = []
    for hour in range(8, 17):
        for minute in ['00', '30']:
            time_slots.append(f"{hour:02d}:{minute}")
    
    lang = session.get('language', 'en')
    return render_template('staff_manage_slots.html',
        user=session,
        staff_clinic=staff_clinic,
        selected_date=selected_date,
        time_slots=time_slots,
        booked_slots=booked_slots,
        unavailable_slots=unavailable_slots,
        today=today,
        lang=lang,
        translate_text=translate_text
    )

# ============================================================
# STAFF - SEARCH PATIENTS - FIX #6 (allergies)
# ============================================================

@app.route('/staff/search', methods=['GET'])
@require_terms_acceptance
def staff_search():
    if 'user_id' not in session or session.get('role') not in ['nurse', 'staff']:
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT staff_clinic FROM users WHERE id = ?', (session['user_id'],))
    user = cursor.fetchone()
    conn.close()
    
    staff_clinic = user['staff_clinic'] if user else ''
    
    if not staff_clinic:
        flash('Please select your clinic first', 'warning')
        return redirect(url_for('staff_profile_setup'))
    
    query = request.args.get('q', '').strip()
    search_results = []
    search_performed = False
    
    if query:
        search_performed = True
        # ===== FIX #10: use DB_PATH =====
        conn = sqlite3.connect(DB_PATH)
        # ===== END FIX #10 =====
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # ===== FIX #6: allergies included =====
        cursor.execute('''
        SELECT id, email, full_name, phone, role, age, health_conditions, location, 
               staff_clinic, is_active, created_at, gender, id_number, allergies
        FROM users 
        WHERE role = 'patient' 
        AND (full_name LIKE ? OR phone LIKE ? OR email LIKE ?)
        ORDER BY full_name
        ''', ('%' + query + '%', '%' + query + '%', '%' + query + '%'))
        # ===== END FIX #6 =====
        
        patients = cursor.fetchall()
        conn.close()
        
        for patient in patients:
            # ===== FIX #10: use DB_PATH =====
            conn = sqlite3.connect(DB_PATH)
            # ===== END FIX #10 =====
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            cursor.execute('''
            SELECT COUNT(*) as total, 
                   SUM(CASE WHEN status = 'Completed' THEN 1 ELSE 0 END) as completed,
                   SUM(CASE WHEN status = 'No-Show' THEN 1 ELSE 0 END) as no_shows,
                   SUM(CASE WHEN status = 'Cancelled' THEN 1 ELSE 0 END) as cancelled
            FROM appointments 
            WHERE patient_email = ?
            ''', (patient['email'],))
            
            stats = cursor.fetchone()
            conn.close()
            
            health_count = len([c for c in (patient['health_conditions'] or '').split(',') if c.strip()])
            age = patient['age'] or 30
            no_show_history = stats['no_shows'] or 0
            
            risk_score = 15
            if age < 30: risk_score += 10
            elif age > 60: risk_score += 5
            if health_count > 1: risk_score += 15
            elif health_count > 0: risk_score += 10
            if no_show_history > 0: risk_score += 20 * min(no_show_history, 3)
            if health_count >= 3: risk_score += 10
            risk_score = min(risk_score, 95)
            
            if risk_score < 30:
                risk_category = 'Low'
            elif risk_score < 50:
                risk_category = 'Medium'
            elif risk_score < 70:
                risk_category = 'High'
            else:
                risk_category = 'Very High'
            
            # ===== FIX #6: allergies in results =====
            search_results.append({
                'id': patient['id'],
                'full_name': patient['full_name'],
                'email': patient['email'],
                'phone': patient['phone'] or 'N/A',
                'age': age,
                'gender': patient['gender'] or 'N/A',
                'id_number': patient['id_number'] or 'N/A',
                'allergies': patient['allergies'] or '',
                'health_conditions': patient['health_conditions'] or 'None',
                'location': patient['location'] or 'N/A',
                'is_active': patient['is_active'],
                'total_appointments': stats['total'] or 0,
                'completed': stats['completed'] or 0,
                'no_shows': stats['no_shows'] or 0,
                'cancelled': stats['cancelled'] or 0,
                'risk_score': risk_score,
                'risk_category': risk_category,
                'created_at': patient['created_at']
            })
            # ===== END FIX #6 =====
    
    lang = session.get('language', 'en')
    return render_template('staff_search_results.html',
        user=session,
        search_query=query,
        search_results=search_results,
        search_performed=search_performed,
        staff_clinic=staff_clinic,
        lang=lang,
        translate_text=translate_text
    )

# ============================================================
# STAFF - EXPORT
# ============================================================

@app.route('/staff/export')
@require_terms_acceptance
def staff_export():
    if 'user_id' not in session or session.get('role') not in ['nurse', 'staff']:
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT staff_clinic FROM users WHERE id = ?', (session['user_id'],))
    user = cursor.fetchone()
    conn.close()
    
    staff_clinic = user['staff_clinic'] if user else ''
    
    if not staff_clinic:
        flash('Please select your clinic first', 'warning')
        return redirect(url_for('staff_profile_setup'))
    
    export_type = request.args.get('type', 'today')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if export_type == 'today':
        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute('''
        SELECT patient_name, patient_phone, appointment_time, status, reason, created_at
        FROM appointments 
        WHERE clinic_name = ? AND appointment_date = ?
        ORDER BY appointment_time
        ''', (staff_clinic, today))
        appointments = cursor.fetchall()
        filename = f"appointments_{staff_clinic.replace(' ', '_')}_{today}.csv"
    else:
        cursor.execute('''
        SELECT patient_name, patient_phone, appointment_date, appointment_time, status, reason, created_at
        FROM appointments 
        WHERE clinic_name = ?
        ORDER BY appointment_date DESC, appointment_time
        ''', (staff_clinic,))
        appointments = cursor.fetchall()
        filename = f"all_appointments_{staff_clinic.replace(' ', '_')}.csv"
    
    conn.close()
    
    if not appointments:
        flash('No appointments found to export.', 'warning')
        return redirect(url_for('staff_dashboard'))
    
    output = StringIO()
    writer = csv.writer(output)
    
    writer.writerow(['Patient Name', 'Phone', 'Date', 'Time', 'Status', 'Reason', 'Booked On'])
    
    for appt in appointments:
        writer.writerow([
            appt['patient_name'],
            appt['patient_phone'] or 'N/A',
            appt['appointment_date'] if 'appointment_date' in appt.keys() else 'Today',
            appt['appointment_time'],
            appt['status'],
            appt['reason'] or 'N/A',
            appt['created_at'][:10] if appt['created_at'] else 'N/A'
        ])
    
    response = Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename={filename}',
            'Content-Type': 'text/csv'
        }
    )
    
    return response

@app.route('/staff/export-patients')
@require_terms_acceptance
def staff_export_patients():
    if 'user_id' not in session or session.get('role') not in ['nurse', 'staff']:
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT staff_clinic FROM users WHERE id = ?', (session['user_id'],))
    user = cursor.fetchone()
    conn.close()
    
    staff_clinic = user['staff_clinic'] if user else ''
    
    if not staff_clinic:
        flash('Please select your clinic first', 'warning')
        return redirect(url_for('staff_profile_setup'))
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT DISTINCT u.id, u.full_name, u.email, u.phone, u.age, u.health_conditions, u.location,
           (SELECT COUNT(*) FROM appointments WHERE patient_email = u.email AND clinic_name = ?) as total_visits
    FROM users u
    INNER JOIN appointments a ON LOWER(a.patient_email) = LOWER(u.email)
    WHERE u.role = 'patient' AND a.clinic_name = ?
    ORDER BY u.full_name
    ''', (staff_clinic, staff_clinic))
    
    patients = cursor.fetchall()
    conn.close()
    
    if not patients:
        flash('No patients found for this clinic.', 'warning')
        return redirect(url_for('staff_dashboard'))
    
    output = StringIO()
    writer = csv.writer(output)
    
    writer.writerow(['Patient Name', 'Email', 'Phone', 'Age', 'Health Conditions', 'Location', 'Total Visits'])
    
    for patient in patients:
        writer.writerow([
            patient['full_name'],
            patient['email'],
            patient['phone'] or 'N/A',
            patient['age'] or 'N/A',
            patient['health_conditions'] or 'None',
            patient['location'] or 'N/A',
            patient['total_visits'] or 0
        ])
    
    filename = f"patients_{staff_clinic.replace(' ', '_')}.csv"
    
    response = Response(
        output.getvalue(),
        mimetype='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename={filename}',
            'Content-Type': 'text/csv'
        }
    )
    
    return response

# ============================================================
# STAFF - PATIENT PROFILE - FIX #6 (allergies + emergency)
# ============================================================

@app.route('/staff/patient/<int:patient_id>')
@require_terms_acceptance
def staff_patient_history(patient_id):
    if 'user_id' not in session or session.get('role') not in ['nurse', 'staff', 'admin']:
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # ===== FIX #6: allergies + emergency fields included =====
    cursor.execute('''
    SELECT id, email, full_name, phone, role, age, health_conditions, location, 
           is_active, created_at, gender, id_number, allergies,
           emergency_name, emergency_phone, emergency_relationship
    FROM users 
    WHERE id = ? AND role = 'patient'
    ''', (patient_id,))
    # ===== END FIX #6 =====
    patient = cursor.fetchone()
    
    if not patient:
        flash('Patient not found.', 'error')
        conn.close()
        return redirect(url_for('staff_dashboard'))
    
    cursor.execute('''
    SELECT id, clinic_name, appointment_date, appointment_time, status, reason, created_at
    FROM appointments 
    WHERE patient_email = ?
    ORDER BY appointment_date DESC, appointment_time DESC
    ''', (patient['email'],))
    appointments = cursor.fetchall()
    conn.close()
    
    total = len(appointments)
    completed = sum(1 for a in appointments if a['status'] == 'Completed')
    no_shows = sum(1 for a in appointments if a['status'] == 'No-Show')
    cancelled = sum(1 for a in appointments if a['status'] == 'Cancelled')
    
    no_show_rate = round((no_shows / total * 100) if total > 0 else 0, 1)
    
    age = patient['age'] if patient['age'] else 30
    health_conditions = patient['health_conditions'] if patient['health_conditions'] else ''
    health_count = len([c for c in health_conditions.split(',') if c.strip()]) if health_conditions else 0
    
    risk_score = 15
    if age < 30: risk_score += 10
    elif age > 60: risk_score += 5
    if health_count > 1: risk_score += 15
    elif health_count > 0: risk_score += 10
    if no_shows > 0: risk_score += 20 * min(no_shows, 3)
    if health_count >= 3: risk_score += 10
    risk_score = min(risk_score, 95)
    
    if risk_score < 30:
        risk_category = 'Low'
    elif risk_score < 50:
        risk_category = 'Medium'
    elif risk_score < 70:
        risk_category = 'High'
    else:
        risk_category = 'Very High'
    
    health_conditions_list = [c.strip() for c in health_conditions.split(',') if c.strip()]
    
    no_show_prediction = predict_no_show(patient['email'])
    
    lang = session.get('language', 'en')
    return render_template('staff_patient_history.html',
        user=session,
        patient=patient,
        appointments=appointments,
        total=total,
        completed=completed,
        no_shows=no_shows,
        cancelled=cancelled,
        no_show_rate=no_show_rate,
        age=age,
        risk_score=risk_score,
        risk_category=risk_category,
        health_conditions_list=health_conditions_list,
        health_count=health_count,
        no_show_prediction=no_show_prediction,
        lang=lang,
        translate_text=translate_text
    )

# ============================================================
# ADMIN - DASHBOARD
# ============================================================

@app.route('/admin/dashboard')
@require_terms_acceptance
def admin_dashboard():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'patient'")
    total_patients = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM users WHERE role IN ('nurse', 'staff', 'admin')")
    total_staff = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM users WHERE is_active = 1")
    active_users = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM appointments")
    total_appointments = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM appointments WHERE status = 'Scheduled'")
    scheduled = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM appointments WHERE status = 'Completed'")
    completed = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM appointments WHERE status = 'Cancelled'")
    cancelled = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM appointments WHERE status = 'No-Show'")
    no_show = cursor.fetchone()[0]
    
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute("SELECT COUNT(*) FROM appointments WHERE appointment_date = ?", (today,))
    today_appointments = cursor.fetchone()[0]
    
    daily_labels = []
    daily_counts = []
    for i in range(6, -1, -1):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        day_name = (datetime.now() - timedelta(days=i)).strftime('%a')
        daily_labels.append(day_name)
        cursor.execute('SELECT COUNT(*) FROM appointments WHERE appointment_date = ?', (date,))
        count = cursor.fetchone()[0]
        daily_counts.append(count)
    
    cursor.execute('''
    SELECT clinic_name, COUNT(*) as count 
    FROM appointments 
    GROUP BY clinic_name 
    ORDER BY count DESC 
    LIMIT 5
    ''')
    clinic_stats = cursor.fetchall()
    clinic_labels = [row['clinic_name'][:15] for row in clinic_stats]
    clinic_counts = [row['count'] for row in clinic_stats]
    
    cursor.execute('''
    SELECT full_name, email, role, created_at 
    FROM users 
    ORDER BY created_at DESC 
    LIMIT 5
    ''')
    recent_users = cursor.fetchall()
    
    no_show_rate = round((no_show / total_appointments * 100) if total_appointments > 0 else 0, 1)
    total_clinics = len(get_clinics_list())
    
    conn.close()
    
    lang = session.get('language', 'en')
    return render_template('admin_dashboard.html',
        user=session,
        total_users=total_users,
        total_patients=total_patients,
        total_staff=total_staff,
        total_appointments=total_appointments,
        today_appointments=today_appointments,
        scheduled=scheduled,
        completed=completed,
        cancelled=cancelled,
        no_show=no_show,
        no_show_rate=no_show_rate,
        active_users=active_users,
        daily_labels=json.dumps(daily_labels),
        daily_counts=json.dumps(daily_counts),
        status_counts=json.dumps([scheduled, completed, cancelled, no_show]),
        clinic_labels=json.dumps(clinic_labels),
        clinic_counts=json.dumps(clinic_counts),
        recent_users=recent_users,
        total_clinics=total_clinics,
        lang=lang,
        translate_text=translate_text
    )

@app.route('/admin/users')
@require_terms_acceptance
def admin_users():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT id, full_name, email, phone, role, is_active, is_verified, created_at, gender, id_number FROM users ORDER BY created_at DESC')
    users = cursor.fetchall()
    conn.close()
    
    lang = session.get('language', 'en')
    return render_template('admin_users.html',
        user=session,
        users=users,
        lang=lang,
        translate_text=translate_text
    )

@app.route('/admin/user/add', methods=['GET', 'POST'])
@require_terms_acceptance
def admin_user_add():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')
    
    error = None
    lang = session.get('language', 'en')
    
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        email = request.form.get('email', '').strip()
        password = request.form.get('password', '').strip()
        phone = request.form.get('phone', '').strip()
        role = request.form.get('role', 'patient').strip()
        staff_code = request.form.get('staff_code', '').strip()
        
        if not full_name or not email or not password:
            error = translate_text('All fields are required.', lang)
            return render_template('admin_user_add.html', user=session, error=error, lang=lang, translate_text=translate_text)
        
        if len(password) < 6:
            error = translate_text('Password must be at least 6 characters.', lang)
            return render_template('admin_user_add.html', user=session, error=error, lang=lang, translate_text=translate_text)
        
        # ===== FIX #10: use DB_PATH =====
        conn = sqlite3.connect(DB_PATH)
        # ===== END FIX #10 =====
        cursor = conn.cursor()
        
        cursor.execute('SELECT id FROM users WHERE email = ?', (email,))
        if cursor.fetchone():
            conn.close()
            error = translate_text('Email already registered.', lang)
            return render_template('admin_user_add.html', user=session, error=error, lang=lang, translate_text=translate_text)
        
        password_hash = hashlib.sha256(password.encode()).hexdigest()
        cursor.execute('''
        INSERT INTO users (email, password_hash, full_name, phone, role, staff_code, is_verified, is_active, terms_accepted)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (email, password_hash, full_name, phone, role, staff_code, 1, 1, 1))
        conn.commit()
        conn.close()
        
        flash(translate_text('User created successfully!', lang), 'success')
        return redirect(url_for('admin_users'))
    
    return render_template('admin_user_add.html',
        user=session,
        error=error,
        lang=lang,
        translate_text=translate_text
    )

@app.route('/admin/user/edit/<int:user_id>', methods=['GET', 'POST'])
@require_terms_acceptance
def admin_user_edit(user_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    if request.method == 'POST':
        full_name = request.form.get('full_name', '').strip()
        phone = request.form.get('phone', '').strip()
        role = request.form.get('role', 'patient').strip()
        is_active = int(request.form.get('is_active', 1))
        gender = request.form.get('gender', '').strip()
        id_number = request.form.get('id_number', '').strip()
        
        # ===== FIX #1: SA ID validation in admin edit =====
        if id_number:
            valid, msg = validate_sa_id(id_number)
            if not valid:
                flash(f'ID Number: {msg}', 'error')
                return redirect(url_for('admin_user_edit', user_id=user_id))
        # ===== END FIX #1 =====
        
        cursor.execute('''
        UPDATE users SET full_name = ?, phone = ?, role = ?, is_active = ?, gender = ?, id_number = ?
        WHERE id = ?
        ''', (full_name, phone, role, is_active, gender, id_number, user_id))
        conn.commit()
        conn.close()
        
        lang = session.get('language', 'en')
        flash(translate_text('User updated successfully!', lang), 'success')
        return redirect(url_for('admin_users'))
    
    cursor.execute('SELECT id, full_name, email, phone, role, is_active, gender, id_number FROM users WHERE id = ?', (user_id,))
    edit_user = cursor.fetchone()
    conn.close()
    
    if not edit_user:
        flash('User not found.', 'error')
        return redirect(url_for('admin_users'))
    
    lang = session.get('language', 'en')
    return render_template('admin_user_edit.html',
        user=session,
        edit_user=edit_user,
        lang=lang,
        translate_text=translate_text
    )

@app.route('/admin/user/delete/<int:user_id>', methods=['POST'])
@require_terms_acceptance
def admin_user_delete(user_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')
    
    if user_id == session['user_id']:
        flash('You cannot delete your own account.', 'error')
        return redirect(url_for('admin_users'))
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    cursor = conn.cursor()
    cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    
    flash('User deleted successfully.', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/user/toggle/<int:user_id>', methods=['POST'])
@require_terms_acceptance
def admin_user_toggle(user_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    cursor = conn.cursor()
    
    cursor.execute('SELECT is_active FROM users WHERE id = ?', (user_id,))
    user = cursor.fetchone()
    if user:
        new_status = 0 if user[0] == 1 else 1
        cursor.execute('UPDATE users SET is_active = ? WHERE id = ?', (new_status, user_id))
        conn.commit()
    
    conn.close()
    flash('User status updated.', 'success')
    return redirect(url_for('admin_users'))

@app.route('/admin/staff')
@require_terms_acceptance
def admin_staff():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
    SELECT id, full_name, email, role, staff_code, is_verified, is_active 
    FROM users 
    WHERE role IN ('nurse', 'staff', 'admin')
    ORDER BY created_at DESC
    ''')
    staff_members = cursor.fetchall()
    conn.close()
    
    lang = session.get('language', 'en')
    return render_template('admin_staff.html',
        user=session,
        staff_members=staff_members,
        lang=lang,
        translate_text=translate_text
    )

@app.route('/admin/staff/verify/<int:user_id>', methods=['POST'])
@require_terms_acceptance
def admin_staff_verify(user_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    cursor = conn.cursor()
    cursor.execute('UPDATE users SET is_verified = 1 WHERE id = ?', (user_id,))
    conn.commit()
    conn.close()
    
    flash('Staff member verified successfully.', 'success')
    return redirect(url_for('admin_staff'))

@app.route('/admin/clinics')
@require_terms_acceptance
def admin_clinics():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')
    
    clinics = []
    if df_clinics is not None:
        try:
            clinics = df_clinics.to_dict('records')
        except:
            pass
    
    total_clinics = len(clinics)
    lang = session.get('language', 'en')
    
    return render_template('admin_clinics.html',
        user=session,
        clinics=clinics,
        total_clinics=total_clinics,
        lang=lang,
        translate_text=translate_text
    )

@app.route('/admin/clinic/add', methods=['GET', 'POST'])
@require_terms_acceptance
def admin_clinic_add():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')
    
    lang = session.get('language', 'en')
    
    if request.method == 'POST':
        clinic_name = request.form.get('clinic_name', '').strip()
        city = request.form.get('city', '').strip()
        area = request.form.get('area', '').strip()
        province = request.form.get('province', '').strip()
        phone = request.form.get('phone', '').strip()
        services = request.form.get('services', '').strip()
        
        if not clinic_name or not city:
            flash(translate_text('Clinic name and city are required.', lang), 'error')
            return render_template('admin_clinic_add.html', user=session, lang=lang, translate_text=translate_text)
        
        flash(translate_text(f'Clinic "{clinic_name}" added successfully!', lang), 'success')
        return redirect(url_for('admin_clinics'))
    
    return render_template('admin_clinic_add.html',
        user=session,
        lang=lang,
        translate_text=translate_text
    )

@app.route('/admin/clinic/edit/<int:clinic_id>', methods=['GET', 'POST'])
@require_terms_acceptance
def admin_clinic_edit(clinic_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')
    
    clinic = None
    if df_clinics is not None:
        try:
            clinic = df_clinics[df_clinics['Clinic_ID'] == str(clinic_id)].iloc[0].to_dict()
        except:
            pass
    
    if not clinic:
        flash('Clinic not found.', 'error')
        return redirect(url_for('admin_clinics'))
    
    lang = session.get('language', 'en')
    
    if request.method == 'POST':
        flash(translate_text('Clinic updated successfully!', lang), 'success')
        return redirect(url_for('admin_clinics'))
    
    return render_template('admin_clinic_edit.html',
        user=session,
        clinic=clinic,
        lang=lang,
        translate_text=translate_text
    )

@app.route('/admin/clinic/delete/<int:clinic_id>', methods=['POST'])
@require_terms_acceptance
def admin_clinic_delete(clinic_id):
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')
    
    flash('Clinic deleted successfully.', 'success')
    return redirect(url_for('admin_clinics'))

@app.route('/admin/appointments')
@require_terms_acceptance
def admin_appointments():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('''
    SELECT id, patient_name, patient_phone, clinic_name, appointment_date, appointment_time, status, reason, created_at
    FROM appointments 
    ORDER BY appointment_date DESC, appointment_time DESC
    ''')
    appointments = cursor.fetchall()
    conn.close()
    
    lang = session.get('language', 'en')
    return render_template('admin_appointments.html',
        user=session,
        appointments=appointments,
        lang=lang,
        translate_text=translate_text
    )

@app.route('/admin/reports')
@require_terms_acceptance
def admin_reports():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')
    
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM users")
    total_users = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM users WHERE role = 'patient'")
    total_patients = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM users WHERE role IN ('nurse', 'staff', 'admin')")
    total_staff = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM appointments")
    total_appointments = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM appointments WHERE status = 'Scheduled'")
    scheduled = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM appointments WHERE status = 'Completed'")
    completed = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM appointments WHERE status = 'Cancelled'")
    cancelled = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM appointments WHERE status = 'No-Show'")
    no_shows = cursor.fetchone()[0]
    
    daily_stats = []
    for i in range(7):
        date = (datetime.now() - timedelta(days=i)).strftime('%Y-%m-%d')
        cursor.execute('SELECT COUNT(*) FROM appointments WHERE appointment_date = ?', (date,))
        count = cursor.fetchone()[0]
        daily_stats.append({'date': date, 'count': count})
    
    conn.close()
    
    lang = session.get('language', 'en')
    return render_template('admin_reports.html',
        user=session,
        total_users=total_users,
        total_patients=total_patients,
        total_staff=total_staff,
        total_appointments=total_appointments,
        scheduled=scheduled,
        completed=completed,
        cancelled=cancelled,
        no_shows=no_shows,
        daily_stats=daily_stats,
        lang=lang,
        translate_text=translate_text
    )

@app.route('/admin/activity')
@require_terms_acceptance
def admin_activity():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')
    
    activities = []
    # ===== FIX #10: use DB_PATH =====
    conn = sqlite3.connect(DB_PATH)
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT s.search_term, s.result, s.urgency_level, s.created_at, u.full_name
    FROM symptom_history s
    LEFT JOIN users u ON s.user_id = u.id
    ORDER BY s.created_at DESC
    LIMIT 20
    ''')
    history = cursor.fetchall()
    
    for item in history:
        activities.append({
            'time': item['created_at'][:16] if item['created_at'] else '',
            'user': item['full_name'] or 'Unknown',
            'action': 'Searched: ' + (item['search_term'] or 'Unknown'),
            'details': 'Result: ' + (item['result'] or 'N/A')
        })
    
    conn.close()
    
    lang = session.get('language', 'en')
    return render_template('admin_activity.html',
        user=session,
        activities=activities,
        lang=lang,
        translate_text=translate_text
    )

@app.route('/admin/settings', methods=['GET', 'POST'])
@require_terms_acceptance
def admin_settings():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')
    
    lang = session.get('language', 'en')
    
    if request.method == 'POST':
        flash(translate_text('Settings saved successfully!', lang), 'success')
        return redirect(url_for('admin_settings'))
    
    return render_template('admin_settings.html',
        user=session,
        lang=lang,
        translate_text=translate_text
    )

@app.route('/admin/check-no-shows', methods=['GET'])
@require_terms_acceptance
def admin_check_no_shows():
    if 'user_id' not in session or session.get('role') != 'admin':
        return redirect('/login')
    
    expired = check_expired_appointments()
    flash(f'No-shows marked: {expired}', 'success')
    return redirect('/admin/dashboard')

# ============================================================
# USSD ENDPOINT
# ============================================================

@app.route('/ussd', methods=['GET', 'POST'])
def ussd():
    if request.method == 'GET':
        text = request.args.get('text', '')
    else:
        text = request.values.get('text', '')
    
    parts = text.split('*') if text else []
    level = len(parts)
    
    if text == "":
        return "CON Welcome to MediSense\n1. Book Appointment\n2. Check Symptoms\n3. My Appointments\n4. Find Clinic\n5. Health Info\n6. Emergency"
    
    if parts[0] == "1":
        if level == 1:
            clinics = get_clinics_list()
            response = "CON Select clinic:\n"
            for i, clinic in enumerate(clinics[:5], 1):
                response += f"{i}. {clinic}\n"
            return response
        elif level == 2:
            clinics = get_clinics_list()
            try:
                clinic_index = int(parts[1]) - 1
                if 0 <= clinic_index < len(clinics):
                    session['ussd_clinic'] = clinics[clinic_index]
                    return "CON Select date:\n1. Today\n2. Tomorrow"
            except:
                pass
            return "END Invalid selection. Please try again."
        elif level == 3:
            if parts[2] == "1":
                session['ussd_date'] = datetime.now().strftime('%Y-%m-%d')
            elif parts[2] == "2":
                session['ussd_date'] = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            else:
                return "END Invalid date selection."
            return "CON Select time:\n1. 08:00\n2. 09:00\n3. 10:00\n4. 11:00\n5. 14:00\n6. 15:00"
        elif level == 4:
            time_map = {'1': '08:00', '2': '09:00', '3': '10:00', '4': '11:00', '5': '14:00', '6': '15:00'}
            time = time_map.get(parts[3])
            if not time:
                return "END Invalid time selection."
            clinic = session.get('ussd_clinic', 'Clinic')
            date = session.get('ussd_date', datetime.now().strftime('%Y-%m-%d'))
            return f"END Appointment confirmed!\nClinic: {clinic}\nDate: {date}\nTime: {time}\nReference: USSD-{datetime.now().strftime('%Y%m%d%H%M%S')}"
    
    if parts[0] == "2":
        if level == 1:
            return "CON Describe your symptoms:\nExample: fever, cough, headache\nType your symptoms:"
        elif level == 2:
            symptoms = parts[1] if len(parts) > 1 else ""
            if symptoms:
                possible = []
                if 'fever' in symptoms.lower():
                    possible.append("Flu")
                if 'cough' in symptoms.lower():
                    possible.append("Common Cold")
                if 'headache' in symptoms.lower():
                    possible.append("Migraine")
                if not possible:
                    possible = ["Unknown - Please consult a doctor"]
                return f"END Symptoms detected: {symptoms}\nPossible conditions:\n" + "\n".join([f"• {p}" for p in possible]) + "\n\nPlease consult a healthcare professional."
            else:
                return "END No symptoms detected. Please try again."
    
    if parts[0] == "3":
        # ===== FIX #10: use DB_PATH =====
        conn = sqlite3.connect(DB_PATH)
        # ===== END FIX #10 =====
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''
        SELECT clinic_name, appointment_date, appointment_time, status 
        FROM appointments 
        WHERE appointment_date >= date('now')
        ORDER BY appointment_date, appointment_time
        LIMIT 3
        ''')
        appointments = cursor.fetchall()
        conn.close()
        if appointments:
            response = "END Your Appointments:\n"
            for i, appt in enumerate(appointments, 1):
                response += f"{i}. {appt['clinic_name']}\n   {appt['appointment_date']} {appt['appointment_time']}\n   Status: {appt['status']}\n"
            return response
        else:
            return "END No appointments found."
    
    if parts[0] == "4":
        if level == 1:
            return "CON Enter your location:\nExample: Durban, Cape Town"
        elif level == 2:
            location = parts[1] if len(parts) > 1 else ""
            if location:
                if df_clinics is not None:
                    results = df_clinics[
                        df_clinics['City'].str.lower().str.contains(location.lower(), na=False) |
                        df_clinics['Area'].str.lower().str.contains(location.lower(), na=False)
                    ].head(3)
                    if not results.empty:
                        response = f"END Clinics near {location}:\n"
                        for i, row in results.iterrows():
                            response += f"{i+1}. {row['Clinic_Name']}\n"
                            response += f"   Phone: {row['Phone'] if 'Phone' in row else 'N/A'}\n"
                        return response
                    else:
                        return f"END No clinics found near {location}."
                else:
                    return f"END Clinics near {location}:\n1. Empangeni Clinic - 1.2km\n2. Ngwelezane Clinic - 3.5km"
            else:
                return "END Invalid location."
    
    if parts[0] == "5":
        if level == 1:
            return "CON Enter disease name:\nExample: malaria, flu, diabetes"
        elif level == 2:
            disease = parts[1] if len(parts) > 1 else ""
            if disease:
                return f"END Health Info: {disease}\n\nDescription: A medical condition requiring proper diagnosis.\nConsult a healthcare professional for accurate information."
            else:
                return "END No disease entered."
    
    if parts[0] == "6":
        return "END EMERGENCY CONTACTS\n\nAmbulance: 10177\nPolice: 10111\nNational Emergency: 112\n\nNearest Clinic: Empangeni Clinic\nPhone: 035 123 4567"
    
    return "END Invalid option. Please try again."

# ============================================================
# UPDATE LANGUAGE
# ============================================================

@app.route('/update-language', methods=['POST'])
def update_language():
    if 'user_id' not in session:
        return redirect('/login')
    
    lang = request.form.get('language', 'en')
    if lang in ['en', 'zulu']:
        session['language'] = lang
        # ===== FIX #10: use DB_PATH =====
        conn = sqlite3.connect(DB_PATH)
        # ===== END FIX #10 =====
        cursor = conn.cursor()
        try:
            cursor.execute('UPDATE users SET language = ? WHERE id = ?', (lang, session['user_id']))
            conn.commit()
        except:
            pass
        conn.close()
    
    return redirect(request.referrer or url_for('home'))