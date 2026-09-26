from app import app
import sqlite3
import os
import hashlib
from datetime import datetime, timedelta
import threading
import time

# ===== FIX #10: Database persistence for online hosting (e.g. Render) =====
# ===== FIX: Database persistence for online hosting (e.g. Render) =====
def get_db_dir():
    env_dir = os.environ.get('MEDISENSE_DATA_DIR', '').strip()
    if env_dir:
        os.makedirs(env_dir, exist_ok=True)
        return env_dir
    os.makedirs('database', exist_ok=True)
    return 'database'

DATA_DIR = get_db_dir()
DB_PATH = os.path.join(DATA_DIR, 'medisense_users.db')
# ===== END FIX =====

# ===== FIX: SQLite concurrency — WAL + timeout to prevent hangs =====
def get_db_connection():
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL;")
        conn.execute("PRAGMA busy_timeout=30000;")
        conn.execute("PRAGMA synchronous=NORMAL;")
    except Exception:
        pass
    return conn
# ===== END FIX =====

def init_database():
    """Initialize the database with required tables"""
    # ===== FIX #10: use DB_PATH =====
    conn = get_db_connection()
    # ===== END FIX #10 =====
    cursor = conn.cursor()

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        email TEXT UNIQUE NOT NULL,
        password_hash TEXT NOT NULL,
        full_name TEXT NOT NULL,
        phone TEXT,
        role TEXT NOT NULL,
        staff_code TEXT,
        is_verified INTEGER DEFAULT 0,
        is_active INTEGER DEFAULT 1,
        age INTEGER DEFAULT 30,
        health_conditions TEXT DEFAULT '',
        location TEXT DEFAULT '',
        staff_clinic TEXT DEFAULT '',
        language TEXT DEFAULT 'en',
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        last_login TIMESTAMP,
        reset_token TEXT,
        reset_token_expiry TIMESTAMP
    )
    ''')

    cursor.execute("PRAGMA table_info(users)")
    existing_columns = [col[1] for col in cursor.fetchall()]
    
    if 'age' not in existing_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN age INTEGER DEFAULT 30")
    if 'health_conditions' not in existing_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN health_conditions TEXT DEFAULT ''")
    if 'location' not in existing_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN location TEXT DEFAULT ''")
    if 'staff_clinic' not in existing_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN staff_clinic TEXT DEFAULT ''")
    if 'language' not in existing_columns:
        cursor.execute("ALTER TABLE users ADD COLUMN language TEXT DEFAULT 'en'")

    cursor.execute('''
    CREATE TABLE IF NOT EXISTS appointments (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        patient_email TEXT NOT NULL,
        patient_name TEXT NOT NULL,
        patient_phone TEXT,
        clinic_name TEXT NOT NULL,
        appointment_date TEXT NOT NULL,
        appointment_time TEXT NOT NULL,
        reason TEXT,
        status TEXT DEFAULT 'Scheduled',
        booked_by TEXT DEFAULT 'patient',
        reminder_sent INTEGER DEFAULT 0,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
    ''')
    
    cursor.execute("PRAGMA table_info(appointments)")
    appt_columns = [col[1] for col in cursor.fetchall()]
    if 'booked_by' not in appt_columns:
        cursor.execute("ALTER TABLE appointments ADD COLUMN booked_by TEXT DEFAULT 'patient'")
    if 'reminder_sent' not in appt_columns:
        cursor.execute("ALTER TABLE appointments ADD COLUMN reminder_sent INTEGER DEFAULT 0")
    
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
    
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS roles (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role_name TEXT UNIQUE NOT NULL,
        permissions TEXT
    )
    ''')
    
    cursor.execute("SELECT COUNT(*) FROM roles")
    if cursor.fetchone()[0] == 0:
        roles = [
            ('patient', 'book_appointments,view_history,view_health_info'),
            ('nurse', 'manage_appointments,checkin_patients,view_patients'),
            ('staff', 'manage_appointments,checkin_patients,view_patients'),
            ('admin', 'manage_users,manage_clinics,view_analytics,all_permissions')
        ]
        cursor.executemany("INSERT INTO roles (role_name, permissions) VALUES (?, ?)", roles)
        print("Default roles created")
    
    cursor.execute("SELECT COUNT(*) FROM users WHERE email = 'admin@medisense.com'")
    if cursor.fetchone()[0] == 0:
        password_hash = hashlib.sha256("Admin123!".encode()).hexdigest()
        cursor.execute('''
        INSERT INTO users (email, password_hash, full_name, phone, role, staff_code, is_verified, age, health_conditions, location, staff_clinic, language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', ('admin@medisense.com', password_hash, 'System Administrator', '0821234567', 'admin', 'ADMIN2026', 1, 40, '', 'Johannesburg', '', 'en'))
        print("Admin user created")
    
    cursor.execute("SELECT COUNT(*) FROM users WHERE email = 'patient@medisense.com'")
    if cursor.fetchone()[0] == 0:
        password_hash = hashlib.sha256("Patient123!".encode()).hexdigest()
        cursor.execute('''
        INSERT INTO users (email, password_hash, full_name, phone, role, staff_code, is_verified, age, health_conditions, location, staff_clinic, language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', ('patient@medisense.com', password_hash, 'Test Patient', '0823456789', 'patient', '', 1, 35, 'Hypertension, Diabetes', 'Empangeni', '', 'en'))
        print("Patient user created")
    
    cursor.execute("SELECT COUNT(*) FROM users WHERE email = 'nurse@medisense.com'")
    if cursor.fetchone()[0] == 0:
        password_hash = hashlib.sha256("Nurse123!".encode()).hexdigest()
        cursor.execute('''
        INSERT INTO users (email, password_hash, full_name, phone, role, staff_code, is_verified, age, health_conditions, location, staff_clinic, language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', ('nurse@medisense.com', password_hash, 'Sarah Mkhize', '0822345678', 'nurse', 'NURSE2026', 1, 28, '', 'Empangeni', 'Empangeni Clinic', 'en'))
        print("Nurse user created at Empangeni Clinic")
    
    cursor.execute("SELECT COUNT(*) FROM users WHERE email = 'staff@medisense.com'")
    if cursor.fetchone()[0] == 0:
        password_hash = hashlib.sha256("Staff123!".encode()).hexdigest()
        cursor.execute('''
        INSERT INTO users (email, password_hash, full_name, phone, role, staff_code, is_verified, age, health_conditions, location, staff_clinic, language)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', ('staff@medisense.com', password_hash, 'John Staff', '0824567890', 'staff', 'STAFF2026', 1, 30, '', 'Ngwelezane', 'Ngwelezane Clinic', 'en'))
        print("Staff user created at Ngwelezane Clinic")
    
    today = datetime.now().strftime('%Y-%m-%d')
    cursor.execute("SELECT COUNT(*) FROM appointments WHERE appointment_date = ?", (today,))
    if cursor.fetchone()[0] == 0:
        cursor.execute('''
        INSERT INTO appointments (patient_email, patient_name, patient_phone, clinic_name, appointment_date, appointment_time, reason, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', ('patient@medisense.com', 'Test Patient', '0823456789', 'Empangeni Clinic', today, '09:00', 'General Consultation', 'Scheduled'))
        cursor.execute('''
        INSERT INTO appointments (patient_email, patient_name, patient_phone, clinic_name, appointment_date, appointment_time, reason, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', ('patient@medisense.com', 'Test Patient', '0823456789', 'Ngwelezane Clinic', today, '10:30', 'Check-up', 'Scheduled'))
        cursor.execute('''
        INSERT INTO appointments (patient_email, patient_name, patient_phone, clinic_name, appointment_date, appointment_time, reason, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ''', ('patient@medisense.com', 'Test Patient', '0823456789', 'Richards Bay Clinic', today, '14:00', 'Follow-up', 'Scheduled'))
        print("Sample appointments created")
    
    conn.commit()
    conn.close()
    print("Database initialized successfully")

def check_expired_appointments():
    """Check for appointments that have passed 45 minutes and mark as No-Show"""
    now = datetime.now()
    today = now.strftime('%Y-%m-%d')
    
    # ===== FIX #10: use DB_PATH =====
    conn = get_db_connection()
    # ===== END FIX #10 =====
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    cursor.execute('''
    SELECT id, patient_name, appointment_time
    FROM appointments 
    WHERE appointment_date = ? 
    AND status = 'Scheduled'
    ''', (today,))
    
    appointments = cursor.fetchall()
    expired_count = 0
    
    for appt in appointments:
        try:
            appt_datetime = datetime.strptime(f"{today} {appt['appointment_time']}", '%Y-%m-%d %H:%M')
            minutes_passed = (now - appt_datetime).total_seconds() / 60
            
            if minutes_passed >= 45:
                cursor.execute('''
                UPDATE appointments SET status = 'No-Show' 
                WHERE id = ?
                ''', (appt['id'],))
                conn.commit()
                expired_count += 1
                print(f"⏰ Auto No-Show: {appt['patient_name']} at {appt['appointment_time']} ({minutes_passed:.0f} mins passed)")
        except Exception as e:
            print(f"Error checking appointment {appt['id']}: {e}")
    
    conn.close()
    return expired_count

def run_auto_checker():
    """Run the auto no-show checker every minute"""
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Auto No-Show Checker started")
    while True:
        try:
            expired = check_expired_appointments()
            if expired > 0:
                print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Marked {expired} appointments as No-Show")
        except Exception as e:
            print(f"Error in auto checker: {e}")
        time.sleep(300)

if __name__ == '__main__':
    print("Initializing database...")
    init_database()
    
    print("Starting auto no-show checker...")
    checker_thread = threading.Thread(target=run_auto_checker, daemon=True)
    checker_thread.start()
    
    print("="*60)
    print("MEDISENSE SYSTEM")
    print("="*60)
    print("Running on: http://127.0.0.1:5000")
    print("")
    print("Login Credentials:")
    print("  Admin: admin@medisense.com / Admin123!")
    print("  Patient: patient@medisense.com / Patient123!")
    print("  Nurse: nurse@medisense.com / Nurse123! (Assigned to Empangeni Clinic)")
    print("  Staff: staff@medisense.com / Staff123! (Assigned to Ngwelezane Clinic)")
    print("")
    print("Auto No-Show Checker: Running (checks every 5 minutes)")
    print("="*60)
    
    app.run(debug=True, port=5000, host='0.0.0.0')