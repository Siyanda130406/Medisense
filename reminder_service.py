# reminder_service.py
import sqlite3
import os
from datetime import datetime, timedelta
from sms_gateway import SMSService, EmailService

class ReminderService:
    """Service to send automated appointment reminders"""
    
    def __init__(self):
        self.sms_service = SMSService()
        self.email_service = EmailService()
        self.db_path = os.path.join(os.path.dirname(__file__), 'database', 'medisense_users.db')
    
    def send_daily_reminders(self):
        """
        Send reminders for today's appointments that are 1 hour away
        """
        now = datetime.now()
        reminder_time = (now + timedelta(hours=1)).strftime('%H:%M')
        today = now.strftime('%Y-%m-%d')
        
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Get appointments for today that are 1 hour away and haven't been reminded
        cursor.execute('''
        SELECT a.id, a.patient_name, a.patient_email, a.patient_phone, 
               a.clinic_name, a.appointment_date, a.appointment_time,
               u.phone as user_phone, u.email as user_email
        FROM appointments a
        LEFT JOIN users u ON a.patient_email = u.email
        WHERE a.appointment_date = ? 
        AND a.appointment_time = ? 
        AND a.status = 'Scheduled' 
        AND a.reminder_sent = 0
        ''', (today, reminder_time))
        
        appointments = cursor.fetchall()
        conn.close()
        
        sent_count = 0
        failed_count = 0
        
        for appt in appointments:
            # Get phone number (from appointments or users table)
            phone = appt['patient_phone'] or appt['user_phone']
            email = appt['patient_email'] or appt['user_email']
            patient_name = appt['patient_name']
            
            print(f"\n📨 Sending reminder to {patient_name}")
            
            # Send SMS if phone number exists
            if phone:
                sms_result = self.sms_service.send_reminder(
                    patient_name,
                    phone,
                    appt['clinic_name'],
                    appt['appointment_date'],
                    appt['appointment_time']
                )
                
                if sms_result['status'] == 'success':
                    sent_count += 1
                    print(f"   ✅ SMS sent to {phone}")
                else:
                    failed_count += 1
                    print(f"   ❌ SMS failed: {sms_result.get('message', 'Unknown error')}")
            else:
                print(f"   ⚠️ No phone number for {patient_name}")
            
            # Send Email if email exists
            if email:
                email_result = self.email_service.send_reminder(
                    patient_name,
                    email,
                    appt['clinic_name'],
                    appt['appointment_date'],
                    appt['appointment_time']
                )
                
                if email_result['status'] == 'success':
                    print(f"   ✅ Email sent to {email}")
                else:
                    print(f"   ❌ Email failed: {email_result.get('message', 'Unknown error')}")
            
            # Mark reminder as sent
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('''
            UPDATE appointments SET reminder_sent = 1 WHERE id = ?
            ''', (appt['id'],))
            conn.commit()
            conn.close()
        
        return {
            'total': len(appointments),
            'sent': sent_count,
            'failed': failed_count,
            'date': today,
            'time': reminder_time
        }
    
    def send_no_show_warnings(self):
        """Send warnings to patients with 2+ no-shows"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        # Find patients with 2+ no-shows
        cursor.execute('''
        SELECT patient_email, COUNT(*) as no_show_count 
        FROM appointments 
        WHERE status = 'No-Show' 
        GROUP BY patient_email 
        HAVING COUNT(*) >= 2
        ''')
        
        no_show_patients = cursor.fetchall()
        
        warnings_sent = 0
        
        for patient in no_show_patients:
            # Get patient details
            cursor.execute('''
            SELECT full_name, phone, email FROM users WHERE email = ?
            ''', (patient['patient_email'],))
            user = cursor.fetchone()
            
            if user and user['phone']:
                # Get their latest no-show clinic
                cursor.execute('''
                SELECT clinic_name FROM appointments 
                WHERE patient_email = ? AND status = 'No-Show' 
                ORDER BY appointment_date DESC LIMIT 1
                ''', (patient['patient_email'],))
                last_appt = cursor.fetchone()
                
                clinic = last_appt['clinic_name'] if last_appt else 'your clinic'
                
                result = self.sms_service.send_no_show_warning(
                    user['full_name'],
                    user['phone'],
                    clinic,
                    patient['no_show_count']
                )
                
                if result['status'] == 'success':
                    warnings_sent += 1
        
        conn.close()
        return warnings_sent
    
    def send_appointment_reminder_manual(self, appt_id):
        """Send a reminder for a specific appointment (manual trigger)"""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute('''
        SELECT a.id, a.patient_name, a.patient_email, a.patient_phone, 
               a.clinic_name, a.appointment_date, a.appointment_time,
               u.phone as user_phone, u.email as user_email
        FROM appointments a
        LEFT JOIN users u ON a.patient_email = u.email
        WHERE a.id = ?
        ''', (appt_id,))
        
        appt = cursor.fetchone()
        conn.close()
        
        if not appt:
            return {'status': 'error', 'message': 'Appointment not found'}
        
        phone = appt['patient_phone'] or appt['user_phone']
        email = appt['patient_email'] or appt['user_email']
        patient_name = appt['patient_name']
        
        results = {'sms': None, 'email': None}
        
        # Send SMS
        if phone:
            results['sms'] = self.sms_service.send_reminder(
                patient_name, phone, appt['clinic_name'],
                appt['appointment_date'], appt['appointment_time']
            )
        
        # Send Email
        if email:
            results['email'] = self.email_service.send_reminder(
                patient_name, email, appt['clinic_name'],
                appt['appointment_date'], appt['appointment_time']
            )
        
        # Mark as sent
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('UPDATE appointments SET reminder_sent = 1 WHERE id = ?', (appt_id,))
        conn.commit()
        conn.close()
        
        return {
            'status': 'success',
            'patient': patient_name,
            'results': results
        }