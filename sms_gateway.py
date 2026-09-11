import os
import requests
import json
from datetime import datetime
import re

class SMSService:
    """Africa's Talking SMS Gateway Integration"""
    
    def __init__(self):
        self.username = 'sandbox'
        self.api_key = 'atsk_98ffc2f1b0dd8527948e9fa7f5b7f26b7f7761ce8107496aaa3d482b4e0d3ce3825c9987'
        self.base_url = 'https://api.africastalking.com/version1/messaging'
        self.sandbox_mode = True
        
        # Check if API key is set
        if not self.api_key:
            print("⚠️ Africa's Talking API key not set. SMS will be simulated.")
            self.sandbox_mode = True
        else:
            print(f"✅ SMS Service initialized with API key")
            print(f"   Mode: {'SIMULATION' if self.sandbox_mode else 'REAL SMS'}")
    
    def send_sms(self, phone_number, message):
        """
        Send SMS using Africa's Talking API
        """
        # Format phone number
        phone = self._format_phone(phone_number)
        
        if not phone:
            print(f"❌ Invalid phone number: {phone_number}")
            return {
                'status': 'error',
                'message': 'Invalid phone number'
            }
        
        # For sandbox testing (only if API key is missing)
        if self.sandbox_mode or not self.api_key:
            print(f"\n📱 [SMS SIMULATION]")
            print(f"   To: {phone}")
            print(f"   Message: {message}")
            print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            return {
                'status': 'success',
                'message': 'SMS sent (simulated)',
                'recipient': phone
            }
        
        # Real API call
        try:
            headers = {
                'ApiKey': self.api_key,
                'Content-Type': 'application/x-www-form-urlencoded'
            }
            
            data = {
                'username': self.username,
                'to': phone,
                'message': message,
                'from': 'MEDISENSE'
            }
            
            print(f"\n📱 Sending REAL SMS...")
            print(f"   To: {phone}")
            print(f"   Message: {message[:50]}..." if len(message) > 50 else f"   Message: {message}")
            print(f"   Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            
            response = requests.post(
                self.base_url,
                headers=headers,
                data=data,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"   ✅ SMS sent successfully!")
                return {
                    'status': 'success',
                    'response': result,
                    'recipient': phone
                }
            else:
                print(f"   ❌ SMS failed: {response.status_code}")
                return {
                    'status': 'error',
                    'message': f'API error: {response.status_code}',
                    'response': response.text
                }
                
        except Exception as e:
            print(f"   ❌ SMS failed: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def send_reminder(self, patient_name, phone, clinic, date, time):
        """Send appointment reminder"""
        message = f"MediSense Reminder: Hello {patient_name}, you have an appointment at {clinic} on {date} at {time}. Please arrive 15 minutes early. Reply STOP to opt out."
        return self.send_sms(phone, message)
    
    def send_no_show_warning(self, patient_name, phone, clinic, no_show_count):
        """Send no-show warning"""
        message = f"MediSense: {patient_name}, you missed your appointment at {clinic}. You have {no_show_count} no-shows recorded. Please reschedule to avoid penalties."
        return self.send_sms(phone, message)
    
    def send_health_alert(self, phone, alert_message):
        """Send health alert"""
        message = f"MEDISENSE HEALTH ALERT: {alert_message}"
        return self.send_sms(phone, message)
    
    def _format_phone(self, phone):
        """Format phone number to international format"""
        if not phone:
            return None
        
        # Remove any non-digit characters
        phone = ''.join(filter(str.isdigit, phone))
        
        # If number starts with 0, replace with 27 (South Africa)
        if phone.startswith('0'):
            phone = '27' + phone[1:]
        
        # If number starts with 27, keep as is
        if phone.startswith('27'):
            return phone
        
        # If number has 9 digits, assume it's a local number without country code
        if len(phone) == 9:
            return '27' + phone
        
        # If number starts with +, remove it
        if phone.startswith('+'):
            phone = phone[1:]
        
        return phone


class EmailService:
    """Email Service for sending reminders"""
    
    def __init__(self):
        self.smtp_server = os.environ.get('SMTP_SERVER', 'smtp.gmail.com')
        self.smtp_port = int(os.environ.get('SMTP_PORT', 587))
        self.smtp_username = os.environ.get('SMTP_USERNAME', '')
        self.smtp_password = os.environ.get('SMTP_PASSWORD', '')
        self.from_email = os.environ.get('FROM_EMAIL', 'medisense@example.com')
        
        self.enabled = bool(self.smtp_username and self.smtp_password)
        
        if not self.enabled:
            print("⚠️ Email credentials not set. Emails will be simulated.")
        else:
            print(f"✅ Email Service initialized")
            print(f"   From: {self.from_email}")
    
    def send_email(self, to_email, subject, body, html_body=None):
        """Send email using SMTP"""
        if not self.enabled:
            print(f"\n📧 [EMAIL SIMULATION]")
            print(f"   To: {to_email}")
            print(f"   Subject: {subject}")
            print(f"   Body: {body[:100]}..." if len(body) > 100 else f"   Body: {body}")
            return {
                'status': 'success',
                'message': 'Email sent (simulated)',
                'recipient': to_email
            }
        
        try:
            import smtplib
            from email.mime.text import MIMEText
            from email.mime.multipart import MIMEMultipart
            
            msg = MIMEMultipart()
            msg['From'] = self.from_email
            msg['To'] = to_email
            msg['Subject'] = subject
            
            # Plain text version
            msg.attach(MIMEText(body, 'plain'))
            
            # HTML version if provided
            if html_body:
                msg.attach(MIMEText(html_body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.smtp_server, self.smtp_port)
            server.starttls()
            server.login(self.smtp_username, self.smtp_password)
            server.send_message(msg)
            server.quit()
            
            print(f"\n📧 Email sent to {to_email}")
            return {
                'status': 'success',
                'message': 'Email sent successfully',
                'recipient': to_email
            }
            
        except Exception as e:
            print(f"\n❌ Email failed: {str(e)}")
            return {
                'status': 'error',
                'message': str(e)
            }
    
    def send_reminder(self, patient_name, to_email, clinic, date, time):
        """Send appointment reminder email"""
        subject = f"MediSense: Appointment Reminder - {clinic} on {date}"
        
        body = f"""
Dear {patient_name},

This is a reminder of your upcoming appointment:

Clinic: {clinic}
Date: {date}
Time: {time}

Please arrive 15 minutes early.

If you need to cancel or reschedule, please log in to MediSense.

Thank you,
MediSense Team
"""
        
        html_body = f"""
<!DOCTYPE html>
<html>
<head>
    <style>
        body {{ font-family: Arial, sans-serif; color: #333; }}
        .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
        .header {{ background-color: #1a73e8; color: white; padding: 15px; text-align: center; }}
        .content {{ padding: 20px; }}
        .footer {{ text-align: center; padding: 10px; font-size: 12px; color: #666; }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h2>MediSense</h2>
            <p>Appointment Reminder</p>
        </div>
        <div class="content">
            <h3>Dear {patient_name},</h3>
            <p>This is a reminder of your upcoming appointment:</p>
            <table>
                <tr><td><strong>Clinic:</strong></td><td>{clinic}</td></tr>
                <tr><td><strong>Date:</strong></td><td>{date}</td></tr>
                <tr><td><strong>Time:</strong></td><td>{time}</td></tr>
            </table>
            <p>Please arrive 15 minutes early.</p>
            <p>If you need to cancel or reschedule, please log in to MediSense.</p>
        </div>
        <div class="footer">
            &copy; 2026 MediSense. All rights reserved.
        </div>
    </div>
</body>
</html>
"""
        
        return self.send_email(to_email, subject, body, html_body)