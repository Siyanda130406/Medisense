# scheduler.py
import time
import schedule
from datetime import datetime
from reminder_service import ReminderService

def run_reminder_job():
    """Run the daily reminder job"""
    print(f"\n{'='*60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Running reminder job...")
    print(f"{'='*60}")
    
    service = ReminderService()
    result = service.send_daily_reminders()
    
    print(f"\n Reminder Results:")
    print(f"   Total appointments checked: {result['total']}")
    print(f"   Reminders sent: {result['sent']}")
    print(f"   Failed: {result['failed']}")
    
    # Send no-show warnings weekly
    if datetime.now().weekday() == 0:  # Monday
        print("\n Sending no-show warnings...")
        warnings = service.send_no_show_warnings()
        print(f"   No-show warnings sent: {warnings}")
    
    print(f"{'='*60}\n")

def run_no_show_checker():
    """Check for expired appointments (45 minute rule)"""
    try:
        from app import routes
        expired = routes.check_expired_appointments()
        if expired > 0:
            print(f"   Auto no-shows marked: {expired}")
    except Exception as e:
        print(f"   Error in no-show checker: {e}")

def run_scheduler():
    """Run the scheduler in background"""
    print(f"\n{'='*60}")
    print("MEDISENSE SCHEDULER")
    print(f"{'='*60}")
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Scheduler started")
    print("")
    print("Schedule:")
    print("  Every minute: Check for appointments 45 mins past (Auto No-Show)")
    print("  Every minute: Check for appointments 1 hour away (Reminders)")
    print("  Monday: Send no-show warnings")
    print("  8:00 AM: Send daily reminders")
    print("  6:00 PM: Send evening reminders")
    print(f"{'='*60}\n")
    
    # Run the no-show checker every minute
    schedule.every(1).minutes.do(run_no_show_checker)
    
    # Run reminder checker every minute
    schedule.every(1).minutes.do(run_reminder_job)
    
    # Scheduled times for reminders
    schedule.every().day.at("08:00").do(run_reminder_job)
    schedule.every().day.at("18:00").do(run_reminder_job)
    
    while True:
        schedule.run_pending()
        time.sleep(30)  # Check every 30 seconds

if __name__ == "__main__":
    run_scheduler()