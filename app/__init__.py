from flask import Flask
from dotenv import load_dotenv   # <-- NEW
import os

# Load environment variables from .env file
load_dotenv()   # <-- NEW

# Create Flask app
app = Flask(
    __name__,
    template_folder='../templates',
    static_folder='../static'
)

app.secret_key = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

# Import routes only (auth is merged into routes)
from app import routes