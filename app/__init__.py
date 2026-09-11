from flask import Flask
import os

# Create Flask app
app = Flask(
    __name__,
    template_folder='../templates',
    static_folder='../static'
)

app.secret_key = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

# Import routes only (auth is merged into routes)
from app import routes
