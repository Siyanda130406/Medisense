from flask import Flask
from dotenv import load_dotenv
import os

load_dotenv()

os.environ.setdefault('DISABLE_SENTENCE_TRANSFORMERS', 'true')

app = Flask(
    __name__,
    template_folder='../templates',
    static_folder='../static'
)

app.secret_key = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'

from medisense import routes  # noqa: E402,F401