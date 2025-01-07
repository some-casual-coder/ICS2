from firebase_admin import credentials, initialize_app, firestore
from .config import settings

cred = credentials.Certificate(settings.FIREBASE_CREDS_PATH)
firebase_app = initialize_app(cred)
db = firestore.client()
