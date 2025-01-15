# app/scripts/create_test_data.py
from firebase_admin import credentials, initialize_app, firestore
from pathlib import Path

# Get the project root directory
ROOT_DIR = Path(__file__).resolve().parent.parent.parent

# Initialize Firebase
cred = credentials.Certificate(str(ROOT_DIR / "serviceAccount.json"))
initialize_app(cred)
db = firestore.client()

# Test user data
test_users = {
    "user123": {
        "preferences": {
            "genres": [
                "Action",
                "Adventure",
                "Science Fiction",
                "Drama"
            ]
        }
    },
    "user456": {
        "preferences": {
            "genres": [
                "Drama",
                "Comedy",
                "Romance",
                "Action"
            ]
        }
    }
}

def create_test_users():
    """Create test users in Firebase"""
    for user_id, data in test_users.items():
        try:
            db.collection('users').document(user_id).set(data)
            print(f"Created test user: {user_id}")
        except Exception as e:
            print(f"Error creating user {user_id}: {str(e)}")

if __name__ == "__main__":
    create_test_users()