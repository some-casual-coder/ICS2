from typing import Dict, List
from fastapi import HTTPException
from ..core.firebase import db
from .models import UserPreferences

async def get_user_genres(user_id: str) -> List[Dict[str, int]]:
    try:
        doc = db.collection('user_preferences').document(user_id).get()
        if not doc.exists:
            raise HTTPException(status_code=404, detail="User preferences not found")
        return doc.to_dict().get('genres', [])
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def get_user_liked_movies(user_id: str) -> Dict:
    try:
        doc = db.collection('user_movie_preferences').document(user_id).get()
        if not doc.exists:
            return {}
        
        ratings = doc.to_dict().get('ratings', {})
        return {
            movie_id: data 
            for movie_id, data in ratings.items() 
            if data['rating'] > 0
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
