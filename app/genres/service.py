from typing import Dict, List
from fastapi import HTTPException
import httpx
from firebase_admin import firestore
from ..core.config import settings
from ..core.firebase import db

async def get_tmdb_genres():
    """Fetch genres from TMDB API"""
    headers = {
        "Authorization": f"Bearer {settings.TMDB_TOKEN}",
        "accept": "application/json"
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.TMDB_BASE_URL}/genre/movie/list?language=en",
            headers=headers
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code, 
                              detail="Failed to fetch genres from TMDB")
        
        return response.json()["genres"]

async def save_user_genres(user_id: str, selected_genres: List[Dict[str, int]]):
    """Save user's genre preferences to Firebase"""
    try:
        user_ref = db.collection('user_preferences').document(user_id)
        user_ref.set({
            'genres': selected_genres,
            'updated_at': firestore.SERVER_TIMESTAMP
        }, merge=True)
        return True
    except Exception as e:
        raise HTTPException(status_code=500, 
                          detail=f"Failed to save preferences: {str(e)}")