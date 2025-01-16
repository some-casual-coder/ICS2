from typing import Dict, List, Optional
from fastapi import HTTPException
from ..core.firebase import db
from .models import Era, LanguagePreference, MovieLength, UserPreferences, UserSettings

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
    
async def get_user_settings(user_id: str) -> Optional[UserSettings]:
    try:
        user_ref = db.collection('user_preferences').document(user_id)
        user_doc = user_ref.get()
        
        if not user_doc.exists:
            return None
            
        settings_data = user_doc.to_dict().get('settings', {})
        
        return UserSettings(
            user_id=user_id,
            movie_length=[MovieLength(length) for length in settings_data.get('movie_length', [])],
            preferred_eras=[Era(era) for era in settings_data.get('preferred_eras', [])],
            language_preference=LanguagePreference(settings_data.get('language_preference'))
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
async def save_user_settings(settings: UserSettings) -> bool:
    try:
        user_ref = db.collection('user_preferences').document(settings.user_id)
        user_ref.set({
            'settings': {
                'movie_length': [length.value for length in settings.movie_length],
                'preferred_eras': [era.value for era in settings.preferred_eras],
                'language_preference': settings.language_preference.value
            }
        }, merge=True)
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
