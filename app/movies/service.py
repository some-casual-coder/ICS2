from fastapi import HTTPException
import httpx
from typing import List, Dict
from firebase_admin import firestore

from app.movies.models import UserMoviePreference
from ..core.config import settings
from ..core.firebase import db

async def get_discover_movies(genre_ids: List[int], page: int = 1) -> Dict:
    """Fetch popular movies filtered by genres"""
    headers = {
        "Authorization": f"Bearer {settings.TMDB_TOKEN}",
        "accept": "application/json"
    }
    
    # Convert genre IDs to comma-separated string for AND logic
    genres_param = ",".join(map(str, genre_ids))
    
    params = {
        "include_adult": "false",
        "include_video": "false",
        "language": "en-US",
        "page": page,
        "sort_by": "popularity.desc",
        "with_genres": genres_param
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.TMDB_BASE_URL}/discover/movie",
            headers=headers,
            params=params
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code,
                              detail="Failed to fetch movies from TMDB")
        
        return response.json()

async def search_movies(query: str, page: int = 1) -> Dict:
    """Search for movies by title"""
    headers = {
        "Authorization": f"Bearer {settings.TMDB_TOKEN}",
        "accept": "application/json"
    }
    
    params = {
        "query": query,
        "include_adult": "false",
        "language": "en-US",
        "page": page
    }
    
    async with httpx.AsyncClient() as client:
        response = await client.get(
            f"{settings.TMDB_BASE_URL}/search/movie",
            headers=headers,
            params=params
        )
        
        if response.status_code != 200:
            raise HTTPException(status_code=response.status_code,
                              detail="Failed to search movies from TMDB")
        
        return response.json()

async def save_user_movie_preferences(
    user_id: str,
    preferences: List[UserMoviePreference]
) -> bool:
    """Save user's movie preferences to Firebase"""
    try:
        # Get reference to user's movie preferences
        user_movies_ref = db.collection('user_movie_preferences').document(user_id)
        
        # Convert preferences to dictionary format for Firestore
        movie_ratings = {
            str(pref.movie_id): {
                "rating": pref.rating,
                "timestamp": firestore.SERVER_TIMESTAMP
            }
            for pref in preferences
        }
        
        # Update or create the document with the movie preferences
        user_movies_ref.set({
            "ratings": movie_ratings,
            "updated_at": firestore.SERVER_TIMESTAMP
        }, merge=True)
        
        return True
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save movie preferences: {str(e)}"
        )