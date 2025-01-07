from fastapi import APIRouter, HTTPException, Query
from typing import List
from .models import UserMoviePreference
from .service import get_discover_movies, search_movies, save_user_movie_preferences

router = APIRouter(prefix="/movies", tags=["movies"])

@router.get("/discover")
async def discover_movies(
    genre_ids: str = Query(..., description="Comma-separated list of genre IDs"),
    page: int = Query(1, ge=1)
):
    """
    Get popular movies filtered by genres
    Uses AND logic between genres (movies must have all specified genres)
    """
    # Convert comma-separated string to list of integers
    genre_list = [int(id) for id in genre_ids.split(",") if id.strip()]
    try:
        movies = await get_discover_movies(genre_list, page)
        return movies
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/search")
async def search_movies_route(
    query: str = Query(..., min_length=1),
    page: int = Query(1, ge=1)
):
    """Search for movies by title"""
    try:
        results = await search_movies(query, page)
        return results
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/preferences")
async def save_movie_preferences(preferences: List[UserMoviePreference]):
    """
    Save user's movie preferences
    """
    if not preferences:
        raise HTTPException(status_code=400, detail="No preferences provided")
    
    try:
        user_id = preferences[0].user_id  # All preferences should have same user_id
        await save_user_movie_preferences(user_id, preferences)
        return {"status": "success", "message": "Movie preferences saved successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))