from fastapi import APIRouter, HTTPException
from .models import GenrePreference
from .service import get_tmdb_genres, save_user_genres

router = APIRouter(prefix="/genres", tags=["genres"])

@router.get("/")
async def get_genres():
    """Fetch all available genres from TMDB"""
    try:
        genres = await get_tmdb_genres()
        return {"genres": genres}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/preferences")
async def save_genre_preferences(preferences: GenrePreference):
    """Save user's genre preferences"""
    await save_user_genres(
        preferences.user_id, 
        preferences.selected_genres
    )
    return {
        "status": "success", 
        "message": "Genre preferences saved successfully"
    }