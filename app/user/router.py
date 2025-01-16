from fastapi import APIRouter, HTTPException

from app.user.models import UserSettings
from .service import get_user_genres, get_user_liked_movies, get_user_settings, save_user_settings

router = APIRouter(prefix="/user", tags=["user"])

@router.get("/{user_id}/genres")
async def get_genres(user_id: str):
    return {"genres": await get_user_genres(user_id)}

@router.get("/{user_id}/liked-movies")
async def get_liked_movies(user_id: str):
    return {"movies": await get_user_liked_movies(user_id)}

@router.get("/{user_id}/settings")
async def get_user_preferences(user_id: str):
    settings = await get_user_settings(user_id)
    if settings is None:
        raise HTTPException(status_code=404, detail="User settings not found")
    return settings

@router.post("/settings")
async def save_settings(settings: UserSettings):
    await save_user_settings(settings)
    return {"status": "success", "message": "User settings saved successfully"}