from fastapi import APIRouter
from .service import get_user_genres, get_user_liked_movies

router = APIRouter(prefix="/user", tags=["user"])

@router.get("/{user_id}/genres")
async def get_genres(user_id: str):
    return {"genres": await get_user_genres(user_id)}

@router.get("/{user_id}/liked-movies")
async def get_liked_movies(user_id: str):
    return {"movies": await get_user_liked_movies(user_id)}