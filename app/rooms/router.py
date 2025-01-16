from typing import Optional
from fastapi import APIRouter
from .models import RoomPreferences
from .service import create_room, delete_room, join_room, update_room_preferences, get_room_preferences

router = APIRouter(prefix="/rooms", tags=["rooms"])

@router.post("/create")
async def create_new_room(creator_id: str, room_id: str, preferences: Optional[RoomPreferences] = None):
    room_id = await create_room(creator_id, room_id, preferences)
    return {"room_id": room_id}

@router.post("/{room_id}/join")
async def join_existing_room(room_id: str, user_id: str):
    await join_room(room_id, user_id)
    return {"status": "success"}

@router.get("/{room_id}/preferences")
async def get_room_preferences_endpoint(room_id: str):
    preferences = await get_room_preferences(room_id)
    return {"preferences": preferences.dict() if preferences else None}

@router.put("/{room_id}/preferences")
async def update_preferences(room_id: str, preferences: RoomPreferences):
    await update_room_preferences(room_id, preferences)
    return {"status": "success"}

@router.delete("/{room_id}")
async def delete_existing_room(room_id: str):
    await delete_room(room_id)
    return {"status": "success", "message": "Room deleted successfully"}