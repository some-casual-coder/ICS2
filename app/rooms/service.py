from typing import Optional
from fastapi import HTTPException
from firebase_admin import firestore
from ..core.firebase import db
from .models import Room, RoomPreferences
import uuid
from datetime import datetime

async def create_room(creator_id: str, room_id: str, preferences: Optional[RoomPreferences] = None) -> str:
    try:
        room_ref = db.collection('rooms').document(room_id)
        
        room_data = {
            'room_id': room_id,
            'creator_id': creator_id,
            'members': [creator_id],
            'created_at': firestore.SERVER_TIMESTAMP,
            'updated_at': firestore.SERVER_TIMESTAMP
        }
        
        if preferences:
            room_data['preferences'] = preferences.dict()
        
        room_ref.set(room_data)
        return room_id
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def join_room(room_id: str, user_id: str) -> bool:
    try:
        room_ref = db.collection('rooms').document(room_id)
        room = room_ref.get()
        
        if not room.exists:
            raise HTTPException(status_code=404, detail="Room not found")
            
        room_ref.update({
            'members': firestore.ArrayUnion([user_id]),
            'updated_at': firestore.SERVER_TIMESTAMP
        })
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

async def update_room_preferences(room_id: str, preferences: RoomPreferences) -> bool:
    try:
        room_ref = db.collection('rooms').document(room_id)
        room = room_ref.get()
        
        if not room.exists:
            raise HTTPException(status_code=404, detail="Room not found")
            
        room_ref.update({
            'preferences': preferences.dict(),
            'updated_at': firestore.SERVER_TIMESTAMP
        })
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
async def get_room_preferences(room_id: str) -> Optional[RoomPreferences]:
    try:
        room_ref = db.collection('rooms').document(room_id)
        room_doc = room_ref.get()
        
        if not room_doc.exists:
            raise HTTPException(status_code=404, detail="Room not found")
            
        room_data = room_doc.to_dict()
        preferences_data = room_data.get('preferences')
        
        if not preferences_data:
            return None
            
        return RoomPreferences(**preferences_data)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    
async def delete_room(room_id: str) -> bool:
    try:
        room_ref = db.collection('rooms').document(room_id)
        if not room_ref.get().exists:
            raise HTTPException(status_code=404, detail="Room not found")
        
        room_ref.delete()
        return True
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))