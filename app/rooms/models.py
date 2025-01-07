from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
from datetime import datetime

class RoomPreferences(BaseModel):
    runtime_preference: str = Field(..., description="short/medium/long")
    language_preference: List[str]
    min_rating: float = Field(..., ge=0, le=10)
    release_year_range: Tuple[int, int]

class Room(BaseModel):
    room_id: str
    creator_id: str
    members: List[str]
    preferences: Optional[RoomPreferences]
    created_at: datetime
    updated_at: datetime