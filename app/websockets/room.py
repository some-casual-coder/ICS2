from pydantic import BaseModel
from typing import List, Dict, Optional
from uuid import UUID

class User(BaseModel):
    id: UUID
    name: str
    is_host: bool = False
    status: str = "joined"
    swipe_progress: int = 0
    total_movies: int = 0

class Room(BaseModel):
   id: UUID
   name: str
   code: str
   host_id: UUID
   users: Dict[UUID, User] = {}
   pending_users: Dict[UUID, User] = {}
   latitude: float | None = None
   longitude: float | None = None
