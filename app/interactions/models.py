from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime

class SwipeAction(str, Enum):
    INTERESTED = "interested"        # Swipe right - Want to watch
    WATCHED_LIKED = "watched_liked"  # Swipe up - Already watched and enjoyed
    NOT_INTERESTED = "not_interested"  # Swipe left - Don't want to watch
    NOT_SURE = "not_sure"           # Swipe down - Skip/Undecided

class MovieInteraction(BaseModel):
    user_id: str
    movie_id: int
    action: SwipeAction
    room_id: Optional[str]
    timestamp: Optional[datetime] = Field(default_factory=datetime.now) 

class MovieStats(BaseModel):
    movie_id: int
    total_watches: int  
    interested_count: int
    not_interested_count: int 
    not_sure_count: int
    engagement_ratio: float 
    
class BatchSwipe(BaseModel):
    movie_id: int
    action: SwipeAction

class BatchInteraction(BaseModel):
    user_id: str
    room_id: Optional[str]
    swipes: List[BatchSwipe]

class RoomStatistics(BaseModel):
    room_id: str
    interested_movie_ids: List[int] = []
    not_interested_movie_ids: List[int] = []
    watched_movie_ids: List[int] = []
    not_sure_movie_ids: List[int] = []