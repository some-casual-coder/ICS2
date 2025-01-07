from enum import Enum
from pydantic import BaseModel
from typing import List, Dict, Any

class UserPreferences(BaseModel):
    genres: List[Dict[str, int]]
    movies: Dict[str, Dict[str, Any]]
    
class MovieLength(str, Enum):
    SHORT = "short"     # < 90 mins
    MEDIUM = "medium"   # 90-150 mins
    LONG = "long"       # > 150 mins

class Era(str, Enum):
    RECENT = "recent"   # Last 3 years
    ERA_2010s = "2010s"
    ERA_2000s = "2000s"
    CLASSICS = "classics" # Before 2000

class LanguagePreference(str, Enum):
    LOCAL = "local"
    INTERNATIONAL = "international"
    BOTH = "both"

class UserSettings(BaseModel):
    user_id: str
    movie_length: List[MovieLength]
    preferred_eras: List[Era]
    language_preference: LanguagePreference