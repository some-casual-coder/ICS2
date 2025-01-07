from pydantic import BaseModel
from typing import List, Dict, Any

class UserPreferences(BaseModel):
    genres: List[Dict[str, int]]
    movies: Dict[str, Dict[str, Any]]