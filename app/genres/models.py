from pydantic import BaseModel
from typing import List, Dict

class GenrePreference(BaseModel):
    user_id: str
    selected_genres: List[Dict[str, int]]