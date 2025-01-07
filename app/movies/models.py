from pydantic import BaseModel
from typing import List, Optional

class MovieBase(BaseModel):
    id: int
    title: str
    overview: str
    poster_path: Optional[str]
    vote_average: float
    genre_ids: List[int]
    release_date: str

class UserMoviePreference(BaseModel):
    user_id: str
    movie_id: int
    rating: int