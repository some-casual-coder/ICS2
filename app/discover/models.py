from pydantic import BaseModel
from typing import List, Optional, Dict
from datetime import datetime

class MovieMetadata(BaseModel):
    id: int
    title: str
    vote_count: int
    vote_average: float
    runtime: Optional[int]
    original_language: str
    popularity: float
    poster_path: Optional[str]
    genres: List[str]
    release_date: str
    spoken_languages: List[str]

class DiscoverFilters(BaseModel):
    runtime_preference: Optional[str] = None
    language_preference: Optional[List[str]] = None
    min_rating: Optional[float] = None
    release_year_range: Optional[tuple[int, int]] = None
    genres: Optional[List[int]] = None
    exclude_watched: bool = True