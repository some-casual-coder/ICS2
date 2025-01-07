from pydantic import BaseModel, Field
from typing import Dict, List, Tuple, Optional

class GroupPreferences(BaseModel):
    runtime_preference: str = Field(..., description="short/medium/long")
    genre_weights: Dict[str, float]
    language_preference: List[str]
    min_rating: float = Field(..., ge=0, le=10)
    release_year_range: Tuple[int, int]

class MovieRecommendation(BaseModel):
    movie_id: int
    metadata: Dict
    final_score: float
    hnsw_score: float
    model_score: float
    
class MovieSimilarity(BaseModel):
    title: str
    similarity_score: float
    genres: List[str]

class CandidateResponse(BaseModel):
    total_candidates: int
    original_movies: List[str]
    recommended_movies: List[MovieSimilarity]