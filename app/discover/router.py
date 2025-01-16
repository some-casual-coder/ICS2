from fastapi import APIRouter, Query
from .service import DiscoverService
from .models import DiscoverFilters
from typing import List, Optional

router = APIRouter(prefix="/discover", tags=["discover"])
discover_service = DiscoverService()

@router.get("/movies")
async def discover_movies(
    runtime: Optional[str] = None,
    languages: Optional[str] = None,
    min_rating: Optional[float] = None,
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    genres: Optional[List[int]] = Query(None),
    exclude_watched: bool = True,
    watched_movies: Optional[List[int]] = Query(None),
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100)
):
    """
    Custom discover endpoint using local dataset
    """
    language_list = languages.split(',') if languages else None
    filters = DiscoverFilters(
        runtime_preference=runtime,
        language_preference=language_list,
        min_rating=min_rating,
        release_year_range=(start_year, end_year) if start_year and end_year else None,
        genres=genres,
        exclude_watched=exclude_watched
    )
    
    return await discover_service.get_discover_movies(
        filters,
        watched_movies=watched_movies,
        page=page,
        per_page=per_page
    )