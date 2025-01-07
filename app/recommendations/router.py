from pathlib import Path
from fastapi import APIRouter, Query
from typing import List
from .models import GroupPreferences
from .service import RecommendationService

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = str(ROOT_DIR / "best_two_tower_model.keras")
print(f"Model path: {MODEL_PATH} -----------------------------------------------------------------------")

recommendation_service = RecommendationService(model_path=MODEL_PATH)

@router.post("/group")
async def get_recommendations(
    preferences: GroupPreferences,
    movie_ids: List[int] = Query(...),
    limit: int = Query(10, ge=1, le=50)
):
    """Get group recommendations based on movie preferences and history"""
    recommendations = await recommendation_service.get_group_recommendations(
        movie_ids=movie_ids,
        preferences=preferences,
        limit=limit
    )
    return {"recommendations": recommendations}


@router.post("/group/preferences")
async def create_group_preferences(
    user_ids: List[str],
    runtime: str = Query(..., description="short/medium/long"),
    languages: List[str] = Query(...),
    min_rating: float = Query(..., ge=0, le=10),
    start_year: int = Query(...),
    end_year: int = Query(...)
):
    """Create group preferences from user IDs"""
    group_prefs = await recommendation_service.create_group_preferences(
        user_ids=user_ids,
        runtime_pref=runtime,
        language_pref=languages,
        min_rating=min_rating,
        year_range=(start_year, end_year)
    )
    return group_prefs