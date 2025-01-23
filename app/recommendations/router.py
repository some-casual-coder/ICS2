from pathlib import Path
from fastapi import APIRouter, Query
from typing import List
from .models import CreateGroupPreferencesRequest, GroupPreferences, RecommendationRequest
from .service import RecommendationService

router = APIRouter(prefix="/recommendations", tags=["recommendations"])

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
MODEL_PATH = str(ROOT_DIR / "best_two_tower_model_v2.keras")
print(f"Model path: {MODEL_PATH} -----------------------------------------------------------------------")

recommendation_service = RecommendationService(model_path=MODEL_PATH)

@router.post("/group")
async def get_recommendations(request: RecommendationRequest):
    """Get group recommendations based on movie preferences and history"""
    recommendations = await recommendation_service.get_group_recommendations(
        movie_ids=request.movie_ids,
        not_interested_ids=request.not_interested_ids,
        preferences=request.preferences,
        limit=20
    )
    return {"recommendations": recommendations}


@router.post("/group/preferences")
async def create_group_preferences(request: CreateGroupPreferencesRequest):
    """Create group preferences from user IDs and preferences"""
    group_prefs = await recommendation_service.create_group_preferences(
        user_ids=request.user_ids,
        runtime_pref=request.runtime,
        language_pref=request.languages,
        min_rating=request.min_rating,
        year_range=(request.start_year, request.end_year)
    )
    return group_prefs