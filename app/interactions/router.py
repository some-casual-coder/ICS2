from fastapi import APIRouter
from .models import BatchInteraction, MovieInteraction, RoomStatistics
from .service import InteractionService

router = APIRouter(prefix="/interactions", tags=["interactions"])
interaction_service = InteractionService()

@router.post("/record")
async def record_swipe_action(interaction: MovieInteraction):
    """Record a user's swipe action on a movie"""
    await interaction_service.record_interaction(interaction)
    return {"status": "success"}

@router.get("/movies/{movie_id}/stats")
async def get_movie_stats(movie_id: int):
    """Get interaction statistics for a specific movie"""
    stats = await interaction_service.get_movie_statistics(movie_id)
    return stats

@router.post("/batch-record")
async def record_batch_swipe_actions(batch: BatchInteraction):
    """Record multiple swipe actions in a single request"""
    await interaction_service.record_batch_interactions(batch)
    return {
        "status": "success",
        "message": f"Recorded {len(batch.swipes)} interactions"
    }

@router.get("/rooms/{room_id}/stats")
async def get_room_statistics(room_id: str):
    """Get aggregated movie interactions for a room"""
    stats = await interaction_service.get_room_stats(room_id)
    return stats