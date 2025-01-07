import os
from dotenv import load_dotenv
from fastapi import APIRouter, HTTPException
from qdrant_client import QdrantClient
from scripts.embeddings.group_model import GroupPreferencesModel
from scripts.embeddings.utils import format_candidate_names, generate_candidates_for_two_tower, process_movies

router = APIRouter(prefix="/embeddings", tags=["embeddings"])

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")

client = QdrantClient(
    url=qdrant_url,
    api_key=qdrant_key
)

# @router.get("/generate_embeddings")
# async def generate_embeddings():
#     """Generate embeddings from movies dataset"""
#     try:
#         result = await process_movies(openai_key, qdrant_url, qdrant_key)
#         return {"message": "Embedding process completed", "result": result}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    
@router.post("/recommendation_candidates")
async def recommend_movies(preferences: GroupPreferencesModel):
    try:
        candidates = await generate_candidates_for_two_tower(client, preferences.group_preferences)
        formatted_recommendations = format_candidate_names(candidates, client)
        return {"message": "Fetched Recommendations", "result": formatted_recommendations}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))