import traceback
from fastapi import HTTPException
import pandas as pd
from qdrant_client import QdrantClient
from typing import List, Dict, Tuple
import numpy as np

from app.recommendations.config import GENRE_CONFIG
from app.recommendations.model_loader import load_two_tower_model
from scripts.embeddings.utils import generate_candidates_for_two_tower
from .models import GroupPreferences, MovieRecommendation
from .utils import prepare_group_features, combine_scores
from ..core.config import settings
from ..core.firebase import db

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

tmdb_df = pd.read_csv('datasets/tmdb_movie_metadata_v2.csv')
tmdb_df.set_index('id', inplace=True)


class RecommendationService:
    def __init__(self, model_path: str):
        self.qdrant_client = QdrantClient(
            url=settings.QDRANT_URL,
            api_key=settings.QDRANT_API_KEY
        )
        logger.info("Loading recommendation model...")
        self.model = load_two_tower_model(model_path)
        if self.model is None:
            logger.error("Model loading failed")
            raise Exception(
                "Failed to load recommendation model. Check logs for details.")

    async def aggregate_group_genre_weights(self, user_ids: List[str]) -> Dict[str, float]:
        """Aggregate genre preferences for all users in a group"""
        genre_preferences = []

        # Get all users' genre preferences
        for user_id in user_ids:
            user_doc = db.collection('users').document(user_id).get()
            if user_doc.exists:
                user_data = user_doc.to_dict()
                if 'preferences' in user_data and 'genres' in user_data['preferences']:
                    genre_preferences.append(
                        user_data['preferences']['genres'])

        # Aggregate weights
        aggregated_weights = {}
        num_users = len(genre_preferences)

        # Initialize with CONFIG genres
        for genre in GENRE_CONFIG['genre_weights'].keys():
            genre_count = sum(
                1 for prefs in genre_preferences if genre in prefs)
            aggregated_weights[genre] = genre_count / \
                num_users if num_users > 0 else 0.0

        return aggregated_weights

    async def create_group_preferences(
        self,
        user_ids: List[str],
        runtime_pref: str,
        language_pref: List[str],
        min_rating: float,
        year_range: Tuple[int, int]
    ) -> GroupPreferences:
        """Create GroupPreferences with aggregated genre weights"""
        genre_weights = await self.aggregate_group_genre_weights(user_ids)

        return GroupPreferences(
            runtime_preference=runtime_pref,
            genre_weights=genre_weights,
            language_preference=language_pref,
            min_rating=min_rating,
            release_year_range=year_range
        )

    async def get_group_recommendations(
        self,
        movie_ids: List[int],
        not_interested_ids: List[int],
        preferences: GroupPreferences,
        limit: int = 20
    ) -> List[MovieRecommendation]:
        try:
            # Get candidates with embeddings
            candidates_data = await generate_candidates_for_two_tower(
                client=self.qdrant_client,
                movie_ids=movie_ids,
                not_interested_ids=not_interested_ids
            )

            # Prepare movie features for model
            movie_features = np.array([
                data['embedding']
                for data in candidates_data['candidates'].values()
            ])

            # Prepare group features
            group_features = prepare_group_features(preferences)

            # Repeat group features to match movie features shape
            num_movies = movie_features.shape[0]
            group_features_repeated = np.repeat(
                group_features, num_movies, axis=0)

            # Get model predictions
            similarity_scores = self.model.predict(
                [movie_features, group_features_repeated])

            # Convert ragged tensor to dense and get mean of each row
            similarity_scores = similarity_scores.to_tensor().numpy()
            similarity_scores = np.mean(similarity_scores, axis=1)

            # Combine scores and create recommendations
            recommendations = []
            for (movie_id, data), model_score in zip(
                candidates_data['candidates'].items(),
                similarity_scores
            ):
                metadata = data['metadata']
                final_score = combine_scores(
                    hnsw_score=data['similarity_scores'],
                    model_score=float(model_score),
                    preferences=preferences,
                    metadata=metadata
                )

                recommendations.append(MovieRecommendation(
                    movie_id=movie_id,
                    metadata=metadata,
                    final_score=final_score,
                    hnsw_score=data['similarity_scores'],
                    model_score=float(model_score)
                ))

            # Sort and limit results
            # In your recommendations processing
            logger.info(f"Total recommendations before sorting: {len(recommendations)}")
            recommendations.sort(key=lambda x: x.final_score, reverse=True)
            logger.info(f"Top score: {recommendations[0].final_score if recommendations else 'No recommendations'}")
            top_recommendations = recommendations[:limit]
            logger.info(f"Number of recommendations after limit: {len(top_recommendations)}")

            # Get poster paths from the dataset
            recommendations_with_posters = []
            for rec in top_recommendations:
                movie_data = {
                    'movie_id': rec.movie_id,
                    'title': rec.metadata['title'],
                    'genres': rec.metadata['genres'],
                    'final_score': round(rec.final_score, 3),
                    'hnsw_score': round(rec.hnsw_score, 3),
                    'model_score': round(rec.model_score, 3)
                }
                if rec.movie_id in tmdb_df.index:
                    poster_path = tmdb_df.loc[rec.movie_id, 'poster_path']
                    vote_average = tmdb_df.loc[rec.movie_id, 'vote_average']
                    if pd.notna(poster_path):
                        movie_data['poster_path'] = poster_path
                        movie_data['vote_average'] = vote_average

                recommendations_with_posters.append(movie_data)
                
            logger.info(f"Number of recommendations with posters after limit: {len(recommendations_with_posters)}")

            return {
                'total_candidates': len(recommendations),
                'original_movies': candidates_data['original_movies'],
                'recommendations': recommendations_with_posters
            }
        except Exception as e:
            logger.error(f"Error in get_group_recommendations: {str(e)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            raise HTTPException(status_code=500, detail=str(e))
