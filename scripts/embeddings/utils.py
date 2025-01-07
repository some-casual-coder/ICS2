import gc
import time
from typing import Dict, List
import pandas as pd
from tqdm import tqdm
from qdrant_client.http import models
from qdrant_client import QdrantClient

from scripts.embeddings.config import Config
from scripts.embeddings.data_processor import DataProcessor
from scripts.embeddings.movie_embedding_generator import MovieEmbeddingGenerator
from scripts.embeddings.qdrant_handler import QdrantHandler

def prepare_genres(genres_data):
    if pd.isna(genres_data):
        return []
    # Split on comma and strip whitespace
    return [genre.strip() for genre in str(genres_data).split(',')]

def process_movies(openai_api_key, qdrant_url, qdrant_api_key):
    # Initialize components
    embedding_gen = MovieEmbeddingGenerator(openai_api_key)
    data_processor = DataProcessor()
    qdrant_handler = QdrantHandler(qdrant_url, qdrant_api_key)
    
    # Initialize Qdrant collection
    qdrant_handler.init_collection()
    
    # Load data
    df = data_processor.load_and_prepare_data()
    df = df[~df['id'].isin(data_processor.processed_ids)]

    total_processed = 0
    print(f"Starting processing of {len(df)} movies...")
    
    # Process in chunks
    for chunk_start in tqdm(range(0, len(df), Config.CHUNK_SIZE)):
        chunk_end = min(chunk_start + Config.CHUNK_SIZE, len(df))
        chunk_df = df.iloc[chunk_start:chunk_end].copy()
        
        # Process in batches
        for i in range(0, len(chunk_df), Config.BATCH_SIZE):
            batch = chunk_df.iloc[i:i+Config.BATCH_SIZE]
            points = []
            
            for _, row in batch.iterrows():
                embedding = embedding_gen.get_embedding(row['plot'])
                if embedding is not None:
                    point = models.PointStruct(
                        id=int(row['id']),
                        vector=embedding,
                        payload={
                            'title': str(row['title']),
                            'vote_average': float(row['vote_average']),
                            'genres': prepare_genres(row['genres']),
                            'release_date': str(row['release_date'])
                        }
                    )
                    points.append(point)
                    data_processor.processed_ids.add(row['id'])
                    total_processed += 1
                    if total_processed % 100 == 0:
                        print(f"Processed {total_processed}/{len(df)} movies")
            
            if points:
                qdrant_handler.upsert_batch(points)
            
            # Save progress
            data_processor.save_checkpoint(data_processor.processed_ids)
            
        # Clear memory
        del chunk_df
        gc.collect()
        
        # Prevent timeouts with adaptive sleep
        if chunk_end % (Config.CHUNK_SIZE * 5) == 0:
            time.sleep(0.5)  # Start with shorter sleep
            gc.collect() 
            
# Fetch vector by ID
def get_movie_vector(movie_id, client:QdrantClient):
    # Retrieve points by IDs
    points = client.retrieve(
        collection_name="movie_embeddings",
        ids=[movie_id],
        with_payload=True,
        with_vectors=True,
    )
    
    if points:
        point = points[0]
        return {
            'vector': point.vector,
            'payload': point.payload
        }
    return None

async def generate_candidates_for_two_tower(
    client: QdrantClient,
    movie_ids: List[int],
    num_total_candidates: int = 200,
) -> Dict:
    """
    Generate diverse candidates optimized for Two-Tower model input.
    """
    # Calculate candidates per query to ensure good coverage
    candidates_per_query = min(100, num_total_candidates // len(movie_ids))
    
    # Get group vectors
    group_vectors = []
    for movie_id in movie_ids:
        points = client.retrieve(
            collection_name="movie_embeddings",
            ids=[movie_id],
            with_vectors=True
        )
        if points and points[0].vector:
            group_vectors.append(points[0].vector)
    
    # Batch search for efficiency
    candidates = {}
    seen_movies = set(movie_ids)  # Track to avoid duplicates
    
    # First pass: Get initial candidates
    for vector in group_vectors:
        results = client.search(
            collection_name="movie_embeddings",
            query_vector=vector,
            limit=candidates_per_query,
            with_payload=True,
            with_vectors=True  # Important for Two-Tower input
        )
        
        for result in results:
            if result.id not in seen_movies:
                seen_movies.add(result.id)
                candidates[result.id] = {
                    'vector': result.vector,
                    'payload': result.payload,
                    'similarity_score': result.score
                }
    
    # Prepare for Two-Tower model
    candidate_features = {
        movie_id: {
            'embedding': data['vector'],
            'metadata': data['payload'],
            'similarity_scores': data['similarity_score'],
        }
        for movie_id, data in candidates.items()
    }
    
    return {
        'candidates': candidate_features,
        'total_candidates': len(candidate_features),
        'original_movies': movie_ids
    }
    
def format_candidate_names(candidates_data, client:QdrantClient):
    """
    Format candidate movie names with their similarity scores
    """
    # Format original movies
    original_points = client.retrieve(
        collection_name="movie_embeddings",
        ids=candidates_data['original_movies'],
        with_payload=True
    )
    original_movies = [point.payload['title'] for point in original_points]
    
    # Format and sort candidate movies
    sorted_candidates = sorted(
        candidates_data['candidates'].items(),
        key=lambda x: x[1]['similarity_scores'],
        reverse=True
    )
    
    formatted_candidates = []
    for movie_id, data in sorted_candidates:
        formatted_candidates.append({
            'title': data['metadata']['title'],
            'similarity_score': round(data['similarity_scores'], 3),
            'genres': data['metadata']['genres']
        })
    
    return {
        'total_candidates': candidates_data['total_candidates'],
        'original_movies': original_movies,
        'recommended_movies': formatted_candidates
    }