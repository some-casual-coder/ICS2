import gc
import time
import pandas as pd
from tqdm import tqdm
from qdrant_client.http import models

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