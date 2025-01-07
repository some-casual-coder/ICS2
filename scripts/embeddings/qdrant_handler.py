import time

from qdrant_client import QdrantClient
from qdrant_client.http import models

class QdrantHandler:
    def __init__(self, url, api_key):  # Added api_key parameter
        self.client = QdrantClient(
            url=url,
            api_key=api_key  # Added api_key
        )
        self.collection_name = "movie_embeddings"
    
    def init_collection(self):
        try:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=1536,
                    distance=models.Distance.COSINE
                )
            )
            print(f"Created collection: {self.client.get_collections()}")
        except Exception as e:
            print(f"Collection might already exist: {str(e)}")
    
    def upsert_batch(self, points, max_retries=3):
        for attempt in range(max_retries):
            try:
                self.client.upsert(
                    collection_name=self.collection_name,
                    points=points
                )
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise e
                time.sleep(2 ** attempt)