from openai import OpenAI
import pandas as pd
import time

class MovieEmbeddingGenerator:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
    
    def get_embedding(self, text, max_retries=3):
        text = str(text).strip() if pd.notna(text) else "No plot available."
        for attempt in range(max_retries):
            try:
                response = self.client.embeddings.create(
                    model="text-embedding-3-small",
                    input=text
                )
                return response.data[0].embedding
            except Exception as e:
                if attempt == max_retries - 1:
                    print(f"Failed after {max_retries} attempts: {str(e)}")
                    return None
                time.sleep(2 ** attempt)