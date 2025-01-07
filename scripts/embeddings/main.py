import os
from dotenv import load_dotenv

from scripts.embeddings.utils import process_movies

load_dotenv()

openai_key = os.getenv("OPENAI_API_KEY")
qdrant_url = os.getenv("QDRANT_URL")
qdrant_key = os.getenv("QDRANT_API_KEY")

def main():
    process_movies(openai_key, qdrant_url, qdrant_key)

if __name__ == "__main__":
    main()