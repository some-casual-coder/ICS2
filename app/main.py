from fastapi import FastAPI

from scripts.embeddings.router import router as embeddings_router
from .genres.router import router as genres_router
from .movies.router import router as movies_router
from .user.router import router as user_router

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Duet Recommendation API"}
    
app.include_router(genres_router)
app.include_router(movies_router)
app.include_router(user_router)
app.include_router(embeddings_router)