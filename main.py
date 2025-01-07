from fastapi import FastAPI
from fastapi import FastAPI, HTTPException
from scripts.embeddings.main import main as run_embedding 

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Two-Tower Recommendation API"}

@app.get("/generate_embeddings")
def generate_embeddings():
    try:
        result = run_embedding()
        return {"message": "Embedding process completed", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))