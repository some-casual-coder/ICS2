from fastapi import FastAPI

app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Two-Tower Recommendation API"}

# Add more routes and logic as needed
