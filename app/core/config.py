from pydantic_settings import BaseSettings
from dotenv import load_dotenv
import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv(Path(ROOT_DIR) / '.env')

class Settings(BaseSettings):
    TMDB_BASE_URL: str = "https://api.themoviedb.org/3"
    TMDB_TOKEN: str
    FIREBASE_CREDS_PATH: str

    @property
    def FIREBASE_CREDS_PATH_ABSOLUTE(self) -> Path:
        """Returns absolute path to Firebase credentials file"""
        return ROOT_DIR / self.FIREBASE_CREDS_PATH

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()