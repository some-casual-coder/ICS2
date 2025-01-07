from pydantic_settings import BaseSettings
from dotenv import load_dotenv
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent.parent

load_dotenv(Path(ROOT_DIR) / '.env')

class Settings(BaseSettings):
    TMDB_BASE_URL: str = "https://api.themoviedb.org/3"
    TMDB_TOKEN: str
    FIREBASE_CREDS_PATH: str
    TMDB_METADATA_DATASET_PATH: str
    TMDB_PLOTS_DATASET_PATH: str

    @property
    def FIREBASE_CREDS_PATH_ABSOLUTE(self) -> Path:
        """Returns absolute path to Firebase credentials file"""
        return ROOT_DIR / self.FIREBASE_CREDS_PATH
    
    @property
    def TMDB_METADATA_DATASET_PATH_ABSOLUTE(self) -> Path:
        return ROOT_DIR / self.TMDB_METADATA_DATASET_PATH

    @property
    def TMDB_PLOTS_DATASET_PATH_ABSOLUTE(self) -> Path:
        return ROOT_DIR / self.TMDB_PLOTS_DATASET_PATH

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()