import pandas as pd
from typing import List, Dict, Optional
import json
from pathlib import Path
from fastapi import HTTPException
import asyncio
from datetime import datetime

from app.discover.models import DiscoverFilters

class DiscoverService:
    def __init__(self):
        self.metadata_df = None
        self.plots_df = None
        self._load_datasets()

    def _load_datasets(self):
        """Load and preprocess the datasets"""
        try:
            # Load datasets
            metadata_path = Path("datasets/tmdb_movie_metadata_v1.csv")
            plots_path = Path("datasets/tmdb_movie_plots_v1.csv")
            
            self.metadata_df = pd.read_csv(metadata_path)
            self.plots_df = pd.read_csv(plots_path)
            
            # Preprocess metadata
            self.metadata_df['genres'] = self.metadata_df['genres'].apply(lambda x: [g.strip() for g in str(x).split(',')])
            self.metadata_df['spoken_languages'] = self.metadata_df['spoken_languages'].apply(lambda x: [l.strip() for l in str(x).split(',')])
            self.metadata_df['release_date'] = pd.to_datetime(self.metadata_df['release_date'])
            
            # Filter to only include movies with plots
            valid_ids = set(self.plots_df['id'].unique())
            self.metadata_df = self.metadata_df[self.metadata_df['id'].isin(valid_ids)]
            
        except Exception as e:
            raise Exception(f"Error loading datasets: {str(e)}")

    def _filter_by_runtime(self, df: pd.DataFrame, preference: str) -> pd.DataFrame:
        """Filter movies by runtime preference"""
        if preference == "short":
            return df[df['runtime'] <= 90]
        elif preference == "medium":
            return df[(df['runtime'] > 90) & (df['runtime'] <= 150)]
        elif preference == "long":
            return df[df['runtime'] > 150]
        return df

    def _filter_by_languages(self, df: pd.DataFrame, languages: List[str]) -> pd.DataFrame:
        """Filter movies by language preferences"""
        return df[df['original_language'].isin(languages)]

    def _filter_by_rating(self, df: pd.DataFrame, min_rating: float) -> pd.DataFrame:
        """Filter movies by minimum rating"""
        return df[df['vote_average'] >= min_rating]

    def _filter_by_years(self, df: pd.DataFrame, year_range: tuple) -> pd.DataFrame:
        """Filter movies by release year range"""
        start_year, end_year = year_range
        return df[(df['release_date'].dt.year >= start_year) & 
                 (df['release_date'].dt.year <= end_year)]

    def _filter_by_genres(self, df: pd.DataFrame, genre_ids: List[int]) -> pd.DataFrame:
        """Filter movies by genres"""
        # Map genre IDs to names (you might want to store this mapping somewhere)
        genre_map = {
            28: "Action",
            12: "Adventure",
            16: "Animation",
            35: "Comedy",
            80: "Crime",
            99: "Documentary",
            18: "Drama",
            10751: "Family",
            14: "Fantasy",
            36: "History",
            27: "Horror",
            10402: "Music",
            9648: "Mystery",
            10749: "Romance",
            878: "Science Fiction",
            10770: "TV Movie",
            53: "Thriller",
            10752: "War",
            37: "Western"
        }
        
        target_genres = [genre_map[gid] for gid in genre_ids if gid in genre_map]
        
        def has_genres(movie_genres, target_genres):
            return all(any(t.lower() in g.lower() for g in movie_genres) for t in target_genres)
        
        return df[df['genres'].apply(lambda x: has_genres(x, target_genres))]

    async def get_discover_movies(
        self,
        filters: DiscoverFilters,
        watched_movies: Optional[List[int]] = None,
        page: int = 1,
        per_page: int = 20
    ) -> Dict:
        """Get filtered movies based on preferences"""
        try:
            df = self.metadata_df.copy()
            
            # Apply filters
            if filters.runtime_preference:
                df = self._filter_by_runtime(df, filters.runtime_preference)
            
            if filters.language_preference:
                df = self._filter_by_languages(df, filters.language_preference)
            
            if filters.min_rating:
                df = self._filter_by_rating(df, filters.min_rating)
            
            if filters.release_year_range:
                df = self._filter_by_years(df, filters.release_year_range)
            
            if filters.genres:
                df = self._filter_by_genres(df, filters.genres)
            
            # Exclude watched movies
            if filters.exclude_watched and watched_movies:
                df = df[~df['id'].isin(watched_movies)]
            
            # Sort by popularity and rating
            df = df.sort_values(['popularity', 'vote_average'], ascending=[False, False])
            
            # Paginate results
            total_results = len(df)
            total_pages = (total_results + per_page - 1) // per_page
            
            start_idx = (page - 1) * per_page
            end_idx = start_idx + per_page
            
            page_df = df.iloc[start_idx:end_idx]
            
            # Format results
            results = page_df.to_dict('records')
            
            return {
                "page": page,
                "total_pages": total_pages,
                "total_results": total_results,
                "results": results
            }
            
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))