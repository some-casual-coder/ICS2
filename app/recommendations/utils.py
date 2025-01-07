from typing import Dict
import numpy as np
from .models import GroupPreferences
from .config import GENRE_CONFIG, SCORE_WEIGHTS

def prepare_group_features(group_preferences: GroupPreferences) -> np.ndarray:
    """Convert group preferences into feature vector for model input."""
    features = []
    
    # Genre weights (9 dimensions)
    for genre in GENRE_CONFIG['genre_weights'].keys():
        features.append(group_preferences.genre_weights.get(genre, 0))
    
    # Rating threshold
    features.append(group_preferences.min_rating)
    
    # Runtime preference (normalized)
    runtime_map = {
        'short': 0.0,   # < 90 mins
        'medium': 0.5,  # 90-120 mins
        'long': 1.0     # > 120 mins
    }
    features.append(runtime_map[group_preferences.runtime_preference])
    
    return np.array(features).reshape(1, -1)

def combine_scores(
    hnsw_score: float,
    model_score: float,
    preferences: GroupPreferences,
    metadata: Dict,
    weights: Dict[str, float] = SCORE_WEIGHTS
) -> float:
    """Combine HNSW similarity scores with model predictions and preference bonuses."""
    combined_score = (
        weights['hnsw'] * hnsw_score +
        weights['model'] * model_score
    )
    
    preference_bonus = 0.0
    
    # Language match bonus
    if metadata['original_language'] in preferences.language_preference:
        preference_bonus += 0.1
    
    # Runtime match bonus
    runtime = metadata.get('runtime', 0)
    if preferences.runtime_preference == 'short' and runtime < 90:
        preference_bonus += 0.1
    elif preferences.runtime_preference == 'medium' and 90 <= runtime <= 120:
        preference_bonus += 0.1
    elif preferences.runtime_preference == 'long' and runtime > 120:
        preference_bonus += 0.1
    
    # Release year bonus
    year = int(metadata.get('release_date', '0').split('-')[0])
    if preferences.release_year_range[0] <= year <= preferences.release_year_range[1]:
        preference_bonus += 0.1
    
    return combined_score + (weights['preference_bonus'] * preference_bonus)