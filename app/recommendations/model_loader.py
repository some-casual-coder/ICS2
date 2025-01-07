from pathlib import Path
import traceback
import tensorflow as tf
from tensorflow.python.keras.models import load_model, Model
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_two_tower_model(model_path: str) -> Optional[Model]:
    """Load the saved two-tower model"""
    try:
        logger.info(f"Attempting to load model from path: {model_path}")
        
        # Check if file exists
        if not Path(model_path).exists():
            logger.error(f"Model file not found at: {model_path}")
            return None
            
        logger.info("File exists, attempting to load...")
        try:
            # Modified to explicitly use keras format
            model = tf.keras.models.load_model(
                model_path,
                compile=False
            )
            logger.info("Model loaded successfully")
            return model
        except Exception as model_error:
            logger.error(f"Error during model loading: {str(model_error)}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            return None
            
    except Exception as e:
        logger.error(f"Unexpected error in load_two_tower_model: {str(e)}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        return None