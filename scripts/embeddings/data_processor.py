import os
import numpy as np
import pandas as pd

from scripts.embeddings.config import Config

class DataProcessor:
    def __init__(self):
        self.processed_ids = self.load_checkpoint()
    
    @staticmethod
    def load_checkpoint():
        try:
            return set(np.load(Config.CHECKPOINT_FILE))
        except:
            return set()
    
    def save_checkpoint(self, processed_ids):
        np.save(Config.CHECKPOINT_FILE, list(processed_ids))
    
    def load_and_prepare_data(self):
        dtypes = {
            'id': 'int32',
            'vote_count': 'float32',
            'vote_average': 'float32',
            'runtime': 'float32',
            'popularity': 'float32'
        }
        
        script_dir = os.path.dirname(os.path.realpath(__file__))  # Get the directory of the current script
        plots_path = os.path.join(script_dir, Config.PLOTS_FILE)
        metadata_path = os.path.join(script_dir, Config.METADATA_FILE)
        
        plots_df = pd.read_csv(plots_path, dtype={'id': 'int32'})
        metadata_df = pd.read_csv(metadata_path, dtype=dtypes)
        
        return plots_df.merge(metadata_df, on='id')