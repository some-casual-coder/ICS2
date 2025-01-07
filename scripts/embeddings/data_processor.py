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
        
        plots_df = pd.read_csv(Config.PLOTS_FILE, dtype={'id': 'int32'})
        metadata_df = pd.read_csv(Config.METADATA_FILE, dtype=dtypes)
        
        return plots_df.merge(metadata_df, on='id')