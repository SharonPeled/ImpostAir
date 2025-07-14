import json
from typing import Dict, List, Optional
import pandas as pd
from torch.utils.data import Dataset
import torch
from abc import abstractmethod


class AbstractDataset(Dataset):
    """Abstract Dataset.
       Defines the additional methods that a dataset should implement.
    """
    
    def __init__(self, df: pd.DataFrame, config: dict, transform=None):
        self.config = config
        self.transform = transform  
        self.df = df

    def __len__(self):
        """Return number of trajectories in dataset."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """Load and return a single trajectory."""
        file_path = self.df.iloc[idx]['path']

        df_trajectory = self.load_trajectory(file_path)

        if 'timestamp' in self.config['data']['input_features']:
            df_trajectory = df_trajectory.drop('timestamp', axis=1)
        
        ts = torch.tensor(df_trajectory.values).float()
        if self.transform:
            ts = self.transform(ts)
        return {
            'ts': ts
        }
    
    @abstractmethod
    def load_trajectory(self, file_path: str):
        """Load a single trajectory from file."""
        raise NotImplementedError("Subclasses must implement load_trajectory")

    @staticmethod
    @abstractmethod
    def get_all_ids_df(data_dir: str) -> pd.DataFrame:
        """Get mapping of trajectory IDs to their file paths.
        
        Returns:
            pd.DataFrame: DataFrame with columns 'trackid' and 'path' mapping trajectory IDs to their full file paths
        """
        raise NotImplementedError("Subclasses must implement get_all_ids_df")
    
    
    