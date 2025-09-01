import json
from typing import Dict, List, Optional
import pandas as pd
from torch.utils.data import Dataset
import torch
from abc import abstractmethod
import numpy as np


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
        # timestamps are in milliseconds; handle ISO8601 with/without fractional seconds robustly
        try:
            dt_series = pd.to_datetime(df_trajectory['timestamp'], format='mixed', errors='raise')
        except TypeError:
            # pandas versions without 'mixed' support
            dt_series = pd.to_datetime(df_trajectory['timestamp'], format='ISO8601', errors='raise')
        timestamps = torch.tensor(dt_series.astype(np.int64).values // 10**6)
        df_trajectory.drop(['timestamp', 'file_id'], axis=1, errors='ignore', inplace=True)
        # df_trajectory.drop(['y_detected'], axis=1, errors='ignore', inplace=True)

        # y_detected = torch.tensor(df_trajectory['y_detected'].values).float()
        y_detected = torch.randint(0, 2, (len(df_trajectory),)).float()
        
        ts = torch.tensor(df_trajectory.values).float()
        nan_mask = torch.isnan(ts).any(-1)  # Create a mask where values are NaN

        sample = {'ts': ts, 'nan_mask': nan_mask, 'path': file_path, 'columns': list(df_trajectory.columns), 'timestamps': timestamps, 'y_detected': y_detected}
        if self.transform:
            sample = self.transform(sample)
        return sample
    
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
    
    
    