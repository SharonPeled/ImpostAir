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
        self.callsign_to_idx_dict = {callsign: idx for idx, callsign in enumerate(df['callsign'].unique())}
        self.df_track_annotations = self.load_track_annotations(self.config['paths'].get('track_anomaly_annotations_filepath'))
        

    def __len__(self):
        """Return number of trajectories in dataset."""
        return len(self.df)
    
    def __getitem__(self, idx):
        """Load and return a single trajectory."""
        file_path = self.df.iloc[idx]['path']
        trackid = self.df.iloc[idx]['trackid']
        callsign = self.df.iloc[idx]['callsign']
        callsign_idx = self.callsign_to_idx_dict[callsign]
        
        df_trajectory = self.load_trajectory(file_path)
        dt_series = pd.to_datetime(df_trajectory['timestamp'], format='ISO8601', errors='raise')
        timestamps = torch.tensor(dt_series.astype(np.int64).values // 10**6)

        df_trajectory.drop(['timestamp', 'file_id'], axis=1, errors='ignore', inplace=True)

        y_track_is_anomaly = self.df_track_annotations.is_anomaly.loc[trackid]
        
        ts = torch.tensor(df_trajectory.values).float()
        nan_mask = torch.isnan(ts).any(-1)  # Create a mask where values are NaN

        sample = {'ts': ts, 'nan_mask': nan_mask, 'path': file_path, 'columns': list(df_trajectory.columns), 
        'timestamps': timestamps, 'y_track_is_anomaly': y_track_is_anomaly, 'callsign': callsign, 
        'callsign_idx': callsign_idx}
        
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
    
    def load_track_annotations(self, file_path: [str, None]):
        df_annt = pd.DataFrame({
                'trackid': self.df.trackid.values,
                'is_anomaly': [np.nan] * len(self.df)
            }, index=self.df.trackid.values)
        
        if file_path is None: 
            return df_annt
        
        df_all_annt = pd.read_csv(file_path)
        df_all_annt.trackid = df_all_annt.trackid.astype(str)
        df_all_annt = df_all_annt[df_all_annt.trackid.isin(self.df.trackid.values)].set_index('trackid')

        df_annt.loc[df_all_annt.index, 'is_anomaly'] = df_all_annt.is_anomaly.values
        
        return df_annt
    
    
    