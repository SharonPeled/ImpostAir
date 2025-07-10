import json
from typing import Dict, List, Optional
import pandas as pd
from torch.utils.data import Dataset
import torch
from src.data.AbstractDataset import AbstractDataset
from pathlib import Path
import glob
import os


class SCATDataset(AbstractDataset):
    """PyTorch Dataset for SCAT trajectory data."""
    
    def __init__(self, df: pd.DataFrame, config: dict, transform=None):
        super().__init__(df, config, transform)
        self.input_features = config['data']['input_features']
    
    def load_trajectory(self, file_path: str):
        """Load a single trajectory from file."""
        input_features = self.config['data']['input_features']
        
        with open(file_path, 'r') as f:
            traj_json = json.load(f)
        
        plots = []
        if 'plots' in traj_json:
            file_id = traj_json.get('id')
            for i, plot in enumerate(traj_json['plots']):
                point = {'file_id': file_id}
                
                # Extract coordinates
                if 'I062/105' in plot:
                    if 'lat' in input_features:
                        point['lat'] = plot['I062/105'].get('lat')
                    if 'lon' in input_features:
                        point['lon'] = plot['I062/105'].get('lon')
                
                # Extract altitude
                if 'I062/136' in plot:
                    if 'altitude' in input_features:
                        point['alt'] = plot['I062/136'].get('measured_flight_level')
                
                # Extract timestamp
                if 'time_of_track' in input_features:
                    point['timestamp'] = plot.get('time_of_track')
                
                # Extract velocity if available
                if 'I062/185' in plot:
                    if 'vx' in input_features:
                        point['vx'] = plot['I062/185'].get('vx')
                    if 'vy' in input_features:
                        point['vy'] = plot['I062/185'].get('vy')
                
                plots.append(point)
        
        return pd.DataFrame(plots).values

    @staticmethod
    def get_all_ids_df(data_dir: str) -> pd.DataFrame:
        """Get DataFrame containing all available trajectory IDs."""
        pattern = Path(data_dir) / '[0-9][0-9][0-9][0-9][0-9][0-9].json'
        file_paths = sorted(glob.glob(str(pattern)))
        
        # Create DataFrame with trajectory IDs and file paths
        df = pd.DataFrame({
            'trackid': [os.path.basename(path).replace('.json', '') for path in file_paths],
            'path': file_paths
        })
        
        return df
    

