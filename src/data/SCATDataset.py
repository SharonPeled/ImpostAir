import json
from typing import Dict, List, Optional
import pandas as pd
from torch.utils.data import Dataset
import torch
from src.data.AbstractDataset import AbstractDataset
from pathlib import Path
import glob
import os
# from src.utils import stable_string_to_bucket_id


import hashlib
from typing import Optional
# TODO: move to utils
def stable_string_to_bucket_id(value: Optional[str], num_buckets: int) -> int:
    """
    Map a string to a stable integer ID in [0, num_buckets).
    Empty/None values map to 0.
    """
    if not value:
        return 0
    # Normalize to uppercase and strip spaces to reduce variants
    normalized = value.strip().upper()
    if normalized == "":
        return 0
    # Use sha256 for stable hashing across processes and runs
    digest = hashlib.sha256(normalized.encode("utf-8")).digest()
    # Convert first 8 bytes to int for speed, then mod buckets
    int_val = int.from_bytes(digest[:8], byteorder="big", signed=False)
    return int_val % max(1, num_buckets)

class SCATDataset(AbstractDataset):
    """PyTorch Dataset for SCAT trajectory data."""
    
    def __init__(self, df: pd.DataFrame, config: dict, transform=None):
        super().__init__(df, config, transform)
        self.input_features = config['data']['input_features']
        self.callsign_num_buckets = config.get('data', {}).get('callsign_num_buckets', 4096)

    
    def load_trajectory(self, file_path: str):
        """Load a single trajectory from file."""
        input_features = self.config['data']['input_features']
        
        with open(file_path, 'r') as f:
            traj_json = json.load(f)
        
        plots = []
        # Resolve callsign from multiple possible JSON layouts
        callsign_val = traj_json.get('callsign')
        if not callsign_val:
            fpl_section = traj_json.get('fpl') or {}
            fpl_base = fpl_section.get('fpl_base')
            if isinstance(fpl_base, list) and len(fpl_base) > 0 and isinstance(fpl_base[0], dict):
                callsign_val = fpl_base[0].get('callsign')
            elif isinstance(fpl_base, dict):
                callsign_val = fpl_base.get('callsign')
            # Fallbacks: sometimes the key might be at root under 'fpl_base' as list/dict
            if not callsign_val:
                root_fpl_base = traj_json.get('fpl_base')
                if isinstance(root_fpl_base, list) and len(root_fpl_base) > 0 and isinstance(root_fpl_base[0], dict):
                    callsign_val = root_fpl_base[0].get('callsign')
                elif isinstance(root_fpl_base, dict):
                    callsign_val = root_fpl_base.get('callsign')
        callsign_id = stable_string_to_bucket_id(callsign_val, self.callsign_num_buckets)
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
                    if 'alt' in input_features:
                        point['alt'] = plot['I062/136'].get('measured_flight_level')
                
                # Extract timestamp
                if 'timestamp' in input_features:
                    point['timestamp'] = plot.get('time_of_track')
                
                # Extract velocity if available
                if 'I062/185' in plot:
                    if 'vx' in input_features:
                        point['vx'] = plot['I062/185'].get('vx')
                    if 'vy' in input_features:
                        point['vy'] = plot['I062/185'].get('vy')
                        
                point['callsign_id'] = callsign_id
                plots.append(point)
        
        return pd.DataFrame(plots)

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
    

