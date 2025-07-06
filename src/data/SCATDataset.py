import pytorch_lightning as pl
from torch.utils.data import DataLoader
from src.data.datamodule import TimeSeriesDataModule
from pathlib import Path
import glob
import json
from typing import Dict, List
import pandas as pd
import os
import random


class SCATDataset(TimeSeriesDataModule):
    def __init__(self, config):
        self.config = config
        self.setup(config['paths']['data_dir'], config['data']['input_features'])
        print(f"Loaded {len(self.file_paths)} files")
    
    def setup(self, data_dir: str, input_features: List[str]):
        """Get all JSON files with numeric names from data directory."""
        pattern = Path(data_dir) / '[0-9][0-9][0-9][0-9][0-9][0-9].json'
        self.file_paths = sorted(glob.glob(str(pattern)))
    
    def load_trajectory(self, file_path: str, input_features: List[str]):
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
        
        return pd.DataFrame(plots)
        

    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        return self.load_trajectory(file_path, self.config['data']['input_features'])
    
    def load_trajectory_by_id(self, flight_id: str):
        file_path = next((p for p in self.file_paths if os.path.basename(p).split('.')[0] == flight_id), None)
        if file_path is None:
            raise ValueError(f"Flight ID {flight_id} not found in dataset")
        return self.load_trajectory(file_path, self.config['data']['input_features'])
    
    def load_metadata(self, sample:int = -1):
        metadata_list = []
        if sample == -1:
            sample_paths = self.file_paths
        else:
            sample_paths = random.sample(self.file_paths, sample)
        for file_path in sample_paths:
            metadata = self.extract_flight_metadata(file_path)
            metadata_list.append(metadata)
        self.df_metadata = pd.DataFrame(metadata_list)

    def extract_flight_metadata(self, file_path):
        """Extract metadata from a flight data dictionary."""
        with open(file_path, 'r') as f:
            data = json.load(f)
        metadata = {
            'file_id': None,
            'departure_airport': None,
            'destination_airport': None,
            'aircraft_type': None,
            'callsign': None,
            'flight_rules': None,
            'num_plots': 0,
            'flight_duration_minutes': None,
            'start_time': None,
            'end_time': None
        }
        
        # Extract file ID
        metadata['file_id'] = data.get('id')
        
        # Extract flight plan base info
        if 'fpl' in data and 'fpl_base' in data['fpl'] and data['fpl']['fpl_base']:
            base = data['fpl']['fpl_base'][0]
            metadata.update({
                'departure_airport': base.get('adep'),
                'destination_airport': base.get('ades'),
                'aircraft_type': base.get('aircraft_type'),
                'callsign': base.get('callsign'),
                'flight_rules': base.get('flight_rules')
            })
        
        # Extract trajectory info
        if 'plots' in data and data['plots']:
            plots = data['plots']
            metadata['num_plots'] = len(plots)
            
            if len(plots) > 1:
                start_time = plots[0].get('time_of_track')
                end_time = plots[-1].get('time_of_track')
                
                if start_time and end_time:
                    try:
                        start_dt = pd.to_datetime(start_time)
                        end_dt = pd.to_datetime(end_time)
                        duration = (end_dt - start_dt).total_seconds() / 60
                        metadata.update({
                            'flight_duration_minutes': duration,
                            'start_time': start_time,
                            'end_time': end_time
                        })
                    except:
                        pass
        
        return metadata
