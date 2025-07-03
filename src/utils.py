"""Utility functions for time series forecasting (metrics, helpers, etc.)."""

import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import glob
from collections import Counter
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

def load_trajectory_file(file_path: str) -> Dict:
    """Load a single trajectory JSON file."""
    with open(file_path, 'r') as f:
        return json.load(f)

def extract_flight_metadata(data: Dict) -> Dict:
    """Extract metadata from a flight data dictionary."""
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

def extract_trajectory_data(data: Dict) -> pd.DataFrame:
    """Extract trajectory points from flight data."""
    trajectory_data = []
    
    if 'plots' in data:
        file_id = data.get('id')
        for i, plot in enumerate(data['plots']):
            point = {'file_id': file_id, 'point_index': i}
            
            # Extract coordinates
            if 'I062/105' in plot:
                point['lat'] = plot['I062/105'].get('lat')
                point['lon'] = plot['I062/105'].get('lon')
            
            # Extract altitude
            if 'I062/136' in plot:
                point['altitude'] = plot['I062/136'].get('measured_fl')
            
            # Extract timestamp
            point['timestamp'] = plot.get('time_of_track')
            
            # Extract velocity if available
            if 'I062/185' in plot:
                point['vx'] = plot['I062/185'].get('vx')
                point['vy'] = plot['I062/185'].get('vy')
            
            trajectory_data.append(point)
    
    return pd.DataFrame(trajectory_data)

def get_all_flight_files(data_dir: str) -> List[str]:
    """Get all JSON files with numeric names from data directory."""
    pattern = Path(data_dir) / '[0-9][0-9][0-9][0-9][0-9][0-9].json'
    return glob.glob(str(pattern))

def create_metadata_dataframe(data_dir: str) -> pd.DataFrame:
    """Create a DataFrame with metadata from all flight files."""
    files = get_all_flight_files(data_dir)
    metadata_list = []
    
    for file_path in files:
        try:
            data = load_trajectory_file(file_path)
            metadata = extract_flight_metadata(data)
            metadata['file_path'] = file_path
            metadata_list.append(metadata)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    return pd.DataFrame(metadata_list)

def plot_individual_trajectory(data: Dict, title: Optional[str] = None) -> None:
    """Plot a single flight trajectory."""
    trajectory_df = extract_trajectory_data(data)
    
    if trajectory_df.empty or 'lat' not in trajectory_df.columns or 'lon' not in trajectory_df.columns:
        print("No valid trajectory data found")
        return
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Trajectory on map
    ax1.plot(trajectory_df['lon'], trajectory_df['lat'], 'b-', linewidth=2, alpha=0.7)
    ax1.scatter(trajectory_df['lon'].iloc[0], trajectory_df['lat'].iloc[0], 
                color='green', s=100, label='Start', zorder=5)
    ax1.scatter(trajectory_df['lon'].iloc[-1], trajectory_df['lat'].iloc[-1], 
                color='red', s=100, label='End', zorder=5)
    ax1.set_xlabel('Longitude')
    ax1.set_ylabel('Latitude')
    ax1.set_title('Flight Trajectory')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Altitude over time
    if 'altitude' in trajectory_df.columns:
        ax2.plot(range(len(trajectory_df)), trajectory_df['altitude'], 'g-', linewidth=2)
        ax2.set_xlabel('Point Index')
        ax2.set_ylabel('Altitude (FL)')
        ax2.set_title('Altitude Profile')
        ax2.grid(True, alpha=0.3)
    
    # Plot 3: Speed (if available)
    if 'vx' in trajectory_df.columns and 'vy' in trajectory_df.columns:
        speed = np.sqrt(trajectory_df['vx']**2 + trajectory_df['vy']**2)
        ax3.plot(range(len(trajectory_df)), speed, 'r-', linewidth=2)
        ax3.set_xlabel('Point Index')
        ax3.set_ylabel('Speed (knots)')
        ax3.set_title('Speed Profile')
        ax3.grid(True, alpha=0.3)
    
    # Plot 4: Flight info
    metadata = extract_flight_metadata(data)
    info_text = f"""
    Flight ID: {metadata['file_id']}
    Callsign: {metadata['callsign']}
    Route: {metadata['departure_airport']} → {metadata['destination_airport']}
    Aircraft: {metadata['aircraft_type']}
    Duration: {metadata['flight_duration_minutes']:.1f} min
    Points: {metadata['num_plots']}
    """
    ax4.text(0.1, 0.5, info_text, transform=ax4.transAxes, fontsize=12, 
             verticalalignment='center', bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.axis('off')
    ax4.set_title('Flight Information')
    
    if title:
        fig.suptitle(title, fontsize=16)
    
    plt.tight_layout()
    plt.show()

def plot_summary_statistics(metadata_df: pd.DataFrame) -> None:
    """Plot summary statistics from metadata DataFrame."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
    
    # Plot 1: Top routes
    routes = metadata_df.groupby(['departure_airport', 'destination_airport']).size().reset_index(name='count')
    routes = routes.sort_values('count', ascending=False).head(10)
    route_labels = [f"{row['departure_airport']}→{row['destination_airport']}" for _, row in routes.iterrows()]
    
    ax1.barh(range(len(route_labels)), routes['count'])
    ax1.set_yticks(range(len(route_labels)))
    ax1.set_yticklabels(route_labels)
    ax1.set_xlabel('Number of Flights')
    ax1.set_title('Top 10 Routes by Flight Count')
    ax1.invert_yaxis()
    
    # Plot 2: Airport activity
    all_airports = pd.concat([
        metadata_df['departure_airport'].value_counts(),
        metadata_df['destination_airport'].value_counts()
    ]).groupby(level=0).sum().sort_values(ascending=False).head(15)
    
    ax2.barh(range(len(all_airports)), all_airports.values)
    ax2.set_yticks(range(len(all_airports)))
    ax2.set_yticklabels(all_airports.index)
    ax2.set_xlabel('Number of Flights')
    ax2.set_title('Top 15 Airports by Total Activity')
    ax2.invert_yaxis()
    
    # Plot 3: Flight duration distribution
    if 'flight_duration_minutes' in metadata_df.columns:
        valid_durations = metadata_df['flight_duration_minutes'].dropna()
        ax3.hist(valid_durations, bins=30, alpha=0.7, edgecolor='black')
        ax3.set_xlabel('Flight Duration (minutes)')
        ax3.set_ylabel('Number of Flights')
        ax3.set_title('Flight Duration Distribution')
        ax3.axvline(valid_durations.median(), color='red', linestyle='--', 
                   label=f'Median: {valid_durations.median():.1f} min')
        ax3.legend()
    
    # Plot 4: Number of trajectory points distribution
    if 'num_plots' in metadata_df.columns:
        ax4.hist(metadata_df['num_plots'], bins=30, alpha=0.7, edgecolor='black')
        ax4.set_xlabel('Number of Trajectory Points')
        ax4.set_ylabel('Number of Flights')
        ax4.set_title('Trajectory Points Distribution')
        ax4.axvline(metadata_df['num_plots'].median(), color='red', linestyle='--',
                   label=f'Median: {metadata_df["num_plots"].median():.0f} points')
        ax4.legend()
    
    plt.tight_layout()
    plt.show()

def print_dataset_summary(metadata_df: pd.DataFrame) -> None:
    """Print a summary of the dataset."""
    print("=" * 60)
    print("DATASET SUMMARY")
    print("=" * 60)
    print(f"Total flights: {len(metadata_df)}")
    print(f"Unique departure airports: {metadata_df['departure_airport'].nunique()}")
    print(f"Unique destination airports: {metadata_df['destination_airport'].nunique()}")
    print(f"Unique routes: {metadata_df.groupby(['departure_airport', 'destination_airport']).ngroups}")
    print(f"Unique aircraft types: {metadata_df['aircraft_type'].nunique()}")
    
    if 'flight_duration_minutes' in metadata_df.columns:
        valid_durations = metadata_df['flight_duration_minutes'].dropna()
        print(f"\nFlight Duration Statistics:")
        print(f"  Mean: {valid_durations.mean():.1f} minutes")
        print(f"  Median: {valid_durations.median():.1f} minutes")
        print(f"  Min: {valid_durations.min():.1f} minutes")
        print(f"  Max: {valid_durations.max():.1f} minutes")
    
    if 'num_plots' in metadata_df.columns:
        print(f"\nTrajectory Points Statistics:")
        print(f"  Mean: {metadata_df['num_plots'].mean():.1f} points")
        print(f"  Median: {metadata_df['num_plots'].median():.1f} points")
        print(f"  Min: {metadata_df['num_plots'].min()} points")
        print(f"  Max: {metadata_df['num_plots'].max()} points")
    
    print("\nTop 5 Routes:")
    routes = metadata_df.groupby(['departure_airport', 'destination_airport']).size().reset_index(name='count')
    routes = routes.sort_values('count', ascending=False).head(5)
    for _, row in routes.iterrows():
        print(f"  {row['departure_airport']} → {row['destination_airport']}: {row['count']} flights")
    
    print("=" * 60)
