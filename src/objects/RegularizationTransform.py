import torch
import scipy
import numpy as np
import pandas as pd

class RegularizationTransform:
    def __init__(self, type_, time_interval, error_tolerance):
        """
        Initializes the regularization (making irregular time series regular) transform with the provided type and error tolerance.
        
        Args:
            type_ (str): The type of regularization to apply.
            time_interval (float): The time interval for linear interpolation.
            error_tolerance (float): The error tolerance for linear interpolation.
        """
        if type_ != "linear_interpolation":
            raise NotImplementedError("Only linear interpolation is implemented.")
        self.time_interval = time_interval
        self.error_tolerance = error_tolerance

    def __call__(self, sample):
        """
        Applies regularization on the given sample (time series).
        Args:
            sample (dict): Dict with keys 'ts' (tensor) and 'nan_mask' (tensor)
        Returns:
            dict: The input sample dict with 'ts' regularized (missing values interpolated) and 'nan_mask' updated to True where 
            interpolation error exceeds the error_tolerance.
        """
        ts = sample['ts']
        timestamps = sample['timestamps']
        nan_mask = sample['nan_mask']

        # Create regular grid
        regular_index = create_regular_timestamps_ms(
            timestamps[0], 
            timestamps[-1], 
            self.time_interval * 1000
        )
        
        # Convert to relative time (milliseconds from start)
        time_relative = timestamps - timestamps[0]
        regular_relative = regular_index - timestamps[0]
        
        n_features = ts.shape[1]
        interpolated_values = np.zeros((len(regular_index), n_features))
        

        valid_times = time_relative[~nan_mask]
        valid_ts = ts[~nan_mask]

        # Only interpolate if we have enough valid data points
        # Not enough data points, fill with NaN
        if len(valid_times) <= 1:
            interpolated_values = np.nan

        else:
            # Interpolate each feature
            for i in range(n_features):
                interpolator = scipy.interpolate.interp1d(
                                    valid_times, 
                                    valid_ts[:, i], 
                                    kind='linear',
                                    bounds_error=False,
                                    fill_value=np.nan  # Fill with NaN outside data range
                                )
                interpolated_values[:, i] = interpolator(regular_relative)
                    
        sample['ts'] = torch.tensor(interpolated_values, dtype=ts.dtype, device=ts.device)
        sample['timestamps'] = regular_index

        # Update nan_mask: mark as True where the distance to the nearest observed timestamp exceeds error_tolerance
        dist_to_nearest = torch.min(torch.abs(valid_times[:, None] - regular_relative[None, :]), dim=0).values
        sample['nan_mask'] = dist_to_nearest > self.error_tolerance 
        
        return sample

       
def create_regular_timestamps_ms(start_ms, end_ms, time_interval_ms):
    """
    Create a regular timestamp array from start to end with constant intervals in milliseconds.
    
    Args:
        start_ms: int, start timestamp in milliseconds
        end_ms: int, end timestamp in milliseconds  
        time_interval_ms: int, interval in milliseconds
    
    Returns:
        pd.Series with regular millisecond timestamps
    """
    return torch.tensor(np.arange(start_ms, end_ms, time_interval_ms))