import torch


class GlobalStandardNormalizer:
    def __init__(self, mean, std):
        """
        Initializes the normalization transform with the provided means and stds for each feature.
        
        Args:
            mean (list or tensor): The mean values for each feature (length should match number of features).
            std (list or tensor): The std values for each feature (length should match number of features).
        """
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)

    def __call__(self, sample):
        """
        Applies normalization on the given sample (time series).
        Args:
            sample (dict): Dict with keys 'ts' (tensor) and 'nan_mask' (tensor)
        Returns:
            dict: Updated dict with normalized 'ts' and unchanged 'nan_mask'.
        """
        ts = sample['ts']
        normalized_ts = (ts - self.mean) / self.std
        sample['ts'] = normalized_ts
        return sample
