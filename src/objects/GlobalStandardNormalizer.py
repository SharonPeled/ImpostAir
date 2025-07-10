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
            sample (tensor): The input tensor with shape [time_steps, num_features].
        
        Returns:
            tensor: The normalized tensor with the same shape as the input.
        """
        # Normalize each feature independently (across time steps)
        normalized_sample = (sample - self.mean) / self.std
        return normalized_sample