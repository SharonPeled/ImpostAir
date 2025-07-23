import torch


class ImputationTransform:
    def __init__(self, type_, value):
        """
        Initializes the imputation transform with the provided type and value.
        
        Args:
            type_ (str): The type of imputation to apply.
            value (float): The value to use for imputation.
        """
        if type_ != "constant":
            raise NotImplementedError("Only constant imputation is implemented yet")
        self.value = value

    def __call__(self, sample):
        """
        Applies imputation on the given sample (time series).
        Args:
            sample (dict): Dict with keys 'ts' (tensor) and 'nan_mask' (tensor)
        Returns:
            dict: Updated dict with imputed 'ts' and unchanged 'nan_mask'.
        """
        ts = sample['ts']
        ts[torch.isnan(ts)] = self.value
        sample['ts'] = ts
        return sample
