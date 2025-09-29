import torch
import torch.nn.functional as F


class PaddingTransform:
    def __init__(self, max_len, pad_value, pad_nans=False):
        """
        Initializes the padding transform.

        Args:
            max_len (int): The target maximum sequence length to pad to.
            pad_value (float): The constant value to use for padding the time series.
            pad_nans (bool): Whether to pad all Nan values (impute with pad_value).
        """
        # Keep validations minimal to preserve original interface intent
        self.max_len = int(max_len)
        self.pad_value = pad_value
        self.pad_nans = pad_nans

    def __call__(self, sample):
        """
        Pads the given sample (time series) at the end up to max_len with a constant.

        Args:
            sample (dict): Dict with keys 'ts' (T x C tensor) and 'nan_mask' (T tensor)

        Returns:
            dict: Updated dict with padded 'ts' and updated 'nan_mask'.
        """
        ts = sample['ts']
        nan_mask = sample['nan_mask']
        timestamps = sample['timestamps']
        T, _ = ts.shape

        if T < self.max_len:
            pad_size = self.max_len - T
            # Pad rows at the end (second-to-last dim right)
            ts = F.pad(ts, (0, 0, 0, pad_size), mode='constant', value=float(self.pad_value))
            # Mark padded positions as True in the mask (indicating padded/missing)
            nan_mask = F.pad(nan_mask, (0, pad_size), mode='constant', value=True)
            # padding timestamps 
            timestamps = F.pad(timestamps, (0, pad_size), mode='constant', value=True)

        # truncate ts if longer than max_len
        sample['ts'] = ts[-self.max_len:]
        sample['nan_mask'] = nan_mask[-self.max_len:]
        sample['timestamps'] = timestamps[-self.max_len:]
        return sample