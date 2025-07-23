import torch
import torch.nn.functional as F


class PaddingTransform:
    def __init__(self, objective, type_, patch_len):
        """
        Initializes the padding transform.
        
        Args:
            objective (str): The objective of the padding.
            type_ (str): The type of padding to apply.
            patch_len (int): The length of the patch to pad.
        """
        if objective != "patching":
            raise NotImplementedError("Only patching padding is implemented yet")
        if type_ != "nan_padding":
            raise NotImplementedError("Only nan padding is implemented yet")
        self.patch_len = patch_len

    def __call__(self, sample):
        """
        Applies padding on the given sample (time series).
        Args:
            sample (dict): Dict with keys 'ts' (tensor) and 'nan_mask' (tensor)
        Returns:
            dict: Updated dict with padded 'ts' and unchanged 'nan_mask'.
        """
        ts = sample['ts']
        nan_mask = sample['nan_mask']
        T, _ = ts.shape

        # handle padding when sequence length is not divisible by patch_len
        if T % self.patch_len != 0:
            # Calculate padding needed at start
            pad_size = self.patch_len - (T % self.patch_len)
            ts = F.pad(ts, (0, 0, pad_size, 0), mode='constant', value=torch.nan)
            nan_mask = F.pad(nan_mask, (0, 0, pad_size, 0), mode='constant', value=True)
        
        sample['ts'] = ts
        sample['nan_mask'] = nan_mask

        return sample
