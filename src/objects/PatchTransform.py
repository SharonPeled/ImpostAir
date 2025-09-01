import torch


class PatchTransform:
    def __init__(self, patch_len: int, patch_nan_tolerance_percentage: float):
        """
        Initializes the patch transform that converts time series and nan masks into patch structure.
        
        Args:
            patch_len (int): The length of each patch.
            patch_nan_tolerance_percentage (float): The percentage of NaN values allowed in a patch 
                                                  before it's considered invalid (0.0 to 1.0).
        """
        self.patch_len = patch_len
        self.patch_nan_tolerance_percentage = patch_nan_tolerance_percentage

    def __call__(self, sample):
        """
        Converts time series and nan mask into patch structure.
        
        Args:
            sample (dict): Dict with keys 'ts' (tensor) and 'nan_mask' (tensor)
                          Expected shapes: ts [num_steps, num_features], 
                                         nan_mask [num_steps,]
        
        Returns:
            dict: Updated dict with:
                - 'ts_patches': [num_patches, patch_len * num_features]
                - 'nan_mask': [num_patches, patch_len]
        """
        ts = sample['ts']
        nan_mask = sample['nan_mask']
        y_detected = sample['y_detected']
        
        T, C = ts.shape
        
        assert T % self.patch_len == 0, "Time series is not divisible by patch_len"
        # Calculate number of patches
        N = T // self.patch_len
        
        # Create patches for time series
        ts_patches = ts.reshape(N, self.patch_len, C)
        y_detected_patches = y_detected.reshape(N, self.patch_len)
        
        # Create patches for nan mask
        nan_mask_patches = nan_mask.reshape(N, self.patch_len)
        
        # Generate padding mask (True = ignore/pad)
        nan_counts = nan_mask_patches.sum(dim=-1).float()  # [num_patches, ]
        patch_nan_percentage = nan_counts / self.patch_len  # [num_patches, ]
        nan_mask_patches_binary = patch_nan_percentage > self.patch_nan_tolerance_percentage  # [num_patches, ]

        # Ensure the first patch is valid (not masked out), otherwise advance to the first valid patch.
        # This is necessary for transformer masking: if the first token is filtered, the rest of the sequence
        # can result in NaN after the forward pass.
        nan_mask_patches_binary[0] = False
        
        # Update sample
        sample['ts'] = ts_patches
        sample['nan_mask'] = nan_mask_patches_binary
        sample['y_detected'] = y_detected_patches
        
        return sample