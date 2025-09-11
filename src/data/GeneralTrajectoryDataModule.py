import lightning as pl
from torch.utils.data import DataLoader
from abc import abstractmethod
from typing import List, Optional, Any
import random
from src.utils import get_class_from_path
import pandas as pd


class GeneralTrajectoryDataModule(pl.LightningDataModule):
    """Generic data module for trajectory datasets."""

    def __init__(self, config, transform=None, **dataset_kwargs):
        """Initialize with config dict."""
        super().__init__()
        self.config = config
        self.transform = transform
        self.dataset_kwargs = dataset_kwargs

        self.data_dir = config['paths']['data_dir']
        self.input_features = config['data']['input_features']

        self.dataset_class = get_class_from_path(config['data']['class_path'])  # the actual class (not an instance)

        self.df = self.dataset_class.get_all_ids_df(self.data_dir)  # static method
        print(f"Found {len(self.df)} trajectory files")
        
        self.df_train: pd.DataFrame = pd.DataFrame()
        self.df_val: pd.DataFrame = pd.DataFrame()
        self.df_test: pd.DataFrame = pd.DataFrame()
        
    def setup(self, stage: Optional[str] = None):
        """Setup train/val/test splits."""

        # Sample if debug mode is enabled
        if self.config['debug'].get('sample_size', None):
            df = self.df.sample(self.config['debug']['sample_size']).reset_index(drop=True)
        else:
            df = self.df

        # Split by trajectory using configurable ratios
        total_ids = len(df)
        train_ratio = self.config['data']['train_ratio']
        val_ratio = self.config['data']['val_ratio']
        train_split = int(train_ratio * total_ids)
        val_split = int((train_ratio + val_ratio) * total_ids)
        
        # Shuffle for random split
        shuffled_df = df.sample(frac=1).reset_index(drop=True)
        
        self.df_train = shuffled_df.iloc[:train_split]
        self.df_val = shuffled_df.iloc[train_split:val_split]
        self.df_test = shuffled_df.iloc[val_split:]
        
        print(f"Data splits: Train={len(self.df_train)}, Val={len(self.df_val)}, Test={len(self.df_test)}")
    
    def train_dataloader(self) -> DataLoader:
        """Create training dataloader."""
        dataset = self.dataset_class(self.df_train, self.config, self.transform, **self.dataset_kwargs)
        return DataLoader(
            dataset,
            batch_size=self.config.get('compute', {}).get('batch_size', 1),
            shuffle=True,
            num_workers=self.config.get('compute', {}).get('num_workers', 0),
            drop_last=False
        )
    
    def val_dataloader(self) -> DataLoader:
        """Create validation dataloader."""
        dataset = self.dataset_class(self.df_val, self.config, self.transform, **self.dataset_kwargs)
        return DataLoader(
            dataset,
            batch_size=self.config.get('compute', {}).get('batch_size', 1),
            shuffle=False,
            num_workers=self.config.get('compute', {}).get('num_workers', 0),
            drop_last=False
        )
    
    def test_dataloader(self) -> DataLoader:
        """Create test dataloader."""
        dataset = self.dataset_class(self.df_test, self.config, self.transform, **self.dataset_kwargs)
        return DataLoader(
            dataset,
            batch_size=self.config.get('compute', {}).get('batch_size', 1),
            shuffle=False,
            num_workers=self.config.get('compute', {}).get('num_workers', 0),
            drop_last=False
        )
    
    def predict_dataloader(self) -> DataLoader:
        """Create predict dataloader (defaults to test split)."""
        dataset = self.dataset_class(self.df_test, self.config, self.transform, **self.dataset_kwargs)
        return DataLoader(
            dataset,
            batch_size=self.config.get('compute', {}).get('batch_size', 1),
            shuffle=False,
            num_workers=self.config.get('compute', {}).get('num_workers', 0),
            drop_last=False
        )


