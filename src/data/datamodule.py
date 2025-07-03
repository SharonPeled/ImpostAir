import pytorch_lightning as pl

class TimeSeriesDataModule(pl.LightningDataModule):
    def __init__(self, config):
        """Initialize with config dict."""
        super().__init__()
        self.config = config

    def setup(self, stage=None):
        """Load and preprocess data for train/val/test."""
        pass

    def train_dataloader(self):
        pass

    def val_dataloader(self):
        pass

    def test_dataloader(self):
        pass
