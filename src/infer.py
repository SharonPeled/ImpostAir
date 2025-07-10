import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import json

from src.models.patch_transformer_model import PatchTransformerTimeSeriesModel
from src.data.SCATDataset import SCATDataset


def run_inference(config: Dict, checkpoint_path: str, flight_ids: Optional[List[str]] = None):
    pass
