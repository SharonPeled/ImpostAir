import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional
import json
from src.utils import get_class_from_path, compose_transforms
import lightning as pl
from src.data.GeneralTrajectoryDataModule import GeneralTrajectoryDataModule

def run_inference(config: Dict, checkpoint_path: str, flight_ids: Optional[List[str]] = None):
    """Run next-patch inference using Lightning's predict.

    Returns a list of batch dicts containing raw predictions and masks.
    Optionally filters to provided flight_ids (matching filename stem).
    """
    # Build data module
    transform = compose_transforms(config)
    data_module = GeneralTrajectoryDataModule(config=config, transform=transform)
    data_module.setup(stage="test")

    # Optionally filter dataset by flight_ids on the predict split (defaults to test)
    if flight_ids:
        # Narrow df_test to requested IDs
        id_set = set(str(fid) for fid in flight_ids)
        df = data_module.df_test
        data_module.df_test = df[df['trackid'].astype(str).isin(id_set)].reset_index(drop=True)

    # Load model and set prediction mode
    model_class = get_class_from_path(config['model']['class_path'])
    model = model_class.load_from_checkpoint(checkpoint_path, config=config)
    model.predict_output_mode = 'predictions'
    model.eval()

    trainer = pl.Trainer(
        accelerator=config['compute']['accelerator'],
        devices=config['compute']['devices'],
        logger=False,
        enable_progress_bar=True,
    )
    outputs = trainer.predict(model, datamodule=data_module)

    # Optional save
    outputs_dir = config.get('paths', {}).get('output_dir')
    if outputs_dir:
        out_path = Path(outputs_dir)
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / 'predictions.jsonl'
        with open(out_file, 'w') as f:
            for batch_out in outputs:
                f.write(json.dumps(batch_out) + "\n")

    return outputs


@torch.no_grad()
def run_detection(
    config: Dict,
    checkpoint_path: str,
) -> pd.DataFrame:
    """Run anomaly detection using Lightning's Trainer.predict."""

    # Data
    transform = compose_transforms(config)
    data_module = GeneralTrajectoryDataModule(config=config, transform=transform)
    data_module.setup(stage="test")

    # Load model
    model_class = get_class_from_path(config['model']['class_path'])
    model = model_class.load_from_checkpoint(checkpoint_path, config=config)
    model.eval()

    # Predict
    trainer = pl.Trainer(
        accelerator=config['compute']['accelerator'],
        devices=config['compute']['devices'],
        logger=False,
        enable_progress_bar=True,
    )
    predict_outputs = trainer.predict(model, datamodule=data_module)

    # Aggregate outputs
    rows: List[Dict] = []
    for out in predict_outputs:
        paths = out.get("path", [])
        scores_list = out.get("patch_scores", [])
        flags_list = out.get("is_anomaly", [])
        for b, (p, scores_b, flags_b) in enumerate(zip(paths, scores_list, flags_list)):
            for patch_idx, (score, flag) in enumerate(zip(scores_b, flags_b)):
                rows.append({
                    "path": p,
                    "sample_index": b,
                    "patch_index": patch_idx,
                    "score_se": score,
                    "is_anomaly": flag,
                })

    results_df = pd.DataFrame(rows)

    # Optional: save
    detections_dir = config.get('paths', {}).get('output_dir')
    if detections_dir:
        Path(detections_dir).mkdir(parents=True, exist_ok=True)
        results_df.to_csv(Path(detections_dir) / 'detections.csv', index=False)

    return results_df


