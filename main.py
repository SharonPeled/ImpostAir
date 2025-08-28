import argparse
import yaml
import sys
from src.train import run_training
from src.infer import run_inference
import os 
from pathlib import Path
from typing import Dict, Any
from omegaconf import OmegaConf


def load_config(config_path: str) -> Dict[str, Any]:
    """Load and validate configuration from YAML file using OmegaConf."""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    OmegaConf.register_new_resolver("eval", eval)

    config = OmegaConf.load(config_file)
    
    return config


def main():
    
    parser = argparse.ArgumentParser(description="Time Series Forecasting Pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Train command
    train_parser = subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--config", required=True, help="Path to YAML config file")

    # Inference command
    infer_parser = subparsers.add_parser("infer", help="Run inference with a trained model")
    infer_parser.add_argument("--config", required=True, help="Path to YAML config file")
    infer_parser.add_argument("--checkpoint", required=True, help="Path to model checkpoint")

    args = parser.parse_args()
    config = load_config(args.config)

    if args.command == "train":
        run_training(config)
    elif args.command == "infer":
        run_inference(config, args.checkpoint)
    else:
        print("Unknown command", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
