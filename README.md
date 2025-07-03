# Time Series Forecasting - Modular ML Pipeline

This project provides a modular, extensible pipeline for time series forecasting using PyTorch Lightning and MLflow. The structure is designed for rapid experimentation, easy model/data swapping, and integration of external model codebases (e.g., Chronos).

## Stack
- PyTorch Lightning: Training & abstraction
- MLflow: Experiment tracking
- YAML: Configuration
- Modular code: Easy to extend/replace models, preprocessing, data, etc.

## Quick Start

1. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```
2. Prepare a config file:
   ```bash
   cp config/example.yaml config/exp1.yaml
   # Edit config/exp1.yaml as needed
   ```
3. Train a model:
   ```bash
   python main.py train --config config/exp1.yaml
   ```
4. Run inference:
   ```bash
   python main.py infer --config config/exp1.yaml --checkpoint path/to/model.ckpt
   ```
5. Track experiments with MLflow:
   ```bash
   mlflow ui
   ```

## Structure
- `main.py` - CLI entrypoint
- `config/` - YAML config files
- `src/` - Source code (data, models, training, inference, utils, external wrappers)
- `notebooks/` - EDA, prototyping
- `design/` - Design docs

## Principles
- Modular, config-driven, extensible, and ready for external model integration.