import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import datetime
from lightning.pytorch.loggers import MLFlowLogger
import torch

from src.data.GeneralTrajectoryDataModule import GeneralTrajectoryDataModule
from src.utils import get_class_from_path, compose_transforms


def run_training(config):
    """Train a model using the provided config."""
    
    # Set random seed for reproducibility
    pl.seed_everything(config['project']['seed'])
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')

    torch.autograd.set_detect_anomaly(True)

    # initialize transforms
    print("Initializing transforms...")
    transform = compose_transforms(config)
    print(transform)
    
    # Initialize data module
    print("Setting up data module...")
    data_module = GeneralTrajectoryDataModule(config=config, transform=transform)
    data_module.setup()
    
    # Initialize model
    print("Initializing model...")
    model_class = get_class_from_path(config['model']['class_path'])
    model = model_class(config=config)
    
    # Setup callbacks
    callbacks = []
    
    # Early stopping
    if config['training'].get('early_stopping_monitor', None) is not None:
        early_stopping = EarlyStopping(
            monitor=config['training']['early_stopping_monitor'],
            patience=config['training']['early_stopping_patience'],
            mode='min',
            verbose=True
        )
        callbacks.append(early_stopping)
    
    # Model checkpointing
    if config['paths'].get('checkpoint_dir', None) is not None:
        checkpoint_dir = Path(config['paths']['checkpoint_dir'])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        experiment_name = config['project']['experiment_name']
        run_name = config['project']['run_name']
        model_checkpoint = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename=f"{timestamp}-{experiment_name}-{run_name}-{{epoch:02d}}-{{val_loss:.2f}}",
            monitor=None,  # We're not tracking any metric
            mode='min',
            save_top_k=1,  # Only keep one checkpoint
            every_n_epochs=1,
            save_on_train_epoch_end=True,  # or False depending on when you want to save
            verbose=True
        )
        callbacks.append(model_checkpoint)
    
    # MLflow logger setup
    assert 'mlflow_uri' in config['paths'], "mlflow_uri must be specified in config"
    run_name_base = config['project'].get('run_name', 'run')
    run_name = f"{run_name_base}_{timestamp}"
    mlflow_logger = MLFlowLogger(
        experiment_name=config['project']['experiment_name'],
        run_name=run_name,
        tracking_uri=config['paths']['mlflow_uri']
    )
        
    trainer = pl.Trainer(
        max_epochs=config['training']['max_epochs'],
        accelerator=config['compute']['accelerator'],
        devices=config['compute']['devices'],
        
        # Gradient accumulation configuration
        accumulate_grad_batches=config['training'].get('accumulate_grad_batches', 1),
        
        # Gradient clipping
        gradient_clip_val=config['training'].get('gradient_clip_val'),
        
        # Logging and checkpointing
        logger=mlflow_logger,
        callbacks=callbacks,
        
        # Progress bar and logging
        enable_progress_bar=True,
        # log_every_n_steps=1,
        
        # Validation
        check_val_every_n_epoch=1,
        
        enable_model_summary=True,
        
        # Deterministic training
        deterministic=True
    )
    
    # Train the model
    print("Starting training...")
    trainer.fit(model, data_module)
    
    # Print best checkpoint info
    print(f"\n✓ Training completed!")
    print(f"Model checkpoints saved to: {config['paths'].get('checkpoint_dir', None)}")

    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_results = trainer.test(model, datamodule=data_module)
    print(f"Test results: {test_results}")
    
    return model, trainer
