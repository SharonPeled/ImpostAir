import lightning as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from pathlib import Path
import datetime
from lightning.pytorch.loggers import MLFlowLogger
import torch

from src.data.GeneralTrajectoryDataModule import GeneralTrajectoryDataModule
from src.utils import get_class_from_path, compose_transforms, denormlize_predictions


def run_training(config):
    """Train a model using the provided config."""
    
    # Set random seed for reproducibility
    pl.seed_everything(config['project']['seed'])
    timestamp = datetime.datetime.now().strftime('%Y_%m_%d_%H%M%S')

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
    run_name = config['project'].get('run_name', 'run')
    # run_name = f"{run_name}_{timestamp}"  # auto logged
    mlflow_logger = MLFlowLogger(
        experiment_name=config['project']['experiment_name'],
        run_name=run_name,
        tracking_uri=config['paths']['mlflow_uri']
    )
    print(f"Initialized run: {config['project']['experiment_name']}.{run_name}")
        
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
        log_every_n_steps=1,
        
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

    if config['saving'].get('train_predictions'):
        df_list = treiner.predict(model, dataloaders=data_module.train_dataloader())
        if df_list:
            dirpath = os.path.join(config['paths']['artifact_dir'], 'train', 'forecasts')
            os.makedirs(dirpath, exist_ok=True)
            save_predictions(df_list, dirpath, config, transform)
    if config['saving'].get('test_predictions'):
        df_list = treiner.predict(model, dataloaders=data_module.train_dataloader())
        if df_list:
            dirpath = os.path.join(config['paths']['artifact_dir'], 'test', 'forecasts')
            os.makedirs(dirpath, exist_ok=True)
            save_predictions(df_list, dirpath, config, transform)
    
    return model, trainer


def save_predictions(df_list, dirpath, config, transform):
    df_cat = pd.concat(df_list, ignore_index=True)
    df_cat = denormlize_predictions(df_cat, transform.transforms[0], config)
    for callsign, df_callsign in df_cat.groupby('callsign'):
        df_callsign.to_csv(os.path.join(dirpath, f'df_{callsign}.csv'), index=False)

    log.info(f'Test predictions saved to {dirpath}')