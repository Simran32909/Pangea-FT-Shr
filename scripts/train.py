#!/usr/bin/env python3
"""
Main training script for Pangea-7B HTR fine-tuning on Sharada data.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger, TensorBoardLogger
import logging
import torch

from src.models import PangeaHTRModel
from src.training import SharadaHTRDataModule
from src.data import load_processed_dataset

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Main training function for HTR."""
    
    # Set seed for reproducibility
    pl.seed_everything(cfg.seed)
    
    logger.info("Starting HTR training...")
    logger.info(f"Configuration: {cfg}")
    
    # Load processed dataset
    logger.info("Loading processed HTR dataset...")
    dataset = load_processed_dataset(cfg.data.output_path)
    
    # Initialize data module
    data_module = SharadaHTRDataModule(
        dataset=dataset,
        tokenizer_name=cfg.model.name,
        vision_model_name=cfg.model.vision_model,
        max_seq_length=cfg.data.max_seq_length,
        batch_size=cfg.trainer.get("batch_size", 1),
        num_workers=cfg.trainer.get("num_workers", 4)
    )
    
    # Initialize model
    logger.info("Initializing HTR model...")
    model = PangeaHTRModel(
        model_name=cfg.model.name,
        vision_model_name=cfg.model.vision_model,
        lora_config=cfg.lora,
        model_config=cfg.model,
        optimizer_config=cfg.optimizer,
        metrics_config=cfg.metrics
    )
    
    # Setup callbacks
    callbacks = []
    
    # Model checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        dirpath=cfg.checkpoint.dirpath,
        filename=cfg.checkpoint.filename,
        monitor=cfg.trainer.monitor,
        mode=cfg.trainer.mode,
        save_top_k=cfg.trainer.save_top_k,
        save_last=True,
    )
    callbacks.append(checkpoint_callback)
    
    # Learning rate monitor
    lr_monitor = LearningRateMonitor(logging_interval='step')
    callbacks.append(lr_monitor)
    
    # Setup loggers
    loggers = []
    
    # TensorBoard logger
    tb_logger = TensorBoardLogger(
        save_dir=cfg.logging.save_dir,
        name=cfg.logging.name,
        version=None
    )
    loggers.append(tb_logger)
    
    # WandB logger
    if cfg.logging.get("use_wandb", False):
        wandb_logger = WandbLogger(
            project=cfg.logging.project,
            name=cfg.logging.name,
            entity=cfg.logging.get("wandb_entity", None),
            log_model=cfg.logging.log_model,
            tags=["htr", "sharada", "pangea-7b", "lora"]
        )
        loggers.append(wandb_logger)
        logger.info("WandB logging enabled")
    
    # Initialize trainer
    trainer = pl.Trainer(
        max_epochs=cfg.trainer.max_epochs,
        gradient_clip_val=cfg.trainer.gradient_clip_val,
        accumulate_grad_batches=cfg.trainer.accumulate_grad_batches,
        precision=cfg.trainer.precision,
        accelerator=cfg.trainer.accelerator,
        devices=cfg.trainer.devices,
        strategy=cfg.trainer.strategy,
        log_every_n_steps=cfg.trainer.log_every_n_steps,
        val_check_interval=cfg.trainer.val_check_interval,
        callbacks=callbacks,
        logger=loggers,
        deterministic=False,  # Set to True for exact reproducibility
    )
    
    # Start training
    logger.info("Starting HTR training...")
    trainer.fit(model, data_module)
    
    # Save final model
    logger.info("HTR training completed!")
    logger.info(f"Best model saved at: {checkpoint_callback.best_model_path}")
    
    # Test the model
    logger.info("Running validation...")
    trainer.test(model, data_module)
    
    return model

if __name__ == "__main__":
    main()
