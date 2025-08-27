#!/usr/bin/env python3
"""
Script to prepare Sharada HTR dataset for fine-tuning.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
import logging
from src.data import SharadaHTRDatasetProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Prepare Sharada HTR dataset for fine-tuning."""
    
    logger.info("Starting HTR data preparation...")
    
    # Initialize dataset processor
    processor = SharadaHTRDatasetProcessor(
        dataset_path=cfg.data.dataset_path,
        text_field=cfg.data.text_field,
        image_field=cfg.data.image_field,
        prompt_template=cfg.data.prompt_template,
        image_size=cfg.data.image_size,
        use_augmentation=cfg.data.use_augmentation
    )
    
    # Create dataset
    dataset = processor.create_dataset(
        max_samples=cfg.data.max_samples,
        train_split=cfg.data.train_split
    )
    
    # Save processed dataset
    processor.save_dataset(dataset, cfg.data.output_path)
    
    logger.info("HTR data preparation completed!")
    logger.info(f"Dataset saved to: {cfg.data.output_path}")
    logger.info(f"Train samples: {len(dataset['train'])}")
    logger.info(f"Validation samples: {len(dataset['validation'])}")

if __name__ == "__main__":
    main()
