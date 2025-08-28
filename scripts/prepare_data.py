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
from rich.console import Console
from rich.panel import Panel
from rich.text import Text
from src.data import SharadaHTRDatasetProcessor

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
console = Console()

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Prepare Sharada HTR dataset for fine-tuning."""
    
    console.print(Panel.fit(
        "[bold blue]Sharada HTR Dataset Preparation[/bold blue]",
        border_style="blue"
    ))
    
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
    
    # Display results
    train_samples = len(dataset['train'])
    val_samples = len(dataset['validation'])
    total_samples = train_samples + val_samples
    
    console.print(Panel(
        f"[bold green]Dataset Preparation Complete![/bold green]\n\n"
        f"[bold]Dataset saved to:[/bold] {cfg.data.output_path}\n"
        f"[bold]Total samples:[/bold] {total_samples:,}\n"
        f"[bold]Train samples:[/bold] {train_samples:,}\n"
        f"[bold]Validation samples:[/bold] {val_samples:,}\n"
        f"[bold]Image size:[/bold] {cfg.data.image_size[0]}x{cfg.data.image_size[1]}",
        title="[bold green]✓ Success[/bold green]",
        border_style="green"
    ))
    
    logger.info("HTR data preparation completed!")

if __name__ == "__main__":
    main()
