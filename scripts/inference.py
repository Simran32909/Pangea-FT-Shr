#!/usr/bin/env python3
"""
Inference script for fine-tuned Pangea-7B HTR model on Sharada manuscript images.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import hydra
from omegaconf import DictConfig
import torch
import logging
from transformers import AutoTokenizer, AutoProcessor
from peft import PeftModel, PeftConfig
from src.models import PangeaHTRModel
from PIL import Image
import glob

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@hydra.main(version_base=None, config_path="../configs", config_name="config")
def main(cfg: DictConfig):
    """Run HTR inference with fine-tuned model."""
    
    # Load checkpoint path (you can override this in command line)
    checkpoint_path = cfg.get("checkpoint_path", None)
    if not checkpoint_path:
        # Find the best checkpoint
        checkpoint_dir = cfg.checkpoint.dirpath
        if os.path.exists(checkpoint_dir):
            checkpoints = [f for f in os.listdir(checkpoint_dir) if f.endswith('.ckpt')]
            if checkpoints:
                checkpoint_path = os.path.join(checkpoint_dir, checkpoints[0])
                logger.info(f"Using checkpoint: {checkpoint_path}")
            else:
                logger.error("No checkpoints found!")
                return
        else:
            logger.error(f"Checkpoint directory not found: {checkpoint_dir}")
            return
    
    # Load model from checkpoint
    logger.info("Loading HTR model from checkpoint...")
    model = PangeaHTRModel.load_from_checkpoint(
        checkpoint_path,
        model_name=cfg.model.name,
        vision_model_name=cfg.model.vision_model,
        lora_config=cfg.lora,
        model_config=cfg.model,
        optimizer_config=cfg.optimizer
    )
    
    # Set to evaluation mode
    model.eval()
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Test with sample images from dataset
    logger.info("Testing HTR on sample images...")
    
    # Find some test images
    dataset_path = cfg.data.dataset_path
    test_images = []
    
    # Look for some sample images
    for root, dirs, files in os.walk(dataset_path):
        for file in files:
            if file.endswith('.jpeg') and len(test_images) < 5:
                test_images.append(os.path.join(root, file))
    
    if test_images:
        for i, image_path in enumerate(test_images):
            logger.info(f"\n--- Test {i+1} ---")
            logger.info(f"Image: {image_path}")
            
            try:
                # Generate text from image
                generated_text = model.generate_text_from_image(
                    image_path=image_path,
                    max_length=512,
                    temperature=0.7
                )
                
                logger.info(f"Generated text: {generated_text}")
                logger.info("-" * 50)
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
    else:
        logger.warning("No test images found in dataset")
    
    # Interactive mode
    logger.info("\nEntering interactive mode. Type 'quit' to exit.")
    while True:
        try:
            user_input = input("\nEnter path to Sharada manuscript image (or 'quit'): ")
            if user_input.lower() == 'quit':
                break
            
            if not os.path.exists(user_input):
                print(f"Image not found: {user_input}")
                continue
            
            # Generate text from image
            generated_text = model.generate_text_from_image(
                image_path=user_input,
                max_length=512,
                temperature=0.7
            )
            
            print(f"Generated text: {generated_text}")
            
        except KeyboardInterrupt:
            break
        except Exception as e:
            logger.error(f"Error during generation: {e}")
    
    logger.info("HTR inference completed!")

if __name__ == "__main__":
    main()
