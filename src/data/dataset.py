import os
import json
import glob
from typing import List, Dict, Any, Optional
from datasets import Dataset, DatasetDict
import pandas as pd
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console
import logging
from PIL import Image
import torch
from torchvision import transforms

logger = logging.getLogger(__name__)
console = Console()

class SharadaHTRDatasetProcessor:
    """Process Sharada HTR dataset from JSON files and images to HuggingFace Dataset format."""
    
    def __init__(self, dataset_path: str, text_field: str = "original_text", 
                 image_field: str = "image_path", prompt_template: str = "Below is a Sharada manuscript image. The text reads:\n\n{text}\n\n",
                 image_size: List[int] = [1800, 68], use_original_size: bool = True, 
                 use_augmentation: bool = True):
        self.dataset_path = dataset_path
        self.text_field = text_field
        self.image_field = image_field
        self.prompt_template = prompt_template
        self.image_size = image_size
        self.use_original_size = use_original_size
        self.use_augmentation = use_augmentation
        
        # Setup image transforms
        self.setup_transforms()
        
    def setup_transforms(self):
        """Setup image transforms for manuscript line images using original 1800x68 size."""
        # For manuscript line images, we'll use the original 1800x68 dimensions
        # This will create many more patches but preserve all the text detail
        
        if self.use_original_size:
            # Use original size: 1800x68
            # With 16x16 patches: 112x4 = 448 patches (1800/16 = 112, 68/16 = 4)
            
            if self.use_augmentation:
                self.train_transform = transforms.Compose([
                    # Keep original size, just normalize
                    transforms.ToTensor(),
                    # Light augmentation for manuscript images
                    transforms.RandomHorizontalFlip(p=0.05),  # Very low probability for text
                    transforms.RandomRotation(degrees=1),  # Minimal rotation
                    transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.05),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.train_transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            
            self.val_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info(f"Using original size: {self.image_size} (112x4 = 448 patches with 16x16 patch size)")
        
    def find_json_files(self, max_files: Optional[int] = None) -> List[str]:
        """Find all JSON files in the dataset directory structure."""
        json_files = []
        
        # Walk through all subdirectories
        for root, dirs, files in os.walk(self.dataset_path):
            for file in files:
                if file.endswith('.json'):
                    json_files.append(os.path.join(root, file))
                    if max_files and len(json_files) >= max_files:
                        return json_files
        
        return json_files
    
    def load_image_text_pairs(self, json_files: List[str]) -> List[Dict[str, Any]]:
        """Load and process JSON files with corresponding images into a list of dictionaries."""
        data = []
        
        with Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(),
            console=console
        ) as progress:
            task = progress.add_task("Loading JSON files", total=len(json_files))
            
            for json_file in json_files:
                try:
                    with open(json_file, 'r', encoding='utf-8') as f:
                        item = json.load(f)
                    
                    # Extract text based on specified field
                    text = item.get(self.text_field, "")
                    if not text or not text.strip():
                        progress.advance(task)
                        continue
                    
                    # Get image path - construct from JSON file location
                    json_dir = os.path.dirname(json_file)
                    json_name = os.path.splitext(os.path.basename(json_file))[0]
                    image_path = os.path.join(json_dir, f"{json_name}.jpeg")
                    
                    # Check if image exists
                    if not os.path.exists(image_path):
                        logger.warning(f"Image not found: {image_path}")
                        progress.advance(task)
                        continue
                    
                    # Load and resize image if needed
                    try:
                        with Image.open(image_path) as img:
                            width, height = img.size
                            
                            # Resize if dimensions don't match expected size
                            if width != self.image_size[0] or height != self.image_size[1]:
                                logger.info(f"Resizing image {image_path} from {width}x{height} to {self.image_size[0]}x{self.image_size[1]}")
                                img = img.resize(self.image_size, Image.Resampling.LANCZOS)
                            
                            # Save resized image back to the same location
                            img.save(image_path, 'JPEG', quality=95)
                            
                    except Exception as e:
                        logger.warning(f"Error processing image {image_path}: {e}")
                        progress.advance(task)
                        continue
                    
                    # Create formatted text with prompt template
                    formatted_text = self.prompt_template.format(text=text.strip())
                    
                    data.append({
                        'id': item.get('id', ''),
                        'text': formatted_text,
                        'original_text': text.strip(),
                        'image_path': image_path,
                        'font': item.get('font', ''),
                        'font_size': item.get('font_size', 0),
                        'text_source': item.get('text_source', ''),
                        'WX': item.get('WX', ''),  # Keep transliteration for reference
                    })
                    
                except Exception as e:
                    logger.warning(f"Error loading {json_file}: {e}")
                
                progress.advance(task)
        
        return data
    
    def create_dataset(self, max_samples: Optional[int] = None, 
                      train_split: float = 0.95) -> DatasetDict:
        """Create train/validation split from the dataset."""
        
        # Find JSON files
        json_files = self.find_json_files(max_samples)
        logger.info(f"Found {len(json_files)} JSON files")
        
        # Load data
        data = self.load_image_text_pairs(json_files)
        logger.info(f"Loaded {len(data)} valid image-text pairs")
        
        if max_samples and len(data) > max_samples:
            data = data[:max_samples]
            logger.info(f"Limited to {max_samples} samples for overfitting")
        
        # Convert to DataFrame for easier manipulation
        df = pd.DataFrame(data)
        
        # Split into train/val
        train_size = int(len(df) * train_split)
        train_df = df[:train_size]
        val_df = df[train_size:]
        
        logger.info(f"Train samples: {len(train_df)}, Val samples: {len(val_df)}")
        
        # Create DatasetDict
        dataset_dict = DatasetDict({
            'train': Dataset.from_pandas(train_df),
            'validation': Dataset.from_pandas(val_df)
        })
        
        return dataset_dict
    
    def save_dataset(self, dataset: DatasetDict, output_path: str):
        """Save the processed dataset to disk."""
        os.makedirs(output_path, exist_ok=True)
        dataset.save_to_disk(output_path)
        logger.info(f"Dataset saved to {output_path}")
        
        # Save some statistics
        stats = {
            'train_samples': len(dataset['train']),
            'val_samples': len(dataset['validation']),
            'total_samples': len(dataset['train']) + len(dataset['validation']),
            'text_field': self.text_field,
            'image_field': self.image_field,
            'prompt_template': self.prompt_template,
            'image_size': list(self.image_size) if hasattr(self.image_size, '__iter__') else self.image_size,
            'use_original_size': getattr(self, 'use_original_size', True),
            'patches': f"{self.image_size[0]//16}x{self.image_size[1]//16} = {(self.image_size[0]//16) * (self.image_size[1]//16)} patches"
        }
        
        with open(os.path.join(output_path, 'dataset_stats.json'), 'w') as f:
            json.dump(stats, f, indent=2)
        
        logger.info(f"Dataset statistics: {stats}")


def load_processed_dataset(data_path: str) -> DatasetDict:
    """Load a previously processed dataset from disk."""
    # Normalize path and ensure fsspec recognizes local protocol
    if not isinstance(data_path, str):
        data_path = str(data_path)
    abs_path = os.path.abspath(data_path)
    if abs_path.startswith("/"):
        load_path = f"file://{abs_path}"
    else:
        load_path = abs_path
    return DatasetDict.load_from_disk(load_path)
