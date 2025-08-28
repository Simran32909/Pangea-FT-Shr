import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoProcessor
import torch
from typing import Optional, Dict, Any
import logging
from PIL import Image
import os
from rich.progress import Progress, TextColumn, BarColumn, TaskProgressColumn, TimeRemainingColumn
from rich.console import Console

logger = logging.getLogger(__name__)
console = Console()

class SharadaHTRDataModule(pl.LightningDataModule):
    """PyTorch Lightning DataModule for Sharada HTR dataset."""
    
    def __init__(self, 
                 dataset: DatasetDict,
                 tokenizer_name: str = "neulab/Pangea-7B",
                 vision_model_name: str = "microsoft/git-base-patch16-224",
                 max_seq_length: int = 1024,
                 batch_size: int = 1,
                 num_workers: int = 4):
        super().__init__()
        
        self.dataset = dataset
        self.tokenizer_name = tokenizer_name
        self.vision_model_name = vision_model_name
        self.max_seq_length = max_seq_length
        self.batch_size = batch_size
        self.num_workers = num_workers
        
        self.tokenizer = None
        self.processor = None
        
    def setup(self, stage: Optional[str] = None):
        """Setup tokenizer and processor, and tokenize datasets."""
        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                trust_remote_code=True,
                padding_side="right"
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if self.processor is None:
            self.processor = AutoProcessor.from_pretrained(self.vision_model_name)
        
        # Tokenize datasets
        if stage == "fit" or stage is None:
            self.train_dataset = self.process_dataset(self.dataset['train'])
            self.val_dataset = self.process_dataset(self.dataset['validation'])
            
        if stage == "test":
            self.test_dataset = self.process_dataset(self.dataset['validation'])  # Use val as test
    
    def process_dataset(self, dataset):
        """Process a dataset with both text tokenization and image loading."""
        def process_function(examples):
            # Tokenize the texts
            tokenized = self.tokenizer(
                examples['text'],
                truncation=True,
                padding=False,
                max_length=self.max_seq_length,
                return_tensors=None,
            )
            
            # Create labels (same as input_ids for causal LM)
            tokenized['labels'] = tokenized['input_ids'].copy()
            
            # Load and process images at original size
            images = []
            
            with Progress(
                TextColumn("[bold green]{task.description}"),
                BarColumn(),
                TaskProgressColumn(),
                TimeRemainingColumn(),
                console=console
            ) as progress:
                task = progress.add_task("Processing images", total=len(examples['image_path']))
                
                for image_path in examples['image_path']:
                    try:
                        image = Image.open(image_path).convert('RGB')
                        
                        # Verify dimensions
                        width, height = image.size
                        if width != 1800 or height != 68:
                            logger.warning(f"Image {image_path} has unexpected dimensions: {width}x{height}")
                        
                        # Process image using vision processor (will handle original size)
                        # The processor should be able to handle variable input sizes
                        processed = self.processor(images=image, return_tensors="pt")
                        images.append(processed['pixel_values'].squeeze(0))
                        
                    except Exception as e:
                        logger.warning(f"Error loading image {image_path}: {e}")
                        # Create a dummy image tensor if loading fails
                        # Use the original size for dummy
                        dummy_image = torch.zeros(3, 68, 1800)  # Original size
                        images.append(dummy_image)
                    
                    progress.advance(task)
            
            tokenized['images'] = images
            
            return tokenized
        
        return dataset.map(
            process_function,
            batched=True,
            remove_columns=dataset.column_names,
            desc="Processing dataset"
        )
    
    def train_dataloader(self):
        """Training dataloader."""
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def val_dataloader(self):
        """Validation dataloader."""
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def test_dataloader(self):
        """Test dataloader."""
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn
        )
    
    def collate_fn(self, batch):
        """Custom collate function to handle variable length sequences and images."""
        # Find max length in this batch
        max_len = max(len(item['input_ids']) for item in batch)
        
        # Pad sequences
        input_ids = []
        attention_mask = []
        labels = []
        images = []
        
        for item in batch:
            seq_len = len(item['input_ids'])
            padding_len = max_len - seq_len
            
            # Pad input_ids
            padded_input_ids = item['input_ids'] + [self.tokenizer.pad_token_id] * padding_len
            input_ids.append(padded_input_ids)
            
            # Pad attention mask
            padded_attention_mask = item['attention_mask'] + [0] * padding_len
            attention_mask.append(padded_attention_mask)
            
            # Pad labels (use -100 for padding tokens)
            padded_labels = item['labels'] + [-100] * padding_len
            labels.append(padded_labels)
            
            # Stack images
            images.append(item['images'])
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'images': torch.stack(images),
        }
