import pytorch_lightning as pl
from torch.utils.data import DataLoader
from datasets import DatasetDict
from transformers import AutoTokenizer, AutoProcessor
import numpy as np
import torch
from typing import Optional, Dict, Any
import logging
from PIL import Image
import torchvision.io as tvio
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
                 max_seq_length: int = 1024,
                 batch_size: int = 8,
                 num_workers: int = 8):
        super().__init__()
        
        self.dataset = dataset
        self.tokenizer_name = tokenizer_name
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
            # Try to load a processor that actually supports images; otherwise leave as None
            proc = None
            try:
                proc = AutoProcessor.from_pretrained(self.tokenizer_name)
            except Exception:
                proc = None
            # Some models return a tokenizer from AutoProcessor; ensure it has image support
            if proc is not None and not (hasattr(proc, "image_processor") or hasattr(proc, "feature_extractor")):
                proc = None
            self.processor = proc
        
        # Tokenize datasets
        if stage == "fit" or stage is None:
            self.train_dataset = self.process_dataset(self.dataset['train'])
            self.val_dataset = self.process_dataset(self.dataset['validation'])
            
        if stage == "test":
            self.test_dataset = self.process_dataset(self.dataset['validation'])  # Use val as test
    
    def process_dataset(self, dataset):
        """Process a dataset with text tokenization only; defer image I/O to collate_fn."""
        def process_function(examples):
            # Tokenize the texts
            tokenized = self.tokenizer(
                examples['original_text'],
                truncation=True,
                padding=False,
                max_length=self.max_seq_length,
                return_tensors=None,
            )
            
            # Create labels (same as input_ids for causal LM)
            tokenized['labels'] = tokenized['input_ids'].copy()
            
            # Defer image loading; keep paths only for collate-time loading
            tokenized['image_path'] = examples['image_path']
            # Keep raw ground-truth for metric computation
            tokenized['original_text'] = examples['original_text']
            
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
        original_texts = []
        
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
            
            # Load image fast via torchvision.io in worker process
            try:
                img = tvio.read_image(item['image_path'], mode=tvio.ImageReadMode.RGB)
                img = img.to(dtype=torch.float32) / 255.0
            except Exception as e:
                logger.warning(f"Error reading image {item['image_path']}: {e}")
                img = torch.zeros(3, 68, 1800, dtype=torch.float32)
            images.append(img)
            # Keep original text for CER/WER computation
            original_texts.append(item.get('original_text', ""))
        
        return {
            'input_ids': torch.tensor(input_ids, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(labels, dtype=torch.long),
            'images': torch.stack(images),
            'original_text': original_texts,
        }
