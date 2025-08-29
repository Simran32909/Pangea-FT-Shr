import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, AutoProcessor
from peft import LoraConfig, get_peft_model, TaskType
import pytorch_lightning as pl
from typing import Dict, Any, Optional
import logging
from PIL import Image
import torchvision.transforms as transforms
import numpy as np
from jiwer import wer
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import re

logger = logging.getLogger(__name__)

class PangeaHTRModel(pl.LightningModule):
    """PyTorch Lightning module for Pangea-7B with vision encoder for HTR fine-tuning."""
    
    def __init__(self, 
                 model_name: str = "neulab/Pangea-7B",
                 lora_config: Dict[str, Any] = None,
                 model_config: Dict[str, Any] = None,
                 optimizer_config: Dict[str, Any] = None,
                 metrics_config: Dict[str, Any] = None):
        super().__init__()
        self.save_hyperparameters()
        
        self.model_name = model_name
        self.lora_config = lora_config or {}
        self.model_config = model_config or {}
        self.optimizer_config = optimizer_config or {}
        self.metrics_config = metrics_config or {}
        
        # Initialize model and tokenizer
        self.tokenizer = None
        self.processor = None
        self.model = None
        self.vision_encoder = None
        self.setup_model()
        
    def setup_model(self):
        """Setup the model with vision encoder and LoRA configuration."""
        logger.info(f"Loading model: {self.model_name}")
        
        # Load tokenizer and processor
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            padding_side="right"
        )
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load processor from the same multimodal Pangea model (built-in vision)
        try:
            self.processor = AutoProcessor.from_pretrained(self.model_name, use_fast=True)
        except Exception:
            self.processor = None
        
        # Quantization config (prefer 4-bit QLoRA if enabled, else 8-bit, else None)
        quantization_config = None
        if self.model_config.get("load_in_4bit", False):
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        elif self.model_config.get("load_in_8bit", False):
            quantization_config = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
            )
        
        # Load base model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=quantization_config,
            torch_dtype=getattr(torch, self.model_config.get("torch_dtype", "bfloat16")),
            trust_remote_code=True,
            device_map="auto" if quantization_config else None,
        )

        # Enable gradient checkpointing if requested
        if self.model_config.get("gradient_checkpointing", False):
            try:
                self.model.gradient_checkpointing_enable()
            except Exception:
                pass
        
        # Do not load a separate vision encoder; Pangea is assumed multimodal
        self.vision_encoder = None
        
        # Apply LoRA to language model
        if self.lora_config:
            self.setup_lora()
        
        logger.info("Model setup complete")
    
    def setup_lora(self):
        """Apply LoRA configuration to the model."""
        logger.info("Applying LoRA configuration")
        
        lora_config = LoraConfig(
            r=self.lora_config.get("r", 16),
            lora_alpha=self.lora_config.get("lora_alpha", 32),
            target_modules=self.lora_config.get("target_modules", 
                ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]),
            lora_dropout=self.lora_config.get("lora_dropout", 0.1),
            bias=self.lora_config.get("bias", "none"),
            task_type=TaskType.CAUSAL_LM,
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def encode_images(self, images):
        """Placeholder retained for compatibility; not used when using Pangea's built-in vision."""
        return images
    
    def forward(self, input_ids, attention_mask=None, labels=None, images=None):
        """Forward pass through the model with optional image input."""
        # Route images directly to the multimodal model if supported
        try:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                images=images
            )
        except TypeError:
            # Fallback: models that don't accept images will ignore them
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
        return outputs
    
    def calculate_cer(self, predictions, targets):
        """Calculate Character Error Rate."""
        total_edits = 0
        total_chars = 0
        
        for pred, target in zip(predictions, targets):
            # Clean and normalize text
            pred = self.clean_text(pred)
            target = self.clean_text(target)
            
            # Calculate Levenshtein distance
            distance = self.levenshtein_distance(pred, target)
            total_edits += distance
            total_chars += len(target)
        
        return total_edits / max(total_chars, 1)
    
    def calculate_wer(self, predictions, targets):
        """Calculate Word Error Rate."""
        pred_texts = [self.clean_text(p) for p in predictions]
        target_texts = [self.clean_text(t) for t in targets]
        
        return wer(target_texts, pred_texts)
    
    def calculate_bleu(self, predictions, targets):
        """Calculate BLEU score."""
        smoothie = SmoothingFunction().method1
        total_bleu = 0
        
        for pred, target in zip(predictions, targets):
            pred_tokens = self.clean_text(pred).split()
            target_tokens = self.clean_text(target).split()
            
            if len(target_tokens) > 0:
                bleu = sentence_bleu([target_tokens], pred_tokens, smoothing_function=smoothie)
                total_bleu += bleu
        
        return total_bleu / len(predictions)
    
    def clean_text(self, text):
        """Clean and normalize text for evaluation."""
        # Remove special tokens and extra whitespace
        text = re.sub(r'<[^>]+>', '', text)  # Remove HTML-like tags
        text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
        text = text.strip()
        return text
    
    def levenshtein_distance(self, s1, s2):
        """Calculate Levenshtein distance between two strings."""
        if len(s1) < len(s2):
            return self.levenshtein_distance(s2, s1)
        
        if len(s2) == 0:
            return len(s1)
        
        previous_row = list(range(len(s2) + 1))
        for i, c1 in enumerate(s1):
            current_row = [i + 1]
            for j, c2 in enumerate(s2):
                insertions = previous_row[j + 1] + 1
                deletions = current_row[j] + 1
                substitutions = previous_row[j] + (c1 != c2)
                current_row.append(min(insertions, deletions, substitutions))
            previous_row = current_row
        
        return previous_row[-1]
    
    def generate_text_from_batch(self, batch, max_length=512, temperature=0.7):
        """Generate text from a batch of images."""
        predictions = []
        
        for i in range(batch['images'].shape[0]):
            # Extract single image
            image = batch['images'][i:i+1]
            
            # Create prompt
            prompt = "Below is a Sharada manuscript image. The text reads:\n\n"
            inputs = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=max_length,
                    temperature=temperature,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            predictions.append(generated_text)
        
        return predictions
    
    def training_step(self, batch, batch_idx):
        """Training step."""
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            images=batch.get('images', None)
        )
        
        loss = outputs.loss
        self.log('train_loss', loss, prog_bar=True, sync_dist=True)
        
        # Optionally log train CER/metrics at a reduced frequency to avoid overhead
        try:
            if self.metrics_config.get("log_cer", False) and (batch_idx % 200 == 0):
                # Generate predictions (best-effort, lightweight sampling)
                predictions = self.generate_text_from_batch(batch)
                  = []
                for text in batch.get('original_text', []):
                    targets.append(text if isinstance(text, str) else "")
                if predictions and targets:
                    cer = self.calculate_cer(predictions, targets)
                    self.log('train_cer', cer, prog_bar=False, sync_dist=True)
                    if self.metrics_config.get("log_wer", False):
                        wer_score = self.calculate_wer(predictions, targets)
                        self.log('train_wer', wer_score, sync_dist=True)
                    if self.metrics_config.get("log_bleu", False):
                        bleu_score = self.calculate_bleu(predictions, targets)
                        self.log('train_bleu', bleu_score, sync_dist=True)
        except Exception as e:
            logger.warning(f"Error calculating train metrics: {e}")
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step."""
        outputs = self(
            input_ids=batch['input_ids'],
            attention_mask=batch['attention_mask'],
            labels=batch['labels'],
            images=batch.get('images', None)
        )
        
        loss = outputs.loss
        self.log('val_loss', loss, prog_bar=True, sync_dist=True)
        
        # Calculate metrics if enabled
        if self.metrics_config.get("log_cer", False) and batch_idx < self.metrics_config.get("cer_samples", 100):
            try:
                # Generate predictions
                predictions = self.generate_text_from_batch(batch)
                
                # Extract ground truth text (remove prompt template)
                targets = []
                for text in batch.get('original_text', []):
                    # Extract text after the prompt
                    if isinstance(text, str):
                        # Remove prompt template to get just the text
                        text = text.replace("Below is a Sharada manuscript image. The text reads:\n\n", "").strip()
                        targets.append(text)
                    else:
                        targets.append("")
                
                # Calculate metrics
                if predictions and targets:
                    cer = self.calculate_cer(predictions, targets)
                    self.log('val_cer', cer, prog_bar=True, sync_dist=True)
                    
                    if self.metrics_config.get("log_wer", False):
                        wer_score = self.calculate_wer(predictions, targets)
                        self.log('val_wer', wer_score, sync_dist=True)
                    
                    if self.metrics_config.get("log_bleu", False):
                        bleu_score = self.calculate_bleu(predictions, targets)
                        self.log('val_bleu', bleu_score, sync_dist=True)
                
            except Exception as e:
                logger.warning(f"Error calculating metrics: {e}")
        
        return loss
    
    def configure_optimizers(self):
        """Configure optimizer and learning rate scheduler."""
        # Get trainable parameters
        trainable_params = [p for p in self.model.parameters() if p.requires_grad]
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=self.optimizer_config.get("lr", 2e-4),
            weight_decay=self.optimizer_config.get("weight_decay", 0.01),
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=0
        )
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
            },
        }
    
    def generate_text_from_image(self, image_path: str, max_length: int = 512, temperature: float = 0.7):
        """Generate text from an image (HTR task)."""
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.processor(images=image, return_tensors="pt")['pixel_values']
        image_tensor = image_tensor.to(self.device)
        
        # Create a simple prompt
        prompt = "Below is a Sharada manuscript image. The text reads:\n\n"
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            # This is a simplified generation - you'll need to adapt based on your vision-language integration
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
