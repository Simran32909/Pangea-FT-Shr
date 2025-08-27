# Pangea-7B Fine-Tuned for HTR

Fine-tune Pangea-7B for Handwritten Text Recognition (HTR) on Sharada manuscript images.


## Dataset

- **Input**: 1800x68 pixel manuscript line images + JSON with ground truth text
- **Location**: `/scratch/tathagata.ghosh/datasets/1MSharada`
- **Format**: JSON files with `original_text` and `image_path` fields

## Quick Start

```bash
# 1. Prepare dataset
python scripts/prepare_data.py

# 2. Train model
python scripts/train.py

# 3. Test model
python scripts/inference.py
```

## Configuration

Key settings in `configs/config.yaml`:

```yaml
data:
  max_samples: 10000      # Number of samples for training
  image_size: [1800, 68]  # Original image dimensions

trainer:
  max_epochs: 10
  batch_size: 1           # For large images

logging:
  use_wandb: true         # Enable WandB logging
```

## Metrics

- **CER**: Character Error Rate
- **WER**: Word Error Rate  
- **BLEU**: Text similarity score
- **Loss**: Training/validation loss

## Output

- **Checkpoints**: `checkpoints/`
- **Logs**: `logs/`
- **WandB**: Real-time metrics dashboard

## Customize

```bash
# Change number of samples
python scripts/prepare_data.py data.max_samples=5000

# Change learning rate
python scripts/train.py optimizer.lr=1e-4
```
