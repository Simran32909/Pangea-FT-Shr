#!/bin/bash

# Complete training pipeline for Sharada fine-tuning
set -e

echo "=== Sharada Fine-tuning Pipeline ==="
echo "Starting at: $(date)"

# Change to project directory
cd /scratch/tathagata.ghosh/sharada_finetune

echo "=== Step 1: Installing dependencies ==="
pip install -r requirements.txt

echo "=== Step 2: Preparing dataset ==="
python scripts/prepare_data.py

echo "=== Step 3: Starting training ==="
python scripts/train.py

echo "=== Step 4: Running inference test ==="
python scripts/inference.py

echo "=== Pipeline completed at: $(date) ==="
echo "Check logs in: /scratch/tathagata.ghosh/sharada_finetune/logs"
echo "Check checkpoints in: /scratch/tathagata.ghosh/sharada_finetune/checkpoints"
