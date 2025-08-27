#!/usr/bin/env python3
"""
Script to setup WandB for logging.
"""

import os
import subprocess
import sys

def setup_wandb():
    """Setup WandB authentication."""
    print("Setting up WandB for logging...")
    
    # Check if wandb is installed
    try:
        import wandb
        print("✓ WandB is installed")
    except ImportError:
        print("✗ WandB not found. Installing...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
        import wandb
    
    # Check if already logged in
    try:
        api = wandb.Api()
        print("✓ WandB is already authenticated")
        return True
    except Exception:
        print("WandB not authenticated. Please login:")
        print("1. Go to https://wandb.ai/authorize")
        print("2. Copy your API key")
        print("3. Run: wandb login")
        
        # Try to run wandb login
        try:
            subprocess.run(["wandb", "login"], check=True)
            print("✓ WandB login successful")
            return True
        except subprocess.CalledProcessError:
            print("✗ WandB login failed. Please run 'wandb login' manually")
            return False

if __name__ == "__main__":
    setup_wandb()
