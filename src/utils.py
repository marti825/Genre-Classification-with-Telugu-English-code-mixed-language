"""Utility functions for the project."""

import os
import json
import logging
import random
import numpy as np
import torch
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight

from config import TRAIN_CONFIG, DATA_PATHS


def setup_logging(name='train'):
    """Setup logging configuration."""
    log_dir = DATA_PATHS['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f'{name}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(name)


def set_seed(seed=TRAIN_CONFIG['seed']):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def compute_class_weights(labels):
    """Compute class weights to handle imbalance."""
    classes = np.unique(labels)
    weights = compute_class_weight(
        class_weight='balanced',
        classes=classes,
        y=labels
    )
    return torch.tensor(weights, dtype=torch.float)


def save_json(data, filepath):
    """Save data to JSON file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def load_json(filepath):
    """Load data from JSON file."""
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_checkpoint(model, optimizer, scheduler, epoch, filepath):
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
    }, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    return checkpoint.get('epoch', 0)
