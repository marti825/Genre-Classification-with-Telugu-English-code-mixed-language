"""Configuration for XLM-RoBERTa Code-Mixed Text Classification."""

# Model Configuration
MODEL_CONFIG = {
    'model_name': 'xlm-roberta-base',
    'max_length': 128,
    'num_labels': 5,
    'dropout_rate': 0.1,
}

# Training Configuration - Optimized for speed
TRAIN_CONFIG = {
    'batch_size': 64,          # Larger batch = fewer iterations/epoch
    'learning_rate': 3e-5,   # Slightly higher LR for fewer epochs
    'weight_decay': 0.01,
    'warmup_steps': 100,     # Reduced for shorter training
    'num_epochs': 5,         # Fewer epochs (early stopping still active)
    'gradient_accumulation_steps': 1,  # Disabled - larger batch covers it
    'max_grad_norm': 1.0,
    'early_stopping_patience': 2,  # Stop faster if no improvement
    'label_smoothing': 0.1,
    'seed': 42,
}

# Evaluation Configuration
EVAL_CONFIG = {
    'batch_size': 64,
}

# Data Paths
DATA_PATHS = {
    'train': 'data/train_dataset.csv',
    'test': 'data/test_dataset.csv',
    'full': 'data/labeled_dataset_33906.csv',
    'output_dir': 'outputs/',
    'model_dir': 'outputs/models/',
    'log_dir': 'outputs/logs/',
    'result_dir': 'outputs/results/',
}

# Label Mapping
LABELS = ['Education', 'Entertainment', 'Politics', 'Sports', 'Miscellaneous']

# ID to Label Mapping
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
# Add lowercase variants for case-insensitive matching
for label in LABELS:
    LABEL2ID[label.lower()] = LABEL2ID[label]

# Device Configuration
import torch
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
