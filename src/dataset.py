"""PyTorch Dataset for code-mixed text classification."""

import torch
from torch.utils.data import Dataset
from transformers import XLMRobertaTokenizer
from config import MODEL_CONFIG, ID2LABEL, LABEL2ID


class CodeMixedDataset(Dataset):
    """Dataset for Telugu-English code-mixed text."""
    
    def __init__(self, texts, labels, tokenizer=None, max_length=None):
        self.texts = texts.reset_index(drop=True) if hasattr(texts, 'reset_index') else texts
        self.labels = labels.reset_index(drop=True) if hasattr(labels, 'reset_index') else labels
        self.max_length = max_length or MODEL_CONFIG['max_length']
        
        # Load tokenizer if not provided
        if tokenizer is None:
            self.tokenizer = XLMRobertaTokenizer.from_pretrained(MODEL_CONFIG['model_name'])
        else:
            self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label_str = str(self.labels[idx]).lower().strip()
        label_id = LABEL2ID.get(label_str, 4)  # Default to Miscellaneous
        
        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label_id, dtype=torch.long),
        }


def create_dataloaders(train_df, test_df, tokenizer, batch_size, val_split=0.1):
    """Create train, validation, and test dataloaders."""
    from torch.utils.data import DataLoader, random_split
    from sklearn.model_selection import train_test_split
    
    # Split train into train/val
    train_split, val_split_df = train_test_split(
        train_df, test_size=val_split, random_state=42, stratify=train_df['actual_label']
    )
    
    # Create datasets
    train_dataset = CodeMixedDataset(
        train_split['comment'], train_split['actual_label'], tokenizer
    )
    val_dataset = CodeMixedDataset(
        val_split_df['comment'], val_split_df['actual_label'], tokenizer
    )
    test_dataset = CodeMixedDataset(
        test_df['comment'], test_df['actual_label'], tokenizer
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size * 2,
        shuffle=False,
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return train_loader, val_loader, test_loader
