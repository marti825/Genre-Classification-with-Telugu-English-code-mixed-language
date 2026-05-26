"""Training module for XLM-RoBERTa code-mixed text classification."""

import os
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from tqdm import tqdm

from config import TRAIN_CONFIG, MODEL_CONFIG, DEVICE, DATA_PATHS
from utils import setup_logging, set_seed, compute_class_weights, save_checkpoint, save_json
from model import build_model
from dataset import create_dataloaders


class EarlyStopping:
    """Early stopping to prevent overfitting."""
    
    def __init__(self, patience=3, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        self.best_checkpoint = None
    
    def __call__(self, val_loss, model, optimizer, scheduler, epoch):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model, optimizer, scheduler, epoch)
        elif val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model, optimizer, scheduler, epoch)
    
    def save_checkpoint(self, model, optimizer, scheduler, epoch):
        self.best_checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        }


class Trainer:
    """Trainer class for model training."""
    
    def __init__(self, model, train_loader, val_loader, config=None):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or TRAIN_CONFIG
        self.device = DEVICE
        self.logger = setup_logging('train')
        
        self.global_step = 0
        self.history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
        
        self._setup_optimizer()
        self._setup_scheduler()
        self.early_stopping = EarlyStopping(
            patience=self.config['early_stopping_patience']
        )
    
    def _setup_optimizer(self):
        """Setup AdamW optimizer."""
        param_groups = self.model.get_param_groups(
            lr_head=self.config['learning_rate'],
            lr_base=self.config['learning_rate'] / 2
        )
        
        self.optimizer = AdamW(
            param_groups,
            weight_decay=self.config['weight_decay']
        )
    
    def _setup_scheduler(self):
        """Setup linear warmup scheduler."""
        total_steps = len(self.train_loader) * self.config['num_epochs']
        warmup_steps = self.config['warmup_steps']
        
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps
        )
    
    def train_epoch(self, epoch):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {epoch+1}/{self.config["num_epochs"]}')
        
        for batch_idx, batch in enumerate(progress_bar):
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(input_ids, attention_mask, labels)
            loss = outputs['loss']
            
            # Gradient accumulation
            loss = loss / self.config['gradient_accumulation_steps']
            loss.backward()
            
            if (batch_idx + 1) % self.config['gradient_accumulation_steps'] == 0:
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['max_grad_norm']
                )
                
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.global_step += 1
            
            total_loss += loss.item() * self.config['gradient_accumulation_steps']
            num_batches += 1
            
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
        
        avg_loss = total_loss / num_batches
        self.history['train_loss'].append(avg_loss)
        
        return avg_loss
    
    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Evaluating', leave=False):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(input_ids, attention_mask, labels)
                loss = outputs['loss']
                logits = outputs['logits']
                
                total_loss += loss.item()
                
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds, average='weighted')
        
        self.history['val_loss'].append(avg_loss)
        self.history['val_acc'].append(accuracy)
        self.history['val_f1'].append(f1)
        
        return avg_loss, accuracy, f1
    
    def train(self):
        """Full training loop."""
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Total epochs: {self.config['num_epochs']}")
        self.logger.info(f"Train batches: {len(self.train_loader)}")
        self.logger.info(f"Val batches: {len(self.val_loader)}")
        
        best_f1 = 0
        
        for epoch in range(self.config['num_epochs']):
            start_time = time.time()
            
            # Train
            train_loss = self.train_epoch(epoch)
            
            # Evaluate
            val_loss, val_acc, val_f1 = self.evaluate()
            
            epoch_time = time.time() - start_time
            
            self.logger.info(
                f"Epoch {epoch+1}/{self.config['num_epochs']} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | "
                f"Val Acc: {val_acc:.4f} | "
                f"Val F1: {val_f1:.4f} | "
                f"Time: {epoch_time:.2f}s"
            )
            
            # Save best model
            if val_f1 > best_f1:
                best_f1 = val_f1
                model_path = os.path.join(DATA_PATHS['model_dir'], 'best_model.pt')
                save_checkpoint(self.model, self.optimizer, self.scheduler, epoch, model_path)
                self.logger.info(f"New best model saved (F1: {best_f1:.4f})")
            
            # Early stopping
            self.early_stopping(val_loss, self.model, self.optimizer, self.scheduler, epoch)
            if self.early_stopping.early_stop:
                self.logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Load best checkpoint
        if self.early_stopping.best_checkpoint:
            self.model.load_state_dict(self.early_stopping.best_checkpoint['model_state_dict'])
        
        # Save training history
        history_path = os.path.join(DATA_PATHS['result_dir'], 'training_history.json')
        save_json(self.history, history_path)
        
        self.logger.info("Training completed")
        
        return self.model, self.history
