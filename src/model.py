"""XLM-RoBERTa model with custom classification head."""

import torch
import torch.nn as nn
from transformers import XLMRobertaModel, XLMRobertaTokenizer
from config import MODEL_CONFIG, DEVICE


class XLMRobertaClassifier(nn.Module):
    """XLM-RoBERTa for code-mixed text classification."""
    
    def __init__(self, num_labels, model_name=None, dropout_rate=None):
        super(XLMRobertaClassifier, self).__init__()
        
        self.model_name = model_name or MODEL_CONFIG['model_name']
        self.num_labels = num_labels
        self.dropout_rate = dropout_rate or MODEL_CONFIG['dropout_rate']
        
        # Load pre-trained XLM-RoBERTa
        self.roberta = XLMRobertaModel.from_pretrained(self.model_name)
        self.hidden_size = self.roberta.config.hidden_size
        
        # Classification head
        self.dropout = nn.Dropout(self.dropout_rate)
        self.classifier = nn.Linear(self.hidden_size, num_labels)
        
        # Initialize classifier weights
        nn.init.xavier_uniform_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)
    
    def forward(self, input_ids, attention_mask, labels=None):
        """Forward pass."""
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Extract [CLS] token (first token) representation
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch, hidden_size]
        
        # Apply dropout
        cls_output = self.dropout(cls_output)
        
        # Classification
        logits = self.classifier(cls_output)
        
        # Compute loss if labels provided
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(label_smoothing=0.1)
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': outputs.last_hidden_state
        }
    
    def freeze_base(self):
        """Freeze base model parameters (for feature extraction)."""
        for param in self.roberta.parameters():
            param.requires_grad = False
    
    def unfreeze_base(self):
        """Unfreeze base model parameters for fine-tuning."""
        for param in self.roberta.parameters():
            param.requires_grad = True
    
    def get_param_groups(self, lr_head=2e-5, lr_base=1e-5):
        """Get parameter groups with different learning rates."""
        base_params = []
        head_params = []
        
        for name, param in self.named_parameters():
            if 'classifier' in name or 'dropout' in name:
                head_params.append(param)
            else:
                base_params.append(param)
        
        return [
            {'params': base_params, 'lr': lr_base},
            {'params': head_params, 'lr': lr_head}
        ]


def build_model(num_labels, model_name=None, freeze_base=False):
    """Build and return model."""
    model = XLMRobertaClassifier(num_labels, model_name)
    
    if freeze_base:
        model.freeze_base()
    
    return model.to(DEVICE)
