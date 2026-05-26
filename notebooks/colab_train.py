"""
Google Colab GPU Training Script
Copy-paste all cells below into a Colab notebook.
"""
# ==== CELL 1 ====
# Check GPU
import torch
print('GPU:', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')

# ==== CELL 2 ====
# Install
!pip install -q transformers sentencepiece torch pandas scikit-learn tqdm matplotlib seaborn

# ==== CELL 3 ====
# Upload dataset files using the left Files panel, or run:
from google.colab import files
import os, shutil
os.makedirs('data', exist_ok=True)
print("Upload train_dataset.csv")
for n, c in files.upload().items(): open(f'data/{n}','wb').write(c)
print("Upload test_dataset.csv")
for n, c in files.upload().items(): open(f'data/{n}','wb').write(c)

# ==== CELL 4 ====
# Imports & Config
import re, json, random, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel, AdamW, get_linear_schedule_with_warmup

SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'xlm-roberta-base'
MAX_LEN = 128
BATCH_SIZE = 64
LR = 3e-5
EPOCHS = 5
WARMUP = 100
PATIENCE = 2

LABELS = ['Education', 'Entertainment', 'Politics', 'Sports', 'Miscellaneous']
ID2LABEL = {i: l for i, l in enumerate(LABELS)}
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
for l in LABELS: LABEL2ID[l.lower()] = LABEL2ID[l]

os.makedirs('outputs/models', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)

# ==== CELL 5 ====
# Preprocessing
class Preproc:
    def __init__(self):
        self.url = re.compile(r'https?://\S+|www\.\S+')
        self.mention = re.compile(r'@\w+')
        self.email = re.compile(r'\S+@\S+\.\S+')
        self.space = re.compile(r'\s+')
        self.repeat = re.compile(r'(.)\1{3,}')
    def clean(self, t):
        t = str(t)
        t = self.url.sub(' ', t); t = self.email.sub(' ', t); t = self.mention.sub(' ', t)
        t = self.repeat.sub(r'\1\1\1', t); t = self.space.sub(' ', t); return t.strip()
    def process(self, path):
        df = pd.read_csv(path, header=None, names=['text','label'])
        df['text'] = df['text'].apply(self.clean)
        df['label'] = df['label'].str.lower().str.strip()
        valid = set(k.lower() for k in LABEL2ID.keys())
        df = df[df['label'].isin(valid)].drop_duplicates('text').reset_index(drop=True)
        print(f"{path}: {len(df)} samples")
        print(df['label'].value_counts())
        return df

train_df = Preproc().process('data/train_dataset.csv')
test_df = Preproc().process('data/test_dataset.csv')

# ==== CELL 6 ====
# Dataset & Model
class CMData(Dataset):
    def __init__(self, texts, labels, tok, max_len=128):
        self.texts = texts.tolist() if hasattr(texts, 'tolist') else texts
        self.labels = labels.tolist() if hasattr(labels, 'tolist') else labels
        self.tok = tok; self.max_len = max_len
    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        lab = LABEL2ID.get(str(self.labels[idx]).lower().strip(), 4)
        e = self.tok(text, max_length=self.max_len, padding='max_length', truncation=True, return_tensors='pt')
        return {'input_ids': e['input_ids'].flatten(), 'attention_mask': e['attention_mask'].flatten(), 'labels': torch.tensor(lab, dtype=torch.long)}

class Classifier(nn.Module):
    def __init__(self, n_labels=5):
        super().__init__()
        self.roberta = XLMRobertaModel.from_pretrained(MODEL_NAME)
        self.drop = nn.Dropout(0.1)
        self.clf = nn.Linear(self.roberta.config.hidden_size, n_labels)
    def forward(self, ids, mask, labels=None):
        out = self.roberta(ids, attention_mask=mask, return_dict=True)
        cls = out.last_hidden_state[:, 0, :]
        cls = self.drop(cls)
        logits = self.clf(cls)
        loss = None
        if labels is not None:
            loss = nn.CrossEntropyLoss(label_smoothing=0.1)(logits, labels)
        return {'loss': loss, 'logits': logits}

# ==== CELL 7 ====
# Build loaders & model
tok = XLMRobertaTokenizer.from_pretrained(MODEL_NAME)
train_s, val_s = train_test_split(train_df, test_size=0.1, random_state=42, stratify=train_df['label'])

train_ds = CMData(train_s['text'], train_s['label'], tok)
val_ds = CMData(val_s['text'], val_s['label'], tok)
test_ds = CMData(test_df['text'], test_df['label'], tok)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE*2)
test_dl = DataLoader(test_ds, batch_size=BATCH_SIZE*2)

model = Classifier().to(DEVICE)
opt = AdamW(model.parameters(), lr=LR)
total = len(train_dl) * EPOCHS
sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=WARMUP, num_training_steps=total)

print(f"Train batches: {len(train_dl)}, Val: {len(val_dl)}, Test: {len(test_dl)}")

# ==== CELL 8 ====
# Training loop
history = {'train_loss': [], 'val_loss': [], 'val_acc': [], 'val_f1': []}
best_f1, patience_cnt = 0, 0
best_state = None

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in tqdm(train_dl, desc=f'Epoch {epoch+1}/{EPOCHS}'):
        ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        labels = batch['labels'].to(DEVICE)
        opt.zero_grad()
        out = model(ids, mask, labels)
        loss = out['loss']
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(train_dl)
    history['train_loss'].append(avg_loss)

    # Validation
    model.eval()
    vloss, preds, trues = 0, [], []
    with torch.no_grad():
        for batch in val_dl:
            ids = batch['input_ids'].to(DEVICE)
            mask = batch['attention_mask'].to(DEVICE)
            labels = batch['labels'].to(DEVICE)
            out = model(ids, mask, labels)
            vloss += out['loss'].item()
            pred = torch.argmax(out['logits'], dim=1)
            preds.extend(pred.cpu().numpy())
            trues.extend(labels.cpu().numpy())
    vloss /= len(val_dl)
    acc = accuracy_score(trues, preds)
    f1 = f1_score(trues, preds, average='weighted', zero_division=0)
    history['val_loss'].append(vloss)
    history['val_acc'].append(acc)
    history['val_f1'].append(f1)

    print(f"Epoch {epoch+1}: train_loss={avg_loss:.4f} val_loss={vloss:.4f} val_acc={acc:.4f} val_f1={f1:.4f}")

    # Early stopping
    if f1 > best_f1:
        best_f1 = f1
        patience_cnt = 0
        best_state = model.state_dict()
        torch.save(best_state, 'outputs/models/best_model.pt')
        print(f"  -> New best model saved (F1={best_f1:.4f})")
    else:
        patience_cnt += 1
        if patience_cnt >= PATIENCE:
            print(f"Early stopping at epoch {epoch+1}")
            break

if best_state:
    model.load_state_dict(best_state)

# ==== CELL 9 ====
# Test evaluation
model.eval()
all_preds, all_trues, all_probs = [], [], []
with torch.no_grad():
    for batch in test_dl:
        ids = batch['input_ids'].to(DEVICE)
        mask = batch['attention_mask'].to(DEVICE)
        out = model(ids, mask)
        probs = torch.softmax(out['logits'], dim=1)
        pred = torch.argmax(out['logits'], dim=1)
        all_preds.extend(pred.cpu().numpy())
        all_trues.extend(batch['labels'].numpy())
        all_probs.extend(probs.cpu().numpy())

# Metrics
acc = accuracy_score(all_trues, all_preds)
f1w = f1_score(all_trues, all_preds, average='weighted', zero_division=0)
f1m = f1_score(all_trues, all_preds, average='macro', zero_division=0)
print(f"\nTest Accuracy: {acc:.4f}")
print(f"Test Weighted F1: {f1w:.4f}")
print(f"Test Macro F1: {f1m:.4f}")
print("\nClassification Report:")
print(classification_report(all_trues, all_preds, target_names=LABELS, zero_division=0))

# Confusion matrix
cm = confusion_matrix(all_trues, all_preds)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=LABELS, yticklabels=LABELS)
plt.title('Confusion Matrix'); plt.xlabel('Predicted'); plt.ylabel('True')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('outputs/results/confusion_matrix.png', dpi=300)
plt.show()

# Training curves
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
axes[0].plot(history['train_loss'], label='Train'); axes[0].plot(history['val_loss'], label='Val')
axes[0].set_title('Loss'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(history['val_acc'], marker='o'); axes[1].set_title('Val Accuracy'); axes[1].grid(True, alpha=0.3)
axes[2].plot(history['val_f1'], marker='o', color='green'); axes[2].set_title('Val F1'); axes[2].grid(True, alpha=0.3)
plt.tight_layout(); plt.savefig('outputs/results/training_curves.png', dpi=300); plt.show()

# Save predictions
results = []
for text, true, pred, probs in zip(test_df['text'].tolist(), all_trues, all_preds, all_probs):
    results.append({'text': text, 'true': ID2LABEL[true], 'pred': ID2LABEL[pred], 'correct': true==pred, 'conf': float(probs[pred])})
pd.DataFrame(results).to_csv('outputs/results/predictions.csv', index=False)

# Save model & tokenizer
tok.save_pretrained('outputs/models/')
torch.save(model.state_dict(), 'outputs/models/final_model.pt')
print("\nModel and results saved to outputs/")

# ==== CELL 10 ====
# Download results
from google.colab import files
files.download('outputs/results/predictions.csv')
files.download('outputs/results/confusion_matrix.png')
files.download('outputs/results/training_curves.png')
