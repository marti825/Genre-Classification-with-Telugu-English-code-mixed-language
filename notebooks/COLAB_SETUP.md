# Google Colab GPU Training - Quick Setup

## Why the .ipynb didn't work
The notebook file was saved as single-line JSON. Jupyter couldn't parse it properly.

## Easiest Way: Copy-Paste Script

Open https://colab.research.google.com, create a new notebook, then paste these 4 cells:

### Cell 1: Setup
```python
!pip install -q transformers sentencepiece torch pandas scikit-learn tqdm matplotlib seaborn
import torch, re, random, numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import XLMRobertaTokenizer, XLMRobertaModel, AdamW, get_linear_schedule_with_warmup
import os

SEED, DEVICE, MODEL_NAME = 42, torch.device('cuda' if torch.cuda.is_available() else 'cpu'), 'xlm-roberta-base'
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available(): torch.cuda.manual_seed_all(SEED)

MAX_LEN, BATCH_SIZE, LR, EPOCHS, WARMUP, PATIENCE = 128, 64, 3e-5, 5, 100, 2
LABELS = ['Education','Entertainment','Politics','Sports','Miscellaneous']
ID2LABEL = {i:l for i,l in enumerate(LABELS)}
LABEL2ID = {l:i for i,l in enumerate(LABELS)}
for l in LABELS: LABEL2ID[l.lower()] = LABEL2ID[l]

os.makedirs('outputs/models', exist_ok=True)
os.makedirs('outputs/results', exist_ok=True)
print('Device:', DEVICE)
```

### Cell 2: Upload Data
Use the Files panel on the left, or run:
```python
from google.colab import files
os.makedirs('data', exist_ok=True)
print("Upload train_dataset.csv"); [open(f'data/{n}','wb').write(c) for n,c in files.upload().items()]
print("Upload test_dataset.csv"); [open(f'data/{n}','wb').write(c) for n,c in files.upload().items()]
```

### Cell 3: Load & Train
Copy the full code from `notebooks/colab_train.py` starting at line 26 (CELL 4 section) through the end.

### Cell 4: Download Results
```python
from google.colab import files
files.download('outputs/results/predictions.csv')
files.download('outputs/results/confusion_matrix.png')
files.download('outputs/models/final_model.pt')
```

## Alternative: Use GitHub
Push this repo to GitHub, then in Colab:
```python
!git clone https://github.com/YOUR_USERNAME/major_project.git
%cd major_project
!python train.py
```
