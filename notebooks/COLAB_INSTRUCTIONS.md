# Google Colab GPU Training Instructions

## Step 1: Open Google Colab
1. Go to https://colab.research.google.com
2. Create a **New Notebook** (File > New notebook)

## Step 2: Enable GPU
- Click **Runtime** in the menu
- Select **Change runtime type**
- Set **Hardware accelerator** to **GPU** (T4)
- Click **Save**

## Step 3: Copy-Paste Cells
Copy each section below into a separate Colab cell, then run them in order.

---

### Cell 1: GPU Check
```python
import torch
print('GPU:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('Device:', torch.cuda.get_device_name(0))
    print('Memory:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
```

### Cell 2: Install
```python
!pip install -q transformers sentencepiece torch pandas scikit-learn tqdm matplotlib seaborn
```

### Cell 3: Upload Dataset
```python
from google.colab import files
import os
os.makedirs('data', exist_ok=True)

print("Upload train_dataset.csv")
for n, c in files.upload().items():
    open(f'data/{n}', 'wb').write(c)

print("Upload test_dataset.csv")
for n, c in files.upload().items():
    open(f'data/{n}', 'wb').write(c)
```

### Cell 4: All Code
Copy everything from `notebooks/colab_train.py` starting from Cell 4 onwards into one big cell, OR copy each CELL section separately.

## Step 4: Run
Press **Runtime > Run all** or run cells one by one.

## Expected Time
- **T4 GPU**: ~2-3 minutes per epoch (5 epochs = ~15 min)
- **CPU**: Not recommended (too slow)

## Download Results
After training completes, the outputs/ folder contains:
- `outputs/models/best_model.pt` - Trained model weights
- `outputs/models/tokenizer/` - Tokenizer files
- `outputs/results/predictions.csv` - Test predictions
- `outputs/results/confusion_matrix.png` - Confusion matrix plot
- `outputs/results/training_curves.png` - Training curves

## Alternative: GitHub Upload
Instead of manual upload, you can also:
1. Push your project to GitHub
2. In Colab: `!git clone <your-repo-url>`
3. `!cd major_project && python train.py`
