"""Text preprocessing for code-mixed Telugu-English text."""

import re
import pandas as pd
from config import LABEL2ID, DATA_PATHS

# Expected column names for text and label
def detect_columns(df):
    """Auto-detect text and label columns from DataFrame."""
    cols = list(df.columns)
    
    # Common text column names
    text_candidates = ['comment', 'text', 'sentence', 'content', 'catalog_content', 
                      'tweet', 'post', 'message', 'review', 'input', '0']
    # Common label column names  
    label_candidates = ['actual_label', 'label', 'category', 'class', 'target',
                     'sentiment', 'type', 'tag', '1']
    
    text_col = None
    label_col = None
    
    for c in text_candidates:
        if c in cols:
            text_col = c
            break
    
    for c in label_candidates:
        if c in cols:
            label_col = c
            break
    
    # Fallback: if exactly 2 columns and no header detected, assume first=text, second=label
    if text_col is None or label_col is None:
        if len(cols) == 2:
            text_col = cols[0]
            label_col = cols[1]
    
    return text_col, label_col


class TextPreprocessor:
    """Preprocessor for code-mixed social media text."""
    
    def __init__(self):
        # Regex patterns
        self.url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}')
        self.mention_pattern = re.compile(r'@\w+')
        self.hashtag_pattern = re.compile(r'#\w+')
        self.email_pattern = re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b')
        # Keep emojis as they can carry sentiment and context
        self.extra_spaces = re.compile(r'\s+')
        self.repeated_chars = re.compile(r'(.)\1{3,}')
        
    def clean_text(self, text):
        """Clean a single text sample."""
        text = str(text)
        
        # Remove URLs
        text = self.url_pattern.sub(' ', text)
        
        # Remove email addresses
        text = self.email_pattern.sub(' ', text)
        
        # Replace mentions with special token or remove
        text = self.mention_pattern.sub(' ', text)
        
        # Replace hashtags with just the word (remove # but keep context)
        text = self.hashtag_pattern.sub(lambda m: m.group(0)[1:], text)
        
        # Normalize repeated characters (e.g., "gooood" -> "good")
        text = self.repeated_chars.sub(r'\1\1\1', text)
        
        # Remove excessive whitespace
        text = self.extra_spaces.sub(' ', text)
        
        # Strip leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def process_dataframe(self, df, text_col=None, label_col=None):
        """Process entire dataframe. Auto-detect columns if not provided."""
        df = df.copy()
        
        # Auto-detect columns
        if text_col is None or label_col is None:
            detected_text, detected_label = detect_columns(df)
            text_col = text_col or detected_text
            label_col = label_col or detected_label
        
        if text_col is None or label_col is None:
            print(f"Available columns: {list(df.columns)}")
            raise ValueError(f"Could not detect text/label columns. Please specify manually.")
        
        print(f"Using text_col='{text_col}', label_col='{label_col}'")
        
        # Rename to standard names for downstream consistency
        df = df.rename(columns={text_col: 'comment', label_col: 'actual_label'})
        text_col, label_col = 'comment', 'actual_label'
        
        # Clean text
        df[text_col] = df[text_col].apply(self.clean_text)
        
        # Remove empty texts
        df = df[df[text_col].str.len() > 0]
        
        # Filter valid labels (case-insensitive)
        df[label_col] = df[label_col].str.lower().str.strip()
        valid_labels = set(k.lower() for k in LABEL2ID.keys())
        df = df[df[label_col].isin(valid_labels)]
        
        # Drop duplicates
        df = df.drop_duplicates(subset=[text_col])
        
        # Reset index
        df = df.reset_index(drop=True)
        
        return df


def load_and_preprocess(train_path, test_path=None, header='infer'):
    """Load and preprocess train and test data.
    
    Args:
        train_path: Path to training CSV
        test_path: Path to test CSV (optional)
        header: 'infer' to auto-detect, None for no header
    """
    preprocessor = TextPreprocessor()
    
    # Try loading with header, fallback to no header
    try:
        train_df = pd.read_csv(train_path, header=header)
        if header == 'infer' and len(train_df.columns) == 2:
            # Check if first row looks like data (not header)
            first_val = str(train_df.iloc[0, 1]).lower()
            if first_val in LABEL2ID.keys():
                pass  # Has proper header
            else:
                # Probably no header - reload without header
                train_df = pd.read_csv(train_path, header=None, 
                                       names=['comment', 'actual_label'])
    except Exception as e:
        print(f"Error loading train: {e}")
        train_df = pd.read_csv(train_path, header=None, 
                               names=['comment', 'actual_label'])
    
    train_df = preprocessor.process_dataframe(train_df)
    
    print(f"Train samples: {len(train_df)}")
    if len(train_df) > 0:
        print("Train label distribution:")
        print(train_df['actual_label'].value_counts())
    
    # Load test if provided
    test_df = None
    if test_path:
        try:
            test_df = pd.read_csv(test_path, header=header)
            if header == 'infer' and len(test_df.columns) == 2:
                first_val = str(test_df.iloc[0, 1]).lower()
                if first_val not in LABEL2ID.keys():
                    test_df = pd.read_csv(test_path, header=None,
                                          names=['comment', 'actual_label'])
        except:
            test_df = pd.read_csv(test_path, header=None,
                               names=['comment', 'actual_label'])
        
        test_df = preprocessor.process_dataframe(test_df)
        print(f"\nTest samples: {len(test_df)}")
        if len(test_df) > 0:
            print("Test label distribution:")
            print(test_df['actual_label'].value_counts())
    
    return train_df, test_df


def oversample_minority(train_df, text_col='comment', label_col='actual_label', 
                      max_multiplier=3):
    """Oversample minority classes up to max_multiplier times."""
    label_counts = train_df[label_col].value_counts()
    max_count = label_counts.max()
    
    dfs = []
    for label, count in label_counts.items():
        label_df = train_df[train_df[label_col] == label]
        
        if count < max_count:
            multiplier = min(max_count // count, max_multiplier)
            if multiplier > 1:
                label_df = pd.concat([label_df] * multiplier, ignore_index=True)
        
        dfs.append(label_df)
    
    balanced_df = pd.concat(dfs, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)
    
    print(f"\nBalanced train samples: {len(balanced_df)}")
    print("Balanced label distribution:")
    print(balanced_df[label_col].value_counts())
    
    return balanced_df
