"""
Standalone evaluation script.
Usage: python evaluate.py --model_path outputs/models/best_model.pt
"""
import sys; sys.path.insert(0, 'src')

import argparse
import torch
from transformers import XLMRobertaTokenizer
from config import DATA_PATHS, LABEL2ID
from utils import setup_logging
from preprocessing import load_and_preprocess
from dataset import create_dataloaders
from model import XLMRobertaClassifier
from evaluate import Evaluator


def evaluate_model(model_path, test_path):
    logger = setup_logging('evaluate')
    logger.info(f"Evaluating model: {model_path}")
    
    # Load test data
    _, test_df = load_and_preprocess(test_path)
    
    # Load tokenizer
    tokenizer = XLMRobertaTokenizer.from_pretrained('xlm-roberta-base')
    
    # Create test dataloader only
    from dataset import CodeMixedDataset, create_dataloaders
    from torch.utils.data import DataLoader
    test_dataset = CodeMixedDataset(test_df['comment'], test_df['actual_label'], tokenizer)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = XLMRobertaClassifier(num_labels=5)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    
    # Evaluate
    evaluator = Evaluator(model, test_loader)
    metrics = evaluator.run_full_evaluation(texts=test_df['comment'].tolist())
    
    logger.info("Evaluation completed!")
    return metrics


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', default='outputs/models/best_model.pt', help='Path to model checkpoint')
    parser.add_argument('--test_path', default='data/test_dataset.csv', help='Path to test data')
    args = parser.parse_args()
    
    evaluate_model(args.model_path, args.test_path)
