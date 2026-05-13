#!/usr/bin/env python
# coding=utf-8

import logging
import os
import sys
import torch
import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    Trainer,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import is_main_process
import evaluate

from src.modeling import AlbertWithEarlyExits

logger = logging.getLogger(__name__)

MODEL_DIR = "./model_output/train_1"


def load_model_and_tokenizer(model_dir=MODEL_DIR):
    """
    Load AlbertWithEarlyExits model and tokenizer from model_output directory
    
    Args:
        model_dir (str): Path to the model output directory
    
    Returns:
        tuple: (model, tokenizer, config)
    """
    print("=" * 60)
    print("Loading AlbertWithEarlyExits model")
    print(f"Model directory: {model_dir}")
    print("=" * 60)
    
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")
    
    # Check for required files
    required_files = ["config.json", "tokenizer_config.json"]
    for f in required_files:
        if not os.path.exists(os.path.join(model_dir, f)):
            raise FileNotFoundError(f"Required file not found: {os.path.join(model_dir, f)}")
    
    # Load tokenizer
    print("\n[1/3] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    print(f"  ✓ Tokenizer loaded: vocab size = {tokenizer.vocab_size}")
    
    # Load model
    print("\n[2/3] Loading model...")
    model = AlbertWithEarlyExits.from_pretrained(
        model_dir
    )
    model.eval()
    print(f"  ✓ Model loaded")
    print(f"  ✓ Hidden size: {model.config.hidden_size}")
    print(f"  ✓ Num hidden layers: {model.config.num_hidden_layers}")
    print(f"  ✓ Early pooler hidden size: {model.config.early_pooler_hidden_size}")
    print(f"  ✓ Use early poolers: {model.config.use_early_poolers}")
    print(f"  ✓ Use meta predictors: {model.config.use_meta_predictors}")
    print(f"  ✓ Use history logits: {model.config.use_history_logits}")
    
    print("\n[3/3] Model summary...")
    print(model)
    
    return model, tokenizer, model.config


def run_inference(model, tokenizer, text, device='cpu'):
    """
    Run inference on a single text input
    
    Args:
        model: AlbertWithEarlyExits model
        tokenizer: tokenizer
        text (str): Input text
        device (str): Device to run inference on ('cpu' or 'cuda')
    
    Returns:
        dict: Inference results
    """
    model = model.to(device)
    model.eval()
    
    # Tokenize input
    inputs = tokenizer(
        text,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=128
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model(**inputs)
    
    # all_logits shape: [batch, num_classifiers+1, num_labels]
    # Use the final classifier output (last index)
    all_logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
    
    if all_logits.dim() == 3:
        # Early exit model: take the final classifier logits
        logits = all_logits[:, -1, :]
    else:
        logits = all_logits
    
    predictions = torch.argmax(logits, dim=-1)
    
    return {
        'logits': logits.cpu().numpy(),
        'all_logits': all_logits.cpu().numpy(),
        'predictions': predictions.cpu().numpy(),
    }


def evaluate_model(model, tokenizer, dataset, device='cpu', max_samples=None):
    """
    Evaluate model on a dataset
    
    Args:
        model: AlbertWithEarlyExits model
        tokenizer: tokenizer
        dataset: HuggingFace dataset
        device (str): Device to run inference on
        max_samples (int): Maximum number of samples to evaluate
    
    Returns:
        dict: Evaluation metrics
    """
    print(f"\nEvaluating on {len(dataset)} samples...")
    
    if max_samples is not None:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    metric = evaluate.load("accuracy")
    model = model.to(device)
    model.eval()
    
    all_predictions = []
    all_labels = []
    
    for i, sample in enumerate(dataset):
        inputs = tokenizer(
            sample['sentence'] if 'sentence' in sample else sample['sentence1'],
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=128
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # all_logits shape: [batch, num_classifiers+1, num_labels]
        # Use the final classifier output (last index)
        all_logits = outputs.logits if hasattr(outputs, 'logits') else outputs[0]
        
        if all_logits.dim() == 3:
            logits = all_logits[:, -1, :]
        else:
            logits = all_logits
        
        prediction = torch.argmax(logits, dim=-1).item()
        
        all_predictions.append(prediction)
        all_labels.append(sample['label'])
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{len(dataset)} samples")
    
    results = metric.compute(predictions=all_predictions, references=all_labels)
    print(f"\nEvaluation results: {results}")
    
    return results


if __name__ == "__main__":
    
    # Load model and tokenizer
    model, tokenizer, config = load_model_and_tokenizer(MODEL_DIR)

    print("\n" + "=" * 60)
    print("Model loaded successfully!")
    print("=" * 60)

    # Example: run inference on a sample text
    print("\n[Testing inference...]")
    sample_texts = [
        "This is a great movie!",
        "The weather is terrible today.",
        "I am feeling neutral about this.",
    ]

    for text in sample_texts:
        result = run_inference(model, tokenizer, text)
        print(f"\nText: '{text}'")
        print(f"  Prediction: {result['predictions']}")
        print(f"  Logits shape: {result['logits'].shape}")

    # Example: evaluate on SST-2
    # print("\n[Evaluating on SST-2...]")
    # dataset = load_dataset("glue", "sst2")
    # results = evaluate_model(model, tokenizer, dataset["validation"], max_samples=100)