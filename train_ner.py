"""
NER Training Script for Turkish Documents

This script fine-tunes a Turkish BERT model for Named Entity Recognition (NER).
It expects a dataset in CoNLL-2003 format or a custom dataset with token-level labels.

Usage:
    python train_ner.py --data_dir ./data/ner --output_dir ./models/ner_model
"""

import os
import argparse
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from transformers import EarlyStoppingCallback
import torch
try:
    from seqeval.metrics import accuracy_score, precision_recall_fscore_support
    SEQEVAL_AVAILABLE = True
except ImportError:
    SEQEVAL_AVAILABLE = False
    print("WARNING: seqeval not installed. Install with: pip install seqeval")
import numpy as np


# Model configuration
MODEL_NAME = 'dbmdz/bert-base-turkish-cased'
MAX_LENGTH = 512


def load_conll_dataset(data_dir: str):
    """
    Load NER dataset in CoNLL-2003 format.
    
    Expected format:
    - train.txt, dev.txt, test.txt
    - Each line: word\tlabel
    - Empty line separates sentences
    """
    def parse_conll_file(file_path):
        sentences = []
        current_sentence = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                else:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        word, label = parts[0], parts[1]
                        current_sentence.append((word, label))
            
            if current_sentence:
                sentences.append(current_sentence)
        
        return sentences
    
    data_dir = Path(data_dir)
    train_sentences = parse_conll_file(data_dir / 'train.txt') if (data_dir / 'train.txt').exists() else []
    dev_sentences = parse_conll_file(data_dir / 'dev.txt') if (data_dir / 'dev.txt').exists() else []
    test_sentences = parse_conll_file(data_dir / 'test.txt') if (data_dir / 'test.txt').exists() else []
    
    def sentences_to_dataset(sentences):
        tokens = []
        labels = []
        for sent in sentences:
            tokens.append([w[0] for w in sent])
            labels.append([w[1] for w in sent])
        return Dataset.from_dict({'tokens': tokens, 'ner_tags': labels})
    
    datasets = {}
    if train_sentences:
        datasets['train'] = sentences_to_dataset(train_sentences)
    if dev_sentences:
        datasets['dev'] = sentences_to_dataset(dev_sentences)
    if test_sentences:
        datasets['test'] = sentences_to_dataset(test_sentences)
    
    return DatasetDict(datasets) if datasets else None


def get_label_list(datasets):
    """Extract unique labels from the dataset."""
    all_labels = set()
    for split in datasets.values():
        for example in split:
            all_labels.update(example['ner_tags'])
    return sorted(list(all_labels))


def tokenize_and_align_labels(examples, tokenizer, label_to_id):
    """Tokenize and align labels with subword tokens."""
    tokenized_inputs = tokenizer(
        examples['tokens'],
        truncation=True,
        padding=True,
        max_length=MAX_LENGTH,
        is_split_into_words=True,
    )
    
    labels = []
    for i, label_seq in enumerate(examples['ner_tags']):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)  # Special tokens: -100 is ignored in loss
            elif word_idx != previous_word_idx:
                label = label_seq[word_idx]
                label_ids.append(label_to_id.get(label, label_to_id.get('O', 0)))
            else:
                # Subword token: use -100 or the same label (BIO scheme: use -100)
                label_ids.append(-100)
            previous_word_idx = word_idx
        
        labels.append(label_ids)
    
    tokenized_inputs['labels'] = labels
    return tokenized_inputs


def compute_metrics_factory(label_list):
    """Factory function to create compute_metrics with access to label_list."""
    def compute_metrics(eval_pred):
        """Compute NER metrics (precision, recall, F1)."""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Remove ignored index (special tokens)
        true_predictions = [
            [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [label_list[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        if SEQEVAL_AVAILABLE:
            results = precision_recall_fscore_support(true_labels, true_predictions, average='weighted')
            return {
                'precision': results[0],
                'recall': results[1],
                'f1': results[2],
                'accuracy': accuracy_score(true_labels, true_predictions)
            }
        else:
            # Fallback: simple accuracy if seqeval not available
            correct = sum(p == l for pred_seq, label_seq in zip(true_predictions, true_labels) 
                         for p, l in zip(pred_seq, label_seq))
            total = sum(len(seq) for seq in true_labels)
            return {'accuracy': correct / total if total > 0 else 0.0}
    return compute_metrics


def main():
    parser = argparse.ArgumentParser(description='Train Turkish NER model')
    parser.add_argument('--data_dir', type=str, default='./data/ner',
                        help='Directory containing train.txt, dev.txt, test.txt')
    parser.add_argument('--output_dir', type=str, default='./models/ner_model',
                        help='Output directory for the trained model')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=2e-5,
                        help='Learning rate')
    parser.add_argument('--use_hf_dataset', type=str, default=None,
                        help='Use Hugging Face dataset name instead of local files')
    
    args = parser.parse_args()
    
    # Load tokenizer
    print(f"Loading tokenizer from {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    # Load dataset
    if args.use_hf_dataset:
        print(f"Loading dataset from Hugging Face: {args.use_hf_dataset}")
        datasets = load_dataset(args.use_hf_dataset)
    else:
        print(f"Loading dataset from {args.data_dir}...")
        datasets = load_conll_dataset(args.data_dir)
        if datasets is None:
            print("ERROR: No dataset found. Please provide data_dir with train.txt, dev.txt, test.txt")
            print("Or use --use_hf_dataset to load from Hugging Face (e.g., 'tner/ontonotes5')")
            return
    
    # Get label list and create mappings
    label_list = get_label_list(datasets)
    label_to_id = {label: i for i, label in enumerate(label_list)}
    id_to_label = {i: label for i, label in enumerate(label_list)}
    num_labels = len(label_list)
    
    print(f"Found {num_labels} unique labels: {label_list}")
    
    # Load model
    print(f"Loading model from {MODEL_NAME}...")
    model = AutoModelForTokenClassification.from_pretrained(
        MODEL_NAME,
        num_labels=num_labels,
        id2label=id_to_label,
        label2id=label_to_id,
    )
    
    # Tokenize dataset
    print("Tokenizing dataset...")
    tokenized_datasets = datasets.map(
        lambda x: tokenize_and_align_labels(x, tokenizer, label_to_id),
        batched=True,
        remove_columns=datasets['train'].column_names if 'train' in datasets else datasets[list(datasets.keys())[0]].column_names,
    )
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        logging_dir=f'{args.output_dir}/logs',
        logging_steps=100,
        eval_strategy='epoch' if 'dev' in datasets or 'validation' in datasets else 'no',
        save_strategy='epoch',
        save_total_limit=3,
        load_best_model_at_end=True if 'dev' in datasets or 'validation' in datasets else False,
        metric_for_best_model='f1' if SEQEVAL_AVAILABLE else 'accuracy',
        greater_is_better=True,
        push_to_hub=False,
    )
    
    # Data collator
    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    
    # Create compute_metrics function with label_list
    compute_metrics = compute_metrics_factory(label_list)
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'] if 'train' in tokenized_datasets else None,
        eval_dataset=tokenized_datasets.get('dev') or tokenized_datasets.get('validation'),
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=3)] if 'dev' in datasets or 'validation' in datasets else None,
    )
    
    # Train
    print("Starting training...")
    trainer.train()
    
    # Save model
    print(f"Saving model to {args.output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Evaluate on test set if available
    if 'test' in tokenized_datasets:
        print("Evaluating on test set...")
        results = trainer.evaluate(tokenized_datasets['test'])
        print(f"Test results: {results}")
    
    print("Training completed!")


if __name__ == '__main__':
    main()
