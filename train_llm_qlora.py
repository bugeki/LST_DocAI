"""
QLoRA Fine-tuning Script for Turkish Document Understanding

This script fine-tunes a large language model (LLM) using QLoRA (Quantized LoRA)
for Turkish document information extraction and summarization.

Requirements:
- GPU with at least 8GB VRAM (for 7B models)
- bitsandbytes library for quantization
- accelerate library for distributed training

Usage:
    python train_llm_qlora.py --base_model meta-llama/Llama-2-7b-chat-hf --data_dir ./data/llm
"""

import os
import argparse
import torch
from pathlib import Path
from datasets import load_dataset, Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    BitsAndBytesConfig,
)
from peft import (
    prepare_model_for_kbit_training,
    LoraConfig,
    get_peft_model,
    TaskType,
)
from transformers import DataCollatorForLanguageModeling
import json


def load_instruction_dataset(data_dir: str):
    """
    Load instruction-following dataset for fine-tuning.
    
    Expected format: JSON file with list of examples:
    [
        {
            "instruction": "Aşağıdaki belgeden bilgi çıkar",
            "input": "Belge metni burada...",
            "output": "Çıkarılan bilgiler: ..."
        },
        ...
    ]
    """
    data_dir = Path(data_dir)
    
    def load_json_file(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    train_data = load_json_file(data_dir / 'train.json') if (data_dir / 'train.json').exists() else []
    val_data = load_json_file(data_dir / 'val.json') if (data_dir / 'val.json').exists() else []
    
    return train_data, val_data


def format_prompt(example, tokenizer):
    """
    Format instruction, input, and output into a prompt for the model.
    
    Uses a template suitable for Turkish instruction-following.
    """
    instruction = example.get('instruction', '')
    input_text = example.get('input', '')
    output = example.get('output', '')
    
    # Format for instruction-following models
    if input_text:
        prompt = f"### Talimat:\n{instruction}\n\n### Girdi:\n{input_text}\n\n### Çıktı:\n{output}"
    else:
        prompt = f"### Talimat:\n{instruction}\n\n### Çıktı:\n{output}"
    
    return prompt


def preprocess_function(examples, tokenizer, max_length=512):
    """
    Tokenize and format the dataset.
    
    When batched=True, examples is a dict with column names as keys and lists as values.
    We need to zip the columns together to get individual examples.
    """
    # Extract columns
    instructions = examples.get('instruction', [])
    inputs = examples.get('input', [])
    outputs = examples.get('output', [])
    
    # Zip columns together to create individual examples
    prompts = []
    for i in range(len(instructions)):
        ex = {
            'instruction': instructions[i] if i < len(instructions) else '',
            'input': inputs[i] if i < len(inputs) else '',
            'output': outputs[i] if i < len(outputs) else '',
        }
        prompts.append(format_prompt(ex, tokenizer))
    
    model_inputs = tokenizer(
        prompts,
        max_length=max_length,
        truncation=True,
        padding='max_length',
    )
    
    # For causal LM, labels are the same as input_ids
    model_inputs['labels'] = model_inputs['input_ids'].copy()
    
    return model_inputs


def main():
    parser = argparse.ArgumentParser(description='Fine-tune LLM with QLoRA for Turkish documents')
    parser.add_argument('--base_model', type=str, default='meta-llama/Llama-2-7b-chat-hf',
                        help='Base model to fine-tune (must support 4-bit quantization)')
    parser.add_argument('--data_dir', type=str, default='./data/llm',
                        help='Directory containing train.json and val.json')
    parser.add_argument('--output_dir', type=str, default='./models/llm_qlora',
                        help='Output directory for the fine-tuned model')
    parser.add_argument('--num_epochs', type=int, default=3,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Training batch size (keep small for QLoRA)')
    parser.add_argument('--learning_rate', type=float, default=2e-4,
                        help='Learning rate for LoRA')
    parser.add_argument('--lora_r', type=int, default=16,
                        help='LoRA rank (r parameter)')
    parser.add_argument('--lora_alpha', type=int, default=32,
                        help='LoRA alpha parameter')
    parser.add_argument('--lora_dropout', type=float, default=0.1,
                        help='LoRA dropout')
    parser.add_argument('--max_length', type=int, default=512,
                        help='Maximum sequence length')
    parser.add_argument('--use_hf_dataset', type=str, default=None,
                        help='Use Hugging Face dataset name instead of local files')
    
    args = parser.parse_args()
    
    # Check for GPU
    if not torch.cuda.is_available():
        print("WARNING: CUDA not available. QLoRA training requires GPU.")
        print("Training will be very slow on CPU. Consider using a GPU instance.")
    
    # Load tokenizer
    print(f"Loading tokenizer from {args.base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    # Configure 4-bit quantization
    print("Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )
    
    # Load model with quantization
    print(f"Loading model from {args.base_model} (this may take a while)...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    
    # Prepare model for k-bit training
    model = prepare_model_for_kbit_training(model)
    
    # Configure LoRA
    print("Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  # Common for LLaMA
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    # Load dataset
    if args.use_hf_dataset:
        print(f"Loading dataset from Hugging Face: {args.use_hf_dataset}")
        datasets = load_dataset(args.use_hf_dataset)
    else:
        print(f"Loading dataset from {args.data_dir}...")
        train_data, val_data = load_instruction_dataset(args.data_dir)
        
        if not train_data:
            print("ERROR: No training data found. Please provide data_dir with train.json")
            print("Expected format: JSON file with list of {'instruction': ..., 'input': ..., 'output': ...}")
            return
        
        train_dataset = Dataset.from_list(train_data)
        val_dataset = Dataset.from_list(val_data) if val_data else None
        
        # Convert to DatasetDict (required for .map() method)
        datasets_dict = {'train': train_dataset}
        if val_dataset:
            datasets_dict['validation'] = val_dataset
        datasets = DatasetDict(datasets_dict)
    
    # Preprocess dataset
    print("Preprocessing dataset...")
    tokenized_datasets = datasets.map(
        lambda x: preprocess_function(x, tokenizer, args.max_length),
        batched=True,
        remove_columns=datasets['train'].column_names,
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
        logging_steps=10,
        eval_strategy='epoch' if 'validation' in tokenized_datasets else 'no',
        save_strategy='epoch',
        save_total_limit=3,
        load_best_model_at_end=True if 'validation' in tokenized_datasets else False,
        gradient_accumulation_steps=4,  # Effective batch size = batch_size * gradient_accumulation_steps
        fp16=True,  # Use mixed precision
        optim="paged_adamw_8bit",  # Memory-efficient optimizer
        push_to_hub=False,
    )
    
    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # Causal LM, not masked LM
    )
    
    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets['train'],
        eval_dataset=tokenized_datasets.get('validation'),
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    
    # Train
    print("Starting training...")
    print("NOTE: First epoch may be slow due to compilation. Subsequent epochs will be faster.")
    trainer.train()
    
    # Save model
    print(f"Saving model to {args.output_dir}...")
    trainer.save_model()
    tokenizer.save_pretrained(args.output_dir)
    
    # Save LoRA adapters separately
    model.save_pretrained(args.output_dir)
    
    print("Training completed!")
    print(f"Model saved to {args.output_dir}")
    print("To use the model, load it with:")
    print(f"  from peft import PeftModel")
    print(f"  model = PeftModel.from_pretrained(base_model, '{args.output_dir}')")


if __name__ == '__main__':
    main()