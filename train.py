#!/usr/bin/env python3
"""
Fine-tuning Script for Afaan Oromoo Passport Chatbot
Uses QLoRA for efficient training on consumer GPUs.

Requirements:
    pip install torch transformers datasets peft accelerate bitsandbytes trl

Usage:
    python train.py --model_name Qwen/Qwen2.5-3B-Instruct --epochs 3 --batch_size 4
"""

import argparse
import torch
from pathlib import Path
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, TaskType
from trl import SFTTrainer

# Default paths
DATA_DIR = Path("/home/z/my-project/download/oromo_chatbot_finetune/data")
OUTPUT_DIR = Path("/home/z/my-project/download/oromo_chatbot_finetune/model")

def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune a model for Afaan Oromoo chatbot")
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen2.5-3B-Instruct",
                        help="Base model to fine-tune")
    parser.add_argument("--data_dir", type=str, default=str(DATA_DIR),
                        help="Directory containing training data")
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_DIR),
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--epochs", type=int, default=3,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size per device")
    parser.add_argument("--learning_rate", type=float, default=2e-4,
                        help="Learning rate")
    parser.add_argument("--max_seq_length", type=int, default=512,
                        help="Maximum sequence length")
    parser.add_argument("--lora_r", type=int, default=16,
                        help="LoRA rank")
    parser.add_argument("--lora_alpha", type=int, default=32,
                        help="LoRA alpha")
    parser.add_argument("--use_4bit", action="store_true", default=True,
                        help="Use 4-bit quantization (QLoRA)")
    return parser.parse_args()

def print_trainable_parameters(model):
    """Print the number of trainable parameters."""
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(f"Trainable parameters: {trainable_params:,} / {all_param:,} ({100 * trainable_params / all_param:.2f}%)")

def main():
    args = parse_args()
    
    print("=" * 60)
    print("Afaan Oromoo Passport Chatbot - Fine-tuning")
    print("=" * 60)
    print(f"\n📦 Model: {args.model_name}")
    print(f"📁 Data directory: {args.data_dir}")
    print(f"💾 Output directory: {args.output_dir}")
    print(f"⚙️  Epochs: {args.epochs}, Batch size: {args.batch_size}")
    print(f"📊 Learning rate: {args.learning_rate}")
    print(f"🔧 LoRA rank: {args.lora_r}, alpha: {args.lora_alpha}")
    
    # Check GPU
    if torch.cuda.is_available():
        print(f"\n✅ GPU detected: {torch.cuda.get_device_name(0)}")
        print(f"   VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        print("\n⚠️  No GPU detected. Training will be very slow on CPU.")
    
    # Quantization config for QLoRA
    if args.use_4bit:
        print("\n🔧 Setting up 4-bit quantization (QLoRA)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    else:
        bnb_config = None
    
    # Load model
    print(f"\n📥 Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16 if not args.use_4bit else None,
    )
    
    # Prepare for k-bit training (needed for QLoRA)
    if args.use_4bit:
        model = prepare_model_for_kbit_training(model)
    
    # LoRA configuration
    print("\n🔧 Configuring LoRA...")
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    
    # Apply LoRA
    model = get_peft_model(model, lora_config)
    print_trainable_parameters(model)
    
    # Load tokenizer
    print("\n📥 Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # Fix for fp16 training
    
    # Load dataset
    print("\n📂 Loading dataset...")
    train_dataset = load_dataset("json", data_files=str(Path(args.data_dir) / "train_instruction.jsonl"), split="train")
    val_dataset = load_dataset("json", data_files=str(Path(args.data_dir) / "val_instruction.jsonl"), split="train")
    print(f"   Training samples: {len(train_dataset)}")
    print(f"   Validation samples: {len(val_dataset)}")
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        fp16=True,
        logging_steps=10,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=2,
        load_best_model_at_end=True,
        optim="paged_adamw_8bit",
        warmup_ratio=0.03,
        weight_decay=0.01,
        report_to="none",  # Disable wandb/tensorboard
        gradient_checkpointing=True,
    )
    
    # Create trainer
    print("\n🚀 Starting training...")
    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        dataset_text_field="text",
    )
    
    # Train
    trainer.train()
    
    # Save model
    print(f"\n💾 Saving model to: {args.output_dir}")
    trainer.model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    
    # Save training config
    config = {
        "model_name": args.model_name,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "max_seq_length": args.max_seq_length,
    }
    import json
    with open(Path(args.output_dir) / "training_config.json", "w") as f:
        json.dump(config, f, indent=2)
    
    print("\n✅ Training complete!")
    print(f"   Model saved to: {args.output_dir}")

if __name__ == "__main__":
    main()
