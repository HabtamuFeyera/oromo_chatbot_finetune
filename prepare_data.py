#!/usr/bin/env python3
"""
Data Preparation Script for Afaan Oromoo Passport Chatbot Fine-tuning
This script converts your raw JSONL data into the proper format for training.
"""

import json
import random
from pathlib import Path

# Configuration
INPUT_FILE = "/home/z/my-project/upload/afaan_oromoo_jsonl_20260225_bfc421.txt"
OUTPUT_DIR = Path("/home/z/my-project/download/oromo_chatbot_finetune/data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# System prompt for the chatbot (in Afaan Oromoo)
SYSTEM_PROMPT = """Ati gargaarsa tajaajila paaspoortii Itoophiyaa. Gaaffiiwwan waa'ee paaspoortii, viizaa, fi imala biyya alaa kanneen Afaan Oromootin gaafataman deebii kenna. Odeeffannoo sirrii, gargaarsa faallaa, fi akkataa jaalalaa qaba. Namoota paaspoortii barbaadan gargaaruuf qophiidha."""

def load_data(filepath):
    """Load JSONL data from file."""
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Warning: Could not parse line: {e}")
    return data

def format_for_training(sample, format_type="instruction"):
    """Format a single sample for training."""
    
    if format_type == "instruction":
        # Simple instruction format
        return {
            "text": f"""### System:
{SYSTEM_PROMPT}

### Instruction:
{sample['instruction']}

### Response:
{sample['output']}"""
        }
    
    elif format_type == "chat":
        # Chat format (for chat models like Qwen, Llama)
        return {
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": sample['instruction']},
                {"role": "assistant", "content": sample['output']}
            ]
        }
    
    elif format_type == "alpaca":
        # Alpaca format
        return {
            "instruction": sample['instruction'],
            "input": "",
            "output": sample['output'],
            "system": SYSTEM_PROMPT
        }
    
    return sample

def split_data(data, train_ratio=0.9, val_ratio=0.1, seed=42):
    """Split data into train and validation sets."""
    random.seed(seed)
    shuffled = data.copy()
    random.shuffle(shuffled)
    
    train_size = int(len(shuffled) * train_ratio)
    train_data = shuffled[:train_size]
    val_data = shuffled[train_size:]
    
    return train_data, val_data

def save_jsonl(data, filepath):
    """Save data to JSONL file."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"Saved {len(data)} samples to {filepath}")

def main():
    print("=" * 60)
    print("Afaan Oromoo Passport Chatbot - Data Preparation")
    print("=" * 60)
    
    # Load raw data
    print(f"\n📂 Loading data from: {INPUT_FILE}")
    raw_data = load_data(INPUT_FILE)
    print(f"   Loaded {len(raw_data)} samples")
    
    # Split data
    print("\n✂️  Splitting data into train/validation sets...")
    train_data, val_data = split_data(raw_data)
    print(f"   Train: {len(train_data)} samples")
    print(f"   Validation: {len(val_data)} samples")
    
    # Format for different model types
    formats = ["instruction", "chat", "alpaca"]
    
    for fmt in formats:
        print(f"\n📝 Creating {fmt} format...")
        formatted_train = [format_for_training(s, fmt) for s in train_data]
        formatted_val = [format_for_training(s, fmt) for s in val_data]
        
        save_jsonl(formatted_train, OUTPUT_DIR / f"train_{fmt}.jsonl")
        save_jsonl(formatted_val, OUTPUT_DIR / f"val_{fmt}.jsonl")
    
    # Create a sample for testing
    print("\n📋 Sample formatted data (instruction format):")
    sample = format_for_training(raw_data[0], "instruction")
    print("-" * 40)
    print(sample["text"][:500] + "...")
    
    # Statistics
    print("\n📊 Dataset Statistics:")
    print(f"   Total samples: {len(raw_data)}")
    print(f"   Training samples: {len(train_data)}")
    print(f"   Validation samples: {len(val_data)}")
    
    # Calculate average lengths
    avg_instruction = sum(len(s['instruction'].split()) for s in raw_data) / len(raw_data)
    avg_output = sum(len(s['output'].split()) for s in raw_data) / len(raw_data)
    print(f"   Avg instruction length: {avg_instruction:.1f} words")
    print(f"   Avg output length: {avg_output:.1f} words")
    
    print("\n✅ Data preparation complete!")
    print(f"   Output directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
