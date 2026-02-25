# Afaan Oromoo Passport Chatbot - Fine-tuning Guide

## 📋 Overview

This project provides a complete pipeline for fine-tuning a small language model to create a chatbot that answers questions about Ethiopian passport services in **Afaan Oromoo** (Oromo language).

## 📊 Dataset Summary

| Metric | Value |
|--------|-------|
| **Total Samples** | 392 |
| **Training Samples** | 352 |
| **Validation Samples** | 40 |
| **Language** | Afaan Oromoo |
| **Domain** | Ethiopian Passport Services |

## 🗂️ Project Structure

```
oromo_chatbot_finetune/
├── prepare_data.py      # Data preparation script
├── train.py             # Fine-tuning script (QLoRA)
├── inference.py         # Testing and inference script
├── requirements.txt     # Python dependencies
├── README.md            # This file
├── data/                # Prepared training data
│   ├── train_instruction.jsonl
│   ├── train_chat.jsonl
│   ├── train_alpaca.jsonl
│   ├── val_instruction.jsonl
│   ├── val_chat.jsonl
│   └── val_alpaca.jsonl
└── model/               # Fine-tuned model (after training)
```

## 🚀 Quick Start

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Prepare Data (Already Done)

```bash
python prepare_data.py
```

### Step 3: Fine-tune the Model

```bash
# Basic training with default settings (Qwen2.5-3B)
python train.py

# Custom configuration
python train.py \
    --model_name Qwen/Qwen2.5-3B-Instruct \
    --epochs 3 \
    --batch_size 4 \
    --learning_rate 2e-4 \
    --lora_r 16
```

### Step 4: Test the Model

```bash
python inference.py --model_path ./model
```

## 🖥️ Hardware Requirements

| Configuration | Minimum GPU | Recommended GPU |
|---------------|-------------|-----------------|
| **QLoRA (3B)** | 8GB VRAM | 12GB+ VRAM |
| **QLoRA (7B)** | 12GB VRAM | 16GB+ VRAM |
| **Full LoRA (3B)** | 16GB VRAM | 24GB+ VRAM |

**Budget Options:**
- Google Colab Free (T4 - 16GB) ✅ Works for QLoRA with 3B models
- Google Colab Pro (A100) - Faster training
- RunPod / Lambda Labs - Cheap GPU rentals ($0.20-0.50/hr)

## ⚙️ Training Configuration

### Default Settings

| Parameter | Value | Description |
|-----------|-------|-------------|
| `model_name` | Qwen/Qwen2.5-3B-Instruct | Base model |
| `epochs` | 3 | Training epochs |
| `batch_size` | 4 | Batch size per device |
| `learning_rate` | 2e-4 | Learning rate |
| `lora_r` | 16 | LoRA rank |
| `lora_alpha` | 32 | LoRA alpha |
| `max_seq_length` | 512 | Max sequence length |

### Recommended Models

| Model | Size | Why Choose It |
|-------|------|---------------|
| **Qwen2.5-3B-Instruct** | 3B | Best balance of size/performance |
| **Llama-3.2-3B-Instruct** | 3B | Latest Llama, good multilingual |
| **Gemma-2-2B** | 2B | Smallest option, efficient |
| **Qwen2.5-1.5B-Instruct** | 1.5B | Very fast, minimal resources |

## 📝 Data Format

### Instruction Format (Used by default)
```json
{
  "text": "### System:\n...\n\n### Instruction:\n...\n\n### Response:\n..."
}
```

### Chat Format (For chat models)
```json
{
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."},
    {"role": "assistant", "content": "..."}
  ]
}
```

## 🔧 Advanced Options

### Using Different Base Models

```bash
# Using Llama 3.2
python train.py --model_name meta-llama/Llama-3.2-3B-Instruct

# Using Gemma 2
python train.py --model_name google/gemma-2-2b-it

# Using Mistral (larger, more capable)
python train.py --model_name mistralai/Mistral-7B-Instruct-v0.3
```

### Adjusting LoRA Parameters

```bash
# Higher LoRA rank = more capacity but more parameters
python train.py --lora_r 32 --lora_alpha 64

# Lower LoRA rank = fewer parameters, faster training
python train.py --lora_r 8 --lora_alpha 16
```

## 📦 Deployment Options

### Option 1: Ollama (Easiest)

```bash
# Convert to GGUF and import to Ollama
# See: https://github.com/ollama/ollama/blob/main/docs/import.md
```

### Option 2: vLLM (Production)

```python
from vllm import LLM, SamplingParams

llm = LLM(model="./model")
params = SamplingParams(temperature=0.7, max_tokens=256)
outputs = llm.generate(["Paaspoortii maali?"], params)
```

### Option 3: Hugging Face Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

base = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
model = PeftModel.from_pretrained(base, "./model")
tokenizer = AutoTokenizer.from_pretrained("./model")
```

### Option 4: FastAPI Server

```python
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class Query(BaseModel):
    text: str

@app.post("/chat")
async def chat(query: Query):
    response = generate_response(model, tokenizer, query.text)
    return {"response": response}
```

## 📈 Improving the Model

### Expand Your Dataset

1. **Add more conversation examples**
2. **Include varied question phrasings**
3. **Add greetings and small talk**
4. **Include edge cases and error handling**

### Data Augmentation Ideas

```python
# Example: Paraphrase questions
original = "Paaspoortii maali?"
paraphrases = [
    "Paaspoortii akkaataadha?",
    "Waa'ee paaspoortii naa gorsi.",
    "Maal paaspoortii jedhama?"
]
```

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `batch_size` to 2 or 1
- Reduce `max_seq_length` to 256
- Use a smaller model (1.5B instead of 3B)

### Slow Training
- Ensure you're using GPU (check `torch.cuda.is_available()`)
- Use `gradient_checkpointing=True` (already enabled)
- Reduce model size or sequence length

### Poor Quality Output
- Increase training epochs
- Expand training dataset
- Adjust `learning_rate` (try 1e-4 or 3e-4)
- Increase LoRA rank (`lora_r`)

## 📚 Resources

- [QLoRA Paper](https://arxiv.org/abs/2305.14314)
- [PEFT Documentation](https://huggingface.co/docs/peft)
- [TRL Documentation](https://huggingface.co/docs/trl)
- [Afaan Oromoo Resources](https://en.wikipedia.org/wiki/Oromo_language)

## 📄 License

This project is provided for educational purposes. Please ensure you comply with the license terms of the base models used (Qwen, Llama, etc.).

---

**Created for fine-tuning Afaan Oromoo chatbot with 392 passport service Q&A pairs.**
