#!/usr/bin/env python3
"""
Inference Script for Afaan Oromoo Passport Chatbot
Test your fine-tuned model with interactive chat.

Usage:
    python inference.py --model_path ./model
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# System prompt for the chatbot
SYSTEM_PROMPT = """Ati gargaarsa tajaajila paaspoortii Itoophiyaa. Gaaffiiwwan waa'ee paaspoortii, viizaa, fi imala biyya alaa kanneen Afaan Oromootin gaafataman deebii kenna. Odeeffannoo sirrii, gargaarsa faallaa, fi akkataa jaalalaa qaba. Namoota paaspoortii barbaadan gargaaruuf qophiidha."""

def parse_args():
    parser = argparse.ArgumentParser(description="Run inference with fine-tuned model")
    parser.add_argument("--model_path", type=str, 
                        default="/home/z/my-project/download/oromo_chatbot_finetune/model",
                        help="Path to fine-tuned model")
    parser.add_argument("--base_model", type=str, default=None,
                        help="Base model (if using LoRA adapters)")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Generation temperature")
    return parser.parse_args()

def load_model(model_path, base_model=None):
    """Load the fine-tuned model."""
    print(f"📥 Loading model from: {model_path}")
    
    # Check if it's a LoRA adapter
    adapter_config = False
    try:
        import json
        with open(f"{model_path}/adapter_config.json", "r") as f:
            adapter_config = True
    except:
        pass
    
    if adapter_config:
        # Load base model and adapter
        if base_model is None:
            # Try to determine base model from config
            base_model = "Qwen/Qwen2.5-3B-Instruct"  # Default
        
        print(f"   Loading base model: {base_model}")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        
        print(f"   Loading LoRA adapter: {model_path}")
        model = PeftModel.from_pretrained(base, model_path)
        tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    else:
        # Load full fine-tuned model
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    
    model.eval()
    return model, tokenizer

def format_prompt(instruction, system_prompt=SYSTEM_PROMPT):
    """Format the prompt for the model."""
    return f"""### System:
{system_prompt}

### Instruction:
{instruction}

### Response:
"""

def generate_response(model, tokenizer, prompt, max_new_tokens=256, temperature=0.7):
    """Generate a response from the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            do_sample=True,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    # Extract only the response part
    if "### Response:" in response:
        response = response.split("### Response:")[-1].strip()
    
    return response

def interactive_chat(model, tokenizer, args):
    """Run interactive chat session."""
    print("\n" + "=" * 60)
    print("🤖 Afaan Oromoo Passport Chatbot")
    print("=" * 60)
    print("Type your question in Afaan Oromoo (or 'quit' to exit)")
    print("-" * 60)
    
    while True:
        try:
            user_input = input("\n👤 You: ").strip()
            
            if user_input.lower() in ['quit', 'exit', 'q', 'bye']:
                print("\n👋 Nagaa dha! Goodbye!")
                break
            
            if not user_input:
                continue
            
            prompt = format_prompt(user_input)
            response = generate_response(
                model, tokenizer, prompt,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature
            )
            
            print(f"\n🤖 Bot: {response}")
            
        except KeyboardInterrupt:
            print("\n\n👋 Nagaa dha! Goodbye!")
            break

def test_samples(model, tokenizer, args):
    """Test with sample questions."""
    print("\n" + "=" * 60)
    print("🧪 Testing with Sample Questions")
    print("=" * 60)
    
    samples = [
        "Paaspoortii maali?",
        "Paaspoortii argachuuf maal barbaachisa?",
        "Finfinne keessatti ofiisii paaspoortii eessa jira?",
        "Paaspoortii daa'imaaf akkamitti argadha?",
        "Baasii paaspoortii meeqa?",
    ]
    
    for i, question in enumerate(samples, 1):
        print(f"\n--- Test {i} ---")
        print(f"❓ Question: {question}")
        
        prompt = format_prompt(question)
        response = generate_response(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature
        )
        
        print(f"💬 Response: {response}")

def main():
    args = parse_args()
    
    # Load model
    model, tokenizer = load_model(args.model_path, args.base_model)
    
    # Run tests
    test_samples(model, tokenizer, args)
    
    # Interactive chat
    interactive_chat(model, tokenizer, args)

if __name__ == "__main__":
    main()
