import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import login

login('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')

def load_model():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-9b-it")
    model = AutoModelForCausalLM.from_pretrained(
        "google/gemma-2-9b-it",
        device_map="cuda",
        torch_dtype=torch.float16, 
        load_in_4bit=True
    )
    return model, tokenizer