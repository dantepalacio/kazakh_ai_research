import torch
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForSeq2SeqLM, TranslationPipeline
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


def load_tilmash():
    login('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')

    model = AutoModelForSeq2SeqLM.from_pretrained('issai/tilmash')
    tokenizer = AutoTokenizer.from_pretrained("issai/tilmash")
    # tilmash = TranslationPipeline(model = model, tokenizer = tokenizer, src_lang = "rus_Cyrl", tgt_lang = "kaz_Cyrl", max_length = 1024)
    return model,tokenizer