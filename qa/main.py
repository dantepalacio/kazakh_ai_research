import time

from load_model import load_model
from utils import split_text
from generation import *

start_time = time.time()

model, tokenizer = load_model()

name_text = '7903324.txt'
with open(f'kaz_audio/{name_text}', 'r', encoding='utf-8') as f:
    text = f.read()

text_result = text.split("'")[1]
# print(text_result)

fragments = split_text(text_result, max_fragment_length=5000)

summarization = summarize(model,tokenizer,fragments)
print(summarization)
ru_qa_response = ru_qa(model,tokenizer,summarization)
print('XXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXXX')
print(ru_qa_response)
translated_qa = translate_qa(model,tokenizer,ru_qa_response)
print('################################')
print(translated_qa)
end_time = time.time()
print(f'Execution: {end_time-start_time}')