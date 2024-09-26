import time

from utils import extract_text_from_docx, replace_text_in_docx
from load_model import load_model, load_tilmash
from generation import translate_qa, translate_tilmash

start = time.time()

input_file_path = 'anketa.docx'
output_file_path = 'translsated.docx'

data_from_docx = extract_text_from_docx(file_path=input_file_path)

# model, tokenizer = load_model()

# translations = []
# for text in data_from_docx:
#     temp_translate = translate_qa(model,tokenizer,text)
#     translations.append(temp_translate)

model,tokenizer = load_tilmash()

translations = []
for source_text in data_from_docx:
    temp_translate = translate_tilmash(model,tokenizer,source_text)
    print(f'Source text - {source_text}, Translated text - {temp_translate} \n\n --------------------')
    translations.append(temp_translate)

replaced_text_in_docx = replace_text_in_docx(input_file_path, output_file_path, translations)


print(translations)
end = time.time()
print(f'execution: {end-start}')