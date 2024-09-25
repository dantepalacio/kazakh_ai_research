def summarize(model,tokenizer, fragments):
    question = '''Твоя задача сократить данный отрезок текста, максимально сохранив контекст, так как это интервью, постарайся в своем ответе рассказать вкратце о словах интервьюируемого человека. Если в заданном отрезке текст неясен, то постарайся на него не отвечать. Твой ответ должен быть на чистом и грамотном русском языке.'''

    response = []
    for text in fragments:
        fragment_messages = [
            {"role": "user", "content": f"{question} \nОтрезок текста:\n {text}"},
        ]
        input_ids = tokenizer.apply_chat_template(fragment_messages, return_tensors="pt", return_dict=True).to("cuda")
        
        outputs = model.generate(
            **input_ids,
            max_new_tokens=1024,
            do_sample=False,
            # temperature = 0.5,
            pad_token_id=tokenizer.eos_token_id, 
            eos_token_id=tokenizer.eos_token_id,
            output_scores=False,
            return_dict_in_generate=False,
        )
        result = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Берём последние 5 слов из input_text для создания маркера
        num_words_for_marker = 5
        input_text_words = text.split()  # Разбиваем текст на слова
        if len(input_text_words) > num_words_for_marker:
            start_marker = ' '.join(input_text_words[-num_words_for_marker:])  # Берём последние n слов
        else:
            start_marker = text  # Если слов меньше, используем весь текст
        
        # Ищем start_marker в сгенерированном ответе и удаляем всё, что до него
        answer_start = result.find(start_marker)
        if answer_start != -1:
            result = result[answer_start + len(start_marker):].strip()
        
        # print(result)
        # print('============')
        response.append(result)
    final_response = ' '.join(response)
    # print(final_response)
    return final_response
    

def ru_qa(model,tokenizer, text):
    question = '''
    - Если говорить о прошедшем месяце, о чем больше всего думал собеседник? Что вас больше всего волновало, вызывало наибольшее беспокойство?
    - По наблюдениям собеседника, цены на какие товары и услуги за прошедшие 12 месяцев выросли очень сильно?
    '''
    # question = '''
    # - Если говорить о прошедшем месяце, о чем больше всего думал собеседник? Что его больше всего волновало, вызывало наибольшее беспокойство?
    # '''

    template = f'''Твоя задача найти ответы в тексте на вопросы. Я дам вам интервью из реальной жизни и вопросы. Твоя задача вернуть только ответ в развернутом и подробном виде, без смайликов и не предлагай помочь с чем-то другим. Ваша задача – найти и вернуть мне ответы в заданном контексте на следующие вопросы: 
    ### Вопрос:
    {question}

    ### Контекст:
    {text}

    ### Ответ:
    '''

    messages = [
        {"role": "user", "content": template},
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")

    outputs = model.generate(
        **input_ids,
        max_new_tokens=1024,
        do_sample=True,
        temperature=0.3,
        repetition_penalty=1.5,
        pad_token_id=tokenizer.eos_token_id, 
        eos_token_id=tokenizer.eos_token_id,
        output_scores=False,
        return_dict_in_generate=False,
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    

    start_marker = "### Ответ:"
    answer_start = result.find(start_marker)
    if answer_start != -1:
        answer_result = result[answer_start + len(start_marker):].strip()
    print('--------------')
    # print(answer_result)
    return answer_result


def translate_qa(model,tokenizer,text):
    translate_template = f'''Ты носитель казахского и русского языков. Твоя задача правильно перевести текст с русского языка на казахский. Твой ответ всегда должен быть чисто на казахском и содержать только перевод:
    ### Текст, который нужно перевести:
    {text}

    ### Ответ:
    '''

    messages = [
        {"role": "user", "content": translate_template},
    ]
    input_ids = tokenizer.apply_chat_template(messages, return_tensors="pt", return_dict=True).to("cuda")

    outputs = model.generate(
        **input_ids,
        max_new_tokens=1024,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id, 
        eos_token_id=tokenizer.eos_token_id,
        output_scores=False,
        return_dict_in_generate=False,
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)

    start_marker = "### Ответ:"
    answer_start = result.find(start_marker)
    if answer_start != -1:
        answer_result = result[answer_start + len(start_marker):].strip()

    # print('---------------------')
    # print(answer_result)
    return result