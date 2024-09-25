import re

def split_text_overlap(text, max_fragment_length, overlap_length):
    sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', text)

    fragments = []
    current_fragment = ""

    for sentence in sentences:
        if len(current_fragment) + len(sentence) <= max_fragment_length:
            current_fragment += sentence
        else:
            if current_fragment:
                fragments.append(current_fragment)

            current_fragment = sentence[:overlap_length]
            overlap = sentence[overlap_length:]
            while len(overlap) > max_fragment_length:
                fragments.append(current_fragment)
                current_fragment = overlap[:overlap_length]
                overlap = overlap[overlap_length:]

    if current_fragment:
        fragments.append(current_fragment)

    return fragments

def split_text(text, max_fragment_length):
    return [text[i:i+max_fragment_length] for i in range(0, len(text), max_fragment_length)]