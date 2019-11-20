# This script is for process the raw data and generate the data for further analyses.

from lxml import etree as ET
import pandas as pd
import re
from nltk.corpus import stopwords

def clean_text(text, remove_digits=False):
    pattern = r'[^a-zA-Z0-9\s]' if not remove_digits else r'[^a-zA-Z\s]'
    text = re.sub(pattern, '', text)
    text = text.replace('\t', ' ').replace('\n', ' ')
    text = ' '.join(text.split()).lower()

    return text

def remove_stopwords(list_of_words, list_of_stopwords):
    filtered_text = []
    for w in list_of_words:
        if w not in list_of_stopwords:
            filtered_text.append(w)

    return 