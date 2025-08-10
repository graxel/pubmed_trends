import os
import json
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
from tqdm import tqdm
import pandas as pd

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')


stop_words = set(stopwords.words('english'))
extra_stopwords = {'et', 'al', 'figure', 'table', 'doi', 'preprint', 'copyright'}
stop_words.update(extra_stopwords)


INPUT_FOLDER = 'data/processed_very_new'
OUTPUT_FOLDER = 'data/tokenized'

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

def is_valid_date(date_str):
    try:
        pd.to_datetime(date_str, errors='raise')
        return True
    except Exception:
        return False

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)

for filename in tqdm(sorted(os.listdir(INPUT_FOLDER))):
    idx = 0
    if not filename.endswith('.json'):
        continue

    input_path = os.path.join(INPUT_FOLDER, filename)
    output_path = os.path.join(OUTPUT_FOLDER, filename)

    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    # Filter elements with valid PubDate
    # filtered = [item for item in data if 'PubDate' in item and isinstance(item['PubDate'], str) and is_valid_date(item['PubDate'])]
    filtered = []
    for item in data:
        if 'PubDate' in item and isinstance(item['PubDate'], str) and is_valid_date(item['PubDate']):
            item['index'] = idx
            item['tokens'] = preprocess_text(item['Abstract'])
            filtered.append(item)
            idx += 1

    # Write filtered data to new file
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(filtered, f_out, indent=4, ensure_ascii=False)

    print(f"Processed {filename}: {len(filtered)} valid entries saved.")
