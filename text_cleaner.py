import json
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import os
from tqdm import tqdm
import pandas as pd

nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')


stop_words = set(stopwords.words('english'))
extra_stopwords = {'et', 'al', 'figure', 'table', 'doi', 'preprint', 'copyright'}
stop_words.update(extra_stopwords)


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    tokens = word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    return ' '.join(tokens)


def preprocess_file(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    docs = []
    for record in data:
        date = record.get('PubDate', '')
        try:
            pd.to_datetime(date)
        except:
            print(input_path, date)
        abstract = record.get('Abstract', '')
        if abstract and isinstance(abstract, str) and len(abstract) > 20:
            cleaned = preprocess_text(abstract)
            if cleaned.strip():
                docs.append(cleaned)

    # Save the cleaned abstracts as a text file, one abstract per line
    with open(output_path, 'w', encoding='utf-8') as f_out:
        for doc in docs:
            f_out.write(doc + '\n')

    # print(f"Processed and saved {len(docs)} abstracts to {output_path}")


# Example: preprocess all json files in a folder
input_folder = "data/processed"
output_folder = "data/cleaned"
os.makedirs(output_folder, exist_ok=True)

for filename in tqdm(os.listdir(input_folder)):
    if filename.endswith('.json'):
        input_path = os.path.join(input_folder, filename)
        # Change output filename extension to .txt for the cleaned text
        output_path = os.path.join(output_folder, filename.replace('.json', '_cleaned.txt'))
        preprocess_file(input_path, output_path)
