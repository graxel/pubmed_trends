import os
import gzip
import shutil
import json
from tqdm import tqdm
import xml.etree.ElementTree as ET
from urllib.request import urlretrieve
import pandas as pd

DATA_DIR = 'data'
DOWNLOAD_DIR = os.path.join(DATA_DIR, 'download')
RAW_DIR = os.path.join(DATA_DIR, 'raw')
PROCESSED_DIR = os.path.join(DATA_DIR, 'processed_new')

# List your files of interest here (or build a function to get them dynamically)
FILE_NAMES = [f"pubmed25n{num:0>4}.xml.gz" for num in range(1, 550)]

BASE_URL = "https://ftp.ncbi.nlm.nih.gov/pubmed/baseline/"

def ensure_dirs():
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    os.makedirs(RAW_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

def download_file(filename):
    url = BASE_URL + filename
    dest_path = os.path.join(DOWNLOAD_DIR, filename)
    if not os.path.exists(dest_path):
        print(f"Downloading {filename}...")
        urlretrieve(url, dest_path)
    # else:
    #     print(f"{filename} already downloaded.")
    return dest_path

def extract_gz(gz_path):
    raw_path = os.path.join(RAW_DIR, os.path.basename(gz_path).replace(".gz", ""))
    if not os.path.exists(raw_path):
        # print(f"Extracting {gz_path} to {raw_path}...")
        with gzip.open(gz_path, 'rb') as f_in:
            with open(raw_path, 'wb') as f_out:
                shutil.copyfileobj(f_in, f_out)
    else:
        print(f"{raw_path} already extracted.")
    return raw_path



def parse_articles(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    articles = []

    for article in root.findall('.//Article'):
        # Abstract
        abstract_elems = article.findall('.//AbstractText')
        abstract = ' '.join(a.text.strip() for a in abstract_elems if a.text)
        if abstract == '':
            continue

        # PubDate
        pubdate_elem = article.find('./Journal/JournalIssue/PubDate')
        year = pubdate_elem.findtext('Year', default='').strip() if pubdate_elem is not None else ''
        month = pubdate_elem.findtext('Month', default='').strip() if pubdate_elem is not None else ''
        day = pubdate_elem.findtext('Day', default='').strip() if pubdate_elem is not None else ''
        # Format as YYYY-MM-DD (or as much as we have)
        pubdate_parts = [year]
        if month:
            pubdate_parts.append(month)
        if day:
            pubdate_parts.append(day)
        pubdate = '-'.join(pubdate_parts)
        try:
            stripped_pubdate = pubdate.strip()
            pd.to_datetime(stripped_pubdate)
        except:
            print(f"found pubdate: {pubdate_elem} - skipping")
            continue

        # ArticleTitle
        title_elem = article.find('ArticleTitle')
        article_title = '' if title_elem.text is None else title_elem.text.strip()
        # article_title = title_elem.text.strip() if title_elem is not None else ''

        # AuthorList
        author_list = []
        for author in article.findall('.//Author'):
            fore_name = author.findtext('ForeName', default='').strip()
            last_name = author.findtext('LastName', default='').strip()
            if fore_name or last_name:
                author_list.append(f"{fore_name} {last_name}".strip())

        

        # Build article dict
        articles.append({
            "ArticleTitle": article_title,
            "AuthorList": author_list,
            "Abstract": abstract,
            "PubDate": stripped_pubdate,
        })

    return articles

def process_file(filename):
    # print(f"Processing {filename} ...")
    gz_path = download_file(filename)
    xml_path = extract_gz(gz_path)
    articles = parse_articles(xml_path)

    base_filename = os.path.basename(xml_path).replace(".xml", "")
    output_path = os.path.join(PROCESSED_DIR, base_filename + ".json")

    # print(f"Saving parsed articles to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f_out:
        json.dump(articles, f_out, indent=4, ensure_ascii=False)

def main():
    ensure_dirs()
    for filename in tqdm(FILE_NAMES, desc="Processing files"):
        process_file(filename)
    print("All files processed.")

if __name__ == "__main__":
    main()
