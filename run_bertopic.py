import os
import json
from datetime import datetime
from bertopic import BERTopic
import pandas as pd
from tqdm import tqdm
import random

# Directory where tokenized JSON files are stored
DATA_DIR = "data/tokenized/"
METATOPIC = 'cancer'
OUTPUT_TOPIC_HTML = f"{METATOPIC}/topic_scatter.html"
OUTPUT_TOPICS_TIME_HTML = f"{METATOPIC}/topic_over_time.html"

def load_all_data(data_dir):
    """Load all JSON files from the directory and combine into a single list"""
    all_records = []
    file_count = 0
    for filename in tqdm(os.listdir(data_dir)):
        if filename.endswith('.json'):
            filepath = os.path.join(data_dir, filename)
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        metatopic_data = [entry for entry in data if METATOPIC in entry['tokens']]
                        all_records.extend(metatopic_data)
                        file_count += 1
                    else:
                        print(f"Warning: File {filename} does not contain a list.")
            except Exception as e:
                print(f"Error reading {filename}: {e}")
    print(f"Loaded {len(all_records)} records from {file_count} files.")
    return all_records

def parse_pubdate(pubdate_str):
    try:
        return pd.to_datetime(pubdate_str).year # .isoformat()
    except Exception:
        # If format differs or parsing fails, return None to ignore that record in timeline
        return None

def prepare_corpus_and_timestamps(records):
    """Extract cleaned documents ('tokens') and their timestamps from loaded records"""
    docs = []
    timestamps = []
    for rec in records:
        tokens = rec.get('tokens', '')
        pubdate = rec.get('PubDate', '')
        if tokens and isinstance(tokens, str) and tokens.strip():
            dt = parse_pubdate(pubdate)
            if dt is not None:
                docs.append(tokens.strip())
                timestamps.append(dt)
    print(f"Prepared {len(docs)} documents with valid timestamps for modeling.")
    return docs, timestamps

def main():
    sampled_records = []
    years = range(2005, 2026)
    # sample_size = 200000 # about 5000 papers per year
    samples_per_year = 10000 # sample_size // len(years)
    print("Sorting data...")
    records = None
    for year in tqdm(years):
        master_file = f"data/master_{year}.json"
        if os.path.exists(master_file):
            # Check if the master file exists for that year, if so, load it
            print(f"Loading {master_file}...")
            with open(master_file, 'r', encoding='utf-8') as rf:
                records_of_year = json.load(rf)
        else:
            if records is None:
                # load records
                print(f"Loading data/master_filtered_2005.json...")
                with open('data/master_filtered_2005.json', 'r', encoding='utf-8') as rf:
                    records = json.load(rf)
            # Extract records_of_year from records
            records_of_year = [record for record in records if pd.to_datetime(record['PubDate']).year == year]
            
            # Save for future:
            print(f"Saving master_file for {year}...")
            with open(master_file, 'w') as wf:
                json.dump(records_of_year, wf)
        print("Sampling data...")
        sample_of_records_of_year = random.sample(records_of_year, samples_per_year)
        sampled_records.extend(sample_of_records_of_year)

    print("Preparing corpus and timestamps...")
    docs, timestamps = prepare_corpus_and_timestamps(sampled_records)

    if len(docs) == 0:
        print("No valid documents found. Exiting.")
        return

    print("Fitting BERTopic model...")
    topic_model = BERTopic(nr_topics=30)  # adjust number of topics as needed
    topics, probs = topic_model.fit_transform(docs)

    print("Saving fitted BERTopic model to disk...")
    joblib.dump(topic_model, "bertopic_model.pkl")

    print(f"Saving interactive visualizations to '{OUTPUT_TOPIC_HTML}' and '{OUTPUT_TOPICS_TIME_HTML}'...")
    
    # Visualize topics scatter plot
    fig_topics = topic_model.visualize_topics()
    fig_topics.write_html(OUTPUT_TOPIC_HTML)

    # Visualize topics over time
    topics_over_time = topic_model.topics_over_time(docs, timestamps, global_tuning=True)

    fig_topics_time = topic_model.visualize_topics_over_time(topics_over_time, height=700, width=1000)
    fig_topics_time.update_yaxes(type="log")
    fig_topics_time.write_html(OUTPUT_TOPICS_TIME_HTML)

    print("All done! You can open the HTML files in a browser or embed them in your webpage.")

if __name__ == "__main__":
    main()
