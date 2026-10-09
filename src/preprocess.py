import os
import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from collections import Counter
from sklearn.model_selection import train_test_split
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

nlp = spacy.load('en_core_web_sm')

def perform_eda(df_laptops, df_restaurants, output_dir):
    """Generates EDA plots and summary."""
    os.makedirs(output_dir, exist_ok=True)
    summary_lines = []
    
    for domain, df in zip(['Laptops', 'Restaurants'], [df_laptops, df_restaurants]):
        df['num_aspects'] = df['aspects'].apply(len)
        df['aspect_terms'] = df['aspects'].apply(lambda x: [a['term'] for a in x])
        df['polarities'] = df['aspects'].apply(lambda x: [a['polarity'] for a in x])
        
        # Sentences with mixed polarities
        df['mixed_polarity'] = df['polarities'].apply(lambda x: len(set(x)) > 1)
        
        # Aspect lengths (words)
        all_aspects = [a for sublist in df['aspect_terms'] for a in sublist]
        aspect_lengths = [len(a.split()) for a in all_aspects]
        
        # Basic stats
        total_sents = len(df)
        total_aspects = len(all_aspects)
        summary_lines.append(f"{domain}: {total_sents} sentences, {total_aspects} aspects.")
        summary_lines.append(f"{domain}: Sentences with mixed polarities: {df['mixed_polarity'].sum()}")
        
        # Plot Number of Aspects per Sentence
        plt.figure(figsize=(6, 4))
        sns.countplot(x='num_aspects', data=df)
        plt.title(f'Number of Aspects per Sentence ({domain})')
        plt.savefig(os.path.join(output_dir, f'{domain}_num_aspects.png'))
        plt.close()
        
        # Plot Polarity Distribution
        all_polarities = [p for sublist in df['polarities'] for p in sublist]
        plt.figure(figsize=(6, 4))
        sns.countplot(x=all_polarities, order=['positive', 'negative', 'neutral'])
        plt.title(f'Polarity Distribution ({domain})')
        plt.savefig(os.path.join(output_dir, f'{domain}_polarities.png'))
        plt.close()
        
        # Top 20 aspects
        top_20 = Counter(all_aspects).most_common(20)
        plt.figure(figsize=(10, 6))
        sns.barplot(x=[x[1] for x in top_20], y=[x[0] for x in top_20])
        plt.title(f'Top 20 Aspect Terms ({domain})')
        plt.savefig(os.path.join(output_dir, f'{domain}_top20_aspects.png'))
        plt.close()

    summary_path = os.path.join(output_dir, '..', 'eda_summary.md')
    with open(summary_path, 'w') as f:
        f.write("\n".join(summary_lines))
    logging.info(f"Saved EDA plots to {output_dir} and summary to {summary_path}")

def get_bio_labels(text, aspects):
    """Converts aspect spans to BIO labels at the token level using spaCy."""
    doc = nlp(text)
    labels = ['O'] * len(doc)
    
    for aspect in aspects:
        start_char = aspect['from']
        end_char = aspect['to']
        
        # Find which tokens overlap with the character span
        for i, token in enumerate(doc):
            if token.idx >= end_char:
                break
            if token.idx + len(token.text) > start_char:
                # Overlap found
                if token.idx == start_char or (labels[i] == 'O' and sum([1 for l in labels[:i] if l.startswith('B-') or l.startswith('I-')]) == 0): # simplification
                    pass # We will do a stricter check below

        # Better BIO labeling based on char offsets
        span_tokens = []
        for i, token in enumerate(doc):
            if token.idx >= start_char and token.idx < end_char:
                span_tokens.append(i)
                
        if span_tokens:
            labels[span_tokens[0]] = 'B-ASP'
            for idx in span_tokens[1:]:
                labels[idx] = 'I-ASP'
                
    tokens = [token.text for token in doc]
    return tokens, labels

def process_data_splits():
    PROCESSED_DIR = os.path.join('data', 'processed')
    
    for domain in ['laptops', 'restaurants']:
        train_path = os.path.join(PROCESSED_DIR, f"{domain}_train.json")
        test_path = os.path.join(PROCESSED_DIR, f"{domain}_test.json")
        
        if not os.path.exists(train_path) or not os.path.exists(test_path):
            logging.warning(f"Data not found for {domain}")
            continue
            
        df_train = pd.read_json(train_path, lines=True)
        df_test = pd.read_json(test_path, lines=True)
        
        if domain == 'laptops':
            df_laptops = df_train
        else:
            df_restaurants = df_train
            
        # Split train into train and val (10% stratified by polarity, roughly)
        # Since a sentence can have multiple polarities, we stratify by majority polarity or just random split
        # We will use random split for simplicity while ensuring 10%
        train, val = train_test_split(df_train, test_size=0.1, random_state=42)
        
        for split_name, df in zip(['train_split', 'val_split', 'test'], [train, val, df_test]):
            parsed_data = []
            for _, row in df.iterrows():
                tokens, bio_labels = get_bio_labels(row['text'], row['aspects'])
                
                # Format (a) Token-level BIO tags
                
                # Format (b) (sentence, aspect, polarity) triples
                triples = []
                for asp in row['aspects']:
                    triples.append({
                        'aspect': asp['term'],
                        'polarity': asp['polarity']
                    })
                    
                parsed_data.append({
                    'sent_id': row['sent_id'],
                    'text': row['text'],
                    'tokens': tokens,
                    'bio_labels': bio_labels,
                    'triples': triples
                })
                
            out_file = os.path.join(PROCESSED_DIR, f"{domain}_{split_name}_processed.json")
            pd.DataFrame(parsed_data).to_json(out_file, orient='records', lines=True)
            logging.info(f"Saved {out_file}")

    if 'df_laptops' in locals() and 'df_restaurants' in locals():
        perform_eda(df_laptops, df_restaurants, os.path.join('results', 'figures'))

if __name__ == "__main__":
    process_data_splits()
