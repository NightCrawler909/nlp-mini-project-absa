import os
import urllib.request
from datasets import load_dataset
import pandas as pd
import xml.etree.ElementTree as ET
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

SEMEVAL_URLS = {
    'laptops_train': 'https://raw.githubusercontent.com/williamsyedi/Aspect-Based-Sentiment-Analysis/master/data/Laptops_Train_v2.xml',
    'laptops_test': 'https://raw.githubusercontent.com/williamsyedi/Aspect-Based-Sentiment-Analysis/master/data/Laptops_Test_Gold.xml',
    'restaurants_train': 'https://raw.githubusercontent.com/williamsyedi/Aspect-Based-Sentiment-Analysis/master/data/Restaurants_Train_v2.xml',
    'restaurants_test': 'https://raw.githubusercontent.com/williamsyedi/Aspect-Based-Sentiment-Analysis/master/data/Restaurants_Test_Gold.xml'
}

RAW_DIR = os.path.join('data', 'raw')
PROCESSED_DIR = os.path.join('data', 'processed')

def download_semeval():
    os.makedirs(RAW_DIR, exist_ok=True)
    for name, url in SEMEVAL_URLS.items():
        filepath = os.path.join(RAW_DIR, f"{name}.xml")
        if not os.path.exists(filepath):
            logging.info(f"Downloading {name} from {url}")
            try:
                urllib.request.urlretrieve(url, filepath)
            except Exception as e:
                logging.error(f"Failed to download {name}: {e}")
                
def download_unlabeled():
    """Download a subset of Amazon US reviews for demo/unlabeled analysis."""
    filepath = os.path.join(RAW_DIR, "amazon_reviews_unlabeled.csv")
    if not os.path.exists(filepath):
        logging.info("Downloading unlabeled Amazon reviews from HuggingFace")
        try:
            # We'll use a small subset of the amazon polarity dataset or similar
            dataset = load_dataset('amazon_polarity', split='test[:5000]')
            df = dataset.to_pandas()
            # Just keep the text column
            df = df[['content']].rename(columns={'content': 'text'})
            df.to_csv(filepath, index=False)
            logging.info(f"Saved {len(df)} reviews to {filepath}")
        except Exception as e:
            logging.error(f"Failed to download unlabeled dataset: {e}")

def parse_xml_to_df(filepath, domain):
    """Parses SemEval XML into a list of dictionaries with sentences and aspects."""
    tree = ET.parse(filepath)
    root = tree.getroot()
    
    data = []
    for sentence in root.findall('sentence'):
        sent_id = sentence.get('id')
        text = sentence.find('text').text
        aspect_terms = []
        
        terms_elem = sentence.find('aspectTerms')
        if terms_elem is not None:
            for term in terms_elem.findall('aspectTerm'):
                term_dict = {
                    'term': term.get('term'),
                    'polarity': term.get('polarity'),
                    'from': int(term.get('from')),
                    'to': int(term.get('to'))
                }
                # Phase 2 rule: Drop or flag the "conflict" polarity
                if term_dict['polarity'] != 'conflict':
                    aspect_terms.append(term_dict)
                    
        data.append({
            'sent_id': sent_id,
            'text': text,
            'aspects': aspect_terms,
            'domain': domain
        })
    return pd.DataFrame(data)

def prepare_data():
    """Main function to prepare datasets for Phase 2."""
    download_semeval()
    download_unlabeled()
    
    # Parse and save SemEval data as easy-to-read JSON or CSV
    for domain in ['laptops', 'restaurants']:
        train_path = os.path.join(RAW_DIR, f"{domain}_train.xml")
        test_path = os.path.join(RAW_DIR, f"{domain}_test.xml")
        
        if os.path.exists(train_path):
            df_train = parse_xml_to_df(train_path, domain)
            df_train.to_json(os.path.join(PROCESSED_DIR, f"{domain}_train.json"), orient='records', lines=True)
            
        if os.path.exists(test_path):
            df_test = parse_xml_to_df(test_path, domain)
            df_test.to_json(os.path.join(PROCESSED_DIR, f"{domain}_test.json"), orient='records', lines=True)

if __name__ == "__main__":
    prepare_data()
