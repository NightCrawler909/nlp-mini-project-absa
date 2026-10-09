import os
import urllib.request
import spacy
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

nlp = spacy.load("en_core_web_sm")

LEXICON_DIR = os.path.join('data', 'lexicon')

def download_opinion_lexicon():
    os.makedirs(LEXICON_DIR, exist_ok=True)
    pos_url = 'https://raw.githubusercontent.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/master/data/opinion-lexicon-English/positive-words.txt'
    neg_url = 'https://raw.githubusercontent.com/jeffreybreen/twitter-sentiment-analysis-tutorial-201107/master/data/opinion-lexicon-English/negative-words.txt'
    
    pos_path = os.path.join(LEXICON_DIR, 'positive-words.txt')
    neg_path = os.path.join(LEXICON_DIR, 'negative-words.txt')
    
    def _download(url, path):
        if not os.path.exists(path):
            logging.info(f"Downloading {url}")
            urllib.request.urlretrieve(url, path)
            
    _download(pos_url, pos_path)
    _download(neg_url, neg_path)

def load_lexicon():
    download_opinion_lexicon()
    opinion_words = set()
    for filename in ['positive-words.txt', 'negative-words.txt']:
        path = os.path.join(LEXICON_DIR, filename)
        with open(path, 'r', encoding='latin-1') as f:
            for line in f:
                if not line.startswith(';') and line.strip():
                    opinion_words.add(line.strip())
    return opinion_words

OPINION_LEXICON = load_lexicon()

def extract_aspects_rule_based(text):
    """
    Extracts aspects using spaCy dependency parsing.
    Rule: Nouns/Noun phrases that are subjects or objects of opinion words.
    """
    doc = nlp(text)
    aspects = []
    
    for token in doc:
        # Check if the token is an opinion word
        if token.lemma_ in OPINION_LEXICON or token.text.lower() in OPINION_LEXICON:
            # Look for subjects or objects connected to this opinion word
            for child in token.children:
                if child.pos_ in ['NOUN', 'PROPN']:
                    if child.dep_ in ['nsubj', 'dobj', 'pobj', 'nsubjpass']:
                        # Simple expansion to noun chunk if possible
                        aspect_span = child.text
                        # Find the noun chunk containing this child
                        for chunk in doc.noun_chunks:
                            if child in chunk:
                                aspect_span = chunk.text
                                break
                        aspects.append(aspect_span)
            
            # If the opinion word is an adjective modifying a noun
            if token.dep_ == 'amod' and token.head.pos_ in ['NOUN', 'PROPN']:
                aspect_span = token.head.text
                for chunk in doc.noun_chunks:
                    if token.head in chunk:
                        # Exclude the opinion word itself if it's part of the chunk
                        # For simplicity, we just take the head noun
                        aspect_span = token.head.text
                        break
                aspects.append(aspect_span)
                
    # Remove duplicates
    return list(set(aspects))

if __name__ == "__main__":
    test_sentence = "The battery life is amazing but the screen is terrible."
    print("Test Sentence:", test_sentence)
    print("Extracted Aspects:", extract_aspects_rule_based(test_sentence))
