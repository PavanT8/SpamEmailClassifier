import os
import re
import nltk

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
NLTK_DATA_DIR = os.path.join(BASE_DIR, "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

if NLTK_DATA_DIR not in nltk.data.path:
    nltk.data.path.insert(0, NLTK_DATA_DIR)

# Safely verify/download NLTK resources
try:
    from nltk.corpus import stopwords
    _ = stopwords.words('english')
except Exception:
    nltk.download('stopwords', download_dir=NLTK_DATA_DIR, quiet=True)

try:
    from nltk.stem import WordNetLemmatizer
    lemmatizer = WordNetLemmatizer()
    _ = lemmatizer.lemmatize("testing")
except Exception:
    nltk.download('wordnet', download_dir=NLTK_DATA_DIR, quiet=True)

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Common high-risk spam keywords and patterns
COMMON_SPAM_TRIGGERS = {
    "claim", "prize", "winner", "cash", "guaranteed", "urgent", "free", 
    "voucher", "congratulations", "win", "won", "reward", "lottery", 
    "call", "txt", "nokia", "ringtone", "rate", "valid", "entry", 
    "pobox", "ppm", "service", "expire", "suspended", "verify", 
    "password", "security", "unauthorized", "login", "bank", "bonus",
    "exclusive", "offer", "refund", "credit", "selected", "click", "gift"
}

def get_lemmatizer():
    return WordNetLemmatizer()

def get_stopwords():
    try:
        return set(stopwords.words('english'))
    except Exception:
        return set()

def preprocess_text(text: str) -> str:
    """
    Standardizes and cleans input text:
    1. Removes non-alphabetic characters
    2. Lowercases text
    3. Removes English stopwords
    4. Applies WordNet lemmatization
    """
    if not text or not isinstance(text, str):
        return ""
    
    lemmatizer = get_lemmatizer()
    stop_words = get_stopwords()
    
    # Remove non-alphabetic characters
    clean_text = re.sub(r'[^a-zA-Z]', ' ', text)
    clean_text = clean_text.lower()
    
    words = clean_text.split()
    processed_words = [
        lemmatizer.lemmatize(word) 
        for word in words 
        if word not in stop_words and len(word) > 1
    ]
    return ' '.join(processed_words)

def extract_suspicious_keywords(text: str, additional_spam_words: set = None) -> list:
    """
    Detects suspicious spam/phishing trigger keywords present in input.
    """
    if not text:
        return []
        
    lemmatizer = get_lemmatizer()
    clean_text = re.sub(r'[^a-zA-Z0-9$%]', ' ', text.lower())
    tokens = clean_text.split()
    
    suspicious_set = set(COMMON_SPAM_TRIGGERS)
    if additional_spam_words:
        suspicious_set.update(additional_spam_words)
        
    detected = []
    seen = set()
    for token in tokens:
        lemmatized_token = lemmatizer.lemmatize(token)
        if (token in suspicious_set or lemmatized_token in suspicious_set) and token not in seen:
            detected.append(token)
            seen.add(token)
            
    return detected[:10]
