import os
import sys
import pytest
import joblib

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.preprocessing import preprocess_text, extract_suspicious_keywords

def test_model_and_vectorizer_exist():
    model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
    assert os.path.exists(model_path), "Champion model file does not exist"
    assert os.path.exists(vectorizer_path), "Vectorizer file does not exist"

def test_preprocessing():
    raw_text = "CONGRATULATIONS! You've won a $1000 prize!! Claim it now."
    processed = preprocess_text(raw_text)
    assert "congratulation" in processed or "prize" in processed
    assert "!" not in processed
    assert "$" not in processed

def test_keyword_extraction():
    raw_text = "Urgent: your account has been suspended. Call now to claim prize."
    triggers = extract_suspicious_keywords(raw_text)
    assert len(triggers) > 0
    assert any(w in triggers for w in ["urgent", "claim", "prize", "suspended", "call"])

def test_model_inference():
    model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
    vectorizer_path = os.path.join(BASE_DIR, "models", "vectorizer.pkl")
    model = joblib.load(model_path)
    vectorizer = joblib.load(vectorizer_path)
    
    spam_sample = "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 0800 now to claim."
    ham_sample = "Hey Alex, are we still meeting up for lunch today at 12 PM?"
    
    X_spam = vectorizer.transform([preprocess_text(spam_sample)])
    X_ham = vectorizer.transform([preprocess_text(ham_sample)])
    
    pred_spam = model.predict(X_spam)[0]
    pred_ham = model.predict(X_ham)[0]
    
    assert pred_spam == 1, "Model failed to classify obvious spam sample"
    assert pred_ham == 0, "Model incorrectly flagged legitimate ham message"
