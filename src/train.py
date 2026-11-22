import os
import sys
import json
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.data_loader import load_data
from src.preprocessing import preprocess_text

MODELS_DIR = os.path.join(BASE_DIR, "models")
APP_MODELS_DIR = os.path.join(BASE_DIR, "app", "models")
METRICS_FILE = os.path.join(MODELS_DIR, "metrics.json")

def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)
    acc = float(accuracy_score(y_test, y_pred))
    prec = float(precision_score(y_test, y_pred, zero_division=0))
    rec = float(recall_score(y_test, y_pred, zero_division=0))
    f1 = float(f1_score(y_test, y_pred, zero_division=0))
    cm = confusion_matrix(y_test, y_pred).tolist()
    
    print(f"\n==========================================")
    print(f"📊 Model: {name}")
    print(f"==========================================")
    print(f"  Accuracy  : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  Precision : {prec:.4f} ({prec*100:.2f}%)")
    print(f"  Recall    : {rec:.4f} ({rec*100:.2f}%)")
    print(f"  F1 Score  : {f1:.4f} ({f1*100:.2f}%)")
    print(f"  Confusion Matrix:\n    {cm}")
    
    return {
        "name": name,
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1_score": round(f1, 4),
        "confusion_matrix": cm
    }

def train():
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(APP_MODELS_DIR, exist_ok=True)
    
    print("Loading spam classification dataset...")
    df = load_data()
    print(f"Total dataset records: {len(df)}")
    
    print("Applying text preprocessing (lowercasing, punctuation removal, lemmatization)...")
    df['processed_text'] = df['text'].apply(preprocess_text)
    
    X = df['processed_text']
    y = df['label']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    print("\nFitting TF-IDF Vectorizer (max_features=5000, ngram_range=(1,2))...")
    vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Candidate 1: Multinomial Naive Bayes
    nb_clf = MultinomialNB(alpha=1.0)
    nb_clf.fit(X_train_tfidf, y_train)
    nb_metrics = evaluate_model("Multinomial Naive Bayes (MultinomialNB)", nb_clf, X_test_tfidf, y_test)
    
    # Candidate 2: Logistic Regression
    lr_clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42)
    lr_clf.fit(X_train_tfidf, y_train)
    lr_metrics = evaluate_model("Logistic Regression (LogisticRegression)", lr_clf, X_test_tfidf, y_test)
    
    # Compare models: Prioritize F1 score & Precision (avoiding false alarms on legitimate emails)
    candidates = [
        {"name": "MultinomialNB", "model": nb_clf, "metrics": nb_metrics},
        {"name": "LogisticRegression", "model": lr_clf, "metrics": lr_metrics}
    ]
    
    best_candidate = max(candidates, key=lambda c: (c["metrics"]["f1_score"] + c["metrics"]["precision"]))
    best_model = best_candidate["model"]
    best_name = best_candidate["name"]
    
    print(f"\n🏆 Champion Model Selected: {best_name}")
    print(f"   Accuracy: {best_candidate['metrics']['accuracy']*100:.2f}% | F1: {best_candidate['metrics']['f1_score']*100:.2f}%")
    
    # Save artifacts
    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    classifier_path = os.path.join(MODELS_DIR, "classifier.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, "vectorizer.pkl")
    legacy_app_model_path = os.path.join(APP_MODELS_DIR, "model.pkl")
    
    joblib.dump(best_model, best_model_path)
    joblib.dump(best_model, classifier_path)
    joblib.dump(vectorizer, vectorizer_path)
    
    # Also save a self-contained Pipeline for backward compatibility in app/models/model.pkl
    from sklearn.pipeline import Pipeline
    full_pipeline = Pipeline([
        ('vectorizer', vectorizer),
        ('classifier', best_model)
    ])
    joblib.dump(full_pipeline, legacy_app_model_path)
    
    metrics_summary = {
        "champion_model": best_name,
        "models_compared": {
            "MultinomialNB": nb_metrics,
            "LogisticRegression": lr_metrics
        },
        "selected_metrics": best_candidate["metrics"],
        "dataset_size": len(df),
        "features_count": len(vectorizer.get_feature_names_out())
    }
    
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics_summary, f, indent=2)
        
    print(f"Model artifacts successfully serialized to {MODELS_DIR}/")
    print(f"Metrics saved to {METRICS_FILE}")
    return metrics_summary

def ensure_model_artifacts():
    """Fallback utility: ensures models exist, otherwise trains them automatically."""
    best_model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    vectorizer_path = os.path.join(MODELS_DIR, "vectorizer.pkl")
    
    if not (os.path.exists(best_model_path) and os.path.exists(vectorizer_path)):
        print("Model artifacts missing. Running automated training fallback...")
        return train()
    return None

if __name__ == "__main__":
    train()
