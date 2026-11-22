import os
import sys
import time
from contextlib import asynccontextmanager
from typing import List, Dict, Any

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

from app.schemas import PredictRequest, PredictResponse, HealthResponse, SampleEmail, EmailStats
from src.preprocessing import preprocess_text, extract_suspicious_keywords
from src.train import ensure_model_artifacts

# App state
app_state = {
    "model": None,
    "vectorizer": None,
    "top_spam_words": set(),
    "model_name": "LogisticRegression",
    "start_time": time.time()
}

def load_ml_components():
    """Load or automatically train and load ML models."""
    if app_state["model"] is not None and app_state["vectorizer"] is not None:
        return
        
    models_dir = os.path.join(BASE_DIR, "models")
    model_path = os.path.join(models_dir, "best_model.pkl")
    vectorizer_path = os.path.join(models_dir, "vectorizer.pkl")
    
    # Auto-train fallback if files don't exist
    if not (os.path.exists(model_path) and os.path.exists(vectorizer_path)):
        print("Model artifacts not found on startup. Triggering automatic fallback training...")
        ensure_model_artifacts()
        
    if os.path.exists(model_path) and os.path.exists(vectorizer_path):
        try:
            model = joblib.load(model_path)
            vectorizer = joblib.load(vectorizer_path)
            app_state["model"] = model
            app_state["vectorizer"] = vectorizer
            app_state["model_name"] = model.__class__.__name__
            
            # Extract top spam vocabulary weights
            feature_names = vectorizer.get_feature_names_out()
            if hasattr(model, "coef_"):
                # Logistic Regression or Linear models
                weights = model.coef_[0]
                top_idx = np.argsort(weights)[-100:]
                app_state["top_spam_words"] = {feature_names[i] for i in top_idx}
            elif hasattr(model, "feature_log_prob_"):
                # Naive Bayes models
                diff = model.feature_log_prob_[1] - model.feature_log_prob_[0]
                top_idx = np.argsort(diff)[-100:]
                app_state["top_spam_words"] = {feature_names[i] for i in top_idx}
                
            print(f"Loaded {app_state['model_name']} with {len(feature_names)} features successfully.")
        except Exception as e:
            print(f"Error loading model artifacts: {e}")

@asynccontextmanager
async def lifespan(app: FastAPI):
    app_state["start_time"] = time.time()
    load_ml_components()
    yield
    app_state["model"] = None
    app_state["vectorizer"] = None
    app_state["top_spam_words"].clear()

app = FastAPI(
    title="Spam Email Classifier API",
    description="Production-ready FastAPI Machine Learning service for Spam & Phishing detection.",
    version="2.0.0",
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Static files mounting
static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
templates_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")
os.makedirs(static_dir, exist_ok=True)
os.makedirs(templates_dir, exist_ok=True)

app.mount("/static", StaticFiles(directory=static_dir), name="static")

@app.get("/", summary="Web Interface")
def serve_index():
    index_file = os.path.join(templates_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    fallback_static = os.path.join(static_dir, "index.html")
    if os.path.exists(fallback_static):
        return FileResponse(fallback_static)
    return {"message": "Spam Email Classifier API is running. UI templates at app/templates/index.html"}

@app.get("/health", response_model=HealthResponse, summary="Health Check")
def health_check():
    load_ml_components()
    model = app_state.get("model")
    vectorizer = app_state.get("vectorizer")
    is_loaded = (model is not None and vectorizer is not None)
    
    return HealthResponse(
        status="healthy" if is_loaded else "degraded",
        model_loaded=is_loaded,
        model_name=app_state.get("model_name", "None"),
        vocabulary_size=len(vectorizer.get_feature_names_out()) if vectorizer else 0,
        uptime_seconds=int(time.time() - app_state["start_time"])
    )

@app.get("/api/info", summary="Model Metadata & Links")
def get_info():
    load_ml_components()
    metrics_path = os.path.join(BASE_DIR, "models", "metrics.json")
    metrics_data = {}
    if os.path.exists(metrics_path):
        import json
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics_data = json.load(f)
            
    return {
        "project": "Spam & Phishing Email Classifier",
        "version": "2.0.0",
        "author": "Pavan T",
        "github_repo": "https://github.com/PavanT8/SpamEmailClassifier",
        "author_profile": "https://github.com/PavanT8",
        "champion_model": app_state.get("model_name", "LogisticRegression"),
        "metrics": metrics_data.get("selected_metrics", {
            "accuracy": 0.9731,
            "precision": 1.0,
            "recall": 0.7987,
            "f1_score": 0.8881
        }),
        "models_compared": metrics_data.get("models_compared", {})
    }

@app.get("/api/samples", response_model=List[SampleEmail], summary="Curated Test Email Presets")
def get_sample_presets():
    return [
        SampleEmail(
            id="spam_lottery",
            category="spam",
            title="🎁 $1,000,000 Prize Winner",
            content="CONGRATULATIONS! You have been selected as the grand prize winner of $1,000,000 cash in our international sweepstakes! Call 0800 123 4567 immediately or reply CLAIM to collect your prize. Guaranteed payout, terms apply. Valid for 24 hours only!"
        ),
        SampleEmail(
            id="spam_bank",
            category="spam",
            title="🚨 Suspended Banking Account",
            content="Urgent Security Notice: Your online banking account access has been temporarily suspended due to multiple unauthorized login attempts. Click here immediately to verify your identity and restore your credentials: http://secure-bank-login-update.com/verify. Failure to do so within 12 hours will result in permanent account deactivation."
        ),
        SampleEmail(
            id="ham_meeting",
            category="ham",
            title="💼 Team Project Sync",
            content="Hi team, please find attached the updated slide deck for tomorrow's Q3 performance review meeting scheduled at 10:00 AM in Conference Room B. Please review the slides beforehand and bring any feedback."
        ),
        SampleEmail(
            id="ham_dinner",
            category="ham",
            title="☕ Coffee & Casual Catchup",
            content="Hey Sarah, are we still meeting up for coffee this afternoon? Let me know what time works best for you. I should finish my lectures around 3:30 PM. See you soon!"
        )
    ]

@app.post("/predict", response_model=PredictResponse, summary="Classify Email Text")
def predict_email(req: PredictRequest):
    load_ml_components()
    model = app_state.get("model")
    vectorizer = app_state.get("vectorizer")
    
    if not model or not vectorizer:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Machine learning model is not loaded."
        )
        
    raw_text = req.text.strip()
    if not raw_text:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Email content cannot be empty or only whitespace."
        )
        
    char_count = len(raw_text)
    word_count = len(raw_text.split())
    
    # Preprocess text
    processed = preprocess_text(raw_text)
    tokens_count = len(processed.split()) if processed else 0
    
    # Vectorize
    text_to_transform = processed if processed else raw_text
    X = vectorizer.transform([text_to_transform])
    
    # Predict
    raw_pred = model.predict(X)[0]
    
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        prob_ham = float(probs[0])
        prob_spam = float(probs[1])
    else:
        # Fallback for models without predict_proba
        decision = model.decision_function(X)[0]
        prob_spam = float(1 / (1 + np.exp(-decision)))
        prob_ham = float(1 - prob_spam)
        
    is_spam = (raw_pred == 1) or (prob_spam >= 0.5)
    pred_label = "Spam" if is_spam else "Ham"
    confidence = prob_spam if is_spam else prob_ham
    spam_score = int(round(prob_spam * 100))
    
    # Risk level classification
    if is_spam:
        if prob_spam >= 0.85:
            risk_level = "High Threat"
        elif prob_spam >= 0.65:
            risk_level = "Suspicious"
        else:
            risk_level = "Moderate"
    else:
        if prob_spam < 0.15:
            risk_level = "Safe"
        else:
            risk_level = "Low Risk"
            
    # Extract suspicious keywords
    detected = extract_suspicious_keywords(raw_text, app_state.get("top_spam_words"))
    
    return PredictResponse(
        prediction=pred_label,
        confidence=round(confidence, 4),
        spam_score=spam_score,
        risk_level=risk_level,
        trigger_keywords=detected,
        probabilities={
            "spam": round(prob_spam, 4),
            "ham": round(prob_ham, 4)
        },
        stats=EmailStats(
            char_count=char_count,
            word_count=word_count,
            tokens_analyzed=tokens_count
        )
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=5000, reload=True)
