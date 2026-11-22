import os
import sys
import pytest
from fastapi.testclient import TestClient

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app.main import app

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as test_client:
        yield test_client

def test_root_serves_html(client):
    res = client.get("/")
    assert res.status_code == 200
    assert "text/html" in res.headers.get("content-type", "")
    assert "SpamGuard AI" in res.text

def test_health_endpoint(client):
    res = client.get("/health")
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert data["vocabulary_size"] > 0
    assert "uptime_seconds" in data

def test_api_info_endpoint(client):
    res = client.get("/api/info")
    assert res.status_code == 200
    data = res.json()
    assert data["author"] == "Pavan T"
    assert "github_repo" in data
    assert "metrics" in data
    assert data["metrics"]["accuracy"] > 0.95

def test_api_samples_endpoint(client):
    res = client.get("/api/samples")
    assert res.status_code == 200
    samples = res.json()
    assert len(samples) >= 4
    categories = {s["category"] for s in samples}
    assert "spam" in categories
    assert "ham" in categories

def test_predict_spam(client):
    payload = {
        "text": "CONGRATULATIONS! You have won $1,000,000 cash in our international lottery! Call 0800 123 4567 to claim your prize now."
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] == "Spam"
    assert data["confidence"] >= 0.70
    assert data["spam_score"] >= 70
    assert data["risk_level"] in ["High Threat", "Suspicious", "Moderate"]
    assert len(data["trigger_keywords"]) > 0

def test_predict_ham(client):
    payload = {
        "text": "Hi team, please find the quarterly presentation attached for tomorrow's 10 AM project review meeting."
    }
    res = client.post("/predict", json=payload)
    assert res.status_code == 200
    data = res.json()
    assert data["prediction"] == "Ham"
    assert data["confidence"] > 0.60
    assert data["risk_level"] in ["Safe", "Low Risk"]

def test_predict_empty_string(client):
    res = client.post("/predict", json={"text": ""})
    assert res.status_code in [400, 422]

def test_predict_whitespace_only(client):
    res = client.post("/predict", json={"text": "   \n\t  "})
    assert res.status_code in [400, 422]
