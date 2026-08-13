import os
import sys
import pytest
import json
import joblib

# Add the parent directory to the path so we can import from app
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.join(current_dir, "..")
sys.path.insert(0, parent_dir)

from app.api import app

@pytest.fixture
def client():
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_model_exists():
    model_path = os.path.join(parent_dir, "app", "models", "model.pkl")
    assert os.path.exists(model_path), "Model file does not exist"

def test_model_prediction():
    model_path = os.path.join(parent_dir, "app", "models", "model.pkl")
    model = joblib.load(model_path)
    
    ham_text = "Hey, how are you doing today?"
    spam_text = "WINNER!! As a valued network customer you have been selected to receivea £900 prize reward!"
    
    # Predict
    ham_pred = model.predict([ham_text])[0]
    spam_pred = model.predict([spam_text])[0]
    
    assert ham_pred == 0, "Model incorrectly classified ham"
    assert spam_pred == 1, "Model incorrectly classified spam"

def test_api_predict_ham(client):
    response = client.post('/predict', data=json.dumps({'text': 'Hey, are we still meeting today?'}), content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert data['prediction'] == 'ham'

def test_api_predict_spam(client):
    response = client.post('/predict', data=json.dumps({'text': 'URGENT! You have won a 1 week FREE membership in our £100,000 Prize Jackpot!'}), content_type='application/json')
    assert response.status_code == 200
    data = json.loads(response.data)
    assert 'prediction' in data
    assert data['prediction'] == 'spam'

def test_api_predict_missing_text(client):
    response = client.post('/predict', data=json.dumps({'not_text': 'Hello'}), content_type='application/json')
    assert response.status_code == 400
    data = json.loads(response.data)
    assert 'error' in data
