from flask import Flask, request, jsonify, render_template
import joblib
import os

app = Flask(__name__)

# Load the model
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, "models", "model.pkl")

model = None
if os.path.exists(model_path):
    model = joblib.load(model_path)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500
    
    data = request.get_json(force=True)
    if not data or 'text' not in data:
        return jsonify({"error": "Missing 'text' field in JSON request"}), 400
        
    text = data['text']
    
    try:
        # Predict expects an iterable of strings
        pred = model.predict([text])[0]
        label = "spam" if pred == 1 else "ham"
        return jsonify({"prediction": label})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=5000)
