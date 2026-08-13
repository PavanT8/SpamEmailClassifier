# Spam Email Classifier

This project is an end-to-end Machine Learning web application designed to classify text messages or emails as **Spam** or **Ham (Not Spam)**. It uses a Natural Language Processing (NLP) pipeline and a Multinomial Naive Bayes model trained on the SMS Spam Collection dataset.

## Table of Contents
- [Problem Statement](#problem-statement)
- [Dataset](#dataset)
- [Architecture](#architecture)
- [Installation](#installation)
- [Usage](#usage)
- [API Reference](#api-reference)
- [Limitations & Future Improvements](#limitations--future-improvements)

## Problem Statement
With the increasing volume of digital communication, spam messages have become a significant problem. They clutter inboxes, waste time, and can sometimes be malicious (phishing, scams). This project aims to automatically detect and filter out spam messages from legitimate ones using machine learning techniques.

## Dataset
The model is trained on the **SMS Spam Collection Data Set** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection). 
- It contains a set of SMS tagged messages that have been collected for SMS Spam research.
- **Format:** Tab-separated values with two columns (`label` and `text`).
- Labels are either `spam` or `ham`.

## Architecture
The project is structured into three main components:
1. **Data Pipeline & Training:** Scripts to download the data, parse it, preprocess the text (TF-IDF vectorization), and train a Machine Learning model.
2. **REST API:** A Flask-based backend server that loads the pre-trained model and exposes a POST endpoint for real-time inference.
3. **Web UI:** A clean, modern frontend built with HTML, CSS, and Vanilla JavaScript that allows users to interact with the model directly from their browser.

### Model Details
- **Preprocessing:** `TfidfVectorizer` (Term Frequency-Inverse Document Frequency) is used to convert raw text into a matrix of TF-IDF features. English stop words are removed.
- **Algorithm:** `MultinomialNB` (Multinomial Naive Bayes) is employed as the classifier, which is highly effective and standard for text classification tasks.
- **Evaluation:** The model is evaluated based on Accuracy, Precision, Recall, and F1-score during the training phase.

## Installation

### Prerequisites
- Python 3.8+
- Git

### Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/PavanT8/SpamEmailClassifier.git
   cd SpamEmailClassifier
   ```

2. **Create and activate a virtual environment (optional but recommended):**
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   ```

3. **Install the dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download data and train the model:**
   First, download the dataset:
   ```bash
   python src/download_data.py
   ```
   Then, train and save the model:
   ```bash
   python src/train.py
   ```
   *This will generate a `model.pkl` file in the `app/models/` directory.*

## Usage

1. **Start the Flask server:**
   ```bash
   python app/api.py
   ```
   The server will start on `http://localhost:5000` (or `http://0.0.0.0:5000`).

2. **Access the Web Interface:**
   Open your web browser and navigate to `http://localhost:5000`. You will see a professional user interface where you can paste email or SMS content and click "Classify Email" to get an immediate prediction (Spam or Ham).

## API Reference

The Flask application provides a REST API that you can call directly.

### POST `/predict`
Classifies a given text message.

**Request Header:**
`Content-Type: application/json`

**Request Body:**
```json
{
  "text": "Congratulations! You've won a $1,000 Walmart gift card. Click here to claim your prize."
}
```

**Response (200 OK):**
```json
{
  "prediction": "spam"
}
```
*(If the text is not spam, the prediction will be `"ham"`)*

## Limitations & Future Improvements

**Current Limitations:**
- **Short Texts Bias:** The model is trained on the SMS Spam Collection, which consists mainly of short text messages. It might not generalize perfectly to long, complex email threads.
- **Basic Preprocessing:** We are using basic TF-IDF without advanced lemmatization, stemming, or handling of specific entities (like URLs or phone numbers).

**Future Improvements:**
- **Advanced Models:** Experiment with Deep Learning models (like LSTMs or Transformers like BERT) for better contextual understanding.
- **Expanded Dataset:** Combine the SMS dataset with larger Email Spam datasets (e.g., Enron dataset) to improve email-specific classification.
- **Feature Engineering:** Add custom features like "contains URL", "length of message", or "contains special characters".
- **Dockerization:** Add a `Dockerfile` and `docker-compose.yml` to containerize the application for easier deployment.
