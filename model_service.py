import pandas as pd
import json
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import joblib
import re
import os
from nltk.corpus import stopwords
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# --- NLTK Data Check ---
print("Checking for NLTK data (wordnet, omw-1.4, stopwords)...")
try:
    WordNetLemmatizer().lemmatize('running')
    nltk.data.find('corpora/stopwords')
    print("NLTK data already present.")
except LookupError:
    print("NLTK data not found, downloading...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords')
    print("NLTK data download complete.")

# --- Model Training and Saving ---
MODEL_DIR = 'model_artifacts'
os.makedirs(MODEL_DIR, exist_ok=True)
print(f"Model artifacts will be stored in: {MODEL_DIR}")

MODEL_PATH = os.path.join(MODEL_DIR, 'best_knn_model.joblib')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')
TRAINING_DATA_PATH = os.path.join(MODEL_DIR, 'training_data.joblib')

# Initialize model and transformers
best_knn = None
vectorizer = None
label_encoder = None
X_train = np.array([])
y_train = np.array([])

# --- Global NLTK components ---
lemmatizer = WordNetLemmatizer()
custom_stop_words = set(stopwords.words('english'))

def preprocess_text(text):
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    tokens = text.split()
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in custom_stop_words and word.strip()]
    return " ".join(tokens)

def predict_top_diagnoses(symptom, top_n=20):
    processed_symptom = preprocess_text(symptom)

    if not processed_symptom.strip():
        return []

    if vectorizer is None or best_knn is None:
        print("Error: Model or vectorizer not loaded/trained before prediction attempt.")
        return []

    symptom_tfidf = vectorizer.transform([processed_symptom]).toarray()
    n_neighbors_for_prediction = min(max(1, top_n * 2), X_train.shape[0])

    distances, indices = best_knn.kneighbors(symptom_tfidf, n_neighbors=n_neighbors_for_prediction)

    diagnosis_distance_pairs = []
    for i, idx in enumerate(indices.flatten()):
        if idx < len(y_train):
            diagnosis = label_encoder.inverse_transform([y_train[idx]])[0]
            distance = distances.flatten()[i]
            diagnosis_distance_pairs.append((diagnosis, distance))

    from collections import defaultdict
    diagnosis_scores = defaultdict(float)
    for diagnosis, dist in diagnosis_distance_pairs:
        score = 1.0 / (dist + 1e-6)
        diagnosis_scores[diagnosis] += score

    sorted_diagnoses = sorted(diagnosis_scores.items(), key=lambda item: item[1], reverse=True)
    return [{"disorder": diagnosis, "probability": round(score * 100, 2)} for diagnosis, score in sorted_diagnoses[:top_n]]

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        symptoms = data.get('symptoms', [])
        
        if not symptoms:
            return jsonify({"error": "No symptoms provided"}), 400

        # Join symptoms into a single string
        symptom_text = " ".join(symptoms)
        
        # Get predictions
        predictions = predict_top_diagnoses(symptom_text)
        
        return jsonify({
            "predictions": predictions,
            "input_symptoms": symptoms
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Load or train the model
    if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH) and \
            os.path.exists(LABEL_ENCODER_PATH) and os.path.exists(TRAINING_DATA_PATH):
        try:
            best_knn = joblib.load(MODEL_PATH)
            vectorizer = joblib.load(VECTORIZER_PATH)
            label_encoder = joblib.load(LABEL_ENCODER_PATH)
            X_train, y_train = joblib.load(TRAINING_DATA_PATH)
            print("Model and transformers loaded successfully.")
        except Exception as e:
            print(f"Error loading model artifacts: {e}")
            exit(1)
    else:
        print("Model artifacts not found. Please train the model first.")
        exit(1)

    # Start the Flask server
    app.run(host='0.0.0.0', port=5000) 