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
import joblib  # Import joblib for saving and loading models
import re
import os  # Import os to check if files exist
from nltk.corpus import stopwords # Import stopwords

# --- NLTK Data Check ---
print("Checking for NLTK data (wordnet, omw-1.4, stopwords)...")
try:
    WordNetLemmatizer().lemmatize('running')
    nltk.data.find('corpora/stopwords') # Check for stopwords too
    print("NLTK data already present.")
except LookupError:
    print("NLTK data not found, downloading...")
    nltk.download('wordnet')
    nltk.download('omw-1.4')
    nltk.download('stopwords') # Ensure stopwords are downloaded
    print("NLTK data download complete.")

# --- Model Training and Saving ---
MODEL_DIR = 'model_artifacts'
os.makedirs(MODEL_DIR, exist_ok=True)  # Create directory if it doesn't exist
print(f"Model artifacts will be stored in: {MODEL_DIR}")

MODEL_PATH = os.path.join(MODEL_DIR, 'best_knn_model.joblib')
VECTORIZER_PATH = os.path.join(MODEL_DIR, 'tfidf_vectorizer.joblib')
LABEL_ENCODER_PATH = os.path.join(MODEL_DIR, 'label_encoder.joblib')
TRAINING_DATA_PATH = os.path.join(MODEL_DIR, 'training_data.joblib')  # To save X_train and y_train for kneighbors

# Initialize model and transformers to None
best_knn = None
vectorizer = None
label_encoder = None
X_train = np.array([])  # Initialize X_train and y_train as empty arrays
y_train = np.array([])

# --- Global NLTK components for efficiency ---
lemmatizer = WordNetLemmatizer()
custom_stop_words = set(stopwords.words('english')) # Populate this set properly


def preprocess_text(text):
    # Ensure text is string and not empty after strip
    if not isinstance(text, str) or not text.strip():
        return ""
    text = text.lower()
    tokens = text.split()
    # Filter out stopwords and lemmatize
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in custom_stop_words and word.strip()]
    return " ".join(tokens)

# Attempt to load model artifacts first
print("\nAttempting to load pre-trained model and transformers...")
if os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH) and \
        os.path.exists(LABEL_ENCODER_PATH) and os.path.exists(TRAINING_DATA_PATH):
    try:
        best_knn = joblib.load(MODEL_PATH)
        vectorizer = joblib.load(VECTORIZER_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        X_train, y_train = joblib.load(TRAINING_DATA_PATH)
        print("Model and transformers loaded successfully.")

        # Re-create X_test and y_test for evaluation, but don't need to reload full JSON if X_train/y_train are loaded
        # Need to reconstruct X_tfidf from training data to get X_test from split
        # This part ensures X_test and y_test are available for initial classification_report/confusion_matrix
        # You'll need to load the full JSON to correctly reproduce the original X and y for splitting
        with open('synthetic_synopsis.json', 'r') as file:
            data_full = json.load(file)
        symptoms_full = []
        diagnoses_full = []
        if isinstance(data_full, dict):
            for syndrome, details in data_full.items():
                if isinstance(details, dict):
                    diagnosis = syndrome
                    for synopsis in details.get('synopsis', []):
                        if isinstance(synopsis, list):
                            processed_synopsis = preprocess_text(" ".join(synopsis))
                            if processed_synopsis:
                                symptoms_full.append(processed_synopsis)
                                diagnoses_full.append(diagnosis)
        df_full = pd.DataFrame({'symptoms': symptoms_full, 'diagnosis': diagnoses_full})
        X_full = df_full['symptoms']
        y_full = df_full['diagnosis']
        y_full_encoded = label_encoder.transform(y_full) # Use loaded encoder
        X_full_tfidf = vectorizer.transform(X_full).toarray() # Use loaded vectorizer

        _, X_test, _, y_test = train_test_split(X_full_tfidf, y_full_encoded, test_size=0.2, random_state=42,
                                                stratify=y_full_encoded)

    except Exception as e:
        print(f"Error loading model artifacts: {e}. Proceeding with new model training.")
        # If loading fails, best_knn remains None, which will trigger training
else:
    print("No pre-trained model found or incomplete artifacts. Training a new model...")
    # Load the JSON data from the file
    json_file_path = 'synthetic_synopsis.json'
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    # Prepare the dataset
    symptoms = []
    diagnoses = []

    if isinstance(data, dict):
        print("Processing JSON data to extract symptoms and diagnoses...")
        for syndrome, details in data.items():
            if isinstance(details, dict):
                diagnosis = syndrome
                for synopsis in details.get('synopsis', []):
                    if isinstance(synopsis, list):
                        clinical_synopsis = " ".join(synopsis)
                        processed_synopsis = preprocess_text(clinical_synopsis)
                        if processed_synopsis:
                            symptoms.append(processed_synopsis)
                            diagnoses.append(diagnosis)
        print(f"Extracted {len(symptoms)} symptom-diagnosis pairs.")
    else:
        print("The data is not in the expected format. Please check 'synthetic_synopsis.json'. Exiting.")
        exit()  # Exit if data format is incorrect

    # Create a DataFrame
    df = pd.DataFrame({'symptoms': symptoms, 'diagnosis': diagnoses})
    print(f"Created DataFrame with {len(df)} entries.")

    # Split the data into features and labels
    X = df['symptoms']
    y = df['diagnosis']

    # Encode the labels
    print("Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"Labels encoded for {len(label_encoder.classes_)} unique diagnoses.")

    # Convert symptoms to TF-IDF features
    print("Fitting TF-IDF vectorizer...")
    vectorizer = TfidfVectorizer(max_features=5000, stop_words=list(custom_stop_words), ngram_range=(1, 2), min_df=5)
    X_tfidf = vectorizer.fit_transform(X).toarray()
    print(f"TF-IDF vectorizer fitted. X_tfidf shape: {X_tfidf.shape}")

    # Split the data into training and testing sets
    print("Splitting data into training and testing sets...")
    X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y_encoded, test_size=0.2, random_state=42,
                                                        stratify=y_encoded)
    print(f'Number of training samples: {X_train.shape[0]}')
    print(f'Number of testing samples: {X_test.shape[0]}')

    # Determine potential k values
    num_samples = X_train.shape[0]
    # Ensure k_values are at least 1 and less than or equal to num_samples
    # Reduced the number of fractions for faster tuning
    k_values = [max(1, int(num_samples * fraction)) for fraction in [0.05, 0.1, 0.15]]
    k_values = [k for k in k_values if k <= num_samples]  # Filter out k values larger than num_samples
    k_values = sorted(list(set(k_values)))  # Remove duplicates and sort
    if not k_values: # Ensure k_values is not empty, add a default if necessary
        k_values = [min(5, num_samples)] if num_samples > 0 else [1] # Fallback to a small k

    # Initialize the KNN classifier with GridSearchCV for hyperparameter tuning
    param_grid = {'n_neighbors': k_values, 'metric': ['cosine'], # Reduced metrics for faster tuning
                  'weights': ['distance']} # Reduced weights for faster tuning
    print(f"\nStarting GridSearchCV for KNN. This may take a while...")
    print(f"Parameters to tune: {param_grid}")
    skf = StratifiedKFold(n_splits=3) # Reduced folds for faster tuning
    grid_search = GridSearchCV(KNeighborsClassifier(), param_grid, cv=skf, scoring='neg_mean_squared_error', n_jobs=-1,
                               verbose=1)
    grid_search.fit(X_train, y_train)

    # Get the best KNN model
    best_knn = grid_search.best_estimator_
    print(f'\nGridSearchCV complete. Best KNN parameters: {grid_search.best_params_}')

    # Save the trained model and transformers
    joblib.dump(best_knn, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    joblib.dump(label_encoder, LABEL_ENCODER_PATH)
    joblib.dump((X_train, y_train), TRAINING_DATA_PATH)  # Save training data for k-neighbors in prediction
    print("Model, vectorizer, and label encoder saved successfully.")

# --- Model Evaluation (always run after training or loading) ---
# Make predictions
print("\n--- Evaluating the model on the test set ---")
print("Making predictions on the test set...")
y_pred = best_knn.predict(X_test)
print("Predictions made.")

# Evaluate the model
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))


# Function to calculate Mean Reciprocal Rank (MRR)
def mean_reciprocal_rank(y_true, y_pred_proba_indices, top_n=20):
    mrr = 0.0
    print(f"Calculating Mean Reciprocal Rank (MRR) for top {top_n}...")
    # Ensure y_true is a numpy array for consistent indexing
    y_true = np.asarray(y_true)
    for i, true_label_encoded in enumerate(y_true):
        # pred_indices are actual indices from X_train/y_train
        neighbor_indices = y_pred_proba_indices[i]

        # Use the actual true label, not its encoded value for direct comparison with original labels
        true_original_label = label_encoder.inverse_transform([true_label_encoded])[0]

        found = False
        for rank, neighbor_idx in enumerate(neighbor_indices):
            # Check if neighbor_idx is valid for y_train
            if neighbor_idx < len(y_train):
                neighbor_label_encoded = y_train[neighbor_idx]
                neighbor_original_label = label_encoder.inverse_transform([neighbor_label_encoded])[0]
                if neighbor_original_label == true_original_label:
                    mrr += 1.0 / (rank + 1)
                    found = True
                    break # Found the true label, move to next test sample
            else:
                print(f"Warning: Neighbor index {neighbor_idx} out of bounds for y_train (len={len(y_train)}). Skipping this neighbor.")
        # If true label not found among top_n, it contributes 0 to MRR for this sample.
    print("MRR calculation complete.")
    return mrr / len(y_true) if len(y_true) > 0 else 0.0

# Get the top k predictions for each test sample
# Ensure n_neighbors does not exceed the number of training samples for MRR calculation
n_neighbors_for_mrr = min(20, X_train.shape[0])
# We need to get the actual neighbors for MRR, not just a predict_proba
# best_knn.kneighbors returns distances and indices of neighbors in the training set
_, y_pred_neighbor_indices = best_knn.kneighbors(X_test, n_neighbors=n_neighbors_for_mrr)

# Calculate MRR
mrr = mean_reciprocal_rank(y_test, y_pred_neighbor_indices, top_n=n_neighbors_for_mrr)
print(f'Mean Reciprocal Rank (Top {n_neighbors_for_mrr}): {mrr:.4f}')


# Function to predict top N diagnoses
def predict_top_diagnoses(symptom, top_n=20): # Set a default top_n for convenience
    processed_symptom = preprocess_text(symptom)

    if not processed_symptom.strip():
        return []

    # Ensure vectorizer and best_knn are not None before using
    if vectorizer is None or best_knn is None:
        print("Error: Model or vectorizer not loaded/trained before prediction attempt.")
        return []

    symptom_tfidf = vectorizer.transform([processed_symptom]).toarray()

    # Ensure n_neighbors for kneighbors doesn't exceed the number of training samples
    # Use max(1, n_neighbors) to avoid n_neighbors=0 if top_n is 0 or less
    n_neighbors_for_prediction = min(max(1, top_n * 2), X_train.shape[0]) # Fetch more neighbors to get diverse top_n

    distances, indices = best_knn.kneighbors(symptom_tfidf, n_neighbors=n_neighbors_for_prediction)

    # Get the predicted diagnoses based on the indices from the training data
    # Filter out any indices that might be out of bounds for y_train
    valid_indices = [idx for idx in indices.flatten() if idx < len(y_train)]
    if not valid_indices:
        return []  # Return empty list if no valid neighbors found

    predicted_diagnoses_encoded_train = y_train[valid_indices]
    predicted_diagnoses_train = label_encoder.inverse_transform(predicted_diagnoses_encoded_train)

    # Count occurrences of each diagnosis and get the top N
    unique, counts = np.unique(predicted_diagnoses_train, return_counts=True)
    diagnosis_counts = dict(zip(unique, counts))

    # Sort diagnoses by count (or by distance if counts are equal) and get the top N
    # For KNN, 'distance' weight means closer neighbors have more influence.
    # If weights='distance' was used, the model effectively already weighted by distance.
    # When generating top N, you want the most frequent of the *weighted* neighbors.
    # Since we fetched a fixed number of neighbors, simply counting unique diagnoses from these neighbors
    # and sorting by count (frequency) is a reasonable approach for presenting 'top diagnoses'.
    # If a very fine-grained ranking by distance is needed, it would involve re-sorting
    # based on the original distances *within each unique diagnosis group*, which is more complex.
    # For now, sorting by count is robust.

    # Collect unique diagnoses with their cumulative distance (or a measure of proximity)
    # This part is a bit tricky with KNN. A simple count might not reflect "top N" perfectly if 'distance' weights are used.
    # If we just count, 'uniform' weights are implicitly assumed for the final top-N selection.
    # A more rigorous approach for 'distance' weighting would be to sum the inverse distances for each diagnosis,
    # but that complicates the simple top-N list.
    # For clarity and common practice in presenting KNN results, we will stick to frequency among top neighbors.
    # If 'weights' in KNN was 'distance', the `best_knn.predict_proba` (if available for KNN)
    # would already give weighted probabilities. However, KNeighborsClassifier.predict_proba()
    # is only for classification based on class majorities in neighbors, not for ranking all possible classes.
    # The current approach of getting `kneighbors` and then counting diagnoses from those neighbors is standard for
    # getting a list of "possible diagnoses."

    # To improve the ranking when `weights='distance'` was used in the model,
    # we can consider the distances when multiple diagnoses have the same count.
    # Create a list of (diagnosis, distance) pairs
    diagnosis_distance_pairs = []
    for i, idx in enumerate(indices.flatten()):
        if idx < len(y_train): # Check if index is valid
            diagnosis = label_encoder.inverse_transform([y_train[idx]])[0]
            distance = distances.flatten()[i]
            diagnosis_distance_pairs.append((diagnosis, distance))

    # Group by diagnosis and calculate average/min distance for each, or just count
    # For simplicity, let's stick to counting for now, but note that this doesn't fully leverage 'distance' weights in sorting.
    # If true distance-weighted ranking is critical, consider other algorithms or a custom scoring method.
    from collections import defaultdict
    diagnosis_scores = defaultdict(float) # Using inverse distance as a 'score'
    for diagnosis, dist in diagnosis_distance_pairs:
        # Avoid division by zero for exact matches (dist=0). Handle small distances by adding a small epsilon
        score = 1.0 / (dist + 1e-6) # Use inverse distance as a proxy for 'closeness' or 'vote strength'
        diagnosis_scores[diagnosis] += score # Summing scores from all occurrences

    # Sort by the score (descending)
    sorted_diagnoses = sorted(diagnosis_scores.items(), key=lambda item: item[1], reverse=True)

    # Return only the top_n diagnosis names
    return [diagnosis for diagnosis, score in sorted_diagnoses[:top_n]]


# --- Main execution block for interactive mode ---
if __name__ == "__main__":
    print("\n--- Interactive Diagnosis Prediction ---")
    print("Type 'exit' to quit.")

    while True:
        user_input = input("\nPlease provide the symptoms the patient has (e.g., 'seizures, developmental delay, heterotpia on MRI'):\n")
        if user_input.lower() == 'exit':
            break

        if not user_input.strip():
            print("No symptoms provided. Please enter some symptoms or type 'exit' to quit.")
            continue

        top_20_diagnoses = predict_top_diagnoses(user_input, top_n=20)

        if top_20_diagnoses:
            print("\nThe most likely 20 genetic neurological conditions based on the symptoms that the patient has are as follows:")
            for i, diag in enumerate(top_20_diagnoses):
                print(f"  {i + 1}. {diag}")
        else:
            print("Could not determine top diagnoses based on the provided symptoms. Please try with more detailed symptoms.")

    print("\nThank you for using the diagnosis prediction tool!")
