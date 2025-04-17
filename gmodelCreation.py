import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Define the dataset directory
DATASET_DIR = 'handsign_dataset'
MODEL_FILENAME = 'handsign_model.pkl'

def load_and_preprocess_data(dataset_dir):
    """Loads data from CSV files, preprocesses it, and combines it."""
    all_data = []
    for filename in os.listdir(dataset_dir):
        if filename.endswith('.csv'):
            filepath = os.path.join(dataset_dir, filename)
            df = pd.read_csv(filepath)
            all_data.append(df)

    if not all_data:
        print("No CSV files found in the dataset directory.")
        return None, None

    combined_df = pd.concat(all_data, ignore_index=True)

    # Create a combined label for hand sign and hand type
    combined_df['target'] = combined_df['label'] + '_' + combined_df['hand_type']

    # Separate features (landmark coordinates) and target
    feature_cols = [col for col in combined_df.columns if col.startswith('landmark_')]
    X = combined_df[feature_cols].values
    y = combined_df['target'].values

    return X, y

def train_model(X_train, y_train):
    """Trains a Support Vector Machine (SVM) model."""
    # Scale the features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # Choose and train the model (you can experiment with different models)
    model = SVC(kernel='rbf', C=1.0, gamma='scale')  # Example: SVM with RBF kernel
    # model = RandomForestClassifier(n_estimators=100, random_state=42) # Example: Random Forest
    model.fit(X_train_scaled, y_train)

    return model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    """Evaluates the trained model."""
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy on the test set: {accuracy:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

def save_model(model, scaler, filename):
    """Saves the trained model and scaler to a file."""
    with open(filename, 'wb') as file:
        pickle.dump((model, scaler), file)
    print(f"Trained model and scaler saved to '{filename}'")

def main():
    # Load and preprocess the data
    X, y = load_and_preprocess_data(DATASET_DIR)

    if X is None or y is None:
        return

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Train the model
    model, scaler = train_model(X_train, y_train)

    # Evaluate the model
    evaluate_model(model, scaler, X_test, y_test)

    # Save the trained model
    save_model(model, scaler, MODEL_FILENAME)

if __name__ == "__main__":
    main()