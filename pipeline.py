import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib
import os

def preprocess_data(data_path):
    """
    Loads and preprocesses data.
    """
    df = pd.read_csv(data_path)
    # Assuming a simple dataset where the last column is the target
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    return X, y

def train_model(X_train, y_train, model_path="model.joblib"):
    """
    Trains a RandomForestClassifier and saves the model.
    """
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    print(f"Model trained and saved to {model_path}")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Evaluates the trained model.
    """
    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)
    print(f"Model accuracy: {accuracy:.4f}")
    return accuracy

if __name__ == "__main__":
    # Create a dummy dataset for demonstration
    dummy_data = {
        'feature1': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        'feature2': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1],
        'target': [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    }
    df_dummy = pd.DataFrame(dummy_data)
    df_dummy.to_csv("dummy_data.csv", index=False)

    print("Starting MLOps pipeline...")
    X, y = preprocess_data("dummy_data.csv")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = train_model(X_train, y_train)
    evaluate_model(model, X_test, y_test)
    print("MLOps pipeline example finished.")

    # Clean up dummy files
    os.remove("dummy_data.csv")
    os.remove("model.joblib")
