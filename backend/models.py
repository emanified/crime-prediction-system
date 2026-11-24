"""
Model utilities and helper functions for crime prediction
"""

import pickle
import os

# Base directory for model files
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def get_model_path(model_name):
    """Get the file path for a model"""
    model_map = {
        "Random Forest": "crime_random_forest_model.pkl",
        "Logistic Regression": "crime_logistic_regression_model.pkl",
        "Decision Tree": "crime_decision_tree_model.pkl",
        "KNN": "crime_knn_model.pkl"
    }
    model_filename = model_map.get(model_name, "")
    if not model_filename:
        raise ValueError(f"Unknown model name: {model_name}")
    return os.path.join(BASE_DIR, model_filename)


def validate_model_file(model_name):
    """Check if model file exists and return the full path"""
    model_path = get_model_path(model_name)
    exists = os.path.exists(model_path)
    if not exists:
        print(f"⚠ Warning: Model file not found: {model_path}")
    return exists, model_path


def save_model(model, scaler, encoders, feature_columns, model_name):
    """Save a trained model with its preprocessing components to D: drive"""
    saved_data = {
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'feature_columns': feature_columns.tolist() if hasattr(feature_columns, 'tolist') else list(feature_columns)
    }
    
    filename = f"crime_{model_name.lower().replace(' ', '_')}_model.pkl"
    filepath = os.path.join(BASE_DIR, filename)
    
    with open(filepath, "wb") as f:
        pickle.dump(saved_data, f)
    
    print(f"✓ {model_name} model saved to {filepath}")
    return filepath

# Note: This script should be run from the notebook after training models
# Or you can modify it to load data and train models here

if __name__ == "__main__":
    print("=" * 60)
    print("Model Saving Script")
    print("=" * 60)
    print("\nThis script is designed to be run from the Jupyter notebook")
    print("after models have been trained.\n")
    print("To save models from the notebook, add this code after each model training:")
    print("""
# Example for Random Forest:
save_data = {
    'model': rf,
    'scaler': None,  # Random Forest doesn't need scaling
    'encoders': encoders,  # Your label encoders dict
    'feature_columns': X_train.columns.tolist()
}
with open('crime_random_forest_model.pkl', 'wb') as f:
    pickle.dump(save_data, f)

# Example for Logistic Regression:
save_data = {
    'model': lr_model,
    'scaler': scaler,  # Your StandardScaler
    'encoders': encoders,
    'feature_columns': X_train.columns.tolist()
}
with open('crime_logistic_regression_model.pkl', 'wb') as f:
    pickle.dump(save_data, f)
    """)
    print("\nAlternatively, you can modify this script to load your data")
    print("and train models directly here.")

