"""
Test script to verify predictions match notebook training
This script tests if models load correctly and predictions are valid
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app import load_model, create_feature_vector, AREA_NAME_TO_ID
import pandas as pd
import numpy as np

def test_model_loading():
    """Test if all models can be loaded correctly"""
    print("=" * 60)
    print("Testing Model Loading")
    print("=" * 60)
    
    models = ["Random Forest", "Logistic Regression", "Decision Tree", "KNN"]
    results = {}
    
    for model_name in models:
        try:
            model_data = load_model(model_name)
            
            # Verify model components
            assert model_data["model"] is not None, f"{model_name}: Model is None"
            assert model_data["feature_columns"] is not None, f"{model_name}: Feature columns missing"
            
            print(f"✓ {model_name}:")
            print(f"  - Model loaded: {type(model_data['model']).__name__}")
            print(f"  - Feature columns: {len(model_data['feature_columns'])}")
            print(f"  - Has scaler: {model_data['scaler'] is not None}")
            print(f"  - Has encoders: {model_data['encoder'] is not None}")
            
            results[model_name] = "PASS"
        except Exception as e:
            print(f"✗ {model_name}: FAILED - {str(e)}")
            results[model_name] = f"FAIL: {str(e)}"
    
    return results


def test_feature_creation():
    """Test if feature vector creation matches expected format"""
    print("\n" + "=" * 60)
    print("Testing Feature Vector Creation")
    print("=" * 60)
    
    # Test input
    test_input = {
        "Area Name": list(AREA_NAME_TO_ID.keys())[0],
        "User Sex": "MALE",
        "User Age": 30,
        "Time of Day": "Morning"
    }
    
    try:
        features = create_feature_vector(test_input)
        
        # Verify required features exist
        required_features = [
            'Area ID', 'Victim Age', 'Victim Sex', 'Hour', 
            'Crime_Severity_Index', 'Time_of_Day', 'Victim_Age_Group'
        ]
        
        missing = [f for f in required_features if f not in features]
        if missing:
            print(f"✗ Missing features: {missing}")
            return False
        
        # Verify feature types and ranges
        assert isinstance(features['Area ID'], (int, np.integer)), "Area ID should be integer"
        assert 10 <= features['Victim Age'] <= 99, "Victim Age should be in valid range"
        assert features['Hour'] in [9, 14, 19, 23], "Hour should match Time of Day"
        assert features['Time_of_Day'] in ["Morning", "Afternoon", "Evening", "Night"]
        assert features['Victim_Age_Group'] in ["Child", "Young Adult", "Adult", "Senior"]
        
        print("✓ Feature vector created successfully")
        print(f"  - Total features: {len(features)}")
        print(f"  - Sample features: Area ID={features['Area ID']}, Hour={features['Hour']}, Age Group={features['Victim_Age_Group']}")
        
        return True
    except Exception as e:
        print(f"✗ Feature creation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def test_prediction_pipeline():
    """Test the full prediction pipeline"""
    print("\n" + "=" * 60)
    print("Testing Prediction Pipeline")
    print("=" * 60)
    
    # Load a model
    try:
        model_data = load_model("Random Forest")
    except Exception as e:
        print(f"✗ Cannot load model: {str(e)}")
        return False
    
    # Create test input
    test_input = {
        "Area Name": list(AREA_NAME_TO_ID.keys())[0],
        "User Sex": "MALE",
        "User Age": 30,
        "Time of Day": "Morning"
    }
    
    try:
        # Create feature vector
        features = create_feature_vector(test_input)
        X = pd.DataFrame([features])
        
        # Apply encoders
        categorical_cols = ['Victim Sex', 'Victim Descent', 'Time_of_Day', 'Crime_Category', 
                           'Crime_Severity_Bucket', 'Victim_Age_Group', 'MO Codes',
                           'Premise Description', 'Weapon Description', 'Status Code', 'Status Description']
        
        for col in categorical_cols:
            if col in X.columns:
                if model_data["encoder"] is not None and col in model_data["encoder"]:
                    try:
                        values = X[col].astype(str).values
                        encoder = model_data["encoder"][col]
                        unseen_mask = ~pd.Series(values).isin(encoder.classes_)
                        if unseen_mask.any():
                            values[unseen_mask] = encoder.classes_[0]
                        X[col] = encoder.transform(values)
                    except:
                        X[col] = pd.Categorical(X[col]).codes
                else:
                    X[col] = pd.Categorical(X[col]).codes
        
        # Reorder columns
        if model_data["feature_columns"] is not None:
            missing_cols = [c for c in model_data["feature_columns"] if c not in X.columns]
            for c in missing_cols:
                X[c] = 0
            X = X[model_data["feature_columns"]]
        
        # Apply scaler if present
        X_scaled = model_data["scaler"].transform(X) if model_data["scaler"] is not None else X
        
        # Make prediction
        prediction = model_data["model"].predict(X_scaled)[0]
        proba = None
        if hasattr(model_data["model"], "predict_proba"):
            proba = float(model_data["model"].predict_proba(X_scaled)[0][1])
        
        print(f"✓ Prediction successful")
        print(f"  - Prediction: {prediction} ({'HIGH RISK' if prediction == 1 else 'LOW RISK'})")
        print(f"  - Probability: {proba:.4f}" if proba is not None else "  - Probability: N/A")
        print(f"  - Input shape: {X_scaled.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Prediction failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("CRIME PREDICTION SYSTEM - VALIDATION TESTS")
    print("=" * 60)
    
    # Run tests
    load_results = test_model_loading()
    feature_test = test_feature_creation()
    prediction_test = test_prediction_pipeline()
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Model Loading: {'PASS' if all(r == 'PASS' for r in load_results.values()) else 'FAIL'}")
    print(f"Feature Creation: {'PASS' if feature_test else 'FAIL'}")
    print(f"Prediction Pipeline: {'PASS' if prediction_test else 'FAIL'}")
    
    if all([all(r == 'PASS' for r in load_results.values()), feature_test, prediction_test]):
        print("\n✓ All tests passed! Predictions should work correctly.")
    else:
        print("\n✗ Some tests failed. Please review the errors above.")

