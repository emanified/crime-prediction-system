"""
Flask Backend for Crime Risk Prediction System
Provides REST API endpoints for the React frontend
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Load area name to ID mapping
BASE_DIR = r'D:\Crime Prediction\backend'
AREA_NAME_TO_ID_PATH = os.path.join(BASE_DIR, 'area_name_to_id.pkl')
AREA_CRIME_PROFILE_PATH = os.path.join(BASE_DIR, 'area_crime_profile.pkl')

# Verify the area mapping file exists
if not os.path.exists(AREA_NAME_TO_ID_PATH):
    raise FileNotFoundError(f"Area mapping file not found at: {AREA_NAME_TO_ID_PATH}")

with open(AREA_NAME_TO_ID_PATH, 'rb') as f:
    AREA_NAME_TO_ID = pickle.load(f)

# Load area crime profile from separate pickle file
AREA_CRIME_PROFILE = None
if os.path.exists(AREA_CRIME_PROFILE_PATH):
    try:
        with open(AREA_CRIME_PROFILE_PATH, 'rb') as f:
            AREA_CRIME_PROFILE = pickle.load(f)
        print(f"✓ Successfully loaded area_crime_profile from {AREA_CRIME_PROFILE_PATH}")
        print(f"  Loaded profiles for {len(AREA_CRIME_PROFILE)} areas")
    except Exception as e:
        print(f"⚠ Warning: Could not load area_crime_profile: {e}")
else:
    print(f"⚠ Warning: area_crime_profile.pkl not found at {AREA_CRIME_PROFILE_PATH}")
    print("  Crime categories will be loaded from model files if available")

# Model paths
MODEL_PATHS = {
    "Random Forest": "crime_random_forest_model.pkl",
    "Logistic Regression": "crime_logistic_regression_model.pkl",
    "Decision Tree": "crime_decision_tree_model.pkl",
    "KNN": "crime_knn_model.pkl"
}

# Global model storage
loaded_models = {}


def load_model(model_name):
    """Load a model from disk"""
    if model_name in loaded_models:
        return loaded_models[model_name]
    
    # Construct full path to model file
    model_path = os.path.join(BASE_DIR, MODEL_PATHS[model_name])
    
    # Verify the model file exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file not found: {model_path}\n"
            f"Expected location: D:\\Crime Prediction\\backend\\{MODEL_PATHS[model_name]}\n"
            f"Please ensure the model file exists in the backend directory."
        )
    
    # Load the model
    try:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        raise IOError(f"Error loading model file {model_path}: {str(e)}")
    
    # Extract model components
    model_data = {
        "model": data.get("model"),
        "scaler": data.get("scaler"),
        "encoder": data.get("encoders"),
        "feature_columns": data.get("feature_columns", None),
        "area_crime_profile": data.get("area_crime_profile", None)
    }
    
    # Validate model was loaded correctly
    if model_data["model"] is None:
        raise ValueError(f"Loaded file does not contain a 'model' key. File: {model_path}")
    
    # Cache the loaded model
    loaded_models[model_name] = model_data
    print(f"✓ Successfully loaded {model_name} from {model_path}")
    return model_data


def create_feature_vector(input_values):
    """Create feature vector from input values matching notebook feature engineering"""
    age = input_values.get('User Age', 35)
    time_of_day = input_values.get("Time of Day", "Morning")
    
    # Map Time of Day to Hour (matching notebook logic)
    hour_map = {
        "Morning": 9,    # 5-12 -> use 9 as representative
        "Afternoon": 14, # 12-17 -> use 14
        "Evening": 19,   # 17-22 -> use 19
        "Night": 23      # 22-5 -> use 23
    }
    hour = hour_map.get(time_of_day, 12)
    
    # Calculate Minutes from Hour (matching notebook: Minutes = Hour * 60)
    minutes = hour * 60
    
    # Calculate Time_Intensity (matching notebook: Time_Intensity = Hour * 60 + Minutes)
    time_intensity = hour * 60 + minutes
    
    # Calculate Time_Occurred_Decimal (matching notebook: Hour + Minutes/60)
    time_occurred_decimal = hour + minutes / 60.0
    
    # Age group (matching notebook function)
    if age < 18:
        age_group = "Child"
    elif age < 30:
        age_group = "Young Adult"
    elif age < 50:
        age_group = "Adult"
    else:
        age_group = "Senior"
    
    # Default values for crime-related features (using median/common values from training)
    crime_code = 624  # BATTERY - SIMPLE ASSAULT (common crime)
    premise_code = -1.0  # Unknown premise
    reporting_district = 1000  # Default district
    
    # Calculate Crime_Severity_Index (matching notebook formula)
    # Crime_Severity_Index = Crime Code * 0.6 + Premise Code * 0.3 + Hour * 0.1
    crime_severity_index = crime_code * 0.6 + premise_code * 0.3 + hour * 0.1
    
    # Determine Crime_Severity_Bucket (matching notebook function)
    if crime_severity_index < 200:
        crime_severity_bucket = "Low"
    elif crime_severity_index < 500:
        crime_severity_bucket = "Medium"
    else:
        crime_severity_bucket = "High"
    
    # Get current date components
    now = datetime.now()
    
    # Calculate weekday (Monday=0, Sunday=6)
    weekday = now.weekday()
    is_weekend = 1 if weekday >= 5 else 0
    
    # Calculate report lag (assuming same day for prediction)
    report_lag_days = 0
    
    features = {
        'Area ID': AREA_NAME_TO_ID[input_values["Area Name"]],
        'Reporting District': reporting_district,
        'Crime Code': crime_code,
        'MO Codes': 0,  # Will be encoded
        'Victim Age': age,
        'Victim Sex': input_values.get('User Sex', 'MALE'),
        'Victim Descent': 'H',  # H is most common in training data
        'Premise Code': premise_code,
        'Premise Description': 0,  # Will be encoded
        'Weapon Description': 0,  # Will be encoded
        'Status Code': 0,  # Will be encoded
        'Status Description': 0,  # Will be encoded
        'Hour': hour,
        'Occur_Year': now.year,
        'Occur_Month': now.month,
        'Occur_Day': now.day,
        'Occur_Weekday': weekday,
        'Occur_is_weekend': is_weekend,
        'Report_Year': now.year,
        'Report_Month': now.month,
        'report_lag_days': report_lag_days,
        'Crime_Severity_Index': crime_severity_index,
        'Minutes': minutes,
        'Time_Intensity': time_intensity,
        'Latitude': 34.0522,  # LA coordinates (default)
        'Longitude': -118.2437,
        'Time_Occurred_Decimal': time_occurred_decimal,
        'Time_of_Day': time_of_day,
        'Crime_Category': 'BATTERY',  # Most common category
        'Crime_Severity_Bucket': crime_severity_bucket,
        'Victim_Age_Group': age_group
    }
    
    return features


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({"status": "ok", "message": "Crime Prediction API is running"})


@app.route('/api/areas', methods=['GET'])
def get_areas():
    """Get list of available areas"""
    areas = sorted(AREA_NAME_TO_ID.keys())
    return jsonify({"areas": areas})


@app.route('/api/models', methods=['GET'])
def get_models():
    """Get list of available models"""
    return jsonify({"models": list(MODEL_PATHS.keys())})


@app.route('/api/models/load', methods=['POST'])
def load_model_endpoint():
    """Load a model"""
    try:
        data = request.json
        model_name = data.get('model_name')
        
        if not model_name or model_name not in MODEL_PATHS:
            return jsonify({"error": "Invalid model name"}), 400
        
        model_data = load_model(model_name)
        
        return jsonify({
            "success": True,
            "message": f"{model_name} model loaded successfully",
            "model_name": model_name
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/predict', methods=['POST'])
def predict():
    """Make a prediction"""
    try:
        data = request.json
        
        # Validate required fields
        required_fields = ['Area Name', 'User Sex', 'User Age', 'Time of Day', 'model_name']
        for field in required_fields:
            if field not in data:
                return jsonify({"error": f"Missing required field: {field}"}), 400
        
        # Validate area name
        if data['Area Name'] not in AREA_NAME_TO_ID:
            return jsonify({"error": "Invalid Area Name"}), 400
        
        # Validate age
        try:
            age = int(data['User Age'])
            if not (10 <= age <= 99):
                return jsonify({"error": "User Age must be between 10 and 99"}), 400
        except ValueError:
            return jsonify({"error": "User Age must be a valid number"}), 400
        
        # Validate time of day
        if data['Time of Day'] not in ["Morning", "Afternoon", "Evening", "Night"]:
            return jsonify({"error": "Invalid Time of Day"}), 400
        
        # Load model if not already loaded
        model_name = data['model_name']
        model_data = load_model(model_name)
        
        # Create feature vector
        features = create_feature_vector(data)
        X = pd.DataFrame([features])
        
        # Apply encoders to categorical columns
        # Note: The notebook uses LabelEncoder for all categorical columns
        # The encoders dict should contain encoders for: Victim Sex, Victim Descent, 
        # Time_of_Day, Crime_Category, Crime_Severity_Bucket, Victim_Age_Group,
        # and also: MO Codes, Premise Description, Weapon Description, Status Code, Status Description
        categorical_cols = ['Victim Sex', 'Victim Descent', 'Time_of_Day', 'Crime_Category', 
                           'Crime_Severity_Bucket', 'Victim_Age_Group', 'MO Codes',
                           'Premise Description', 'Weapon Description', 'Status Code', 'Status Description']
        
        for col in categorical_cols:
            if col in X.columns:
                if model_data["encoder"] is not None and col in model_data["encoder"]:
                    try:
                        # Try to transform using saved encoder
                        # Handle unseen values by using the first class
                        values = X[col].astype(str).values
                        # Check for unseen values
                        encoder = model_data["encoder"][col]
                        unseen_mask = ~pd.Series(values).isin(encoder.classes_)
                        if unseen_mask.any():
                            # Replace unseen values with first class
                            values[unseen_mask] = encoder.classes_[0]
                        X[col] = encoder.transform(values)
                    except Exception as e:
                        # Fallback to categorical codes if encoder fails
                        print(f"Warning: Encoder failed for {col}: {e}. Using fallback encoding.")
                        X[col] = pd.Categorical(X[col]).codes
                else:
                    # Fallback: use categorical codes
                    X[col] = pd.Categorical(X[col]).codes
        
        # Reorder columns to match model
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
        else:
            # If no probability available, use prediction as probability estimate
            proba = float(prediction)
        
        # Classify risk based on probability percentage
        # High Risk: ≥ 50%
        # Moderate Risk: 40-49%
        # Low Risk: < 40% (covers "around 30 or lower")
        proba_percent = proba * 100
        
        if proba_percent >= 50:
            risk_level = "HIGH RISK"
            risk_category = "high"
        elif proba_percent >= 40:
            risk_level = "MODERATE RISK"
            risk_category = "moderate"
        else:  # < 40% (covers "around 30 or lower")
            risk_level = "LOW RISK"
            risk_category = "low"
        
        # Get crime categories for the area from area_crime_profile (data-driven, not hardcoded)
        # Priority: 1) Separate pickle file, 2) Model file (backward compatibility)
        crime_categories = []
        area_name = data['Area Name']
        
        # Use the separately loaded area_crime_profile (preferred)
        area_crime_profile = AREA_CRIME_PROFILE
        
        # Fallback to model data if separate file not available (backward compatibility)
        if area_crime_profile is None:
            area_crime_profile = model_data.get("area_crime_profile")
            if area_crime_profile is not None:
                print(f"Using area_crime_profile from model file")
        
        if area_crime_profile is not None:
            # area_crime_profile is a dict mapping area names to crime categories
            # New format: {"Area Name": [("Crime_Category", percentage), ...]} - list of tuples (multiple crimes per area)
            # Old format: {"Area Name": ("Crime_Category", percentage)} - single tuple (backward compatibility)
            # This is created from actual dataset in the notebook, not hardcoded
            # The notebook now gets top 5 crime categories per area, not just the most common one
            # Try exact match first
            if area_name in area_crime_profile:
                category_info = area_crime_profile[area_name]
                
                # Handle new format: list of tuples (multiple crime types)
                if isinstance(category_info, list):
                    for crime_tuple in category_info:
                        if isinstance(crime_tuple, tuple) and len(crime_tuple) >= 1:
                            crime_cat = crime_tuple[0] if len(crime_tuple) > 0 else None
                            crime_pct = crime_tuple[1] if len(crime_tuple) > 1 else None
                            # Convert percentage to float if it's a number
                            if crime_pct is not None:
                                try:
                                    crime_pct = float(crime_pct)
                                except (ValueError, TypeError):
                                    crime_pct = None
                            
                            if crime_cat:
                                crime_categories.append({
                                    "category": str(crime_cat),
                                    "percentage": crime_pct
                                })
                # Handle old format: single tuple (backward compatibility)
                elif isinstance(category_info, tuple) and len(category_info) >= 1:
                    crime_cat = category_info[0] if len(category_info) > 0 else None
                    crime_pct = category_info[1] if len(category_info) > 1 else None
                    # Convert percentage to float if it's a number
                    if crime_pct is not None:
                        try:
                            crime_pct = float(crime_pct)
                        except (ValueError, TypeError):
                            crime_pct = None
                    
                    if crime_cat:
                        crime_categories.append({
                            "category": str(crime_cat),
                            "percentage": crime_pct
                        })
            else:
                # Try case-insensitive and title case matching
                area_name_upper = area_name.upper().strip()
                area_name_title = area_name.title().strip()
                area_name_lower = area_name.lower().strip()
                
                for key, value in area_crime_profile.items():
                    key_upper = key.upper().strip()
                    key_title = key.title().strip()
                    key_lower = key.lower().strip()
                    
                    if (key_upper == area_name_upper or 
                        key_title == area_name_title or 
                        key_lower == area_name_lower or
                        key.strip() == area_name.strip()):
                        # Handle new format: list of tuples (multiple crime types)
                        if isinstance(value, list):
                            for crime_tuple in value:
                                if isinstance(crime_tuple, tuple) and len(crime_tuple) >= 1:
                                    crime_cat = crime_tuple[0] if len(crime_tuple) > 0 else None
                                    crime_pct = crime_tuple[1] if len(crime_tuple) > 1 else None
                                    # Convert percentage to float if it's a number
                                    if crime_pct is not None:
                                        try:
                                            crime_pct = float(crime_pct)
                                        except (ValueError, TypeError):
                                            crime_pct = None
                                    
                                    if crime_cat:
                                        crime_categories.append({
                                            "category": str(crime_cat),
                                            "percentage": crime_pct
                                        })
                        # Handle old format: single tuple (backward compatibility)
                        elif isinstance(value, tuple) and len(value) >= 1:
                            crime_cat = value[0] if len(value) > 0 else None
                            crime_pct = value[1] if len(value) > 1 else None
                            # Convert percentage to float if it's a number
                            if crime_pct is not None:
                                try:
                                    crime_pct = float(crime_pct)
                                except (ValueError, TypeError):
                                    crime_pct = None
                            
                            if crime_cat:
                                crime_categories.append({
                                    "category": str(crime_cat),
                                    "percentage": crime_pct
                                })
                        break
        else:
            # Debug: Log when area_crime_profile is not available
            print(f"⚠ Warning: area_crime_profile is None for area: {area_name}")
            print(f"  AREA_CRIME_PROFILE global: {AREA_CRIME_PROFILE is not None}")
            print(f"  Model data has area_crime_profile: {model_data.get('area_crime_profile') is not None}")
        
        # Note: If area_crime_profile is missing or area not found, crime_categories will be empty
        # This ensures all crime data comes from the actual dataset (area_crime_profile), not hardcoded values
        
        # Generate analysis text based on risk level
        hour_map = {"Morning": 9, "Afternoon": 14, "Evening": 19, "Night": 23}
        hour = hour_map.get(data['Time of Day'], 12)
        
        analysis = []
        if risk_category == "high":
            analysis.append(f"This area ({area_name}) shows a high likelihood of crime occurrence ({proba_percent:.1f}% risk) based on historical data.")
            if hour >= 18 or hour <= 5:
                analysis.append("Risk levels are significantly higher during evening and night hours.")
            if data['User Sex'] == "FEMALE":
                analysis.append("Extra caution is strongly advised, especially in less crowded or poorly lit areas.")
        elif risk_category == "moderate":
            analysis.append(f"This area ({area_name}) shows a moderate crime risk ({proba_percent:.1f}% risk) during the selected time.")
            analysis.append("Standard safety precautions are recommended. Stay aware of your surroundings.")
            if hour >= 18 or hour <= 5:
                analysis.append("Be extra vigilant during evening and night hours.")
        else:  # low risk
            analysis.append(f"This area ({area_name}) shows a relatively low crime risk ({proba_percent:.1f}% risk) during the selected time.")
            analysis.append("Normal safety awareness is advised, as no area is completely risk-free.")
        
        # Debug: Log crime categories being returned
        print(f"\n=== DEBUG: Crime Categories Lookup ===")
        print(f"Area Name: '{area_name}'")
        print(f"AREA_CRIME_PROFILE loaded: {AREA_CRIME_PROFILE is not None}")
        if AREA_CRIME_PROFILE is not None:
            print(f"Total areas in profile: {len(AREA_CRIME_PROFILE)}")
            print(f"Sample area names: {list(AREA_CRIME_PROFILE.keys())[:5]}")
            print(f"Area '{area_name}' in profile: {area_name in AREA_CRIME_PROFILE}")
            if area_name in AREA_CRIME_PROFILE:
                print(f"  Found data: {AREA_CRIME_PROFILE[area_name]}")
        print(f"Crime categories found: {len(crime_categories)}")
        if crime_categories:
            print(f"  Categories: {crime_categories}")
        print("=" * 40)
        
        return jsonify({
            "success": True,
            "prediction": int(prediction),
            "risk_level": risk_level,
            "risk_category": risk_category,
            "probability": proba,
            "probability_percent": proba_percent,
            "analysis": analysis,
            "crime_categories": crime_categories,
            "model_name": model_name
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000)

