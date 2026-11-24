"""
Quick script to check if area_crime_profile exists in models or as separate file
"""
import pickle
import os

BASE_DIR = r'D:\Crime Prediction\backend'

# Check separate file
separate_file = os.path.join(BASE_DIR, 'area_crime_profile.pkl')
print(f"Checking separate file: {separate_file}")
if os.path.exists(separate_file):
    try:
        with open(separate_file, 'rb') as f:
            profile = pickle.load(f)
        print(f"✓ Found separate area_crime_profile.pkl with {len(profile)} areas")
        print(f"  Sample areas: {list(profile.keys())[:3]}")
    except Exception as e:
        print(f"✗ Error loading separate file: {e}")
else:
    print("✗ Separate area_crime_profile.pkl NOT FOUND")

# Check models
model_files = [
    'crime_random_forest_model.pkl',
    'crime_logistic_regression_model.pkl',
    'crime_decision_tree_model.pkl',
    'crime_knn_model.pkl'
]

print("\nChecking models for embedded area_crime_profile:")
for model_file in model_files:
    model_path = os.path.join(BASE_DIR, model_file)
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                data = pickle.load(f)
            if 'area_crime_profile' in data and data['area_crime_profile'] is not None:
                profile = data['area_crime_profile']
                print(f"✓ {model_file}: Found area_crime_profile with {len(profile)} areas")
            else:
                print(f"✗ {model_file}: No area_crime_profile found")
        except Exception as e:
            print(f"✗ {model_file}: Error loading - {e}")
    else:
        print(f"✗ {model_file}: File not found")

