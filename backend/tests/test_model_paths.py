"""
Test script to verify all model files can be found at the correct paths
"""

import os

BASE_DIR = r'D:\Crime Prediction\backend'

# Required files
REQUIRED_FILES = [
    'area_name_to_id.pkl',
    'crime_random_forest_model.pkl',
    'crime_logistic_regression_model.pkl',
    'crime_decision_tree_model.pkl',
    'crime_knn_model.pkl'
]

print("=" * 60)
print("Model File Path Verification")
print("=" * 60)
print(f"\nBase Directory: {BASE_DIR}\n")

all_found = True
for filename in REQUIRED_FILES:
    filepath = os.path.join(BASE_DIR, filename)
    exists = os.path.exists(filepath)
    status = "✓ FOUND" if exists else "✗ NOT FOUND"
    print(f"{status}: {filename}")
    print(f"  Path: {filepath}")
    if not exists:
        all_found = False
    print()

if all_found:
    print("=" * 60)
    print("✓ All required files are present!")
    print("=" * 60)
else:
    print("=" * 60)
    print("✗ Some files are missing. Please check the paths above.")
    print("=" * 60)

