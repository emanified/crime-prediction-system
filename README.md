# Crime Risk Prediction System

A modern web-based application for predicting crime occurrence risk using machine learning models. Built with React (frontend) and Flask (backend).

## Project Structure

```
crime-prediction-system/
│
├── backend/
│   ├── app.py
│   ├── models.py
│   ├── area_name_to_id.pkl
│   ├── crime_random_forest_model.pkl
│   ├── crime_logistic_regression_model.pkl
│   ├── crime_decision_tree_model.pkl
│   ├── crime_knn_model.pkl
│   ├── requirements.txt
│   └── tests/
│       ├── __init__.py
│       └── test_model_paths.py
│
├── frontend/
│   ├── package.json
│   ├── vite.config.js
│   ├── index.html
│   └── src/
│       ├── main.jsx
│       ├── App.jsx
│       ├── api.js
│       └── styles.css
│
├── docs/
│   ├── README.md
│   └── AI_Project_lab_9_neww.ipynb   # model training notebook
│
└── README.md
```

## Dataset
Models were trained on the Los Angeles crime dataset (`Crime_Data_2010_2017.csv`, 120K+ records used).
The raw file is ~366 MB, so it is **not included** in this repo. To retrain, download the dataset,
then set the CSV path in the first cells of `docs/AI_Project_lab_9_neww.ipynb`.
The trained models needed to run the app are already in `backend/`.

## Features

- **Multiple ML Models**: Choose from Random Forest, Logistic Regression, Decision Tree, or KNN
- **Interactive UI**: Modern yellow, black, and white themed interface
- **Real-time Predictions**: Get instant risk assessments based on input parameters
- **Visual Analytics**: Charts showing risk probability distribution and prediction history
- **Comprehensive Analysis**: Detailed safety recommendations based on predictions

## Setup Instructions

### Backend Setup

1. Navigate to the backend directory:
```bash
cd backend
```

2. Create a virtual environment (recommended):
```bash
python -m venv venv
```

3. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Linux/Mac: `source venv/bin/activate`

4. Install dependencies:
```bash
pip install -r requirements.txt
```

5. Run the Flask server:
```bash
python app.py
```

The backend will run on `http://localhost:5000`

### Frontend Setup

1. Navigate to the frontend directory:
```bash
cd frontend
```

2. Install dependencies:
```bash
npm install
```

3. Start the development server:
```bash
npm run dev
```

The frontend will run on `http://localhost:3000`

## Usage

1. Start both the backend and frontend servers
2. Open your browser and navigate to `http://localhost:3000`
3. Select a machine learning model and click "Load Model"
4. Fill in the input parameters:
   - Area Name
   - User Sex
   - User Age (10-99)
   - Time of Day
5. Click "Predict Risk" to get the prediction
6. View the results, analysis, and charts

## API Endpoints

- `GET /api/health` - Health check
- `GET /api/areas` - Get list of available areas
- `GET /api/models` - Get list of available models
- `POST /api/models/load` - Load a model
- `POST /api/predict` - Make a prediction

## Technologies Used

- **Frontend**: React, Vite, Recharts
- **Backend**: Flask, Flask-CORS
- **ML Libraries**: scikit-learn, pandas, numpy
- **Styling**: CSS3 with modern design principles

## Notes

- Make sure all model files (`.pkl`) are in the `backend` directory
- The backend must be running before using the frontend
- The frontend is configured to proxy API requests to `http://localhost:5000`
