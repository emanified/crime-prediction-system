# Crime Risk Prediction System

A modern web-based application for predicting crime occurrence risk using machine learning models. Built with React (frontend) and Flask (backend).

## Project Structure

```
crime-prediction-system/
â”‚
â”œâ”€â”€ backend/
â”‚   â”œâ”€â”€ app.py
â”‚   â”œâ”€â”€ models.py
â”‚   â”œâ”€â”€ area_name_to_id.pkl
â”‚   â”œâ”€â”€ crime_random_forest_model.pkl
â”‚   â”œâ”€â”€ crime_logistic_regression_model.pkl
â”‚   â”œâ”€â”€ crime_decision_tree_model.pkl
â”‚   â”œâ”€â”€ crime_knn_model.pkl
â”‚   â”œâ”€â”€ requirements.txt
â”‚   â””â”€â”€ tests/
â”‚       â”œâ”€â”€ __init__.py
â”‚       â””â”€â”€ test_model_paths.py
â”‚
â”œâ”€â”€ frontend/
â”‚   â”œâ”€â”€ package.json
â”‚   â”œâ”€â”€ vite.config.js
â”‚   â”œâ”€â”€ index.html
â”‚   â””â”€â”€ src/
â”‚       â”œâ”€â”€ main.jsx
â”‚       â”œâ”€â”€ App.jsx
â”‚       â”œâ”€â”€ api.js
â”‚       â””â”€â”€ styles.css
â”‚
â”œâ”€â”€ docs/
â”‚   â”œâ”€â”€ README.md
â”‚   â””â”€â”€ AI_Project_lab_9_neww.ipynb   # model training notebook
â”‚
â””â”€â”€ README.md
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

Requires **Python 3.11+** and Node.js 18+.

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
