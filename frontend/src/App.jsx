import React, { useState, useEffect } from 'react';
import { getAreas, getModels, loadModel, predict } from './api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, LineChart, Line, Area, AreaChart, Cell } from 'recharts';

function App() {
  const [areas, setAreas] = useState([]);
  const [models, setModels] = useState([]);
  const [selectedModel, setSelectedModel] = useState('Random Forest');
  const [modelLoaded, setModelLoaded] = useState(false);
  const [loading, setLoading] = useState(false);
  const [status, setStatus] = useState('System Ready • Select a model and load it');
  
  // Input fields
  const [areaName, setAreaName] = useState('');
  const [userSex, setUserSex] = useState('MALE');
  const [userAge, setUserAge] = useState('30');
  const [timeOfDay, setTimeOfDay] = useState('Morning');
  
  // Results
  const [predictionResult, setPredictionResult] = useState(null);
  const [predictionHistory, setPredictionHistory] = useState([]);
  const [riskProbData, setRiskProbData] = useState(null);

  useEffect(() => {
    // Load initial data
    const initialize = async () => {
      try {
        const [areasData, modelsData] = await Promise.all([
          getAreas(),
          getModels()
        ]);
        setAreas(areasData);
        setModels(modelsData);
        if (areasData.length > 0) {
          setAreaName(areasData[0]);
        }
      } catch (error) {
        console.error('Error initializing:', error);
        setStatus('Error loading initial data');
      }
    };
    initialize();
  }, []);

  const handleLoadModel = async () => {
    setLoading(true);
    setStatus(`Loading ${selectedModel} model...`);
    try {
      await loadModel(selectedModel);
      setModelLoaded(true);
      setStatus(`✓ ${selectedModel} model loaded successfully • Ready for prediction`);
    } catch (error) {
      console.error('Error loading model:', error);
      setStatus(`Error loading model: ${error.message}`);
      setModelLoaded(false);
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async () => {
    if (!modelLoaded) {
      setStatus('Please load a model first');
      return;
    }

    // Validate inputs
    const age = parseInt(userAge);
    if (isNaN(age) || age < 10 || age > 99) {
      setStatus('User Age must be between 10 and 99');
      return;
    }

    if (!areaName) {
      setStatus('Please select an Area Name');
      return;
    }

    setLoading(true);
    setStatus('Making prediction...');

    try {
      const result = await predict({
        'Area Name': areaName,
        'User Sex': userSex,
        'User Age': age,
        'Time of Day': timeOfDay,
        'model_name': selectedModel
      });

      setPredictionResult(result);
      
      // Debug: Log crime categories received
      console.log('Prediction result:', result);
      console.log('Crime categories:', result.crime_categories);
      
      // Update risk probability chart data (binary: high/low only)
      // High risk threshold is now 50% (≥50% = high, <50% = low)
      if (result.probability !== null) {
        const probPercent = result.probability_percent || (result.probability * 100);
        // Charts remain binary: show actual probability values
        // The chart displays the probability of high risk vs low risk
        const highRiskPercent = probPercent;
        const lowRiskPercent = 100 - probPercent;
        
        setRiskProbData([
          { name: 'Low Risk', value: lowRiskPercent, fill: '#4CAF50' },
          { name: 'High Risk', value: highRiskPercent, fill: '#F44336' }
        ]);
      }

      // Add to prediction history
      const historyEntry = {
        prediction: result.prediction,
        probability: result.probability || 0.5,
        area: areaName,
        time: new Date().toLocaleTimeString(),
        index: predictionHistory.length + 1
      };
      setPredictionHistory([...predictionHistory, historyEntry]);

      setStatus(`Prediction completed • ${result.risk_level}`);
    } catch (error) {
      console.error('Error making prediction:', error);
      setStatus(`Error: ${error.response?.data?.error || error.message}`);
    } finally {
      setLoading(false);
    }
  };

  const handleClear = () => {
    setAreaName(areas.length > 0 ? areas[0] : '');
    setUserSex('MALE');
    setUserAge('30');
    setTimeOfDay('Morning');
    setPredictionResult(null);
    setRiskProbData(null);
    setStatus('Inputs cleared. Ready for new prediction.');
  };

  // Prepare history chart data
  const historyChartData = predictionHistory.map((entry, idx) => ({
    index: idx + 1,
    probability: entry.probability,
    risk: entry.probability >= 0.5 ? 'High' : 'Low'
  }));

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-accent"></div>
        <div className="header-content">
          <h1 className="title">Crime Risk Prediction System</h1>
          <p className="subtitle">AI-Powered Crime Occurrence Risk Assessment</p>
        </div>
      </header>

      <div className="container">
        {/* Model Selection Card */}
        <div className="card">
          <div className="card-header">
            <div className="accent-line"></div>
            <h2 className="card-title">📊 Model Selection</h2>
          </div>
          <div className="card-content">
            <div className="input-group">
              <label>ML Model:</label>
              <select 
                value={selectedModel} 
                onChange={(e) => {
                  setSelectedModel(e.target.value);
                  setModelLoaded(false);
                }}
                className="select-input"
              >
                {models.map(model => (
                  <option key={model} value={model}>{model}</option>
                ))}
              </select>
              <button 
                onClick={handleLoadModel} 
                className="btn btn-primary"
                disabled={loading}
              >
                {loading ? (
                  <>
                    <span className="loading-spinner"></span>
                    <span style={{ marginLeft: '8px' }}>Loading...</span>
                  </>
                ) : (
                  'Load Model'
                )}
              </button>
            </div>
          </div>
        </div>

        {/* Input Parameters Card */}
        <div className="card">
          <div className="card-header">
            <div className="accent-line"></div>
            <h2 className="card-title">📝 Input Parameters</h2>
          </div>
          <div className="card-content">
            <div className="input-group">
              <label>Area Name:</label>
              <select 
                value={areaName} 
                onChange={(e) => setAreaName(e.target.value)}
                className="select-input"
              >
                {areas.map(area => (
                  <option key={area} value={area}>{area}</option>
                ))}
              </select>
            </div>

            <div className="input-group">
              <label>User Sex:</label>
              <select 
                value={userSex} 
                onChange={(e) => setUserSex(e.target.value)}
                className="select-input"
              >
                <option value="MALE">MALE</option>
                <option value="FEMALE">FEMALE</option>
                <option value="UNKNOWN">UNKNOWN</option>
              </select>
            </div>

            <div className="input-group">
              <label>User Age:</label>
              <input 
                type="number" 
                value={userAge} 
                onChange={(e) => setUserAge(e.target.value)}
                min="10"
                max="99"
                className="text-input"
                placeholder="10-99"
              />
            </div>

            <div className="input-group">
              <label>Time of Day:</label>
              <select 
                value={timeOfDay} 
                onChange={(e) => setTimeOfDay(e.target.value)}
                className="select-input"
              >
                <option value="Morning">Morning</option>
                <option value="Afternoon">Afternoon</option>
                <option value="Evening">Evening</option>
                <option value="Night">Night</option>
              </select>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="button-group">
          <button 
            onClick={handlePredict} 
            className="btn btn-primary btn-large"
            disabled={loading || !modelLoaded}
          >
            {loading ? (
              <>
                <span className="loading-spinner"></span>
                <span style={{ marginLeft: '8px' }}>Analyzing...</span>
              </>
            ) : (
              '🔮 Predict Risk'
            )}
          </button>
          <button 
            onClick={handleClear} 
            className="btn btn-secondary"
            disabled={loading}
          >
            Clear All
          </button>
        </div>

        {/* Prediction Results Card */}
        <div className="card">
          <div className="card-header">
            <div className="accent-line"></div>
            <h2 className="card-title">📊 Prediction Results</h2>
          </div>
          <div className="card-content">
            <div className="result-box">
              {predictionResult ? (
                <div className="result-content" key={predictionResult.risk_level}>
                  <div className="result-header">
                    <span>Predicted Risk Level: </span>
                    <span className={`risk-label risk-${predictionResult.risk_category || (predictionResult.probability_percent > 60 ? 'high' : predictionResult.probability_percent >= 40 ? 'moderate' : 'low')}`}>
                      {predictionResult.risk_level}
                    </span>
                  </div>
                  
                  {predictionResult.probability !== null && (
                    <div className="result-probability">
                      Estimated Risk Probability: {predictionResult.probability_percent?.toFixed(2) || (predictionResult.probability * 100).toFixed(2)}%
                    </div>
                  )}

                  <div className="result-analysis">
                    <h3>Safety Analysis:</h3>
                    <ul>
                      {predictionResult.analysis.map((item, idx) => (
                        <li key={idx}>{item}</li>
                      ))}
                    </ul>

                    {/* Crime Categories Section */}
                    {predictionResult.crime_categories && predictionResult.crime_categories.length > 0 ? (
                      <div className="crime-categories-section">
                        <h4>Potential Crime Types in This Area:</h4>
                        <div className="crime-categories-list">
                          {predictionResult.crime_categories.map((crime, idx) => (
                            <div key={idx} className="crime-category-item">
                              <span className="crime-category-name">{crime.category || 'Unknown'}</span>
                              {crime.percentage !== null && crime.percentage !== undefined && (
                                <span className="crime-category-percentage">
                                  ({typeof crime.percentage === 'number' ? crime.percentage.toFixed(1) : crime.percentage}% of crimes)
                                </span>
                              )}
                            </div>
                          ))}
                        </div>
                      </div>
                    ) : (
                      <div className="crime-categories-section" style={{ opacity: 0.6 }}>
                        <p style={{ fontStyle: 'italic', color: '#666' }}>
                          Crime type data not available for this area.
                        </p>
                      </div>
                    )}

                    {predictionResult.risk_category === 'high' || (predictionResult.probability_percent && predictionResult.probability_percent > 60) ? (
                      <div>
                        <h4>Recommended Precautions:</h4>
                        <ul>
                          <li>Avoid isolated routes</li>
                          <li>Stay alert to surroundings</li>
                          <li>Prefer traveling in groups</li>
                          <li>Use well-lit and busy streets</li>
                          <li>Consider alternative routes if possible</li>
                        </ul>
                      </div>
                    ) : predictionResult.risk_category === 'moderate' || (predictionResult.probability_percent && predictionResult.probability_percent >= 40 && predictionResult.probability_percent <= 60) ? (
                      <div>
                        <h4>Moderate Risk Precautions:</h4>
                        <ul>
                          <li>Stay aware of your surroundings</li>
                          <li>Prefer well-lit areas when possible</li>
                          <li>Travel with others when feasible</li>
                          <li>Keep personal belongings secure</li>
                        </ul>
                      </div>
                    ) : (
                      <div>
                        <h4>General Advice:</h4>
                        <ul>
                          <li>Remain aware of surroundings</li>
                          <li>Follow local safety guidelines</li>
                          <li>Trust your instincts</li>
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              ) : (
                <div className="result-placeholder">
                  Enter parameters above and click 'Predict Risk' to see results here.
                  <br /><br />
                  The system will analyze the input and provide a risk assessment.
                </div>
              )}
            </div>
          </div>
        </div>

        {/* Analysis Graphs Card */}
        <div className="card">
          <div className="card-header">
            <div className="accent-line"></div>
            <h2 className="card-title">📈 Analysis Graphs</h2>
          </div>
          <div className="card-content">
            <div className="graphs-container">
              {/* Risk Probability Chart */}
              <div className="chart-container">
                <h3 className="chart-title">Risk Probability Distribution</h3>
                {riskProbData ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={riskProbData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(128, 128, 128, 0.15)" />
                      <XAxis 
                        dataKey="name" 
                        tick={{ fill: '#E0E0E0', fontSize: 12 }}
                        axisLine={{ stroke: '#808080', strokeWidth: 1 }}
                      />
                      <YAxis 
                        tick={{ fill: '#E0E0E0', fontSize: 12 }}
                        axisLine={{ stroke: '#808080', strokeWidth: 1 }}
                      />
                      <Tooltip 
                        formatter={(value) => `${value.toFixed(1)}%`}
                        contentStyle={{
                          backgroundColor: '#1E1E1E',
                          border: '1px solid rgba(128, 128, 128, 0.4)',
                          borderRadius: '8px',
                          color: '#FFD700',
                          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.4), 0 0 10px rgba(128, 128, 128, 0.2)'
                        }}
                      />
                      <Bar 
                        dataKey="value" 
                        radius={[8, 8, 0, 0]}
                        className="glow-bar"
                      >
                        {riskProbData.map((entry, index) => (
                          <Cell 
                            key={`cell-${index}`} 
                            fill={entry.fill}
                            className={entry.fill === '#4CAF50' ? 'glow-green' : 'glow-red'}
                          />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="chart-placeholder">
                    No prediction yet<br />
                    Run a prediction to see results
                  </div>
                )}
              </div>

              {/* Prediction History Chart */}
              <div className="chart-container">
                <h3 className="chart-title">Prediction History</h3>
                {historyChartData.length > 0 ? (
                  <ResponsiveContainer width="100%" height={300}>
                    <AreaChart data={historyChartData}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(128, 128, 128, 0.15)" />
                      <XAxis 
                        dataKey="index" 
                        label={{ value: 'Prediction Number', position: 'insideBottom', offset: -5 }}
                        tick={{ fill: '#E0E0E0', fontSize: 12 }}
                        axisLine={{ stroke: '#808080', strokeWidth: 1 }}
                      />
                      <YAxis 
                        label={{ value: 'Risk Probability', angle: -90, position: 'insideLeft' }}
                        tick={{ fill: '#E0E0E0', fontSize: 12 }}
                        axisLine={{ stroke: '#808080', strokeWidth: 1 }}
                      />
                      <Tooltip 
                        contentStyle={{
                          backgroundColor: '#1E1E1E',
                          border: '1px solid rgba(128, 128, 128, 0.4)',
                          borderRadius: '8px',
                          color: '#FFD700',
                          boxShadow: '0 4px 20px rgba(0, 0, 0, 0.4), 0 0 10px rgba(128, 128, 128, 0.2)'
                        }}
                      />
                      <Area 
                        type="monotone" 
                        dataKey="probability" 
                        stroke="#FFD700" 
                        strokeWidth={2.5}
                        fill="url(#areaGradient)" 
                        fillOpacity={0.3}
                        className="glow-area"
                      />
                      <defs>
                        <linearGradient id="areaGradient" x1="0" y1="0" x2="0" y2="1">
                          <stop offset="0%" stopColor="#FFD700" stopOpacity={0.4} />
                          <stop offset="100%" stopColor="#808080" stopOpacity={0.1} />
                        </linearGradient>
                      </defs>
                      <Line 
                        type="monotone" 
                        dataKey="probability" 
                        stroke="#FFD700" 
                        strokeWidth={2}
                        dot={{ fill: '#FFA500', stroke: '#000000', strokeWidth: 1.5, r: 4 }}
                      />
                    </AreaChart>
                  </ResponsiveContainer>
                ) : (
                  <div className="chart-placeholder">
                    No predictions yet<br />
                    Run predictions to see history
                  </div>
                )}
              </div>
            </div>
          </div>
        </div>

        {/* Status Bar */}
        <div className="status-bar">
          <div className="status-accent"></div>
          <div className="status-content">
            <span className="status-dot">●</span>
            <span className="status-text">{status}</span>
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;

