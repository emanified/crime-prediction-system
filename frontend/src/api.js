/**
 * API client for Crime Risk Prediction System
 */

import axios from 'axios';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:5000';

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const getAreas = async () => {
  const response = await api.get('/api/areas');
  return response.data.areas;
};

export const getModels = async () => {
  const response = await api.get('/api/models');
  return response.data.models;
};

export const loadModel = async (modelName) => {
  const response = await api.post('/api/models/load', { model_name: modelName });
  return response.data;
};

export const predict = async (data) => {
  const response = await api.post('/api/predict', data);
  return response.data;
};

export default api;

