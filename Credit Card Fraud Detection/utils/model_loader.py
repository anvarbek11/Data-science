import joblib
import os
import streamlit as st


def load_model_artifacts():
    """Load model artifacts with error handling"""
    try:
        # Load from the correct path (adjust as needed)
        model_path = os.path.join('models/isolation_forest_20250508_2214.pkl')
        scaler_path = os.path.join('models/scaler_20250508_2214.pkl')
        features_path = os.path.join('models/features_20250508_2214.pkl')

        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        features = joblib.load(features_path)

        return model, scaler, features

    except Exception as e:
        st.error(f"Error loading model artifacts: {str(e)}")
        raise