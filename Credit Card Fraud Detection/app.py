import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from utils.model_loader import load_model_artifacts
from utils.visualizations import (
    plot_feature_distributions,
    plot_confusion_matrix,
    plot_metrics,
    generate_simulation_results
)
import warnings
from datetime import datetime
import os

# Suppress sklearn warnings
warnings.filterwarnings('ignore', category=UserWarning)

# Page configuration
st.set_page_config(
    page_title="Credit Card Fraud Detection",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main {
        max-width: 1200px;
    }
    .stNumberInput, .stTextInput {
        margin-bottom: 1rem;
    }
    .fraud-alert {
        background-color: #fff8f8;
        border-left: 4px solid #ff4b4b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .legit-alert {
        background-color: #f8fff8;
        border-left: 4px solid #4bff4b;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 4px;
    }
    .metric-box {
        border: 1px solid #e0e0e0;
        border-radius: 5px;
        padding: 1rem;
        margin-bottom: 1rem;
        background-color: #f9f9f9;
    }
    .stButton>button {
        background-color: #4CAF50;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .stDownloadButton>button {
        background-color: #2196F3;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Load model artifacts with caching and validation
@st.cache_resource(ttl=3600)
def load_artifacts():
    """Load and validate model artifacts"""
    try:
        model, scaler, features = load_model_artifacts()

        # Validate loaded artifacts
        if not all([model, scaler, features]):
            raise ValueError("One or more model artifacts failed to load")

        # Ensure features doesn't contain 'Class'
        features = [f for f in features if f != 'Class']

        return model, scaler, features

    except Exception as e:
        st.error(f"Failed to load model artifacts: {str(e)}")
        st.stop()


model, scaler, features = load_artifacts()


# Simplified prediction function without feature contributions
def predict_fraud(input_data):
    """Make prediction with input validation"""
    try:
        # Validate input data
        if not input_data or len(input_data) != len(features):
            raise ValueError("Invalid input data dimensions")

        # Convert input to numpy array
        input_values = np.array([input_data[f] for f in features]).reshape(1, -1)

        # Scale features
        scaled_input = scaler.transform(input_values)

        # Make prediction
        prediction = model.predict(scaled_input)
        decision_score = model.decision_function(scaled_input)

        # Convert to probability using sigmoid
        fraud_prob = 1 / (1 + np.exp(-decision_score))

        return {
            'prediction': 'Fraud' if prediction[0] == -1 else 'Legitimate',
            'confidence': float(fraud_prob[0] if prediction[0] == -1 else 1 - fraud_prob[0])
        }

    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        st.stop()


# Sample data
SAMPLE_FRAUD = {
    'V1': -3.043540, 'V2': -3.157307, 'V3': 1.088463,
    'V4': 2.288644, 'V5': 1.359805, 'V6': -1.064823,
    'V7': 0.325574, 'V9': -0.294166, 'V10': -0.932235,
    'V11': 1.231782, 'V12': -0.386003, 'V14': -2.384855,
    'V16': -2.682008, 'V17': -4.382873, 'V18': -2.924271,
    'V19': 0.133735, 'V20': -0.021053, 'V21': 0.277838,
    'V24': -0.110474, 'V27': -0.189115, 'V28': 0.133558,
    'V29': -0.021053, 'V30': 0.066928
}

SAMPLE_LEGIT = {
    'V1': 1.234821, 'V2': 0.456789, 'V3': -0.123456,
    'V4': 0.987654, 'V5': -0.654321, 'V6': 0.321098,
    'V7': -0.789012, 'V9': 0.543210, 'V10': -0.210987,
    'V11': 0.876543, 'V12': -0.432109, 'V14': 0.765432,
    'V16': -0.345678, 'V17': 0.678901, 'V18': -0.901234,
    'V19': 0.123456, 'V20': 0.456789, 'V21': -0.789012,
    'V24': 0.234567, 'V27': -0.567890, 'V28': 0.890123,
    'V29': -0.123456, 'V30': 0.456789
}

# Navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("", ["Demo", "Model Analysis", "Business Impact", "About"], index=0)

# Demo Page
if page == "Demo":
    st.title("💳 Credit Card Fraud Detection")
    st.markdown("""
    This interactive demo uses machine learning to detect potentially fraudulent credit card transactions.
    Enter transaction details below to analyze fraud risk.
    """)

    with st.expander("ℹ️ How to use", expanded=True):
        st.markdown("""
        1. **Enter values** for each feature (or use our sample data)
        2. Click **"Analyze Transaction"**
        3. Review the **fraud risk assessment**
        """)

    # Initialize session state for sample data
    if 'sample_data' not in st.session_state:
        st.session_state.sample_data = None

    # Sample data buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load Sample Fraud Transaction"):
            st.session_state.sample_data = SAMPLE_FRAUD
    with col2:
        if st.button("Load Sample Legitimate Transaction"):
            st.session_state.sample_data = SAMPLE_LEGIT

    # Input form
    with st.form("transaction_form"):
        st.subheader("Transaction Features")

        # Create input fields dynamically
        input_data = {}
        cols = st.columns(3)
        for i, feature in enumerate(features):
            with cols[i % 3]:
                # Set default value to sample data if available, otherwise 0
                default_value = st.session_state.sample_data.get(feature, 0.0) if st.session_state.sample_data else 0.0
                input_data[feature] = st.number_input(
                    label=feature,
                    value=default_value,
                    format="%.6f",
                    step=0.000001,
                    key=f"input_{feature}"
                )

        submitted = st.form_submit_button("Analyze Transaction")

    if submitted:
        # Validate inputs
        if all(abs(val) < 0.0001 for val in input_data.values()):
            st.warning("⚠️ Please enter values for at least some features (all values are near zero)")
            st.stop()

        # Make prediction
        with st.spinner("Analyzing transaction..."):
            result = predict_fraud(input_data)

        # Display results
        st.subheader("Fraud Assessment")

        if result['prediction'] == 'Fraud':
            st.error(f"🚨 **Fraud Detected** (Confidence: {result['confidence']:.1%})")
            st.markdown("""
            <div class="fraud-alert">
                <h4>⚠️ Action Recommended</h4>
                <p>This transaction has been flagged as potentially fraudulent with high confidence. 
                Please review carefully and consider:</p>
                <ul>
                    <li>Additional verification steps</li>
                    <li>Contacting the cardholder</li>
                    <li>Reviewing transaction history</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.success(f"✅ **Legitimate Transaction** (Confidence: {result['confidence']:.1%})")
            st.markdown("""
            <div class="legit-alert">
                <h4>✓ Likely Safe</h4>
                <p>This transaction appears legitimate based on our analysis. 
                Standard processing recommended.</p>
            </div>
            """, unsafe_allow_html=True)

# Model Analysis Page
elif page == "Model Analysis":
    st.title("Model Performance Analysis")

    st.markdown("""
    ## Isolation Forest Model Characteristics

    Our fraud detection system uses an Isolation Forest algorithm with the following performance:
    """)

    # Metrics display
    metrics = {
        'Recall': 0.87,
        'Precision': 0.03,
        'F1 Score': 0.06,
        'ROC AUC': 0.91
    }

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(plot_metrics(metrics), use_container_width=True)
    with col2:
        st.markdown("""
        <div class="metric-box">
            <h4>Key Metrics Explained</h4>
            <ul>
                <li><strong>Recall (87%)</strong>: Captures 87% of actual fraud cases</li>
                <li><strong>Precision (3%)</strong>: Only 3% of flagged transactions are truly fraudulent</li>
                <li><strong>F1 Score (0.06)</strong>: Balance between precision and recall</li>
                <li><strong>ROC AUC (0.91)</strong>: Excellent discrimination between fraud and legitimate transactions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Confusion matrix
    st.subheader("Confusion Matrix")
    cm = np.array([[270000, 14315], [64, 428]])  # TN, FP, FN, TP
    st.plotly_chart(plot_confusion_matrix(cm))

    # Feature importance
    st.subheader("Feature Distributions")
    st.plotly_chart(plot_feature_distributions())

# Business Impact Page
# Business Impact Page
elif page == "Business Impact":
    st.title("Business Impact Analysis")

    st.markdown("""
    ## Financial Impact Simulation

    Adjust the parameters below to estimate how this fraud detection system could impact your business.
    """)

    with st.expander("Simulation Parameters", expanded=True):
        col1, col2 = st.columns(2)
        with col1:
            avg_transaction = st.number_input(
                "Average Transaction Amount ($)",
                min_value=1,
                value=100,
                step=10
            )
            fraud_rate = st.slider(
                "Expected Fraud Rate (%)",
                min_value=0.01,
                max_value=5.0,
                value=0.17,
                step=0.01
            ) / 100  # Convert percentage to decimal
        with col2:
            transaction_volume = st.number_input(
                "Monthly Transaction Volume",
                min_value=100,
                value=10000,
                step=100
            )
            fraud_cost_multiplier = st.slider(
                "Fraud Cost Multiplier",
                min_value=1.0,
                max_value=3.0,
                value=1.5,
                step=0.1,
                help="Includes chargebacks, fees, and other costs"
            )

    if st.button("Calculate Impact"):
        # Calculate results directly instead of calling generate_simulation_results
        potential_losses = avg_transaction * transaction_volume * fraud_rate * fraud_cost_multiplier
        estimated_savings = potential_losses * 0.87 * 0.5  # 87% recall, 50% effectiveness
        savings_percentage = estimated_savings / potential_losses if potential_losses > 0 else 0
        roi = estimated_savings / (avg_transaction * transaction_volume * 0.0001)  # Assuming small cost

        results = {
            'potential_losses': potential_losses,
            'estimated_savings': estimated_savings,
            'savings_percentage': savings_percentage,
            'roi': roi
        }

        st.subheader("Financial Impact")

        cols = st.columns(3)
        with cols[0]:
            st.metric(
                "Potential Monthly Losses",
                f"${results['potential_losses']:,.2f}",
                help="Expected losses without fraud detection"
            )
        with cols[1]:
            st.metric(
                "Estimated Monthly Savings",
                f"${results['estimated_savings']:,.2f}",
                delta=f"{results['savings_percentage']:.1%} reduction",
                delta_color="inverse"
            )
        with cols[2]:
            st.metric(
                "Return on Investment",
                f"{results['roi']:.1f}x",
                help="Savings per $1 spent on fraud prevention"
            )

        # Create and display a simple bar chart
        fig = px.bar(
            x=["Without Detection", "With Detection"],
            y=[results['potential_losses'], results['potential_losses'] - results['estimated_savings']],
            labels={'x': 'Scenario', 'y': 'Amount ($)'},
            title="Monthly Fraud Loss Comparison"
        )
        st.plotly_chart(fig, use_container_width=True)

# About Page
elif page == "About":
    st.title("About This Project")

    st.markdown("""
    ## Credit Card Fraud Detection System

    This application uses machine learning to identify potentially fraudulent credit card transactions
    in real-time. The model was trained on a dataset of anonymized credit card transactions.
    """)

    with st.expander("Technical Details", expanded=True):
        st.markdown("""
        - **Model**: Isolation Forest
        - **Training Data**: 284,807 transactions (492 fraud cases)
        - **Features**: 23 principal components + 2 engineered features
        - **Algorithm**: Unsupervised anomaly detection
        - **Performance**: 87% recall, 0.91 ROC AUC
        """)

    with st.expander("Implementation Details", expanded=False):
        st.markdown("""
        - **Frontend**: Streamlit
        - **Backend**: Scikit-learn
        - **Visualizations**: Plotly
        - **Deployment**: Hugging Face Spaces
        """)

    st.markdown("""
    Developed for educational purposes using Python and Streamlit.
    For questions or support, please contact the development team.
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("""
**Fraud Detection System**  
Version 1.0 
Model by Anvarbek Kuziboev 
[GitHub Repository](https://github.com/anvarbek11)  
""")