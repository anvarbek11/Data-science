# 💳 Credit Card Fraud Detection System

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-%233F4F75.svg?style=for-the-badge&logo=plotly&logoColor=white)

An interactive machine learning system that detects fraudulent credit card transactions in real-time using anomaly detection algorithms.

➡️ **Live Demo**: [![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://credit-card-fraud-detection-dufdbkc4mbhq6kcndxhrqn.streamlit.app/)
![1](https://github.com/user-attachments/assets/05e1e8f1-3ee3-436f-bc99-3fc2ade07fcf)

## ✨ Features

- **Real-time Fraud Prediction**: Analyzes transaction features using Isolation Forest
- **Interactive Dashboard**: 
  - Demo interface with sample fraud/legit transactions
  - Model performance visualizations
  - Business impact simulator
- **Explainable AI**: 
  - Confidence scores for predictions
- **Production-Ready**:
  - Cached model loading for fast inference
  - Responsive Streamlit UI

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Frontend** | Streamlit |
| **Machine Learning** | Scikit-learn (Isolation Forest) |
| **Visualization** | Plotly, Altair |
| **Model Serialization** | Joblib |
| **Deployment** | Streamlit Community Cloud |


## 🚀 Deployment

### Local Development
1. Clone the repository:
2. Install dependencies: pip install -r requirements.txt
3. Run the app: streamlit run app.py
## 🧠 Model Details
| Metric | Value |
|-----------|------------|
| **Algorithms** | Isolation Forest|
| **Recall** | 87% |
| **Precision** |3% |
| **ROC AUC** |0.91 |
| **Training Data** | 284,807 transactions (492 fraud cases)|
| **Features** | 23 PCA components + engineered features|
