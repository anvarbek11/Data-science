# IMDB Movie Review Sentiment Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange)
![Gradio](https://img.shields.io/badge/Gradio-Interface-purple)
![NLTK](https://img.shields.io/badge/NLTK-Text%20Processing-green)

A deep learning model that classifies IMDB movie reviews as positive or negative using LSTM neural networks.
![s1](https://github.com/user-attachments/assets/46b7a88d-d9ac-4f8a-b77a-87c1ca0b74fc)


## 🎯 Project Overview
- **Model**: LSTM (Long Short-Term Memory) neural network
- **Accuracy**: 85% on test set
- **Features**: Text preprocessing, word embeddings, sentiment classification
- **Deployment**: Live Gradio interface on [Hugging Face Spaces](https://huggingface.co/spaces/Anvarbekk/sentiment_analysis_imdb)
- **Dataset**: [Kaggle] (https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews/code?datasetId=134715&sortBy=voteCount)

## 📋 Dataset
- **Source**: IMDB Dataset (50,000 reviews)
- **Columns**:
  - `review`: Text of the movie review
  - `sentiment`: Label (positive/negative)

## 🛠️ Technical Implementation
model = Sequential([
    Embedding(input_dim=5000, output_dim=128, input_length=200),
    LSTM(128, dropout=0.2, recurrent_dropout=0.2),
    Dense(1, activation='sigmoid')
])
## Training Results
Epoch	Training Accuracy	Validation Accuracy
* 1	        73.29%	       85.93%
* 5	        92.73%	       85.03%
## 🚀 How to Use
- **Clone the repository**:
  git clone https://github.com/yourusername/imdb-sentiment-analysis.git
cd imdb-sentiment-analysis
- **Install requirements**:pip install -r requirements.txt
- **Run Jupyter notebook**:jupyter notebook IMDB_Sentiment_Analysis.ipynb
