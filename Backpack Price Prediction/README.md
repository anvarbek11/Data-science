# Backpack Price Prediction System

[![Python 3.10](https://img.shields.io/badge/python-3.10-blue.svg)](https://www.python.org/downloads/)
[![AWS EC2](https://img.shields.io/badge/Deployed%20on-AWS%20EC2-orange.svg)](https://aws.amazon.com/ec2/)

An end-to-end machine learning system that predicts backpack prices based on product features, served through a Flask API with a web interface, deployed on AWS EC2 using Nginx.

![Screenshot 2025-04-25 121257](https://github.com/user-attachments/assets/0c50869b-3166-40c4-9bb6-0b32abb22f00)

**Live Deployment**: [http://ec2-51-20-189-73.eu-north-1.compute.amazonaws.com/](http://ec2-51-20-189-73.eu-north-1.compute.amazonaws.com/)

## Key Features

### Data Science Pipeline
- **Comprehensive Data Preprocessing**
  - Size categorization (Small/Medium/Large → 0/1/2)
  - Boolean conversion for features (Yes/No → 1/0)
  - Feature engineering: Weight per compartment calculation
  - Missing value imputation using median values
  - Ordinal encoding for categorical variables

- **Advanced Modeling**
  - LightGBM regression with GPU acceleration
  - Early stopping with 100-round patience
  - Feature fraction randomization (0.8)
  - Explicit dtype conversion for memory optimization
  - K-fold cross validation (through LightGBM's internal validation)

- **Model Evaluation**
  - RMSE metric for performance measurement
  - Train-test split with random state reproducibility
  - Comprehensive feature validation

### Web Interface
- Dynamic form with dropdown inputs
- Real-time price predictions via API
- Mobile-responsive design

### Production Deployment
- Flask API 
- Nginx reverse proxy configuration
- AWS EC2 instance management
- Allow inbound HTTP (80) traffic
- Open SSH port (22) for management

---
📌 **Author**: Anvarbek Kuziboev  
📄 **Note**: This project is part of my personal portfolio.  
🚫 Unauthorized copying or use without attribution is not permitted.

